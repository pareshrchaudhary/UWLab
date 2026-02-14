# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adversary-controlled event functions for manipulation tasks."""

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.version import get_isaac_sim_version

from uwlab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from uwlab_tasks.manager_based.manipulation.reset_states.mdp import utils

from .success_monitor_cfg import SuccessMonitorCfg


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _get_action_term_raw_actions(env: ManagerBasedEnv, action_name: str) -> torch.Tensor:
    """Fetch raw action tensor from the action manager."""
    action_term = env.action_manager._terms.get(action_name)
    if action_term is None:
        raise ValueError(f"Action term '{action_name}' not found in action manager.")
    if not hasattr(action_term, "raw_actions"):
        raise ValueError(f"Action term '{action_name}' does not expose 'raw_actions'.")
    raw_actions = action_term.raw_actions
    if raw_actions is None:
        raise ValueError(f"Action term '{action_name}' has no 'raw_actions' set.")
    return raw_actions


# ---------------------------------------------------------------------------
# AdversaryControlledReset
# ---------------------------------------------------------------------------
class AdversaryControlledReset(ManagerTermBase):
    """Adversary-controlled reset that directly selects grasp and partial assembly poses from pools.

    Adversary action layout (indices relative to grasp_action_start_idx):
        [0]  grasp_logit           sigmoid -> P(use grasp)
        [1]  grasp_index           continuous [-1,1] mapped to pool index
        [2]  assembly_logit        sigmoid -> P(use partial assembly)
        [3]  assembly_index        continuous [-1,1] mapped to pool index
        [4]  receptive_x           mapped to workspace range
        [5]  receptive_y           mapped to workspace range
        [6-8]   ee_pos  (x,y,z)   when do_grasp=False
        [9-11]  ee_orient (r,p,y)  when do_grasp=False
        [12-14] ins_pos  (x,y,z)  when do_assembly=False
        [15-17] ins_orient (r,p,y) when do_assembly=False
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.action_name: str = cfg.params.get("action_name")
        self.grasp_action_start_idx: int = cfg.params.get("grasp_action_start_idx", 9)
        self.insertive_object_cfg: SceneEntityCfg = cfg.params.get("insertive_object_cfg")
        self.receptive_object_cfg: SceneEntityCfg = cfg.params.get("receptive_object_cfg")
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg")
        gripper_cfg: SceneEntityCfg = cfg.params.get("gripper_cfg")
        self.grasp_base_path: str = cfg.params.get("grasp_base_path")
        self.partial_assembly_base_path: str = cfg.params.get("partial_assembly_base_path")
        self.init_grasp_prob: float = cfg.params.get("init_grasp_prob", 0.5)
        self.init_assembly_prob: float = cfg.params.get("init_assembly_prob", 0.5)

        self.workspace_x_range: tuple[float, float] = cfg.params.get("workspace_x_range", (0.3, 0.55))
        self.workspace_y_range: tuple[float, float] = cfg.params.get("workspace_y_range", (-0.1, 0.3))
        self.workspace_z_range: tuple[float, float] = cfg.params.get("workspace_z_range", (0.0, 0.3))

        # Assets
        self.insertive_object: RigidObject = env.scene[self.insertive_object_cfg.name]
        self.receptive_object: RigidObject = env.scene[self.receptive_object_cfg.name]
        self.robot: Articulation = env.scene[robot_ik_cfg.name]
        self.gripper: Articulation = env.scene[gripper_cfg.name]

        # IK joint setup
        self.joint_ids: list[int] | slice = robot_ik_cfg.joint_ids
        self.n_joints: int = self.robot.num_joints if isinstance(self.joint_ids, slice) else len(self.joint_ids)
        self.gripper_joint_ids: list[int] | slice = gripper_cfg.joint_ids

        # IK solver
        robot_ik_solver_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name=robot_ik_cfg.name,
            joint_names=robot_ik_cfg.joint_names,
            body_name=robot_ik_cfg.body_names,
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
        )
        self.solver: DifferentialInverseKinematicsAction = robot_ik_solver_cfg.class_type(robot_ik_solver_cfg, env)

        # Gripper metadata
        metadata = utils.read_metadata_from_usd_directory(self.robot.cfg.spawn.usd_path)
        self.gripper_open_joint_angle = metadata.get("finger_open_joint_angle", 0.04)

        # Load datasets
        self._load_grasp_dataset(env)
        self._load_partial_assembly_dataset(env)

        # Success monitor (4 modes from 2 binary flags)
        success_monitor_cfg = SuccessMonitorCfg(monitored_history_len=100, num_monitored_data=4, device=env.device)
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)
        self.mode_id = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    # ---- dataset loading ----

    def _load_grasp_dataset(self, env):
        insertive_usd_path = self.insertive_object.cfg.spawn.usd_path
        object_hash = utils.compute_assembly_hash(insertive_usd_path)
        path = f"{self.grasp_base_path}/{object_hash}.pt"
        local_path = retrieve_file_path(path)
        data = torch.load(local_path, map_location="cpu")

        grasp_group = data.get("grasp_relative_pose", data)
        rel_pos_list = grasp_group.get("relative_position", [])
        rel_quat_list = grasp_group.get("relative_orientation", [])
        gripper_jp_dict = grasp_group.get("gripper_joint_positions", {})

        n = len(rel_pos_list)
        if n == 0:
            raise ValueError(f"No grasp data in {path}")

        self.grasp_rel_positions = torch.stack(
            [(p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float32)) for p in rel_pos_list]
        ).to(env.device, dtype=torch.float32)
        self.grasp_rel_quaternions = torch.stack(
            [(q if isinstance(q, torch.Tensor) else torch.as_tensor(q, dtype=torch.float32)) for q in rel_quat_list]
        ).to(env.device, dtype=torch.float32)

        gripper_joint_list = (
            list(range(self.robot.num_joints))[self.gripper_joint_ids]
            if isinstance(self.gripper_joint_ids, slice)
            else list(self.gripper_joint_ids)
        )
        self.gripper_joint_positions = torch.zeros((n, len(gripper_joint_list)), device=env.device, dtype=torch.float32)
        for gi, rji in enumerate(gripper_joint_list):
            jname = self.robot.joint_names[rji]
            series = gripper_jp_dict.get(jname, [0.0] * n)
            self.gripper_joint_positions[:, gi] = torch.stack(
                [(j if isinstance(j, torch.Tensor) else torch.as_tensor(j, dtype=torch.float32)) for j in series]
            ).to(env.device, dtype=torch.float32)

        print(f"[AdversaryControlledReset] Loaded {n} grasps from {path}")

    def _load_partial_assembly_dataset(self, env):
        ins_path = self.insertive_object.cfg.spawn.usd_path
        rec_path = self.receptive_object.cfg.spawn.usd_path
        h = utils.compute_assembly_hash(ins_path, rec_path)
        path = f"{self.partial_assembly_base_path}/{h}.pt"
        local_path = retrieve_file_path(path)
        data = torch.load(local_path, map_location="cpu")

        rel_pos = data.get("relative_position")
        rel_quat = data.get("relative_orientation")
        if rel_pos is None or rel_quat is None or len(rel_pos) == 0:
            raise ValueError(f"No partial assembly data in {path}")

        if not isinstance(rel_pos, torch.Tensor):
            rel_pos = torch.as_tensor(rel_pos, dtype=torch.float32)
        if not isinstance(rel_quat, torch.Tensor):
            rel_quat = torch.as_tensor(rel_quat, dtype=torch.float32)

        self.assembly_rel_positions = rel_pos.to(env.device, dtype=torch.float32)
        self.assembly_rel_quaternions = rel_quat.to(env.device, dtype=torch.float32)
        print(f"[AdversaryControlledReset] Loaded {len(rel_pos)} partial assemblies from {path}")

    # ---- main reset call ----

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        action_name: str = "",
        grasp_action_start_idx: int = 9,
        insertive_object_cfg: SceneEntityCfg | None = None,
        receptive_object_cfg: SceneEntityCfg | None = None,
        robot_ik_cfg: SceneEntityCfg | None = None,
        gripper_cfg: SceneEntityCfg | None = None,
        grasp_base_path: str = "",
        partial_assembly_base_path: str = "",
        init_grasp_prob: float = 0.5,
        init_assembly_prob: float = 0.5,
        workspace_x_range: tuple[float, float] = (0.3, 0.55),
        workspace_y_range: tuple[float, float] = (-0.1, 0.3),
        workspace_z_range: tuple[float, float] = (0.0, 0.3),
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        num = len(env_ids)
        ws_x, ws_y, ws_z = self.workspace_x_range, self.workspace_y_range, self.workspace_z_range

        # Read adversary actions (only the slice starting at grasp_action_start_idx)
        raw = _get_action_term_raw_actions(env, self.action_name)
        a = raw[env_ids, self.grasp_action_start_idx:]  # (num, 18)

        grasp_logit       = a[:, 0]
        grasp_idx_raw     = a[:, 1]
        assembly_logit    = a[:, 2]
        assembly_idx_raw  = a[:, 3]
        recep_x_raw       = a[:, 4]
        recep_y_raw       = a[:, 5]
        ee_pos_raw        = a[:, 6:9]
        ee_orient_raw     = a[:, 9:12]
        ins_pos_raw       = a[:, 12:15]
        ins_orient_raw    = a[:, 15:18]

        # Sample binary decisions
        # Fallback to init probs when adversary hasn't output anything yet
        if a.abs().mean() < 1e-6:
            grasp_prob = torch.full_like(grasp_logit, self.init_grasp_prob)
            assembly_prob = torch.full_like(assembly_logit, self.init_assembly_prob)
        else:
            grasp_prob = torch.sigmoid(grasp_logit)
            assembly_prob = torch.sigmoid(assembly_logit)
        do_grasp = torch.bernoulli(grasp_prob).bool()
        do_assembly = torch.bernoulli(assembly_prob).bool()

        # Mode tracking (grasp*2 + assembly -> 0..3)
        self.mode_id[env_ids] = do_grasp.long() * 2 + do_assembly.long()

        # Map continuous [-1,1] to pool indices
        grasp_indices = ((grasp_idx_raw + 1) / 2 * len(self.grasp_rel_positions)).long().clamp(0, len(self.grasp_rel_positions) - 1)
        assembly_indices = ((assembly_idx_raw + 1) / 2 * len(self.assembly_rel_positions)).long().clamp(0, len(self.assembly_rel_positions) - 1)

        # ---- Place receptive object ----
        recep_x = (recep_x_raw + 1) / 2 * (ws_x[1] - ws_x[0]) + ws_x[0]
        recep_y = (recep_y_raw + 1) / 2 * (ws_y[1] - ws_y[0]) + ws_y[0]
        table_z = env.scene["table"].data.root_pos_w[env_ids, 2]
        recep_z = table_z + 0.881
        recep_pos = torch.stack([recep_x, recep_y, recep_z], dim=1)
        recep_quat = torch.zeros((num, 4), device=env.device)
        recep_quat[:, 0] = 1.0

        self.receptive_object.write_root_state_to_sim(
            root_state=torch.cat([recep_pos, recep_quat, torch.zeros((num, 6), device=env.device)], dim=-1),
            env_ids=env_ids,
        )

        # ---- Place insertive object ----
        ins_pos = torch.zeros((num, 3), device=env.device)
        ins_quat = torch.zeros((num, 4), device=env.device)
        support_z = env.scene["ur5_metal_support"].data.root_pos_w[env_ids, 2]

        if do_assembly.any():
            m = do_assembly
            sp = self.assembly_rel_positions[assembly_indices[m]]
            sq = self.assembly_rel_quaternions[assembly_indices[m]]
            pw, qw = math_utils.combine_frame_transforms(recep_pos[m], recep_quat[m], sp, sq)
            ins_pos[m] = pw
            ins_quat[m] = qw

        if (~do_assembly).any():
            m = ~do_assembly
            ax = (ins_pos_raw[m, 0] + 1) / 2 * (ws_x[1] - ws_x[0]) + ws_x[0]
            ay = (ins_pos_raw[m, 1] + 1) / 2 * (ws_y[1] - ws_y[0]) + ws_y[0]
            az = (ins_pos_raw[m, 2] + 1) / 2 * (ws_z[1] - ws_z[0]) + ws_z[0] + support_z[m] + 0.013
            ins_pos[m] = torch.stack([ax, ay, az], dim=1)
            ins_quat[m] = math_utils.quat_from_euler_xyz(
                ins_orient_raw[m, 0] * np.pi, ins_orient_raw[m, 1] * np.pi, ins_orient_raw[m, 2] * np.pi
            )

        self.insertive_object.write_root_state_to_sim(
            root_state=torch.cat([ins_pos, ins_quat, torch.zeros((num, 6), device=env.device)], dim=-1),
            env_ids=env_ids,
        )

        # ---- Place gripper via IK ----
        if do_grasp.any():
            m = do_grasp
            eids = env_ids[m]
            gp = self.grasp_rel_positions[grasp_indices[m]]
            gq = self.grasp_rel_quaternions[grasp_indices[m]]
            gw_pos, gw_quat = math_utils.combine_frame_transforms(ins_pos[m], ins_quat[m], gp, gq)

            pos_b, quat_b = self.solver._compute_frame_pose()
            pos_b[eids], quat_b[eids] = math_utils.subtract_frame_transforms(
                self.robot.data.root_link_pos_w[eids], self.robot.data.root_link_quat_w[eids], gw_pos, gw_quat
            )
            self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))
            for _ in range(25):
                self.solver.apply_actions()
                delta = 0.25 * (self.robot.data.joint_pos_target[eids] - self.robot.data.joint_pos[eids])
                self.robot.write_joint_state_to_sim(
                    position=(delta + self.robot.data.joint_pos[eids])[:, self.joint_ids],
                    velocity=torch.zeros((len(eids), self.n_joints), device=env.device),
                    joint_ids=self.joint_ids, env_ids=eids,
                )
            self.robot.write_joint_state_to_sim(
                position=self.gripper_joint_positions[grasp_indices[m]],
                velocity=torch.zeros_like(self.gripper_joint_positions[grasp_indices[m]]),
                joint_ids=self.gripper_joint_ids, env_ids=eids,
            )

        if (~do_grasp).any():
            m = ~do_grasp
            eids = env_ids[m]
            n_ng = len(eids)
            ex = (ee_pos_raw[m, 0] + 1) / 2 * (ws_x[1] - ws_x[0]) + ws_x[0]
            ey = (ee_pos_raw[m, 1] + 1) / 2 * (ws_y[1] - ws_y[0]) + ws_y[0]
            ez = (ee_pos_raw[m, 2] + 1) / 2 * 0.2 + 0.3
            eq = math_utils.quat_from_euler_xyz(
                ee_orient_raw[m, 0] * np.pi, ee_orient_raw[m, 1] * np.pi, ee_orient_raw[m, 2] * np.pi
            )

            pos_b, quat_b = self.solver._compute_frame_pose()
            pos_b[eids] = torch.stack([ex, ey, ez], dim=1)
            pos_b[eids], quat_b[eids] = math_utils.subtract_frame_transforms(
                self.robot.data.root_link_pos_w[eids], self.robot.data.root_link_quat_w[eids], pos_b[eids], eq
            )
            self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))
            for _ in range(25):
                self.solver.apply_actions()
                delta = 0.25 * (self.robot.data.joint_pos_target[eids] - self.robot.data.joint_pos[eids])
                self.robot.write_joint_state_to_sim(
                    position=(delta + self.robot.data.joint_pos[eids])[:, self.joint_ids],
                    velocity=torch.zeros((n_ng, self.n_joints), device=env.device),
                    joint_ids=self.joint_ids, env_ids=eids,
                )
            n_gj = len(self.gripper_joint_ids) if isinstance(self.gripper_joint_ids, list) else 1
            open_pos = torch.full((n_ng, n_gj), float(self.gripper_open_joint_angle), device=env.device)
            self.robot.write_joint_state_to_sim(
                position=open_pos, velocity=torch.zeros_like(open_pos),
                joint_ids=self.gripper_joint_ids, env_ids=eids,
            )

        # Reset velocities
        self.robot.set_joint_velocity_target(torch.zeros_like(self.robot.data.joint_vel[env_ids]), env_ids=env_ids)

        # Log
        self._log_metrics(env, env_ids, grasp_prob, assembly_prob, do_grasp, do_assembly)

    def _log_metrics(self, env, env_ids, grasp_prob, assembly_prob, do_grasp, do_assembly):
        if hasattr(env, "reward_manager") and hasattr(env.reward_manager, "get_term_cfg"):
            pc = env.reward_manager.get_term_cfg("progress_context")
            if pc is not None and hasattr(pc.func, "success"):
                sm = torch.where(pc.func.success[env_ids], 1.0, 0.0)
                self.success_monitor.success_update(self.mode_id[env_ids], sm)
                sr = self.success_monitor.get_success_rate()
                if "log" not in env.extras:
                    env.extras["log"] = {}
                names = ["AnywhereAnywhere", "AnywhereGrasped", "PartialAnywhere", "PartialGrasped"]
                for i in range(4):
                    env.extras["log"][f"Metrics/mode_{i}_{names[i]}_success_rate"] = sr[i].item()

        if "log" not in env.extras:
            env.extras["log"] = {}
        env.extras["log"]["Metrics/adversary_grasp_prob"] = grasp_prob.mean().item()
        env.extras["log"]["Metrics/adversary_assembly_prob"] = assembly_prob.mean().item()
        env.extras["log"]["Metrics/actual_grasp_rate"] = do_grasp.float().mean().item()
        env.extras["log"]["Metrics/actual_assembly_rate"] = do_assembly.float().mean().item()


# ---------------------------------------------------------------------------
# Physics adversary events (moved from events.py)
# ---------------------------------------------------------------------------
def adversary_operational_space_controller_gains_from_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    adversary_action_name: str,
    osc_action_name: str,
    stiffness_scale_range: tuple[float, float],
    damping_scale_range: tuple[float, float],
    action_stiffness_index: int = 7,
    action_damping_index: int = 8,
) -> None:
    """Set OSC stiffness/damping gains from adversary action (reset-only)."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    osc_term = env.action_manager._terms.get(osc_action_name)
    if osc_term is None:
        raise ValueError(f"Action term '{osc_action_name}' not found in action manager.")
    if not hasattr(osc_term, "_osc") or not hasattr(osc_term._osc, "cfg"):
        raise ValueError(f"Action term '{osc_action_name}' does not appear to be an operational space controller.")
    controller = osc_term._osc
    adv_raw = _get_action_term_raw_actions(env, adversary_action_name)
    a = adv_raw[env_ids]
    original_stiffness = torch.tensor(controller.cfg.motion_stiffness_task, device=env.device)
    original_damping = torch.tensor(controller.cfg.motion_damping_ratio_task, device=env.device)
    for i in range(len(env_ids)):
        env_id = env_ids[i : i + 1]
        stiff_scale = torch.clamp(a[i, action_stiffness_index], stiffness_scale_range[0], stiffness_scale_range[1])
        damp_scale = torch.clamp(a[i, action_damping_index], damping_scale_range[0], damping_scale_range[1])
        new_stiffness = torch.zeros((1, 6), device=env.device)
        new_stiffness[:, 0:3] = original_stiffness[0:3] * stiff_scale
        new_stiffness[:, 3:6] = original_stiffness[3:6] * stiff_scale
        controller._motion_p_gains_task[env_id] = torch.diag_embed(new_stiffness)
        controller._motion_p_gains_task[env_id] = (
            controller._selection_matrix_motion_task[env_id] @ controller._motion_p_gains_task[env_id]
        )
        new_damping_ratios = torch.zeros((1, 6), device=env.device)
        new_damping_ratios[:, 0:3] = original_damping[0:3] * damp_scale
        new_damping_ratios[:, 3:6] = original_damping[3:6] * damp_scale
        controller._motion_d_gains_task[env_id] = torch.diag_embed(
            2 * torch.diagonal(controller._motion_p_gains_task[env_id], dim1=-2, dim2=-1).sqrt() * new_damping_ratios
        )


def adversary_robot_material_from_action(
    env: ManagerBasedEnv, env_ids: torch.Tensor, action_name: str, asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float], dynamic_friction_range: tuple[float, float],
    num_buckets: int = 256, make_consistent: bool = True,
) -> None:
    """Set robot material params from adversary action (reset-only). Indices [0],[1]."""
    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_cpu, env_ids_dev = env_ids.cpu(), env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]
    asset = env.scene[asset_cfg.name]
    sf = torch.clamp(a[:, 0], static_friction_range[0], static_friction_range[1]).to("cpu")
    df = torch.clamp(a[:, 1], dynamic_friction_range[0], dynamic_friction_range[1]).to("cpu")
    if make_consistent:
        df = torch.minimum(df, sf)
    materials = asset.root_physx_view.get_material_properties()
    materials[env_ids_cpu, :, 0] = sf.view(-1, 1)
    materials[env_ids_cpu, :, 1] = df.view(-1, 1)
    materials[env_ids_cpu, :, 2] = 0.0
    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)


def adversary_robot_mass_from_action(
    env: ManagerBasedEnv, env_ids: torch.Tensor, action_name: str, asset_cfg: SceneEntityCfg,
    mass_scale_range: tuple[float, float], recompute_inertia: bool = True,
) -> None:
    """Set robot mass scaling from adversary action (reset-only). Index [2]."""
    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_cpu, env_ids_dev = env_ids.cpu(), env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]
    asset = env.scene[asset_cfg.name]
    body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu") if asset_cfg.body_ids == slice(None) else torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    masses = asset.root_physx_view.get_masses()
    masses[env_ids_cpu[:, None], body_ids] = asset.data.default_mass[env_ids_cpu[:, None], body_ids].clone()
    mass_scale = torch.clamp(a[:, 2], mass_scale_range[0], mass_scale_range[1]).to("cpu")
    masses[env_ids_cpu[:, None], body_ids] *= mass_scale.view(-1, 1)
    masses = torch.clamp(masses, min=1e-6)
    asset.root_physx_view.set_masses(masses, env_ids_cpu)
    if recompute_inertia:
        ratios = masses[env_ids_cpu[:, None], body_ids] / asset.data.default_mass[env_ids_cpu[:, None], body_ids]
        inertias = asset.root_physx_view.get_inertias()
        if isinstance(asset, Articulation):
            inertias[env_ids_cpu[:, None], body_ids] = asset.data.default_inertia[env_ids_cpu[:, None], body_ids] * ratios[..., None]
        else:
            if ratios.ndim == 2 and ratios.shape[1] == 1:
                ratios = ratios[:, 0]
            inertias[env_ids_cpu] = asset.data.default_inertia[env_ids_cpu] * ratios.view(-1, 1)
        asset.root_physx_view.set_inertias(inertias, env_ids_cpu)


def adversary_robot_joint_parameters_from_action(
    env: ManagerBasedEnv, env_ids: torch.Tensor, action_name: str, asset_cfg: SceneEntityCfg,
    friction_scale_range: tuple[float, float], armature_scale_range: tuple[float, float],
) -> None:
    """Set robot joint friction/armature from adversary action (reset-only). Indices [3],[4]."""
    asset = env.scene[asset_cfg.name]
    env_ids = env_ids.to(asset.device)
    raw_actions = _get_action_term_raw_actions(env, action_name)
    a = raw_actions[env_ids.to(raw_actions.device)]
    joint_ids = slice(None) if asset_cfg.joint_ids == slice(None) else torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)
    env_ids_for_slice = env_ids[:, None] if env_ids != slice(None) and joint_ids != slice(None) else env_ids
    friction_scale = torch.clamp(a[:, 3], friction_scale_range[0], friction_scale_range[1]).to(asset.device)
    armature_scale = torch.clamp(a[:, 4], armature_scale_range[0], armature_scale_range[1]).to(asset.device)

    friction_coeff = asset.data.default_joint_friction_coeff.clone()
    friction_coeff[env_ids_for_slice, joint_ids] *= friction_scale.view(-1, 1)
    friction_coeff = torch.clamp(friction_coeff, min=0.0)
    static_friction_coeff = friction_coeff[env_ids_for_slice, joint_ids]
    if get_isaac_sim_version().major >= 5:
        dfc = asset.data.default_joint_dynamic_friction_coeff.clone()
        vfc = asset.data.default_joint_viscous_friction_coeff.clone()
        dfc[env_ids_for_slice, joint_ids] *= friction_scale.view(-1, 1)
        vfc[env_ids_for_slice, joint_ids] *= friction_scale.view(-1, 1)
        dfc = torch.clamp(dfc, min=0.0)
        vfc = torch.clamp(vfc, min=0.0)
        dfc = torch.minimum(dfc, friction_coeff)
        dfc = dfc[env_ids_for_slice, joint_ids]
        vfc = vfc[env_ids_for_slice, joint_ids]
    else:
        dfc = None
        vfc = None
    asset.write_joint_friction_coefficient_to_sim(
        joint_friction_coeff=static_friction_coeff, joint_dynamic_friction_coeff=dfc,
        joint_viscous_friction_coeff=vfc, joint_ids=joint_ids, env_ids=env_ids,
    )
    armature = asset.data.default_joint_armature.clone()
    armature[env_ids_for_slice, joint_ids] *= armature_scale.view(-1, 1)
    armature = torch.clamp(armature, min=0.0)
    asset.write_joint_armature_to_sim(armature[env_ids_for_slice, joint_ids], joint_ids=joint_ids, env_ids=env_ids)


def adversary_gripper_actuator_gains_from_action(
    env: ManagerBasedEnv, env_ids: torch.Tensor, action_name: str, asset_cfg: SceneEntityCfg,
    stiffness_scale_range: tuple[float, float], damping_scale_range: tuple[float, float],
) -> None:
    """Set gripper actuator stiffness/damping from adversary action (reset-only). Indices [5],[6]."""
    asset = env.scene[asset_cfg.name]
    env_ids = env_ids.to(asset.device)
    raw_actions = _get_action_term_raw_actions(env, action_name)
    a = raw_actions[env_ids.to(raw_actions.device)]
    stiffness_scale = torch.clamp(a[:, 5], stiffness_scale_range[0], stiffness_scale_range[1]).to(asset.device)
    damping_scale = torch.clamp(a[:, 6], damping_scale_range[0], damping_scale_range[1]).to(asset.device)
    for actuator in asset.actuators.values():
        if isinstance(asset_cfg.joint_ids, slice):
            actuator_indices = slice(None)
            if isinstance(actuator.joint_indices, slice):
                global_indices = slice(None)
            elif isinstance(actuator.joint_indices, torch.Tensor):
                global_indices = actuator.joint_indices.to(asset.device)
            else:
                raise TypeError("Actuator joint indices must be a slice or a torch.Tensor.")
        elif isinstance(actuator.joint_indices, slice):
            global_indices = actuator_indices = torch.tensor(asset_cfg.joint_ids, device=asset.device)
        else:
            aji = actuator.joint_indices
            asset_ji = torch.tensor(asset_cfg.joint_ids, device=asset.device)
            actuator_indices = torch.nonzero(torch.isin(aji, asset_ji)).view(-1)
            if len(actuator_indices) == 0:
                continue
            global_indices = aji[actuator_indices]
        stiffness = actuator.stiffness[env_ids].clone()
        stiffness[:, actuator_indices] = asset.data.default_joint_stiffness[env_ids][:, global_indices].clone()
        stiffness[:, actuator_indices] *= stiffness_scale.view(-1, 1)
        actuator.stiffness[env_ids] = stiffness
        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)
        damping = actuator.damping[env_ids].clone()
        damping[:, actuator_indices] = asset.data.default_joint_damping[env_ids][:, global_indices].clone()
        damping[:, actuator_indices] *= damping_scale.view(-1, 1)
        actuator.damping[env_ids] = damping
        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)


def adversary_insertive_object_material_from_action(
    env: ManagerBasedEnv, env_ids: torch.Tensor, action_name: str, asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float], dynamic_friction_range: tuple[float, float],
    num_buckets: int = 256, make_consistent: bool = True,
) -> None:
    """Set insertive object material from adversary action. Indices [9],[10]."""
    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_cpu, env_ids_dev = env_ids.cpu(), env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]
    asset: RigidObject = env.scene[asset_cfg.name]
    sf = torch.clamp(a[:, 9], static_friction_range[0], static_friction_range[1]).to("cpu")
    df = torch.clamp(a[:, 10], dynamic_friction_range[0], dynamic_friction_range[1]).to("cpu")
    if make_consistent:
        df = torch.minimum(df, sf)
    materials = asset.root_physx_view.get_material_properties()
    materials[env_ids_cpu, :, 0] = sf.view(-1, 1)
    materials[env_ids_cpu, :, 1] = df.view(-1, 1)
    materials[env_ids_cpu, :, 2] = 0.0
    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)


def adversary_receptive_object_material_from_action(
    env: ManagerBasedEnv, env_ids: torch.Tensor, action_name: str, asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float], dynamic_friction_range: tuple[float, float],
    num_buckets: int = 256, make_consistent: bool = True,
) -> None:
    """Set receptive object material from adversary action. Indices [12],[13]."""
    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_cpu, env_ids_dev = env_ids.cpu(), env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]
    asset: RigidObject = env.scene[asset_cfg.name]
    sf = torch.clamp(a[:, 12], static_friction_range[0], static_friction_range[1]).to("cpu")
    df = torch.clamp(a[:, 13], dynamic_friction_range[0], dynamic_friction_range[1]).to("cpu")
    if make_consistent:
        df = torch.minimum(df, sf)
    materials = asset.root_physx_view.get_material_properties()
    materials[env_ids_cpu, :, 0] = sf.view(-1, 1)
    materials[env_ids_cpu, :, 1] = df.view(-1, 1)
    materials[env_ids_cpu, :, 2] = 0.0
    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)


def adversary_insertive_object_mass_from_action(
    env: ManagerBasedEnv, env_ids: torch.Tensor, action_name: str, asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float], recompute_inertia: bool = True,
) -> None:
    """Set insertive object mass from adversary action. Index [11]."""
    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_cpu, env_ids_dev = env_ids.cpu(), env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]
    asset: RigidObject = env.scene[asset_cfg.name]
    body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu") if asset_cfg.body_ids == slice(None) else torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    masses = asset.root_physx_view.get_masses()
    mass_value = torch.clamp(a[:, 11], mass_range[0], mass_range[1]).to("cpu")
    masses[env_ids_cpu[:, None], body_ids] = mass_value.view(-1, 1)
    masses = torch.clamp(masses, min=1e-6)
    asset.root_physx_view.set_masses(masses, env_ids_cpu)
    if recompute_inertia:
        ratios = masses[env_ids_cpu[:, None], body_ids] / asset.data.default_mass[env_ids_cpu[:, None], body_ids]
        inertias = asset.root_physx_view.get_inertias()
        if ratios.ndim == 2 and ratios.shape[1] == 1:
            ratios = ratios[:, 0]
        inertias[env_ids_cpu] = asset.data.default_inertia[env_ids_cpu] * ratios.view(-1, 1)
        asset.root_physx_view.set_inertias(inertias, env_ids_cpu)


def adversary_receptive_object_mass_from_action(
    env: ManagerBasedEnv, env_ids: torch.Tensor, action_name: str, asset_cfg: SceneEntityCfg,
    mass_scale_range: tuple[float, float], recompute_inertia: bool = True,
) -> None:
    """Set receptive object mass from adversary action. Index [14]."""
    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_cpu, env_ids_dev = env_ids.cpu(), env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]
    asset: RigidObject = env.scene[asset_cfg.name]
    body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu") if asset_cfg.body_ids == slice(None) else torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    masses = asset.root_physx_view.get_masses()
    masses[env_ids_cpu[:, None], body_ids] = asset.data.default_mass[env_ids_cpu[:, None], body_ids].clone()
    mass_scale = torch.clamp(a[:, 14], mass_scale_range[0], mass_scale_range[1]).to("cpu")
    masses[env_ids_cpu[:, None], body_ids] *= mass_scale.view(-1, 1)
    masses = torch.clamp(masses, min=1e-6)
    asset.root_physx_view.set_masses(masses, env_ids_cpu)
    if recompute_inertia:
        ratios = masses[env_ids_cpu[:, None], body_ids] / asset.data.default_mass[env_ids_cpu[:, None], body_ids]
        inertias = asset.root_physx_view.get_inertias()
        if ratios.ndim == 2 and ratios.shape[1] == 1:
            ratios = ratios[:, 0]
        inertias[env_ids_cpu] = asset.data.default_inertia[env_ids_cpu] * ratios.view(-1, 1)
        asset.root_physx_view.set_inertias(inertias, env_ids_cpu)
