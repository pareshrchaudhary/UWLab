# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adversary-controlled event functions for manipulation tasks."""

import numpy as np
import torch
from types import SimpleNamespace
import os

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
from uwlab_tasks.manager_based.manipulation.reset_states.mdp.collision_analyzer_cfg import CollisionAnalyzerCfg

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
# Helpers
# ---------------------------------------------------------------------------
def _map_range(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Map values from [-1, 1] to [lo, hi]."""
    return (raw + 1.0) / 2.0 * (hi - lo) + lo


def _continuous_to_index(raw: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Map continuous [-1, 1] to integer dataset index in [0, pool_size-1]."""
    return ((raw + 1.0) / 2.0 * pool_size).long().clamp(0, pool_size - 1)


def _adv_debug_enabled(env: ManagerBasedEnv | None = None) -> bool:
    if env is not None and bool(getattr(env, "_adversary_debug", False)):
        return True
    v = os.environ.get("UWLAB_ADVERSARY_DEBUG", "0").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _adv_debug(msg: str, env: ManagerBasedEnv | None = None) -> None:
    if _adv_debug_enabled(env):
        print(f"[AdversaryDebug] {msg}", flush=True)


# ---------------------------------------------------------------------------
# AdversaryControlledReset
# ---------------------------------------------------------------------------
class AdversaryControlledReset(ManagerTermBase):
    """MLP-based 4-mode adversary-controlled reset state generator.

    The adversary outputs 16 continuous values (indices relative to ``grasp_action_start_idx``).
    A mode is assigned **uniformly at random** (25 % each, not adversary-controlled).
    Only the subset of outputs relevant to the assigned mode is used.

    Adversary reset-state output layout (16 dims):
        [0]  rec_x            receptive object x          [rec_x_range]
        [1]  rec_y            receptive object y          [rec_y_range]
        [2]  rec_yaw          receptive object yaw        [rec_yaw_range] (degrees)
        [3]  ins_x            insertive object x          [ins_x_range]      (modes A, B)
        [4]  ins_y            insertive object y          [ins_y_range]      (modes A, B)
        [5]  ins_z            insertive object z          [ins_z_range]      (modes A, B)
        [6]  ins_roll         insertive roll              [-pi, pi]          (modes A, B)
        [7]  ins_pitch        insertive pitch             [-pi, pi]          (modes A, B)
        [8]  ins_yaw          insertive yaw               [-pi, pi]          (modes A, B)
        [9]  assembly_idx     partial-assembly pool idx   continuous->index  (modes C, D)
        [10] grasp_idx        grasp pool idx              continuous->index  (modes B, D)
        [11] ee_x             end-effector x              [ee_x_range]       (modes A, C)
        [12] ee_y             end-effector y              [ee_y_range]       (modes A, C)
        [13] ee_z             end-effector z              [ee_z_range]       (modes A, C)
        [14] ee_pitch         end-effector pitch          bounded            (modes A, C)
        [15] ee_yaw           end-effector yaw            bounded            (modes A, C)

    4 Modes (uniform 25 % each):
        A (0): Free insertive  + Free EE       (gripper open)
        B (1): Free insertive  + Grasped EE    (gripper closed)
        C (2): Assembly insert + Free EE       (gripper open)
        D (3): Assembly insert + Grasped EE    (gripper closed)
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

        # Workspace ranges
        self.rec_x_range: tuple[float, float] = cfg.params.get("rec_x_range", (0.30, 0.55))
        self.rec_y_range: tuple[float, float] = cfg.params.get("rec_y_range", (-0.10, 0.30))
        self.rec_yaw_range: tuple[float, float] = cfg.params.get("rec_yaw_range", (-15.0, 15.0))
        self.ins_x_range: tuple[float, float] = cfg.params.get("ins_x_range", (0.30, 0.55))
        self.ins_y_range: tuple[float, float] = cfg.params.get("ins_y_range", (-0.10, 0.50))
        self.ins_z_range: tuple[float, float] = cfg.params.get("ins_z_range", (0.00, 0.30))
        self.ee_x_range: tuple[float, float] = cfg.params.get("ee_x_range", (0.30, 0.70))
        self.ee_y_range: tuple[float, float] = cfg.params.get("ee_y_range", (-0.40, 0.40))
        self.ee_z_range: tuple[float, float] = cfg.params.get("ee_z_range", (0.00, 0.50))
        self.max_validation_attempts: int = cfg.params.get("max_validation_attempts", 10)
        self.collision_num_points: int = cfg.params.get("collision_num_points", 256)
        self.max_physics_retries: int = cfg.params.get("max_physics_retries", 3)
        self.physics_validation_steps: int = cfg.params.get("physics_validation_steps", 50)
        self.consecutive_stability_steps: int = cfg.params.get("consecutive_stability_steps", 5)
        self.max_object_pos_deviation: float = cfg.params.get("max_object_pos_deviation", 0.025)
        self.max_robot_pos_deviation: float = cfg.params.get("max_robot_pos_deviation", 0.05)
        self.pos_z_threshold: float = cfg.params.get("pos_z_threshold", -0.02)
        self.debug_print: bool = cfg.params.get("debug_print", False)

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
        self.gripper_approach_direction = tuple(metadata.get("gripper_approach_direction", (0.0, 0.0, 1.0)))

        # EE body index for pose deviation tracking
        ee_body_name = robot_ik_cfg.body_names if isinstance(robot_ik_cfg.body_names, str) else robot_ik_cfg.body_names[0]
        self.ee_body_idx = self.robot.data.body_names.index(ee_body_name)

        # Assets to check for physics validation (objects + robot)
        self.object_assets = [self.insertive_object, self.receptive_object]
        self.assets_to_check = self.object_assets + [self.robot]

        # Number of gripper joints for open-gripper writes
        self._n_gripper_joints = (
            len(self.gripper_joint_ids) if isinstance(self.gripper_joint_ids, list) else 1
        )

        # Load datasets
        self._load_grasp_dataset(env)
        self._load_partial_assembly_dataset(env)

        # Success monitor (4 modes: A=0, B=1, C=2, D=3)
        success_monitor_cfg = SuccessMonitorCfg(monitored_history_len=100, num_monitored_data=4, device=env.device)
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)
        self.mode_id = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        # Collision analyzers (SDF-based, same as check_reset_state_success)
        robot_collision_cfg = SceneEntityCfg("robot")
        robot_collision_cfg.resolve(env.scene)
        ca_cfgs = [
            CollisionAnalyzerCfg(
                num_points=self.collision_num_points, max_dist=0.5, min_dist=-0.0005,
                asset_cfg=robot_collision_cfg,
                obstacle_cfgs=[self.insertive_object_cfg],
            ),
            CollisionAnalyzerCfg(
                num_points=self.collision_num_points, max_dist=0.5, min_dist=0.0,
                asset_cfg=robot_collision_cfg,
                obstacle_cfgs=[self.receptive_object_cfg],
            ),
            CollisionAnalyzerCfg(
                num_points=self.collision_num_points, max_dist=0.5, min_dist=-0.0005,
                asset_cfg=self.insertive_object_cfg,
                obstacle_cfgs=[self.receptive_object_cfg],
            ),
        ]
        self.collision_analyzers = [ca_cfg.class_type(ca_cfg, env) for ca_cfg in ca_cfgs]
        _adv_debug(
            f"AdversaryControlledReset initialized: action='{self.action_name}', "
            f"grasp_start={self.grasp_action_start_idx}, collision_points={self.collision_num_points}",
            env,
        )

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

    # ---- pose composition from raw actions ----

    def _decode_actions(self, a: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode the 16-dim action slice into named fields."""
        return {
            "rec_x_raw": a[:, 0],
            "rec_y_raw": a[:, 1],
            "rec_yaw_raw": a[:, 2],
            "ins_x_raw": a[:, 3],
            "ins_y_raw": a[:, 4],
            "ins_z_raw": a[:, 5],
            "ins_roll_raw": a[:, 6],
            "ins_pitch_raw": a[:, 7],
            "ins_yaw_raw": a[:, 8],
            "asm_idx_raw": a[:, 9],
            "grasp_idx_raw": a[:, 10],
            "ee_x_raw": a[:, 11],
            "ee_y_raw": a[:, 12],
            "ee_z_raw": a[:, 13],
            "ee_pitch_raw": a[:, 14],
            "ee_yaw_raw": a[:, 15],
        }

    def _compose_receptive_pose(self, d: dict, env: ManagerBasedEnv, env_ids: torch.Tensor, num: int):
        """Compose receptive object world-frame pose (all modes)."""
        rec_x = _map_range(d["rec_x_raw"], *self.rec_x_range)
        rec_y = _map_range(d["rec_y_raw"], *self.rec_y_range)
        rec_yaw_deg = _map_range(d["rec_yaw_raw"], *self.rec_yaw_range)
        rec_yaw_rad = rec_yaw_deg * (np.pi / 180.0)

        table_z = env.scene["table"].data.root_pos_w[env_ids, 2]
        recep_z = table_z + 0.881
        recep_pos = torch.stack([rec_x, rec_y, recep_z], dim=1)
        recep_quat = math_utils.quat_from_euler_xyz(
            torch.zeros(num, device=env.device),
            torch.zeros(num, device=env.device),
            rec_yaw_rad,
        )
        return recep_pos, recep_quat

    def _compose_insertive_pose(
        self, d: dict, env: ManagerBasedEnv, env_ids: torch.Tensor, num: int,
        mode: torch.Tensor, recep_pos: torch.Tensor, recep_quat: torch.Tensor,
    ):
        """Compose insertive object world-frame pose (mode-dependent)."""
        ins_pos = torch.zeros((num, 3), device=env.device)
        ins_quat = torch.zeros((num, 4), device=env.device)
        ins_quat[:, 0] = 1.0  # default identity

        ins_free = (mode == 0) | (mode == 1)  # modes A, B
        ins_assembly = (mode == 2) | (mode == 3)  # modes C, D

        support_z = env.scene["ur5_metal_support"].data.root_pos_w[env_ids, 2]

        # Modes A, B: free insertive from adversary outputs
        if ins_free.any():
            m = ins_free
            ix = _map_range(d["ins_x_raw"][m], *self.ins_x_range)
            iy = _map_range(d["ins_y_raw"][m], *self.ins_y_range)
            iz = _map_range(d["ins_z_raw"][m], *self.ins_z_range) + support_z[m] + 0.013
            ins_pos[m] = torch.stack([ix, iy, iz], dim=1)
            ins_quat[m] = math_utils.quat_from_euler_xyz(
                d["ins_roll_raw"][m] * np.pi,
                d["ins_pitch_raw"][m] * np.pi,
                d["ins_yaw_raw"][m] * np.pi,
            )

        # Modes C, D: from partial assembly dataset relative to receptive
        if ins_assembly.any():
            m = ins_assembly
            asm_indices = _continuous_to_index(d["asm_idx_raw"][m], len(self.assembly_rel_positions))
            sp = self.assembly_rel_positions[asm_indices]
            sq = self.assembly_rel_quaternions[asm_indices]
            pw, qw = math_utils.combine_frame_transforms(recep_pos[m], recep_quat[m], sp, sq)
            ins_pos[m] = pw
            ins_quat[m] = qw

        return ins_pos, ins_quat

    # ---- IK helpers ----

    def _solve_ik_and_write(self, env: ManagerBasedEnv, eids: torch.Tensor,
                            target_pos_w: torch.Tensor, target_quat_w: torch.Tensor):
        """Solve IK for target world-frame EE pose and write robot joint states."""
        pos_b, quat_b = self.solver._compute_frame_pose()
        pos_b[eids], quat_b[eids] = math_utils.subtract_frame_transforms(
            self.robot.data.root_link_pos_w[eids],
            self.robot.data.root_link_quat_w[eids],
            target_pos_w,
            target_quat_w,
        )
        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))
        for _ in range(25):
            self.solver.apply_actions()
            delta = 0.25 * (self.robot.data.joint_pos_target[eids] - self.robot.data.joint_pos[eids])
            self.robot.write_joint_state_to_sim(
                position=(delta + self.robot.data.joint_pos[eids])[:, self.joint_ids],
                velocity=torch.zeros((len(eids), self.n_joints), device=env.device),
                joint_ids=self.joint_ids,
                env_ids=eids,
            )

    def _set_gripper_open(self, env: ManagerBasedEnv, eids: torch.Tensor):
        """Set gripper to open position."""
        n = len(eids)
        open_pos = torch.full((n, self._n_gripper_joints), float(self.gripper_open_joint_angle), device=env.device)
        self.robot.write_joint_state_to_sim(
            position=open_pos, velocity=torch.zeros_like(open_pos),
            joint_ids=self.gripper_joint_ids, env_ids=eids,
        )

    def _set_gripper_from_dataset(self, env: ManagerBasedEnv, eids: torch.Tensor, grasp_indices: torch.Tensor):
        """Set gripper to closed positions from grasp dataset."""
        self.robot.write_joint_state_to_sim(
            position=self.gripper_joint_positions[grasp_indices],
            velocity=torch.zeros_like(self.gripper_joint_positions[grasp_indices]),
            joint_ids=self.gripper_joint_ids,
            env_ids=eids,
        )

    # ---- EE placement ----

    def _place_ee(self, d: dict, env: ManagerBasedEnv, env_ids: torch.Tensor,
                  mode: torch.Tensor, ins_pos: torch.Tensor, ins_quat: torch.Tensor):
        """Place the end-effector via IK (mode-dependent)."""
        ee_free = (mode == 0) | (mode == 2)  # modes A, C
        ee_grasped = (mode == 1) | (mode == 3)  # modes B, D

        # Modes A, C: free EE from adversary outputs
        if ee_free.any():
            m = ee_free
            eids = env_ids[m]
            ex = _map_range(d["ee_x_raw"][m], *self.ee_x_range)
            ey = _map_range(d["ee_y_raw"][m], *self.ee_y_range)
            ez = _map_range(d["ee_z_raw"][m], *self.ee_z_range)
            # Roll fixed (pi for downward-facing), pitch and yaw from adversary
            ee_roll = torch.full_like(d["ee_pitch_raw"][m], np.pi)
            ee_quat = math_utils.quat_from_euler_xyz(
                ee_roll,
                d["ee_pitch_raw"][m] * (np.pi / 2.0),  # bounded [-pi/2, pi/2]
                d["ee_yaw_raw"][m] * np.pi,             # bounded [-pi, pi]
            )
            ee_pos = torch.stack([ex, ey, ez], dim=1)
            self._solve_ik_and_write(env, eids, ee_pos, ee_quat)
            self._set_gripper_open(env, eids)

        # Modes B, D: grasped EE from grasp dataset relative to insertive
        if ee_grasped.any():
            m = ee_grasped
            eids = env_ids[m]
            grasp_indices = _continuous_to_index(d["grasp_idx_raw"][m], len(self.grasp_rel_positions))
            gp = self.grasp_rel_positions[grasp_indices]
            gq = self.grasp_rel_quaternions[grasp_indices]
            gw_pos, gw_quat = math_utils.combine_frame_transforms(ins_pos[m], ins_quat[m], gp, gq)
            self._solve_ik_and_write(env, eids, gw_pos, gw_quat)
            self._set_gripper_from_dataset(env, eids, grasp_indices)

    # ---- validation ----

    def _validate_poses(self, env: ManagerBasedEnv, env_ids: torch.Tensor,
                        recep_pos: torch.Tensor, ins_pos: torch.Tensor,
                        mode: torch.Tensor) -> torch.Tensor:
        """Validate proposed object poses geometrically. Returns bool mask of valid envs."""
        num = len(env_ids)
        valid = torch.ones(num, dtype=torch.bool, device=env.device)

        support_z = env.scene["ur5_metal_support"].data.root_pos_w[env_ids, 2]
        floor_z = support_z + 0.013

        ins_free = (mode == 0) | (mode == 1)

        # Check 1: insertive object not below floor (free modes only)
        if ins_free.any():
            below_floor = ins_pos[ins_free, 2] < floor_z[ins_free]
            valid_sub = valid[ins_free].clone()
            valid_sub[below_floor] = False
            valid[ins_free] = valid_sub

        # Check 2: insertive within robot reachable workspace
        ins_xy_dist = torch.norm(ins_pos[:, :2], dim=1)
        valid &= ins_xy_dist < 0.85

        # Check 3: minimum separation between insertive and receptive (free modes)
        if ins_free.any():
            dist = torch.norm(ins_pos[ins_free, :2] - recep_pos[ins_free, :2], dim=1)
            valid_sub = valid[ins_free].clone()
            valid_sub[dist < 0.03] = False
            valid[ins_free] = valid_sub

        # Check 4: receptive within robot reachable workspace
        rec_xy_dist = torch.norm(recep_pos[:, :2], dim=1)
        valid &= rec_xy_dist < 0.85

        return valid

    # ---- collision checking ----

    def _check_collisions(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> torch.Tensor:
        """Run SDF-based collision checks (robot↔objects, insertive↔receptive).

        Returns bool mask: True = collision-free."""
        return torch.all(
            torch.stack([ca(env, env_ids) for ca in self.collision_analyzers]),
            dim=0,
        )

    # ---- physics validation ----

    def _physics_validate(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> torch.Tensor:
        """Full physics validation replicating check_reset_state_success.

        Saves sim state for ALL envs, steps physics N times, checks stability/collisions/
        deviation/floor/gripper orientation on env_ids, then restores ALL envs to saved state.
        Returns bool mask (len(env_ids),): True = valid.
        """
        num = len(env_ids)
        all_env_ids = torch.arange(env.num_envs, device=env.device)

        # ---- save ALL env states ----
        saved_robot_root = self.robot.data.root_state_w.clone()
        saved_robot_jp = self.robot.data.joint_pos.clone()
        saved_robot_jv = self.robot.data.joint_vel.clone()
        saved_ins = self.insertive_object.data.root_state_w.clone()
        saved_rec = self.receptive_object.data.root_state_w.clone()

        # ---- record initial poses for deviation checking (env_ids only) ----
        init_ee_pos = self.robot.data.body_link_pos_w[env_ids, self.ee_body_idx].clone()
        init_obj_pos = {}
        for asset in self.object_assets:
            init_obj_pos[id(asset)] = asset.data.root_pos_w[env_ids].clone()

        # ---- step physics and track stability ----
        stability_counter = torch.zeros(num, device=env.device, dtype=torch.int32)
        for _ in range(self.physics_validation_steps):
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

            step_stable = torch.ones(num, device=env.device, dtype=torch.bool)
            # Robot (Articulation) velocity check
            step_stable &= self.robot.data.joint_vel[env_ids].abs().sum(dim=1) < 5.0
            # Object (RigidObject) velocity checks
            for obj in self.object_assets:
                step_stable &= obj.data.body_lin_vel_w[env_ids].abs().sum(dim=2).sum(dim=1) < 0.1
                step_stable &= obj.data.body_ang_vel_w[env_ids].abs().sum(dim=2).sum(dim=1) < 1.0

            stability_counter = torch.where(
                step_stable, stability_counter + 1, torch.zeros_like(stability_counter)
            )

        stability_reached = stability_counter >= self.consecutive_stability_steps

        # ---- abnormal gripper state ----
        abnormal_gripper = (
            self.robot.data.joint_vel[env_ids].abs() > (self.robot.data.joint_vel_limits[env_ids] * 2)
        ).any(dim=1)

        # ---- gripper orientation (approach direction within 60-deg cone downward) ----
        ee_quat = self.robot.data.body_link_quat_w[env_ids, self.ee_body_idx]
        approach_local = torch.tensor(
            self.gripper_approach_direction, device=env.device, dtype=torch.float32
        ).expand(num, -1)
        approach_world = math_utils.quat_apply(ee_quat, approach_local)
        gripper_ok = approach_world[:, 2] < -0.5  # cos(60deg) = 0.5

        # ---- pose deviation and floor penetration ----
        excessive_dev = torch.zeros(num, device=env.device, dtype=torch.bool)
        below_floor = torch.zeros(num, device=env.device, dtype=torch.bool)

        # EE deviation
        cur_ee_pos = self.robot.data.body_link_pos_w[env_ids, self.ee_body_idx]
        skip_ee = (
            torch.isnan(self.robot.data.root_pos_w[env_ids]).any(dim=1)
            | torch.isnan(self.robot.data.root_quat_w[env_ids]).any(dim=1)
        )
        ee_dev = (cur_ee_pos - init_ee_pos).norm(dim=1)
        ee_dev = torch.where(~skip_ee, ee_dev, torch.zeros_like(ee_dev))
        excessive_dev |= ee_dev > self.max_robot_pos_deviation
        below_floor |= cur_ee_pos[:, 2] < self.pos_z_threshold

        # Object deviations
        for asset in self.object_assets:
            cur_pos = asset.data.root_pos_w[env_ids]
            skip = torch.isnan(cur_pos).any(dim=1) | torch.isnan(asset.data.root_quat_w[env_ids]).any(dim=1)
            dev = (cur_pos - init_obj_pos[id(asset)]).norm(dim=1)
            dev = torch.where(~skip, dev, torch.zeros_like(dev))
            excessive_dev |= dev > self.max_object_pos_deviation
            below_floor |= cur_pos[:, 2] < self.pos_z_threshold

        # ---- collision check (SDF) ----
        collision_free = self._check_collisions(env, env_ids)

        # ---- combine all checks ----
        valid = (
            (~abnormal_gripper)
            & gripper_ok
            & stability_reached
            & (~excessive_dev)
            & (~below_floor)
            & collision_free
        )

        # ---- restore ALL env states ----
        self.robot.write_root_state_to_sim(saved_robot_root, all_env_ids)
        self.robot.write_joint_state_to_sim(saved_robot_jp, saved_robot_jv, env_ids=all_env_ids)
        self.insertive_object.write_root_state_to_sim(saved_ins, all_env_ids)
        self.receptive_object.write_root_state_to_sim(saved_rec, all_env_ids)
        # Refresh data buffers so subsequent IK reads correct robot state
        env.scene.update(dt=env.physics_dt)

        return valid

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
        rec_x_range: tuple[float, float] = (0.30, 0.55),
        rec_y_range: tuple[float, float] = (-0.10, 0.30),
        rec_yaw_range: tuple[float, float] = (-15.0, 15.0),
        ins_x_range: tuple[float, float] = (0.30, 0.55),
        ins_y_range: tuple[float, float] = (-0.10, 0.50),
        ins_z_range: tuple[float, float] = (0.00, 0.30),
        ee_x_range: tuple[float, float] = (0.30, 0.70),
        ee_y_range: tuple[float, float] = (-0.40, 0.40),
        ee_z_range: tuple[float, float] = (0.00, 0.50),
        max_validation_attempts: int = 10,
        collision_num_points: int = 256,
        max_physics_retries: int = 3,
        physics_validation_steps: int = 50,
        consecutive_stability_steps: int = 5,
        max_object_pos_deviation: float = 0.025,
        max_robot_pos_deviation: float = 0.05,
        pos_z_threshold: float = -0.02,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        num = len(env_ids)
        _adv_debug(f"reset_from_adversary start: num_envs={num}", env)

        # Read adversary actions (16-dim slice starting at grasp_action_start_idx)
        raw = _get_action_term_raw_actions(env, self.action_name)
        a = raw[env_ids, self.grasp_action_start_idx:]  # (num, 16)
        _adv_debug(
            f"reset actions slice: shape={tuple(a.shape)} mean={a.mean().item():.4f} std={a.std().item():.4f}",
            env,
        )

        # Assign mode uniformly at random (NOT adversary-controlled)
        # A=0, B=1, C=2, D=3 — each with 25% probability
        mode = torch.randint(0, 4, (num,), device=env.device)
        self.mode_id[env_ids] = mode
        _adv_debug(
            "mode counts: "
            f"A={int((mode==0).sum().item())}, "
            f"B={int((mode==1).sum().item())}, "
            f"C={int((mode==2).sum().item())}, "
            f"D={int((mode==3).sum().item())}",
            env,
        )

        # Decode actions
        d = self._decode_actions(a)

        # Compose receptive pose (all modes)
        recep_pos, recep_quat = self._compose_receptive_pose(d, env, env_ids, num)

        # Compose insertive pose (mode-dependent)
        ins_pos, ins_quat = self._compose_insertive_pose(d, env, env_ids, num, mode, recep_pos, recep_quat)

        # Validate proposed poses; re-sample invalid ones from uniform random
        valid = self._validate_poses(env, env_ids, recep_pos, ins_pos, mode)
        rejection_count = 0

        for _attempt in range(self.max_validation_attempts):
            invalid = ~valid
            if not invalid.any():
                break
            rejection_count += invalid.sum().item()
            # Re-sample invalid envs with uniform random actions [-1, 1]
            n_inv = invalid.sum()
            a_resample = torch.rand((n_inv, 16), device=env.device) * 2.0 - 1.0
            d_resample = self._decode_actions(a_resample)
            rp, rq = self._compose_receptive_pose(d_resample, env, env_ids[invalid], int(n_inv))
            ip, iq = self._compose_insertive_pose(
                d_resample, env, env_ids[invalid], int(n_inv), mode[invalid], rp, rq,
            )
            # Update the invalid entries
            recep_pos[invalid] = rp
            recep_quat[invalid] = rq
            ins_pos[invalid] = ip
            ins_quat[invalid] = iq
            # Also update decoded actions for EE placement later
            for k in d:
                d[k][invalid] = d_resample[k]
            # Re-validate
            valid[invalid] = self._validate_poses(env, env_ids[invalid], rp, ip, mode[invalid])

        # Physics validation with retries (replicates check_reset_state_success)
        physics_rejection_count = 0
        for attempt in range(1 + self.max_physics_retries):
            # Write objects to sim
            self.receptive_object.write_root_state_to_sim(
                root_state=torch.cat([recep_pos, recep_quat, torch.zeros((num, 6), device=env.device)], dim=-1),
                env_ids=env_ids,
            )
            self.insertive_object.write_root_state_to_sim(
                root_state=torch.cat([ins_pos, ins_quat, torch.zeros((num, 6), device=env.device)], dim=-1),
                env_ids=env_ids,
            )
            # Place EE via IK (mode-dependent)
            self._place_ee(d, env, env_ids, mode, ins_pos, ins_quat)

            # Full physics validation (save → step → check → restore)
            valid = self._physics_validate(env, env_ids)

            if valid.all() or attempt == self.max_physics_retries:
                break

            physics_rejection_count += int((~valid).sum().item())

            # Re-sample failing envs
            failing = ~valid
            n_fail = int(failing.sum())
            fail_ids = env_ids[failing]

            a_new = torch.rand((n_fail, 16), device=env.device) * 2.0 - 1.0
            d_new = self._decode_actions(a_new)
            mode[failing] = torch.randint(0, 4, (n_fail,), device=env.device)
            self.mode_id[fail_ids] = mode[failing]

            rp, rq = self._compose_receptive_pose(d_new, env, fail_ids, n_fail)
            ip, iq = self._compose_insertive_pose(d_new, env, fail_ids, n_fail, mode[failing], rp, rq)

            geo_ok = self._validate_poses(env, fail_ids, rp, ip, mode[failing])
            update_mask = failing.clone()
            update_mask[failing] &= geo_ok

            if update_mask.any():
                recep_pos[update_mask] = rp[geo_ok]
                recep_quat[update_mask] = rq[geo_ok]
                ins_pos[update_mask] = ip[geo_ok]
                ins_quat[update_mask] = iq[geo_ok]
                for k in d:
                    d[k][update_mask] = d_new[k][geo_ok]

        # Final write (physics_validate restores sim, so re-write validated states)
        self.receptive_object.write_root_state_to_sim(
            root_state=torch.cat([recep_pos, recep_quat, torch.zeros((num, 6), device=env.device)], dim=-1),
            env_ids=env_ids,
        )
        self.insertive_object.write_root_state_to_sim(
            root_state=torch.cat([ins_pos, ins_quat, torch.zeros((num, 6), device=env.device)], dim=-1),
            env_ids=env_ids,
        )
        self._place_ee(d, env, env_ids, mode, ins_pos, ins_quat)

        # Reset velocities
        self.robot.set_joint_velocity_target(torch.zeros_like(self.robot.data.joint_vel[env_ids]), env_ids=env_ids)

        # Log metrics
        self._log_metrics(env, env_ids, mode, rejection_count, physics_rejection_count)
        _adv_debug(
            f"reset_from_adversary done: rejection_count={rejection_count}, "
            f"physics_rejection_count={physics_rejection_count}",
            env,
        )

    def _log_metrics(self, env, env_ids, mode, rejection_count, physics_rejection_count=0):
        if hasattr(env, "reward_manager") and hasattr(env.reward_manager, "get_term_cfg"):
            pc = env.reward_manager.get_term_cfg("progress_context")
            if pc is not None and hasattr(pc.func, "success"):
                sm = torch.where(pc.func.success[env_ids], 1.0, 0.0)
                self.success_monitor.success_update(self.mode_id[env_ids], sm)
                sr = self.success_monitor.get_success_rate()
                if "log" not in env.extras:
                    env.extras["log"] = {}
                names = [
                    "A_FreeObj_FreeEE",
                    "B_FreeObj_GraspEE",
                    "C_AssemblyObj_FreeEE",
                    "D_AssemblyObj_GraspEE",
                ]
                for i in range(4):
                    env.extras["log"][f"Metrics/mode_{i}_{names[i]}_success_rate"] = sr[i].item()

        if "log" not in env.extras:
            env.extras["log"] = {}
        for i in range(4):
            env.extras["log"][f"Metrics/mode_{i}_fraction"] = (mode == i).float().mean().item()
        env.extras["log"]["Metrics/validation_rejection_count"] = rejection_count
        env.extras["log"]["Metrics/physics_rejection_count"] = physics_rejection_count


def adversary_controlled_reset(
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
    rec_x_range: tuple[float, float] = (0.30, 0.55),
    rec_y_range: tuple[float, float] = (-0.10, 0.30),
    rec_yaw_range: tuple[float, float] = (-15.0, 15.0),
    ins_x_range: tuple[float, float] = (0.30, 0.55),
    ins_y_range: tuple[float, float] = (-0.10, 0.50),
    ins_z_range: tuple[float, float] = (0.00, 0.30),
    ee_x_range: tuple[float, float] = (0.30, 0.70),
    ee_y_range: tuple[float, float] = (-0.40, 0.40),
    ee_z_range: tuple[float, float] = (0.00, 0.50),
    max_validation_attempts: int = 10,
    collision_num_points: int = 256,
    max_physics_retries: int = 3,
    physics_validation_steps: int = 50,
    consecutive_stability_steps: int = 5,
    max_object_pos_deviation: float = 0.025,
    max_robot_pos_deviation: float = 0.05,
    pos_z_threshold: float = -0.02,
    debug_print: bool = False,
) -> None:
    """Wrapper that lazily creates and reuses the stateful class term."""
    if debug_print:
        setattr(env, "_adversary_debug", True)
    cache_attr = "_adversary_controlled_reset_term"
    term = getattr(env, cache_attr, None)
    if term is None:
        params = {
            "action_name": action_name,
            "grasp_action_start_idx": grasp_action_start_idx,
            "insertive_object_cfg": insertive_object_cfg,
            "receptive_object_cfg": receptive_object_cfg,
            "robot_ik_cfg": robot_ik_cfg,
            "gripper_cfg": gripper_cfg,
            "grasp_base_path": grasp_base_path,
            "partial_assembly_base_path": partial_assembly_base_path,
            "rec_x_range": rec_x_range,
            "rec_y_range": rec_y_range,
            "rec_yaw_range": rec_yaw_range,
            "ins_x_range": ins_x_range,
            "ins_y_range": ins_y_range,
            "ins_z_range": ins_z_range,
            "ee_x_range": ee_x_range,
            "ee_y_range": ee_y_range,
            "ee_z_range": ee_z_range,
            "max_validation_attempts": max_validation_attempts,
            "collision_num_points": collision_num_points,
            "max_physics_retries": max_physics_retries,
            "physics_validation_steps": physics_validation_steps,
            "consecutive_stability_steps": consecutive_stability_steps,
            "max_object_pos_deviation": max_object_pos_deviation,
            "max_robot_pos_deviation": max_robot_pos_deviation,
            "pos_z_threshold": pos_z_threshold,
            "debug_print": debug_print,
        }
        term = AdversaryControlledReset(SimpleNamespace(params=params), env)
        setattr(env, cache_attr, term)
        _adv_debug("created cached AdversaryControlledReset term", env)
    term(env, env_ids)


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
    _adv_debug(
        f"osc gains update: n={len(env_ids)}, stiff_idx={action_stiffness_index}, damp_idx={action_damping_index}",
        env,
    )
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
    _adv_debug(
        f"robot material: n={len(env_ids_cpu)}, sf_mean={sf.mean().item():.4f}, df_mean={df.mean().item():.4f}",
        env,
    )
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
    _adv_debug(f"robot mass: n={len(env_ids_cpu)}, mass_scale_mean={mass_scale.mean().item():.4f}", env)
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
    _adv_debug(
        f"robot joints: n={len(env_ids)}, friction_scale_mean={friction_scale.mean().item():.4f}, "
        f"armature_scale_mean={armature_scale.mean().item():.4f}",
        env,
    )

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
    _adv_debug(
        f"gripper gains: n={len(env_ids)}, stiffness_scale_mean={stiffness_scale.mean().item():.4f}, "
        f"damping_scale_mean={damping_scale.mean().item():.4f}",
        env,
    )
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
    _adv_debug(
        f"insertive material: n={len(env_ids_cpu)}, sf_mean={sf.mean().item():.4f}, df_mean={df.mean().item():.4f}",
        env,
    )
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
    _adv_debug(
        f"receptive material: n={len(env_ids_cpu)}, sf_mean={sf.mean().item():.4f}, df_mean={df.mean().item():.4f}",
        env,
    )
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
    _adv_debug(f"insertive mass: n={len(env_ids_cpu)}, mass_mean={mass_value.mean().item():.4f}", env)
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
    _adv_debug(f"receptive mass: n={len(env_ids_cpu)}, mass_scale_mean={mass_scale.mean().item():.4f}", env)
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
