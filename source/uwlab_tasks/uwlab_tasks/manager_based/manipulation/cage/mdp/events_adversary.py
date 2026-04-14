# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adversary event functions for CAGE manipulation tasks."""

import numpy as np
import scipy.stats as stats
import torch
import trimesh
import trimesh.transformations as tra

import isaaclab.utils.math as math_utils
import omni.usd
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from pxr import UsdGeom

from uwlab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from uwlab_tasks.manager_based.manipulation.cage.mdp import utils

from ..assembly_keypoints import Offset


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


def adversary_rigid_body_material_from_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str,
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    action_static_index: int,
    action_dynamic_index: int,
    make_consistent: bool = True,
) -> None:
    """Set rigid-body material friction from adversary action (reset-only).

    Uses ``action_static_index`` / ``action_dynamic_index`` into the adversary action vector.
    """

    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_cpu = env_ids.cpu()
    env_ids_dev = env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]

    asset = env.scene[asset_cfg.name]

    static_friction = torch.clamp(
        a[:, action_static_index], static_friction_range[0], static_friction_range[1]
    ).to("cpu")
    dynamic_friction = torch.clamp(
        a[:, action_dynamic_index], dynamic_friction_range[0], dynamic_friction_range[1]
    ).to("cpu")
    if make_consistent:
        dynamic_friction = torch.minimum(dynamic_friction, static_friction)

    materials = asset.root_physx_view.get_material_properties()
    materials[env_ids_cpu, :, 0] = static_friction.view(-1, 1)
    materials[env_ids_cpu, :, 1] = dynamic_friction.view(-1, 1)
    materials[env_ids_cpu, :, 2] = 0.0  # restitution

    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)


def adversary_rigid_body_mass_scale_from_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str,
    asset_cfg: SceneEntityCfg,
    mass_scale_range: tuple[float, float],
    mass_scale_index: int,
    recompute_inertia: bool = True,
) -> None:
    """Scale rigid-body masses from default using one adversary action index (reset-only)."""

    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_cpu = env_ids.cpu()
    env_ids_dev = env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]

    asset = env.scene[asset_cfg.name]

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    masses = asset.root_physx_view.get_masses()

    masses[env_ids_cpu[:, None], body_ids] = asset.data.default_mass[env_ids_cpu[:, None], body_ids].clone()
    mass_scale = torch.clamp(a[:, mass_scale_index], mass_scale_range[0], mass_scale_range[1]).to("cpu")
    masses[env_ids_cpu[:, None], body_ids] *= mass_scale.view(-1, 1)
    masses = torch.clamp(masses, min=1e-6)

    asset.root_physx_view.set_masses(masses, env_ids_cpu)

    if recompute_inertia:
        ratios = masses[env_ids_cpu[:, None], body_ids] / asset.data.default_mass[env_ids_cpu[:, None], body_ids]
        inertias = asset.root_physx_view.get_inertias()
        if isinstance(asset, Articulation):
            inertias[env_ids_cpu[:, None], body_ids] = (
                asset.data.default_inertia[env_ids_cpu[:, None], body_ids] * ratios[..., None]
            )
        else:
            if ratios.ndim == 2 and ratios.shape[1] == 1:
                ratios = ratios[:, 0]
            inertias[env_ids_cpu] = asset.data.default_inertia[env_ids_cpu] * ratios.view(-1, 1)
        asset.root_physx_view.set_inertias(inertias, env_ids_cpu)


def adversary_rigid_body_mass_abs_from_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str,
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float],
    mass_index: int,
    recompute_inertia: bool = True,
) -> None:
    """Set total rigid-body mass from adversary (reset-only).

    Scales default body masses uniformly so the sum over ``body_ids`` equals the clamped
    target mass (matches ``randomize_rigid_body_mass`` with ``operation="abs"`` intent).
    """

    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_cpu = env_ids.cpu()
    env_ids_dev = env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]

    asset = env.scene[asset_cfg.name]

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    masses = asset.root_physx_view.get_masses()
    default_chunk = asset.data.default_mass[env_ids_cpu[:, None], body_ids].clone()
    total_default = default_chunk.sum(dim=1, keepdim=True)
    target_total = torch.clamp(a[:, mass_index], mass_range[0], mass_range[1]).to("cpu").view(-1, 1)
    scale = target_total / torch.clamp(total_default, min=1e-9)
    masses[env_ids_cpu[:, None], body_ids] = default_chunk * scale
    masses = torch.clamp(masses, min=1e-6)

    asset.root_physx_view.set_masses(masses, env_ids_cpu)

    if recompute_inertia:
        ratios = masses[env_ids_cpu[:, None], body_ids] / asset.data.default_mass[env_ids_cpu[:, None], body_ids]
        inertias = asset.root_physx_view.get_inertias()
        if isinstance(asset, Articulation):
            inertias[env_ids_cpu[:, None], body_ids] = (
                asset.data.default_inertia[env_ids_cpu[:, None], body_ids] * ratios[..., None]
            )
        else:
            if ratios.ndim == 2 and ratios.shape[1] == 1:
                ratios = ratios[:, 0]
            inertias[env_ids_cpu] = asset.data.default_inertia[env_ids_cpu] * ratios.view(-1, 1)
        asset.root_physx_view.set_inertias(inertias, env_ids_cpu)


def adversary_gripper_actuator_gains_from_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str,
    asset_cfg: SceneEntityCfg,
    stiffness_scale_range: tuple[float, float],
    damping_scale_range: tuple[float, float],
    action_stiffness_index: int = 5,
    action_damping_index: int = 6,
) -> None:
    """Set gripper actuator stiffness/damping scaling from adversary action (reset-only)."""

    asset = env.scene[asset_cfg.name]
    env_ids = env_ids.to(asset.device)

    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_dev = env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]

    stiffness_scale = torch.clamp(
        a[:, action_stiffness_index], stiffness_scale_range[0], stiffness_scale_range[1]
    ).to(asset.device)
    damping_scale = torch.clamp(
        a[:, action_damping_index], damping_scale_range[0], damping_scale_range[1]
    ).to(asset.device)

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
            actuator_joint_indices = actuator.joint_indices
            asset_joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)
            actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
            if len(actuator_indices) == 0:
                continue
            global_indices = actuator_joint_indices[actuator_indices]

        # Stiffness
        stiffness = actuator.stiffness[env_ids].clone()
        stiffness[:, actuator_indices] = asset.data.default_joint_stiffness[env_ids][:, global_indices].clone()
        stiffness[:, actuator_indices] *= stiffness_scale.view(-1, 1)
        actuator.stiffness[env_ids] = stiffness
        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)

        # Damping
        damping = actuator.damping[env_ids].clone()
        damping[:, actuator_indices] = asset.data.default_joint_damping[env_ids][:, global_indices].clone()
        damping[:, actuator_indices] *= damping_scale.view(-1, 1)
        actuator.damping[env_ids] = damping
        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)


def adversary_joint_friction_armature_from_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str,
    asset_cfg: SceneEntityCfg,
    friction_scale_range: tuple[float, float],
    armature_scale_range: tuple[float, float],
    action_friction_index: int,
    action_armature_index: int,
) -> None:
    """Scale joint friction and armature from adversary action (reset-only).

    Reads two scalar indices from the adversary action vector, clamps them to the
    given ranges, and multiplies the default joint friction / armature values.
    """

    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_dev = env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]

    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = env_ids.to(asset.device)

    friction_scale = torch.clamp(
        a[:, action_friction_index], friction_scale_range[0], friction_scale_range[1]
    ).to(asset.device)
    armature_scale = torch.clamp(
        a[:, action_armature_index], armature_scale_range[0], armature_scale_range[1]
    ).to(asset.device)

    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = asset_cfg.joint_ids

    # Friction: scale default values
    default_friction = asset.data.default_joint_friction_coeff[env_ids]
    if joint_ids != slice(None):
        default_friction = default_friction[:, joint_ids]
    new_friction = default_friction.clone() * friction_scale.view(-1, 1)
    asset.write_joint_friction_coefficient_to_sim(new_friction, joint_ids=joint_ids, env_ids=env_ids)

    # Armature: scale default values
    default_armature = asset.data.default_joint_armature[env_ids]
    if joint_ids != slice(None):
        default_armature = default_armature[:, joint_ids]
    new_armature = default_armature.clone() * armature_scale.view(-1, 1)
    asset.write_joint_armature_to_sim(new_armature, joint_ids=joint_ids, env_ids=env_ids)


def adversary_osc_gains_from_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str,
    adversary_action_name: str,
    stiffness_scale_range: tuple[float, float],
    damping_scale_range: tuple[float, float],
    action_stiffness_index: int,
    action_damping_index: int,
) -> None:
    """Scale OSC controller Kp/Kd from adversary action (reset-only).

    Reads two scalar indices from the adversary action vector, clamps them,
    and uniformly scales the default Kp and derives Kd via the default damping ratio.
    """

    from .actions.task_space_actions import RelCartesianOSCAction

    raw_actions = _get_action_term_raw_actions(env, adversary_action_name)
    env_ids_dev = env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]

    action_term = env.action_manager._terms.get(action_name)
    if action_term is None or not isinstance(action_term, RelCartesianOSCAction):
        raise ValueError(f"Action term '{action_name}' is not a RelCartesianOSCAction.")

    env_ids = env_ids.to(action_term.device)

    stiffness_scale = torch.clamp(
        a[:, action_stiffness_index], stiffness_scale_range[0], stiffness_scale_range[1]
    ).to(action_term.device)
    damping_scale = torch.clamp(
        a[:, action_damping_index], damping_scale_range[0], damping_scale_range[1]
    ).to(action_term.device)

    kp_default = action_term._kp_default  # (6,)
    dr_default = action_term._damping_ratio_default  # (6,)

    new_kp = kp_default.unsqueeze(0) * stiffness_scale.view(-1, 1)
    new_dr = dr_default.unsqueeze(0) * damping_scale.view(-1, 1)

    action_term._kp[env_ids] = new_kp
    action_term._kd[env_ids] = 2.0 * torch.sqrt(new_kp) * new_dr


class adversary_reset_receptive_object_pose_from_action(ManagerTermBase):
    """Reset the receptive object's root state using adversary action for x, y, yaw.

    The adversary outputs 3 values which are clamped to the pose ranges for
    x, y, and yaw.  z, roll, pitch are fixed at their configured values
    (typically 0).  The result is added to the asset's default root state
    plus an optional offset asset, mirroring ``reset_root_states_uniform``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        pose_range_dict = cfg.params.get("pose_range")
        self.pose_range = torch.tensor(
            [pose_range_dict.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=env.device,
        )
        self.asset_cfgs = list(cfg.params.get("asset_cfgs", dict()).values())
        self.offset_asset_cfg = cfg.params.get("offset_asset_cfg")
        self.use_bottom_offset = cfg.params.get("use_bottom_offset", False)
        self.action_name: str = cfg.params["action_name"]
        self.action_x_index: int = cfg.params.get("action_x_index", 0)
        self.action_y_index: int = cfg.params.get("action_y_index", 1)
        self.action_yaw_index: int = cfg.params.get("action_yaw_index", 2)

        if self.use_bottom_offset:
            self.bottom_offset_positions = dict()
            for asset_cfg in self.asset_cfgs:
                asset: RigidObject | Articulation = env.scene[asset_cfg.name]
                usd_path = asset.cfg.spawn.usd_path
                metadata = utils.read_metadata_from_usd_directory(usd_path)
                bottom_offset = metadata.get("bottom_offset")
                self.bottom_offset_positions[asset_cfg.name] = (
                    torch.tensor(bottom_offset.get("pos"), device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
                )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        asset_cfgs: dict[str, SceneEntityCfg] = dict(),
        offset_asset_cfg: SceneEntityCfg = None,
        use_bottom_offset: bool = False,
        action_name: str = "adversaryaction",
        action_x_index: int = 0,
        action_y_index: int = 1,
        action_yaw_index: int = 2,
    ) -> None:
        raw_actions = _get_action_term_raw_actions(env, self.action_name)
        env_ids_dev = env_ids.to(raw_actions.device)
        a = raw_actions[env_ids_dev]

        # Clamp adversary outputs to pose ranges
        x = torch.clamp(a[:, self.action_x_index], self.pose_range[0, 0], self.pose_range[0, 1])
        y = torch.clamp(a[:, self.action_y_index], self.pose_range[1, 0], self.pose_range[1, 1])
        yaw = torch.clamp(a[:, self.action_yaw_index], self.pose_range[5, 0], self.pose_range[5, 1])

        # Fixed pose components
        z = torch.full_like(x, (self.pose_range[2, 0] + self.pose_range[2, 1]) * 0.5)
        roll = torch.full_like(x, (self.pose_range[3, 0] + self.pose_range[3, 1]) * 0.5)
        pitch = torch.full_like(x, (self.pose_range[4, 0] + self.pose_range[4, 1]) * 0.5)

        positions_offset = torch.stack([x, y, z], dim=-1)
        orientations_delta = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        for asset_cfg in self.asset_cfgs:
            asset: RigidObject | Articulation = env.scene[asset_cfg.name]
            root_states = asset.data.default_root_state[env_ids].clone()

            positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + positions_offset

            if self.offset_asset_cfg:
                offset_asset: RigidObject | Articulation = env.scene[self.offset_asset_cfg.name]
                offset_positions = offset_asset.data.default_root_state[env_ids].clone()
                positions += offset_positions[:, 0:3]

            if self.use_bottom_offset:
                bottom_offset_position = self.bottom_offset_positions[asset_cfg.name]
                positions -= bottom_offset_position[env_ids, 0:3]

            orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
            velocities = root_states[:, 7:13]

            asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
            asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


class adversary_reset_insertive_object_pose_from_assembled_offset(ManagerTermBase):
    """Reset insertive object pose using partial assembly dataset as base + adversary offsets.

    Samples a partial assembly state from a pre-recorded dataset (relative pose of the
    insertive w.r.t. the receptive object), then applies adversary action values as
    body-frame offsets on top.  Zero offsets → partial assembly state from dataset;
    large offsets → displaced (e.g. on table).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.receptive_object_cfg = cfg.params.get("receptive_object_cfg")
        self.receptive_object = env.scene[self.receptive_object_cfg.name]
        self.insertive_object_cfg = cfg.params.get("insertive_object_cfg")
        self.insertive_object = env.scene[self.insertive_object_cfg.name]

        # Load partial assembly dataset (same as reset_insertive_object_from_partial_assembly_dataset)
        dataset_dir: str = cfg.params.get("dataset_dir")
        insertive_usd_path = self.insertive_object.cfg.spawn.usd_path
        receptive_usd_path = self.receptive_object.cfg.spawn.usd_path
        pair = utils.compute_pair_dir(insertive_usd_path, receptive_usd_path)
        dataset_path = f"{dataset_dir}/Resets/{pair}/partial_assemblies.pt"

        local_path = utils.safe_retrieve_file_path(dataset_path)
        data = torch.load(local_path, map_location="cpu")

        rel_pos = data.get("relative_position")
        rel_quat = data.get("relative_orientation")
        if rel_pos is None or rel_quat is None or len(rel_pos) == 0:
            raise ValueError(f"No partial assembly data found in {dataset_path}")

        if not isinstance(rel_pos, torch.Tensor):
            rel_pos = torch.as_tensor(rel_pos, dtype=torch.float32)
        if not isinstance(rel_quat, torch.Tensor):
            rel_quat = torch.as_tensor(rel_quat, dtype=torch.float32)

        self.rel_positions = rel_pos.to(env.device, dtype=torch.float32)
        self.rel_quaternions = rel_quat.to(env.device, dtype=torch.float32)

        # Parse pose_range_b for clamping adversary offsets
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b", dict())
        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.ranges = torch.tensor(range_list, device=env.device)

        self.action_name: str = cfg.params["action_name"]
        self.action_start_index: int = cfg.params.get("action_start_index", 3)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        dataset_dir: str = "",
        receptive_object_cfg: SceneEntityCfg = None,
        insertive_object_cfg: SceneEntityCfg = None,
        pose_range_b: dict[str, tuple[float, float]] = dict(),
        action_name: str = "adversaryaction",
        action_start_index: int = 3,
    ) -> None:
        num_envs = len(env_ids)

        # 1. Get receptive object's current world pose (already reset by adversary)
        receptive_pos_w = self.receptive_object.data.root_pos_w[env_ids]
        receptive_quat_w = self.receptive_object.data.root_quat_w[env_ids]

        # 2. Sample partial assembly states from dataset
        assembly_indices = torch.randint(0, len(self.rel_positions), (num_envs,), device=env.device)
        sampled_rel_pos = self.rel_positions[assembly_indices]
        sampled_rel_quat = self.rel_quaternions[assembly_indices]

        # Transform to world coordinates: T_insertive_w = T_receptive_w * T_relative
        base_pos_w, base_quat_w = math_utils.combine_frame_transforms(
            receptive_pos_w, receptive_quat_w, sampled_rel_pos, sampled_rel_quat
        )

        # 3. Split envs: 50% get adversary offsets, 50% keep raw dataset samples
        adversary_mask = torch.rand(num_envs, device=env.device) < 0.5

        insertive_pos_w = base_pos_w.clone()
        insertive_quat_w = base_quat_w.clone()

        if adversary_mask.any():
            # Read adversary action values for the selected envs
            raw_actions = _get_action_term_raw_actions(env, self.action_name)
            env_ids_dev = env_ids.to(raw_actions.device)
            a = raw_actions[env_ids_dev]
            si = self.action_start_index
            offset_values = a[adversary_mask, si : si + 6]

            # Clamp to pose_range_b
            clamped = torch.clamp(offset_values, self.ranges[:, 0], self.ranges[:, 1])

            # Apply adversary offsets in body frame
            offset_positions = clamped[:, 0:3]
            offset_orientations = math_utils.quat_from_euler_xyz(clamped[:, 3], clamped[:, 4], clamped[:, 5])

            adv_pos_w, adv_quat_w = math_utils.combine_frame_transforms(
                base_pos_w[adversary_mask], base_quat_w[adversary_mask], offset_positions, offset_orientations
            )

            insertive_pos_w[adversary_mask] = adv_pos_w
            insertive_quat_w[adversary_mask] = adv_quat_w

        # 6. Write insertive root state to sim
        self.insertive_object.write_root_state_to_sim(
            root_state=torch.cat(
                [
                    insertive_pos_w,
                    insertive_quat_w,
                    torch.zeros((num_envs, 6), device=env.device),
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )


class adversary_reset_end_effector_from_action(ManagerTermBase):
    """Reset end-effector pose and gripper state from the adversary's action.

    Mirrors the structure of omnireset's ``reset_end_effector_round_fixed_asset``:
    set IK target → 10 damped iterations → write joint state. The only difference
    is that the target comes from the clamped adversary action
    (``action[start:start+6]`` for EE pose, ``action[gripper_action_index]`` for
    the gripper finger joint), not a uniform sample. After the IK loop, the
    gripper joint is written explicitly.

    No validation, no resampling, no fallback. Bad proposals → bad robot pose →
    physics settling fails → ``abnormal_robot`` or ``check_reset_state_success``
    rejects the state → adversary gets a low reward and learns to avoid them.
    The downstream physics validation is the only validator.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        fixed_asset_cfg: SceneEntityCfg = cfg.params.get("fixed_asset_cfg")  # type: ignore
        fixed_asset_offset: Offset = cfg.params.get("fixed_asset_offset")  # type: ignore
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg", SceneEntityCfg("robot"))
        gripper_cfg: SceneEntityCfg = cfg.params.get(
            "gripper_cfg", SceneEntityCfg("robot", joint_names=["finger_joint"])
        )

        # Pose range for clamping the adversary's 6D EE proposal
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b")  # type: ignore
        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.ranges = torch.tensor(range_list, device=env.device)

        # Gripper range
        gripper_range: tuple[float, float] = cfg.params.get("gripper_range", (0.0, 0.785398))
        self.gripper_range = torch.tensor(gripper_range, device=env.device)

        # Fixed asset (robot base) for position offset
        self.fixed_asset: Articulation | RigidObject = env.scene[fixed_asset_cfg.name]
        self.fixed_asset_offset: Offset = fixed_asset_offset

        # Robot and IK solver — same setup as omnireset's reset_end_effector_round_fixed_asset
        self.robot: Articulation = env.scene[robot_ik_cfg.name]
        self.joint_ids: list[int] | slice = robot_ik_cfg.joint_ids
        self.n_joints: int = self.robot.num_joints if isinstance(self.joint_ids, slice) else len(self.joint_ids)

        robot_ik_solver_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name=robot_ik_cfg.name,
            joint_names=robot_ik_cfg.joint_names,  # type: ignore
            body_name=robot_ik_cfg.body_names,  # type: ignore
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
        )
        self.solver: DifferentialInverseKinematicsAction = robot_ik_solver_cfg.class_type(robot_ik_solver_cfg, env)  # type: ignore

        # Gripper joint
        self.gripper_joint_ids: list[int] | slice = gripper_cfg.joint_ids

        # Action indices
        self.action_name: str = cfg.params["action_name"]
        self.action_start_index: int = cfg.params.get("action_start_index", 9)
        self.gripper_action_index: int = cfg.params.get("gripper_action_index", 15)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        fixed_asset_cfg: SceneEntityCfg = None,
        fixed_asset_offset: Offset = None,
        pose_range_b: dict[str, tuple[float, float]] = dict(),
        robot_ik_cfg: SceneEntityCfg = None,
        gripper_cfg: SceneEntityCfg = None,
        gripper_range: tuple[float, float] = (0.0, 0.785398),
        action_name: str = "adversaryaction",
        action_start_index: int = 9,
        gripper_action_index: int = 15,
    ) -> None:
        # Read adversary actions for the envs being reset
        raw_actions = _get_action_term_raw_actions(env, self.action_name)
        a = raw_actions[env_ids.to(raw_actions.device)]

        # Clamp the 6-DOF EE pose to the configured range
        si = self.action_start_index
        clamped = torch.clamp(a[:, si : si + 6], self.ranges[:, 0], self.ranges[:, 1])

        # World-frame target = fixed asset tip + adversary offset
        if self.fixed_asset_offset is not None:
            fixed_tip_pos_w, _ = self.fixed_asset_offset.apply(self.fixed_asset)
        else:
            fixed_tip_pos_w = self.fixed_asset.data.root_pos_w

        pos_w = fixed_tip_pos_w[env_ids] + clamped[:, 0:3]
        quat_w = math_utils.quat_from_euler_xyz(clamped[:, 3], clamped[:, 4], clamped[:, 5])

        # Convert world target to base frame, fill only env_ids slots
        pos_b, quat_b = self.solver._compute_frame_pose()
        pos_b[env_ids], quat_b[env_ids] = math_utils.subtract_frame_transforms(
            self.robot.data.root_link_pos_w[env_ids],
            self.robot.data.root_link_quat_w[env_ids],
            pos_w,
            quat_w,
        )
        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))

        # 10 damped IK iterations (matches omnireset; ~5% residual error)
        for _ in range(10):
            self.solver.apply_actions()
            delta_joint_pos = 0.25 * (self.robot.data.joint_pos_target[env_ids] - self.robot.data.joint_pos[env_ids])
            self.robot.write_joint_state_to_sim(
                position=(delta_joint_pos + self.robot.data.joint_pos[env_ids])[:, self.joint_ids],
                velocity=torch.zeros((len(env_ids), self.n_joints), device=env.device),
                joint_ids=self.joint_ids,
                env_ids=env_ids,
            )

        # Write gripper joint
        gripper_value = torch.clamp(
            a[:, self.gripper_action_index], self.gripper_range[0], self.gripper_range[1]
        ).unsqueeze(-1)
        self.robot.write_joint_state_to_sim(
            position=gripper_value,
            velocity=torch.zeros_like(gripper_value),
            joint_ids=self.gripper_joint_ids,
            env_ids=env_ids,
        )


class reset_from_state_buffer(ManagerTermBase):
    """Load validated scene state from the runner's ``ResetStateBuffer`` into env_ids.

    Runs LAST in the reset chain so that during Phase B policy training it
    overwrites whatever the adversary reset events placed with the validated
    state from the buffer. No-op when the buffer is detached or empty
    (i.e. during Phase A generation), letting the adversary events take effect
    instead.

    Stored states use ``scene.get_state(is_relative=True)`` format: per-env root
    poses are relative to the env origin, so we add ``env_origins[env_ids]`` to
    the xyz before writing to sim.

    Only handles the asset categories the cage scene actually has — articulations
    (the robot) and rigid objects (insertive/receptive). No deformables, no
    surface grippers.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        buf = getattr(env, "reset_state_buffer", None)
        if buf is None or buf.occupancy == 0:
            return

        state = buf.sample_for_envs(env_ids)
        scene = env.scene
        env_origins = scene.env_origins[env_ids]

        # Articulations (e.g. robot)
        for asset_name, articulation in scene._articulations.items():
            asset_state = state["articulation"].get(asset_name)
            if asset_state is None:
                continue
            root_pose = asset_state["root_pose"].clone()
            root_pose[:, :3] += env_origins
            articulation.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            articulation.write_root_velocity_to_sim(asset_state["root_velocity"], env_ids=env_ids)
            joint_position = asset_state["joint_position"]
            joint_velocity = asset_state["joint_velocity"]
            articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
            # Refresh PD targets so the controller doesn't yank the joints back.
            articulation.set_joint_position_target(joint_position, env_ids=env_ids)
            articulation.set_joint_velocity_target(joint_velocity, env_ids=env_ids)

        # Rigid objects (e.g. insertive_object, receptive_object)
        for asset_name, rigid_object in scene._rigid_objects.items():
            asset_state = state["rigid_object"].get(asset_name)
            if asset_state is None:
                continue
            root_pose = asset_state["root_pose"].clone()
            root_pose[:, :3] += env_origins
            rigid_object.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            rigid_object.write_root_velocity_to_sim(asset_state["root_velocity"], env_ids=env_ids)

        # Propagate joint targets to the simulator.
        scene.write_data_to_sim()


