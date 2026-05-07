# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP functions for manipulation tasks."""

import numpy as np
import torch

import isaacsim.core.utils.bounds as bounds_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import math as math_utils

from uwlab_tasks.manager_based.manipulation.cage.mdp import utils

from ..assembly_keypoints import Offset
from .collision_analyzer_cfg import CollisionAnalyzerCfg
from .pose_delta_adversary import (
    POSE_DELTA_ACCEPTED_RECORDS_ATTR,
    POSE_DELTA_CANDIDATE_ACTION_ATTR,
    POSE_DELTA_CANDIDATE_VALID_ATTR,
    POSE_DELTA_COMMITTED_ACTION_ATTR,
    POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_DEBUG_ATTR,
    POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_FAMILY_IDS_ATTR,
    POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_ROW_IDS_ATTR,
    POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_STATS_ATTR,
    POSE_DELTA_FAMILY_LANE_PERTURB_DEBUG_ATTR,
    POSE_DELTA_FAMILY_LANE_PERTURB_STATS_ATTR,
    POSE_DELTA_FAMILY_LANE_INVALID_MASK_ATTR,
    POSE_DELTA_FAMILY_LANE_SCRIPTED_MASK_ATTR,
    POSE_DELTA_FAMILY_LANE_VECTOR_VALID_ATTR,
    POSE_DELTA_FAMILY_LANE_VECTORS_ATTR,
    POSE_RESET_FAMILY_NAMES,
    commit_pose_delta_candidate_tensors,
)


def _check_obb_overlap(centroids_a, axes_a, half_extents_a, centroids_b, axes_b, half_extents_b) -> torch.Tensor:
    """
    OBB overlap check.

    Args:
        centroids_a: Centers of OBB A for all envs (num_envs, 3) - torch tensor on GPU
        axes_a: Orientation axes of OBB A for all envs (num_envs, 3, 3) - torch tensor on GPU
        half_extents_a: Half extents of OBB A (3,) - torch tensor on GPU
        centroids_b: Centers of OBB B for all envs (num_envs, 3) - torch tensor on GPU
        axes_b: Orientation axes of OBB B for all envs (num_envs, 3, 3) - torch tensor on GPU
        half_extents_b: Half extents of OBB B (3,) - torch tensor on GPU

    Returns:
        torch.Tensor: Boolean tensor (num_envs,) indicating overlap for each environment
    """
    num_envs = centroids_a.shape[0]
    device = centroids_a.device

    # Vector between centroids for all envs (num_envs, 3)
    d = centroids_b - centroids_a

    # Matrix C = A^T * B (rotation from A to B) for all envs (num_envs, 3, 3)
    C = torch.bmm(axes_a.transpose(1, 2), axes_b)
    abs_C = torch.abs(C)

    # Initialize overlap results (assume all overlap initially)
    overlap_results = torch.ones(num_envs, device=device, dtype=torch.bool)

    # Test all axes of A at once (vectorized across all 3 axes and all environments)
    # axes_a: (num_envs, 3, 3), d: (num_envs, 3) -> projections: (num_envs, 3)
    projections_on_axes_a = torch.abs(torch.bmm(d.unsqueeze(1), axes_a).squeeze(1))  # (num_envs, 3)
    ra_all = half_extents_a.unsqueeze(0).expand(num_envs, -1)  # (num_envs, 3)
    rb_all = torch.sum(half_extents_b.unsqueeze(0).unsqueeze(0) * abs_C, dim=2)  # (num_envs, 3)
    no_overlap_a = projections_on_axes_a > (ra_all + rb_all)  # (num_envs, 3)
    overlap_results &= ~torch.any(no_overlap_a, dim=1)  # (num_envs,)

    # Test all axes of B at once (vectorized across all 3 axes and all environments)
    # axes_b: (num_envs, 3, 3), d: (num_envs, 3) -> projections: (num_envs, 3)
    projections_on_axes_b = torch.abs(torch.bmm(d.unsqueeze(1), axes_b).squeeze(1))  # (num_envs, 3)
    ra_all_b = torch.sum(half_extents_a.unsqueeze(0).unsqueeze(0) * abs_C.transpose(1, 2), dim=2)  # (num_envs, 3)
    rb_all_b = half_extents_b.unsqueeze(0).expand(num_envs, -1)  # (num_envs, 3)
    no_overlap_b = projections_on_axes_b > (ra_all_b + rb_all_b)  # (num_envs, 3)
    overlap_results &= ~torch.any(no_overlap_b, dim=1)  # (num_envs,)

    # Test all cross products at once (9 cross products per environment)
    # Reshape axes for broadcasting: axes_a (num_envs, 3, 1, 3), axes_b (num_envs, 1, 3, 3)
    axes_a_expanded = axes_a.unsqueeze(2)  # (num_envs, 3, 1, 3)
    axes_b_expanded = axes_b.unsqueeze(1)  # (num_envs, 1, 3, 3)

    # Compute all 9 cross products at once: (num_envs, 3, 3, 3)
    cross_products = torch.cross(axes_a_expanded, axes_b_expanded, dim=3)  # (num_envs, 3, 3, 3)

    # Compute norms and filter out near-parallel axes: (num_envs, 3, 3)
    cross_norms = torch.norm(cross_products, dim=3)  # (num_envs, 3, 3)
    valid_crosses = cross_norms > 1e-6  # (num_envs, 3, 3)

    # Normalize cross products (set invalid ones to zero)
    normalized_crosses = torch.where(
        valid_crosses.unsqueeze(3),
        cross_products / cross_norms.unsqueeze(3).clamp(min=1e-6),
        torch.zeros_like(cross_products),
    )  # (num_envs, 3, 3, 3)

    # Project d onto all cross product axes: (num_envs, 3, 3)
    d_expanded = d.unsqueeze(1).unsqueeze(1)  # (num_envs, 1, 1, 3)
    projections_cross = torch.abs(torch.sum(d_expanded * normalized_crosses, dim=3))  # (num_envs, 3, 3)

    # Compute ra for all cross products: (num_envs, 3, 3)
    # half_extents_a: (3,), axes_a: (num_envs, 3, 3), normalized_crosses: (num_envs, 3, 3, 3)
    axes_a_cross_dots = torch.abs(
        torch.sum(axes_a.unsqueeze(1).unsqueeze(1) * normalized_crosses.unsqueeze(3), dim=4)
    )  # (num_envs, 3, 3, 3)
    ra_cross = torch.sum(
        half_extents_a.unsqueeze(0).unsqueeze(0).unsqueeze(0) * axes_a_cross_dots, dim=3
    )  # (num_envs, 3, 3)

    # Compute rb for all cross products: (num_envs, 3, 3)
    axes_b_cross_dots = torch.abs(
        torch.sum(axes_b.unsqueeze(1).unsqueeze(1) * normalized_crosses.unsqueeze(4), dim=4)
    )  # (num_envs, 3, 3, 3)
    rb_cross = torch.sum(
        half_extents_b.unsqueeze(0).unsqueeze(0).unsqueeze(0) * axes_b_cross_dots, dim=3
    )  # (num_envs, 3, 3)

    # Check separating condition for all cross products: (num_envs, 3, 3)
    no_overlap_cross = projections_cross > (ra_cross + rb_cross)  # (num_envs, 3, 3)
    # Only consider valid cross products
    no_overlap_cross_valid = no_overlap_cross & valid_crosses  # (num_envs, 3, 3)
    overlap_results &= ~torch.any(no_overlap_cross_valid.view(num_envs, -1), dim=1)  # (num_envs,)

    return overlap_results


class check_grasp_success(ManagerTermBase):
    """Check if grasp is successful based on object stability, gripper closure, and collision detection."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._env = env

        self.object_cfg = cfg.params.get("object_cfg")
        self.gripper_cfg = cfg.params.get("gripper_cfg")
        self.collision_analyzer_cfg = cfg.params.get("collision_analyzer_cfg")
        self.collision_analyzer = self.collision_analyzer_cfg.class_type(self.collision_analyzer_cfg, self._env)
        self.max_pos_deviation = cfg.params.get("max_pos_deviation")
        self.pos_z_threshold = cfg.params.get("pos_z_threshold")
        self.consecutive_stability_steps = cfg.params.get("consecutive_stability_steps", 5)
        self.stability_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)

        object_asset = self._env.scene[self.object_cfg.name]
        if not hasattr(object_asset, "initial_pos"):
            object_asset.initial_pos = object_asset.data.root_pos_w.clone()
            object_asset.initial_quat = object_asset.data.root_quat_w.clone()
        else:
            object_asset.initial_pos[env_ids] = object_asset.data.root_pos_w[env_ids].clone()
            object_asset.initial_quat[env_ids] = object_asset.data.root_quat_w[env_ids].clone()

        if env_ids is None:
            self.stability_counter.zero_()
        else:
            self.stability_counter[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedEnv,
        object_cfg: SceneEntityCfg,
        gripper_cfg: SceneEntityCfg,
        collision_analyzer_cfg: CollisionAnalyzerCfg,
        max_pos_deviation: float = 0.05,
        pos_z_threshold: float = 0.05,
        consecutive_stability_steps: int = 5,
    ) -> torch.Tensor:
        # Get object and gripper from scene
        object_asset = env.scene[self.object_cfg.name]
        gripper_asset = env.scene[self.gripper_cfg.name]

        # Check time out
        time_out = env.episode_length_buf >= env.max_episode_length

        # Check for abnormal gripper state (excessive joint velocities)
        abnormal_gripper_state = (gripper_asset.data.joint_vel.abs() > (gripper_asset.data.joint_vel_limits * 2)).any(
            dim=1
        )

        # Check if asset velocities are small
        current_step_stable = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        # Check gripper (articulation) velocities
        current_step_stable &= gripper_asset.data.joint_vel.abs().sum(dim=1) < 5.0
        # Check object (rigid object) velocities
        if isinstance(object_asset, RigidObject):
            current_step_stable &= object_asset.data.body_lin_vel_w.abs().sum(dim=2).sum(dim=1) < 0.05
            current_step_stable &= object_asset.data.body_ang_vel_w.abs().sum(dim=2).sum(dim=1) < 1.0
        elif isinstance(object_asset, RigidObjectCollection):
            current_step_stable &= object_asset.data.object_lin_vel_w.abs().sum(dim=2).sum(dim=1) < 0.05
            current_step_stable &= object_asset.data.object_ang_vel_w.abs().sum(dim=2).sum(dim=1) < 1.0

        self.stability_counter = torch.where(
            current_step_stable,
            self.stability_counter + 1,  # Increment counter if stable
            torch.zeros_like(self.stability_counter),  # Reset counter if not stable
        )

        stability_reached = self.stability_counter >= self.consecutive_stability_steps

        # Skip if position or quaternion is NaN
        pos_is_nan = torch.isnan(object_asset.data.root_pos_w).any(dim=1)
        quat_is_nan = torch.isnan(object_asset.data.root_quat_w).any(dim=1)
        skip_check = pos_is_nan | quat_is_nan

        # Object has excessive pose deviation if position exceeds thresholds
        pos_deviation = (object_asset.data.root_pos_w - object_asset.initial_pos).norm(dim=1)
        valid_pos_deviation = torch.where(~skip_check, pos_deviation, torch.zeros_like(pos_deviation))
        excessive_pose_deviation = valid_pos_deviation > self.max_pos_deviation

        # Object is above ground if position is greater than z threshold
        pos_above_ground = object_asset.data.root_pos_w[:, 2] >= self.pos_z_threshold

        # Check for collisions between gripper and object
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        collision_free = self.collision_analyzer(env, all_env_ids)

        grasp_success = (
            (~abnormal_gripper_state)
            & stability_reached
            & (~excessive_pose_deviation)
            & pos_above_ground
            & collision_free
            & time_out
        )

        return grasp_success


class check_reset_state_success(ManagerTermBase):
    """Check if grasp is successful based on object stability, gripper closure, and collision detection."""

    _SUCCESS_STATE_ATTR = "_cage_reset_success_scene_state"
    _SUCCESS_STATE_VALID_ATTR = "_cage_reset_success_scene_state_valid"
    _SUCCESS_METADATA_ATTR = "_cage_reset_success_metadata"
    _SUCCESS_METADATA_TENSOR_ATTRS = (
        "_cage_adversary_gripper_action_target",
        "_cage_adversary_settling_robot_joint_pos_target",
        "_cage_adversary_settling_robot_joint_vel_target",
        "_cage_adversary_settling_robot_joint_target_valid",
        "_cage_grasp_survival_expected_obj_pos_ee",
        "_cage_grasp_survival_expected_obj_quat_ee",
        "_cage_grasp_survival_reference_valid",
        "_cage_reset_grasp_branch",
        "_cage_reset_branch_valid",
        POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_FAMILY_IDS_ATTR,
        POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_ROW_IDS_ATTR,
        POSE_DELTA_COMMITTED_ACTION_ATTR,
        POSE_DELTA_CANDIDATE_ACTION_ATTR,
        POSE_DELTA_CANDIDATE_VALID_ATTR,
    )

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._env = env

        self.object_cfgs = cfg.params.get("object_cfgs")
        self.robot_cfg = cfg.params.get("robot_cfg")
        self.ee_body_name = cfg.params.get("ee_body_name")
        self.collision_analyzer_cfgs = cfg.params.get("collision_analyzer_cfgs")
        self.collision_analyzers = [
            collision_analyzer_cfg.class_type(collision_analyzer_cfg, self._env)
            for collision_analyzer_cfg in self.collision_analyzer_cfgs
        ]
        self.max_robot_pos_deviation = cfg.params.get("max_robot_pos_deviation")
        self.max_object_pos_deviation = cfg.params.get("max_object_pos_deviation")
        self.pos_z_threshold = cfg.params.get("pos_z_threshold")
        self.consecutive_stability_steps = cfg.params.get("consecutive_stability_steps", 5)
        self.validate_grasp_survival = cfg.params.get("validate_grasp_survival", False)
        self.grasp_survival_pos_threshold = cfg.params.get("grasp_survival_pos_threshold", 0.03)
        self.grasp_survival_rot_threshold = cfg.params.get("grasp_survival_rot_threshold", 0.75)
        self.validate_family_semantics = cfg.params.get("validate_family_semantics", False)
        self.family_resting_rel_z_min = cfg.params.get("family_resting_rel_z_min", -0.05)
        self.family_resting_rel_z_max = cfg.params.get("family_resting_rel_z_max", 0.08)
        self.family_partial_rel_dist_max = cfg.params.get("family_partial_rel_dist_max", 0.12)
        self.family_partial_rel_xy_max = cfg.params.get("family_partial_rel_xy_max", 0.08)
        self.collision_check_chunk_size = int(cfg.params.get("collision_check_chunk_size", 0) or env.num_envs)
        if self.collision_check_chunk_size <= 0:
            self.collision_check_chunk_size = env.num_envs

        # Load gripper_approach_direction from metadata
        robot_asset = env.scene[self.robot_cfg.name]
        usd_path = robot_asset.cfg.spawn.usd_path
        metadata = utils.read_metadata_from_usd_directory(usd_path)
        self.gripper_approach_direction = tuple(metadata.get("gripper_approach_direction"))

        # Initialize stability counter for consecutive stability checking
        self.stability_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

        self.object_assets = [env.scene[cfg.name] for cfg in self.object_cfgs]
        self.robot_asset = env.scene[self.robot_cfg.name]
        self.assets_to_check = self.object_assets + [self.robot_asset]
        self.asset_names_to_check = [cfg.name for cfg in self.object_cfgs] + [self.robot_cfg.name]
        self.object_asset_by_name = dict(zip(self.asset_names_to_check, self.assets_to_check))
        self.insertive_asset = self.object_asset_by_name.get("insertive_object")
        self.receptive_asset = self.object_asset_by_name.get("receptive_object")
        self.ee_body_idx = self.robot_asset.data.body_names.index(self.ee_body_name)
        self._debug_gripper_joint_ids = [
            idx
            for idx, joint_name in enumerate(self.robot_asset.data.joint_names)
            if joint_name == "finger_joint" or "left" in joint_name or "right" in joint_name
        ]

        # Optional assembly alignment filter
        self.assembly_success_prob = cfg.params.get("assembly_success_prob")
        if self.assembly_success_prob is not None:
            insertive_asset_cfg = cfg.params.get("insertive_asset_cfg")
            receptive_asset_cfg = cfg.params.get("receptive_asset_cfg")
            self.insertive_asset = env.scene[insertive_asset_cfg.name]
            self.receptive_asset = env.scene[receptive_asset_cfg.name]

            insertive_meta = utils.read_metadata_from_usd_directory(self.insertive_asset.cfg.spawn.usd_path)
            receptive_meta = utils.read_metadata_from_usd_directory(self.receptive_asset.cfg.spawn.usd_path)
            self.insertive_asset_offset = Offset(
                pos=tuple(insertive_meta.get("assembled_offset").get("pos")),
                quat=tuple(insertive_meta.get("assembled_offset").get("quat")),
            )
            self.receptive_asset_offset = Offset(
                pos=tuple(receptive_meta.get("assembled_offset").get("pos")),
                quat=tuple(receptive_meta.get("assembled_offset").get("quat")),
            )
            assembly_threshold_scale = cfg.params.get("assembly_threshold_scale", 1.0)
            self.assembly_pos_threshold: float = (
                receptive_meta.get("success_thresholds").get("position") * assembly_threshold_scale
            )
            self.assembly_ori_threshold: float = (
                receptive_meta.get("success_thresholds").get("orientation") * assembly_threshold_scale
            )
            self.require_assembly_success = torch.rand(env.num_envs, device=env.device) < self.assembly_success_prob
            self._pending_reflip = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

        setattr(env, self._SUCCESS_STATE_VALID_ATTR, torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))

    def _grasp_tensor_attr(self, env: ManagerBasedEnv, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        value = getattr(env, name, None)
        if not isinstance(value, torch.Tensor) or value.shape != shape:
            raise RuntimeError(f"CAGE grasp survival validation expected env.{name} with shape {shape}.")
        return value

    def _empty_scene_state_like(self, value):
        if isinstance(value, dict):
            return {key: self._empty_scene_state_like(child) for key, child in value.items()}
        return torch.zeros_like(value)

    def _copy_scene_state_ids(self, dst, src, env_ids: torch.Tensor) -> None:
        if isinstance(src, dict):
            for key, child in src.items():
                if not isinstance(dst, dict) or key not in dst:
                    continue
                self._copy_scene_state_ids(dst[key], child, env_ids)
            return
        dst[env_ids] = src[env_ids].clone()

    def _cache_reset_success_state(self, env: ManagerBasedEnv, reset_success: torch.Tensor) -> None:
        success_ids = reset_success.nonzero(as_tuple=False).squeeze(-1)
        if success_ids.numel() == 0:
            return

        scene_state = env.scene.get_state(is_relative=True)
        cached_state = getattr(env, self._SUCCESS_STATE_ATTR, None)
        if not isinstance(cached_state, dict):
            cached_state = self._empty_scene_state_like(scene_state)
            setattr(env, self._SUCCESS_STATE_ATTR, cached_state)

        valid = getattr(env, self._SUCCESS_STATE_VALID_ATTR, None)
        if not isinstance(valid, torch.Tensor) or valid.shape != (env.num_envs,):
            valid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            setattr(env, self._SUCCESS_STATE_VALID_ATTR, valid)

        self._copy_scene_state_ids(cached_state, scene_state, success_ids)
        valid[success_ids] = True

        cached_metadata = getattr(env, self._SUCCESS_METADATA_ATTR, None)
        if not isinstance(cached_metadata, dict):
            cached_metadata = {}
            setattr(env, self._SUCCESS_METADATA_ATTR, cached_metadata)
        for attr_name in self._SUCCESS_METADATA_TENSOR_ATTRS:
            value = getattr(env, attr_name, None)
            if not isinstance(value, torch.Tensor) or value.shape[0] != env.num_envs:
                continue
            cached_value = cached_metadata.get(attr_name)
            if not isinstance(cached_value, torch.Tensor) or cached_value.shape != value.shape:
                cached_value = torch.zeros_like(value)
                cached_metadata[attr_name] = cached_value
            cached_value[success_ids.to(cached_value.device)] = value[success_ids.to(value.device)].to(
                device=cached_value.device, dtype=cached_value.dtype
            ).clone()

    def _settling_resolution_mask(self, env: ManagerBasedEnv, time_out: torch.Tensor) -> torch.Tensor:
        task_command = None
        try:
            task_command = env.command_manager.get_term("task_command")
        except (AttributeError, KeyError, ValueError):
            task_command = None
        live_mask = getattr(task_command, "_live_mask", None)
        if isinstance(live_mask, torch.Tensor) and live_mask.numel() == env.num_envs:
            return time_out & ~live_mask.to(device=env.device, dtype=torch.bool).view(-1)
        return time_out

    def _commit_accepted_family_lane_vectors(self, env: ManagerBasedEnv, accepted_ids: torch.Tensor) -> None:
        if accepted_ids.numel() == 0:
            return
        lane_vectors = getattr(env, POSE_DELTA_FAMILY_LANE_VECTORS_ATTR, None)
        lane_vector_valid = getattr(env, POSE_DELTA_FAMILY_LANE_VECTOR_VALID_ATTR, None)
        family_ids = getattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_FAMILY_IDS_ATTR, None)
        candidate = getattr(env, POSE_DELTA_CANDIDATE_ACTION_ATTR, None)
        if not (
            isinstance(lane_vectors, torch.Tensor)
            and lane_vectors.ndim == 3
            and lane_vectors.shape[:2] == (len(POSE_RESET_FAMILY_NAMES), env.num_envs)
            and isinstance(lane_vector_valid, torch.Tensor)
            and lane_vector_valid.shape == lane_vectors.shape[:2]
            and isinstance(family_ids, torch.Tensor)
            and family_ids.shape == (env.num_envs,)
            and isinstance(candidate, torch.Tensor)
            and candidate.shape == (env.num_envs, lane_vectors.shape[-1])
        ):
            return

        accepted_ids = accepted_ids.to(env.device)
        local_family_ids = family_ids[accepted_ids.to(family_ids.device)].to(env.device, dtype=torch.long)
        for family_idx in range(len(POSE_RESET_FAMILY_NAMES)):
            lane_ids = accepted_ids[local_family_ids == family_idx]
            if lane_ids.numel() == 0:
                continue
            lane_vectors[family_idx, lane_ids.to(lane_vectors.device)] = candidate[
                lane_ids.to(candidate.device)
            ].to(device=lane_vectors.device, dtype=lane_vectors.dtype)
            lane_vector_valid[family_idx, lane_ids.to(lane_vector_valid.device)] = True

    def _accepted_reset_diagnostics(
        self,
        env: ManagerBasedEnv,
        accepted_ids: torch.Tensor,
        previous: torch.Tensor,
        accepted: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        previous_lane = previous
        family_ids = getattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_FAMILY_IDS_ATTR, None)
        lane_vectors = getattr(env, POSE_DELTA_FAMILY_LANE_VECTORS_ATTR, None)
        if (
            isinstance(family_ids, torch.Tensor)
            and family_ids.shape == (env.num_envs,)
            and isinstance(lane_vectors, torch.Tensor)
            and lane_vectors.ndim == 3
            and lane_vectors.shape[:2] == (len(POSE_RESET_FAMILY_NAMES), env.num_envs)
            and lane_vectors.shape[-1] == accepted.shape[-1]
        ):
            local_family_ids = family_ids[accepted_ids.to(family_ids.device)].to(lane_vectors.device, dtype=torch.long)
            valid_family = (local_family_ids >= 0) & (local_family_ids < len(POSE_RESET_FAMILY_NAMES))
            if bool(valid_family.all().item()):
                previous_lane = lane_vectors[
                    local_family_ids,
                    accepted_ids.to(lane_vectors.device),
                ].to(device=accepted.device, dtype=accepted.dtype).detach().clone()

        diagnostics: dict[str, torch.Tensor] = {
            "accepted_reset_previous_lane_states": previous_lane,
            "accepted_reset_candidate_states": accepted,
            "accepted_reset_true_lane_deltas": accepted - previous_lane,
        }

        if isinstance(family_ids, torch.Tensor) and family_ids.shape == (env.num_envs,):
            diagnostics["accepted_reset_family_ids"] = family_ids[accepted_ids.to(family_ids.device)].detach().clone()

        row_ids = getattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_ROW_IDS_ATTR, None)
        if isinstance(row_ids, torch.Tensor) and row_ids.shape == (env.num_envs,):
            diagnostics["accepted_reset_row_ids"] = row_ids[accepted_ids.to(row_ids.device)].detach().clone()

        scripted_mask = getattr(env, POSE_DELTA_FAMILY_LANE_SCRIPTED_MASK_ATTR, None)
        if isinstance(scripted_mask, torch.Tensor) and scripted_mask.shape == (env.num_envs,):
            diagnostics["accepted_reset_is_scripted"] = scripted_mask[
                accepted_ids.to(scripted_mask.device)
            ].detach().clone()

        return diagnostics

    def _maybe_record_pose_delta_candidate_validation(
        self,
        env: ManagerBasedEnv,
        settling_resolved: torch.Tensor,
        settling_success: torch.Tensor,
        rejection_masks: dict[str, torch.Tensor],
    ) -> None:
        if not bool(getattr(env, POSE_DELTA_FAMILY_LANE_PERTURB_DEBUG_ATTR, False)):
            return
        stats = getattr(env, POSE_DELTA_FAMILY_LANE_PERTURB_STATS_ATTR, None)
        candidate_valid = getattr(env, POSE_DELTA_CANDIDATE_VALID_ATTR, None)
        family_ids = getattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_FAMILY_IDS_ATTR, None)
        invalid_scripted = getattr(env, POSE_DELTA_FAMILY_LANE_INVALID_MASK_ATTR, None)
        if not (
            isinstance(stats, dict)
            and isinstance(candidate_valid, torch.Tensor)
            and candidate_valid.shape == (env.num_envs,)
            and isinstance(family_ids, torch.Tensor)
            and family_ids.shape == (env.num_envs,)
        ):
            return

        resolved_candidate = settling_resolved & candidate_valid.to(device=env.device, dtype=torch.bool)
        accepted = settling_success & candidate_valid.to(device=env.device, dtype=torch.bool)
        rejected = resolved_candidate & ~accepted
        if not bool(resolved_candidate.any().item()):
            return

        accepted_counts = stats.get("accepted")
        rejected_counts = stats.get("rejected")
        invalid_accepted_counts = stats.get("invalid_accepted")
        invalid_rejected_counts = stats.get("invalid_rejected")
        family_ids = family_ids.to(device=env.device, dtype=torch.long)
        if isinstance(invalid_scripted, torch.Tensor) and invalid_scripted.shape == (env.num_envs,):
            invalid_scripted = invalid_scripted.to(device=env.device, dtype=torch.bool)
        else:
            invalid_scripted = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for family_idx in range(len(POSE_RESET_FAMILY_NAMES)):
            family_mask = family_ids == family_idx
            if isinstance(accepted_counts, torch.Tensor):
                accepted_counts[family_idx] += (accepted & family_mask).sum().to(accepted_counts.dtype)
            if isinstance(rejected_counts, torch.Tensor):
                rejected_counts[family_idx] += (rejected & family_mask).sum().to(rejected_counts.dtype)
            if isinstance(invalid_accepted_counts, torch.Tensor):
                invalid_accepted_counts[family_idx] += (accepted & invalid_scripted & family_mask).sum().to(
                    invalid_accepted_counts.dtype
                )
            if isinstance(invalid_rejected_counts, torch.Tensor):
                invalid_rejected_counts[family_idx] += (rejected & invalid_scripted & family_mask).sum().to(
                    invalid_rejected_counts.dtype
                )

        reason_counts = stats.setdefault("reject_reasons", {})
        if isinstance(reason_counts, dict):
            for reason, mask in rejection_masks.items():
                count = int((rejected & mask.to(device=env.device, dtype=torch.bool)).sum().item())
                if count > 0:
                    reason_counts[reason] = int(reason_counts.get(reason, 0)) + count

        if not (isinstance(accepted_counts, torch.Tensor) and isinstance(rejected_counts, torch.Tensor)):
            return
        total_resolved = int((accepted_counts.sum() + rejected_counts.sum()).item())
        last = int(stats.get("last_resolved_print", 0))
        interval = max(1, int(stats.get("print_interval", 20)))
        if total_resolved - last < interval:
            return
        stats["last_resolved_print"] = total_resolved
        parts = []
        for family_idx, reset_type in enumerate(POSE_RESET_FAMILY_NAMES):
            ok = int(accepted_counts[family_idx].item())
            bad = int(rejected_counts[family_idx].item())
            total = ok + bad
            pct = 100.0 * ok / max(total, 1)
            parts.append(f"{reset_type}={ok}/{total} ({pct:.1f}%)")
        if isinstance(reason_counts, dict) and len(reason_counts) > 0:
            top_reject = ", ".join(
                f"{reason}={count}" for reason, count in sorted(reason_counts.items(), key=lambda item: -item[1])[:4]
            )
        else:
            top_reject = "none"
        if isinstance(invalid_accepted_counts, torch.Tensor) and isinstance(invalid_rejected_counts, torch.Tensor):
            invalid_accepted_total = int(invalid_accepted_counts.sum().item())
            invalid_resolved_total = int((invalid_accepted_counts.sum() + invalid_rejected_counts.sum()).item())
            invalid_text = f" invalid_accept={invalid_accepted_total}/{invalid_resolved_total}"
        else:
            invalid_text = ""
        print(
            f"[CAGE family perturb] resolved total={total_resolved} "
            + ", ".join(parts)
            + f" top_reject={top_reject}"
            + invalid_text,
            flush=True,
        )

    def _commit_adversary_pose_candidate(self, env: ManagerBasedEnv, settling_success: torch.Tensor) -> None:
        if not bool(settling_success.any().item()):
            return

        committed = getattr(env, POSE_DELTA_COMMITTED_ACTION_ATTR, None)
        candidate = getattr(env, POSE_DELTA_CANDIDATE_ACTION_ATTR, None)
        candidate_valid = getattr(env, POSE_DELTA_CANDIDATE_VALID_ATTR, None)
        if not (
            isinstance(committed, torch.Tensor)
            and isinstance(candidate, torch.Tensor)
            and isinstance(candidate_valid, torch.Tensor)
            and committed.shape == candidate.shape
            and committed.ndim == 2
            and candidate_valid.shape == (env.num_envs,)
        ):
            return

        valid_candidate = candidate_valid.to(device=settling_success.device, dtype=torch.bool)
        invalid_scripted = getattr(env, POSE_DELTA_FAMILY_LANE_INVALID_MASK_ATTR, None)
        if isinstance(invalid_scripted, torch.Tensor) and invalid_scripted.shape == (env.num_envs,):
            valid_candidate &= ~invalid_scripted.to(device=settling_success.device, dtype=torch.bool)
        accepted_ids = (settling_success & valid_candidate).nonzero(as_tuple=False).squeeze(-1)
        if accepted_ids.numel() > 0:
            target_ids = accepted_ids.to(committed.device)
            source_ids = accepted_ids.to(candidate.device)
            previous = committed[target_ids].detach().clone()
            accepted = candidate[source_ids].to(device=committed.device, dtype=committed.dtype).detach().clone()
            records = getattr(env, POSE_DELTA_ACCEPTED_RECORDS_ATTR, None)
            if not isinstance(records, list):
                records = []
                setattr(env, POSE_DELTA_ACCEPTED_RECORDS_ATTR, records)
            records.append(
                {
                    "accepted_reset_env_ids": accepted_ids.detach().clone(),
                    "accepted_reset_states": accepted,
                    "accepted_reset_deltas": accepted - previous,
                    **self._accepted_reset_diagnostics(env, accepted_ids, previous, accepted),
                }
            )
            self._commit_accepted_family_lane_vectors(env, accepted_ids)
        commit_pose_delta_candidate_tensors(committed, candidate, candidate_valid, settling_success)

    def _apply_grasp_survival_filter(
        self, env: ManagerBasedEnv, reset_success: torch.Tensor
    ) -> torch.Tensor:
        self._last_grasp_survival_pos_err = torch.full((env.num_envs,), float("nan"), device=env.device)
        self._last_grasp_survival_rot_err = torch.full((env.num_envs,), float("nan"), device=env.device)

        if not self.validate_grasp_survival:
            return reset_success

        success_ids = reset_success.nonzero(as_tuple=False).squeeze(-1)
        if success_ids.numel() == 0:
            return reset_success

        branch = self._grasp_tensor_attr(env, "_cage_reset_grasp_branch", (env.num_envs,))
        branch_valid = self._grasp_tensor_attr(env, "_cage_reset_branch_valid", (env.num_envs,))
        if not bool(branch_valid[success_ids].all().item()):
            bad = success_ids[~branch_valid[success_ids]][:10].detach().cpu().tolist()
            raise RuntimeError(f"CAGE grasp survival validation missing branch labels for reset-success envs: {bad}")

        grasp_ids = success_ids[branch[success_ids]]
        if grasp_ids.numel() == 0:
            return reset_success

        pos_ref = self._grasp_tensor_attr(env, "_cage_grasp_survival_expected_obj_pos_ee", (env.num_envs, 3))
        quat_ref = self._grasp_tensor_attr(env, "_cage_grasp_survival_expected_obj_quat_ee", (env.num_envs, 4))
        ref_valid = self._grasp_tensor_attr(env, "_cage_grasp_survival_reference_valid", (env.num_envs,))
        if not bool(ref_valid[grasp_ids].all().item()):
            bad = grasp_ids[~ref_valid[grasp_ids]][:10].detach().cpu().tolist()
            raise RuntimeError(f"CAGE grasp survival validation missing object-in-EE references: {bad}")

        object_asset_name = getattr(env, "_cage_grasp_survival_object_asset_name")
        insertive_object = env.scene[object_asset_name]
        ee_pos_w = self.robot_asset.data.body_link_pos_w[grasp_ids, self.ee_body_idx]
        ee_quat_w = self.robot_asset.data.body_link_quat_w[grasp_ids, self.ee_body_idx]
        obj_pos_w = insertive_object.data.root_pos_w[grasp_ids]
        obj_quat_w = insertive_object.data.root_quat_w[grasp_ids]
        cur_obj_pos_ee, cur_obj_quat_ee = math_utils.subtract_frame_transforms(
            ee_pos_w, ee_quat_w, obj_pos_w, obj_quat_w
        )

        pos_err = torch.linalg.norm(cur_obj_pos_ee - pos_ref[grasp_ids].to(env.device), dim=-1)
        rot_err = math_utils.quat_error_magnitude(cur_obj_quat_ee, quat_ref[grasp_ids].to(env.device))
        self._last_grasp_survival_pos_err[grasp_ids] = pos_err
        self._last_grasp_survival_rot_err[grasp_ids] = rot_err
        survived = (pos_err < self.grasp_survival_pos_threshold) & (rot_err < self.grasp_survival_rot_threshold)

        filtered_success = reset_success.clone()
        filtered_success[grasp_ids] &= survived
        return filtered_success

    def _insertive_receptive_relative_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self.insertive_asset, RigidObject) or not isinstance(self.receptive_asset, RigidObject):
            raise RuntimeError("CAGE family semantic validation requires insertive_object and receptive_object assets.")
        return math_utils.subtract_frame_transforms(
            self.receptive_asset.data.root_pos_w,
            self.receptive_asset.data.root_quat_w,
            self.insertive_asset.data.root_pos_w,
            self.insertive_asset.data.root_quat_w,
        )

    def _apply_family_semantic_filter(
        self, env: ManagerBasedEnv, reset_success: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        empty = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        rejection_masks = {
            "family_branch": empty.clone(),
            "family_grasp_ref": empty.clone(),
            "family_resting": empty.clone(),
            "family_partial": empty.clone(),
        }
        if not self.validate_family_semantics:
            return reset_success, rejection_masks

        family_ids = getattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_FAMILY_IDS_ATTR, None)
        if not isinstance(family_ids, torch.Tensor) or family_ids.shape != (env.num_envs,):
            return reset_success, rejection_masks

        family_ids = family_ids.to(device=env.device, dtype=torch.long)
        valid_family = (family_ids >= 0) & (family_ids < len(POSE_RESET_FAMILY_NAMES))
        if not bool(valid_family.any().item()):
            return reset_success, rejection_masks

        branch = getattr(env, "_cage_reset_grasp_branch", None)
        branch_valid = getattr(env, "_cage_reset_branch_valid", None)
        ref_valid = getattr(env, "_cage_grasp_survival_reference_valid", None)
        if not (
            isinstance(branch, torch.Tensor)
            and isinstance(branch_valid, torch.Tensor)
            and isinstance(ref_valid, torch.Tensor)
            and branch.shape == (env.num_envs,)
            and branch_valid.shape == (env.num_envs,)
            and ref_valid.shape == (env.num_envs,)
        ):
            rejection_masks["family_branch"] |= valid_family
            return reset_success & ~rejection_masks["family_branch"], rejection_masks

        branch = branch.to(device=env.device, dtype=torch.bool)
        branch_valid = branch_valid.to(device=env.device, dtype=torch.bool)
        ref_valid = ref_valid.to(device=env.device, dtype=torch.bool)

        grasped_family = (
            (family_ids == POSE_RESET_FAMILY_NAMES.index("ObjectRestingEEGrasped"))
            | (family_ids == POSE_RESET_FAMILY_NAMES.index("ObjectAnywhereEEGrasped"))
            | (family_ids == POSE_RESET_FAMILY_NAMES.index("ObjectPartiallyAssembledEEGrasped"))
        )
        nongrasp_family = family_ids == POSE_RESET_FAMILY_NAMES.index("ObjectAnywhereEEAnywhere")

        rejection_masks["family_branch"] = valid_family & (~branch_valid | (branch != grasped_family))
        rejection_masks["family_grasp_ref"] = valid_family & grasped_family & ~ref_valid

        rel_pos, _ = self._insertive_receptive_relative_pose()
        rel_xy = torch.linalg.norm(rel_pos[:, :2], dim=1)
        rel_dist = torch.linalg.norm(rel_pos, dim=1)
        rel_z = rel_pos[:, 2]

        resting_family = family_ids == POSE_RESET_FAMILY_NAMES.index("ObjectRestingEEGrasped")
        resting_valid = (rel_z >= self.family_resting_rel_z_min) & (rel_z <= self.family_resting_rel_z_max)
        rejection_masks["family_resting"] = valid_family & resting_family & ~resting_valid

        partial_family = family_ids == POSE_RESET_FAMILY_NAMES.index("ObjectPartiallyAssembledEEGrasped")
        partial_valid = (rel_dist <= self.family_partial_rel_dist_max) & (rel_xy <= self.family_partial_rel_xy_max)
        rejection_masks["family_partial"] = valid_family & partial_family & ~partial_valid

        semantic_invalid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for mask in rejection_masks.values():
            semantic_invalid |= mask
        semantic_invalid &= valid_family | nongrasp_family
        return reset_success & ~semantic_invalid, rejection_masks

    def _compute_collision_free(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> torch.Tensor:
        if len(self.collision_analyzers) == 0:
            return torch.ones(len(env_ids), device=env.device, dtype=torch.bool)

        if len(env_ids) <= self.collision_check_chunk_size:
            return torch.all(
                torch.stack([collision_analyzer(env, env_ids) for collision_analyzer in self.collision_analyzers]),
                dim=0,
            )

        collision_free = torch.ones(len(env_ids), device=env.device, dtype=torch.bool)
        for start in range(0, len(env_ids), self.collision_check_chunk_size):
            end = min(start + self.collision_check_chunk_size, len(env_ids))
            chunk_env_ids = env_ids[start:end]
            collision_free[start:end] = torch.all(
                torch.stack([collision_analyzer(env, chunk_env_ids) for collision_analyzer in self.collision_analyzers]),
                dim=0,
            )
        return collision_free

    def _bootstrap_replay_row_text(self, env: ManagerBasedEnv, env_id: int) -> str:
        row_ids = getattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_ROW_IDS_ATTR, None)
        if not isinstance(row_ids, torch.Tensor) or row_ids.shape != (env.num_envs,):
            return " row=unknown"
        return f" row={int(row_ids[env_id].item())}"

    def _bootstrap_replay_family_geometry_text(self, env_id: int) -> str:
        if not self.validate_family_semantics:
            return ""
        if not isinstance(self.insertive_asset, RigidObject) or not isinstance(self.receptive_asset, RigidObject):
            return ""
        env_id_tensor = torch.tensor([env_id], device=self._env.device, dtype=torch.long)
        rel_pos, _ = math_utils.subtract_frame_transforms(
            self.receptive_asset.data.root_pos_w[env_id_tensor],
            self.receptive_asset.data.root_quat_w[env_id_tensor],
            self.insertive_asset.data.root_pos_w[env_id_tensor],
            self.insertive_asset.data.root_quat_w[env_id_tensor],
        )
        rel_xy = torch.linalg.norm(rel_pos[:, :2], dim=1)
        rel_dist = torch.linalg.norm(rel_pos, dim=1)
        return (
            f" family_rel_dist={float(rel_dist.item()):.4f}"
            f" family_rel_xy={float(rel_xy.item()):.4f}"
            f" family_rel_z={float(rel_pos[0, 2].item()):.4f}"
        )

    def _bootstrap_replay_gripper_text(self, env: ManagerBasedEnv, env_id: int) -> str:
        parts = []
        gripper_action_target = getattr(env, "_cage_adversary_gripper_action_target", None)
        if isinstance(gripper_action_target, torch.Tensor) and gripper_action_target.shape == (env.num_envs,):
            parts.append(f"gripper_action_target={float(gripper_action_target[env_id].item()):.3f}")

        target_joint_pos = getattr(env, "_cage_adversary_settling_robot_joint_pos_target", None)
        target_joint_valid = getattr(env, "_cage_adversary_settling_robot_joint_target_valid", None)
        if isinstance(target_joint_valid, torch.Tensor) and target_joint_valid.shape == (env.num_envs,):
            parts.append(f"joint_target_valid={bool(target_joint_valid[env_id].item())}")

        if (
            self._debug_gripper_joint_ids
            and isinstance(target_joint_pos, torch.Tensor)
            and target_joint_pos.shape == (env.num_envs, self.robot_asset.num_joints)
        ):
            joint_ids = torch.as_tensor(self._debug_gripper_joint_ids, device=env.device, dtype=torch.long)
            current = self.robot_asset.data.joint_pos[env_id, joint_ids]
            target = target_joint_pos[env_id, joint_ids].to(env.device)
            velocity = self.robot_asset.data.joint_vel[env_id, joint_ids]
            parts.append(f"gripper_joint_err={float((current - target).abs().amax().item()):.5f}")
            parts.append(f"gripper_joint_vel={float(velocity.abs().amax().item()):.5f}")
            parts.append(f"gripper_joint_cur_mean={float(current.mean().item()):.5f}")
            parts.append(f"gripper_joint_tgt_mean={float(target.mean().item()):.5f}")

        return (" " + " ".join(parts)) if parts else ""

    def _bootstrap_replay_velocity_text(self, env_id: int) -> str:
        parts = []
        for asset_name, asset in zip(self.asset_names_to_check, self.assets_to_check):
            if isinstance(asset, Articulation):
                joint_vel = asset.data.joint_vel[env_id].abs()
                parts.append(f"{asset_name}_jv_sum={float(joint_vel.sum().item()):.5f}")
                parts.append(f"{asset_name}_jv_max={float(joint_vel.amax().item()):.5f}")
            elif isinstance(asset, RigidObject):
                lin_vel = asset.data.body_lin_vel_w[env_id].abs()
                ang_vel = asset.data.body_ang_vel_w[env_id].abs()
                parts.append(f"{asset_name}_lin_sum={float(lin_vel.sum().item()):.5f}")
                parts.append(f"{asset_name}_lin_max={float(lin_vel.amax().item()):.5f}")
                parts.append(f"{asset_name}_ang_sum={float(ang_vel.sum().item()):.5f}")
                parts.append(f"{asset_name}_ang_max={float(ang_vel.amax().item()):.5f}")
            elif isinstance(asset, RigidObjectCollection):
                lin_vel = asset.data.object_lin_vel_w[env_id].abs()
                ang_vel = asset.data.object_ang_vel_w[env_id].abs()
                parts.append(f"{asset_name}_lin_sum={float(lin_vel.sum().item()):.5f}")
                parts.append(f"{asset_name}_lin_max={float(lin_vel.amax().item()):.5f}")
                parts.append(f"{asset_name}_ang_sum={float(ang_vel.sum().item()):.5f}")
                parts.append(f"{asset_name}_ang_max={float(ang_vel.amax().item()):.5f}")
        return (" " + " ".join(parts)) if parts else ""

    def _bootstrap_replay_grasp_reference_text(self, env: ManagerBasedEnv, env_id: int) -> str:
        branch = getattr(env, "_cage_reset_grasp_branch", None)
        branch_valid = getattr(env, "_cage_reset_branch_valid", None)
        ref_valid = getattr(env, "_cage_grasp_survival_reference_valid", None)
        parts = []
        if isinstance(branch_valid, torch.Tensor) and branch_valid.shape == (env.num_envs,):
            parts.append(f"branch_valid={bool(branch_valid[env_id].item())}")
        if isinstance(branch, torch.Tensor) and branch.shape == (env.num_envs,):
            parts.append(f"grasp_branch={bool(branch[env_id].item())}")
        if isinstance(ref_valid, torch.Tensor) and ref_valid.shape == (env.num_envs,):
            parts.append(f"ref_valid={bool(ref_valid[env_id].item())}")

        pos_ref = getattr(env, "_cage_grasp_survival_expected_obj_pos_ee", None)
        quat_ref = getattr(env, "_cage_grasp_survival_expected_obj_quat_ee", None)
        object_asset_name = getattr(env, "_cage_grasp_survival_object_asset_name", None)
        if (
            isinstance(ref_valid, torch.Tensor)
            and ref_valid.shape == (env.num_envs,)
            and bool(ref_valid[env_id].item())
            and isinstance(pos_ref, torch.Tensor)
            and pos_ref.shape == (env.num_envs, 3)
            and isinstance(quat_ref, torch.Tensor)
            and quat_ref.shape == (env.num_envs, 4)
            and isinstance(object_asset_name, str)
        ):
            env_id_tensor = torch.tensor([env_id], device=env.device, dtype=torch.long)
            insertive_object = env.scene[object_asset_name]
            ee_pos_w = self.robot_asset.data.body_link_pos_w[env_id_tensor, self.ee_body_idx]
            ee_quat_w = self.robot_asset.data.body_link_quat_w[env_id_tensor, self.ee_body_idx]
            obj_pos_w = insertive_object.data.root_pos_w[env_id_tensor]
            obj_quat_w = insertive_object.data.root_quat_w[env_id_tensor]
            cur_obj_pos_ee, cur_obj_quat_ee = math_utils.subtract_frame_transforms(
                ee_pos_w, ee_quat_w, obj_pos_w, obj_quat_w
            )
            pos_err = torch.linalg.norm(cur_obj_pos_ee - pos_ref[env_id_tensor].to(env.device), dim=-1)
            rot_err = math_utils.quat_error_magnitude(cur_obj_quat_ee, quat_ref[env_id_tensor].to(env.device))
            parts.append(f"ref_pos_err={float(pos_err.item()):.4f}")
            parts.append(f"ref_rot_err={float(rot_err.item()):.4f}")

        return (" " + " ".join(parts)) if parts else ""

    def _maybe_print_bootstrap_replay_failure_details(
        self,
        env: ManagerBasedEnv,
        stats: dict,
        resolved_mask: torch.Tensor,
        reset_success: torch.Tensor,
        rejection_masks: dict[str, torch.Tensor],
        family_ids: torch.Tensor,
        current_step_stable: torch.Tensor,
        gripper_approach_z: torch.Tensor,
        asset_pos_deviations: dict[str, torch.Tensor],
    ) -> None:
        max_detail_prints = 32
        detail_print_count = int(stats.get("detail_print_count", 0))
        if detail_print_count >= max_detail_prints:
            return

        failed_ids = (resolved_mask & ~reset_success).nonzero(as_tuple=False).squeeze(-1)
        if failed_ids.numel() == 0:
            return

        remaining = max_detail_prints - detail_print_count
        target_family = POSE_RESET_FAMILY_NAMES.index("ObjectAnywhereEEGrasped")
        failed_family_ids = family_ids[failed_ids.to(family_ids.device)].to(env.device, dtype=torch.long)
        target_failed_ids = failed_ids[failed_family_ids == target_family]
        other_failed_ids = failed_ids[failed_family_ids != target_family]
        failed_ids = torch.cat((target_failed_ids, other_failed_ids), dim=0)[: min(remaining, 8)]
        for env_id in failed_ids.detach().cpu().tolist():
            family_idx = int(family_ids[env_id].item())
            family_name = (
                POSE_RESET_FAMILY_NAMES[family_idx]
                if 0 <= family_idx < len(POSE_RESET_FAMILY_NAMES)
                else f"unknown:{family_idx}"
            )
            reasons = [
                reason
                for reason, reason_mask in rejection_masks.items()
                if bool(reason_mask[env_id].item())
            ]
            reason_text = ",".join(reasons) if reasons else "none"
            drift_parts = []
            for asset_name in ("robot", "insertive_object", "receptive_object"):
                drift = asset_pos_deviations.get(asset_name)
                if isinstance(drift, torch.Tensor):
                    drift_parts.append(f"{asset_name}_drift={float(drift[env_id].item()):.4f}")
            grasp_pos_err = getattr(self, "_last_grasp_survival_pos_err", None)
            grasp_rot_err = getattr(self, "_last_grasp_survival_rot_err", None)
            if isinstance(grasp_pos_err, torch.Tensor) and isinstance(grasp_rot_err, torch.Tensor):
                grasp_text = (
                    f" grasp_pos_err={float(grasp_pos_err[env_id].item()):.4f}"
                    f" grasp_rot_err={float(grasp_rot_err[env_id].item()):.4f}"
                )
            else:
                grasp_text = ""
            message = (
                "[CAGE bootstrap fail] "
                f"env={env_id} family={family_name} reasons={reason_text} "
                + self._bootstrap_replay_row_text(env, env_id)
                + " "
                f"stable_count={int(self.stability_counter[env_id].item())}/{self.consecutive_stability_steps} "
                f"current_stable={bool(current_step_stable[env_id].item())} "
                f"approach_z={float(gripper_approach_z[env_id].item()):.3f} "
                + " ".join(drift_parts)
                + grasp_text
                + self._bootstrap_replay_family_geometry_text(env_id)
                + self._bootstrap_replay_gripper_text(env, env_id)
                + self._bootstrap_replay_velocity_text(env_id)
                + self._bootstrap_replay_grasp_reference_text(env, env_id)
            )
            print(message, flush=True)
            detail_print_count += 1
        stats["detail_print_count"] = detail_print_count

    def _maybe_print_bootstrap_replay_validation(
        self,
        env: ManagerBasedEnv,
        resolved_mask: torch.Tensor,
        reset_success: torch.Tensor,
        rejection_masks: dict[str, torch.Tensor],
        current_step_stable: torch.Tensor,
        gripper_approach_z: torch.Tensor,
        asset_pos_deviations: dict[str, torch.Tensor],
    ) -> None:
        if not bool(getattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_DEBUG_ATTR, False)):
            return

        family_ids = getattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_FAMILY_IDS_ATTR, None)
        if not isinstance(family_ids, torch.Tensor) or family_ids.shape != (env.num_envs,):
            return

        resolved_ids = resolved_mask.nonzero(as_tuple=False).squeeze(-1)
        if resolved_ids.numel() == 0:
            return

        stats = getattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_STATS_ATTR, None)
        if not isinstance(stats, dict) or not stats:
            stats = {
                "resolved": torch.zeros(len(POSE_RESET_FAMILY_NAMES), device=env.device, dtype=torch.long),
                "success": torch.zeros(len(POSE_RESET_FAMILY_NAMES), device=env.device, dtype=torch.long),
                "rejections": {
                    name: torch.zeros(len(POSE_RESET_FAMILY_NAMES), device=env.device, dtype=torch.long)
                    for name in rejection_masks
                },
                "last_print_total": 0,
                "detail_print_count": 0,
            }
            setattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_STATS_ATTR, stats)
        else:
            for name in rejection_masks:
                if name not in stats["rejections"]:
                    stats["rejections"][name] = torch.zeros(
                        len(POSE_RESET_FAMILY_NAMES), device=env.device, dtype=torch.long
                    )

        ids = resolved_ids.to(env.device)
        families = family_ids[ids].to(env.device, dtype=torch.long)
        valid_family = (families >= 0) & (families < len(POSE_RESET_FAMILY_NAMES))
        ids = ids[valid_family]
        families = families[valid_family]
        if ids.numel() == 0:
            return

        success_values = reset_success[ids].to(env.device, dtype=torch.bool)
        for family_idx in range(len(POSE_RESET_FAMILY_NAMES)):
            mask = families == family_idx
            if not bool(mask.any().item()):
                continue
            stats["resolved"][family_idx] += mask.sum().to(torch.long)
            stats["success"][family_idx] += success_values[mask].sum().to(torch.long)
            for reason, reason_mask in rejection_masks.items():
                stats["rejections"][reason][family_idx] += reason_mask[ids][mask].sum().to(torch.long)

        total = int(stats["resolved"].sum().item())
        print_every = max(1, env.num_envs)
        if total - int(stats["last_print_total"]) < print_every:
            return
        stats["last_print_total"] = total

        self._maybe_print_bootstrap_replay_failure_details(
            env,
            stats,
            resolved_mask,
            reset_success,
            rejection_masks,
            family_ids,
            current_step_stable,
            gripper_approach_z,
            asset_pos_deviations,
        )

        success_total = int(stats["success"].sum().item())
        family_parts = []
        for family_idx, family_name in enumerate(POSE_RESET_FAMILY_NAMES):
            resolved = int(stats["resolved"][family_idx].item())
            success = int(stats["success"][family_idx].item())
            rate = 100.0 * success / max(resolved, 1)
            family_reasons = {
                reason: int(values[family_idx].item())
                for reason, values in stats["rejections"].items()
                if int(values[family_idx].item()) > 0
            }
            top_family_reasons = sorted(family_reasons.items(), key=lambda item: item[1], reverse=True)[:3]
            family_reason_text = ", ".join(
                f"{reason}={count}" for reason, count in top_family_reasons
            ) or "none"
            family_parts.append(f"{family_name}={success}/{resolved} ({rate:.1f}%, reject={family_reason_text})")

        reason_totals = {
            reason: int(values.sum().item())
            for reason, values in stats["rejections"].items()
            if int(values.sum().item()) > 0
        }
        top_reasons = sorted(reason_totals.items(), key=lambda item: item[1], reverse=True)[:4]
        reason_text = ", ".join(f"{reason}={count}" for reason, count in top_reasons) or "none"
        overall_rate = 100.0 * success_total / max(total, 1)
        print(
            f"[CAGE bootstrap validation] success={success_total}/{total} ({overall_rate:.1f}%) "
            + ", ".join(family_parts)
            + f" top_reject={reason_text}",
            flush=True,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)

        for asset in self.assets_to_check:
            if asset is self.robot_asset:
                asset_pos = asset.data.body_link_pos_w[:, self.ee_body_idx].clone()
            else:
                asset_pos = asset.data.root_pos_w.clone()
            if not hasattr(asset, "initial_pos") or env_ids is None:
                asset.initial_pos = asset_pos
            else:
                asset.initial_pos[env_ids] = asset_pos[env_ids].clone()

        if env_ids is None:
            self.stability_counter.zero_()
        else:
            self.stability_counter[env_ids] = 0

        if self.assembly_success_prob is not None:
            if env_ids is None:
                self.require_assembly_success = (
                    torch.rand(self._env.num_envs, device=self._env.device) < self.assembly_success_prob
                )
                self._pending_reflip.zero_()
            else:
                reflip_mask = self._pending_reflip[env_ids]
                if reflip_mask.any():
                    reflip_ids = env_ids[reflip_mask]
                    self.require_assembly_success[reflip_ids] = (
                        torch.rand(reflip_ids.shape[0], device=self._env.device) < self.assembly_success_prob
                    )
                self._pending_reflip[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedEnv,
        object_cfgs: list[SceneEntityCfg],
        robot_cfg: SceneEntityCfg,
        ee_body_name: str,
        collision_analyzer_cfgs: list[CollisionAnalyzerCfg],
        max_robot_pos_deviation: float = 0.1,
        max_object_pos_deviation: float = 0.1,
        pos_z_threshold: float = -0.01,
        consecutive_stability_steps: int = 5,
        insertive_asset_cfg: SceneEntityCfg | None = None,
        receptive_asset_cfg: SceneEntityCfg | None = None,
        assembly_success_prob: float | None = None,
        assembly_threshold_scale: float = 1.0,
        validate_grasp_survival: bool = False,
        grasp_survival_pos_threshold: float = 0.03,
        grasp_survival_rot_threshold: float = 0.75,
        validate_family_semantics: bool = False,
        family_resting_rel_z_min: float = -0.05,
        family_resting_rel_z_max: float = 0.08,
        family_partial_rel_dist_max: float = 0.12,
        family_partial_rel_xy_max: float = 0.08,
        collision_check_chunk_size: int = 0,
    ) -> torch.Tensor:

        # Check time out
        time_out = env.episode_length_buf >= env.max_episode_length

        # Check for abnormal gripper state (excessive joint velocities)
        abnormal_gripper_state = (
            self.robot_asset.data.joint_vel.abs() > (self.robot_asset.data.joint_vel_limits * 2)
        ).any(dim=1)

        # Check if gripper orientation is pointing downward within 60 degrees of vertical
        ee_quat = self.robot_asset.data.body_link_quat_w[:, self.ee_body_idx]
        gripper_approach_local = torch.tensor(
            self.gripper_approach_direction, device=env.device, dtype=torch.float32
        ).expand(env.num_envs, -1)
        gripper_approach_world = math_utils.quat_apply(ee_quat, gripper_approach_local)
        gripper_approach_z = gripper_approach_world[:, 2]
        gripper_orientation_within_range = (
            gripper_approach_z < -0.5
        )  # cos(60°) = 0.5, so z < -0.5 for 60° cone

        # Check if asset velocities are small
        current_step_stable = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        for asset in self.assets_to_check:
            if isinstance(asset, Articulation):
                current_step_stable &= asset.data.joint_vel.abs().sum(dim=1) < 5.0
            elif isinstance(asset, RigidObject):
                current_step_stable &= asset.data.body_lin_vel_w.abs().sum(dim=2).sum(dim=1) < 0.1
                current_step_stable &= asset.data.body_ang_vel_w.abs().sum(dim=2).sum(dim=1) < 1.0
            elif isinstance(asset, RigidObjectCollection):
                current_step_stable &= asset.data.object_lin_vel_w.abs().sum(dim=2).sum(dim=1) < 0.1
                current_step_stable &= asset.data.object_ang_vel_w.abs().sum(dim=2).sum(dim=1) < 1.0

        self.stability_counter = torch.where(
            current_step_stable,
            self.stability_counter + 1,  # Increment counter if stable
            torch.zeros_like(self.stability_counter),  # Reset counter if not stable
        )

        stability_reached = self.stability_counter >= self.consecutive_stability_steps

        # Reset initial positions on first check or after env reset
        excessive_pose_deviation = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        pos_below_threshold = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        asset_pos_deviations: dict[str, torch.Tensor] = {}
        for asset_name, asset in zip(self.asset_names_to_check, self.assets_to_check):
            if asset is self.robot_asset:
                asset_pos = asset.data.body_link_pos_w[:, self.ee_body_idx].clone()
            else:
                asset_pos = asset.data.root_pos_w.clone()

            # Skip if position or quaternion is NaN
            pos_is_nan = torch.isnan(asset.data.root_pos_w).any(dim=1)
            quat_is_nan = torch.isnan(asset.data.root_quat_w).any(dim=1)
            skip_check = pos_is_nan | quat_is_nan

            # Asset has excessive pose deviation if position exceeds thresholds
            pos_deviation = (asset_pos - asset.initial_pos).norm(dim=1)
            valid_pos_deviation = torch.where(~skip_check, pos_deviation, torch.zeros_like(pos_deviation))
            asset_pos_deviations[asset_name] = valid_pos_deviation
            if asset is self.robot_asset:
                excessive_pose_deviation |= valid_pos_deviation > self.max_robot_pos_deviation
            else:
                excessive_pose_deviation |= valid_pos_deviation > self.max_object_pos_deviation

            # Asset is above ground if position is greater than z threshold
            pos_below_threshold |= asset_pos[:, 2] < self.pos_z_threshold

        # Check for collisions between gripper and object
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        collision_free = self._compute_collision_free(env, all_env_ids)

        reset_success = (
            (~abnormal_gripper_state)
            & gripper_orientation_within_range
            & stability_reached
            & (~excessive_pose_deviation)
            & (~pos_below_threshold)
            & collision_free
            & time_out
        )

        if self.assembly_success_prob is not None:
            ins_pos_w, ins_quat_w = self.insertive_asset_offset.apply(self.insertive_asset)
            rec_pos_w, rec_quat_w = self.receptive_asset_offset.apply(self.receptive_asset)
            rel_pos, rel_quat = math_utils.subtract_frame_transforms(rec_pos_w, rec_quat_w, ins_pos_w, ins_quat_w)
            e_x, e_y, _ = math_utils.euler_xyz_from_quat(rel_quat)
            euler_xy_dist = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
            xyz_dist = torch.norm(rel_pos, dim=1)
            assembly_success = (xyz_dist < self.assembly_pos_threshold) & (euler_xy_dist < self.assembly_ori_threshold)
            assembly_match = torch.where(self.require_assembly_success, assembly_success, ~assembly_success)
            reset_success = reset_success & assembly_match
        else:
            assembly_match = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

        reset_success_before_grasp = reset_success.clone()
        reset_success_after_grasp = self._apply_grasp_survival_filter(env, reset_success)
        reset_success, family_rejection_masks = self._apply_family_semantic_filter(env, reset_success_after_grasp)
        settling_resolved = self._settling_resolution_mask(env, time_out)
        rejection_masks = {
            "abnormal_gripper": abnormal_gripper_state,
            "bad_gripper_orientation": ~gripper_orientation_within_range,
            "unstable": ~stability_reached,
            "pose_drift": excessive_pose_deviation,
            "z_below": pos_below_threshold,
            "collision": ~collision_free,
            "assembly": ~assembly_match,
            "grasp_survival": reset_success_before_grasp & ~reset_success_after_grasp,
        }
        rejection_masks.update(family_rejection_masks)
        invalid_scripted = getattr(env, POSE_DELTA_FAMILY_LANE_INVALID_MASK_ATTR, None)
        if isinstance(invalid_scripted, torch.Tensor) and invalid_scripted.shape == (env.num_envs,):
            invalid_scripted = invalid_scripted.to(device=env.device, dtype=torch.bool)
        else:
            invalid_scripted = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        rejection_masks["scripted_invalid"] = invalid_scripted
        reset_success = reset_success & ~invalid_scripted
        self._maybe_print_bootstrap_replay_validation(
            env,
            settling_resolved,
            reset_success,
            rejection_masks,
            current_step_stable,
            gripper_approach_z,
            asset_pos_deviations,
        )
        settling_success = reset_success & settling_resolved
        self._maybe_record_pose_delta_candidate_validation(
            env,
            settling_resolved,
            settling_success,
            rejection_masks,
        )
        self._commit_adversary_pose_candidate(env, settling_success)
        self._cache_reset_success_state(env, settling_success)

        if self.assembly_success_prob is not None:
            self._pending_reflip |= reset_success

        return reset_success


class check_obb_no_overlap_termination(ManagerTermBase):
    """Termination condition that checks if OBBs of two objects no longer overlap."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._env = env

        self.insertive_object_cfg = cfg.params.get("insertive_object_cfg")
        self.insertive_object = env.scene[self.insertive_object_cfg.name]

        self.enable_visualization = cfg.params.get("enable_visualization", False)

        # Initialize OBB computation cache and compute OBBs once
        self._bbox_cache = bounds_utils.create_bbox_cache()
        self._compute_object_obbs()

        # Initialize placeholders for initial poses (will be set in reset)
        self._insertive_initial_pos = None
        self._insertive_initial_quat = None

        # Store debug draw interface if visualization is enabled
        if self.enable_visualization:
            import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

            self._omni_debug_draw = omni_debug_draw
        else:
            self._omni_debug_draw = None

    def _compute_object_obbs(self):
        """Compute OBB for insertive object and convert to body frame."""
        # Get prim path (use env 0 as template)
        insertive_prim_path = self.insertive_object.cfg.prim_path.replace(".*", "0", 1)

        # Compute OBB in world frame using Isaac Sim's built-in functions
        insertive_centroid_world, insertive_axes_world, insertive_half_extents = bounds_utils.compute_obb(
            self._bbox_cache, insertive_prim_path
        )

        # Get current world pose of object (env 0) to convert OBB to body frame
        insertive_pos_world = self.insertive_object.data.root_pos_w[0]  # (3,)
        insertive_quat_world = self.insertive_object.data.root_quat_w[0]  # (4,)

        device = self._env.device

        # Convert world frame OBB data to torch tensors
        insertive_centroid_world_tensor = torch.tensor(insertive_centroid_world, device=device, dtype=torch.float32)
        insertive_axes_world_tensor = torch.tensor(insertive_axes_world, device=device, dtype=torch.float32)

        # Convert centroid from world frame to body frame
        insertive_centroid_body = math_utils.quat_apply_inverse(
            insertive_quat_world, insertive_centroid_world_tensor - insertive_pos_world
        )

        # Convert axes from world frame to body frame
        insertive_rot_matrix_world = math_utils.matrix_from_quat(insertive_quat_world.unsqueeze(0))[0]  # (3, 3)

        # Transform axes: R_world_to_body @ world_axes = R_world^T @ world_axes
        # Note: Isaac Sim's compute_obb returns axes as column vectors, so we need to transpose
        insertive_axes_body = torch.matmul(insertive_rot_matrix_world.T, insertive_axes_world_tensor.T).T

        # Cache OBB data in body frame as torch tensors on device for fast access
        self._insertive_obb_centroid = insertive_centroid_body
        self._insertive_obb_axes = insertive_axes_body
        self._insertive_obb_half_extents = torch.tensor(insertive_half_extents, device=device, dtype=torch.float32)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Store initial pose of insertive object when environments are reset."""
        super().reset(env_ids)

        insertive_pos = self.insertive_object.data.root_pos_w.clone()
        insertive_quat = self.insertive_object.data.root_quat_w.clone()

        if self._insertive_initial_pos is None or self._insertive_initial_quat is None or env_ids is None:
            # First time initialization or reset all environments
            self._insertive_initial_pos = insertive_pos
            self._insertive_initial_quat = insertive_quat
        else:
            # Update only the reset environments
            self._insertive_initial_pos[env_ids] = insertive_pos[env_ids]
            self._insertive_initial_quat[env_ids] = insertive_quat[env_ids]

    def _compute_obb_corners_batch(self, centroids, axes, half_extents):
        """
        Compute the 8 corners of Oriented Bounding Boxes for all environments using Isaac Sim's built-in function.

        Args:
            centroids: Centers of OBBs (num_envs, 3)
            axes: Orientation axes of OBBs (num_envs, 3, 3) - rows are the axes
            half_extents: Half extents of OBB along its axes (3,)

        Returns:
            corners: 8 corners of the OBBs (num_envs, 8, 3)
        """
        num_envs = centroids.shape[0]
        device = centroids.device

        # Convert torch tensors to numpy for Isaac Sim functions
        centroids_np = centroids.detach().cpu().numpy()
        axes_np = axes.detach().cpu().numpy()
        half_extents_np = half_extents.detach().cpu().numpy()

        # Compute corners for each environment using Isaac Sim's function
        all_corners = []
        for env_idx in range(num_envs):
            # Use Isaac Sim's get_obb_corners function
            corners_np = bounds_utils.get_obb_corners(
                centroids_np[env_idx], axes_np[env_idx], half_extents_np
            )  # (8, 3)
            all_corners.append(corners_np)

        # Convert back to torch tensor
        corners_tensor = torch.tensor(np.stack(all_corners), device=device, dtype=torch.float32)
        return corners_tensor  # (num_envs, 8, 3)

    def _visualize_bounding_boxes(self, env: ManagerBasedEnv):
        """Visualize oriented bounding boxes for initial and current insertive object positions using wireframe edges."""
        # Clear previous debug lines
        draw_interface = self._omni_debug_draw.acquire_debug_draw_interface()
        draw_interface.clear_lines()

        # Get current world poses of insertive object for all environments
        insertive_pos = self.insertive_object.data.root_pos_w  # (num_envs, 3)
        insertive_quat = self.insertive_object.data.root_quat_w  # (num_envs, 4)

        # Transform current insertive object OBB centroid from body frame to world coordinates for all environments
        insertive_obb_centroid_body = self._insertive_obb_centroid
        insertive_current_world_centroids = insertive_pos + math_utils.quat_apply(
            insertive_quat, insertive_obb_centroid_body.unsqueeze(0).expand(env.num_envs, -1)
        )  # (num_envs, 3)

        # Transform current insertive object OBB orientation from body frame to world coordinates for all environments
        insertive_current_rot_matrices = math_utils.matrix_from_quat(insertive_quat)  # (num_envs, 3, 3)
        insertive_obb_axes_body = self._insertive_obb_axes
        insertive_current_world_axes = torch.bmm(
            insertive_current_rot_matrices,
            insertive_obb_axes_body.unsqueeze(0).expand(env.num_envs, -1, -1).transpose(1, 2),
        ).transpose(
            1, 2
        )  # (num_envs, 3, 3)

        # Compute OBB corners for current position visualization using Isaac Sim's built-in function
        insertive_current_corners = self._compute_obb_corners_batch(
            insertive_current_world_centroids, insertive_current_world_axes, self._insertive_obb_half_extents
        )  # (num_envs, 8, 3)

        # Transform initial insertive object OBB centroid from body frame to world coordinates for all environments
        insertive_initial_world_centroids = self._insertive_initial_pos + math_utils.quat_apply(
            self._insertive_initial_quat, insertive_obb_centroid_body.unsqueeze(0).expand(env.num_envs, -1)
        )  # (num_envs, 3)

        # Transform initial insertive object OBB orientation from body frame to world coordinates
        insertive_initial_rot_matrices = math_utils.matrix_from_quat(self._insertive_initial_quat)  # (num_envs, 3, 3)
        insertive_initial_world_axes = torch.bmm(
            insertive_initial_rot_matrices,
            insertive_obb_axes_body.unsqueeze(0).expand(env.num_envs, -1, -1).transpose(1, 2),
        ).transpose(
            1, 2
        )  # (num_envs, 3, 3)

        # Compute OBB corners for initial position visualization using Isaac Sim's built-in function
        insertive_initial_corners = self._compute_obb_corners_batch(
            insertive_initial_world_centroids, insertive_initial_world_axes, self._insertive_obb_half_extents
        )  # (num_envs, 8, 3)

        # Draw wireframe boxes for each environment
        for env_idx in range(env.num_envs):
            # Draw current insertive object bounding box edges (blue)
            self._draw_obb_wireframe(
                insertive_current_corners[env_idx],  # (8, 3)
                color=(0.0, 0.5, 1.0, 1.0),  # Bright blue
                line_width=4.0,
                draw_interface=draw_interface,
            )

            # Draw initial insertive object bounding box edges (red)
            self._draw_obb_wireframe(
                insertive_initial_corners[env_idx],  # (8, 3)
                color=(1.0, 0.2, 0.0, 1.0),  # Bright red
                line_width=4.0,
                draw_interface=draw_interface,
            )

    def _draw_obb_wireframe(
        self, corners: torch.Tensor, color: tuple = (1.0, 1.0, 1.0, 1.0), line_width: float = 2.0, draw_interface=None
    ):
        """
        Draw wireframe edges of an oriented bounding box.

        Args:
            corners: 8 corners of the OBB (8, 3)
            color: RGBA color tuple for the lines
            line_width: Width of the lines
            draw_interface: Debug draw interface (optional, will acquire if not provided)
        """
        # Define the edges of a cube by connecting corner indices
        # Corners are ordered as: [0-3] bottom face, [4-7] top face
        edge_indices = [
            # Bottom face edges
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            # Top face edges
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            # Vertical edges connecting bottom to top
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
            # Diagonal X on top face (4-5-6-7)
            (4, 6),
            (5, 7),
            # Diagonal X on bottom face (0-1-2-3)
            (0, 2),
            (1, 3),
            # Diagonal X on front face (0-1-4-5)
            (0, 5),
            (1, 4),
            # Diagonal X on back face (2-3-6-7)
            (2, 7),
            (3, 6),
            # Diagonal X on left face (0-3-4-7)
            (0, 7),
            (3, 4),
            # Diagonal X on right face (1-2-5-6)
            (1, 6),
            (2, 5),
        ]

        # Create line segments for all edges
        line_starts = []
        line_ends = []

        for start_idx, end_idx in edge_indices:
            line_starts.append(corners[start_idx].cpu().numpy().tolist())
            line_ends.append(corners[end_idx].cpu().numpy().tolist())

        # Use provided interface or acquire new one
        if draw_interface is None:
            draw_interface = self._omni_debug_draw.acquire_debug_draw_interface()

        colors = [list(color)] * len(edge_indices)
        line_thicknesses = [line_width] * len(edge_indices)

        # Draw all edges at once
        draw_interface.draw_lines(line_starts, line_ends, colors, line_thicknesses)

    def __call__(
        self,
        env: ManagerBasedEnv,
        insertive_object_cfg: SceneEntityCfg,
        enable_visualization: bool = False,
    ) -> torch.Tensor:
        """Check if OBB overlap condition is violated between initial and current insertive object positions."""

        # Get current world poses of insertive object for all environments
        insertive_pos = self.insertive_object.data.root_pos_w  # (num_envs, 3)
        insertive_quat = self.insertive_object.data.root_quat_w  # (num_envs, 4)

        # Transform current insertive object centroid from body frame to world coordinates for all environments
        insertive_obb_centroid_body = self._insertive_obb_centroid
        insertive_current_world_centroids = insertive_pos + math_utils.quat_apply(
            insertive_quat, insertive_obb_centroid_body.unsqueeze(0).expand(env.num_envs, -1)
        )  # (num_envs, 3)

        # Transform initial insertive object centroid from body frame to world coordinates for all environments
        insertive_initial_world_centroids = self._insertive_initial_pos + math_utils.quat_apply(
            self._insertive_initial_quat, insertive_obb_centroid_body.unsqueeze(0).expand(env.num_envs, -1)
        )  # (num_envs, 3)

        # Transform OBB axes to world coordinates
        insertive_current_rot_matrices = math_utils.matrix_from_quat(insertive_quat)  # (num_envs, 3, 3)
        insertive_initial_rot_matrices = math_utils.matrix_from_quat(self._insertive_initial_quat)  # (num_envs, 3, 3)

        insertive_obb_axes_body = self._insertive_obb_axes

        # Transform axes from body frame to world frame: R @ body_axes for all environments
        # Since axes are stored as row vectors, we need to handle the transpose properly
        insertive_current_world_axes = torch.bmm(
            insertive_current_rot_matrices,
            insertive_obb_axes_body.unsqueeze(0).expand(env.num_envs, -1, -1).transpose(1, 2),
        ).transpose(
            1, 2
        )  # (num_envs, 3, 3)

        insertive_initial_world_axes = torch.bmm(
            insertive_initial_rot_matrices,
            insertive_obb_axes_body.unsqueeze(0).expand(env.num_envs, -1, -1).transpose(1, 2),
        ).transpose(
            1, 2
        )  # (num_envs, 3, 3)

        # Check OBB overlap between initial and current insertive object positions for all environments
        obb_overlap = _check_obb_overlap(
            insertive_current_world_centroids,
            insertive_current_world_axes,
            self._insertive_obb_half_extents,
            insertive_initial_world_centroids,
            insertive_initial_world_axes,
            self._insertive_obb_half_extents,
        )

        # Visualize bounding boxes if enabled
        if self.enable_visualization:
            self._visualize_bounding_boxes(env)

        return ~obb_overlap


def consecutive_success_state(env: ManagerBasedRLEnv, num_consecutive_successes: int = 10):
    # Get the progress context to access assets and offsets
    context_term = env.reward_manager.get_term_cfg("progress_context").func  # type: ignore
    continuous_success_counter = getattr(context_term, "continuous_success_counter")

    return continuous_success_counter >= num_consecutive_successes


def consecutive_success_state_with_min_length(
    env: ManagerBasedRLEnv, num_consecutive_successes: int = 10, min_episode_length: int = 0
):
    """Like consecutive_success_state but rejects episodes shorter than min_episode_length.

    Episodes that start already assembled will reach num_consecutive_successes quickly,
    but won't be marked as success until min_episode_length steps have passed.
    Combined with a separate early termination, these episodes get terminated as failures.
    """
    context_term = env.reward_manager.get_term_cfg("progress_context").func  # type: ignore
    continuous_success_counter = getattr(context_term, "continuous_success_counter")
    success = continuous_success_counter >= num_consecutive_successes
    if min_episode_length > 0:
        success = success & (env.episode_length_buf >= min_episode_length)
    return success


def early_success_termination(env: ManagerBasedRLEnv, num_consecutive_successes: int = 5, min_episode_length: int = 10):
    """Terminates episodes that achieve success before min_episode_length steps.

    Paired with consecutive_success_state_with_min_length as the 'success' term:
    that term gates success until min_episode_length, while this term terminates
    the episode early (as a non-success failure) to avoid wasting sim time.
    """
    context_term = env.reward_manager.get_term_cfg("progress_context").func  # type: ignore
    continuous_success_counter = getattr(context_term, "continuous_success_counter")
    is_successful = continuous_success_counter >= num_consecutive_successes
    is_too_short = env.episode_length_buf < min_episode_length
    return is_successful & is_too_short


def corrupted_camera_detected(
    env: ManagerBasedRLEnv, camera_names: list[str], std_threshold: float = 10.0
) -> torch.Tensor:
    """
    Detect corrupted camera images by checking if standard deviation is below threshold.

    Corrupted cameras typically show uniform gray/black images with very low variance.
    This function checks all specified cameras and returns True for environments where
    any camera shows corruption (std < threshold).

    Args:
        env: The environment instance.
        camera_names: List of camera sensor names to check (e.g., ["front_camera", "wrist_camera"]).
        std_threshold: Standard deviation threshold below which image is considered corrupted.
                      Default 10.0 is conservative - normal images have std > 20.

    Returns:
        Boolean tensor of shape (num_envs,) indicating which environments have corrupted cameras.
    """
    num_envs = env.num_envs
    device = env.device

    # Initialize as no corruption
    is_corrupted = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # Check each camera
    for camera_name in camera_names:
        # Get camera sensor from scene
        camera = env.scene[camera_name]

        # Get RGB data: shape (num_envs, height, width, 3)
        rgb_data = camera.data.output["rgb"]

        # Compute standard deviation across spatial and channel dimensions
        # Reshape to (num_envs, -1) to compute std per environment
        rgb_flat = rgb_data.reshape(num_envs, -1).float()
        std_per_env = torch.std(rgb_flat, dim=1)

        # Mark as corrupted if std is below threshold
        is_corrupted |= std_per_env < std_threshold

    return is_corrupted
