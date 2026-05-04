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
from .pose_delta_adversary import (
    POSE_DELTA_ACCEPTED_RECORDS_ATTR,
    POSE_DELTA_BASE_ACTION_ATTR,
    POSE_DELTA_BASE_VALID_ATTR,
    POSE_DELTA_CANDIDATE_ACTION_ATTR,
    POSE_DELTA_CANDIDATE_VALID_ATTR,
    POSE_DELTA_COMMITTED_ACTION_ATTR,
    POSE_DELTA_LAST_PHYSICAL_DELTA_ATTR,
    POSE_DELTA_RESET_TYPE_ATTR,
    POSE_RESET_TYPE_OBJECT_ANYWHERE_EE_ANYWHERE,
    POSE_RESET_TYPE_OBJECT_ANYWHERE_EE_GRASPED,
    POSE_RESET_TYPE_OBJECT_PARTIALLY_ASSEMBLED_EE_GRASPED,
    POSE_RESET_TYPE_OBJECT_RESTING_EE_GRASPED,
    POSE_RESET_TYPE_PROBS,
    POSE_RESET_DELTA_NAMES,
    POSE_RESET_STATE_NAMES,
    compute_pose_delta_candidate,
)


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


def _as_1d_action_tensor(value, device: torch.device, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence, got shape {tuple(tensor.shape)}.")
    return tensor


def _get_pose_delta_candidate_actions(env: ManagerBasedEnv, env_ids: torch.Tensor) -> torch.Tensor:
    candidate = getattr(env, POSE_DELTA_CANDIDATE_ACTION_ATTR, None)
    valid = getattr(env, POSE_DELTA_CANDIDATE_VALID_ATTR, None)
    if not (
        isinstance(candidate, torch.Tensor)
        and isinstance(valid, torch.Tensor)
        and candidate.ndim == 2
        and valid.ndim == 1
        and candidate.shape[0] == env.num_envs
        and valid.shape[0] == env.num_envs
    ):
        raise RuntimeError(
            "CAGE pose reset expected a prepared delta candidate. "
            "Add adversary_prepare_pose_delta_action before pose reset events."
        )

    target_ids = env_ids.to(candidate.device)
    if target_ids.numel() > 0 and not bool(valid[target_ids].all().item()):
        bad = env_ids[~valid[target_ids].to(env_ids.device)][:10].detach().cpu().tolist()
        raise RuntimeError(f"CAGE pose reset missing prepared delta candidates for envs: {bad}")
    return candidate[target_ids].to(device=env.device, dtype=torch.float32)


def _get_pose_delta_reset_types(env: ManagerBasedEnv, env_ids: torch.Tensor) -> torch.Tensor:
    reset_types = getattr(env, POSE_DELTA_RESET_TYPE_ATTR, None)
    if not isinstance(reset_types, torch.Tensor) or reset_types.shape != (env.num_envs,):
        raise RuntimeError("CAGE pose reset expected internal reset-type metadata.")
    return reset_types[env_ids.to(reset_types.device)].to(device=env.device, dtype=torch.long)


def _quat_to_euler_action(quat: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat)
    return torch.stack(
        [math_utils.wrap_to_pi(roll), math_utils.wrap_to_pi(pitch), math_utils.wrap_to_pi(yaw)],
        dim=-1,
    )


class adversary_prepare_pose_delta_action(ManagerTermBase):
    """Prepare continuous reset actions from raw adversary delta commands."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.action_name: str = cfg.params["action_name"]
        self.action_low = _as_1d_action_tensor(cfg.params["action_low"], env.device, "action_low")
        self.action_high = _as_1d_action_tensor(cfg.params["action_high"], env.device, "action_high")
        self.neutral_action = _as_1d_action_tensor(cfg.params["neutral_action"], env.device, "neutral_action")
        self.delta_scale = _as_1d_action_tensor(cfg.params["delta_scale"], env.device, "delta_scale")
        self.action_dim = int(self.action_low.numel())
        for name, tensor in (
            ("action_high", self.action_high),
            ("neutral_action", self.neutral_action),
            ("delta_scale", self.delta_scale),
        ):
            if tensor.shape != self.action_low.shape:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} must match action_low {tuple(self.action_low.shape)}."
                )
        if not bool((self.action_low <= self.action_high).all().item()):
            raise ValueError("action_low must be <= action_high for every CAGE pose adversary dimension.")

        self._committed_action = self.neutral_action.unsqueeze(0).repeat(env.num_envs, 1).clone()
        self._base_action = self._committed_action.clone()
        self._base_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._candidate_action = self._committed_action.clone()
        self._candidate_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._last_physical_delta = torch.zeros_like(self._committed_action)
        self._reset_type = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        setattr(env, POSE_DELTA_COMMITTED_ACTION_ATTR, self._committed_action)
        setattr(env, POSE_DELTA_BASE_ACTION_ATTR, self._base_action)
        setattr(env, POSE_DELTA_BASE_VALID_ATTR, self._base_valid)
        setattr(env, POSE_DELTA_CANDIDATE_ACTION_ATTR, self._candidate_action)
        setattr(env, POSE_DELTA_CANDIDATE_VALID_ATTR, self._candidate_valid)
        setattr(env, POSE_DELTA_LAST_PHYSICAL_DELTA_ATTR, self._last_physical_delta)
        setattr(env, POSE_DELTA_RESET_TYPE_ATTR, self._reset_type)
        setattr(env, "_cage_adversary_pose_prepare_event", self)

    def begin_semantic_base(self, env_ids: torch.Tensor) -> None:
        target_ids = env_ids.to(self._base_action.device)
        probs = torch.tensor(POSE_RESET_TYPE_PROBS, device=self._base_action.device, dtype=torch.float32)
        sampled = torch.multinomial(probs, target_ids.numel(), replacement=True)
        self._reset_type[target_ids] = sampled
        self._base_action[target_ids] = self.neutral_action.to(self._base_action.device)
        self._base_valid[target_ids] = False
        self._candidate_valid[target_ids] = False

    def finish_semantic_base(self, env_ids: torch.Tensor) -> None:
        self._base_valid[env_ids.to(self._base_valid.device)] = True

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        action_name: str = "adversaryaction",
        action_low: tuple[float, ...] = (),
        action_high: tuple[float, ...] = (),
        neutral_action: tuple[float, ...] = (),
        delta_scale: tuple[float, ...] = (),
    ) -> None:
        raw_actions = _get_action_term_raw_actions(env, self.action_name)
        env_ids_dev = env_ids.to(raw_actions.device)
        raw_delta = raw_actions[env_ids_dev, : self.action_dim].to(env.device, dtype=torch.float32)

        target_ids = env_ids.to(self._committed_action.device)
        base = self._base_action[target_ids]
        base_valid = self._base_valid[target_ids].to(device=base.device, dtype=torch.bool).unsqueeze(-1)
        committed = self._committed_action[target_ids]
        candidate_base = torch.where(base_valid, base, committed)
        candidate = compute_pose_delta_candidate(
            raw_delta,
            candidate_base,
            self.delta_scale,
            self.action_low,
            self.action_high,
        )
        self._candidate_action[target_ids] = candidate.to(self._candidate_action.device)
        self._candidate_valid[target_ids] = True


class CageAdversaryResetRuntimeHooks:
    """CAGE-specific runtime hooks used by the generic multi-agent runner."""

    _SUCCESS_STATE_ATTR = "_cage_reset_success_scene_state"
    _SUCCESS_STATE_VALID_ATTR = "_cage_reset_success_scene_state_valid"
    _SUCCESS_METADATA_ATTR = "_cage_reset_success_metadata"

    def __init__(self, env: ManagerBasedEnv, robot: Articulation):
        self.env = env
        self.robot = robot
        self._live_anchor_grasp_branch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._live_anchor_branch_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # Rows: non-grasp branch, grasp branch. Columns: done count, task-success count.
        self._branch_episode_stats = torch.zeros((2, 2), dtype=torch.float32, device=env.device)

    def _tensor_attr(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        value = getattr(self.env, name, None)
        if not isinstance(value, torch.Tensor) or value.shape != shape:
            raise RuntimeError(f"CAGE runtime hook expected env.{name} with shape {shape}.")
        return value

    def _clone_scene_state(self, state):
        if isinstance(state, dict):
            return {key: self._clone_scene_state(value) for key, value in state.items()}
        return state.clone()

    def _copy_scene_state_ids(self, dst, src, env_ids: torch.Tensor) -> None:
        if isinstance(src, dict):
            for key, value in src.items():
                self._copy_scene_state_ids(dst[key], value, env_ids)
            return
        target_ids = env_ids.to(dst.device)
        source_ids = env_ids.to(src.device)
        dst[target_ids] = src[source_ids].to(device=dst.device, dtype=dst.dtype).clone()

    def _restore_reset_success_metadata(self, env_ids: torch.Tensor) -> None:
        cached_metadata = getattr(self.env, self._SUCCESS_METADATA_ATTR, None)
        if not isinstance(cached_metadata, dict):
            return
        for attr_name, cached_value in cached_metadata.items():
            target = getattr(self.env, attr_name, None)
            if not (
                isinstance(target, torch.Tensor)
                and isinstance(cached_value, torch.Tensor)
                and target.shape == cached_value.shape
            ):
                continue
            target_ids = env_ids.to(target.device)
            source_ids = env_ids.to(cached_value.device)
            target[target_ids] = cached_value[source_ids].to(device=target.device, dtype=target.dtype).clone()

    def consume_reset_success_state(self, env_ids: torch.Tensor, fallback_state: dict) -> dict:
        """Return the termination-time reset state before Isaac's internal reset clears it."""
        if env_ids.numel() == 0:
            return fallback_state

        cached_state = getattr(self.env, self._SUCCESS_STATE_ATTR, None)
        valid = getattr(self.env, self._SUCCESS_STATE_VALID_ATTR, None)
        if not isinstance(cached_state, dict) or not isinstance(valid, torch.Tensor):
            return fallback_state
        if valid.numel() != self.env.num_envs:
            return fallback_state

        target_ids = env_ids.to(valid.device)
        hit_mask = valid[target_ids].to(dtype=torch.bool)
        hit_count = int(hit_mask.sum().item())
        if hit_count == 0:
            return fallback_state

        hit_ids = env_ids[hit_mask.to(env_ids.device)]
        merged_state = self._clone_scene_state(fallback_state)
        self._copy_scene_state_ids(merged_state, cached_state, hit_ids)
        self._restore_reset_success_metadata(hit_ids)
        valid[target_ids[hit_mask]] = False
        return merged_state

    @staticmethod
    def _reset_manager(manager, env_ids: torch.Tensor) -> bool:
        reset = getattr(manager, "reset", None)
        if not callable(reset):
            return False
        reset(env_ids)
        return True

    def prepare_adversary_proposal_context(self, env_ids: torch.Tensor) -> None:
        """Stage a physical semantic base before the runner samples adversary deltas."""
        if env_ids.numel() == 0:
            return
        target_ids = env_ids.to(self.env.device)
        prepare_event = getattr(self.env, "_cage_adversary_pose_prepare_event", None)
        receptive_event = getattr(self.env, "_cage_adversary_receptive_reset_event", None)
        insertive_event = getattr(self.env, "_cage_adversary_insertive_reset_event", None)
        ee_event = getattr(self.env, "_cage_adversary_ee_reset_event", None)
        missing = [
            name
            for name, value in (
                ("prepare_pose_delta_action", prepare_event),
                ("reset_receptive_object_pose", receptive_event),
                ("reset_insertive_object", insertive_event),
                ("reset_end_effector_pose", ee_event),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(f"CAGE adversary proposal staging missing events: {missing}")

        prepare_event.begin_semantic_base(target_ids)
        receptive_event.stage_semantic_base(self.env, target_ids)
        insertive_event.stage_semantic_base(self.env, target_ids)
        ee_event.stage_semantic_base(self.env, target_ids)
        prepare_event.finish_semantic_base(target_ids)
        self.env.scene.write_data_to_sim()

    def on_runner_state_written(self, env_ids: torch.Tensor) -> None:
        """Refresh IsaacLab managers after the runner restores a cached scene state."""
        if env_ids.numel() == 0:
            return
        target_ids = env_ids.to(self.env.device)

        observation_manager = getattr(self.env, "observation_manager", None)
        if not self._reset_manager(observation_manager, target_ids):
            history_buffers = getattr(observation_manager, "_group_obs_term_history_buffer", {})
            for group_buffers in history_buffers.values():
                for circular_buffer in group_buffers.values():
                    circular_buffer.reset(batch_ids=target_ids)

        action_manager = getattr(self.env, "action_manager", None)
        self._reset_manager(action_manager, target_ids)
        prev_action = getattr(action_manager, "_prev_action", None)
        if isinstance(prev_action, torch.Tensor):
            prev_action[target_ids.to(prev_action.device)] = 0.0

        termination_manager = getattr(self.env, "termination_manager", None)
        self._reset_manager(termination_manager, target_ids)

    def settling_gripper_targets(self, _default_action: float) -> torch.Tensor:
        target = self._tensor_attr("_cage_adversary_gripper_action_target", (self.env.num_envs,))
        return target.to(self.env.device, dtype=torch.float32).view(-1)

    def apply_settling_state_targets(self, env_ids: torch.Tensor, write_state: bool = True) -> None:
        if env_ids.numel() == 0:
            return
        expected_shape = (self.env.num_envs, self.robot.num_joints)
        pos_target = self._tensor_attr("_cage_adversary_settling_robot_joint_pos_target", expected_shape)
        vel_target = self._tensor_attr("_cage_adversary_settling_robot_joint_vel_target", expected_shape)
        valid = self._tensor_attr("_cage_adversary_settling_robot_joint_target_valid", (self.env.num_envs,))
        target_ids = env_ids.to(self.env.device)
        valid_mask = valid[target_ids].to(dtype=torch.bool)
        if not bool(valid_mask.all().item()):
            bad = target_ids[~valid_mask][:10].detach().cpu().tolist()
            raise RuntimeError(f"CAGE settling joint targets missing for SETTLING envs: {bad}")

        joint_pos = pos_target[target_ids].to(self.env.device, dtype=torch.float32)
        joint_vel = vel_target[target_ids].to(self.env.device, dtype=torch.float32)

        if write_state:
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=target_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=target_ids)
        self.robot.set_joint_velocity_target(joint_vel, env_ids=target_ids)
        self.env.scene.write_data_to_sim()

    def _reset_branch_values(self, env_ids: torch.Tensor) -> torch.Tensor:
        branch = self._tensor_attr("_cage_reset_grasp_branch", (self.env.num_envs,))
        valid = self._tensor_attr("_cage_reset_branch_valid", (self.env.num_envs,))
        target_ids = env_ids.to(branch.device)
        branch_valid = valid[target_ids].to(dtype=torch.bool)
        if not bool(branch_valid.all().item()):
            bad = target_ids[~branch_valid][:10].detach().cpu().tolist()
            raise RuntimeError(f"CAGE branch tracking missing reset labels for envs: {bad}")
        return branch[target_ids].to(self.env.device, dtype=torch.bool)

    def on_live_anchor_start(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        branch_values = self._reset_branch_values(env_ids)
        target_ids = env_ids.to(self.env.device)
        self._live_anchor_grasp_branch[target_ids] = branch_values
        self._live_anchor_branch_valid[target_ids] = True

    def on_live_anchor_clear(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._live_anchor_branch_valid[env_ids.to(self.env.device)] = False

    def on_live_episode_done(self, live_done_ids: torch.Tensor) -> None:
        if live_done_ids.numel() == 0:
            return

        env_ids = live_done_ids.to(self.env.device)
        tracked_ids = env_ids[self._live_anchor_branch_valid[env_ids]]
        if tracked_ids.numel() == 0:
            return

        task_success_values = self._tensor_attr("_cage_task_command_last_episode_success", (self.env.num_envs,))
        task_success_valid = self._tensor_attr("_cage_task_command_last_episode_success_valid", (self.env.num_envs,))
        success_ids = tracked_ids.to(task_success_valid.device)
        success_valid = task_success_valid[success_ids].to(self.env.device, dtype=torch.bool)
        if not bool(success_valid.all().item()):
            bad = tracked_ids[~success_valid][:10].detach().cpu().tolist()
            raise RuntimeError(f"CAGE branch tracking found LIVE dones without command success labels: {bad}")

        branches = self._live_anchor_grasp_branch[tracked_ids].to(torch.long)
        task_successes = task_success_values[success_ids].to(self.env.device, dtype=torch.float32)
        for branch_idx in (0, 1):
            mask = branches == branch_idx
            if not bool(mask.any().item()):
                continue
            self._branch_episode_stats[branch_idx, 0] += mask.sum().to(torch.float32)
            self._branch_episode_stats[branch_idx, 1] += task_successes[mask].sum()

    def consume_iter_metrics(self, is_distributed: bool = False) -> dict[str, float]:
        metrics: dict[str, float] = {}
        stats = self._branch_episode_stats.reshape(-1).clone()

        if is_distributed:
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

        self._branch_episode_stats.zero_()
        if float(stats.sum().item()) <= 0.0:
            return metrics

        episode_stats = stats.view(2, 2)
        branch_names = ("nongrasp", "grasp")
        for branch_idx, branch_name in enumerate(branch_names):
            count = episode_stats[branch_idx, 0]
            task_success = episode_stats[branch_idx, 1]
            if float(count.item()) > 0.0:
                metrics[f"BranchSuccess/{branch_name}_task_success_rate"] = float((task_success / count).item())

        total_count = episode_stats[:, 0].sum()
        total_task_success = episode_stats[:, 1].sum()
        if float(total_count.item()) > 0.0:
            metrics["BranchSuccess/overall_task_success_rate"] = float((total_task_success / total_count).item())
        return metrics

    def consume_adversary_hdf5_records(self) -> dict | None:
        records = getattr(self.env, POSE_DELTA_ACCEPTED_RECORDS_ATTR, None)
        if not isinstance(records, list) or len(records) == 0:
            return None
        setattr(self.env, POSE_DELTA_ACCEPTED_RECORDS_ATTR, [])

        datasets: dict[str, torch.Tensor] = {}
        for key in ("accepted_reset_env_ids", "accepted_reset_states", "accepted_reset_deltas"):
            values = [record[key] for record in records if isinstance(record.get(key), torch.Tensor)]
            if values:
                datasets[key] = torch.cat(values, dim=0).detach()
        if "accepted_reset_states" not in datasets or datasets["accepted_reset_states"].numel() == 0:
            return None

        return {
            "file_name": "accepted_reset_states.h5",
            "datasets": datasets,
            "attrs": {
                "record_type": "accepted_reset_states",
                "description": "Accepted reset-state records only; rejected adversary proposals are not written.",
                "state_names": POSE_RESET_STATE_NAMES,
                "delta_names": POSE_RESET_DELTA_NAMES,
            },
        }


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
        setattr(env, "_cage_adversary_receptive_reset_event", self)

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

    def _sample_receptive_action(self, env_ids: torch.Tensor) -> torch.Tensor:
        num_envs = env_ids.numel()
        values = torch.zeros((num_envs, 3), device=env_ids.device, dtype=torch.float32)
        values[:, 0] = math_utils.sample_uniform(
            self.pose_range[0, 0], self.pose_range[0, 1], (num_envs,), device=env_ids.device
        )
        values[:, 1] = math_utils.sample_uniform(
            self.pose_range[1, 0], self.pose_range[1, 1], (num_envs,), device=env_ids.device
        )
        values[:, 2] = math_utils.sample_uniform(
            self.pose_range[5, 0], self.pose_range[5, 1], (num_envs,), device=env_ids.device
        )
        return values

    def _write_receptive_from_action(self, env: ManagerBasedEnv, env_ids: torch.Tensor, action_values: torch.Tensor) -> None:
        x = torch.clamp(action_values[:, self.action_x_index], self.pose_range[0, 0], self.pose_range[0, 1])
        y = torch.clamp(action_values[:, self.action_y_index], self.pose_range[1, 0], self.pose_range[1, 1])
        yaw = torch.clamp(action_values[:, self.action_yaw_index], self.pose_range[5, 0], self.pose_range[5, 1])

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

    def stage_semantic_base(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        base = getattr(env, POSE_DELTA_BASE_ACTION_ATTR)
        target_ids = env_ids.to(base.device)
        receptive_action = self._sample_receptive_action(target_ids)
        base[target_ids, self.action_x_index] = receptive_action[:, 0].to(base.device)
        base[target_ids, self.action_y_index] = receptive_action[:, 1].to(base.device)
        base[target_ids, self.action_yaw_index] = receptive_action[:, 2].to(base.device)
        self._write_receptive_from_action(env, env_ids, base[target_ids].to(env.device))

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
        a = _get_pose_delta_candidate_actions(env, env_ids)
        self._write_receptive_from_action(env, env_ids, a)


class adversary_reset_insertive_object_pose_from_assembled_offset(ManagerTermBase):
    """Reset insertive object pose from staged semantic bases plus adversary offsets.

    Partially assembled semantics use the existing partial-assembly dataset as a
    base. Other semantics generate legal relative poses procedurally. The final
    reset uses the staged base plus the adversary's continuous delta.
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
        setattr(env, "_cage_adversary_insertive_reset_event", self)

    def _sample_anywhere_relative_pose(self, num_envs: int, device: torch.device) -> torch.Tensor:
        return math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (num_envs, 6), device=device)

    def _sample_resting_relative_pose(self, num_envs: int, device: torch.device) -> torch.Tensor:
        samples = self._sample_anywhere_relative_pose(num_envs, device)
        samples[:, 2] = self.ranges[2, 0]
        samples[:, 3] = 0.0
        samples[:, 4] = 0.0
        samples[:, 5] = math_utils.sample_uniform(self.ranges[5, 0], self.ranges[5, 1], (num_envs,), device=device)
        return samples

    def _sample_partial_assembly_relative_pose(self, num_envs: int, device: torch.device) -> torch.Tensor:
        assembly_indices = torch.randint(0, len(self.rel_positions), (num_envs,), device=device)
        rel_pos = self.rel_positions[assembly_indices].to(device)
        rel_euler = _quat_to_euler_action(self.rel_quaternions[assembly_indices].to(device))
        return torch.clamp(torch.cat([rel_pos, rel_euler], dim=-1), self.ranges[:, 0], self.ranges[:, 1])

    def stage_semantic_base(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        base = getattr(env, POSE_DELTA_BASE_ACTION_ATTR)
        reset_types = _get_pose_delta_reset_types(env, env_ids)
        target_ids = env_ids.to(base.device)
        base_values = torch.zeros((env_ids.numel(), 6), device=env.device, dtype=torch.float32)

        anywhere_mask = (reset_types == POSE_RESET_TYPE_OBJECT_ANYWHERE_EE_ANYWHERE) | (
            reset_types == POSE_RESET_TYPE_OBJECT_ANYWHERE_EE_GRASPED
        )
        if bool(anywhere_mask.any().item()):
            base_values[anywhere_mask] = self._sample_anywhere_relative_pose(
                int(anywhere_mask.sum().item()), env.device
            )

        resting_mask = reset_types == POSE_RESET_TYPE_OBJECT_RESTING_EE_GRASPED
        if bool(resting_mask.any().item()):
            base_values[resting_mask] = self._sample_resting_relative_pose(
                int(resting_mask.sum().item()), env.device
            )

        partial_mask = reset_types == POSE_RESET_TYPE_OBJECT_PARTIALLY_ASSEMBLED_EE_GRASPED
        if bool(partial_mask.any().item()):
            base_values[partial_mask] = self._sample_partial_assembly_relative_pose(
                int(partial_mask.sum().item()), env.device
            )

        si = self.action_start_index
        base[target_ids, si : si + 6] = base_values.to(base.device)
        self._write_insertive_from_action(env, env_ids, base[target_ids].to(env.device))

    def _write_insertive_from_action(
        self, env: ManagerBasedEnv, env_ids: torch.Tensor, action_values: torch.Tensor
    ) -> None:
        receptive_pos_w = self.receptive_object.data.root_pos_w[env_ids]
        receptive_quat_w = self.receptive_object.data.root_quat_w[env_ids]

        si = self.action_start_index
        clamped = torch.clamp(action_values[:, si : si + 6], self.ranges[:, 0], self.ranges[:, 1])
        rel_pos = clamped[:, 0:3]
        rel_quat = math_utils.quat_from_euler_xyz(clamped[:, 3], clamped[:, 4], clamped[:, 5])
        insertive_pos_w, insertive_quat_w = math_utils.combine_frame_transforms(
            receptive_pos_w, receptive_quat_w, rel_pos, rel_quat
        )

        self.insertive_object.write_root_state_to_sim(
            root_state=torch.cat(
                [
                    insertive_pos_w,
                    insertive_quat_w,
                    torch.zeros((env_ids.numel(), 6), device=env.device),
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )

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
        a = _get_pose_delta_candidate_actions(env, env_ids)
        self._write_insertive_from_action(env, env_ids, a)


class adversary_reset_end_effector_from_action(ManagerTermBase):
    """Reset end-effector pose and gripper from staged semantic bases plus adversary deltas.

    CAGE samples OmniReset-style reset semantics before adversary action sampling.
    Grasped semantics use a saved OmniReset grasp as the local base; non-grasp
    semantics use an adversary-perturbed EE offset relative to the insertive object.
    Downstream physics settling validates the final candidate.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.dataset_dir: str = cfg.params.get("dataset_dir")
        grasped_asset_cfg: SceneEntityCfg = cfg.params.get("grasped_asset_cfg")  # type: ignore
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg", SceneEntityCfg("robot"))
        gripper_cfg: SceneEntityCfg = cfg.params.get(
            "gripper_cfg", SceneEntityCfg("robot", joint_names=["finger_joint", ".*right.*", ".*left.*"])
        )

        # Pose range for clamping the adversary's 6D EE offset relative to insertive object
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b")  # type: ignore
        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.ranges = torch.tensor(range_list, device=env.device)

        # Gripper range
        gripper_range: tuple[float, float] = cfg.params.get("gripper_range", (0.0, 0.785398))
        self.gripper_range = torch.tensor(gripper_range, device=env.device)

        # Robot and IK solver
        self.robot_asset_name: str = robot_ik_cfg.name
        self.grasped_asset_name: str = grasped_asset_cfg.name
        if robot_ik_cfg.body_names is None:
            raise ValueError("adversary_reset_end_effector_from_action requires robot_ik_cfg.body_names.")
        self.ee_body_name: str = (
            robot_ik_cfg.body_names if isinstance(robot_ik_cfg.body_names, str) else robot_ik_cfg.body_names[0]
        )
        self.robot: Articulation = env.scene[robot_ik_cfg.name]
        setattr(env, "_multi_agent_runner_hooks", CageAdversaryResetRuntimeHooks(env, self.robot))
        setattr(env, "_cage_grasp_survival_robot_asset_name", self.robot_asset_name)
        setattr(env, "_cage_grasp_survival_object_asset_name", self.grasped_asset_name)
        setattr(env, "_cage_grasp_survival_ee_body_name", self.ee_body_name)
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
        self.grasped_asset: RigidObject = env.scene[grasped_asset_cfg.name]
        self.grasp_dataset_path = self._compute_grasp_dataset_path()
        self._load_and_precompute_grasps(env)

        # Action indices
        self.action_name: str = cfg.params["action_name"]
        self.action_start_index: int = cfg.params.get("action_start_index", 9)

        self._settling_gripper_target = -torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
        self._settling_robot_joint_pos_target = self.robot.data.default_joint_pos.to(
            env.device, dtype=torch.float32
        ).clone()
        self._settling_robot_joint_vel_target = torch.zeros(
            (env.num_envs, self.robot.num_joints), device=env.device, dtype=torch.float32
        )
        self._settling_robot_joint_target_valid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._grasp_survival_expected_obj_pos_ee = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
        self._grasp_survival_expected_obj_quat_ee = torch.zeros(env.num_envs, 4, device=env.device, dtype=torch.float32)
        self._grasp_survival_expected_obj_quat_ee[:, 0] = 1.0
        self._grasp_survival_reference_valid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._reset_grasp_branch = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._reset_branch_valid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._reset_grasp_index = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        setattr(env, "_cage_adversary_gripper_action_target", self._settling_gripper_target)
        setattr(env, "_cage_adversary_settling_robot_joint_pos_target", self._settling_robot_joint_pos_target)
        setattr(env, "_cage_adversary_settling_robot_joint_vel_target", self._settling_robot_joint_vel_target)
        setattr(env, "_cage_adversary_settling_robot_joint_target_valid", self._settling_robot_joint_target_valid)
        setattr(env, "_cage_grasp_survival_expected_obj_pos_ee", self._grasp_survival_expected_obj_pos_ee)
        setattr(env, "_cage_grasp_survival_expected_obj_quat_ee", self._grasp_survival_expected_obj_quat_ee)
        setattr(env, "_cage_grasp_survival_reference_valid", self._grasp_survival_reference_valid)
        setattr(env, "_cage_reset_grasp_branch", self._reset_grasp_branch)
        setattr(env, "_cage_reset_branch_valid", self._reset_branch_valid)
        setattr(env, "_cage_adversary_ee_reset_event", self)

    def _compute_grasp_dataset_path(self) -> str:
        usd_path = self.grasped_asset.cfg.spawn.usd_path
        obj_name = utils.object_name_from_usd(usd_path)
        return f"{self.dataset_dir}/Grasps/{obj_name}/grasps.pt"

    def _load_and_precompute_grasps(self, env: ManagerBasedEnv) -> None:
        """Load successful grasp relative poses and gripper joint states."""
        local_path = utils.safe_retrieve_file_path(self.grasp_dataset_path)
        data = torch.load(local_path, map_location="cpu")
        grasp_group = data.get("grasp_relative_pose", data)

        rel_pos_list = grasp_group.get("relative_position", [])
        rel_quat_list = grasp_group.get("relative_orientation", [])
        gripper_joint_positions_dict = grasp_group.get("gripper_joint_positions", {})

        num_grasps = len(rel_pos_list)
        if num_grasps == 0:
            raise ValueError(f"No grasp data found in {self.grasp_dataset_path}")

        self.rel_positions = torch.stack(
            [
                (pos if isinstance(pos, torch.Tensor) else torch.as_tensor(pos, dtype=torch.float32))
                for pos in rel_pos_list
            ],
            dim=0,
        ).to(env.device, dtype=torch.float32)
        self.rel_quaternions = torch.stack(
            [
                (quat if isinstance(quat, torch.Tensor) else torch.as_tensor(quat, dtype=torch.float32))
                for quat in rel_quat_list
            ],
            dim=0,
        ).to(env.device, dtype=torch.float32)

        if isinstance(self.gripper_joint_ids, slice):
            gripper_joint_list = list(range(self.robot.num_joints))[self.gripper_joint_ids]
        else:
            gripper_joint_list = list(self.gripper_joint_ids)
        self.gripper_joint_list = gripper_joint_list

        self.gripper_joint_positions = torch.zeros(
            (num_grasps, len(gripper_joint_list)), device=env.device, dtype=torch.float32
        )
        for gripper_idx, robot_joint_idx in enumerate(gripper_joint_list):
            joint_name = self.robot.joint_names[robot_joint_idx]
            joint_series = gripper_joint_positions_dict.get(joint_name, [0.0] * num_grasps)
            joint_tensor = torch.stack(
                [(j if isinstance(j, torch.Tensor) else torch.as_tensor(j, dtype=torch.float32)) for j in joint_series],
                dim=0,
            ).to(env.device, dtype=torch.float32)
            self.gripper_joint_positions[:, gripper_idx] = joint_tensor

        self.gripper_open_positions = self.robot.data.default_joint_pos[0, gripper_joint_list].to(
            env.device, dtype=torch.float32
        ).clone()
        closed_sign = torch.sign(self.gripper_joint_positions.mean(dim=0))
        closed_sign = torch.where(closed_sign == 0.0, torch.ones_like(closed_sign), closed_sign)
        closed_magnitude = torch.full_like(self.gripper_open_positions, float(self.gripper_range[1].item()))
        self.gripper_closed_positions = closed_sign * closed_magnitude

    def _store_settling_gripper_targets(self, env_ids: torch.Tensor, gripper_action_targets: torch.Tensor) -> None:
        env_ids = env_ids.to(self._settling_gripper_target.device)
        self._settling_gripper_target[env_ids] = gripper_action_targets.to(
            self._settling_gripper_target.device, dtype=self._settling_gripper_target.dtype
        )

    def _store_settling_robot_joint_targets(
        self,
        env_ids: torch.Tensor,
        joint_position_targets: torch.Tensor,
        joint_velocity_targets: torch.Tensor,
    ) -> None:
        env_ids = env_ids.to(self._settling_robot_joint_pos_target.device)
        self._settling_robot_joint_pos_target[env_ids] = joint_position_targets.to(
            self._settling_robot_joint_pos_target.device, dtype=self._settling_robot_joint_pos_target.dtype
        )
        self._settling_robot_joint_vel_target[env_ids] = joint_velocity_targets.to(
            self._settling_robot_joint_vel_target.device, dtype=self._settling_robot_joint_vel_target.dtype
        )
        self._settling_robot_joint_target_valid[env_ids] = True

    def _clear_grasp_survival_reference(self, env_ids: torch.Tensor) -> None:
        self._grasp_survival_reference_valid[env_ids.to(self._grasp_survival_reference_valid.device)] = False

    def _store_grasp_survival_reference(
        self,
        grasp_ids: torch.Tensor,
        expected_obj_pos_ee: torch.Tensor,
        expected_obj_quat_ee: torch.Tensor,
    ) -> None:
        grasp_ids = grasp_ids.to(self._grasp_survival_reference_valid.device)
        self._grasp_survival_expected_obj_pos_ee[grasp_ids] = expected_obj_pos_ee.to(
            self._grasp_survival_expected_obj_pos_ee.device, dtype=self._grasp_survival_expected_obj_pos_ee.dtype
        )
        self._grasp_survival_expected_obj_quat_ee[grasp_ids] = expected_obj_quat_ee.to(
            self._grasp_survival_expected_obj_quat_ee.device, dtype=self._grasp_survival_expected_obj_quat_ee.dtype
        )
        self._grasp_survival_reference_valid[grasp_ids] = True

    def _store_reset_branch_labels(self, env_ids: torch.Tensor, grasped_mask: torch.Tensor) -> None:
        env_ids = env_ids.to(self._reset_grasp_branch.device)
        self._reset_grasp_branch[env_ids] = grasped_mask.to(self._reset_grasp_branch.device, dtype=torch.bool)
        self._reset_branch_valid[env_ids] = True

    def _sample_anywhere_ee_pose(self, num_envs: int, device: torch.device) -> torch.Tensor:
        return math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (num_envs, 6), device=device)

    def _sample_grasp_ee_pose(self, env_ids: torch.Tensor) -> torch.Tensor:
        grasp_indices = torch.randint(0, len(self.rel_positions), (env_ids.numel(),), device=env_ids.device)
        self._reset_grasp_index[env_ids.to(self._reset_grasp_index.device)] = grasp_indices.to(
            self._reset_grasp_index.device
        )
        rel_pos = self.rel_positions[grasp_indices].to(env_ids.device)
        rel_euler = _quat_to_euler_action(self.rel_quaternions[grasp_indices].to(env_ids.device))
        return torch.clamp(torch.cat([rel_pos, rel_euler], dim=-1), self.ranges[:, 0], self.ranges[:, 1])

    def stage_semantic_base(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        base = getattr(env, POSE_DELTA_BASE_ACTION_ATTR)
        reset_types = _get_pose_delta_reset_types(env, env_ids)
        target_ids = env_ids.to(base.device)
        base_values = torch.zeros((env_ids.numel(), 6), device=env.device, dtype=torch.float32)

        nongrasp_mask = reset_types == POSE_RESET_TYPE_OBJECT_ANYWHERE_EE_ANYWHERE
        if bool(nongrasp_mask.any().item()):
            base_values[nongrasp_mask] = self._sample_anywhere_ee_pose(int(nongrasp_mask.sum().item()), env.device)

        grasped_mask = ~nongrasp_mask
        if bool(grasped_mask.any().item()):
            grasp_ids = env_ids[grasped_mask]
            base_values[grasped_mask] = self._sample_grasp_ee_pose(grasp_ids.to(env.device))

        si = self.action_start_index
        base[target_ids, si : si + 6] = base_values.to(base.device)
        self._write_ee_from_action(env, env_ids, base[target_ids].to(env.device), reset_types)

    def _write_ee_from_action(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        action_values: torch.Tensor,
        reset_types: torch.Tensor,
    ) -> None:
        num_envs = env_ids.numel()
        si = self.action_start_index
        clamped = torch.clamp(action_values[:, si : si + 6], self.ranges[:, 0], self.ranges[:, 1])
        rel_quat = math_utils.quat_from_euler_xyz(clamped[:, 3], clamped[:, 4], clamped[:, 5])
        ee_pos_w, ee_quat_w = math_utils.combine_frame_transforms(
            self.grasped_asset.data.root_pos_w[env_ids],
            self.grasped_asset.data.root_quat_w[env_ids],
            clamped[:, 0:3],
            rel_quat,
        )

        pos_b, quat_b = self.solver._compute_frame_pose()
        pos_b[env_ids], quat_b[env_ids] = math_utils.subtract_frame_transforms(
            self.robot.data.root_link_pos_w[env_ids],
            self.robot.data.root_link_quat_w[env_ids],
            ee_pos_w,
            ee_quat_w,
        )

        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))

        for _ in range(25):
            self.solver.apply_actions()
            delta_joint_pos = 0.25 * (self.robot.data.joint_pos_target[env_ids] - self.robot.data.joint_pos[env_ids])
            self.robot.write_joint_state_to_sim(
                position=(delta_joint_pos + self.robot.data.joint_pos[env_ids])[:, self.joint_ids],
                velocity=torch.zeros((num_envs, self.n_joints), device=env.device),
                joint_ids=self.joint_ids,
                env_ids=env_ids,
            )

        grasped_mask = reset_types != POSE_RESET_TYPE_OBJECT_ANYWHERE_EE_ANYWHERE
        self._store_reset_branch_labels(env_ids, grasped_mask)
        self._clear_grasp_survival_reference(env_ids)

        num_gripper_joints = self.gripper_joint_positions.shape[1]
        gripper_positions = self.gripper_open_positions.unsqueeze(0).expand(num_envs, -1).clone()
        gripper_action_targets = torch.ones(num_envs, device=env.device, dtype=torch.float32)

        if bool(grasped_mask.any().item()):
            grasp_local = grasped_mask.nonzero(as_tuple=False).squeeze(-1)
            grasp_ids = env_ids[grasp_local]
            grasp_indices = self._reset_grasp_index[grasp_ids.to(self._reset_grasp_index.device)].to(env.device)
            valid_grasp = grasp_indices >= 0
            if not bool(valid_grasp.all().item()):
                bad = grasp_ids[~valid_grasp][:10].detach().cpu().tolist()
                raise RuntimeError(f"CAGE staged grasp reset missing grasp indices for envs: {bad}")
            gripper_positions[grasp_local] = self.gripper_joint_positions[grasp_indices]
            gripper_action_targets[grasp_local] = -1.0

            expected_obj_pos_ee, expected_obj_quat_ee = math_utils.subtract_frame_transforms(
                ee_pos_w[grasp_local],
                ee_quat_w[grasp_local],
                self.grasped_asset.data.root_pos_w[grasp_ids],
                self.grasped_asset.data.root_quat_w[grasp_ids],
            )
            self._store_grasp_survival_reference(grasp_ids, expected_obj_pos_ee, expected_obj_quat_ee)

        self.robot.write_joint_state_to_sim(
            position=gripper_positions,
            velocity=torch.zeros_like(gripper_positions),
            joint_ids=self.gripper_joint_ids,
            env_ids=env_ids,
        )

        full_joint_pos = self.robot.data.joint_pos[env_ids].clone()
        full_joint_pos[:, self.gripper_joint_list] = gripper_positions
        full_joint_vel = torch.zeros_like(full_joint_pos)
        self.robot.set_joint_position_target(full_joint_pos, env_ids=env_ids)
        self.robot.set_joint_velocity_target(full_joint_vel, env_ids=env_ids)

        self._store_settling_gripper_targets(env_ids, gripper_action_targets)
        self._store_settling_robot_joint_targets(env_ids, full_joint_pos, full_joint_vel)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        dataset_dir: str = "",
        grasped_asset_cfg: SceneEntityCfg = None,
        pose_range_b: dict[str, tuple[float, float]] = dict(),
        robot_ik_cfg: SceneEntityCfg = None,
        gripper_cfg: SceneEntityCfg = None,
        gripper_range: tuple[float, float] = (0.0, 0.785398),
        action_name: str = "adversaryaction",
        action_start_index: int = 9,
    ) -> None:
        a = _get_pose_delta_candidate_actions(env, env_ids)
        reset_types = _get_pose_delta_reset_types(env, env_ids)
        self._write_ee_from_action(env, env_ids, a, reset_types)
        env.scene.write_data_to_sim()
