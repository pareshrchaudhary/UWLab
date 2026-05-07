# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adversary event functions for CAGE manipulation tasks."""

import os

import numpy as np
import scipy.stats as stats
import torch

import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from uwlab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from uwlab_tasks.manager_based.manipulation.cage.mdp import utils

from .pose_delta_adversary import (
    POSE_DELTA_ACCEPTED_RECORDS_ATTR,
    POSE_DELTA_CANDIDATE_ACTION_ATTR,
    POSE_DELTA_CANDIDATE_VALID_ATTR,
    POSE_DELTA_COMMITTED_ACTION_ATTR,
    POSE_DELTA_FAMILY_BOOTSTRAP_COUNTS_ATTR,
    POSE_DELTA_FAMILY_BOOTSTRAP_DATASETS_ATTR,
    POSE_DELTA_FAMILY_BOOTSTRAP_PAIR_ATTR,
    POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_DEBUG_ATTR,
    POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_FAMILY_IDS_ATTR,
    POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_ROW_IDS_ATTR,
    POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_STATS_ATTR,
    POSE_DELTA_FAMILY_BOOTSTRAP_ROW_INDICES_ATTR,
    POSE_DELTA_FAMILY_BOOTSTRAP_TYPES_ATTR,
    POSE_DELTA_FAMILY_LANE_CONTEXT_VALID_ATTR,
    POSE_DELTA_FAMILY_LANE_PERTURB_DEBUG_ATTR,
    POSE_DELTA_FAMILY_LANE_PERTURB_STATS_ATTR,
    POSE_DELTA_FAMILY_LANE_INVALID_MASK_ATTR,
    POSE_DELTA_FAMILY_LANE_ROW_IDS_ATTR,
    POSE_DELTA_FAMILY_LANE_SCRIPTED_MASK_ATTR,
    POSE_DELTA_FAMILY_LANE_STATES_ATTR,
    POSE_DELTA_FAMILY_LANE_VECTOR_VALID_ATTR,
    POSE_DELTA_FAMILY_LANE_VECTORS_ATTR,
    POSE_RESET_FAMILY_NAMES,
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


def _required_nested_keys(data: dict, path: tuple[str, ...]) -> object:
    cursor: object = data
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            joined = ".".join(path)
            raise ValueError(f"CAGE bootstrap reset dataset missing key '{joined}'.")
        cursor = cursor[key]
    return cursor


def _reset_dataset_row_count(dataset: dict) -> int:
    values = _required_nested_keys(dataset, ("initial_state", "articulation", "robot", "joint_position"))
    if not isinstance(values, list):
        raise ValueError("CAGE bootstrap reset dataset robot joint_position must be a list.")
    return len(values)


def _validate_reset_dataset_schema(dataset: dict, reset_type: str) -> int:
    row_count = _reset_dataset_row_count(dataset)
    if row_count <= 0:
        raise ValueError(f"CAGE bootstrap reset dataset '{reset_type}' has no rows.")

    required_paths = (
        ("initial_state", "articulation", "robot", "root_pose"),
        ("initial_state", "articulation", "robot", "root_velocity"),
        ("initial_state", "articulation", "robot", "joint_velocity"),
        ("initial_state", "rigid_object", "insertive_object", "root_pose"),
        ("initial_state", "rigid_object", "insertive_object", "root_velocity"),
        ("initial_state", "rigid_object", "receptive_object", "root_pose"),
        ("initial_state", "rigid_object", "receptive_object", "root_velocity"),
    )
    for path in required_paths:
        values = _required_nested_keys(dataset, path)
        if not isinstance(values, list):
            joined = ".".join(path)
            raise ValueError(f"CAGE bootstrap reset dataset '{reset_type}' key '{joined}' must be a list.")
        if len(values) != row_count:
            joined = ".".join(path)
            raise ValueError(
                f"CAGE bootstrap reset dataset '{reset_type}' key '{joined}' has {len(values)} rows, "
                f"expected {row_count}."
            )
    return row_count


def _stack_reset_dataset_list_rows(values: list, name: str) -> torch.Tensor:
    rows = [
        value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
        for value in values
    ]
    if len(rows) == 0:
        raise ValueError(f"CAGE bootstrap reset dataset key '{name}' has no rows.")
    return torch.stack(rows, dim=0).to(dtype=torch.float32)


def _filter_reset_dataset_rows_by_saved_velocity(
    dataset: dict,
    reset_type: str,
    max_robot_joint_velocity_sum: float | None,
    max_object_root_velocity_sum: float | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    row_count = _reset_dataset_row_count(dataset)
    keep = torch.ones(row_count, dtype=torch.bool)

    robot_joint_velocity = _stack_reset_dataset_list_rows(
        _required_nested_keys(dataset, ("initial_state", "articulation", "robot", "joint_velocity")),
        f"{reset_type}.initial_state.articulation.robot.joint_velocity",
    )
    insertive_root_velocity = _stack_reset_dataset_list_rows(
        _required_nested_keys(dataset, ("initial_state", "rigid_object", "insertive_object", "root_velocity")),
        f"{reset_type}.initial_state.rigid_object.insertive_object.root_velocity",
    )
    receptive_root_velocity = _stack_reset_dataset_list_rows(
        _required_nested_keys(dataset, ("initial_state", "rigid_object", "receptive_object", "root_velocity")),
        f"{reset_type}.initial_state.rigid_object.receptive_object.root_velocity",
    )

    robot_velocity_sum = robot_joint_velocity.abs().sum(dim=1)
    insertive_velocity_sum = insertive_root_velocity.abs().sum(dim=1)
    receptive_velocity_sum = receptive_root_velocity.abs().sum(dim=1)

    if max_robot_joint_velocity_sum is not None:
        keep &= robot_velocity_sum <= float(max_robot_joint_velocity_sum)
    if max_object_root_velocity_sum is not None:
        max_object_root_velocity_sum = float(max_object_root_velocity_sum)
        keep &= insertive_velocity_sum <= max_object_root_velocity_sum
        keep &= receptive_velocity_sum <= max_object_root_velocity_sum

    indices = torch.nonzero(keep, as_tuple=False).squeeze(-1).to(dtype=torch.long)
    if indices.numel() == 0:
        raise ValueError(
            f"CAGE bootstrap velocity filter removed all rows for '{reset_type}'. "
            f"robot_limit={max_robot_joint_velocity_sum}, object_limit={max_object_root_velocity_sum}"
        )

    diagnostics = {
        "robot_median": float(robot_velocity_sum.median().item()),
        "robot_max_kept": float(robot_velocity_sum[indices].max().item()),
        "insertive_median": float(insertive_velocity_sum.median().item()),
        "insertive_max_kept": float(insertive_velocity_sum[indices].max().item()),
        "receptive_median": float(receptive_velocity_sum.median().item()),
        "receptive_max_kept": float(receptive_velocity_sum[indices].max().item()),
    }
    return indices, diagnostics


def _normalize_env_ids(env: ManagerBasedEnv, env_ids) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _sample_reset_dataset_rows_to_device(dataset: dict, indices: torch.Tensor, device: torch.device) -> dict:
    result = {}
    cpu_indices = indices.detach().cpu().tolist()
    for key, value in dataset.items():
        if isinstance(value, dict):
            result[key] = _sample_reset_dataset_rows_to_device(value, indices, device)
        elif isinstance(value, list):
            rows = [
                row if isinstance(row, torch.Tensor) else torch.as_tensor(row, dtype=torch.float32)
                for row in (value[index] for index in cpu_indices)
            ]
            result[key] = torch.stack(rows, dim=0).to(device=device, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported CAGE bootstrap reset dataset value type: {type(value)}")
    return result


def _select_reset_state_rows(state: dict, indices: torch.Tensor, device: torch.device) -> dict:
    result = {}
    for key, value in state.items():
        if isinstance(value, dict):
            result[key] = _select_reset_state_rows(value, indices, device)
        elif isinstance(value, torch.Tensor):
            result[key] = value[indices.to(value.device)].to(device=device, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported CAGE reset lane state value type: {type(value)}")
    return result


def _write_relative_scene_state_to_sim(
    env: ManagerBasedEnv,
    state: dict,
    env_ids: torch.Tensor,
    zero_velocities: bool = False,
) -> None:
    env_ids = env_ids.to(env.device)
    env_origins = env.scene.env_origins[env_ids]

    for asset_name, articulation in env.scene._articulations.items():
        asset_states = state.get("articulation", {})
        if asset_name not in asset_states:
            continue
        asset_state = asset_states[asset_name]
        root_pose = asset_state["root_pose"].clone()
        root_pose[:, :3] += env_origins
        root_pose[:, 3:7] = _normalize_quat_tensor(root_pose[:, 3:7])
        root_velocity = asset_state["root_velocity"].clone()
        joint_position = asset_state["joint_position"].clone()
        joint_velocity = asset_state["joint_velocity"].clone()
        if zero_velocities:
            root_velocity.zero_()
            joint_velocity.zero_()
        articulation.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        articulation.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
        articulation.set_joint_position_target(joint_position, env_ids=env_ids)
        articulation.set_joint_velocity_target(joint_velocity, env_ids=env_ids)

    for asset_name, rigid_object in env.scene._rigid_objects.items():
        asset_states = state.get("rigid_object", {})
        if asset_name not in asset_states:
            continue
        asset_state = asset_states[asset_name]
        root_pose = asset_state["root_pose"].clone()
        root_pose[:, :3] += env_origins
        root_pose[:, 3:7] = _normalize_quat_tensor(root_pose[:, 3:7])
        root_velocity = asset_state["root_velocity"].clone()
        if zero_velocities:
            root_velocity.zero_()
        rigid_object.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)

    env.scene.write_data_to_sim()


def _normalize_quat_tensor(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp(min=1e-9)


def _euler_xyz_tensor(quat: torch.Tensor) -> torch.Tensor:
    return torch.stack(math_utils.euler_xyz_from_quat(_normalize_quat_tensor(quat)), dim=-1)


def _bottom_offset_tensor(asset: RigidObject | Articulation, env: ManagerBasedEnv) -> torch.Tensor:
    metadata = utils.read_metadata_from_usd_directory(asset.cfg.spawn.usd_path)
    bottom_offset = metadata.get("bottom_offset")
    return torch.tensor(bottom_offset.get("pos"), device=env.device, dtype=torch.float32).view(1, 3)


def _encode_receptive_pose_to_reset_vector(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    receptive_object: RigidObject,
    offset_asset: RigidObject | Articulation | None,
    bottom_offset: torch.Tensor | None,
) -> torch.Tensor:
    root_state = receptive_object.data.default_root_state[env_ids]
    pos_offset = receptive_object.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids] - root_state[:, 0:3]
    if offset_asset is not None:
        pos_offset -= offset_asset.data.default_root_state[env_ids, 0:3]
    if bottom_offset is not None:
        pos_offset += bottom_offset.expand(env_ids.numel(), -1)

    root_quat = _normalize_quat_tensor(root_state[:, 3:7])
    receptive_quat_w = _normalize_quat_tensor(receptive_object.data.root_quat_w[env_ids])
    rel_quat = math_utils.quat_mul(math_utils.quat_inv(root_quat), receptive_quat_w)
    yaw = _euler_xyz_tensor(rel_quat)[:, 2]
    return torch.stack((pos_offset[:, 0], pos_offset[:, 1], yaw), dim=-1)


def _decode_receptive_pose_from_reset_vector(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    reset_vector: torch.Tensor,
    receptive_object: RigidObject,
    offset_asset: RigidObject | Articulation | None,
    bottom_offset: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    root_state = receptive_object.data.default_root_state[env_ids]
    pos_offset = torch.zeros((env_ids.numel(), 3), device=env.device, dtype=torch.float32)
    pos_offset[:, 0:2] = reset_vector[:, 0:2]

    pos_w = root_state[:, 0:3] + env.scene.env_origins[env_ids] + pos_offset
    if offset_asset is not None:
        pos_w += offset_asset.data.default_root_state[env_ids, 0:3]
    if bottom_offset is not None:
        pos_w -= bottom_offset.expand(env_ids.numel(), -1)

    zeros = torch.zeros(env_ids.numel(), device=env.device, dtype=torch.float32)
    root_quat = _normalize_quat_tensor(root_state[:, 3:7])
    delta_quat = _normalize_quat_tensor(math_utils.quat_from_euler_xyz(zeros, zeros, reset_vector[:, 2]))
    quat_w = _normalize_quat_tensor(math_utils.quat_mul(root_quat, delta_quat))
    return pos_w, quat_w


def _encode_scene_to_cage_reset_vector(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    insertive_object: RigidObject,
    receptive_object: RigidObject,
    robot: Articulation,
    ee_body_idx: int,
    offset_asset: RigidObject | Articulation | None,
    bottom_offset: torch.Tensor | None,
) -> torch.Tensor:
    receptive_values = _encode_receptive_pose_to_reset_vector(
        env, env_ids, receptive_object, offset_asset, bottom_offset
    )

    receptive_quat_w = _normalize_quat_tensor(receptive_object.data.root_quat_w[env_ids])
    insertive_quat_w = _normalize_quat_tensor(insertive_object.data.root_quat_w[env_ids])
    ee_quat_w = _normalize_quat_tensor(robot.data.body_link_quat_w[env_ids, ee_body_idx])

    insertive_pos_rel, insertive_quat_rel = math_utils.subtract_frame_transforms(
        receptive_object.data.root_pos_w[env_ids],
        receptive_quat_w,
        insertive_object.data.root_pos_w[env_ids],
        insertive_quat_w,
    )
    insertive_euler_rel = _euler_xyz_tensor(insertive_quat_rel)

    ee_pos_rel, ee_quat_rel = math_utils.subtract_frame_transforms(
        insertive_object.data.root_pos_w[env_ids],
        insertive_quat_w,
        robot.data.body_link_pos_w[env_ids, ee_body_idx],
        ee_quat_w,
    )
    ee_euler_rel = _euler_xyz_tensor(ee_quat_rel)

    return torch.cat((receptive_values, insertive_pos_rel, insertive_euler_rel, ee_pos_rel, ee_euler_rel), dim=-1)


def _decode_cage_reset_vector_to_scene_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    reset_vector: torch.Tensor,
    insertive_object: RigidObject,
    receptive_object: RigidObject,
    offset_asset: RigidObject | Articulation | None,
    bottom_offset: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    receptive_pos_w, receptive_quat_w = _decode_receptive_pose_from_reset_vector(
        env, env_ids, reset_vector, receptive_object, offset_asset, bottom_offset
    )

    insertive_rel_quat = _normalize_quat_tensor(
        math_utils.quat_from_euler_xyz(reset_vector[:, 6], reset_vector[:, 7], reset_vector[:, 8])
    )
    insertive_pos_w, insertive_quat_w = math_utils.combine_frame_transforms(
        receptive_pos_w, receptive_quat_w, reset_vector[:, 3:6], insertive_rel_quat
    )
    insertive_quat_w = _normalize_quat_tensor(insertive_quat_w)

    ee_rel_quat = _normalize_quat_tensor(
        math_utils.quat_from_euler_xyz(reset_vector[:, 12], reset_vector[:, 13], reset_vector[:, 14])
    )
    ee_pos_w, ee_quat_w = math_utils.combine_frame_transforms(
        insertive_pos_w, insertive_quat_w, reset_vector[:, 9:12], ee_rel_quat
    )
    return (
        receptive_pos_w,
        receptive_quat_w,
        insertive_pos_w,
        insertive_quat_w,
        ee_pos_w,
        _normalize_quat_tensor(ee_quat_w),
    )


class cage_load_omnireset_family_bootstrap_states(ManagerTermBase):
    """Load OmniReset family reset datasets for CAGE bootstrap checks.

    This term stores CPU datasets and counts on the env, but does not write
    scene state or alter adversary actions.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        dataset_dir: str = cfg.params["dataset_dir"]
        reset_types: tuple[str, ...] = tuple(cfg.params.get("reset_types", POSE_RESET_FAMILY_NAMES))
        debug_print: bool = bool(cfg.params.get("debug_print", False))
        filter_saved_velocities: bool = bool(cfg.params.get("filter_saved_velocities", True))
        max_robot_joint_velocity_sum = cfg.params.get("max_robot_joint_velocity_sum", 0.2)
        max_object_root_velocity_sum = cfg.params.get("max_object_root_velocity_sum", 0.1)

        insertive_usd_path = env.scene["insertive_object"].cfg.spawn.usd_path
        receptive_usd_path = env.scene["receptive_object"].cfg.spawn.usd_path
        pair = utils.compute_pair_dir(insertive_usd_path, receptive_usd_path)

        datasets: dict[str, dict] = {}
        counts: dict[str, int] = {}
        row_indices: dict[str, torch.Tensor] = {}
        filter_diagnostics: dict[str, dict[str, float]] = {}
        for reset_type in reset_types:
            dataset_path = f"{dataset_dir}/Resets/{pair}/resets_{reset_type}.pt"
            local_path = utils.safe_retrieve_file_path(dataset_path)
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"CAGE bootstrap reset dataset not found: {dataset_path}")

            dataset = torch.load(local_path, map_location="cpu")
            row_count = _validate_reset_dataset_schema(dataset, reset_type)
            datasets[reset_type] = dataset
            counts[reset_type] = row_count
            if filter_saved_velocities:
                valid_indices, diagnostics = _filter_reset_dataset_rows_by_saved_velocity(
                    dataset,
                    reset_type,
                    max_robot_joint_velocity_sum=max_robot_joint_velocity_sum,
                    max_object_root_velocity_sum=max_object_root_velocity_sum,
                )
                row_indices[reset_type] = valid_indices
                filter_diagnostics[reset_type] = diagnostics
            else:
                row_indices[reset_type] = torch.arange(row_count, dtype=torch.long)

        setattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_DATASETS_ATTR, datasets)
        setattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_COUNTS_ATTR, counts)
        setattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_ROW_INDICES_ATTR, row_indices)
        setattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_PAIR_ATTR, pair)
        setattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_TYPES_ATTR, reset_types)

        if debug_print:
            count_text = ", ".join(
                f"{reset_type}={counts[reset_type]},kept={int(row_indices[reset_type].numel())}"
                for reset_type in reset_types
            )
            print(f"[CAGE family bootstrap] pair={pair} {count_text}", flush=True)
            if filter_saved_velocities:
                for reset_type in reset_types:
                    diagnostics = filter_diagnostics[reset_type]
                    print(
                        "[CAGE family bootstrap filter] "
                        f"{reset_type} robot_med={diagnostics['robot_median']:.5f} "
                        f"robot_max_kept={diagnostics['robot_max_kept']:.5f} "
                        f"insertive_med={diagnostics['insertive_median']:.5f} "
                        f"insertive_max_kept={diagnostics['insertive_max_kept']:.5f} "
                        f"receptive_med={diagnostics['receptive_median']:.5f} "
                        f"receptive_max_kept={diagnostics['receptive_max_kept']:.5f}",
                        flush=True,
                    )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        dataset_dir: str = "",
        reset_types: tuple[str, ...] = POSE_RESET_FAMILY_NAMES,
        debug_print: bool = False,
        filter_saved_velocities: bool = True,
        max_robot_joint_velocity_sum: float | None = 0.2,
        max_object_root_velocity_sum: float | None = 0.1,
    ) -> None:
        return None


class cage_replay_omnireset_family_bootstrap_states(ManagerTermBase):
    """Replay sampled OmniReset bootstrap states into sim for validation.

    This is intentionally zero-delta replay. It writes only dataset states and
    prepares the settling/grasp metadata consumed by the existing validator.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.reset_types: tuple[str, ...] = tuple(cfg.params.get("reset_types", POSE_RESET_FAMILY_NAMES))
        probabilities = cfg.params.get("probs", [1.0 / len(self.reset_types)] * len(self.reset_types))
        if len(probabilities) != len(self.reset_types):
            raise ValueError("CAGE bootstrap replay requires one probability per reset family.")
        self.probs = torch.as_tensor(probabilities, device=env.device, dtype=torch.float32)
        self.probs = self.probs / torch.clamp(self.probs.sum(), min=1e-9)
        self.debug_print: bool = bool(cfg.params.get("debug_print", False))
        self.debug_print_interval: int = max(1, int(cfg.params.get("debug_print_interval", 20)))
        self.gripper_closed_threshold: float = float(cfg.params.get("gripper_closed_threshold", 0.1))
        self.zero_velocities: bool = bool(cfg.params.get("zero_velocities", True))

        datasets = getattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_DATASETS_ATTR, None)
        counts = getattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_COUNTS_ATTR, None)
        row_indices = getattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_ROW_INDICES_ATTR, None)
        if not isinstance(datasets, dict) or not isinstance(counts, dict) or not isinstance(row_indices, dict):
            raise RuntimeError(
                "CAGE bootstrap replay expected family datasets from "
                "cage_load_omnireset_family_bootstrap_states."
            )
        missing = [reset_type for reset_type in self.reset_types if reset_type not in datasets]
        if missing:
            raise RuntimeError(f"CAGE bootstrap replay missing loaded reset families: {missing}")
        missing_indices = [reset_type for reset_type in self.reset_types if reset_type not in row_indices]
        if missing_indices:
            raise RuntimeError(f"CAGE bootstrap replay missing filtered row indices: {missing_indices}")

        self.datasets = datasets
        self.counts = counts
        self.row_indices = row_indices
        self.family_to_idx = {reset_type: idx for idx, reset_type in enumerate(self.reset_types)}
        self._reset_call_count = 0
        self._sample_counts = torch.zeros(len(self.reset_types), device=env.device, dtype=torch.long)
        self._family_ids = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self._row_ids = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        setattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_FAMILY_IDS_ATTR, self._family_ids)
        setattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_ROW_IDS_ATTR, self._row_ids)
        setattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_DEBUG_ATTR, self.debug_print)
        setattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_STATS_ATTR, {})

        robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        grasped_asset_cfg: SceneEntityCfg = cfg.params.get("grasped_asset_cfg", SceneEntityCfg("insertive_object"))
        robot: Articulation = env.scene[robot_cfg.name]
        self.robot = robot
        self.grasped_asset: RigidObject = env.scene[grasped_asset_cfg.name]
        self.robot_asset_name = robot_cfg.name
        self.grasped_asset_name = grasped_asset_cfg.name
        self.ee_body_name: str = cfg.params.get("ee_body_name", "robotiq_base_link")
        self.ee_body_idx = robot.data.body_names.index(self.ee_body_name)

        gripper_cfg: SceneEntityCfg = cfg.params.get(
            "gripper_cfg", SceneEntityCfg("robot", joint_names=["finger_joint", ".*right.*", ".*left.*"])
        )
        self.gripper_joint_ids: list[int] | slice = gripper_cfg.joint_ids
        if isinstance(self.gripper_joint_ids, slice):
            self.gripper_joint_list = list(range(robot.num_joints))[self.gripper_joint_ids]
        else:
            self.gripper_joint_list = list(self.gripper_joint_ids)

        setattr(env, "_multi_agent_runner_hooks", CageAdversaryResetRuntimeHooks(env, robot))
        setattr(env, "_cage_grasp_survival_robot_asset_name", self.robot_asset_name)
        setattr(env, "_cage_grasp_survival_object_asset_name", self.grasped_asset_name)
        setattr(env, "_cage_grasp_survival_ee_body_name", self.ee_body_name)

        self._settling_gripper_target = -torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
        self._settling_robot_joint_pos_target = robot.data.default_joint_pos.to(env.device, dtype=torch.float32).clone()
        self._settling_robot_joint_vel_target = torch.zeros(
            (env.num_envs, robot.num_joints), device=env.device, dtype=torch.float32
        )
        self._settling_robot_joint_target_valid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._grasp_survival_expected_obj_pos_ee = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
        self._grasp_survival_expected_obj_quat_ee = torch.zeros(env.num_envs, 4, device=env.device, dtype=torch.float32)
        self._grasp_survival_expected_obj_quat_ee[:, 0] = 1.0
        self._grasp_survival_reference_valid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._reset_grasp_branch = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._reset_branch_valid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        setattr(env, "_cage_adversary_gripper_action_target", self._settling_gripper_target)
        setattr(env, "_cage_adversary_settling_robot_joint_pos_target", self._settling_robot_joint_pos_target)
        setattr(env, "_cage_adversary_settling_robot_joint_vel_target", self._settling_robot_joint_vel_target)
        setattr(env, "_cage_adversary_settling_robot_joint_target_valid", self._settling_robot_joint_target_valid)
        setattr(env, "_cage_grasp_survival_expected_obj_pos_ee", self._grasp_survival_expected_obj_pos_ee)
        setattr(env, "_cage_grasp_survival_expected_obj_quat_ee", self._grasp_survival_expected_obj_quat_ee)
        setattr(env, "_cage_grasp_survival_reference_valid", self._grasp_survival_reference_valid)
        setattr(env, "_cage_reset_grasp_branch", self._reset_grasp_branch)
        setattr(env, "_cage_reset_branch_valid", self._reset_branch_valid)

    @staticmethod
    def _is_grasped_family(reset_type: str) -> bool:
        return reset_type.endswith("EEGrasped")

    def _update_replay_metadata(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        state: dict,
        family_ids: torch.Tensor,
    ) -> None:
        target_ids = env_ids.to(env.device)
        robot_state = state["articulation"][self.robot_asset_name]
        joint_position = robot_state["joint_position"].to(env.device, dtype=torch.float32)
        joint_velocity = robot_state["joint_velocity"].to(env.device, dtype=torch.float32)
        if self.zero_velocities:
            joint_velocity = torch.zeros_like(joint_velocity)

        self._settling_robot_joint_pos_target[target_ids] = joint_position
        self._settling_robot_joint_vel_target[target_ids] = joint_velocity
        self._settling_robot_joint_target_valid[target_ids] = True

        gripper_pos = joint_position[:, self.gripper_joint_list]
        closed_like = gripper_pos.abs().mean(dim=1) > self.gripper_closed_threshold
        self._settling_gripper_target[target_ids] = torch.where(
            closed_like,
            -torch.ones_like(self._settling_gripper_target[target_ids]),
            torch.ones_like(self._settling_gripper_target[target_ids]),
        )

        grasped_mask = torch.zeros(env_ids.numel(), device=env.device, dtype=torch.bool)
        for local_idx, reset_type in enumerate(self.reset_types):
            if self._is_grasped_family(reset_type):
                grasped_mask |= family_ids == local_idx
        self._reset_grasp_branch[target_ids] = grasped_mask
        self._reset_branch_valid[target_ids] = True
        self._grasp_survival_reference_valid[target_ids] = False

        if bool(grasped_mask.any().item()):
            grasp_ids = target_ids[grasped_mask]
            ee_pos_w = self.robot.data.body_link_pos_w[grasp_ids, self.ee_body_idx]
            ee_quat_w = self.robot.data.body_link_quat_w[grasp_ids, self.ee_body_idx]
            obj_pos_w = self.grasped_asset.data.root_pos_w[grasp_ids]
            obj_quat_w = self.grasped_asset.data.root_quat_w[grasp_ids]
            obj_pos_ee, obj_quat_ee = math_utils.subtract_frame_transforms(
                ee_pos_w, ee_quat_w, obj_pos_w, obj_quat_w
            )
            self._grasp_survival_expected_obj_pos_ee[grasp_ids] = obj_pos_ee
            self._grasp_survival_expected_obj_quat_ee[grasp_ids] = obj_quat_ee
            self._grasp_survival_reference_valid[grasp_ids] = True

    def _maybe_print_replay_summary(self) -> None:
        if not self.debug_print:
            return
        self._reset_call_count += 1
        if self._reset_call_count != 1 and self._reset_call_count % self.debug_print_interval != 0:
            return
        total = int(self._sample_counts.sum().item())
        if total <= 0:
            return
        parts = []
        for idx, reset_type in enumerate(self.reset_types):
            count = int(self._sample_counts[idx].item())
            pct = 100.0 * count / max(total, 1)
            parts.append(f"{reset_type}={count} ({pct:.1f}%)")
        print(f"[CAGE bootstrap replay] sampled total={total} " + ", ".join(parts), flush=True)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        reset_types: tuple[str, ...] = POSE_RESET_FAMILY_NAMES,
        probs: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        grasped_asset_cfg: SceneEntityCfg = SceneEntityCfg("insertive_object"),
        gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["finger_joint", ".*right.*", ".*left.*"]),
        ee_body_name: str = "robotiq_base_link",
        debug_print: bool = False,
        debug_print_interval: int = 20,
        gripper_closed_threshold: float = 0.1,
        zero_velocities: bool = True,
    ) -> None:
        env_ids = _normalize_env_ids(env, env_ids)
        if env_ids.numel() == 0:
            return

        family_ids = torch.multinomial(self.probs, env_ids.numel(), replacement=True)
        self._family_ids[env_ids] = family_ids.to(self._family_ids.device)

        for family_idx, reset_type in enumerate(self.reset_types):
            local_mask = family_ids == family_idx
            if not bool(local_mask.any().item()):
                continue
            current_env_ids = env_ids[local_mask]
            count = int(current_env_ids.numel())
            valid_state_indices = self.row_indices[reset_type].to(device=env.device, dtype=torch.long)
            sample_offsets = torch.randint(0, int(valid_state_indices.numel()), (count,), device=env.device)
            state_indices = valid_state_indices[sample_offsets]
            self._row_ids[current_env_ids] = state_indices.to(self._row_ids.device)
            sampled = _sample_reset_dataset_rows_to_device(
                self.datasets[reset_type]["initial_state"], state_indices, env.device
            )
            _write_relative_scene_state_to_sim(
                env,
                sampled,
                current_env_ids,
                zero_velocities=self.zero_velocities,
            )
            self._update_replay_metadata(env, current_env_ids, sampled, family_ids[local_mask])
            self._sample_counts[family_idx] += count

        self._maybe_print_replay_summary()


class cage_replay_family_lane_states(cage_replay_omnireset_family_bootstrap_states):
    """Replay one committed reset-state lane per family and env for Stage-5 checks."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        lane_states: list[dict] = []
        lane_row_ids = torch.full(
            (len(self.reset_types), env.num_envs), -1, device=env.device, dtype=torch.long
        )
        for family_idx, reset_type in enumerate(self.reset_types):
            valid_state_indices = self.row_indices[reset_type].to(device=env.device, dtype=torch.long)
            sample_offsets = torch.randint(0, int(valid_state_indices.numel()), (env.num_envs,), device=env.device)
            state_indices = valid_state_indices[sample_offsets]
            lane_row_ids[family_idx] = state_indices
            lane_states.append(
                _sample_reset_dataset_rows_to_device(
                    self.datasets[reset_type]["initial_state"], state_indices, env.device
                )
            )

        self.lane_states = lane_states
        self.lane_row_ids = lane_row_ids
        self._lane_cycle_cursor = 0
        setattr(env, POSE_DELTA_FAMILY_LANE_STATES_ATTR, self.lane_states)
        setattr(env, POSE_DELTA_FAMILY_LANE_ROW_IDS_ATTR, self.lane_row_ids)

        if self.debug_print:
            parts = [
                f"{reset_type}=env_lanes={env.num_envs}"
                for reset_type in self.reset_types
            ]
            print("[CAGE family lanes] initialized " + ", ".join(parts), flush=True)

    def _sample_family_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(env_ids.numel(), device=env_ids.device, dtype=torch.long)
        family_ids = (offsets + self._lane_cycle_cursor) % len(self.reset_types)
        self._lane_cycle_cursor = (self._lane_cycle_cursor + int(env_ids.numel())) % len(self.reset_types)
        return family_ids

    def _maybe_print_replay_summary(self) -> None:
        if not self.debug_print:
            return
        self._reset_call_count += 1
        if self._reset_call_count != 1 and self._reset_call_count % self.debug_print_interval != 0:
            return
        total = int(self._sample_counts.sum().item())
        if total <= 0:
            return
        parts = []
        for idx, reset_type in enumerate(self.reset_types):
            count = int(self._sample_counts[idx].item())
            pct = 100.0 * count / max(total, 1)
            parts.append(f"{reset_type}={count} ({pct:.1f}%)")
        print(f"[CAGE family lanes] replayed total={total} " + ", ".join(parts), flush=True)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        reset_types: tuple[str, ...] = POSE_RESET_FAMILY_NAMES,
        probs: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        grasped_asset_cfg: SceneEntityCfg = SceneEntityCfg("insertive_object"),
        gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["finger_joint", ".*right.*", ".*left.*"]),
        ee_body_name: str = "robotiq_base_link",
        debug_print: bool = False,
        debug_print_interval: int = 20,
        gripper_closed_threshold: float = 0.1,
        zero_velocities: bool = True,
    ) -> None:
        env_ids = _normalize_env_ids(env, env_ids)
        if env_ids.numel() == 0:
            return

        family_ids = self._sample_family_ids(env_ids)
        self._family_ids[env_ids] = family_ids.to(self._family_ids.device)

        for family_idx, reset_type in enumerate(self.reset_types):
            local_mask = family_ids == family_idx
            if not bool(local_mask.any().item()):
                continue
            current_env_ids = env_ids[local_mask]
            count = int(current_env_ids.numel())
            self._row_ids[current_env_ids] = self.lane_row_ids[
                family_idx, current_env_ids.to(self.lane_row_ids.device)
            ].to(self._row_ids.device)
            sampled = _select_reset_state_rows(self.lane_states[family_idx], current_env_ids, env.device)
            _write_relative_scene_state_to_sim(
                env,
                sampled,
                current_env_ids,
                zero_velocities=self.zero_velocities,
            )
            self._update_replay_metadata(env, current_env_ids, sampled, family_ids[local_mask])
            self._sample_counts[family_idx] += count

        self._maybe_print_replay_summary()


class cage_replay_family_lane_perturbation_states(cage_replay_family_lane_states):
    """Replay family lanes after applying scripted or learned 15D reset-vector perturbations."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        insertive_asset_cfg: SceneEntityCfg = cfg.params.get("insertive_asset_cfg", SceneEntityCfg("insertive_object"))
        receptive_asset_cfg: SceneEntityCfg = cfg.params.get("receptive_asset_cfg", SceneEntityCfg("receptive_object"))
        offset_asset_cfg: SceneEntityCfg | None = cfg.params.get("offset_asset_cfg")
        robot_ik_cfg: SceneEntityCfg = cfg.params.get(
            "robot_ik_cfg",
            SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"),
        )

        self.insertive_object: RigidObject = env.scene[insertive_asset_cfg.name]
        self.receptive_object: RigidObject = env.scene[receptive_asset_cfg.name]
        self.offset_asset: RigidObject | Articulation | None = (
            env.scene[offset_asset_cfg.name] if offset_asset_cfg is not None else None
        )
        self.bottom_offset = None
        if bool(cfg.params.get("use_bottom_offset", False)):
            self.bottom_offset = _bottom_offset_tensor(self.receptive_object, env)

        if robot_ik_cfg.body_names is None:
            raise ValueError("cage_replay_family_lane_perturbation_states requires robot_ik_cfg.body_names.")
        self.ik_joint_ids: list[int] | slice = robot_ik_cfg.joint_ids
        self.n_ik_joints: int = self.robot.num_joints if isinstance(self.ik_joint_ids, slice) else len(self.ik_joint_ids)
        self.ik_iterations: int = max(1, int(cfg.params.get("ik_iterations", 25)))
        self.ik_step_scale: float = float(cfg.params.get("ik_step_scale", 0.25))

        robot_ik_solver_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name=robot_ik_cfg.name,
            joint_names=robot_ik_cfg.joint_names,  # type: ignore
            body_name=robot_ik_cfg.body_names,  # type: ignore
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
        )
        self.solver: DifferentialInverseKinematicsAction = robot_ik_solver_cfg.class_type(robot_ik_solver_cfg, env)  # type: ignore

        self.action_low = _as_1d_action_tensor(cfg.params["action_low"], env.device, "action_low")
        self.action_high = _as_1d_action_tensor(cfg.params["action_high"], env.device, "action_high")
        self.scripted_delta = _as_1d_action_tensor(cfg.params["scripted_delta"], env.device, "scripted_delta")
        self.invalid_scripted_delta = _as_1d_action_tensor(
            cfg.params.get("invalid_scripted_delta", cfg.params["scripted_delta"]),
            env.device,
            "invalid_scripted_delta",
        )
        self.use_learned_delta: bool = bool(cfg.params.get("use_learned_delta", False))
        self.action_name: str = str(cfg.params.get("action_name", "adversaryaction"))
        self.delta_scale = _as_1d_action_tensor(
            cfg.params.get("delta_scale", cfg.params["scripted_delta"]),
            env.device,
            "delta_scale",
        )
        for name, tensor in (
            ("action_high", self.action_high),
            ("scripted_delta", self.scripted_delta),
            ("invalid_scripted_delta", self.invalid_scripted_delta),
            ("delta_scale", self.delta_scale),
        ):
            if tensor.shape != self.action_low.shape:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} must match action_low {tuple(self.action_low.shape)}."
                )
        if self.action_low.numel() != len(POSE_RESET_STATE_NAMES):
            raise ValueError(
                f"CAGE lane perturbation expected {len(POSE_RESET_STATE_NAMES)} reset dims, "
                f"got {self.action_low.numel()}."
            )
        if not bool((self.action_low <= self.action_high).all().item()):
            raise ValueError("action_low must be <= action_high for every CAGE reset-vector dimension.")

        self.invalid_every_n_resets: int = max(0, int(cfg.params.get("invalid_every_n_resets", 0)))
        self.invalid_count_per_family: int = max(0, int(cfg.params.get("invalid_count_per_family", 1)))

        self._committed_action = torch.zeros(
            (env.num_envs, self.action_low.numel()), device=env.device, dtype=torch.float32
        )
        self._candidate_action = self._committed_action.clone()
        self._candidate_valid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._invalid_scripted_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._context_valid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._scripted_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        self.lane_vectors = torch.zeros(
            (len(self.reset_types), env.num_envs, self.action_low.numel()), device=env.device, dtype=torch.float32
        )
        self.lane_vector_valid = torch.zeros(
            (len(self.reset_types), env.num_envs), device=env.device, dtype=torch.bool
        )
        setattr(env, POSE_DELTA_COMMITTED_ACTION_ATTR, self._committed_action)
        setattr(env, POSE_DELTA_CANDIDATE_ACTION_ATTR, self._candidate_action)
        setattr(env, POSE_DELTA_CANDIDATE_VALID_ATTR, self._candidate_valid)
        setattr(env, POSE_DELTA_FAMILY_LANE_INVALID_MASK_ATTR, self._invalid_scripted_mask)
        setattr(env, POSE_DELTA_FAMILY_LANE_CONTEXT_VALID_ATTR, self._context_valid)
        setattr(env, POSE_DELTA_FAMILY_LANE_SCRIPTED_MASK_ATTR, self._scripted_mask)
        setattr(env, POSE_DELTA_FAMILY_LANE_VECTORS_ATTR, self.lane_vectors)
        setattr(env, POSE_DELTA_FAMILY_LANE_VECTOR_VALID_ATTR, self.lane_vector_valid)
        setattr(env, POSE_DELTA_FAMILY_LANE_PERTURB_DEBUG_ATTR, self.debug_print)

        stats = {
            "prepared": torch.zeros(len(self.reset_types), device=env.device, dtype=torch.long),
            "invalid_scripted": torch.zeros(len(self.reset_types), device=env.device, dtype=torch.long),
            "invalid_accepted": torch.zeros(len(self.reset_types), device=env.device, dtype=torch.long),
            "invalid_rejected": torch.zeros(len(self.reset_types), device=env.device, dtype=torch.long),
            "accepted": torch.zeros(len(self.reset_types), device=env.device, dtype=torch.long),
            "rejected": torch.zeros(len(self.reset_types), device=env.device, dtype=torch.long),
            "max_trans_delta": torch.zeros(len(self.reset_types), device=env.device, dtype=torch.float32),
            "max_rot_delta": torch.zeros(len(self.reset_types), device=env.device, dtype=torch.float32),
            "last_prepared_print": 0,
            "last_resolved_print": 0,
            "print_interval": self.debug_print_interval,
            "reject_reasons": {},
        }
        setattr(env, POSE_DELTA_FAMILY_LANE_PERTURB_STATS_ATTR, stats)
        self._perturb_call_count = 0

        if self.debug_print:
            configured_delta = self.delta_scale if self.use_learned_delta else self.scripted_delta
            trans_delta = float(configured_delta[[0, 1, 3, 4, 5, 9, 10, 11]].abs().max().item()) * 1000.0
            rot_delta = float(configured_delta[[2, 6, 7, 8, 12, 13, 14]].abs().max().item()) * 180.0 / np.pi
            mode = "learned" if self.use_learned_delta else "scripted"
            print(
                f"[CAGE family perturb] enabled mode={mode} max_trans={trans_delta:.3f}mm "
                f"max_rot={rot_delta:.3f}deg invalid_every={self.invalid_every_n_resets}",
                flush=True,
            )

    def _apply_reset_vector_to_scene(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        reset_vector: torch.Tensor,
        base_state: dict,
    ) -> None:
        (
            receptive_pos_w,
            receptive_quat_w,
            insertive_pos_w,
            insertive_quat_w,
            ee_pos_w,
            ee_quat_w,
        ) = _decode_cage_reset_vector_to_scene_poses(
            env,
            env_ids,
            reset_vector,
            self.insertive_object,
            self.receptive_object,
            self.offset_asset,
            self.bottom_offset,
        )

        zeros_vel = torch.zeros((env_ids.numel(), 6), device=env.device, dtype=torch.float32)
        self.receptive_object.write_root_pose_to_sim(
            torch.cat((receptive_pos_w, receptive_quat_w), dim=-1),
            env_ids=env_ids,
        )
        self.receptive_object.write_root_velocity_to_sim(zeros_vel, env_ids=env_ids)
        self.insertive_object.write_root_pose_to_sim(
            torch.cat((insertive_pos_w, insertive_quat_w), dim=-1),
            env_ids=env_ids,
        )
        self.insertive_object.write_root_velocity_to_sim(zeros_vel, env_ids=env_ids)

        pos_b, quat_b = self.solver._compute_frame_pose()
        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            self.robot.data.root_link_pos_w[env_ids],
            self.robot.data.root_link_quat_w[env_ids],
            ee_pos_w,
            ee_quat_w,
        )
        pos_b[env_ids] = target_pos_b
        quat_b[env_ids] = target_quat_b
        self.solver.process_actions(torch.cat((pos_b, quat_b), dim=1))

        for _ in range(self.ik_iterations):
            self.solver.apply_actions()
            delta_joint_pos = self.ik_step_scale * (
                self.robot.data.joint_pos_target[env_ids] - self.robot.data.joint_pos[env_ids]
            )
            self.robot.write_joint_state_to_sim(
                position=(delta_joint_pos + self.robot.data.joint_pos[env_ids])[:, self.ik_joint_ids],
                velocity=torch.zeros((env_ids.numel(), self.n_ik_joints), device=env.device),
                joint_ids=self.ik_joint_ids,
                env_ids=env_ids,
            )

        base_joint_pos = base_state["articulation"][self.robot_asset_name]["joint_position"].to(
            env.device, dtype=torch.float32
        )
        gripper_positions = base_joint_pos[:, self.gripper_joint_list]
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
        env.scene.write_data_to_sim()

    def _maybe_print_perturb_prepared_summary(self, env: ManagerBasedEnv) -> None:
        if not self.debug_print:
            return
        stats = getattr(env, POSE_DELTA_FAMILY_LANE_PERTURB_STATS_ATTR, None)
        if not isinstance(stats, dict):
            return
        prepared = stats.get("prepared")
        if not isinstance(prepared, torch.Tensor):
            return
        total = int(prepared.sum().item())
        last = int(stats.get("last_prepared_print", 0))
        interval = max(1, int(stats.get("print_interval", self.debug_print_interval)))
        if total <= 0 or total - last < interval:
            return
        stats["last_prepared_print"] = total
        invalid = stats.get("invalid_scripted")
        max_trans = stats.get("max_trans_delta")
        max_rot = stats.get("max_rot_delta")
        parts = []
        for family_idx, reset_type in enumerate(self.reset_types):
            count = int(prepared[family_idx].item())
            invalid_count = int(invalid[family_idx].item()) if isinstance(invalid, torch.Tensor) else 0
            trans_mm = float(max_trans[family_idx].item()) * 1000.0 if isinstance(max_trans, torch.Tensor) else 0.0
            rot_deg = float(max_rot[family_idx].item()) * 180.0 / np.pi if isinstance(max_rot, torch.Tensor) else 0.0
            parts.append(f"{reset_type}=n{count},invalid{invalid_count},max={trans_mm:.2f}mm/{rot_deg:.2f}deg")
        print(f"[CAGE family perturb] prepared total={total} " + "; ".join(parts), flush=True)

    def _stage_family_lane_context(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        """Write lane bases into sim so the adversary observes the state it will perturb."""
        env_ids = _normalize_env_ids(env, env_ids)
        if env_ids.numel() == 0:
            return

        target_candidate_ids = env_ids.to(self._candidate_valid.device)
        self._candidate_valid[target_candidate_ids] = False
        self._invalid_scripted_mask[env_ids.to(self._invalid_scripted_mask.device)] = False
        self._context_valid[env_ids.to(self._context_valid.device)] = False

        family_ids = self._sample_family_ids(env_ids)
        self._family_ids[env_ids] = family_ids.to(self._family_ids.device)

        for family_idx, reset_type in enumerate(self.reset_types):
            local_mask = family_ids == family_idx
            if not bool(local_mask.any().item()):
                continue

            current_env_ids = env_ids[local_mask]
            count = int(current_env_ids.numel())
            self._row_ids[current_env_ids] = self.lane_row_ids[
                family_idx, current_env_ids.to(self.lane_row_ids.device)
            ].to(self._row_ids.device)

            base_state = _select_reset_state_rows(self.lane_states[family_idx], current_env_ids, env.device)
            _write_relative_scene_state_to_sim(env, base_state, current_env_ids, zero_velocities=self.zero_velocities)
            base_vector = _encode_scene_to_cage_reset_vector(
                env,
                current_env_ids,
                self.insertive_object,
                self.receptive_object,
                self.robot,
                self.ee_body_idx,
                self.offset_asset,
                self.bottom_offset,
            )

            target_ids = current_env_ids.to(self.lane_vectors.device)
            vector_valid = self.lane_vector_valid[family_idx, target_ids].to(env.device)
            stored_vector = self.lane_vectors[family_idx, target_ids].to(env.device)
            committed_vector = torch.where(vector_valid.unsqueeze(-1), stored_vector, base_vector)
            if not bool(vector_valid.all().item()):
                self.lane_vectors[family_idx, target_ids] = committed_vector.to(self.lane_vectors.device)
                self.lane_vector_valid[family_idx, target_ids] = True

            self._committed_action[current_env_ids.to(self._committed_action.device)] = committed_vector.to(
                self._committed_action.device
            )
            self._context_valid[current_env_ids.to(self._context_valid.device)] = True
            self._scripted_mask[current_env_ids.to(self._scripted_mask.device)] = not self.use_learned_delta
            self._sample_counts[family_idx] += count

        self._maybe_print_replay_summary()

    def prepare_adversary_proposal_context(self, env: ManagerBasedEnv, env_ids: torch.Tensor | None = None) -> bool:
        """Runner hook: stage lane bases before adversary action sampling."""
        if not self.use_learned_delta:
            return False
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        self._stage_family_lane_context(env, env_ids)
        return True

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        reset_types: tuple[str, ...] = POSE_RESET_FAMILY_NAMES,
        probs: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        grasped_asset_cfg: SceneEntityCfg = SceneEntityCfg("insertive_object"),
        gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["finger_joint", ".*right.*", ".*left.*"]),
        ee_body_name: str = "robotiq_base_link",
        insertive_asset_cfg: SceneEntityCfg = SceneEntityCfg("insertive_object"),
        receptive_asset_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
        robot_ik_cfg: SceneEntityCfg = SceneEntityCfg(
            "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
        ),
        offset_asset_cfg: SceneEntityCfg | None = None,
        use_bottom_offset: bool = False,
        action_low: tuple[float, ...] = (),
        action_high: tuple[float, ...] = (),
        scripted_delta: tuple[float, ...] = (),
        invalid_scripted_delta: tuple[float, ...] = (),
        use_learned_delta: bool = False,
        action_name: str = "adversaryaction",
        delta_scale: tuple[float, ...] = (),
        invalid_every_n_resets: int = 0,
        invalid_count_per_family: int = 1,
        ik_iterations: int = 25,
        ik_step_scale: float = 0.25,
        debug_print: bool = False,
        debug_print_interval: int = 20,
        gripper_closed_threshold: float = 0.1,
        zero_velocities: bool = True,
    ) -> None:
        env_ids = _normalize_env_ids(env, env_ids)
        if env_ids.numel() == 0:
            return

        self._perturb_call_count += 1
        if self.use_learned_delta:
            context_valid = self._context_valid[env_ids.to(self._context_valid.device)]
            if not bool(context_valid.all().item()):
                self._stage_family_lane_context(env, env_ids)
        else:
            self._stage_family_lane_context(env, env_ids)

        family_ids = self._family_ids[env_ids.to(self._family_ids.device)].to(env.device, dtype=torch.long)
        stats = getattr(env, POSE_DELTA_FAMILY_LANE_PERTURB_STATS_ATTR, None)
        raw_actions = _get_action_term_raw_actions(env, self.action_name) if self.use_learned_delta else None

        for family_idx, reset_type in enumerate(self.reset_types):
            local_mask = family_ids == family_idx
            if not bool(local_mask.any().item()):
                continue
            current_env_ids = env_ids[local_mask]
            count = int(current_env_ids.numel())

            base_state = _select_reset_state_rows(self.lane_states[family_idx], current_env_ids, env.device)
            _write_relative_scene_state_to_sim(env, base_state, current_env_ids, zero_velocities=self.zero_velocities)
            invalid_mask = torch.zeros(count, device=env.device, dtype=torch.bool)
            target_ids = current_env_ids.to(self._committed_action.device)
            committed_vector = self._committed_action[target_ids].to(env.device)
            if self.use_learned_delta:
                if raw_actions is None:
                    raise RuntimeError("CAGE learned family lane perturbation missing adversary raw actions.")
                raw_delta = raw_actions[current_env_ids.to(raw_actions.device), : self.action_low.numel()].to(
                    env.device, dtype=torch.float32
                )
                delta = torch.tanh(raw_delta) * self.delta_scale.unsqueeze(0)
            else:
                delta = self.scripted_delta.unsqueeze(0).expand(count, -1).clone()
            if (
                not self.use_learned_delta
                and self.invalid_every_n_resets > 0
                and self._perturb_call_count % self.invalid_every_n_resets == 0
            ):
                invalid_count = min(count, self.invalid_count_per_family)
                if invalid_count > 0:
                    invalid_mask[:invalid_count] = True
                    delta[invalid_mask] = self.invalid_scripted_delta

            effective_low = torch.minimum(self.action_low.unsqueeze(0), committed_vector)
            effective_high = torch.maximum(self.action_high.unsqueeze(0), committed_vector)
            candidate = torch.clamp(committed_vector + delta, effective_low, effective_high)
            self._committed_action[target_ids] = committed_vector.to(self._committed_action.device)
            self._candidate_action[target_ids] = candidate.to(self._candidate_action.device)
            self._candidate_valid[target_ids] = True
            self._invalid_scripted_mask[target_ids] = invalid_mask.to(self._invalid_scripted_mask.device)
            self._scripted_mask[target_ids] = not self.use_learned_delta

            if isinstance(stats, dict):
                prepared = stats.get("prepared")
                invalid = stats.get("invalid_scripted")
                max_trans = stats.get("max_trans_delta")
                max_rot = stats.get("max_rot_delta")
                if isinstance(prepared, torch.Tensor):
                    prepared[family_idx] += count
                if isinstance(invalid, torch.Tensor):
                    invalid[family_idx] += invalid_mask.sum().to(invalid.dtype)
                if isinstance(max_trans, torch.Tensor):
                    trans_mag = torch.linalg.norm(delta[:, [0, 1, 3, 4, 5, 9, 10, 11]], dim=-1).max()
                    max_trans[family_idx] = torch.maximum(max_trans[family_idx], trans_mag.to(max_trans.device))
                if isinstance(max_rot, torch.Tensor):
                    rot_mag = delta[:, [2, 6, 7, 8, 12, 13, 14]].abs().amax(dim=1).max()
                    max_rot[family_idx] = torch.maximum(max_rot[family_idx], rot_mag.to(max_rot.device))

            self._apply_reset_vector_to_scene(env, current_env_ids, candidate, base_state)
            perturbed_state = _select_reset_state_rows(env.scene.get_state(is_relative=True), current_env_ids, env.device)
            self._update_replay_metadata(env, current_env_ids, perturbed_state, family_ids[local_mask])

        self._context_valid[env_ids.to(self._context_valid.device)] = False
        self._maybe_print_perturb_prepared_summary(env)


class cage_debug_reset_vector_roundtrip(ManagerTermBase):
    """Console-only check for scene <-> 15D reset-vector round trips."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        insertive_asset_cfg: SceneEntityCfg = cfg.params.get("insertive_asset_cfg", SceneEntityCfg("insertive_object"))
        receptive_asset_cfg: SceneEntityCfg = cfg.params.get("receptive_asset_cfg", SceneEntityCfg("receptive_object"))
        robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        gripper_cfg: SceneEntityCfg = cfg.params.get(
            "gripper_cfg", SceneEntityCfg("robot", joint_names=["finger_joint", ".*right.*", ".*left.*"])
        )
        offset_asset_cfg: SceneEntityCfg | None = cfg.params.get("offset_asset_cfg")

        self.insertive_object: RigidObject = env.scene[insertive_asset_cfg.name]
        self.receptive_object: RigidObject = env.scene[receptive_asset_cfg.name]
        self.robot: Articulation = env.scene[robot_cfg.name]
        self.offset_asset: RigidObject | Articulation | None = (
            env.scene[offset_asset_cfg.name] if offset_asset_cfg is not None else None
        )
        self.ee_body_name: str = cfg.params.get("ee_body_name", "robotiq_base_link")
        self.ee_body_idx = self.robot.data.body_names.index(self.ee_body_name)
        self.debug_print: bool = bool(cfg.params.get("debug_print", False))
        self.debug_print_interval: int = max(1, int(cfg.params.get("debug_print_interval", 20)))
        self.pos_tolerance: float = float(cfg.params.get("pos_tolerance", 1.0e-4))
        self.rot_tolerance: float = float(cfg.params.get("rot_tolerance", 1.0e-3))
        self.gripper_tolerance: float = float(cfg.params.get("gripper_tolerance", 1.0e-5))

        self.gripper_joint_ids: list[int] | slice = gripper_cfg.joint_ids
        if isinstance(self.gripper_joint_ids, slice):
            self.gripper_joint_list = list(range(self.robot.num_joints))[self.gripper_joint_ids]
        else:
            self.gripper_joint_list = list(self.gripper_joint_ids)

        self.bottom_offset = None
        if bool(cfg.params.get("use_bottom_offset", False)):
            self.bottom_offset = _bottom_offset_tensor(self.receptive_object, env)

        self._stats = {
            "count": torch.zeros(len(POSE_RESET_FAMILY_NAMES), device=env.device, dtype=torch.long),
            "bad": torch.zeros(len(POSE_RESET_FAMILY_NAMES), device=env.device, dtype=torch.long),
            "invalid_skipped": torch.zeros(len(POSE_RESET_FAMILY_NAMES), device=env.device, dtype=torch.long),
            "last_print_total": 0,
            "max_receptive_pos": torch.zeros(len(POSE_RESET_FAMILY_NAMES), device=env.device),
            "max_receptive_rot": torch.zeros(len(POSE_RESET_FAMILY_NAMES), device=env.device),
            "max_insertive_pos": torch.zeros(len(POSE_RESET_FAMILY_NAMES), device=env.device),
            "max_insertive_rot": torch.zeros(len(POSE_RESET_FAMILY_NAMES), device=env.device),
            "max_ee_pos": torch.zeros(len(POSE_RESET_FAMILY_NAMES), device=env.device),
            "max_ee_rot": torch.zeros(len(POSE_RESET_FAMILY_NAMES), device=env.device),
            "max_gripper": torch.zeros(len(POSE_RESET_FAMILY_NAMES), device=env.device),
        }

    def _update_stat_max(self, name: str, family_ids: torch.Tensor, values: torch.Tensor) -> None:
        for family_idx in range(len(POSE_RESET_FAMILY_NAMES)):
            mask = family_ids == family_idx
            if bool(mask.any().item()):
                self._stats[name][family_idx] = torch.maximum(self._stats[name][family_idx], values[mask].max())

    def _maybe_print_summary(self) -> None:
        if not self.debug_print:
            return
        total = int(self._stats["count"].sum().item())
        print_every = max(1, self.debug_print_interval)
        if total - int(self._stats["last_print_total"]) < print_every:
            return
        self._stats["last_print_total"] = total

        family_parts = []
        for family_idx, family_name in enumerate(POSE_RESET_FAMILY_NAMES):
            count = int(self._stats["count"][family_idx].item())
            bad = int(self._stats["bad"][family_idx].item())
            invalid_skipped = int(self._stats["invalid_skipped"][family_idx].item())
            if count == 0:
                continue
            family_parts.append(
                f"{family_name}=n{count},bad{bad},skip_invalid{invalid_skipped},"
                f"rec={float(self._stats['max_receptive_pos'][family_idx].item()) * 1000.0:.3f}mm/"
                f"{float(self._stats['max_receptive_rot'][family_idx].item()):.4f}rad,"
                f"ins={float(self._stats['max_insertive_pos'][family_idx].item()) * 1000.0:.3f}mm/"
                f"{float(self._stats['max_insertive_rot'][family_idx].item()):.4f}rad,"
                f"ee={float(self._stats['max_ee_pos'][family_idx].item()) * 1000.0:.3f}mm/"
                f"{float(self._stats['max_ee_rot'][family_idx].item()):.4f}rad,"
                f"grip={float(self._stats['max_gripper'][family_idx].item()):.6f}"
            )
        bad_total = int(self._stats["bad"].sum().item())
        print(f"[CAGE reset-vector roundtrip] total={total} bad={bad_total} " + "; ".join(family_parts), flush=True)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        insertive_asset_cfg: SceneEntityCfg = SceneEntityCfg("insertive_object"),
        receptive_asset_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["finger_joint", ".*right.*", ".*left.*"]),
        offset_asset_cfg: SceneEntityCfg | None = None,
        ee_body_name: str = "robotiq_base_link",
        use_bottom_offset: bool = False,
        debug_print: bool = False,
        debug_print_interval: int = 20,
        pos_tolerance: float = 1.0e-4,
        rot_tolerance: float = 1.0e-3,
        gripper_tolerance: float = 1.0e-5,
    ) -> None:
        env_ids = _normalize_env_ids(env, env_ids)
        if env_ids.numel() == 0:
            return

        family_ids = getattr(env, POSE_DELTA_FAMILY_BOOTSTRAP_REPLAY_FAMILY_IDS_ATTR, None)
        if not isinstance(family_ids, torch.Tensor) or family_ids.shape != (env.num_envs,):
            return
        local_family_ids = family_ids[env_ids.to(family_ids.device)].to(env.device, dtype=torch.long)
        valid_family = (local_family_ids >= 0) & (local_family_ids < len(POSE_RESET_FAMILY_NAMES))
        if not bool(valid_family.any().item()):
            return

        env_ids = env_ids[valid_family]
        local_family_ids = local_family_ids[valid_family]
        invalid_scripted = getattr(env, POSE_DELTA_FAMILY_LANE_INVALID_MASK_ATTR, None)
        if isinstance(invalid_scripted, torch.Tensor) and invalid_scripted.shape == (env.num_envs,):
            invalid_local = invalid_scripted[env_ids.to(invalid_scripted.device)].to(env.device, dtype=torch.bool)
            for family_idx in range(len(POSE_RESET_FAMILY_NAMES)):
                mask = local_family_ids == family_idx
                if bool(mask.any().item()):
                    self._stats["invalid_skipped"][family_idx] += (invalid_local & mask).sum().to(torch.long)
            keep_mask = ~invalid_local
            if not bool(keep_mask.any().item()):
                self._maybe_print_summary()
                return
            env_ids = env_ids[keep_mask]
            local_family_ids = local_family_ids[keep_mask]

        reset_vector = _encode_scene_to_cage_reset_vector(
            env,
            env_ids,
            self.insertive_object,
            self.receptive_object,
            self.robot,
            self.ee_body_idx,
            self.offset_asset,
            self.bottom_offset,
        )
        candidate = getattr(env, POSE_DELTA_CANDIDATE_ACTION_ATTR, None)
        candidate_valid = getattr(env, POSE_DELTA_CANDIDATE_VALID_ATTR, None)
        if (
            isinstance(candidate, torch.Tensor)
            and candidate.ndim == 2
            and candidate.shape[0] == env.num_envs
            and candidate.shape[1] == len(POSE_RESET_STATE_NAMES)
            and isinstance(candidate_valid, torch.Tensor)
            and candidate_valid.shape == (env.num_envs,)
        ):
            candidate_valid_local = candidate_valid[env_ids.to(candidate_valid.device)].to(env.device, dtype=torch.bool)
            if bool(candidate_valid_local.any().item()):
                reset_vector[candidate_valid_local] = candidate[
                    env_ids[candidate_valid_local].to(candidate.device)
                ].to(device=env.device, dtype=reset_vector.dtype)
        (
            receptive_pos_rt,
            receptive_quat_rt,
            insertive_pos_rt,
            insertive_quat_rt,
            ee_pos_rt,
            ee_quat_rt,
        ) = _decode_cage_reset_vector_to_scene_poses(
            env,
            env_ids,
            reset_vector,
            self.insertive_object,
            self.receptive_object,
            self.offset_asset,
            self.bottom_offset,
        )

        receptive_pos_err = torch.linalg.norm(receptive_pos_rt - self.receptive_object.data.root_pos_w[env_ids], dim=-1)
        receptive_rot_err = math_utils.quat_error_magnitude(
            receptive_quat_rt, _normalize_quat_tensor(self.receptive_object.data.root_quat_w[env_ids])
        )
        insertive_pos_err = torch.linalg.norm(insertive_pos_rt - self.insertive_object.data.root_pos_w[env_ids], dim=-1)
        insertive_rot_err = math_utils.quat_error_magnitude(
            insertive_quat_rt, _normalize_quat_tensor(self.insertive_object.data.root_quat_w[env_ids])
        )
        ee_pos_err = torch.linalg.norm(ee_pos_rt - self.robot.data.body_link_pos_w[env_ids, self.ee_body_idx], dim=-1)
        ee_rot_err = math_utils.quat_error_magnitude(
            ee_quat_rt, _normalize_quat_tensor(self.robot.data.body_link_quat_w[env_ids, self.ee_body_idx])
        )

        target_joint_pos = getattr(env, "_cage_adversary_settling_robot_joint_pos_target", None)
        if isinstance(target_joint_pos, torch.Tensor) and target_joint_pos.shape == (env.num_envs, self.robot.num_joints):
            gripper_err = (
                self.robot.data.joint_pos[env_ids][:, self.gripper_joint_list]
                - target_joint_pos[env_ids.to(target_joint_pos.device)][:, self.gripper_joint_list].to(env.device)
            ).abs().amax(dim=1)
        else:
            gripper_err = torch.full((env_ids.numel(),), float("inf"), device=env.device)

        bad = (
            (receptive_pos_err > self.pos_tolerance)
            | (receptive_rot_err > self.rot_tolerance)
            | (insertive_pos_err > self.pos_tolerance)
            | (insertive_rot_err > self.rot_tolerance)
            | (ee_pos_err > self.pos_tolerance)
            | (ee_rot_err > self.rot_tolerance)
            | (gripper_err > self.gripper_tolerance)
        )

        for family_idx in range(len(POSE_RESET_FAMILY_NAMES)):
            mask = local_family_ids == family_idx
            if bool(mask.any().item()):
                self._stats["count"][family_idx] += mask.sum().to(torch.long)
                self._stats["bad"][family_idx] += bad[mask].sum().to(torch.long)
        self._update_stat_max("max_receptive_pos", local_family_ids, receptive_pos_err)
        self._update_stat_max("max_receptive_rot", local_family_ids, receptive_rot_err)
        self._update_stat_max("max_insertive_pos", local_family_ids, insertive_pos_err)
        self._update_stat_max("max_insertive_rot", local_family_ids, insertive_rot_err)
        self._update_stat_max("max_ee_pos", local_family_ids, ee_pos_err)
        self._update_stat_max("max_ee_rot", local_family_ids, ee_rot_err)
        self._update_stat_max("max_gripper", local_family_ids, gripper_err)
        self._maybe_print_summary()


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
        self._candidate_action = self._committed_action.clone()
        self._candidate_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        setattr(env, POSE_DELTA_COMMITTED_ACTION_ATTR, self._committed_action)
        setattr(env, POSE_DELTA_CANDIDATE_ACTION_ATTR, self._candidate_action)
        setattr(env, POSE_DELTA_CANDIDATE_VALID_ATTR, self._candidate_valid)

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
        committed = self._committed_action[target_ids]
        candidate = compute_pose_delta_candidate(
            raw_delta,
            committed,
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
        self._settling_control_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        setattr(env, "_cage_settling_control_mask", self._settling_control_mask)
        self._live_anchor_grasp_branch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._live_anchor_branch_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # Rows: non-grasp branch, grasp branch. Columns: done count, task-success count.
        self._branch_episode_stats = torch.zeros((2, 2), dtype=torch.float32, device=env.device)
        self._patch_gripper_action_term()

    def _tensor_attr(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        value = getattr(self.env, name, None)
        if not isinstance(value, torch.Tensor) or value.shape != shape:
            raise RuntimeError(f"CAGE runtime hook expected env.{name} with shape {shape}.")
        return value

    def _patch_gripper_action_term(self) -> None:
        action_manager = getattr(self.env, "action_manager", None)
        terms = getattr(action_manager, "_terms", None)
        if not isinstance(terms, dict):
            return
        gripper_term = terms.get("gripper")
        if gripper_term is None or bool(getattr(gripper_term, "_cage_settling_target_patch", False)):
            return
        original_process_actions = gripper_term.process_actions

        def _process_actions_with_settling_targets(actions):
            original_process_actions(actions)
            mask = getattr(self.env, "_cage_settling_control_mask", None)
            target_joint_pos = getattr(self.env, "_cage_adversary_settling_robot_joint_pos_target", None)
            target_joint_valid = getattr(self.env, "_cage_adversary_settling_robot_joint_target_valid", None)
            joint_ids = getattr(gripper_term, "_joint_ids", None)
            processed_actions = getattr(gripper_term, "_processed_actions", None)
            if not (
                isinstance(mask, torch.Tensor)
                and isinstance(target_joint_pos, torch.Tensor)
                and isinstance(target_joint_valid, torch.Tensor)
                and isinstance(processed_actions, torch.Tensor)
                and mask.shape == (self.env.num_envs,)
                and target_joint_pos.shape == (self.env.num_envs, self.robot.num_joints)
                and target_joint_valid.shape == (self.env.num_envs,)
                and joint_ids is not None
            ):
                return
            active = mask.to(device=processed_actions.device, dtype=torch.bool) & target_joint_valid.to(
                device=processed_actions.device, dtype=torch.bool
            )
            if not bool(active.any().item()):
                return
            active_ids = active.nonzero(as_tuple=False).squeeze(-1)
            processed_actions[active_ids] = target_joint_pos[
                active_ids.to(target_joint_pos.device)
            ][:, joint_ids].to(device=processed_actions.device, dtype=processed_actions.dtype)

        gripper_term.process_actions = _process_actions_with_settling_targets
        setattr(gripper_term, "_cage_settling_target_patch", True)

    def set_settling_control_mask(self, mask: torch.Tensor) -> None:
        if mask.numel() != self.env.num_envs:
            raise RuntimeError(
                f"CAGE settling control mask has {mask.numel()} entries, expected {self.env.num_envs}."
            )
        self._settling_control_mask.copy_(mask.to(device=self.env.device, dtype=torch.bool).view(-1))

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

    def prepare_adversary_proposal_context(self, env_ids: torch.Tensor | None = None) -> bool:
        """Stage a CAGE family lane before the runner samples the adversary action."""
        event_manager = getattr(self.env, "event_manager", None)
        mode_names = getattr(event_manager, "_mode_term_names", {})
        mode_cfgs = getattr(event_manager, "_mode_term_cfgs", {})
        reset_names = mode_names.get("reset", []) if isinstance(mode_names, dict) else []
        reset_cfgs = mode_cfgs.get("reset", []) if isinstance(mode_cfgs, dict) else []
        for name, term_cfg in zip(reset_names, reset_cfgs):
            if name != "replay_family_lane_states":
                continue
            term = getattr(term_cfg, "func", None)
            hook = getattr(term, "prepare_adversary_proposal_context", None)
            if callable(hook):
                return bool(hook(self.env, env_ids))
        return False

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
        record_keys = (
            "accepted_reset_env_ids",
            "accepted_reset_family_ids",
            "accepted_reset_row_ids",
            "accepted_reset_states",
            "accepted_reset_previous_lane_states",
            "accepted_reset_candidate_states",
            "accepted_reset_deltas",
            "accepted_reset_true_lane_deltas",
            "accepted_reset_is_scripted",
        )
        for key in record_keys:
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
                "family_names": POSE_RESET_FAMILY_NAMES,
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
        a = _get_pose_delta_candidate_actions(env, env_ids)

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
            # Read prepared continuous reset actions for the selected envs.
            a = _get_pose_delta_candidate_actions(env, env_ids)
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
    """Reset end-effector pose and gripper state from a 50/50 grasp/action mix.

    Half of the reset envs use a saved OmniReset grasp for the insertive object;
    the other half place the EE at an adversary-chosen offset *relative to the
    insertive object* (body frame), so the adversary directly controls approach
    distance and angle rather than absolute workspace coordinates.
    Downstream physics settling validates both branches.
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
        setattr(env, "_cage_adversary_gripper_action_target", self._settling_gripper_target)
        setattr(env, "_cage_adversary_settling_robot_joint_pos_target", self._settling_robot_joint_pos_target)
        setattr(env, "_cage_adversary_settling_robot_joint_vel_target", self._settling_robot_joint_vel_target)
        setattr(env, "_cage_adversary_settling_robot_joint_target_valid", self._settling_robot_joint_target_valid)
        setattr(env, "_cage_grasp_survival_expected_obj_pos_ee", self._grasp_survival_expected_obj_pos_ee)
        setattr(env, "_cage_grasp_survival_expected_obj_quat_ee", self._grasp_survival_expected_obj_quat_ee)
        setattr(env, "_cage_grasp_survival_reference_valid", self._grasp_survival_reference_valid)
        setattr(env, "_cage_reset_grasp_branch", self._reset_grasp_branch)
        setattr(env, "_cage_reset_branch_valid", self._reset_branch_valid)

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
        num_envs = len(env_ids)

        grasped_mask = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        grasped_mask[torch.randperm(num_envs, device=env.device)[: num_envs // 2]] = True
        adv_local = (~grasped_mask).nonzero(as_tuple=False).squeeze(-1)
        grasp_local = grasped_mask.nonzero(as_tuple=False).squeeze(-1)
        adv_ids = env_ids[adv_local]
        grasp_ids = env_ids[grasp_local]
        self._store_reset_branch_labels(env_ids, grasped_mask)
        self._clear_grasp_survival_reference(env_ids)

        pos_b, quat_b = self.solver._compute_frame_pose()

        if adv_ids.numel() > 0:
            si = self.action_start_index
            adv_clamped = torch.clamp(a[adv_local, si : si + 6], self.ranges[:, 0], self.ranges[:, 1])
            adv_rel_quat = math_utils.quat_from_euler_xyz(adv_clamped[:, 3], adv_clamped[:, 4], adv_clamped[:, 5])
            adv_pos_w, adv_quat_w = math_utils.combine_frame_transforms(
                self.grasped_asset.data.root_pos_w[adv_ids],
                self.grasped_asset.data.root_quat_w[adv_ids],
                adv_clamped[:, 0:3],
                adv_rel_quat,
            )
            pos_b[adv_ids], quat_b[adv_ids] = math_utils.subtract_frame_transforms(
                self.robot.data.root_link_pos_w[adv_ids],
                self.robot.data.root_link_quat_w[adv_ids],
                adv_pos_w,
                adv_quat_w,
            )

        grasp_indices = None
        if grasp_ids.numel() > 0:
            grasp_indices = torch.randint(0, len(self.rel_positions), (grasp_ids.numel(),), device=env.device)
            grasp_pos_w, grasp_quat_w = math_utils.combine_frame_transforms(
                self.grasped_asset.data.root_pos_w[grasp_ids],
                self.grasped_asset.data.root_quat_w[grasp_ids],
                self.rel_positions[grasp_indices],
                self.rel_quaternions[grasp_indices],
            )
            expected_obj_pos_ee, expected_obj_quat_ee = math_utils.subtract_frame_transforms(
                grasp_pos_w,
                grasp_quat_w,
                self.grasped_asset.data.root_pos_w[grasp_ids],
                self.grasped_asset.data.root_quat_w[grasp_ids],
            )
            pos_b[grasp_ids], quat_b[grasp_ids] = math_utils.subtract_frame_transforms(
                self.robot.data.root_link_pos_w[grasp_ids],
                self.robot.data.root_link_quat_w[grasp_ids],
                grasp_pos_w,
                grasp_quat_w,
            )

        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))

        # Match the grasp-dataset reset's convergence target.
        for _ in range(25):
            self.solver.apply_actions()
            delta_joint_pos = 0.25 * (self.robot.data.joint_pos_target[env_ids] - self.robot.data.joint_pos[env_ids])
            self.robot.write_joint_state_to_sim(
                position=(delta_joint_pos + self.robot.data.joint_pos[env_ids])[:, self.joint_ids],
                velocity=torch.zeros((num_envs, self.n_joints), device=env.device),
                joint_ids=self.joint_ids,
                env_ids=env_ids,
            )

        num_gripper_joints = self.gripper_joint_positions.shape[1]
        gripper_positions = torch.zeros((num_envs, num_gripper_joints), device=env.device)
        gripper_action_targets = -torch.ones(num_envs, device=env.device, dtype=torch.float32)

        if adv_local.numel() > 0:
            adv_gripper_targets = torch.where(
                torch.rand(adv_local.numel(), device=env.device) < 0.5,
                torch.ones(adv_local.numel(), device=env.device, dtype=torch.float32),
                -torch.ones(adv_local.numel(), device=env.device, dtype=torch.float32),
            )
            gripper_action_targets[adv_local] = adv_gripper_targets
            open_positions = self.gripper_open_positions.unsqueeze(0).expand(adv_local.numel(), -1)
            closed_positions = self.gripper_closed_positions.unsqueeze(0).expand(adv_local.numel(), -1)
            gripper_positions[adv_local] = torch.where(
                (adv_gripper_targets > 0.0).unsqueeze(-1),
                open_positions,
                closed_positions,
            )

        if grasp_local.numel() > 0:
            gripper_positions[grasp_local] = self.gripper_joint_positions[grasp_indices]

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
        env.scene.write_data_to_sim()

        self._store_settling_gripper_targets(env_ids, gripper_action_targets)
        self._store_settling_robot_joint_targets(env_ids, full_joint_pos, full_joint_vel)
        if grasp_ids.numel() > 0:
            self._store_grasp_survival_reference(grasp_ids, expected_obj_pos_ee, expected_obj_quat_ee)
