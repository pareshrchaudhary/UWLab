"""Tensor helpers for the CAGE continuous pose-delta adversary."""

from __future__ import annotations

import torch


POSE_DELTA_COMMITTED_ACTION_ATTR = "_cage_adversary_pose_committed_action"
POSE_DELTA_CANDIDATE_ACTION_ATTR = "_cage_adversary_pose_candidate_action"
POSE_DELTA_CANDIDATE_VALID_ATTR = "_cage_adversary_pose_candidate_valid"
POSE_DELTA_ACCEPTED_RECORDS_ATTR = "_cage_adversary_pose_accepted_records"

POSE_RESET_STATE_NAMES = (
    "receptive_object_x",
    "receptive_object_y",
    "receptive_object_yaw",
    "insertive_object_rel_x",
    "insertive_object_rel_y",
    "insertive_object_rel_z",
    "insertive_object_rel_roll",
    "insertive_object_rel_pitch",
    "insertive_object_rel_yaw",
    "end_effector_rel_x",
    "end_effector_rel_y",
    "end_effector_rel_z",
    "end_effector_rel_roll",
    "end_effector_rel_pitch",
    "end_effector_rel_yaw",
)

POSE_RESET_DELTA_NAMES = tuple(f"delta_{name}" for name in POSE_RESET_STATE_NAMES)


def compute_pose_delta_candidate(
    raw_actions: torch.Tensor,
    committed_actions: torch.Tensor,
    delta_scale: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    """Convert raw adversary delta commands into bounded physical reset actions."""

    candidate = committed_actions + torch.tanh(raw_actions) * delta_scale
    return torch.clamp(candidate, action_low, action_high)


def commit_pose_delta_candidate_tensors(
    committed: torch.Tensor,
    candidate: torch.Tensor,
    candidate_valid: torch.Tensor,
    commit_mask: torch.Tensor,
) -> torch.Tensor:
    """Commit prepared pose adversary candidates for successful settling envs."""

    commit_ids = (commit_mask & candidate_valid.to(device=commit_mask.device, dtype=torch.bool)).nonzero(
        as_tuple=False
    ).squeeze(-1)
    if commit_ids.numel() > 0:
        target_ids = commit_ids.to(committed.device)
        source_ids = commit_ids.to(candidate.device)
        committed[target_ids] = candidate[source_ids].to(device=committed.device, dtype=committed.dtype)
    return commit_ids
