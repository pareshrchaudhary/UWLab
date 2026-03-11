# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Vision-related termination terms for assembly tasks."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


def corrupted_camera_detected(
    env: ManagerBasedRLEnv,
    camera_names: list[str],
    std_threshold: float = 10.0
) -> torch.Tensor:
    """Detect corrupted camera images by checking if standard deviation is below threshold.

    Corrupted cameras typically show uniform gray/black images with very low variance.
    Returns True for environments where any camera shows corruption (std < threshold).

    Args:
        env: The environment instance.
        camera_names: List of camera sensor names to check.
        std_threshold: Std below which an image is considered corrupted. Default 10.0.

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    num_envs = env.num_envs
    device = env.device

    is_corrupted = torch.zeros(num_envs, dtype=torch.bool, device=device)

    for camera_name in camera_names:
        camera = env.scene[camera_name]
        rgb_data = camera.data.output["rgb"]
        rgb_flat = rgb_data.reshape(num_envs, -1).float()
        std_per_env = torch.std(rgb_flat, dim=1)
        is_corrupted |= (std_per_env < std_threshold)

    return is_corrupted


def consecutive_success_state(env: ManagerBasedRLEnv, num_consecutive_successes: int = 10) -> torch.Tensor:
    """Done when progress_context's continuous_success_counter >= num_consecutive_successes."""
    context_term = env.reward_manager.get_term_cfg("progress_context").func  # type: ignore
    continuous_success_counter = getattr(context_term, "continuous_success_counter")
    return continuous_success_counter >= num_consecutive_successes
