# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def policy_last_action(env: ManagerBasedRLEnv, adversary_action_dim: int = 19) -> torch.Tensor:
    """For MARL environments where actions are [policy_actions | adversary_actions],
    returns only the policy portion."""
    full_actions = env.action_manager.action
    policy_action_dim = full_actions.shape[-1] - adversary_action_dim
    return full_actions[:, :policy_action_dim]


def adversary_noise(env: ManagerBasedEnv, dim: int = 8) -> torch.Tensor:
    """Standard normal noise for adversary policy input."""
    return torch.randn((env.num_envs, dim), device=env.device, dtype=torch.float)
