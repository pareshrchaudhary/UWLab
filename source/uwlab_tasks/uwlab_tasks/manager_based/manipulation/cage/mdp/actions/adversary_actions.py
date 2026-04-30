# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm


class AdversaryAction(ActionTerm):
    """Storage-only action term for adversary outputs."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.cfg.action_dim, device=self.device)
        self._previous_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def previous_actions(self) -> torch.Tensor:
        return self._previous_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.to(self.device)

    def set_previous_actions(self, actions: torch.Tensor, env_ids: torch.Tensor | None = None):
        actions = actions.to(self.device)
        if env_ids is None:
            self._previous_actions[:] = actions
        else:
            env_ids = env_ids.to(self.device)
            self._previous_actions[env_ids] = actions[env_ids]

    def apply_actions(self):
        pass
