# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm

from . import actions_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class AdversaryAction(ActionTerm):
    """Storage-only action term for adversary outputs."""

    cfg: actions_cfg.AdversaryActionCfg

    def __init__(self, cfg: actions_cfg.AdversaryActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.cfg.action_dim, device=self.device)

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.to(self.device)

    def apply_actions(self):
        pass
