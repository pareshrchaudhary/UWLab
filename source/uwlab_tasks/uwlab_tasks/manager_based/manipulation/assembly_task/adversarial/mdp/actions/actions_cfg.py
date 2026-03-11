# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import task_space_actions


@configclass
class AdversaryActionCfg(ActionTermCfg):
    """Storage-only action term for adversary outputs."""

    class_type: type[ActionTerm] = task_space_actions.AdversaryAction

    asset_name: str = "robot"

    action_dim: int = 13
