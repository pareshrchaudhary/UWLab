# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import configclass

from .adversary_actions import AdversaryAction


# Receptive root: x, y, yaw (0–2). Insertive offset: 6-D body frame (3–8).
# EE world-frame pose: x, y, z, roll, pitch, yaw (9–14). Gripper finger_joint (15).
ADVERSARY_POSE_ACTION_DIM = 16

@configclass
class AdversaryActionCfg(ActionTermCfg):
    """Storage-only action term for adversary outputs."""

    class_type: type = AdversaryAction

    asset_name: str = "robot"

    action_dim: int = ADVERSARY_POSE_ACTION_DIM
