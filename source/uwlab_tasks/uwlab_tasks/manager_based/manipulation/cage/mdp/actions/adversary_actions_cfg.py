# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import configclass

from .adversary_actions import AdversaryAction


# Receptive root: x, y, yaw (0-2). Insertive offset: 6-D body frame (3-8).
# EE pose relative to insertive object: x, y, z, roll, pitch, yaw (9-14).
ADVERSARY_POSE_ACTION_DIM = 15

# Parameter adversary (AdversaryAdvancedEventCfg): friction, mass, joint/actuator/OSC gains (18 total).
# Friction: robot(0-1), insertive(2-3), receptive(4-5), table(6-7) — static, dynamic each.
# Mass scale: robot(8), insertive(9), receptive(10), table(11).
# Joint dynamics: friction_scale(12), armature_scale(13).
# Gripper actuator: stiffness_scale(14), damping_scale(15).
# OSC gains: stiffness_scale(16), damping_scale(17).
ADVERSARY_ADVANCED_ACTION_DIM = 18

@configclass
class AdversaryActionCfg(ActionTermCfg):
    """Storage-only action term for adversary outputs."""

    class_type: type = AdversaryAction

    asset_name: str = "robot"

    action_dim: int = ADVERSARY_POSE_ACTION_DIM
