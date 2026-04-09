# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from uwlab_assets.robots.ur5e_robotiq_gripper.actions import ROBOTIQ_GRIPPER_BINARY_ACTIONS

from ...mdp.actions.actions_cfg import RelCartesianOSCActionCfg
from ...mdp.actions.adversary_actions_cfg import ADVERSARY_ADVANCED_ACTION_DIM, ADVERSARY_POSE_ACTION_DIM, AdversaryActionCfg

# Pre-train gains (soft initial Kp; curriculum ramps to stiff terminal)
UR5E_ROBOTIQ_2F85_RELATIVE_OSC = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    motion_stiffness=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    motion_damping_ratio=(3.0, 3.0, 3.0, 1.0, 1.0, 1.0),
    torque_limit=(150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
)

# Eval / sim2real gains (high Kp matched to sysid friction, end-of-curriculum values)
UR5E_ROBOTIQ_2F85_RELATIVE_OSC_EVAL = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=(150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
)

# Unscaled (for sysid scripts)
UR5E_ROBOTIQ_2F85_RELATIVE_OSC_UNSCALED = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=(150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
)


@configclass
class Ur5eRobotiq2f85RelativeOSCAction:
    """Action config using the analytical OSC + binary gripper."""

    arm = UR5E_ROBOTIQ_2F85_RELATIVE_OSC
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS


@configclass
class Ur5eRobotiq2f85RelativeOSCEvalAction:
    """Action config with high Kp gains (end-of-curriculum values) for eval / data-collection."""

    arm = UR5E_ROBOTIQ_2F85_RELATIVE_OSC_EVAL
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS


@configclass
class Ur5eRobotiq2f85SysidOSCAction:
    """Unscaled arm action (Cartesian delta) + binary gripper. For Sysid env / scripts."""

    arm = UR5E_ROBOTIQ_2F85_RELATIVE_OSC_UNSCALED
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS


# Adversary action config (must match AdversaryBaseEventCfg index map in rl_state_cfg)
UR5E_ROBOTIQ_2F85_ADVERSARY_ACTION = AdversaryActionCfg(action_dim=ADVERSARY_POSE_ACTION_DIM)


@configclass
class Ur5eRobotiq2f85AdversaryOSCAction:
    """Action config for CAGE adversarial training: arm OSC + binary gripper + pose adversary."""

    arm = UR5E_ROBOTIQ_2F85_RELATIVE_OSC
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    adversaryaction = UR5E_ROBOTIQ_2F85_ADVERSARY_ACTION


# Advanced adversary action config (AdversaryAdvancedEventCfg: friction, mass, actuator gains, OSC gains)
UR5E_ROBOTIQ_2F85_ADVERSARY_ADVANCED_ACTION = AdversaryActionCfg(action_dim=ADVERSARY_ADVANCED_ACTION_DIM)


@configclass
class Ur5eRobotiq2f85AdversaryAdvancedOSCAction:
    """Action config for CAGE advanced-adversary training: arm OSC + binary gripper + parameter adversary."""

    arm = UR5E_ROBOTIQ_2F85_RELATIVE_OSC
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    adversaryaction = UR5E_ROBOTIQ_2F85_ADVERSARY_ADVANCED_ACTION
