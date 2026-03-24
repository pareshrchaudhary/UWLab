# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment config for BC-PPO online fine-tuning with RGB observations.

Inherits the full RGB scene (cameras, curtains, visual randomizations) from the
data-collection config and adds:
  - history_length=4 on the policy group (matching BC n_obs_steps)
  - a critic group with privileged state
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp as task_mdp
from .data_collection_rgb_cfg import (
    Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg,
    RGBObservationsCfg,
)


@configclass
class BCPPOObservationsCfg:
    """Observation groups for BC-PPO: policy (RGB), critic (privileged state)."""

    @configclass
    class BCPPOPolicyCfg(RGBObservationsCfg.RGBPolicyCfg):
        """Same as RGBPolicyCfg but with history_length for frame stacking."""

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False
            self.history_length = 4
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged state observations for the critic (flat vector)."""

        prev_actions = ObsTerm(func=task_mdp.last_action)

        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_offset_metadata_key": "gripper_offset",
                "root_asset_offset_metadata_key": "offset",
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_offset_metadata_key": "gripper_offset",
                "rotation_repr": "axis_angle",
            },
        )

        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_in_receptive_asset_frame = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )

        time_left = ObsTerm(func=task_mdp.time_left)
        joint_vel = ObsTerm(func=task_mdp.joint_vel)

        end_effector_vel_lin_ang_b = ObsTerm(
            func=task_mdp.asset_link_velocity_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        robot_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        insertive_object_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("insertive_object")}
        )
        receptive_object_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("receptive_object")}
        )
        table_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("table")}
        )

        robot_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("robot")})
        insertive_object_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("insertive_object")})
        receptive_object_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("receptive_object")})
        table_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("table")})

        robot_joint_friction = ObsTerm(func=task_mdp.get_joint_friction, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_joint_armature = ObsTerm(func=task_mdp.get_joint_armature, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_joint_stiffness = ObsTerm(func=task_mdp.get_joint_stiffness, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_joint_damping = ObsTerm(func=task_mdp.get_joint_damping, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: BCPPOPolicyCfg = BCPPOPolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Ur5eRobotiq2f85BCPPORGBCfg(Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg):
    """BC-PPO RGB config: RGB scene + asymmetric obs (policy=images, critic=privileged state)."""

    observations: BCPPOObservationsCfg = BCPPOObservationsCfg()
