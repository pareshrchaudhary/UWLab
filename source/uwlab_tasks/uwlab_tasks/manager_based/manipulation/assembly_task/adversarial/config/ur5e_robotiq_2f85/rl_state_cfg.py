# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR  # type: ignore
from uwlab_assets.robots.ur5e_robotiq_gripper import (
    EXPLICIT_UR5E_ROBOTIQ_2F85,
    IMPLICIT_UR5E_ROBOTIQ_2F85,
    Ur5eRobotiq2f85RelativeJointPositionAction,
)

from .actions import (
    UR5E_ROBOTIQ_2F85_ADVERSARY_ACTION,
    Ur5eRobotiq2f85RelativeOSCAction,
)

from ... import mdp as task_mdp
from ....common.config.ur5e_robotiq_2f85.rl_state_base_cfg import (
    CommandsCfg,
    RewardsCfg,
    RlStateSceneCfg,
    TerminationsCfg,
    Ur5eRobotiq2f85RlStateCfg as BaseRlStateCfg,
    variants,
)

#########################################################
# Base Event Configuration
#########################################################
@configclass
class BaseEventCfg:
    """Configuration for events."""

    #########################################################
    # Robot (Adversary-controlled parameters)
    #########################################################
    # Adversary action indices: [0] static_friction, [1] dynamic_friction
    robot_material = EventTerm(
        func=task_mdp.adversary_robot_material_from_action,  # type: ignore
        mode="reset",
        params={
            "action_name": "adversaryaction",
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.2, 1.0),
            "num_buckets": 256,
            "make_consistent": True,
        },
    )

    # Adversary action index: [2] mass_scale
    randomize_robot_mass = EventTerm(
        func=task_mdp.adversary_robot_mass_from_action,  # type: ignore
        mode="reset",
        params={
            "action_name": "adversaryaction",
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_scale_range": (0.7, 1.3),
            "recompute_inertia": True,
        },
    )

    # Adversary action indices: [3] friction_scale, [4] armature_scale
    randomize_robot_joint_parameters = EventTerm(
        func=task_mdp.adversary_robot_joint_parameters_from_action,  # type: ignore
        mode="reset",
        params={
            "action_name": "adversaryaction",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*", "finger_joint"]),
            "friction_scale_range": (0.25, 4.0),
            "armature_scale_range": (0.25, 4.0),
        },
    )

    # Adversary action indices: [5] stiffness_scale, [6] damping_scale
    randomize_gripper_actuator_parameters = EventTerm(
        func=task_mdp.adversary_gripper_actuator_gains_from_action,  # type: ignore
        mode="reset",
        params={
            "action_name": "adversaryaction",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "stiffness_scale_range": (0.5, 2.0),
            "damping_scale_range": (0.5, 2.0),
        },
    )

    #########################################################
    # Insertive Object
    #########################################################
    insertive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (1.0, 2.0),
            "dynamic_friction_range": (0.9, 1.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "make_consistent": True,
        },
    )

    randomize_insertive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "mass_distribution_params": (0.02, 0.2),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    #########################################################
    # Receptive Object
    #########################################################
    receptive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (1.0, 2.0),
            "dynamic_friction_range": (0.9, 1.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "make_consistent": True,
        },
    )
    
    randomize_receptive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    #########################################################
    # Table
    #########################################################
    randomize_table_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    table_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.6),
            "dynamic_friction_range": (0.2, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("table"),
            "make_consistent": True,
        },
    )

    # mode: reset
    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={}) # type: ignore

    # Training Events
    # Adversary prob logits are the last 4 dims of the adversary action.
    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "base_paths": [
                f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectAnywhereEEAnywhere",
                f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectRestingEEGrasped",
                f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectAnywhereEEGrasped",
                f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectPartiallyAssembledEEGrasped",
            ],
            "action_name": "adversaryaction",
            "action_prob_start_idx": 9,
            "probs": [0.10, 0.20, 0.30, 0.40],
            "min_prob": 0.05,  # Minimum probability cap for each reset state
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )



#########################################################
# Observations 
#########################################################
@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

        prev_actions = ObsTerm(func=task_mdp.policy_last_action, params={"adversary_action_dim": UR5E_ROBOTIQ_2F85_ADVERSARY_ACTION.action_dim})

        joint_pos = ObsTerm(func=task_mdp.joint_pos) # type: ignore

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

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class CriticCfg(ObsGroup):

        prev_actions = ObsTerm(func=task_mdp.policy_last_action, params={"adversary_action_dim": UR5E_ROBOTIQ_2F85_ADVERSARY_ACTION.action_dim})

        joint_pos = ObsTerm(func=task_mdp.joint_pos) # type: ignore

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

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )

        # privileged observations
        time_left = ObsTerm(func=task_mdp.time_left) # type: ignore

        joint_vel = ObsTerm(func=task_mdp.joint_vel) # type: ignore 

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

        insertive_object_mass = ObsTerm(
            func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("insertive_object")}
        )

        receptive_object_mass = ObsTerm(
            func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("receptive_object")}
        )

        table_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("table")})

        robot_joint_friction = ObsTerm(func=task_mdp.get_joint_friction, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_joint_armature = ObsTerm(func=task_mdp.get_joint_armature, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_joint_stiffness = ObsTerm(
            func=task_mdp.get_joint_stiffness, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        robot_joint_damping = ObsTerm(func=task_mdp.get_joint_damping, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

# Adversary Observations
@configclass
class AdversaryObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        noise = ObsTerm(func=task_mdp.adversary_noise, params={"dim": UR5E_ROBOTIQ_2F85_ADVERSARY_ACTION.action_dim})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()

# CAGE Observations
@configclass
class MARLObservationsCfg:
    """Combined observation groups for training on shared environment.
    
    ObservationManager requires ObsGroup instances as direct attributes.
    Keys in obs_buf:
        - "policy": policy actor observations
        - "critic": policy critic observations
        - "adversary_policy": adversary actor observations
    """

    # Policy observations
    policy: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()
    critic: ObservationsCfg.CriticCfg = ObservationsCfg.CriticCfg()

    # Adversary observations
    adversary_policy: AdversaryObservationsCfg.PolicyCfg = AdversaryObservationsCfg.PolicyCfg()



#########################################################
# RL State Configuration
#########################################################
@configclass
class Ur5eRobotiq2f85RlStateCfg(BaseRlStateCfg):
    observations: MARLObservationsCfg = MARLObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    events: BaseEventCfg = MISSING # type: ignore



#########################################################
# Training Configuration 
#########################################################
@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainCfg(Ur5eRobotiq2f85RlStateCfg):
    """Training configuration for Relative Cartesian OSC action space."""

    events: BaseEventCfg = BaseEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Adversary action indices: [7] osc_stiffness_scale, [8] osc_damping_scale
        self.events.randomize_robot_actuator_parameters = EventTerm(  # type: ignore
            func=task_mdp.adversary_operational_space_controller_gains_from_action,
            mode="reset",
            params={
                "adversary_action_name": "adversaryaction",
                "osc_action_name": "arm",
                "stiffness_scale_range": (0.7, 1.3),
                "damping_scale_range": (0.9, 1.1),
                "action_stiffness_index": 7,
                "action_damping_index": 8,
            },
        )

@configclass
class Ur5eRobotiq2f85RelJointPosTrainCfg(Ur5eRobotiq2f85RlStateCfg):
    """Training configuration for Relative Joint Position action space."""

    events: BaseEventCfg = BaseEventCfg()
    actions: Ur5eRobotiq2f85RelativeJointPositionAction = Ur5eRobotiq2f85RelativeJointPositionAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.events.randomize_robot_actuator_parameters = EventTerm( # type: ignore
            func=task_mdp.randomize_actuator_gains, # type: ignore
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*", "finger_joint"]),
                "stiffness_distribution_params": (0.5, 2.0),
                "damping_distribution_params": (0.5, 2.0),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )

