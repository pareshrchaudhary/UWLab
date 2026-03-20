# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from ... import mdp as task_mdp
from .data_collection_rgb_cfg import Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg, Ur5eRobotiq2f85EvalRGBRelCartesianOSCOODCfg


@configclass
class RGBOODOSCCfg(Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg):
    """OOD OSC controller gains (stiffness/damping)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_robot_actuator_parameters = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_operational_space_controller_gains,
            mode="reset",
            params={
                "action_name": "arm",
                "stiffness_distribution_params": (0.5, 2.4),
                "damping_distribution_params": (0.5, 1.6),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.reset_from_reset_states = EventTerm(
            func=task_mdp.MultiResetManager,
            mode="reset",
            params={
                "base_paths": [f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectAnywhereEEAnywhere"],
                "probs": [1.0],
                "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            },
        )


@configclass
class RGBOODRobotCfg(Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg):
    """OOD robot dynamics (mass, material, joints, gripper gains)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.robot_material = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (0.2, 1.8),
                "dynamic_friction_range": (0.2, 1.8),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 256,
                "asset_cfg": SceneEntityCfg("robot"),
                "make_consistent": True,
            },
        )
        self.events.randomize_robot_mass = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "mass_distribution_params": (0.2, 1.8),
                "operation": "scale",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )
        self.events.randomize_robot_joint_parameters = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_joint_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*", "finger_joint"]),
                "friction_distribution_params": (0.1, 5.0),
                "armature_distribution_params": (0.1, 5.0),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
        self.events.randomize_gripper_actuator_parameters = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
                "stiffness_distribution_params": (0.2, 2.8),
                "damping_distribution_params": (0.2, 2.8),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
        self.events.randomize_robot_actuator_parameters = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_operational_space_controller_gains,
            mode="reset",
            params={
                "action_name": "arm",
                "stiffness_distribution_params": (0.7, 1.3),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.reset_from_reset_states = EventTerm(
            func=task_mdp.MultiResetManager,
            mode="reset",
            params={
                "base_paths": [f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectAnywhereEEAnywhere"],
                "probs": [1.0],
                "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            },
        )


@configclass
class RGBOODObjectCfg(Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg):
    """OOD object properties (insertive/receptive material and mass)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.insertive_object_material = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (0.6, 1.4),
                "dynamic_friction_range": (0.6, 1.4),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 256,
                "asset_cfg": SceneEntityCfg("insertive_object"),
                "make_consistent": True,
            },
        )
        self.events.randomize_insertive_object_mass = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("insertive_object"),
                "mass_distribution_params": (0.02, 0.4),
                "operation": "abs",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )
        self.events.receptive_object_material = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (0.6, 1.2),
                "dynamic_friction_range": (0.6, 1.2),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 256,
                "asset_cfg": SceneEntityCfg("receptive_object"),
                "make_consistent": True,
            },
        )
        self.events.randomize_robot_actuator_parameters = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_operational_space_controller_gains,
            mode="reset",
            params={
                "action_name": "arm",
                "stiffness_distribution_params": (0.7, 1.3),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.reset_from_reset_states = EventTerm(
            func=task_mdp.MultiResetManager,
            mode="reset",
            params={
                "base_paths": [f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectAnywhereEEAnywhere"],
                "probs": [1.0],
                "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            },
        )


@configclass
class RGBOODAllCfg(Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg):
    """OOD everything: OSC gains, robot dynamics, and object properties."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_robot_actuator_parameters = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_operational_space_controller_gains,
            mode="reset",
            params={
                "action_name": "arm",
                "stiffness_distribution_params": (0.5, 2.4),
                "damping_distribution_params": (0.5, 1.6),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.robot_material = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (0.2, 1.8),
                "dynamic_friction_range": (0.2, 1.8),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 256,
                "asset_cfg": SceneEntityCfg("robot"),
                "make_consistent": True,
            },
        )
        self.events.randomize_robot_mass = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "mass_distribution_params": (0.2, 1.8),
                "operation": "scale",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )
        self.events.randomize_robot_joint_parameters = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_joint_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*", "finger_joint"]),
                "friction_distribution_params": (0.1, 5.0),
                "armature_distribution_params": (0.1, 5.0),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
        self.events.randomize_gripper_actuator_parameters = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
                "stiffness_distribution_params": (0.2, 2.8),
                "damping_distribution_params": (0.2, 2.8),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
        self.events.insertive_object_material = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (0.6, 1.4),
                "dynamic_friction_range": (0.6, 1.4),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 256,
                "asset_cfg": SceneEntityCfg("insertive_object"),
                "make_consistent": True,
            },
        )
        self.events.randomize_insertive_object_mass = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("insertive_object"),
                "mass_distribution_params": (0.02, 0.4),
                "operation": "abs",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )
        self.events.receptive_object_material = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (0.6, 1.2),
                "dynamic_friction_range": (0.6, 1.2),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 256,
                "asset_cfg": SceneEntityCfg("receptive_object"),
                "make_consistent": True,
            },
        )
        self.events.reset_from_reset_states = EventTerm(
            func=task_mdp.MultiResetManager,
            mode="reset",
            params={
                "base_paths": [f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectAnywhereEEAnywhere"],
                "probs": [1.0],
                "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            },
        )


@configclass
class OODRGBOODAllCfg(Ur5eRobotiq2f85EvalRGBRelCartesianOSCOODCfg):
    """OOD RGB (textures/HDRIs) + OOD physics (OSC gains, robot dynamics, object properties)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_robot_actuator_parameters = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_operational_space_controller_gains,
            mode="reset",
            params={
                "action_name": "arm",
                "stiffness_distribution_params": (0.5, 2.4),
                "damping_distribution_params": (0.5, 1.6),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.robot_material = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (0.2, 1.8),
                "dynamic_friction_range": (0.2, 1.8),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 256,
                "asset_cfg": SceneEntityCfg("robot"),
                "make_consistent": True,
            },
        )
        self.events.randomize_robot_mass = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "mass_distribution_params": (0.2, 1.8),
                "operation": "scale",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )
        self.events.randomize_robot_joint_parameters = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_joint_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*", "finger_joint"]),
                "friction_distribution_params": (0.1, 5.0),
                "armature_distribution_params": (0.1, 5.0),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
        self.events.randomize_gripper_actuator_parameters = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
                "stiffness_distribution_params": (0.2, 2.8),
                "damping_distribution_params": (0.2, 2.8),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
        self.events.insertive_object_material = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (0.6, 1.4),
                "dynamic_friction_range": (0.6, 1.4),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 256,
                "asset_cfg": SceneEntityCfg("insertive_object"),
                "make_consistent": True,
            },
        )
        self.events.randomize_insertive_object_mass = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("insertive_object"),
                "mass_distribution_params": (0.02, 0.4),
                "operation": "abs",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )
        self.events.receptive_object_material = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (0.6, 1.2),
                "dynamic_friction_range": (0.6, 1.2),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 256,
                "asset_cfg": SceneEntityCfg("receptive_object"),
                "make_consistent": True,
            },
        )
        self.events.reset_from_reset_states = EventTerm(
            func=task_mdp.MultiResetManager,
            mode="reset",
            params={
                "base_paths": [f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectAnywhereEEAnywhere"],
                "probs": [1.0],
                "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            },
        )

