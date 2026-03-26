# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from ... import mdp as task_mdp
from .rl_state_cfg import Ur5eRobotiq2f85RelCartesianOSCEvalCfg


@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalCfgV0(Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    """Eval Variant 0: In-Distribution OSC gains: In-Distribution Parameters"""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_osc_gains = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rel_cartesian_osc_gains_fixed,
            mode="reset",
            params={
                "action_name": "arm",
                "scale_range": (0.7, 1.3),
            },
        )
        self.events.reset_from_reset_states = EventTerm(
            func=task_mdp.MultiResetManager,
            mode="reset",
            params={
                "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
                "reset_types": ["ObjectAnywhereEEAnywhere"],
                "probs": [1.0],
                "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            },
        )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalCfgV1(Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    """Eval Variant 1: Out-of-Distribution OSC gains: In-Distribution Parameters"""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_osc_gains = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rel_cartesian_osc_gains_fixed,
            mode="reset",
            params={
                "action_name": "arm",
                "scale_range": (0.2, 2.4),
            },
        )
        self.events.reset_from_reset_states = EventTerm(
            func=task_mdp.MultiResetManager,
            mode="reset",
            params={
                "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
                "reset_types": ["ObjectAnywhereEEAnywhere"],
                "probs": [1.0],
                "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            },
        )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalCfgV2(Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    """Eval Variant 2: In-Distribution OSC gains: Out-of-Distribution Parameters"""

    def __post_init__(self):
        super().__post_init__()
        self.events.robot_material = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (0.2, 2.4),
                "dynamic_friction_range": (0.2, 2.4),
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
                "mass_distribution_params": (0.2, 2.4),
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
        self.events.randomize_osc_gains = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rel_cartesian_osc_gains_fixed,
            mode="reset",
            params={
                "action_name": "arm",
                "scale_range": (0.7, 1.3),
            },
        )
        self.events.reset_from_reset_states = EventTerm(
            func=task_mdp.MultiResetManager,
            mode="reset",
            params={
                "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
                "reset_types": ["ObjectAnywhereEEAnywhere"],
                "probs": [1.0],
                "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            },
        )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalCfgV3(Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    """Eval Variant 3: Out-of-Distribution Object Parameters"""

    def __post_init__(self):
        super().__post_init__()
        self.events.insertive_object_material = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (0.2, 2.8),
                "dynamic_friction_range": (0.2, 2.8),
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
                "mass_distribution_params": (0.02, 0.6),
                "operation": "abs",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )
        self.events.receptive_object_material = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (0.2, 2.8),
                "dynamic_friction_range": (0.2, 2.8),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 256,
                "asset_cfg": SceneEntityCfg("receptive_object"),
                "make_consistent": True,
            },
        )
        self.events.randomize_osc_gains = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rel_cartesian_osc_gains_fixed,
            mode="reset",
            params={
                "action_name": "arm",
                "scale_range": (0.7, 1.3),
            },
        )
        self.events.reset_from_reset_states = EventTerm(
            func=task_mdp.MultiResetManager,
            mode="reset",
            params={
                "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
                "reset_types": ["ObjectAnywhereEEAnywhere"],
                "probs": [1.0],
                "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            },
        )


EVAL_VARIANTS = [
    Ur5eRobotiq2f85RelCartesianOSCEvalCfgV0,
    Ur5eRobotiq2f85RelCartesianOSCEvalCfgV1,
    Ur5eRobotiq2f85RelCartesianOSCEvalCfgV2,
    Ur5eRobotiq2f85RelCartesianOSCEvalCfgV3,
]
