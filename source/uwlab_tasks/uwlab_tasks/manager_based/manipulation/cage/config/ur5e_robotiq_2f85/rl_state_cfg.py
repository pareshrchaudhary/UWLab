# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
from uwlab_assets.robots.ur5e_robotiq_gripper import IMPLICIT_UR5E_ROBOTIQ_2F85

from uwlab_tasks.manager_based.manipulation.cage.config.ur5e_robotiq_2f85.actions import (
    UR5E_ROBOTIQ_2F85_ADVERSARY_ACTION,
    UR5E_ROBOTIQ_2F85_ADVERSARY_ADVANCED_ACTION,
    Ur5eRobotiq2f85AdversaryOSCAction,
    Ur5eRobotiq2f85AdversaryAdvancedOSCAction,
    Ur5eRobotiq2f85RelativeOSCAction,
)

from ... import mdp as task_mdp


@configclass
class RlStateSceneCfg(InteractiveSceneCfg):
    """Scene configuration for RL state environment."""

    robot = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

    insertive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    receptive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Environment
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -0.881), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention/pat_vention.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    ur5_metal_support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/UR5MetalSupport",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0, -0.013), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention2/Ur5MetalSupport/ur5plate.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.868)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class AdversaryBaseEventCfg:
    """Base events for adversary training with fixed material/mass values.

    Material and mass properties are held constant here. Adversary-controlled
    randomization of these parameters is deferred to later stages.
    """

    reset_robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.3),
            "dynamic_friction_range": (0.2, 0.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("robot"),
            "make_consistent": True,
        },
    )

    insertive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.3),
            "dynamic_friction_range": (0.2, 0.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "make_consistent": True,
        },
    )

    receptive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.3),
            "dynamic_friction_range": (0.2, 0.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "make_consistent": True,
        },
    )

    table_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.3),
            "dynamic_friction_range": (0.2, 0.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("table"),
            "make_consistent": True,
        },
    )

    randomize_robot_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_insertive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "mass_distribution_params": (0.02, 0.02),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_receptive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_table_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_gripper_actuator_parameters = EventTerm(
        func=task_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})

    reset_receptive_object_pose = EventTerm(
        func=task_mdp.adversary_reset_root_states_from_action,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.3, 0.55),
                "y": (-0.1, 0.3),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-np.pi / 12, np.pi / 12),
            },
            "asset_cfgs": {"receptive_object": SceneEntityCfg("receptive_object")},
            "offset_asset_cfg": SceneEntityCfg("ur5_metal_support"),
            "use_bottom_offset": True,
            "action_name": "adversaryaction",
            "action_x_index": 0,
            "action_y_index": 1,
            "action_yaw_index": 2,
        },
    )

    reset_insertive_object = EventTerm(
        func=task_mdp.adversary_reset_insertive_from_assembled_offset,
        mode="reset",
        params={
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "pose_range_b": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.3),
                "roll": (-np.pi, np.pi),
                "pitch": (-np.pi, np.pi),
                "yaw": (-np.pi, np.pi),
            },
            "action_name": "adversaryaction",
            "action_start_index": 3,
        },
    )

    reset_end_effector_pose = EventTerm(
        func=task_mdp.adversary_reset_end_effector_from_action,
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("robot"),
            "fixed_asset_offset": None,
            "pose_range_b": {
                "x": (0.3, 0.7),
                "y": (-0.4, 0.5),
                "z": (0.05, 0.5),
                "roll": (-np.pi / 16, np.pi / 16),
                "pitch": (np.pi / 4 - np.pi / 16, 3 * np.pi / 4 + np.pi / 16),
                "yaw": (np.pi / 2 - np.pi / 16, 3 * np.pi / 2 + np.pi / 16),
            },
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "gripper_range": (0.0, 0.785398),
            "action_name": "adversaryaction",
            "action_start_index": 9,
            "gripper_action_index": 15,
        },
    )


@configclass
class AdversaryAdvancedEventCfg:
    """Advanced adversary events: adversary controls friction, mass, joint and actuator
    gains, and OSC gains while pose resets use standard randomization.

    Action index map (18D total, ``ADVERSARY_ADVANCED_ACTION_DIM``):
        0-1:   robot static/dynamic friction
        2-3:   insertive object static/dynamic friction
        4-5:   receptive object static/dynamic friction
        6-7:   table static/dynamic friction
        8:     robot mass scale
        9:     insertive object mass (absolute)
        10:    receptive object mass scale
        11:    table mass scale
        12:    robot joint friction scale
        13:    robot joint armature scale
        14:    gripper stiffness scale
        15:    gripper damping scale
        16:    OSC stiffness scale
        17:    OSC damping scale
    """

    # --- Adversary-controlled friction ---

    adversary_robot_material = EventTerm(
        func=task_mdp.adversary_rigid_body_material_from_action,
        mode="reset",
        params={
            "action_name": "adversaryaction",
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.05, 1.0),
            "dynamic_friction_range": (0.05, 1.0),
            "action_static_index": 0,
            "action_dynamic_index": 1,
            "make_consistent": True,
        },
    )

    adversary_insertive_object_material = EventTerm(
        func=task_mdp.adversary_rigid_body_material_from_action,
        mode="reset",
        params={
            "action_name": "adversaryaction",
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "static_friction_range": (0.05, 1.0),
            "dynamic_friction_range": (0.05, 1.0),
            "action_static_index": 2,
            "action_dynamic_index": 3,
            "make_consistent": True,
        },
    )

    adversary_receptive_object_material = EventTerm(
        func=task_mdp.adversary_rigid_body_material_from_action,
        mode="reset",
        params={
            "action_name": "adversaryaction",
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "static_friction_range": (0.05, 1.0),
            "dynamic_friction_range": (0.05, 1.0),
            "action_static_index": 4,
            "action_dynamic_index": 5,
            "make_consistent": True,
        },
    )

    adversary_table_material = EventTerm(
        func=task_mdp.adversary_rigid_body_material_from_action,
        mode="reset",
        params={
            "action_name": "adversaryaction",
            "asset_cfg": SceneEntityCfg("table"),
            "static_friction_range": (0.05, 1.0),
            "dynamic_friction_range": (0.05, 1.0),
            "action_static_index": 6,
            "action_dynamic_index": 7,
            "make_consistent": True,
        },
    )

    # --- Adversary-controlled mass ---

    adversary_robot_mass = EventTerm(
        func=task_mdp.adversary_rigid_body_mass_scale_from_action,
        mode="reset",
        params={
            "action_name": "adversaryaction",
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_scale_range": (0.5, 2.0),
            "mass_scale_index": 8,
        },
    )

    adversary_insertive_object_mass = EventTerm(
        func=task_mdp.adversary_rigid_body_mass_abs_from_action,
        mode="reset",
        params={
            "action_name": "adversaryaction",
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "mass_range": (0.005, 0.1),
            "mass_index": 9,
        },
    )

    adversary_receptive_object_mass = EventTerm(
        func=task_mdp.adversary_rigid_body_mass_scale_from_action,
        mode="reset",
        params={
            "action_name": "adversaryaction",
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "mass_scale_range": (0.5, 2.0),
            "mass_scale_index": 10,
        },
    )

    adversary_table_mass = EventTerm(
        func=task_mdp.adversary_rigid_body_mass_scale_from_action,
        mode="reset",
        params={
            "action_name": "adversaryaction",
            "asset_cfg": SceneEntityCfg("table"),
            "mass_scale_range": (0.5, 2.0),
            "mass_scale_index": 11,
        },
    )

    # --- Adversary-controlled joint dynamics ---

    adversary_robot_joint_params = EventTerm(
        func=task_mdp.adversary_joint_friction_armature_from_action,
        mode="reset",
        params={
            "action_name": "adversaryaction",
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_scale_range": (0.5, 2.0),
            "armature_scale_range": (0.5, 2.0),
            "action_friction_index": 12,
            "action_armature_index": 13,
        },
    )

    # --- Adversary-controlled gripper actuator gains ---

    adversary_gripper_gains = EventTerm(
        func=task_mdp.adversary_gripper_actuator_gains_from_action,
        mode="reset",
        params={
            "action_name": "adversaryaction",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "stiffness_scale_range": (0.5, 2.0),
            "damping_scale_range": (0.5, 2.0),
            "action_stiffness_index": 14,
            "action_damping_index": 15,
        },
    )

    # --- Adversary-controlled OSC gains ---

    adversary_osc_gains = EventTerm(
        func=task_mdp.adversary_osc_gains_from_action,
        mode="reset",
        params={
            "action_name": "arm",
            "adversary_action_name": "adversaryaction",
            "stiffness_scale_range": (0.5, 2.0),
            "damping_scale_range": (0.5, 2.0),
            "action_stiffness_index": 16,
            "action_damping_index": 17,
        },
    )

    # --- Standard pose resets (not adversary-controlled) ---

    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})

    reset_receptive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.3, 0.55),
                "y": (-0.1, 0.3),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-np.pi / 12, np.pi / 12),
            },
            "velocity_range": {},
            "asset_cfgs": {"receptive_object": SceneEntityCfg("receptive_object")},
            "offset_asset_cfg": SceneEntityCfg("ur5_metal_support"),
            "use_bottom_offset": True,
        },
    )

    reset_insertive_object = EventTerm(
        func=task_mdp.reset_insertive_object_from_partial_assembly_dataset,
        mode="reset",
        params={
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "pose_range_b": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.3),
                "roll": (-np.pi, np.pi),
                "pitch": (-np.pi, np.pi),
                "yaw": (-np.pi, np.pi),
            },
        },
    )

    reset_end_effector_pose = EventTerm(
        func=task_mdp.reset_end_effector_round_fixed_asset,
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("robot"),
            "fixed_asset_offset": None,
            "pose_range_b": {
                "x": (0.3, 0.7),
                "y": (-0.4, 0.5),
                "z": (0.05, 0.5),
                "roll": (-np.pi / 16, np.pi / 16),
                "pitch": (np.pi / 4 - np.pi / 16, 3 * np.pi / 4 + np.pi / 16),
                "yaw": (np.pi / 2 - np.pi / 16, 3 * np.pi / 2 + np.pi / 16),
            },
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    task_command = task_mdp.TaskCommandCfg(
        asset_cfg=SceneEntityCfg("robot", body_names="body"),
        resampling_time_range=(1e6, 1e6),
        insertive_asset_cfg=SceneEntityCfg("insertive_object"),
        receptive_asset_cfg=SceneEntityCfg("receptive_object"),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        prev_actions = ObsTerm(
            func=task_mdp.policy_last_action,
            params={"adversary_action_dim": UR5E_ROBOTIQ_2F85_ADVERSARY_ACTION.action_dim},
        )

        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )

        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
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
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations for policy group."""

        prev_actions = ObsTerm(
            func=task_mdp.policy_last_action,
            params={"adversary_action_dim": UR5E_ROBOTIQ_2F85_ADVERSARY_ACTION.action_dim},
        )

        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )

        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
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
        time_left = ObsTerm(func=task_mdp.time_left)

        joint_vel = ObsTerm(func=task_mdp.joint_vel)

        end_effector_vel_lin_ang_b = ObsTerm(
            func=task_mdp.asset_link_velocity_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
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

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class AdversaryPolicyCfg(ObsGroup):
    """Adversary actor observations (bandit-style: noise input only)."""

    noise = ObsTerm(
        func=task_mdp.adversary_noise,
        params={"dim": UR5E_ROBOTIQ_2F85_ADVERSARY_ACTION.action_dim},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class AdversaryAdvancedPolicyCfg(ObsGroup):
    """Advanced adversary actor observations (bandit-style: noise input only, 18D)."""

    noise = ObsTerm(
        func=task_mdp.adversary_noise,
        params={"dim": UR5E_ROBOTIQ_2F85_ADVERSARY_ADVANCED_ACTION.action_dim},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class MARLObservationsCfg:
    """Combined observation groups for MARL training (pose adversary).

    Keys in obs_buf:
        - "policy": policy actor observations (with adversary action stripped from prev_actions)
        - "critic": policy critic observations
        - "adversary_policy": adversary actor observations (noise only)
    """

    policy: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()
    critic: ObservationsCfg.CriticCfg = ObservationsCfg.CriticCfg()
    adversary_policy: AdversaryPolicyCfg = AdversaryPolicyCfg()


@configclass
class AdversaryAdvancedMARLObservationsCfg:
    """Combined observation groups for MARL training (AdversaryAdvancedEventCfg).

    Keys in obs_buf:
        - "policy": policy actor observations (with adversary action stripped from prev_actions)
        - "critic": policy critic observations
        - "adversary_policy": advanced adversary actor observations (noise only, 18D)
    """

    policy: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()
    critic: ObservationsCfg.CriticCfg = ObservationsCfg.CriticCfg()
    adversary_policy: AdversaryAdvancedPolicyCfg = AdversaryAdvancedPolicyCfg()


@configclass
class RewardsCfg:

    # safety rewards

    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)

    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-1e-3)

    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_clamped,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])},
    )

    abnormal_robot = RewTerm(func=task_mdp.abnormal_robot_state, weight=-100.0)

    # task rewards

    progress_context = RewTerm(
        func=task_mdp.ProgressContext,  # type: ignore
        weight=0.1,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    ee_asset_distance = RewTerm(
        func=task_mdp.ee_asset_distance_tanh,
        weight=0.1,
        params={
            "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_offset_metadata_key": "gripper_offset",
            "std": 1.0,
        },
    )

    dense_success_reward = RewTerm(func=task_mdp.dense_success_reward, weight=0.1, params={"std": 1.0})

    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)


@configclass
class NoCurriculumsCfg:
    """No curriculum"""

    pass


def make_insertive_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_receptive_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


variants = {
    "scene.insertive_object": {
        "fbleg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareLeg/square_leg.usd"),
        "fbdrawerbottom": make_insertive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBottom/drawer_bottom.usd"
        ),
        "peg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd"),
        "cupcake": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/CupCake/cupcake.usd"),
        "cube": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/InsertiveCube/insertive_cube.usd"),
        "rectangle": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Rectangle/rectangle.usd"),
    },
    "scene.receptive_object": {
        "fbtabletop": make_receptive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareTableTop/square_table_top.usd"
        ),
        "fbdrawerbox": make_receptive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBox/drawer_box.usd"
        ),
        "peghole": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd"),
        "plate": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Plate/plate.usd"),
        "cube": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/ReceptiveCube/receptive_cube.usd"),
        "wall": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Wall/wall.usd"),
    },
}


@configclass
class Ur5eRobotiq2f85RlStateCfg(ManagerBasedRLEnvCfg):
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: AdversaryBaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0

        # Contact and solver settings
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31

        # Render settings
        self.sim.render.enable_dlssg = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True


@configclass
class Ur5eRobotiq2f85AdversaryTrainCfg(Ur5eRobotiq2f85RlStateCfg):
    """CAGE adversarial training: policy + pose adversary MARL with implicit actuator."""

    events: AdversaryBaseEventCfg = AdversaryBaseEventCfg()
    actions: Ur5eRobotiq2f85AdversaryOSCAction = Ur5eRobotiq2f85AdversaryOSCAction()
    observations: MARLObservationsCfg = MARLObservationsCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()


@configclass
class Ur5eRobotiq2f85AdversaryAdvancedTrainCfg(Ur5eRobotiq2f85RlStateCfg):
    """CAGE adversarial training: policy + AdversaryAdvancedEventCfg (parameter adversary) MARL.

    The adversary controls 18 parameters (friction, mass, actuator gains,
    OSC gains) while poses are reset with standard randomization.
    """

    events: AdversaryAdvancedEventCfg = AdversaryAdvancedEventCfg()
    actions: Ur5eRobotiq2f85AdversaryAdvancedOSCAction = Ur5eRobotiq2f85AdversaryAdvancedOSCAction()
    observations: AdversaryAdvancedMARLObservationsCfg = AdversaryAdvancedMARLObservationsCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
