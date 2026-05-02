# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from ... import mdp as task_mdp
from .rl_state_cfg import Ur5eRobotiq2f85RelCartesianOSCEvalCfg


# ---------------------------------------------------------------------------
# Sim2real eval helpers (V4+). Each helper collapses one DR axis to a point
# estimate at the training-DR midpoint so the variant is reproducible without
# overrides; the swept value is supplied via Hydra at the CLI.
# ---------------------------------------------------------------------------


def _apply_friction_sweep_overrides(events):
    """V4: collapse peg + hole friction to a fixed point. Default sits past the
    upper edge of training DR for both surfaces — bites the omnireset policy on
    its own without any Hydra override. Sweep individual axes via the Hydra
    paths in the variant docstring.

    Training DR: peg static (1.0, 2.0) / hole static (0.2, 0.6).
    Default:     peg s=8.0 / hole s=2.5  (4× past upper edge of both — peg
    s=4.0/hole s=1.0 still gave 93% on omnireset, friction is a stubbornly
    robust axis for this policy).
    """
    events.insertive_object_material.params["static_friction_range"] = (8.0, 8.0)
    events.insertive_object_material.params["dynamic_friction_range"] = (7.2, 7.2)
    events.receptive_object_material.params["static_friction_range"] = (2.5, 2.5)
    events.receptive_object_material.params["dynamic_friction_range"] = (2.25, 2.25)


def _apply_peg_mass_sweep_overrides(events):
    """V5: collapse peg mass to a fixed point in kg. Default = 2.0 kg, 10× the
    upper edge of training DR (0.02, 0.2). Earlier sweep showed m=2.0 → 9%
    on the omnireset policy."""
    events.randomize_insertive_object_mass.params["mass_distribution_params"] = (2.0, 2.0)


def _apply_grasp_offset_overrides(observations, events):
    """V7: persistent SE(3) grasp offset paired with a peg-pose perception bias.

    Solo grasp offset was confirmed null at all magnitudes (up to 25 mm / 30°
    → 92%) because the policy obs already exposes the *actual* peg pose, so
    the policy compensates regardless of the offset. To make V7 a meaningful
    standalone test, we *also* poison the `insertive_asset_pose` ObsTerm with
    a per-episode bias. Now the policy commands gripper motions assuming the
    peg is in one place when it is actually offset elsewhere — the test
    becomes 'open-loop misalignment under uncertain peg pose'.

    Default: 15 mm / 8° physics-side grasp offset, 15 mm / 8° per-episode peg
    pose perception bias. (Earlier 5 mm / 3° still gave 96% on omnireset —
    the offsets need to be much larger than realistic to bite.)
    """
    # peg-pose perception bias (so policy can't see the actual offset)
    observations.policy.insertive_asset_pose = ObsTerm(
        func=task_mdp.target_asset_pose_with_perception_noise,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "rotation_repr": "axis_angle",
            "trans_std": 0.0,
            "rot_std": 0.0,
            "trans_bias_std": 0.015,  # 15 mm
            "rot_bias_std": 0.140,    # ≈8°
        },
    )
    # physics-side grasp offset
    events.apply_peg_grasp_offset = EventTerm(
        func=task_mdp.apply_peg_grasp_offset,
        mode="reset",
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "ee_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "trans_std": 0.015,  # 15 mm
            "rot_std": 0.140,    # ≈8°
        },
    )


def _apply_hole_pose_noise_overrides(observations):
    """V6: replace policy-group hole pose ObsTerm with a perception-error variant.

    Two error modes (sweep independently or together):
        params.trans_std       — per-step translation jitter [m]
        params.rot_std         — per-step rotation jitter    [rad]
        params.trans_bias_std  — per-episode translation bias [m]   (resamples on reset)
        params.rot_bias_std    — per-episode rotation bias    [rad] (resamples on reset)

    Default = 40 mm translation + 6° rotation per-episode bias — well outside
    realistic calibrated-vision magnitudes, picked so the variant bites the
    omnireset policy on its own. Bias is the real-world-relevant axis;
    per-step jitter averages out over history.
    """
    observations.policy.receptive_asset_pose = ObsTerm(
        func=task_mdp.target_asset_pose_with_perception_noise,
        params={
            "target_asset_cfg": SceneEntityCfg("receptive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "rotation_repr": "axis_angle",
            "trans_std": 0.0,
            "rot_std": 0.0,
            "trans_bias_std": 0.040,  # 40 mm
            "rot_bias_std": 0.105,    # ≈6°
        },
    )


def _apply_combo_stress_overrides(events, observations):
    """V8: multi-axis stress test. Per-axis values match the V4/V5/V6/V7
    standalone defaults — V8 is now "all biting axes simultaneously":
      - peg friction s=8.0, hole friction s=2.5         (from V4)
      - peg mass = 2.0 kg                               (from V5)
      - hole pose bias = 40 mm / 6°                     (from V6)
      - peg pose bias = 15 mm / 8°                      (from V7)
      - persistent grasp offset = 15 mm / 8°            (from V7)

    Expectation: V8 alone collapses to <= the worst single-axis result
    (V5 = 11% on the omnireset checkpoint, so V8 alone should be ~0–10%).

    NOT set here (use the matching CLI flag):
      - action delay  → --action_delay_steps N  (recommended N=1, ≈100 ms)
      - per-step action noise  → --action_noise <std>
    """
    # peg friction (from V4)
    events.insertive_object_material.params["static_friction_range"] = (8.0, 8.0)
    events.insertive_object_material.params["dynamic_friction_range"] = (7.2, 7.2)
    # hole friction (from V4)
    events.receptive_object_material.params["static_friction_range"] = (2.5, 2.5)
    events.receptive_object_material.params["dynamic_friction_range"] = (2.25, 2.25)
    # peg mass (from V5) — 10× training-DR upper of 0.2
    events.randomize_insertive_object_mass.params["mass_distribution_params"] = (2.0, 2.0)
    # hole pose perception bias (from V6)
    observations.policy.receptive_asset_pose = ObsTerm(
        func=task_mdp.target_asset_pose_with_perception_noise,
        params={
            "target_asset_cfg": SceneEntityCfg("receptive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "rotation_repr": "axis_angle",
            "trans_std": 0.0,
            "rot_std": 0.0,
            "trans_bias_std": 0.040,  # 40 mm
            "rot_bias_std": 0.105,    # ≈6°
        },
    )
    # peg pose perception bias (from V7)
    observations.policy.insertive_asset_pose = ObsTerm(
        func=task_mdp.target_asset_pose_with_perception_noise,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "rotation_repr": "axis_angle",
            "trans_std": 0.0,
            "rot_std": 0.0,
            "trans_bias_std": 0.015,  # 15 mm
            "rot_bias_std": 0.140,    # ≈8°
        },
    )
    # persistent grasp offset (from V7)
    events.apply_peg_grasp_offset = EventTerm(
        func=task_mdp.apply_peg_grasp_offset,
        mode="reset",
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "ee_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "trans_std": 0.015,  # 15 mm
            "rot_std": 0.140,    # ≈8°
        },
    )


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
    """Eval Variant 2: In-Distribution OSC gains: Out-of-Distribution Parameters
    (robot-side)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.robot_material = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (0.05, 5.0),
                "dynamic_friction_range": (0.05, 5.0),
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
                "mass_distribution_params": (0.1, 4.0),
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
                "friction_distribution_params": (0.05, 10.0),
                "armature_distribution_params": (0.05, 10.0),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
        self.events.randomize_gripper_actuator_parameters = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
                "stiffness_distribution_params": (0.1, 5.0),
                "damping_distribution_params": (0.1, 5.0),
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
    """Eval Variant 3: Out-of-Distribution Object Parameters. Ranges widened
    (peg/hole material + peg mass) to bite the omnireset policy."""

    def __post_init__(self):
        super().__post_init__()
        self.events.insertive_object_material = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (0.05, 5.0),
                "dynamic_friction_range": (0.05, 5.0),
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
                "mass_distribution_params": (0.005, 1.5),
                "operation": "abs",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )
        self.events.receptive_object_material = EventTerm(  # type: ignore[attr-defined]
            func=task_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "static_friction_range": (0.05, 5.0),
                "dynamic_friction_range": (0.05, 5.0),
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


# ---------------------------------------------------------------------------
# V4 — peg + hole contact-friction sweep (sim2real test #1)
# ---------------------------------------------------------------------------


@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalCfgV4(Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    """V4 (Stage 1): peg+hole friction collapsed to a fixed point.

    Sweep via Hydra:
        events.insertive_object_material.params.static_friction_range="[s,s]"
        events.insertive_object_material.params.dynamic_friction_range="[0.9*s,0.9*s]"
        events.receptive_object_material.params.static_friction_range="[s,s]"
        events.receptive_object_material.params.dynamic_friction_range="[0.9*s,0.9*s]"
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_friction_sweep_overrides(self.events)


# ---------------------------------------------------------------------------
# V5 — peg mass sweep (sim2real test #10)
# ---------------------------------------------------------------------------


@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalCfgV5(Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    """V5 (Stage 1): peg mass collapsed to a fixed value (kg, abs).

    Sweep via Hydra:
        env.events.randomize_insertive_object_mass.params.mass_distribution_params="[m,m]"
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_peg_mass_sweep_overrides(self.events)


# ---------------------------------------------------------------------------
# V6 — hole pose perception noise (sim2real test #3)
# ---------------------------------------------------------------------------


@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalCfgV6(Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    """V6: per-step Gaussian noise on the hole pose obs (translation + rotation).

    Proxies real-world perception error: a deployed state-based policy receives
    its hole pose from some perception system (vision, ICP, fiducial), which has
    error. This variant injects that error into the policy's input.

    Defaults to zero noise (no-op). Sweep via Hydra:
        env.observations.policy.receptive_asset_pose.params.trans_std=0.005   # m
        env.observations.policy.receptive_asset_pose.params.rot_std=0.0087    # rad
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_hole_pose_noise_overrides(self.observations)


# ---------------------------------------------------------------------------
# V7 — persistent grasp offset (sim2real test #4)
# ---------------------------------------------------------------------------


@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalCfgV7(Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    """V7: per-episode SE(3) offset between the peg and the end-effector grasp frame.

    Models real-world grasp imperfection — every grasp leaves the peg slightly
    off-centre in the gripper. The peg's actual insertion axis is then offset
    from the gripper's commanded axis. The policy still observes the peg pose
    accurately, but its commanded gripper motions don't align the peg with the
    hole.

    Defaults to zero offset (no-op). Sweep via Hydra:
        env.events.apply_peg_grasp_offset.params.trans_std=0.001   # m  (1 mm)
        env.events.apply_peg_grasp_offset.params.rot_std=0.0175    # rad (1°)
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_grasp_offset_overrides(self.observations, self.events)


# ---------------------------------------------------------------------------
# V8 — Realistic-magnitude multi-axis combo stress test
# ---------------------------------------------------------------------------


@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalCfgV8(Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    """V8: multi-axis combo at realistic deployment magnitudes.

    Stacks friction + mass + perception bias + grasp offset, each at the upper
    edge of training DR or just outside it. Each axis individually is mild;
    combined, they expose the limits of pure DR — a policy trained to be
    robust *on average* across each axis independently is not necessarily
    robust to *worst-case combinations* of multiple axes.

    Pair with `--action_delay_steps 1` (≈100 ms latency) for the full combo,
    and `--track_tries` to see the attempt-distribution shift.

    Headline DR-vs-adversarial test: the success rate here is the "DR ceiling"
    — adversarial training is hypothesized to produce higher success on this
    same eval.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_combo_stress_overrides(self.events, self.observations)
