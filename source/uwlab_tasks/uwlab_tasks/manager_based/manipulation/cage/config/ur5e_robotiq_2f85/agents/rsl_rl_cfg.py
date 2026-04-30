# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from uwlab_rl.rsl_rl.rl_cfg import (
    BehaviorCloningCfg,
    OffPolicyAlgorithmCfg,
    RslRlFancyActorCriticCfg,
    RslRlFancyPpoAlgorithmCfg,
)

from rsl_rl.utils.logger import extract_cage_physics_params

from uwlab_tasks.manager_based.manipulation.cage.mdp.actions.adversary_actions_cfg import (
    ADVERSARY_ADVANCED_ACTION_DIM,
    ADVERSARY_POSE_ACTION_DIM,
)

CAGE_ADVERSARY_PARAMETER_NAMES = [
    "receptive_object_x", "receptive_object_y", "receptive_object_yaw",
    "insertive_object_rel_x", "insertive_object_rel_y", "insertive_object_rel_z",
    "insertive_object_rel_roll", "insertive_object_rel_pitch", "insertive_object_rel_yaw",
    "end_effector_rel_x", "end_effector_rel_y", "end_effector_rel_z",
    "end_effector_rel_roll", "end_effector_rel_pitch", "end_effector_rel_yaw",
]

CAGE_ADVERSARY_ADVANCED_PARAMETER_NAMES = [
    "robot_static_friction", "robot_dynamic_friction",
    "insertive_object_static_friction", "insertive_object_dynamic_friction",
    "receptive_object_static_friction", "receptive_object_dynamic_friction",
    "table_static_friction", "table_dynamic_friction",
    "robot_mass_scale", "insertive_object_mass_abs",
    "receptive_object_mass_scale", "table_mass_scale",
    "arm_armature_shoulder_pan", "arm_armature_shoulder_lift",
    "arm_armature_elbow", "arm_armature_wrist_1",
    "arm_armature_wrist_2", "arm_armature_wrist_3",
    "arm_friction_shoulder_pan", "arm_friction_shoulder_lift",
    "arm_friction_elbow", "arm_friction_wrist_1",
    "arm_friction_wrist_2", "arm_friction_wrist_3",
    "gripper_stiffness_scale", "gripper_damping_scale",
    "osc_kp_xyz_scale", "osc_kp_rpy_scale",
    "osc_damping_ratio_xyz_scale", "osc_damping_ratio_rpy_scale",
]


def my_experts_observation_func(env):
    obs = env.unwrapped.obs_buf["expert_obs"]
    return obs


@configclass
class Base_PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 100
    resume = False
    experiment_name = "ur5e_robotiq_2f85_cage_agent"
    policy = RslRlFancyActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Base_DAggerRunnerCfg(Base_PPORunnerCfg):
    algorithm = RslRlFancyPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        offline_algorithm_cfg=OffPolicyAlgorithmCfg(
            behavior_cloning_cfg=BehaviorCloningCfg(
                experts_path=[""],
                experts_loader="torch.jit.load",
                experts_observation_group_cfg="uwlab_tasks.manager_based.manipulation.cage.config.ur5e_robotiq_2f85.rl_state_cfg:ObservationsCfg.PolicyCfg",
                experts_observation_func=my_experts_observation_func,
                experts_action_group_cfg="uwlab_tasks.manager_based.manipulation.cage.config.ur5e_robotiq_2f85.actions:Ur5eRobotiq2f85RelativeOSCAction",
                cloning_loss_coeff=1.0,
                loss_decay=1.0,
            )
        ),
    )


# =============================================================================
# Adversary Base Runner
# =============================================================================


@configclass
class AdversaryBaseRunner(RslRlBaseRunnerCfg):
    """Runner for inline-settling adversary + protagonist training.

    Each env continuously alternates between LIVE (protagonist-driven,
    contributes PPO gradient) and SETTLING (adversary-driven, ``settle_max_steps``
    window terminated by Isaac's natural time_out — transitions stored with
    ``valid_mask=0``). On a successful settle, the pre-success scene state is
    restored and the env spends ``regret_k`` LIVE episodes on it before
    rotating to a fresh adversary proposal. See
    ``notes/adversary-dip-investigation.md`` §5 for the design motivation.
    """

    class_name: str = "MultiAgentRunner"
    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 100
    experiment_name = "cage_adversary_base"

    # Regret window: number of LIVE episodes the protagonist gets on each
    # validated start state before the env flips to SETTLING with a fresh
    # proposal. Regret = max-mean over these K returns, attributed to the
    # adversary's proposal as its learning signal.
    regret_k: int = 6
    # Weight on the raw gen_reward term when combining with regret:
    # ``combined = beta_gen_reward * gen_reward + regret``. 0 disables shaping
    # and leaves regret as the sole signal.
    beta_gen_reward: float = 1.0
    # KL(π_{n-1} || π_n) penalty coefficient added to the adversary loss.
    # Anchors the adversary's new distribution to the previous one. 0 disables.
    adversary_kl_penalty_coef: float = 0.0

    # Inline-settling knobs. Everything here is a production parameter;
    # there are no stage-gating flags.
    inline_settling: dict = {
        "settle_max_steps": 20,               # 2.0s / 0.1s control-step
        "invalid_settle_penalty": -1.0,       # teacher reward on forced-LIVE commits
        "max_resample_retries": 5,            # per-env settle attempts before giving up
        "adversary_update_batch_size": None,  # None ⇒ num_envs (per-rank)
        "settling_gripper_default_action": -1.0,
    }

    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    adversary_obs_groups = {
        "policy": ["adversary_policy"],
    }
    policy = RslRlFancyActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # Adversary policy and algorithm
    adversary_policy = RslRlFancyActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[1],
        activation="elu",
        noise_std_type="log",
        state_dependent_std=False,
    )
    adversary_algorithm = RslRlPpoAlgorithmCfg(
        class_name="Reinforce",
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-4,
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    adversary_robot_parameters = ADVERSARY_POSE_ACTION_DIM
    adversary_parameter_names = CAGE_ADVERSARY_PARAMETER_NAMES
    adversary_param_extractor_fn = None


# =============================================================================
# AdversaryAdvancedEventCfg runner
# =============================================================================

@configclass
class AdversaryAdvancedRunner(RslRlBaseRunnerCfg):
    """Runner for advanced (parameter) adversary + policy training.

    Same inline-settling pipeline as ``AdversaryBaseRunner``; the adversary's
    action space is the full parameter set (friction, mass, armature, OSC
    gains, etc.) instead of just the end-effector pose.
    """

    class_name: str = "MultiAgentRunner"
    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 100
    experiment_name = "cage_adversary_advanced"

    regret_k: int = 6
    beta_gen_reward: float = 1.0
    adversary_kl_penalty_coef: float = 0.0

    inline_settling: dict = {
        "settle_max_steps": 20,
        "invalid_settle_penalty": -1.0,
        "max_resample_retries": 5,
        "adversary_update_batch_size": None,
        # Parameter-only adversary: dataset resets are pre-validated, no
        # SETTLING gate needed. Adversary reward = -ep_return per rollout.
        "skip_settling": True,
    }

    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    adversary_obs_groups = {
        "policy": ["adversary_policy"],
    }
    policy = RslRlFancyActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # Advanced adversary policy and algorithm
    adversary_policy = RslRlFancyActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        actor_hidden_dims=[128, 64, 32],
        critic_obs_normalization=False,
        critic_hidden_dims=[1],
        activation="elu",
        noise_std_type="log",
        state_dependent_std=False,
    )
    adversary_algorithm = RslRlPpoAlgorithmCfg(
        class_name="Reinforce",
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.05,  # bumped to maintain exploration across 30-D param action
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-4,
        schedule="fixed",
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    adversary_robot_parameters = ADVERSARY_ADVANCED_ACTION_DIM
    adversary_parameter_names = CAGE_ADVERSARY_ADVANCED_PARAMETER_NAMES
    adversary_param_extractor_fn = extract_cage_physics_params
