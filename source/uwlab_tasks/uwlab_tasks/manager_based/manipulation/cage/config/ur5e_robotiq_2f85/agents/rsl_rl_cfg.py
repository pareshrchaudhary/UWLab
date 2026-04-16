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
    "robot_static_friction", "robot_dynamic_friction",
    "insertive_object_static_friction", "insertive_object_dynamic_friction",
    "receptive_object_static_friction", "receptive_object_dynamic_friction",
    "table_static_friction", "table_dynamic_friction",
    "robot_mass_scale", "insertive_object_mass_scale",
    "receptive_object_mass_scale", "table_mass_scale",
    "robot_joint_friction_scale", "robot_joint_armature_scale",
    "gripper_stiffness_scale", "gripper_damping_scale",
    "osc_stiffness_scale", "osc_damping_scale",
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
    """Runner for adversary reset state generation + policy training.

    Uses MultiAgentRunner which alternates between:
    - Generation loop: adversary proposes states, physics settles, valid → buffer
    - Training loop: policy trains on buffer-loaded states, regret → adversary
    """

    class_name: str = "MultiAgentRunner"
    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 100
    experiment_name = "cage_adversary_base"

    # Generation-specific config
    generation_max_steps: int = 1000  # ~50 episode cycles at 20 steps/episode
    generation_episode_length_s: float = 2.0  # Phase A override; Phase B uses the env's natural episode_length_s
    # K-episode cycle: each buffer slot is credited with max-mean regret once
    # it has banked `regret_k` protagonist episodes. When every slot crosses K,
    # Phase A refills the buffer using the current adversary policy.
    regret_k: int = 3
    max_iters_per_cycle: int = 50  # safety cap; force-refill even if some slots haven't reached K
    # Scale on gen_reward in the combined adversary reward:
    #   adv_reward[i] = beta_gen_reward * gen_reward[i] + regret[i]
    # 0.0 disables gen_reward shaping entirely (regret-only curriculum signal).
    beta_gen_reward: float = 1.0

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

    # Adversary policy and algorithm (bandit-style)
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
        class_name="SimplePPO",
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.05,  # bumped from 0.006 to maintain exploration spread
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-4,
        schedule="fixed",
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    adversary_robot_parameters = ADVERSARY_POSE_ACTION_DIM
    adversary_parameter_names = CAGE_ADVERSARY_PARAMETER_NAMES
    adversary_param_extractor_fn = extract_cage_physics_params


# =============================================================================
# AdversaryAdvancedEventCfg runner (parameter adversary MARL)
# =============================================================================


@configclass
class AdversaryAdvancedRunner(RslRlBaseRunnerCfg):
    """Runner for advanced (parameter) adversary + policy training.

    Same validated-reset framework as ``AdversaryBaseRunner``: Phase A samples
    adversary parameter actions, physics settles, valid states flow into the
    reset state buffer; Phase B trains the policy from pinned buffer slots
    with regret crediting the originating Phase A action.
    """

    class_name: str = "MultiAgentRunner"
    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 100
    experiment_name = "cage_adversary_advanced"

    # Generation-specific config (same semantics as AdversaryBaseRunner).
    generation_max_steps: int = 1000
    generation_episode_length_s: float = 2.0
    regret_k: int = 3
    max_iters_per_cycle: int = 50
    beta_gen_reward: float = 1.0

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
        class_name="SimplePPO",
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