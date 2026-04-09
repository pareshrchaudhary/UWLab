# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from uwlab_rl.rsl_rl import (
    RslRlAsymmetricActorCriticCfg,
    RslRlMARLFullRecurrentRunnerCfg,
    RslRlMARLRecurrentRunnerCfg,
    RslRlMARLRunnerCfg,
    RslRLFancyActorCriticRecurrentCfg,
)
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
# CAGE Adversary MARL Runner Configs
# =============================================================================


@configclass
class MultiAgentRunner(RslRlMARLRunnerCfg):
    class_name: str = "MultiAgentRunner"
    num_steps_per_env = 32
    adversary_update_every_k_episodes = 5
    max_iterations = 40000
    save_interval = 100
    experiment_name = "cage_adversary"
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

    # Adversary policy and algorithm (one-step bandit-style)
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
        entropy_coef=0.006,
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


@configclass
class MultiAgentRecurrentRunner(RslRlMARLRecurrentRunnerCfg):
    class_name: str = "MultiAgentRunner"
    num_steps_per_env = 32
    adversary_update_every_k_episodes = 5
    max_iterations = 40000
    save_interval = 100
    experiment_name = "cage_adversary_recurrent"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    adversary_obs_groups = {
        "policy": ["adversary_policy"],
    }
    policy = RslRlAsymmetricActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # Adversary policy and algorithm (one-step bandit-style)
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
        clip_param=0.1,
        entropy_coef=0.006,
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


@configclass
class MultiAgentFullRecurrentRunner(RslRlMARLFullRecurrentRunnerCfg):
    class_name: str = "MultiAgentRunner"
    num_steps_per_env = 32
    adversary_update_every_k_episodes = 5
    max_iterations = 40000
    save_interval = 100
    experiment_name = "cage_adversary_full_recurrent"
    adversary_robot_parameters = ADVERSARY_POSE_ACTION_DIM
    adversary_parameter_names = CAGE_ADVERSARY_PARAMETER_NAMES
    adversary_param_extractor_fn = extract_cage_physics_params
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    adversary_obs_groups = {
        "policy": ["adversary_policy"],
    }
    policy = RslRLFancyActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
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
        clip_param=0.1,
        entropy_coef=0.006,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-4,
        schedule="fixed",
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# =============================================================================
# AdversaryAdvancedEventCfg runner (parameter adversary MARL)
# =============================================================================


@configclass
class AdversaryAdvancedRunner(RslRlMARLRunnerCfg):
    class_name: str = "MultiAgentRunner"
    num_steps_per_env = 32
    adversary_update_every_k_episodes = 5
    max_iterations = 40000
    save_interval = 100
    experiment_name = "cage_adversary_advanced"
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
        entropy_coef=0.006,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-4,
        schedule="fixed",
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    adversary_robot_parameters = ADVERSARY_ADVANCED_ACTION_DIM
    adversary_parameter_names = CAGE_ADVERSARY_PARAMETER_NAMES
    adversary_param_extractor_fn = extract_cage_physics_params
