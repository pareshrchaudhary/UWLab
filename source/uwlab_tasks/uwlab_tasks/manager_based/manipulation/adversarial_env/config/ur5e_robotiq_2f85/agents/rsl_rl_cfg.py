# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from uwlab_rl.rsl_rl import RslRlAdversarialRunnerCfg, RslRlFancyActorCriticCfg

def my_experts_observation_func(env):
    obs = env.unwrapped.obs_buf["expert_obs"]
    return obs


@configclass
class Base_PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 100
    resume = False
    experiment_name = "ur5e_robotiq_2f85_reset_states_agent"
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
class AdversarialPPORunnerCfg(RslRlAdversarialRunnerCfg):
    """Adversarial runner with Protagonist and Adversary policies.

    The obs_groups uses standard "policy"/"critic" keys that rsl_rl expects.
    The custom runner will remap these for each agent using the agent-specific
    obs_groups (protagonist_obs_groups, adversary_obs_groups).
    """

    class_name: str = "AdversarialOnPolicyRunner"

    num_steps_per_env = 32
    adversary_update_every_k_steps = 5
    max_iterations = 40000
    save_interval = 100
    experiment_name = "ur5e_robotiq_2f85_adversarial"

    # Protagonist obs_groups: maps "policy"/"critic" to env observation groups
    protagonist_obs_groups = {
        "policy": ["protagonist_policy"],
        "critic": ["protagonist_critic"],
    }

    # Adversary obs_groups: maps "policy"/"critic" to env observation groups
    adversary_obs_groups = {
        "policy": ["adversary_policy"],
        "critic": ["adversary_critic"],
    }

    # Protagonist policy and algorithm
    protagonist_policy = RslRlFancyActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,
    )
    protagonist_algorithm = RslRlPpoAlgorithmCfg(
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

    # Antagonist policy and algorithm (one-step bandit-style)
    adversary_policy = RslRlFancyActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="log",
        state_dependent_std=False,
    )
    adversary_algorithm = RslRlPpoAlgorithmCfg(
        # REINFORCE-like for one-step setting
        value_loss_coef=0.0,
        use_clipped_value_loss=False,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.0,
        entropy_coef=0.006,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=1.0,
        lam=1.0,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
