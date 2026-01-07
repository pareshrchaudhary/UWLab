# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from uwlab_rl.rsl_rl.rl_cfg import RslRlFancyActorCriticCfg


def my_experts_observation_func(env):
    obs = env.unwrapped.obs_buf["expert_obs"]
    return obs


@configclass
class Protagonist_PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 100
    resume = False
    experiment_name = "ur5e_robotiq_2f85_protagonist_agent"
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
class Adversary_PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # One-step adversary: only acts on reset (bandit-style).
    num_steps_per_env = 1
    max_iterations = 40000
    save_interval = 100
    resume = False
    experiment_name = "ur5e_robotiq_2f85_adversary_agent"
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
        # Make PPO behave closer to REINFORCE for the one-step setting:
        # - no critic loss
        # - single epoch/minibatch
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


@configclass
class AdversarialPPORunnerCfg():
    protagonist: Protagonist_PPORunnerCfg = Protagonist_PPORunnerCfg()
    adversary: Adversary_PPORunnerCfg = Adversary_PPORunnerCfg()
    
    # Which obs keys to use
    protagonist_obs_key: str = "protagonist_policy"
    protagonist_critic_key: str = "protagonist_critic"
    adversary_obs_key: str = "adversary_actor"
    adversary_critic_key: str = "adversary_critic"