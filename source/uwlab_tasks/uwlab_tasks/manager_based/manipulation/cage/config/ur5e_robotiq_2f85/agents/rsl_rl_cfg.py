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

from uwlab_tasks.manager_based.manipulation.cage.mdp.actions.adversary_actions_cfg import (
    ADVERSARY_ADVANCED_ACTION_DIM,
    ADVERSARY_POSE_ACTION_DIM,
)


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


@configclass
class AdversaryBaseRunner(RslRlBaseRunnerCfg):
    class_name: str = "MultiAgentRunner"
    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 100
    experiment_name = "cage_adversary_base"

    regret_k: int = 3
    beta_gen_reward: float = 1.0

    inline_settling: dict = {
        "settle_max_steps": 20,               # 2.0s / 0.1s control-step
        "invalid_settle_penalty": -1.0,       # teacher reward for exhausted settle proposals
        "max_resample_retries": 20,           # avoid hard LIVE shifts from early settle failures
        "force_live_after_max_retries": False,
        "live_handoff_hold_steps": 4,         # mask PPO while holding the validated reset pose
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
        desired_kl=0.003,
        max_grad_norm=1.0,
    )
    adversary_robot_parameters = ADVERSARY_POSE_ACTION_DIM


@configclass
class AdversaryAdvancedRunner(RslRlBaseRunnerCfg):
    class_name: str = "MultiAgentRunner"
    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 100
    experiment_name = "cage_adversary_advanced"

    regret_k: int = 6
    beta_gen_reward: float = 1.0

    inline_settling: dict = {
        "settle_max_steps": 20,
        "invalid_settle_penalty": -1.0,
        "max_resample_retries": 20,
        "adversary_update_batch_size": None,
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
        entropy_coef=0.05,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-4,
        schedule="fixed",
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    adversary_robot_parameters = ADVERSARY_ADVANCED_ACTION_DIM
