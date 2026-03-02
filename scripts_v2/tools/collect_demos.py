# Copyright (c) 2024-2025, The UW Lab Project Developers.
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

"""Script to collect demonstrations from a trained RL policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import os
import gymnasium as gym
import torch
from tqdm import tqdm

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations from trained RL policy.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.zarr", help="Output dataset path.")
parser.add_argument("--num_demos", type=int, default=10, help="Number of demonstrations to record.")
parser.add_argument("--experts_path", type=str, default=None, help="Path to the expert policy checkpoint.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.envs import (
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg
)

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401

# Import dataset handlers
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab.managers.recorder_manager import DatasetExportMode

from uwlab.utils.datasets import ZarrDatasetFileHandler
from uwlab_tasks.utils.hydra import hydra_task_compose
from uwlab_tasks.manager_based.manipulation.reset_states.mdp.recorders.recorders_cfg import (
    ActionStateRecorderManagerCfg,
)

from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class ExpertPolicyWrapper:
    """Wrapper to load and run a saved RL policy checkpoint."""

    def __init__(self, checkpoint_path: str, device: torch.device):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]

        # Get actor dimensions from state dict
        # actor.0.weight shape: [hidden_dim_0, input_dim]
        # actor.2.weight shape: [hidden_dim_1, hidden_dim_0]
        # ... until the last layer which outputs num_actions
        actor_input_dim = state_dict["actor.0.weight"].shape[1]

        # Collect all linear layer weight shapes
        actor_layer_dims = []
        idx = 0
        while f"actor.{idx}.weight" in state_dict:
            actor_layer_dims.append(state_dict[f"actor.{idx}.weight"].shape[0])
            idx += 2  # Skip activation layers

        # Last dim is num_actions, rest are hidden dims
        num_actions = actor_layer_dims[-1]
        actor_hidden_dims = actor_layer_dims[:-1]  # Exclude output layer

        # Get critic input dim (may differ from actor)
        critic_input_dim = state_dict["critic.0.weight"].shape[1]

        # Build critic hidden dims similarly
        critic_layer_dims = []
        idx = 0
        while f"critic.{idx}.weight" in state_dict:
            critic_layer_dims.append(state_dict[f"critic.{idx}.weight"].shape[0])
            idx += 2
        critic_hidden_dims = critic_layer_dims[:-1]  # Exclude output layer (which is 1)

        # Build obs spec with correct dimensions
        obs = TensorDict(
            {"policy": torch.zeros(1, actor_input_dim), "critic": torch.zeros(1, critic_input_dim)},
            batch_size=[1]
        )
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}

        # Create model
        self.policy = ActorCritic(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_obs_normalization="actor_obs_normalizer._mean" in state_dict,
        ).to(device)

        # Load state dict
        self.policy.load_state_dict(state_dict, strict=False)
        self.policy.eval()

        self.device = device
        self.num_actions = num_actions

        # Get noise std type and values
        if "std" in state_dict:
            # scalar std: shape [num_actions]
            self.std = state_dict["std"].to(device)
        elif "log_std" in state_dict:
            log_std = state_dict["log_std"]
            if log_std.dim() == 2:
                # gSDE: log_std shape is [last_hidden_dim, num_actions]
                # Compute per-action std as the norm across the hidden dim
                self.std = torch.exp(log_std).norm(dim=0).to(device)
            else:
                # standard: log_std shape is [num_actions]
                self.std = torch.exp(log_std).to(device)
        else:
            self.std = torch.ones(num_actions, device=device)

    def compute_distribution(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std for the action distribution."""
        with torch.no_grad():
            # obs may be a TensorDict or tensor - handle both
            if isinstance(obs, TensorDict):
                # If it's already a TensorDict, use directly
                # But we need to ensure it has the right keys for our model
                if "policy" not in obs.keys():
                    # Assume obs contains the actor obs directly - wrap it
                    actor_obs = self.policy.get_actor_obs(obs)
                    obs_dict = TensorDict({"policy": actor_obs, "critic": actor_obs}, batch_size=obs.shape[0])
                else:
                    obs_dict = obs
            else:
                # obs is a tensor - wrap it
                obs_dict = TensorDict({"policy": obs, "critic": obs}, batch_size=[obs.shape[0]])

            mean = self.policy.act_inference(obs_dict)
            std = self.std.expand_as(mean)
            return mean, std

    def to(self, device: torch.device):
        self.policy = self.policy.to(device)
        self.std = self.std.to(device)
        self.device = device
        return self

    def eval(self):
        self.policy.eval()
        return self


def load_rl_expert(checkpoint_path: str) -> ExpertPolicyWrapper:
    """Load an RL expert from a checkpoint file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ExpertPolicyWrapper(checkpoint_path, device)


def process_agent_cfg(env_cfg, agent_cfg):
    if hasattr(agent_cfg.algorithm, "behavior_cloning_cfg"):
        if agent_cfg.algorithm.behavior_cloning_cfg is None:
            del agent_cfg.algorithm.behavior_cloning_cfg
        else:
            bc_cfg = agent_cfg.algorithm.behavior_cloning_cfg
            if bc_cfg.experts_observation_group_cfg is not None:
                import importlib

                # resolve path to the module location
                mod_name, attr_name = bc_cfg.experts_observation_group_cfg.split(":")
                mod = importlib.import_module(mod_name)
                cfg_cls = mod
                for attr in attr_name.split("."):
                    cfg_cls = getattr(cfg_cls, attr)
                cfg = cfg_cls()
                setattr(env_cfg.observations, "expert_obs", cfg)

    if hasattr(agent_cfg.algorithm, "offline_algorithm_cfg"):
        if agent_cfg.algorithm.offline_algorithm_cfg is None:
            del agent_cfg.algorithm.offline_algorithm_cfg
        else:
            if agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg is None:
                del agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg
            else:
                bc_cfg = agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg
                if bc_cfg.experts_observation_group_cfg is not None:
                    import importlib

                    # resolve path to the module location
                    mod_name, attr_name = bc_cfg.experts_observation_group_cfg.split(":")
                    mod = importlib.import_module(mod_name)
                    cfg_cls = mod
                    for attr in attr_name.split("."):
                        cfg_cls = getattr(cfg_cls, attr)
                    cfg = cfg_cls()
                    setattr(env_cfg.observations, "expert_obs", cfg)
    return agent_cfg


@hydra_task_compose(args_cli.task, "rsl_rl_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Collect demonstrations from the environment using RSL-RL policy."""
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.basename(args_cli.dataset_file)

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # add recordermanager to save data
    use_zarr_format = args_cli.dataset_file.endswith('.zarr')
    if use_zarr_format:
        dataset_handler = ZarrDatasetFileHandler
    else:
        dataset_handler = HDF5DatasetFileHandler

    # Setup recorder for raw actions
    env_cfg.recorders = ActionStateRecorderManagerCfg()

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = dataset_handler

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = None

    # add expert obs into env_cfg
    agent_cfg = process_agent_cfg(env_cfg, agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load expert
    bc = agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg
    assert len(bc.experts_path) == 1, "Only one expert is supported for now."
    if args_cli.experts_path is not None:
        bc.experts_path[0] = args_cli.experts_path
    assert bc.experts_path[0], "experts_path is not set. Pass --experts_path <path> or set it in the config."
    expert_obs_fn = bc.experts_observation_func

    # Use custom loader that handles native RL checkpoints (not TorchScript)
    expert_policy = load_rl_expert(bc.experts_path[0]).to(env_cfg.sim.device)
    expert_policy.eval()

    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        # Initialize tqdm progress bar if num_demos > 0
        pbar = tqdm(total=args_cli.num_demos, desc="Recording Demonstrations", unit="demo")

        while True:
            # agent stepping
            expert_policy_obs = expert_obs_fn(env)
            # actions = expert_policy(expert_policy_obs)
            mean, std = expert_policy.compute_distribution(expert_policy_obs)
            actions = torch.normal(mean, std)

            # Mask actions to zero for environments in their first step after reset since first image may not be valid
            first_step_mask = (env.unwrapped.episode_length_buf == 0)
            if torch.any(first_step_mask):
                actions[first_step_mask, :-1] = 0.0
                actions[first_step_mask, -1] = -1.0  # close gripper

            # env stepping
            env.step(actions)

            # print out the current demo count if it has changed
            new_count = env.unwrapped.recorder_manager.exported_successful_episode_count
            if new_count > current_recorded_demo_count:
                increment = new_count - current_recorded_demo_count
                current_recorded_demo_count = new_count
                pbar.update(increment)

            if args_cli.num_demos > 0 and new_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break

        pbar.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function - the decorator handles parameter passing
    main()  # type: ignore
    # close sim app
    simulation_app.close()
