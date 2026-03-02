# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a trained diffusion policy."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play policy trained using diffusion policy for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to diffusion policy checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel.")
parser.add_argument("--num_trajectories", type=int, default=100, help="Number of trajectories to evaluate. If None, run until simulation is stopped.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--use_amp", action="store_true", default=False, help="Use automatic mixed precision.")
parser.add_argument("--save_video", action="store_true", default=False, help="Save video of the policy.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, remaining_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import dill
import hydra
from contextlib import nullcontext
from tqdm import tqdm
import random
import numpy as np

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from uwlab_tasks.utils.hydra import hydra_task_compose

# Diffusion policy imports
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# Import the Diffusion policy wrapper
from uwlab_apps.utils.wrappers.diffusion import DiffusionPolicyWrapper

import imageio
import matplotlib.pyplot as plt


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    """Run a trained diffusion policy with Isaac Lab environment."""
    # Set seeds for reproducibility
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check device is available
    device = torch.device(args_cli.device if args_cli.device else 'cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Override configurations with CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    # Set environment seed
    env_cfg.seed = args_cli.seed
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # Load diffusion policy checkpoint
    ckpt_path = args_cli.checkpoint
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Load policy based on configuration
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.eval().to(device)

    # Wrap policy to handle Isaac Lab observations
    wrapped_policy = DiffusionPolicyWrapper(policy, device, n_obs_steps=policy.n_obs_steps, num_envs=args_cli.num_envs)

    # reset environment
    obs_dict, _ = env.reset()
    dones = torch.ones(args_cli.num_envs, dtype=torch.bool, device=device)
    reset_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
    wrapped_policy.reset(reset_ids)

    # Get termination term names to identify success
    term_names = env.unwrapped.termination_manager._term_names  # type: ignore
    assert "success" in term_names, "Success term not found in termination manager"

    episodes = 0
    steps = 0
    successful_episodes = 0  # Track successful episodes

    # Track all episode metrics
    episode_metrics = {}

    # Initialize progress bar if num_trajectories is specified
    pbar = None
    if args_cli.num_trajectories is not None:
        pbar = tqdm(total=args_cli.num_trajectories, desc="Evaluating trajectories (Success: 0.00%)")

    # simulate environment
    if args_cli.save_video:
        env_frames = [[] for _ in range(args_cli.num_envs)]
        frames_to_save = []
        # Try to get camera keys from observations first, fallback to scene sensors
        cam_keys = sorted([key for key in obs_dict['policy'].keys() if 'rgb' in key])
        # If no rgb obs, try to get cameras from scene
        scene_cam_names = []
        if not cam_keys:
            for name, sensor in env.unwrapped.scene._sensors.items():
                if hasattr(sensor, 'data') and hasattr(sensor.data, 'output'):
                    if 'rgb' in sensor.data.output:
                        scene_cam_names.append(name)
            scene_cam_names = sorted(scene_cam_names)
            print(f"Using scene cameras for video: {scene_cam_names}")
    while simulation_app.is_running():
        # Check if we've reached the desired number of trajectories
        if args_cli.num_trajectories is not None and episodes >= args_cli.num_trajectories:
            if pbar is not None:
                pbar.close()
            print(f"\nReached target number of trajectories ({args_cli.num_trajectories}). Stopping evaluation.")
            break

        # run everything in inference mode
        with torch.inference_mode(), torch.autocast(device_type=device.type) if args_cli.use_amp else nullcontext():
            # compute actions using wrapped diffusion policy
            actions = wrapped_policy.predict_action(obs_dict)

            if args_cli.save_video:
                for i in range(args_cli.num_envs):
                    imgs = []
                    # Get images from observations if available
                    if cam_keys:
                        for cam in cam_keys:
                            img = obs_dict['policy'][cam][i].detach().cpu().permute(1, 2, 0).numpy()
                            img = (img * 255).clip(0, 255).astype('uint8')
                            imgs.append(img)
                    # Otherwise get from scene sensors directly
                    elif scene_cam_names:
                        for cam_name in scene_cam_names:
                            sensor = env.unwrapped.scene._sensors[cam_name]
                            img = sensor.data.output['rgb'][i].detach().cpu().numpy()
                            # Handle different image formats (CHW vs HWC)
                            if img.shape[0] in [1, 3, 4] and img.shape[0] < img.shape[1]:
                                img = img.transpose(1, 2, 0)
                            if img.dtype != np.uint8:
                                img = (img * 255).clip(0, 255).astype('uint8')
                            # Keep only RGB channels if RGBA
                            if img.shape[-1] == 4:
                                img = img[..., :3]
                            imgs.append(img)
                    if imgs:
                        frame = np.concatenate(imgs, axis=1)
                        env_frames[i].append(frame)

            # apply actions using environment
            step_result = env.step(actions)
            if len(step_result) == 4:
                obs_dict, rewards, dones, infos = step_result
            else:
                # Handle gymnasium v0.26+ format with 5 return values
                obs_dict, rewards, terminated, truncated, infos = step_result
                dones = terminated | truncated

            steps += 1

            # Clear data for completed episodes
            new_ids = []
            if isinstance(dones, torch.Tensor):
                new_ids = (dones > 0).nonzero(as_tuple=False)
                episodes += len(new_ids)
            else:
                # Handle scalar done value
                if dones:
                    episodes += 1
                    new_ids = [0]  # Single episode done

            if isinstance(dones, torch.Tensor) and dones.any():
                reset_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
                num_new_successes = 0

                term_dones = env.unwrapped.termination_manager._term_dones[reset_ids]  # type: ignore
                for env_idx, term_row in enumerate(term_dones):
                    active_term_idx = term_row.nonzero(as_tuple=False)
                    if active_term_idx.numel() > 0:
                        # Handle multiple active termination conditions
                        active_term_indices = active_term_idx.flatten().cpu().tolist()
                        for term_idx in active_term_indices:
                            if term_names[term_idx] == "success":
                                num_new_successes += 1
                                break  # Count each environment only once

                successful_episodes += num_new_successes

                wrapped_policy.reset(reset_ids)

                # Store metrics for completed episodes
                if "log" in infos:
                    # Store all metrics from this episode
                    for key, value in infos["log"].items():
                        if key.startswith("Metrics/") or key.startswith("Episode_Reward/"):
                            if key not in episode_metrics:
                                episode_metrics[key] = []
                            episode_metrics[key].append(value)

                steps = 0

                if args_cli.save_video:
                    for i in reset_ids:
                        frames_to_save.extend(env_frames[i])
                        env_frames[i] = []
                    imageio.mimsave("policy_cameras.mp4", frames_to_save, fps=10, codec='libx264')

                # Update progress bar with success rate
                if pbar is not None:
                    pbar.update(len(new_ids))
                    success_rate = (successful_episodes / episodes * 100) if episodes > 0 else 0.0
                    pbar.set_description(f"Evaluating trajectories (Success: {success_rate:.2f}%)")

    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Total trajectories evaluated: {episodes}")
    if successful_episodes > 0 or "Episode_Termination/success" in episode_metrics:
        print(f"Successful trajectories: {successful_episodes}")
        print(f"Success rate: {successful_episodes/episodes*100:.2f}%")
    else:
        print("Success rate: Not calculable (success metric not found in environment)")

    # Print metrics statistics
    if episode_metrics:
        print("\nAverage Metrics:")
        for metric_name, values in sorted(episode_metrics.items()):
            if values:  # Only print if we have values
                values = [float(v) if isinstance(v, torch.Tensor) else v for v in values]
                mean = sum(values) / len(values)
                print(f"{metric_name}: {mean:.4f}")

    # Cleanup
    if pbar is not None:
        pbar.close()
    env.close()


if __name__ == "__main__":
    # run the main function - the decorator handles parameter passing
    main()  # type: ignore
    # close sim app
    simulation_app.close()
