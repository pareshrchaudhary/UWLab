# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to test reset states simulation using IsaacLab framework."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import time
import torch
from typing import cast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test reset states simulation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task", type=str, default="OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0", help="Name of the task."
)
parser.add_argument("--disable_recording", action="store_true", default=True, help="Disable pose data collection.")

AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    """Main function to run reset states simulation."""

    # Remove pose data collection event if requested
    if args_cli.disable_recording and hasattr(env_cfg.events, 'pose_data_collection'):
        print("Disabling pose data collection...")
        delattr(env_cfg.events, 'pose_data_collection')

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Create environment
    print(f"Creating environment: {args_cli.task}")
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped

    # Print configuration info
    print(f"Number of environments: {env.num_envs}")
    print(f"Simulation timestep: {env_cfg.sim.dt}")
    print(f"Episode length: {env_cfg.episode_length_s}s")
    print(f"Running simulation indefinitely... Press Ctrl+C to stop.")
    print("=" * 80)

    # Reset environment
    env.reset()

    # Run simulation loop indefinitely
    actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)

    # Close gripper for grasped tasks
    if "EEGrasped" in args_cli.task:
        actions[:, -1] = -1.0

    step = 0
    episode_count = 0
    total_episodes = 0

    try:
        while True:
            # Step environment
            _, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated

            # Print object states periodically to show simulation progress
            if step % 60 == 0:  # Print every 60 steps (~0.5s)
                insertive_obj = env.scene["insertive_object"]
                pos = insertive_obj.data.root_pos_w[0]
                vel = insertive_obj.data.root_lin_vel_w[0]
                ang_vel = insertive_obj.data.root_ang_vel_w[0]
                print(f"Step {step:6d} | Insertive Object State:")
                print(f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
                print(f"  Lin Vel:  [{vel[0]:.4f}, {vel[1]:.4f}, {vel[2]:.4f}]")
                print(f"  Ang Vel:  [{ang_vel[0]:.4f}, {ang_vel[1]:.4f}, {ang_vel[2]:.4f}]")

            # Print metrics when any environment resets
            if dones.any():
                done_indices = torch.where(dones)[0]
                for idx in done_indices:
                    episode_count += 1
                    print(f"\nEpisode {episode_count:4d} | Env {idx.item():2d} | "
                          f"Reward: {rewards[idx].item():.6f}\n")
                total_episodes += len(done_indices)

            # Check if simulation should stop
            if env.sim.is_stopped():
                print("\nSimulation stopped by user.")
                break

            step += 1

            # Slow down simulation for visualization
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print(f"Simulation interrupted at step {step}")
        print(f"Total episodes completed: {total_episodes}")
        print("=" * 80)

    env.close()

    # Delay at end
    time.sleep(2.0)


if __name__ == "__main__":
    main()
    simulation_app.close()
