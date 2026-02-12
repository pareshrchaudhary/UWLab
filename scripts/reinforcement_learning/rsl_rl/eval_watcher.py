# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Eval script that processes all checkpoints from a finished training run.

This script evaluates all checkpoints in a completed training run and logs
metrics to the same W&B run (resumed).

Usage:
    /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/eval_watcher.py \
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
        --log_dir logs/rsl_rl/experiment/2026-02-11_14-10-15 \
        --device cuda:0 \
        --headless \
        env.scene.insertive_object=fbleg \
        env.scene.receptive_object=fbtabletop
"""

import argparse
import glob
import os
import re
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate checkpoints from a finished training run.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments for eval.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--steps", type=int, default=None, help="Steps per eval rollout (default: 1 episode length).")
parser.add_argument(
    "--log_dir",
    type=str,
    required=True,
    help="Path to the finished training run's log directory.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import yaml

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedRLEnv, ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from uwlab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

# Import eval variants setup
from uwlab_tasks.manager_based.manipulation.reset_states.config.ur5e_robotiq_2f85.eval_cfg import (
    NUM_EVAL_VARIANTS,
    apply_multi_eval_events,
)

# Try to import wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("[EvalWatcher] wandb not installed, metrics will not be logged")


def get_checkpoints(log_dir):
    """Get all checkpoint files sorted by iteration number.
    
    Returns:
        List of (iteration, checkpoint_path) tuples sorted by iteration.
    """
    pattern = os.path.join(log_dir, "model_*.pt")
    checkpoint_files = glob.glob(pattern)
    
    checkpoints = []
    for path in checkpoint_files:
        basename = os.path.basename(path)
        match = re.match(r"model_(\d+)\.pt", basename)
        if match:
            iteration = int(match.group(1))
            checkpoints.append((iteration, path))
    
    # Sort by iteration
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def find_wandb_run_id(log_dir, project=None):
    """Find W&B run ID by querying W&B API using run name.
    
    Args:
        log_dir: The training log directory.
        project: The W&B project name.
        
    Returns:
        W&B run ID string, or None if not found.
    """
    run_name = os.path.basename(log_dir)
    print(f"[EvalWatcher] Looking for W&B run with name: {run_name}")
    
    if not HAS_WANDB:
        print("[EvalWatcher] wandb not installed")
        return None
    
    if not project:
        print("[EvalWatcher] No W&B project specified")
        return None
    
    # Query W&B API to find run by name
    try:
        entity = os.environ.get("WANDB_USERNAME") or os.environ.get("WANDB_ENTITY")
        api = wandb.Api()
        
        path = f"{entity}/{project}" if entity else project
        print(f"[EvalWatcher] Querying W&B API: {path}")
        
        runs = api.runs(path, filters={"display_name": run_name})
        runs_list = list(runs)
        
        if runs_list:
            run = runs_list[0]
            print(f"[EvalWatcher] Found W&B run: {run.name} -> ID: {run.id}")
            return run.id
        else:
            print(f"[EvalWatcher] No W&B run found with name '{run_name}'")
    except Exception as e:
        print(f"[EvalWatcher] W&B API query failed: {e}")
    
    return None


def run_eval_rollout(env, policy, policy_nn, num_steps, device, num_variants, envs_per_variant):
    """Run evaluation rollout and collect per-variant statistics.
    
    Variants use contiguous env blocks: variant_id = env_id // envs_per_variant
    
    Args:
        env: The environment.
        policy: The policy function.
        policy_nn: The policy neural network (for hidden states).
        num_steps: Number of steps to run.
        device: The device to use.
        num_variants: Number of eval variants.
        envs_per_variant: Number of envs per variant.
        
    Returns:
        Dictionary of evaluation statistics per variant.
    """
    obs, _ = env.reset()
    
    # Track episode rewards per env
    episode_rewards = torch.zeros(env.num_envs, device=device)
    
    # Track per-variant statistics (using contiguous blocks like play_cage.py)
    variant_ep_count = torch.zeros(num_variants, device=device)
    variant_rew_sum = torch.zeros(num_variants, device=device)
    variant_rew_min = torch.full((num_variants,), float('inf'), device=device)
    variant_rew_max = torch.full((num_variants,), float('-inf'), device=device)
    
    # Reset hidden states if recurrent
    if hasattr(policy_nn, "reset"):
        policy_nn.reset()
    
    for step in range(num_steps):
        with torch.inference_mode():
            actions = policy(obs)
        
        # RslRlVecEnvWrapper returns (obs, rewards, dones, extras)
        obs, rewards, dones, extras = env.step(actions)
        
        episode_rewards += rewards
        
        # Collect completed episodes by variant (contiguous blocks)
        done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        
        for eid in done_ids.tolist():
            variant_id = eid // envs_per_variant
            if variant_id < num_variants:  # Safety check
                ep_reward = episode_rewards[eid]
                variant_rew_sum[variant_id] += ep_reward
                variant_ep_count[variant_id] += 1
                variant_rew_min[variant_id] = torch.min(variant_rew_min[variant_id], ep_reward)
                variant_rew_max[variant_id] = torch.max(variant_rew_max[variant_id], ep_reward)
        
        episode_rewards[done_ids] = 0
        
        # Reset hidden states for done envs
        if hasattr(policy_nn, "reset"):
            policy_nn.reset(dones)
    
    # Compute per-variant statistics
    stats = {}
    for variant_id in range(num_variants):
        if variant_ep_count[variant_id] > 0:
            mean_rew = (variant_rew_sum[variant_id] / variant_ep_count[variant_id]).item()
            stats[f"eval/variant_{variant_id}/mean_reward"] = mean_rew
            stats[f"eval/variant_{variant_id}/min_reward"] = variant_rew_min[variant_id].item()
            stats[f"eval/variant_{variant_id}/max_reward"] = variant_rew_max[variant_id].item()
    
    return stats


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run evaluation on all checkpoints."""
    # Update config
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    
    # Validate log directory
    log_dir = os.path.abspath(args_cli.log_dir)
    if not os.path.exists(log_dir):
        print(f"[EvalWatcher] Error: Log directory does not exist: {log_dir}")
        return
    
    print(f"[EvalWatcher] Processing checkpoints in: {log_dir}")
    
    # Get all checkpoints
    checkpoints = get_checkpoints(log_dir)
    if not checkpoints:
        print(f"[EvalWatcher] No checkpoints found in: {log_dir}")
        return
    
    print(f"[EvalWatcher] Found {len(checkpoints)} checkpoint(s)")
    for iteration, path in checkpoints:
        print(f"  - model_{iteration}.pt")
    
    # Get W&B project and find run ID
    wandb_project = getattr(agent_cfg, "wandb_project", None)
    wandb_run_id = None
    wandb_initialized = False
    
    if wandb_project:
        wandb_run_id = find_wandb_run_id(log_dir, project=wandb_project)
    
    # Initialize W&B (resume the training run)
    if HAS_WANDB and wandb_project and wandb_run_id:
        try:
            wandb.init(
                project=wandb_project,
                id=wandb_run_id,
                resume="allow",
            )
            wandb_initialized = True
            print(f"[EvalWatcher] Resumed W&B run: {wandb_run_id}")
            
            # Define custom x-axis for eval metrics
            wandb.define_metric("eval_iteration")
            wandb.define_metric("eval/*", step_metric="eval_iteration")
        except Exception as e:
            print(f"[EvalWatcher] Failed to init W&B: {e}")
    elif HAS_WANDB and wandb_project:
        print("[EvalWatcher] Could not find W&B run ID, metrics will not be logged")
    
    # Set up environment config
    ENVS_PER_VARIANT = 100
    num_envs = args_cli.num_envs
    if num_envs is None or num_envs <= 1:
        num_envs = NUM_EVAL_VARIANTS * ENVS_PER_VARIANT
    env_cfg.scene.num_envs = num_envs
    envs_per_variant = num_envs // NUM_EVAL_VARIANTS
    
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.log_dir = log_dir
    
    # Calculate eval steps (1 episode length)
    eval_steps = args_cli.steps
    if eval_steps is None:
        episode_length_s = getattr(env_cfg, "episode_length_s", 16.0)
        sim_dt = getattr(env_cfg.sim, "dt", 1/120.0)
        decimation = getattr(env_cfg, "decimation", 12)
        step_dt = sim_dt * decimation
        eval_steps = int(episode_length_s / step_dt)
    
    print(f"[EvalWatcher] Using num_envs={num_envs} ({NUM_EVAL_VARIANTS} variants, {envs_per_variant} per variant)")
    print(f"[EvalWatcher] eval_steps={eval_steps}")
    
    # Apply multi-variant eval events (different OSC gains + reset datasets per variant)
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        print(f"[EvalWatcher] Applying multi-eval events...")
        apply_multi_eval_events(env_cfg, variants_cli=[], include_per_env_reset=True)
        print(f"[EvalWatcher] Multi-eval events applied.")
    
    print(f"[EvalWatcher] Creating environment on {env_cfg.sim.device}...")
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    
    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    print(f"[EvalWatcher] Environment created. Processing {len(checkpoints)} checkpoint(s)...")
    
    # Create runner for loading checkpoints
    device = env_cfg.sim.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    
    # Process each checkpoint
    for iteration, checkpoint_path in checkpoints:
        print(f"\n[EvalWatcher] Evaluating checkpoint: model_{iteration}.pt")
        
        # Load checkpoint
        try:
            runner.load(checkpoint_path, load_optimizer=False)
        except Exception as e:
            print(f"[EvalWatcher] Failed to load checkpoint: {e}")
            continue
        
        # Get policy
        policy = runner.get_inference_policy(device=device)
        
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
        
        # Run evaluation
        print(f"[EvalWatcher] Running eval ({eval_steps} steps)...")
        stats = run_eval_rollout(env, policy, policy_nn, eval_steps, device, NUM_EVAL_VARIANTS, envs_per_variant)
        
        # Print stats
        print(f"[EvalWatcher] Iteration {iteration}:")
        print(f"  eval_iteration: {iteration}")
        for key, value in stats.items():
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Log to W&B
        if wandb_initialized:
            try:
                log_data = {"eval_iteration": iteration}
                log_data.update(stats)
                wandb.log(log_data, commit=True)
                print(f"[EvalWatcher] Logged to W&B (eval_iteration={iteration})")
            except Exception as e:
                print(f"[EvalWatcher] Failed to log to W&B: {e}")
    
    # Cleanup
    print("\n[EvalWatcher] Finished processing all checkpoints.")
    
    try:
        env.close()
    except Exception as e:
        print(f"[EvalWatcher] Warning: Error closing env: {e}")
    
    if wandb_initialized:
        try:
            wandb.finish()
            print("[EvalWatcher] W&B run finished.")
        except Exception as e:
            print(f"[EvalWatcher] Warning: Error finishing W&B: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[EvalWatcher] Error: {e}")
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
