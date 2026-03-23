# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Eval script that processes all checkpoints from a finished training run.

Usage:
    /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/eval.py \
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
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate checkpoints from a finished training run.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments for eval.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--steps", type=int, default=None, help="Steps per eval rollout (default: 1 episode length).")
parser.add_argument("--log_dir", type=str, required=True, help="Path to the training run's log directory.")
parser.add_argument("--new_eval_run", action="store_true", default=False,
                    help="Create a new eval run directory instead of resuming the original.")
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

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedRLEnv, ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab.utils.io import dump_yaml
from uwlab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

# Import eval variants setup
from uwlab_tasks.manager_based.manipulation.assembly_task.omnireset.mdp.eval import (
    NUM_EVAL_VARIANTS,
    apply_multi_eval_events,
)

import wandb

def get_checkpoints(log_dir):
    """Return (iteration, path) tuples sorted by iteration."""
    checkpoints = []
    for path in glob.glob(os.path.join(log_dir, "model_*.pt")):
        match = re.match(r"model_(\d+)\.pt", os.path.basename(path))
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def find_wandb_run_id(log_dir, project):
    """Find W&B run ID by run name via the API."""
    if not project:
        return None
    run_name = os.path.basename(log_dir)
    try:
        entity = os.environ.get("WANDB_USERNAME") or os.environ.get("WANDB_ENTITY")
        path = f"{entity}/{project}" if entity else project
        runs = list(wandb.Api().runs(path, filters={"display_name": run_name}))
        if runs:
            return runs[0].id
    except Exception as e:
        print(f"[Eval] W&B API query failed: {e}")
    return None


def run_eval_rollout(env, policy, policy_nn, num_steps, device, num_variants, envs_per_variant):
    """Run evaluation rollout and return per-variant reward statistics."""
    obs, _ = env.reset()

    episode_rewards = torch.zeros(env.num_envs, device=device)
    variant_ep_count = torch.zeros(num_variants, device=device)
    variant_rew_sum = torch.zeros(num_variants, device=device)

    if hasattr(policy_nn, "reset"):
        policy_nn.reset()

    for _ in range(num_steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, rewards, dones, _ = env.step(actions)
        episode_rewards += rewards

        done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        for eid in done_ids.tolist():
            vid = eid // envs_per_variant
            if vid < num_variants:
                ep_rew = episode_rewards[eid]
                variant_rew_sum[vid] += ep_rew
                variant_ep_count[vid] += 1
        episode_rewards[done_ids] = 0

        if hasattr(policy_nn, "reset"):
            policy_nn.reset(dones)

    stats = {}
    for vid in range(num_variants):
        if variant_ep_count[vid] > 0:
            stats[f"eval/variant_{vid}/mean_reward"] = (variant_rew_sum[vid] / variant_ep_count[vid]).item()
    return stats


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run evaluation on all checkpoints."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    log_dir = os.path.abspath(args_cli.log_dir)
    if not os.path.exists(log_dir):
        print(f"[Eval] Log directory not found: {log_dir}")
        return

    # Resolve eval logging directory
    if args_cli.new_eval_run:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            run_name += f"_{agent_cfg.run_name}"
        eval_log_dir = os.path.join(os.path.dirname(log_dir), run_name)
        os.makedirs(eval_log_dir, exist_ok=False)
        print(f"[Eval] New eval run: {eval_log_dir}")
    else:
        eval_log_dir = log_dir

    checkpoints = get_checkpoints(log_dir)
    if not checkpoints:
        print(f"[Eval] No checkpoints found in: {log_dir}")
        return
    print(f"[Eval] Found {len(checkpoints)} checkpoint(s): {[f'model_{i}.pt' for i, _ in checkpoints]}")

    # Initialize W&B
    wandb_initialized = False
    wandb_project = getattr(agent_cfg, "wandb_project", None)
    if wandb_project:
        try:
            if args_cli.new_eval_run:
                wandb.init(project=wandb_project, name=os.path.basename(eval_log_dir),
                           resume="never", dir=eval_log_dir)
            else:
                run_id = find_wandb_run_id(log_dir, project=wandb_project)
                if run_id:
                    wandb.init(project=wandb_project, id=run_id, resume="allow")
                else:
                    print("[Eval] Could not find W&B run ID, metrics will not be logged")
            if wandb.run:
                wandb.define_metric("eval_iteration")
                wandb.define_metric("eval/*", step_metric="eval_iteration")
                wandb_initialized = True
        except Exception as e:
            print(f"[Eval] Failed to init W&B: {e}")

    # Configure environment
    ENVS_PER_VARIANT = 100
    num_envs = args_cli.num_envs if (args_cli.num_envs and args_cli.num_envs > 1) else NUM_EVAL_VARIANTS * ENVS_PER_VARIANT
    envs_per_variant = num_envs // NUM_EVAL_VARIANTS
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.log_dir = eval_log_dir

    if args_cli.steps is not None:
        eval_steps = args_cli.steps
    else:
        episode_length_s = getattr(env_cfg, "episode_length_s", 16.0)
        step_dt = getattr(env_cfg.sim, "dt", 1/120.0) * getattr(env_cfg, "decimation", 12)
        eval_steps = int(episode_length_s / step_dt)

    print(f"[Eval] num_envs={num_envs} ({NUM_EVAL_VARIANTS} variants x {envs_per_variant}), eval_steps={eval_steps}")

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        apply_multi_eval_events(env_cfg, variants_cli=[], include_per_env_reset=True)

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg, render_mode=None))
    device = env_cfg.sim.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(),
                            log_dir=eval_log_dir if args_cli.new_eval_run else None, device=device)

    if args_cli.new_eval_run:
        dump_yaml(os.path.join(eval_log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(eval_log_dir, "params", "agent.yaml"), agent_cfg)

    for iteration, checkpoint_path in checkpoints:
        print(f"\n[Eval] Evaluating model_{iteration}.pt ({eval_steps} steps)...")
        try:
            runner.load(checkpoint_path, load_optimizer=False)
        except Exception as e:
            print(f"[Eval] Failed to load checkpoint: {e}")
            continue

        policy = runner.get_inference_policy(device=device)
        policy_nn = runner.alg.policy

        stats = run_eval_rollout(env, policy, policy_nn, eval_steps, device, NUM_EVAL_VARIANTS, envs_per_variant)

        print(f"[Eval] Iteration {iteration}:")
        for vid in range(NUM_EVAL_VARIANTS):
            key = f"eval/variant_{vid}/mean_reward"
            if key in stats:
                print(f"  variant {vid}: mean_reward = {stats[key]:.3f}")
            else:
                print(f"  variant {vid}: mean_reward = (no completed episodes)")

        if wandb_initialized:
            try:
                wandb.log({"eval_iteration": iteration, **stats}, commit=True)
            except Exception as e:
                print(f"[Eval] Failed to log to W&B: {e}")

    print("\n[Eval] Done.")
    env.close()
    if wandb_initialized:
        wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Eval] Error: {e}")
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
