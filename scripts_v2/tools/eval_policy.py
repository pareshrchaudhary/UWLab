# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained RSL-RL state policy checkpoint."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "reinforcement_learning", "rsl_rl"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL state policy for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to run in parallel.")
parser.add_argument(
    "--num_trajectories",
    type=int,
    default=100,
    help="Number of trajectories to evaluate. If None, run until simulation is stopped.",
)
parser.add_argument("--action_noise", type=float, default=2.0, help="Std of Gaussian noise added to actions.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--save_video", action="store_true", default=False, help="Save video of the policy.")
parser.add_argument(
    "--eval_summary_log",
    type=str,
    default=None,
    help="If set, append task name, final statistics, and average metrics to this file.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.checkpoint is None:
    parser.error("--checkpoint is required")
if args_cli.save_video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import imageio
import numpy as np
import random
import torch
from tqdm import tqdm

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.cage import mdp as cage_mdp
from uwlab_tasks.manager_based.manipulation.omnireset import mdp as omnireset_mdp
from uwlab_tasks.utils.hydra import hydra_task_config


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _discover_cameras(obs, env):
    """Return (cam_keys, scene_cam_names) for video recording."""
    cam_keys = []
    if isinstance(obs, dict) and "policy" in obs and isinstance(obs["policy"], dict):
        cam_keys = sorted(k for k in obs["policy"] if "rgb" in k)
        if cam_keys:
            return cam_keys, []

    scene_cam_names = sorted(
        name
        for name, sensor in env.unwrapped.scene._sensors.items()
        if hasattr(sensor, "data") and hasattr(sensor.data, "output") and "rgb" in sensor.data.output
    )
    if scene_cam_names:
        print(f"Using scene cameras for video: {scene_cam_names}")
    return cam_keys, scene_cam_names


def _capture_frame(obs, env, env_idx: int, cam_keys: list, scene_cam_names: list) -> np.ndarray | None:
    """Capture and concatenate camera images for one environment."""
    imgs = []
    if isinstance(obs, dict) and "policy" in obs and cam_keys:
        for cam in cam_keys:
            img = obs["policy"][cam][env_idx].detach().cpu().permute(1, 2, 0).numpy()
            imgs.append((img * 255).clip(0, 255).astype("uint8"))
    elif scene_cam_names:
        for cam_name in scene_cam_names:
            img = env.unwrapped.scene._sensors[cam_name].data.output["rgb"][env_idx].detach().cpu().numpy()
            if img.shape[0] in [1, 3, 4] and img.shape[0] < img.shape[1]:
                img = img.transpose(1, 2, 0)
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype("uint8")
            if img.shape[-1] == 4:
                img = img[..., :3]
            imgs.append(img)
    return np.concatenate(imgs, axis=1) if imgs else None


def _count_successes(env, reset_ids: torch.Tensor, term_names: list[str]) -> int:
    success_idx = term_names.index("success")
    count = 0
    term_dones = env.unwrapped.termination_manager._term_dones[reset_ids]
    for term_row in term_dones:
        active = term_row.nonzero(as_tuple=False).flatten().cpu().tolist()
        if success_idx in active:
            count += 1
    return count


def _collect_metrics(infos: dict, episode_metrics: dict):
    if "log" not in infos:
        return
    for key, value in infos["log"].items():
        if key.startswith("Metrics/") or key.startswith("Episode_Reward/"):
            episode_metrics.setdefault(key, []).append(value)


def _metric_to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        t = value.detach().float().cpu()
        # end_of_episode_* metrics are only written for just-reset envs; all
        # other env slots remain 0. Average only the non-zero entries so the
        # result is not diluted across idle envs.
        nonzero = t[t != 0]
        return float(nonzero.mean()) if len(nonzero) > 0 else 0.0
    return float(value)


def _print_results(
    task: str,
    episodes: int,
    successful_episodes: int,
    episode_metrics: dict,
    has_success_term: bool,
    log_path: str | None = None,
):
    lines: list[str] = ["", f"Task: {task}", "Final Statistics:"]
    lines.append(f"Total trajectories evaluated: {episodes}")
    if not has_success_term:
        lines.append("Success rate: N/A (no 'success' termination term in this task config)")
    elif episodes > 0:
        lines.append(f"Successful trajectories: {successful_episodes}")
        lines.append(f"Success rate: {successful_episodes / episodes * 100:.2f}%")
    else:
        lines.append("Success rate: Not calculable (no completed trajectories)")
    if episode_metrics:
        lines.append("")
        lines.append("Average Metrics:")
        for metric_name, values in sorted(episode_metrics.items()):
            if values:
                floats = [_metric_to_float(v) for v in values]
                lines.append(f"{metric_name}: {sum(floats) / len(floats):.4f}")

    summary = "\n".join(lines) + "\n"
    print(summary, end="")

    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(summary)


def _get_policy_nn(runner):
    try:
        return runner.alg.policy
    except AttributeError:
        return runner.alg.actor_critic


def _get_task_mdp(task_name: str):
    if task_name.startswith("OmniReset-"):
        return omnireset_mdp
    if task_name.startswith("Cage-"):
        return cage_mdp
    return None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate with an RSL-RL agent."""
    _set_seeds(args_cli.seed)

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    task_mdp = _get_task_mdp(args_cli.task)
    if task_mdp is not None:
        env_cfg.terminations.success = DoneTerm(
            func=task_mdp.consecutive_success_state_with_min_length,
            params={"num_consecutive_successes": 5, "min_episode_length": 10},
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.save_video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    term_names = env.unwrapped.termination_manager._term_names
    has_success_term = "success" in term_names
    if not has_success_term:
        print(
            f"[WARNING]: 'success' not in termination terms {term_names}. "
            "Success rate will not be reported. Add a 'success' DoneTerm to the task config "
            "or use the Play task variant that includes one."
        )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = _get_policy_nn(runner)

    obs = env.get_observations()
    dones = torch.ones(args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    policy_nn.reset(dones)

    episodes, successful_episodes = 0, 0
    episode_metrics: dict = {}

    pbar = None
    if args_cli.num_trajectories is not None:
        pbar = tqdm(total=args_cli.num_trajectories, desc="Evaluating trajectories (Success: 0.00%)")

    cam_keys, scene_cam_names, env_frames, frames_to_save = [], [], [], []
    if args_cli.save_video:
        cam_keys, scene_cam_names = _discover_cameras(obs, env)
        env_frames = [[] for _ in range(args_cli.num_envs)]

    while simulation_app.is_running():
        if args_cli.num_trajectories is not None and episodes >= args_cli.num_trajectories:
            print(f"\nReached target number of trajectories ({args_cli.num_trajectories}). Stopping evaluation.")
            break

        with torch.inference_mode():
            actions = policy(obs)
            if args_cli.action_noise > 0.0:
                actions = actions + args_cli.action_noise * torch.randn_like(actions)

            if args_cli.save_video:
                for i in range(args_cli.num_envs):
                    frame = _capture_frame(obs, env, i, cam_keys, scene_cam_names)
                    if frame is not None:
                        env_frames[i].append(frame)

            obs, _, dones, infos = env.step(actions)
            policy_nn.reset(dones)

        if dones.any():
            reset_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
            episodes += len(reset_ids)
            if has_success_term:
                successful_episodes += _count_successes(env, reset_ids, term_names)
            _collect_metrics(infos, episode_metrics)

            if args_cli.save_video:
                for env_id in reset_ids:
                    idx = env_id.item()
                    frames_to_save.extend(env_frames[idx])
                    env_frames[idx] = []
                if frames_to_save:
                    imageio.mimsave("policy_cameras.mp4", frames_to_save, fps=10, codec="libx264")

            if pbar is not None:
                pbar.update(len(reset_ids))
                rate = (successful_episodes / episodes * 100) if episodes > 0 else 0.0
                pbar.set_description(f"Evaluating trajectories (Success: {rate:.2f}%)")

    _print_results(args_cli.task, episodes, successful_episodes, episode_metrics, has_success_term, args_cli.eval_summary_log)
    if pbar is not None:
        pbar.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
