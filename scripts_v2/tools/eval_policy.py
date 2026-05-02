# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate trained RSL-RL state policy checkpoint(s)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import re
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
    "--action_delay_steps",
    type=int,
    default=0,
    help="Delay the action passed to env.step by N timesteps via a ring buffer. Default 0 (no delay). "
    "Models real-world communication/firmware latency between policy host and motor controller. "
    "Tier-agnostic — sits outside the env, works for both Implicit and Explicit actuator checkpoints.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
cli_args.add_rsl_rl_args(parser)
for action in parser._actions:
    if action.dest == "checkpoint":
        action.help = "Path to a checkpoint file, or a directory containing .pt/.pth checkpoints to evaluate."
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


def _collect_metrics(infos: dict, episode_metrics: dict, episode_count: int):
    if "log" not in infos:
        return
    for key, value in infos["log"].items():
        if key.startswith("Metrics/") or key.startswith("Episode_Reward/"):
            metric_stats = episode_metrics.setdefault(key, {"weighted_sum": 0.0, "count": 0})
            metric_stats["weighted_sum"] += _metric_to_float(value) * episode_count
            metric_stats["count"] += episode_count


def _metric_to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        t = value.detach().float().cpu()
        return float(t.mean()) if t.numel() > 0 else 0.0
    return float(value)


def _metric_average(metric_stats: dict) -> float:
    count = metric_stats["count"]
    return metric_stats["weighted_sum"] / count if count > 0 else 0.0


def _get_eval_type(task: str) -> str:
    if "-Eval-" in task:
        return task.split("-Eval-", 1)[1]
    if "-Eval" in task:
        return task.split("-Eval", 1)[1].lstrip("-")
    return task


def _get_eval_tag(task: str) -> str:
    eval_type = _get_eval_type(task)
    match = re.fullmatch(r"Cfg(V\d+)", eval_type)
    if match:
        return match.group(1)
    return eval_type


def _checkpoint_sort_key(path: str):
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", os.path.basename(path))
    )


def _resolve_checkpoint_paths(checkpoint_path: str) -> list[str]:
    local_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if os.path.isdir(local_path):
        checkpoint_paths = [
            os.path.join(local_path, name)
            for name in os.listdir(local_path)
            if os.path.isfile(os.path.join(local_path, name)) and os.path.splitext(name)[1].lower() in {".pt", ".pth"}
        ]
        if not checkpoint_paths:
            raise FileNotFoundError(f"No .pt or .pth checkpoints found in directory: {checkpoint_path}")
        return sorted(checkpoint_paths, key=_checkpoint_sort_key)

    return [retrieve_file_path(checkpoint_path)]


def _get_checkpoint_step(checkpoint_path: str, fallback_step: int) -> int:
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    matches = re.findall(r"\d+", checkpoint_name)
    if matches:
        return int(matches[-1])
    return fallback_step


def _get_checkpoint_run_dir(checkpoint_root: str) -> str:
    root_path = os.path.abspath(os.path.expanduser(checkpoint_root))
    if os.path.isdir(root_path):
        return root_path
    return os.path.dirname(root_path)


def _get_eval_experiment_name(checkpoint_root: str, task: str) -> str:
    run_dir = _get_checkpoint_run_dir(checkpoint_root)
    checkpoint_run = os.path.basename(os.path.normpath(run_dir))
    training_experiment = os.path.basename(os.path.normpath(os.path.dirname(run_dir)))
    if not training_experiment or training_experiment in {"logs", "rsl_rl"}:
        training_experiment = checkpoint_run
        checkpoint_run = ""
    parts = [training_experiment, checkpoint_run, _get_eval_tag(task)]
    return "_".join(part for part in parts if part)


def _get_wandb_run_name(eval_experiment_name: str) -> str:
    return eval_experiment_name


def _build_wandb_summary(
    episodes: int,
    successful_episodes: int,
    episode_metrics: dict,
    has_success_term: bool,
) -> dict:
    """W&B logs one number per checkpoint: success rate. Nothing else."""
    if has_success_term and episodes > 0:
        return {"eval/success_rate_percent": successful_episodes / episodes * 100.0}
    return {}


def _print_results(
    task: str,
    episodes: int,
    successful_episodes: int,
    episode_metrics: dict,
    has_success_term: bool,
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
            lines.append(f"{metric_name}: {_metric_average(values):.4f}")

    print("\n".join(lines) + "\n", end="")


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


def _create_eval_env_and_runner(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    log_dir: str,
):
    """Create one eval environment and runner for the whole checkpoint sweep."""
    _set_seeds(args_cli.seed)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.log_dir = log_dir

    task_mdp = _get_task_mdp(args_cli.task)
    if task_mdp is not None:
        env_cfg.terminations.success = DoneTerm(
            func=task_mdp.consecutive_success_state_with_min_length,
            params={"num_consecutive_successes": 5, "min_episode_length": 10},
        )

    print("[INFO]: Creating eval environment once for checkpoint sweep.")
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

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    return env, runner, term_names, has_success_term


def _evaluate_checkpoint(
    env: RslRlVecEnvWrapper,
    runner,
    resume_path: str,
    checkpoint_index: int,
    num_checkpoints: int,
    term_names: list[str],
    has_success_term: bool,
) -> dict:
    """Evaluate one checkpoint with an RSL-RL agent."""
    _set_seeds(args_cli.seed)

    pbar = None

    try:
        print(f"\n[INFO]: Evaluating checkpoint {checkpoint_index}/{num_checkpoints}: {resume_path}")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path, load_optimizer=False)

        policy = runner.get_inference_policy(device=env.unwrapped.device)
        policy_nn = _get_policy_nn(runner)

        env.seed(args_cli.seed)
        obs, _ = env.reset()
        dones = torch.ones(env.num_envs, dtype=torch.bool, device=env.unwrapped.device)
        policy_nn.reset(dones)

        episodes, successful_episodes = 0, 0
        episode_metrics: dict = {}

        if args_cli.num_trajectories is not None:
            desc = f"Evaluating checkpoint {checkpoint_index}/{num_checkpoints} (Success: 0.00%)"
            pbar = tqdm(total=args_cli.num_trajectories, desc=desc)

        cam_keys, scene_cam_names, env_frames, frames_to_save = [], [], [], []
        if args_cli.save_video:
            cam_keys, scene_cam_names = _discover_cameras(obs, env)
            env_frames = [[] for _ in range(args_cli.num_envs)]

        # --action_delay_steps: ring buffer of size (delay+1, num_envs, action_dim).
        # Each step writes the policy's action at head, advances head, reads the slot
        # that's about to be overwritten next step (= the action from N steps ago).
        # Allocated upfront (outside inference_mode) so per-env reset writes don't
        # trip the inference-tensor guard.
        delay_state = None
        if args_cli.action_delay_steps > 0:
            size = args_cli.action_delay_steps + 1
            delay_state = {
                "buf": torch.zeros(size, env.num_envs, env.num_actions, device=env.unwrapped.device),
                "head": 0,
                "size": size,
            }

        while simulation_app.is_running():
            if args_cli.num_trajectories is not None and episodes >= args_cli.num_trajectories:
                print(f"\nReached target number of trajectories ({args_cli.num_trajectories}). Stopping evaluation.")
                break

            with torch.inference_mode():
                actions = policy(obs)
                if args_cli.action_noise > 0.0:
                    actions = actions + args_cli.action_noise * torch.randn_like(actions)
            actions = actions.detach().clone()

            if args_cli.save_video:
                for i in range(args_cli.num_envs):
                    frame = _capture_frame(obs, env, i, cam_keys, scene_cam_names)
                    if frame is not None:
                        env_frames[i].append(frame)

            if delay_state is not None:
                delay_state["buf"][delay_state["head"]].copy_(actions)
                delay_state["head"] = (delay_state["head"] + 1) % delay_state["size"]
                actions_to_step = delay_state["buf"][delay_state["head"]].clone()
            else:
                actions_to_step = actions

            obs, _, dones, infos = env.step(actions_to_step)
            policy_nn.reset(dones)

            if dones.any():
                reset_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
                if args_cli.num_trajectories is not None:
                    remaining_episodes = args_cli.num_trajectories - episodes
                    counted_reset_ids = reset_ids[:remaining_episodes]
                else:
                    counted_reset_ids = reset_ids
                num_counted_episodes = len(counted_reset_ids)
                episodes += num_counted_episodes
                if has_success_term:
                    successful_episodes += _count_successes(env, counted_reset_ids, term_names)
                _collect_metrics(infos, episode_metrics, num_counted_episodes)

                if delay_state is not None:
                    delay_state["buf"][:, reset_ids, :] = 0

                if args_cli.save_video:
                    for env_id in reset_ids:
                        idx = env_id.item()
                        frames_to_save.extend(env_frames[idx])
                        env_frames[idx] = []
                    if frames_to_save:
                        imageio.mimsave("policy_cameras.mp4", frames_to_save, fps=10, codec="libx264")

                if pbar is not None:
                    pbar.update(num_counted_episodes)
                    rate = (successful_episodes / episodes * 100) if episodes > 0 else 0.0
                    pbar.set_description(
                        f"Evaluating checkpoint {checkpoint_index}/{num_checkpoints} (Success: {rate:.2f}%)"
                    )

        _print_results(args_cli.task, episodes, successful_episodes, episode_metrics, has_success_term)
        return _build_wandb_summary(episodes, successful_episodes, episode_metrics, has_success_term)
    finally:
        if pbar is not None:
            pbar.close()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate with an RSL-RL agent."""
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

    checkpoint_paths = _resolve_checkpoint_paths(args_cli.checkpoint)
    eval_experiment_name = args_cli.experiment_name or _get_eval_experiment_name(args_cli.checkpoint, args_cli.task)
    # Always disambiguate by delay/noise so runs that differ only by these CLI
    # flags don't collide in the W&B UI under the same auto-derived name.
    if args_cli.action_delay_steps > 0:
        eval_experiment_name = f"{eval_experiment_name}_delay{args_cli.action_delay_steps}"
    if args_cli.action_noise > 0:
        eval_experiment_name = f"{eval_experiment_name}_an{args_cli.action_noise:g}"
    print(f"[INFO]: Found {len(checkpoint_paths)} checkpoint(s) to evaluate.")
    print(f"[INFO]: Eval experiment name: {eval_experiment_name}")
    wandb_run = None
    if agent_cfg.logger == "wandb":
        import wandb

        wandb_run = wandb.init(
            project=agent_cfg.wandb_project,
            name=_get_wandb_run_name(eval_experiment_name),
            group=eval_experiment_name,
            config={
                "task": args_cli.task,
                "checkpoint_root": os.path.abspath(os.path.expanduser(args_cli.checkpoint)),
                "num_checkpoints": len(checkpoint_paths),
                "experiment_name": eval_experiment_name,
                "source_experiment_name": os.path.basename(
                    os.path.normpath(os.path.dirname(_get_checkpoint_run_dir(args_cli.checkpoint)))
                ),
                "eval_type": _get_eval_type(args_cli.task),
                "num_envs": args_cli.num_envs,
                "num_trajectories": args_cli.num_trajectories,
                "action_noise": args_cli.action_noise,
                "action_delay_steps": args_cli.action_delay_steps,
                "seed": args_cli.seed,
            },
        )
        # Use the checkpoint training-step (extracted from filename) as the x-axis
        # for eval/* metrics. Logged as a metric value in each log dict; do NOT
        # also pass step= to wandb.log (forcing the global step breaks any non-
        # eval metric and conflates "training step" with "log call ordinal").
        wandb.define_metric("checkpoint")
        wandb.define_metric("eval/*", step_metric="checkpoint")

    env = None
    try:
        env, runner, term_names, has_success_term = _create_eval_env_and_runner(
            env_cfg, agent_cfg, os.path.dirname(checkpoint_paths[0])
        )
        for checkpoint_index, resume_path in enumerate(checkpoint_paths, start=1):
            if not simulation_app.is_running():
                print("[INFO]: Simulation app stopped before all checkpoints were evaluated.")
                break
            summary = _evaluate_checkpoint(
                env, runner, resume_path, checkpoint_index, len(checkpoint_paths), term_names, has_success_term
            )
            if wandb_run is not None:
                checkpoint_step = _get_checkpoint_step(resume_path, checkpoint_index)
                wandb_run.log({"checkpoint": checkpoint_step, **summary})
    finally:
        if env is not None:
            env.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
    simulation_app.close()
