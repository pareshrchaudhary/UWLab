# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play script for RSL-RL agents using AsymmetricActorCritic (recurrent policy with obs_groups).

Use this when the agent config uses AsymmetricActorCritic (e.g. RslRlAsymmetricActorCriticCfg).
Handles recurrent hidden state reset and TensorDict observations correctly.

With --multi_eval: runs N envs (N = len(EVAL_VARIANTS)) with N different eval configs (OSC gains + reset state datasets).
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play an RL checkpoint with AsymmetricActorCritic (recurrent policy)."
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--multi_eval",
    action="store_true",
    help="Run 4 envs with 4 different eval configs (OSC gains + reset state datasets).",
)
parser.add_argument("--steps", type=int, default=2000, help="Max inference steps (ignored when --video).")
parser.add_argument(
    "--variant",
    action="append",
    default=[],
    help=(
        "Per-env OSC variant: 'stiff_min,stiff_max,damp_min,damp_max'. "
        "Provide N variants; env i uses variant i. If omitted with --multi_eval, defaults are used."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from rsl_rl.modules import AsymmetricActorCritic

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config
from uwlab_tasks.manager_based.manipulation.reset_states.config.ur5e_robotiq_2f85.eval_cfg import (
    NUM_EVAL_VARIANTS,
    apply_multi_eval_events,
)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent using AsymmetricActorCritic."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    num_envs = args_cli.num_envs
    if num_envs is None:
        num_envs = NUM_EVAL_VARIANTS if args_cli.multi_eval else env_cfg.scene.num_envs
    env_cfg.scene.num_envs = num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    if args_cli.multi_eval and isinstance(env_cfg, ManagerBasedRLEnvCfg):
        apply_multi_eval_events(
            env_cfg,
            env_cfg.scene.num_envs,
            args_cli.variant,
            include_per_env_reset=True,
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_asymmetric"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if not isinstance(policy_nn, AsymmetricActorCritic):
        print("[WARN] Loaded policy is not AsymmetricActorCritic. Recurrent reset may not work correctly.")

    normalizer = (
        getattr(policy_nn, "actor_obs_normalizer", None)
        or getattr(policy_nn, "student_obs_normalizer", None)
    )

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    max_steps = args_cli.video_length if args_cli.video else args_cli.steps
    ep_rew = torch.zeros(env.num_envs, device=env.unwrapped.device)
    ep_len = torch.zeros(env.num_envs, device=env.unwrapped.device)

    for step_i in range(max_steps):
        if not simulation_app.is_running():
            break
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            ep_rew += rewards
            ep_len += 1
            if args_cli.multi_eval:
                done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                for eid in done_ids.tolist():
                    print(
                        f"[Eval] env={eid} ep_rew={ep_rew[eid].item():.3f} ep_len={int(ep_len[eid].item())}"
                    )
                ep_rew[done_ids] = 0
                ep_len[done_ids] = 0
            policy_nn.reset(dones)
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
