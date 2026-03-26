# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Eval-specific configs and event terms for reset states."""

from __future__ import annotations

import inspect
import os

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm, ManagerTermBase

from . import utils
from .events import (
    MultiResetManager,
    randomize_rel_cartesian_osc_gains_fixed,
    sample_from_nested_dict,
    sample_state_data_set,
)
from .success_monitor_cfg import SuccessMonitorCfg

from ..config.ur5e_robotiq_2f85.eval_cfg import EVAL_VARIANTS


def _get_event_term_names(events_cfg) -> list[str]:
    """Get all event term names from events config."""
    names = []
    for name in dir(events_cfg):
        if name.startswith("_"):
            continue
        val = getattr(events_cfg, name)
        if hasattr(val, "func") and hasattr(val, "params") and hasattr(val, "mode"):
            names.append(name)
    return names


NUM_EVAL_VARIANTS = len(EVAL_VARIANTS)


def _load_single_reset_dataset(env: ManagerBasedEnv, dataset_dir: str, reset_type: str):
    """Load one resets_*.pt file for the scene object pair (same layout as MultiResetManager)."""
    insertive_usd_path = env.scene["insertive_object"].cfg.spawn.usd_path
    receptive_usd_path = env.scene["receptive_object"].cfg.spawn.usd_path
    pair = utils.compute_pair_dir(insertive_usd_path, receptive_usd_path)
    dataset_file = f"{dataset_dir}/Resets/{pair}/resets_{reset_type}.pt"
    local_file_path = utils.safe_retrieve_file_path(dataset_file)
    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"Dataset file {dataset_file} could not be accessed or downloaded.")
    dataset = torch.load(local_file_path)
    n = len(dataset["initial_state"]["articulation"]["robot"]["joint_position"])
    init_indices = torch.arange(n, device=env.device)
    return sample_state_data_set(dataset, init_indices, env.device), n


def parse_osc_variants(variants_cli: list[str], variant_cfgs: list) -> list[dict]:
    """Parse --variant strings into list of OSC gain dicts (one per variant, not per env)."""
    if not variants_cli:
        if not variant_cfgs:
            raise ValueError("EVAL_VARIANTS is empty. Cannot parse OSC variants.")
        osc_variants = []
        for cfg in variant_cfgs:
            p = cfg.events.randomize_osc_gains.params
            osc_variants.append({
                "action_name": p["action_name"],
                "scale_range": p["scale_range"],
            })
        return osc_variants

    parsed = []
    for s in variants_cli:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid --variant '{s}'. Expected 2 comma-separated floats (lo,hi).")
        lo, hi = float(parts[0]), float(parts[1])
        parsed.append({
            "action_name": "arm",
            "scale_range": (lo, hi),
        })
    return parsed


def _make_multi_variant_event(base_func):
    """Return a wrapper that applies base_func batched by variant group."""

    def wrapper(env, env_ids: torch.Tensor | None, variants: list[dict]) -> None:
        if not variants:
            return
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device)
        num_variants = len(variants)
        for variant_idx, params in enumerate(variants):
            mask = (env_ids % num_variants) == variant_idx
            variant_env_ids = env_ids[mask]
            if len(variant_env_ids) == 0:
                continue
            base_func(env, variant_env_ids, **params)

    return wrapper


def _make_multi_variant_class_event(base_class):
    """Return a ManagerTermBase subclass that dispatches per variant group.

    For class-based event terms (e.g. randomize_rigid_body_material), each variant
    needs its own instance since __init__ pre-computes internal state (e.g. material
    buckets) from the config params.
    """

    class MultiVariantTerm(ManagerTermBase):
        def __init__(self, cfg: EventTerm, env: ManagerBasedEnv):
            super().__init__(cfg, env)
            variants = cfg.params["variants"]
            self.num_variants = len(variants)
            # Create one instance per variant with its own params
            self.variant_instances = []
            for variant_params in variants:
                variant_cfg = EventTerm(func=base_class, mode=cfg.mode, params=variant_params)
                self.variant_instances.append(base_class(variant_cfg, env))

        def __call__(self, env, env_ids, variants):
            if env_ids is None:
                env_ids = torch.arange(env.scene.num_envs, device=env.device)
            for variant_idx, (instance, params) in enumerate(
                zip(self.variant_instances, variants)
            ):
                mask = (env_ids % self.num_variants) == variant_idx
                variant_env_ids = env_ids[mask]
                if len(variant_env_ids) == 0:
                    continue
                instance(env, variant_env_ids, **params)

    return MultiVariantTerm


def apply_multi_eval_events(
    env_cfg,
    variants_cli: list[str],
    *,
    include_per_env_reset: bool = True,
) -> None:
    """Apply multi-eval event overrides from EVAL_VARIANTS. Varying events get per-env params."""
    if not hasattr(env_cfg, "events"):
        return

    if not EVAL_VARIANTS:
        raise ValueError("EVAL_VARIANTS is empty. Cannot apply multi-eval events.")

    # Cache variant configs to avoid recreating them for every env
    variant_cfgs = [variant_cls() for variant_cls in EVAL_VARIANTS]

    osc_variants = parse_osc_variants(variants_cli, variant_cfgs)
    first_variant_cfg = variant_cfgs[0]

    for name in _get_event_term_names(env_cfg.events):
        if not hasattr(first_variant_cfg.events, name):
            continue

        first_term = getattr(first_variant_cfg.events, name)
        func, mode = first_term.func, first_term.mode

        if name == "randomize_osc_gains":
            # OSC gains: use class-based multi-variant wrapper
            wrapper_func = _make_multi_variant_class_event(randomize_rel_cartesian_osc_gains_fixed)
            env_cfg.events.randomize_osc_gains = EventTerm(  # type: ignore[attr-defined]
                func=wrapper_func,
                mode=mode,
                params={"variants": osc_variants},
            )
        elif name == "reset_from_reset_states" and include_per_env_reset:
            variant_reset_params = []
            for variant_idx in range(len(EVAL_VARIANTS)):
                p = dict(variant_cfgs[variant_idx].events.reset_from_reset_states.params)
                p.pop("success", None)
                variant_reset_params.append(p)

            env_cfg.events.reset_from_reset_states = EventTerm(
                func=PerEnvMultiResetManager,
                mode=mode,
                params={
                    "variant_reset_params": variant_reset_params,
                    "probs": [1.0 / len(EVAL_VARIANTS)] * len(EVAL_VARIANTS),
                    "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
                },
            )
        elif callable(func):
            params_list = [
                dict(getattr(variant_cfgs[variant_idx].events, name).params)
                for variant_idx in range(len(EVAL_VARIANTS))
            ]

            # Skip wrapping if all variants have identical params for this event
            if all(p == params_list[0] for p in params_list):
                continue

            # Use class-based wrapper for ManagerTermBase subclasses,
            # plain function wrapper otherwise
            if inspect.isclass(func) and issubclass(func, ManagerTermBase):
                wrapper_func = _make_multi_variant_class_event(func)
            else:
                wrapper_func = _make_multi_variant_event(func)

            env_cfg.events.__setattr__(name, EventTerm(  # type: ignore[attr-defined]
                func=wrapper_func,
                mode=mode,
                params={"variants": params_list},
            ))


class PerEnvMultiResetManager(MultiResetManager):
    """MultiResetManager for eval: assigns each env to a fixed dataset (env_id % num_tasks).

    Expects ``variant_reset_params``: list of dicts with ``dataset_dir``, ``reset_types`` (one
    type per variant), and ``probs`` (per-type weights for that variant, length 1), matching
    :class:`MultiResetManager` loading. Top-level ``probs`` is used only for per-task metrics.
    """

    def __init__(self, cfg: EventTerm, env: ManagerBasedEnv):
        ManagerTermBase.__init__(self, cfg, env)

        variant_specs = cfg.params.get("variant_reset_params")
        if not variant_specs:
            raise ValueError("PerEnvMultiResetManager requires variant_reset_params")

        self.datasets = []
        num_states: list[int] = []
        for spec in variant_specs:
            rts = spec["reset_types"]
            if len(rts) != 1:
                raise ValueError(
                    "PerEnvMultiResetManager expects exactly one reset_type per eval variant; got "
                    f"{rts!r}"
                )
            pr = spec.get("probs", [])
            if len(pr) != 1:
                raise ValueError(
                    "PerEnvMultiResetManager expects one probability per variant reset_types entry; "
                    f"got probs={pr!r}"
                )
            ds, n = _load_single_reset_dataset(env, spec["dataset_dir"], rts[0])
            self.datasets.append(ds)
            num_states.append(n)

        self.num_states = torch.tensor(num_states, device=env.device)
        self.num_tasks = len(self.datasets)

        log_probs = cfg.params.get("probs")
        if log_probs is None or len(log_probs) != self.num_tasks:
            raise ValueError("Top-level probs length must match len(variant_reset_params)")
        self.eval_task_probs = torch.tensor(log_probs, device=env.device, dtype=torch.float32)

        if cfg.params.get("success") is not None:
            success_monitor_cfg = SuccessMonitorCfg(
                monitored_history_len=100, num_monitored_data=self.num_tasks, device=env.device
            )
            self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)
        else:
            self.success_monitor = None

        self.task_id = torch.randint(0, self.num_tasks, (self.num_envs,), device=self.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        variant_reset_params: list[dict],
        probs: list[float],
        success: str | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._env.device)

        if success is not None and self.success_monitor is not None:
            success_mask = torch.where(eval(success)[env_ids], 1.0, 0.0)
            self.success_monitor.success_update(self.task_id[env_ids], success_mask)
            success_rates = self.success_monitor.get_success_rate()
            if "log" not in self._env.extras:
                self._env.extras["log"] = {}
            for task_idx in range(self.num_tasks):
                self._env.extras["log"].update({
                    f"Metrics/task_{task_idx}_success_rate": success_rates[task_idx].item(),
                    f"Metrics/task_{task_idx}_prob": self.eval_task_probs[task_idx].item(),
                    f"Metrics/task_{task_idx}_normalized_prob": self.eval_task_probs[task_idx].item(),
                })
            ep_lengths = self._env.episode_length_buf[env_ids].float()
            self._env.extras["log"]["Metrics/mean_episode_length"] = ep_lengths.mean().item()

        dataset_indices = (env_ids % self.num_tasks).long()
        self.task_id[env_ids] = dataset_indices

        for dataset_idx in range(self.num_tasks):
            mask = dataset_indices == dataset_idx
            if not mask.any():
                continue
            current_env_ids = env_ids[mask]
            state_indices = torch.randint(
                0, int(self.num_states[dataset_idx].item()), (len(current_env_ids),), device=self._env.device
            )
            states_to_reset_from = sample_from_nested_dict(self.datasets[dataset_idx], state_indices)
            self._reset_to(states_to_reset_from["initial_state"], env_ids=current_env_ids, is_relative=True)

        robot: Articulation = self._env.scene["robot"]
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel[env_ids]), env_ids=env_ids)
