# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Eval-specific configs and event terms for reset states."""

from __future__ import annotations

import inspect

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm, ManagerTermBase

from ... import mdp as task_mdp
from ...mdp.events import MultiResetManager, sample_from_nested_dict

from .rl_state_cfg import EVAL_VARIANTS


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


def parse_osc_variants(variants_cli: list[str], variant_cfgs: list) -> list[dict]:
    """Parse --variant strings into list of OSC gain dicts (one per variant, not per env)."""
    if not variants_cli:
        if not variant_cfgs:
            raise ValueError("EVAL_VARIANTS is empty. Cannot parse OSC variants.")
        osc_variants = []
        for cfg in variant_cfgs:
            p = cfg.events.randomize_robot_actuator_parameters.params
            osc_variants.append({
                "stiffness_distribution_params": p["stiffness_distribution_params"],
                "damping_distribution_params": p["damping_distribution_params"],
            })
        return osc_variants

    parsed = []
    for s in variants_cli:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Invalid --variant '{s}'. Expected 4 comma-separated floats.")
        smn, smx, dmn, dmx = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        parsed.append({
            "stiffness_distribution_params": (smn, smx),
            "damping_distribution_params": (dmn, dmx)
        })
    return parsed


def multi_variant_osc_gains(
    env,
    env_ids: torch.Tensor,
    *,
    action_name: str,
    variants: list[dict],
    operation: str = "scale",
    distribution: str = "uniform",
) -> None:
    """Apply per-variant OSC gain variants, batching envs by variant group."""
    if not variants:
        return
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    num_variants = len(variants)
    for variant_idx, v in enumerate(variants):
        mask = (env_ids % num_variants) == variant_idx
        variant_env_ids = env_ids[mask]
        if len(variant_env_ids) == 0:
            continue
        task_mdp.randomize_operational_space_controller_gains(
            env,
            variant_env_ids,
            action_name=action_name,
            stiffness_distribution_params=v["stiffness_distribution_params"],
            damping_distribution_params=v["damping_distribution_params"],
            operation=operation,
            distribution=distribution,
        )


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

        if name == "randomize_robot_actuator_parameters":
            env_cfg.events.randomize_robot_actuator_parameters = EventTerm(  # type: ignore[attr-defined]
                func=multi_variant_osc_gains,
                mode=mode,
                params={"action_name": "arm", "variants": osc_variants, "operation": "scale", "distribution": "uniform"},
            )
        elif name == "reset_from_reset_states" and include_per_env_reset:
            # Extract unique base_paths from variants (one per variant, not per env)
            base_paths = [
                variant_cfgs[variant_idx].events.reset_from_reset_states.params["base_paths"][0]
                for variant_idx in range(len(EVAL_VARIANTS))
            ]

            env_cfg.events.reset_from_reset_states = EventTerm(
                func=PerEnvMultiResetManager,
                mode=mode,
                params={
                    "base_paths": base_paths,
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
    """MultiResetManager for eval: assigns each env to a fixed dataset (env_id % num_tasks)."""

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        base_paths: list[str],
        probs: list[float],
        success: str | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._env.device)

        if success is not None:
            success_mask = torch.where(eval(success)[env_ids], 1.0, 0.0)
            self.success_monitor.success_update(self.task_id[env_ids], success_mask)
            success_rates = self.success_monitor.get_success_rate()
            if "log" not in self._env.extras:
                self._env.extras["log"] = {}
            for task_idx in range(self.num_tasks):
                self._env.extras["log"].update({
                    f"Metrics/task_{task_idx}_success_rate": success_rates[task_idx].item(),
                    f"Metrics/task_{task_idx}_prob": self.probs[task_idx].item(),
                    f"Metrics/task_{task_idx}_normalized_prob": self.probs[task_idx].item(),
                })

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
            self._env.scene.reset_to(states_to_reset_from["initial_state"], env_ids=current_env_ids, is_relative=True)

        robot: Articulation = self._env.scene["robot"]
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel[env_ids]), env_ids=env_ids)
