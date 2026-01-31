# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Eval-specific configs and event terms for reset states."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm

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


def _params_equal(a: dict, b: dict) -> bool:
    """Compare params dicts (handles common types; complex objects compared by str)."""
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        try:
            if va == vb:
                continue
            if isinstance(va, (tuple, list)) and isinstance(vb, (tuple, list)) and len(va) == len(vb):
                if all(x == y for x, y in zip(va, vb)):
                    continue
            if isinstance(va, dict) and isinstance(vb, dict) and _params_equal(va, vb):
                continue
            return False
        except (TypeError, ValueError):
            if str(va) != str(vb):
                return False
    return True


def _extract_params_from_eval_variants():
    """Extract all event params from EVAL_VARIANTS. Returns per-variant events and derived constants."""
    variant_events: list[dict] = []
    all_event_names: set[str] = set()
    for variant_cls in EVAL_VARIANTS:
        cfg = variant_cls()
        events_dict = {}
        for name in _get_event_term_names(cfg.events):
            term = getattr(cfg.events, name)
            events_dict[name] = {"func": term.func, "mode": term.mode, "params": dict(term.params)}
            all_event_names.add(name)
        variant_events.append(events_dict)

    event_names = sorted(all_event_names)
    varying_events = []
    for name in event_names:
        first_params = variant_events[0].get(name, {}).get("params", {})
        for i in range(1, len(variant_events)):
            other_params = variant_events[i].get(name, {}).get("params", {})
            if not _params_equal(other_params, first_params):
                varying_events.append(name)
                break

    base_paths = []
    osc_variants = []
    for ev in variant_events:
        if "reset_from_reset_states" in ev:
            base_paths.append(ev["reset_from_reset_states"]["params"]["base_paths"][0])
        if "randomize_robot_actuator_parameters" in ev:
            p = ev["randomize_robot_actuator_parameters"]["params"]
            osc_variants.append({
                "stiffness_distribution_params": p["stiffness_distribution_params"],
                "damping_distribution_params": p["damping_distribution_params"],
            })

    return {
        "variant_events": variant_events,
        "varying_events": varying_events,
        "reset_state_base_paths": base_paths,
        "default_osc_variants": osc_variants,
    }


EVAL_EXTRACTED = _extract_params_from_eval_variants()
NUM_EVAL_VARIANTS = len(EVAL_EXTRACTED["variant_events"])


def parse_osc_variants(variants_cli: list[str], num_envs: int) -> list[dict]:
    """Parse --variant strings into list of OSC gain dicts."""
    if not variants_cli:
        return EVAL_EXTRACTED["default_osc_variants"][:num_envs]
    parsed: list[dict] = []
    for s in variants_cli:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Invalid --variant '{s}'. Expected 4 comma-separated floats.")
        smn, smx, dmn, dmx = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
        parsed.append(
            {"stiffness_distribution_params": (smn, smx), "damping_distribution_params": (dmn, dmx)}
        )
    if len(parsed) != num_envs:
        raise ValueError(f"Provided {len(parsed)} variants but num_envs={num_envs}. Provide exactly one per env.")
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
    """Apply per-env OSC gain variants (one config per env index)."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    for env_id in env_ids.tolist():
        if env_id < 0 or env_id >= len(variants):
            continue
        v = variants[env_id]
        task_mdp.randomize_operational_space_controller_gains(
            env,
            torch.tensor([env_id], device=env.device, dtype=torch.long),
            action_name=action_name,
            stiffness_distribution_params=v["stiffness_distribution_params"],
            damping_distribution_params=v["damping_distribution_params"],
            operation=operation,
            distribution=distribution,
        )


def _make_multi_variant_event(base_func):
    """Return a wrapper that applies base_func per-env with variant-specific params."""

    def wrapper(env, env_ids: torch.Tensor | None, *, variants: list[dict], **kwargs) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device)
        for env_id in env_ids.tolist():
            if env_id < 0 or env_id >= len(variants):
                continue
            params = {**kwargs, **variants[env_id]}
            base_func(env, torch.tensor([env_id], device=env.device, dtype=torch.long), **params)

    return wrapper


def apply_multi_eval_events(
    env_cfg,
    num_envs: int,
    variants_cli: list[str],
    *,
    include_per_env_reset: bool = True,
) -> None:
    """Apply multi-eval event overrides from EVAL_VARIANTS. Varying events get per-env params."""
    if not hasattr(env_cfg, "events"):
        return
    osc_variants = parse_osc_variants(variants_cli, num_envs)
    variant_events = [ev for ev in EVAL_EXTRACTED["variant_events"][:num_envs]]
    varying_names = EVAL_EXTRACTED["varying_events"]

    for name in _get_event_term_names(env_cfg.events):
        if name not in varying_names:
            continue
        first = variant_events[0].get(name)
        if not first:
            continue
        func, mode = first["func"], first["mode"]

        if name == "randomize_robot_actuator_parameters":
            env_cfg.events.randomize_robot_actuator_parameters = EventTerm(  # type: ignore[attr-defined]
                func=multi_variant_osc_gains,
                mode=mode,
                params={"action_name": "arm", "variants": osc_variants, "operation": "scale", "distribution": "uniform"},
            )
        elif name == "reset_from_reset_states" and include_per_env_reset:
            env_cfg.events.reset_from_reset_states = EventTerm(
                func=PerEnvMultiResetManager,
                mode=mode,
                params={
                    "base_paths": EVAL_EXTRACTED["reset_state_base_paths"][:num_envs],
                    "probs": [1.0 / num_envs] * num_envs,
                    "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
                },
            )
        elif callable(func):
            params_list = [variant_events[i][name]["params"] for i in range(len(variant_events))]
            env_cfg.events.__setattr__(name, EventTerm(  # type: ignore[attr-defined]
                func=_make_multi_variant_event(func),
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
