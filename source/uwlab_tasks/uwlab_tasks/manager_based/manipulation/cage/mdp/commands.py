# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

import inspect
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm

from ..assembly_keypoints import Offset
from . import utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import TaskCommandCfg, TaskDependentCommandCfg


class TaskDependentCommand(CommandTerm):
    cfg: TaskDependentCommandCfg

    def __init__(self, cfg: TaskDependentCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        self.reset_terms_when_resample = cfg.reset_terms_when_resample
        self.interval_reset_terms = []
        self.reset_terms = []
        self.ALL_INDICES = torch.arange(self.num_envs, device=self.device)
        for name, term_cfg in self.reset_terms_when_resample.items():
            if not (term_cfg.mode == "reset" or term_cfg.mode == "interval"):
                raise ValueError(f"Term '{name}' in 'reset_terms_when_resample' must have mode 'reset' or 'interval'")
            if inspect.isclass(term_cfg.func):
                term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)
            if term_cfg.mode == "reset":
                self.reset_terms.append(term_cfg)
            elif term_cfg.mode == "interval":
                if term_cfg.interval_range_s != (0, 0):
                    raise ValueError(
                        "task dependent events term with interval mode current only supports range of (0, 0)"
                    )
                self.interval_reset_terms.append(term_cfg)

    def _resample_command(self, env_ids: Sequence[int]):
        for term in self.reset_terms:
            func = term.func
            func(self._env, env_ids, **term.params)
        for term in self.interval_reset_terms:
            func = term.func
            func.reset(env_ids)

    def _update_command(self):
        for term in self.interval_reset_terms:
            func = term.func
            func(self._env, self.ALL_INDICES, **term.params)

    def get_event(self, event_term_name: str):
        """Get the event term by name."""
        return self.reset_terms_when_resample.get(event_term_name).func


class TaskCommand(TaskDependentCommand):
    """Command generator that generates pose commands based on the terrain.

    This command generator samples the position commands from the valid patches of the terrain.
    The heading commands are either set to point towards the target or are sampled uniformly.

    It expects the terrain to have a valid flat patches under the key 'target'.
    """

    cfg: TaskCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TaskCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the terrain asset
        self.insertive_asset: Articulation | RigidObject = env.scene[cfg.insertive_asset_cfg.name]
        self.receptive_asset: Articulation | RigidObject = env.scene[cfg.receptive_asset_cfg.name]
        insertive_meta = utils.read_metadata_from_usd_directory(self.insertive_asset.cfg.spawn.usd_path)
        receptive_meta = utils.read_metadata_from_usd_directory(self.receptive_asset.cfg.spawn.usd_path)
        self.insertive_asset_offset = Offset(
            pos=tuple(insertive_meta.get("assembled_offset").get("pos")),
            quat=tuple(insertive_meta.get("assembled_offset").get("quat")),
        )
        self.receptive_asset_offset = Offset(
            pos=tuple(receptive_meta.get("assembled_offset").get("pos")),
            quat=tuple(receptive_meta.get("assembled_offset").get("quat")),
        )
        self.success_position_threshold: float = receptive_meta.get("success_thresholds").get("position")
        self.success_orientation_threshold: float = receptive_meta.get("success_thresholds").get("orientation")

        self.metrics["average_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["average_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_success_rate"] = torch.zeros(self.num_envs, device=self.device)

        self.orientation_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.position_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.euler_xy_distance = torch.zeros((self._env.num_envs), device=self._env.device)
        self.xyz_distance = torch.zeros((self._env.num_envs), device=self._env.device)

        # Inline settling: the runner flips envs between LIVE (protagonist-
        # driven) and SETTLING (adversary-driven) mid-rollout. SETTLING rows
        # would otherwise pollute `Metrics/task_command/*` with adversary
        # perturbation data. The runner writes this mask each step; default
        # all-True keeps standard single-policy runs unchanged.
        self._live_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_live_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_episode_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._last_episode_success_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        setattr(self._env, "_cage_task_command_last_episode_success", self._last_episode_success)
        setattr(self._env, "_cage_task_command_last_episode_success_valid", self._last_episode_success_valid)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 3, device=self.device)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs current data (computes euler_xy_distance / xyz_distance for
        # all envs — unconditional because downstream consumers read these)
        insertive_asset_alignment_pos_w, insertive_asset_alignment_quat_w = self.insertive_asset_offset.apply(
            self.insertive_asset
        )
        receptive_asset_alignment_pos_w, receptive_asset_alignment_quat_w = self.receptive_asset_offset.apply(
            self.receptive_asset
        )
        insertive_asset_in_receptive_asset_frame_pos, insertive_asset_in_receptive_asset_frame_quat = (
            math_utils.subtract_frame_transforms(
                receptive_asset_alignment_pos_w,
                receptive_asset_alignment_quat_w,
                insertive_asset_alignment_pos_w,
                insertive_asset_alignment_quat_w,
            )
        )
        e_x, e_y, _ = math_utils.euler_xyz_from_quat(insertive_asset_in_receptive_asset_frame_quat)
        self.euler_xy_distance[:] = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
        self.xyz_distance[:] = torch.norm(insertive_asset_in_receptive_asset_frame_pos, dim=1)
        self.position_aligned[:] = self.xyz_distance < self.success_position_threshold
        self.orientation_aligned[:] = self.euler_xy_distance < self.success_orientation_threshold

        # Metric writes are LIVE-only. `became_live` catches SETTLING→LIVE
        # flips that skip Isaac's `_reset_idx` (e.g. valid-settle restores
        # state via runner-side write); without it `end_of_episode_*` would
        # stay at the stale zero from the prior SETTLING reset.
        live = self._live_mask
        became_live = live & ~self._prev_live_mask
        episode_start = ((self._env.episode_length_buf == 0) & live) | became_live
        self.metrics["end_of_episode_rot_align_error"][episode_start] = self.euler_xy_distance[episode_start]
        self.metrics["end_of_episode_pos_align_error"][episode_start] = self.xyz_distance[episode_start]
        last_episode_success = (self.orientation_aligned & self.position_aligned)[episode_start]
        self.metrics["end_of_episode_success_rate"][episode_start] = last_episode_success.float()
        self.metrics["average_rot_align_error"][live] = self.euler_xy_distance[live]
        self.metrics["average_pos_align_error"][live] = self.xyz_distance[live]
        self._prev_live_mask.copy_(live)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Override to compute the logged mean over LIVE-only reset envs.

        Isaac's default ``CommandTerm.reset`` averages each metric over every
        env that reset this step. In inline settling, SETTLING-mode resets
        hold stale or zero metric values (``_update_metrics`` skips them),
        so pooling them into the mean would bias the scalar. We filter
        ``env_ids`` to LIVE envs for the mean, but still zero + resample on
        all reset envs to preserve Isaac's bookkeeping.
        """
        if env_ids is None:
            env_ids_t = self.ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            env_ids_t = env_ids
        else:
            env_ids_t = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)
        live_ids = env_ids_t[self._live_mask[env_ids_t]]
        extras: dict[str, float] = {}

        for metric_name, metric_value in self.metrics.items():
            if metric_name == "end_of_episode_success_rate":
                self._last_episode_success[env_ids_t] = metric_value[env_ids_t]
                self._last_episode_success_valid[env_ids_t] = self._live_mask[env_ids_t]
            if live_ids.numel() > 0:
                extras[metric_name] = torch.mean(metric_value[live_ids]).item()
            metric_value[env_ids_t] = 0.0
        self.command_counter[env_ids_t] = 0
        self._resample(env_ids_t)
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

    def _update_command(self):
        super()._update_command()

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
