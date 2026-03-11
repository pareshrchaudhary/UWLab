# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import tempfile

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase
from isaaclab.utils.assets import retrieve_file_path

from uwlab_tasks.manager_based.manipulation.assembly_task.common.mdp import utils
from uwlab_tasks.manager_based.manipulation.assembly_task.common.mdp.success_monitor_cfg import SuccessMonitorCfg
from uwlab_tasks.manager_based.manipulation.assembly_task.common.mdp.utils import sample_from_nested_dict, sample_state_data_set


class MultiResetManager(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        base_paths: list[str] = cfg.params.get("base_paths", [])
        probabilities: list[float] = cfg.params.get("probs", [])

        if not base_paths:
            raise ValueError("No base paths provided")
        if len(base_paths) != len(probabilities):
            raise ValueError("Number of base paths must match number of probabilities")

        # Compute dataset paths using object hash
        insertive_usd_path = env.scene["insertive_object"].cfg.spawn.usd_path
        receptive_usd_path = env.scene["receptive_object"].cfg.spawn.usd_path
        reset_state_hash = utils.compute_assembly_hash(insertive_usd_path, receptive_usd_path)

        # Generate dataset paths using provided base paths
        dataset_files = []
        for base_path in base_paths:
            dataset_files.append(f"{base_path}/{reset_state_hash}.pt")

        # Load all datasets
        self.datasets = []
        num_states = []
        rank = int(os.getenv("RANK", "0"))
        download_dir = os.path.join(tempfile.gettempdir(), f"rank_{rank}")
        for dataset_file in dataset_files:
            local_file_path = retrieve_file_path(dataset_file, download_dir=download_dir)

            if not os.path.exists(local_file_path):
                raise FileNotFoundError(f"Dataset file {dataset_file} could not be accessed or downloaded.")

            dataset = torch.load(local_file_path)
            num_states.append(len(dataset["initial_state"]["articulation"]["robot"]["joint_position"]))
            init_indices = torch.arange(num_states[-1], device=env.device)
            self.datasets.append(sample_state_data_set(dataset, init_indices, env.device))

        # Normalize probabilities and store dataset lengths
        self.probs = torch.tensor(probabilities, device=env.device) / sum(probabilities)
        self.num_states = torch.tensor(num_states, device=env.device)
        self.num_tasks = len(self.datasets)

        # Initialize success monitor
        if cfg.params.get("success") is not None:
            success_monitor_cfg = SuccessMonitorCfg(
                monitored_history_len=100, num_monitored_data=self.num_tasks, device=env.device
            )
            self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

        self.task_id = torch.randint(0, self.num_tasks, (self.num_envs,), device=self.device)

    def _reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: torch.Tensor | None = None,
        is_relative: bool = False
    ):
        """Reset entities in the scene to provided state, skipping assets not in the loaded state."""
        if env_ids is None:
            env_ids = self._env.scene._ALL_INDICES

        for asset_name, rigid_object in self._env.scene._rigid_objects.items():
            if asset_name not in state.get("rigid_object", {}):
                continue
            asset_state = state["rigid_object"][asset_name]
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += self._env.scene.env_origins[env_ids]
            root_velocity = asset_state["root_velocity"].clone()
            root_state = torch.cat([root_pose, root_velocity], dim=-1)
            rigid_object.write_root_state_to_sim(root_state, env_ids=env_ids)

        for asset_name, articulation in self._env.scene._articulations.items():
            if asset_name not in state.get("articulation", {}):
                continue
            asset_state = state["articulation"][asset_name]
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += self._env.scene.env_origins[env_ids]
            root_velocity = asset_state["root_velocity"].clone()
            articulation.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            articulation.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
            joint_position = asset_state["joint_position"].clone()
            joint_velocity = asset_state["joint_velocity"].clone()
            articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
            articulation.set_joint_position_target(joint_position, env_ids=env_ids)
            articulation.set_joint_velocity_target(joint_velocity, env_ids=env_ids)

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

        dataset_indices = torch.multinomial(self.probs, len(env_ids), replacement=True)
        self.task_id[env_ids] = dataset_indices

        for dataset_idx in range(self.num_tasks):
            mask = dataset_indices == dataset_idx
            if not mask.any():
                continue

            current_env_ids = env_ids[mask]
            state_indices = torch.randint(
                0, self.num_states[dataset_idx], (len(current_env_ids),), device=self._env.device
            )
            states_to_reset_from = sample_from_nested_dict(self.datasets[dataset_idx], state_indices)
            self._reset_to(states_to_reset_from["initial_state"], env_ids=current_env_ids, is_relative=True)

        robot: Articulation = self._env.scene["robot"]
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel[env_ids]), env_ids=env_ids)


