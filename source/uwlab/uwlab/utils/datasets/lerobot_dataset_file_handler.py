# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
LeRobot dataset file handler for recorder_manager.

Implements DatasetFileHandlerBase so collect_demos can write LeRobot-format
datasets (for OpenPI finetuning) via the same recorder pipeline as Zarr/HDF5.
Expects episode data from ActionStateRecorderManagerCfg: obs = data_collection
group (front_rgb, side_rgb, wrist_rgb, arm_joint_pos, end_effector_pose,
last_gripper_action), actions [T, 7].
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from collections.abc import Iterable

import numpy as np

from isaaclab.utils.datasets.dataset_file_handler_base import DatasetFileHandlerBase
from isaaclab.utils.datasets.episode_data import EpisodeData


def _create_lerobot_dataset(root: Path, repo_id: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if root.exists():
        shutil.rmtree(root)
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="ur5e",
        fps=10,
        features={
            "front_rgb": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "side_rgb": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "wrist_rgb": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (13,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


class LeRobotDatasetFileHandler(DatasetFileHandlerBase):
    """Dataset file handler that writes episodes in LeRobot format (directory)."""

    def __init__(self, task_name: str = ""):
        self._dataset = None
        self._dataset_path = None
        self._env_name = None
        self._episode_count = 0
        self._task_name = task_name

    def create(self, file_path: str, env_name: str | None = None):
        """Create a new LeRobot dataset at file_path (directory)."""
        self._dataset_path = Path(file_path).resolve()
        self._env_name = env_name or "isaac_lab_env"
        self._task_name = self._task_name or (env_name or "")
        parent = self._dataset_path.parent
        if str(parent) != ".":
            os.makedirs(parent, exist_ok=True)
        repo_id = self._dataset_path.name or "lerobot_demos"
        self._dataset = _create_lerobot_dataset(self._dataset_path, repo_id)
        self._episode_count = 0

    def open(self, file_path: str, mode: str = "r"):
        raise NotImplementedError("Open not implemented for LeRobot handler")

    def get_env_name(self) -> str | None:
        return self._env_name

    def get_episode_names(self) -> Iterable[str]:
        if self._dataset is None:
            return []
        return [f"episode_{i:06d}" for i in range(self._episode_count)]

    def get_num_episodes(self) -> int:
        return self._episode_count

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None):
        if self._dataset is None or episode.is_empty():
            return
        data = episode.data
        if "obs" not in data or "actions" not in data:
            return
        obs = data["obs"]
        actions = data["actions"]
        n = actions.shape[0]
        for key in ("front_rgb", "side_rgb", "wrist_rgb", "arm_joint_pos", "end_effector_pose", "last_gripper_action"):
            if key not in obs:
                return
        for t in range(n):
            state = np.concatenate([
                obs["arm_joint_pos"][t].cpu().numpy().flatten(),
                obs["end_effector_pose"][t].cpu().numpy().flatten(),
                obs["last_gripper_action"][t].cpu().numpy().flatten(),
            ]).astype(np.float32)
            frame = {
                "front_rgb": obs["front_rgb"][t].cpu().numpy().astype(np.uint8),
                "side_rgb": obs["side_rgb"][t].cpu().numpy().astype(np.uint8),
                "wrist_rgb": obs["wrist_rgb"][t].cpu().numpy().astype(np.uint8),
                "state": state,
                "actions": actions[t].cpu().numpy().astype(np.float32),
                "task": self._task_name,
            }
            self._dataset.add_frame(frame)
        self._dataset.save_episode()
        if demo_id is None:
            self._episode_count += 1

    def load_episode(self, episode_name: str) -> EpisodeData | None:
        raise NotImplementedError("Load not implemented for LeRobot handler")

    def flush(self):
        pass

    def close(self):
        self._dataset = None

    def add_env_args(self, env_args: dict):
        if "task" in env_args:
            self._task_name = str(env_args["task"])
