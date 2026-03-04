# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from OctiLab octilab_apps.utils.wrappers.diffusion

import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Any, List
from abc import ABC, abstractmethod


class ObservationHistoryManager(ABC):
    """Abstract base class for managing observation history."""

    def __init__(self, num_envs: int, n_obs_steps: int, device: torch.device):
        self.num_envs = num_envs
        self.n_obs_steps = n_obs_steps
        self.device = device
        self.history = None
        self.needs_init = set()

    @abstractmethod
    def initialize(self, processed_obs: Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def update(self, processed_obs: Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def get_batch(self, env_indices: List[int]) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def reset_envs(self, env_indices: List[int]):
        pass


class LowDimObservationHistory(ObservationHistoryManager):
    """Manages observation history for low-dimensional policies."""

    def initialize(self, processed_obs: Dict[str, torch.Tensor]):
        obs_shape = processed_obs["obs"].shape
        history_shape = (self.num_envs, self.n_obs_steps, obs_shape[-1])
        self.history = torch.zeros(history_shape, device=self.device, dtype=processed_obs["obs"].dtype)

    def update(self, processed_obs: Dict[str, torch.Tensor]):
        if self.history is None:
            self.initialize(processed_obs)
        if self.needs_init:
            for env_idx in list(self.needs_init):
                first_obs = processed_obs["obs"][env_idx : env_idx + 1]
                for step in range(self.n_obs_steps):
                    self.history[env_idx, step] = first_obs[0]
                self.needs_init.remove(env_idx)
        self.history[:, :-1] = self.history[:, 1:]
        self.history[:, -1] = processed_obs["obs"]

    def get_batch(self, env_indices: List[int]) -> Dict[str, torch.Tensor]:
        if self.history is None:
            return {"obs": torch.zeros((len(env_indices), self.n_obs_steps, 0), device=self.device)}
        return {"obs": self.history[env_indices]}

    def reset_envs(self, env_indices: List[int]):
        for i in env_indices:
            self.needs_init.add(i)


class ImageObservationHistory(ObservationHistoryManager):
    """Manages observation history for image-based policies."""

    def __init__(self, num_envs: int, obs_keys: List[str], n_obs_steps: int, device: torch.device):
        super().__init__(num_envs, n_obs_steps, device)
        self.obs_keys = obs_keys

    def initialize(self, processed_obs: Dict[str, torch.Tensor]):
        self.history = {}
        for key in self.obs_keys:
            obs_shape = processed_obs[key].shape
            history_shape = (self.num_envs, self.n_obs_steps) + obs_shape[1:]
            self.history[key] = torch.zeros(history_shape, device=self.device, dtype=processed_obs[key].dtype)

    def update(self, processed_obs: Dict[str, torch.Tensor]):
        if self.history is None:
            self.initialize(processed_obs)
        if self.needs_init and self.obs_keys is not None:
            for env_idx in list(self.needs_init):
                if env_idx < self.num_envs:
                    for key in self.obs_keys:
                        first_obs = processed_obs[key][env_idx : env_idx + 1]
                        for step in range(self.n_obs_steps):
                            self.history[key][env_idx, step] = first_obs[0]
                    self.needs_init.remove(env_idx)
        if self.obs_keys is not None:
            for key in self.obs_keys:
                self.history[key][:, :-1] = self.history[key][:, 1:].clone()
                self.history[key][:, -1] = processed_obs[key]

    def get_batch(self, env_indices: List[int]) -> Dict[str, torch.Tensor]:
        if self.history is None or self.obs_keys is None:
            return {}
        return {key: self.history[key][env_indices] for key in self.obs_keys}

    def reset_envs(self, env_indices: List[int]):
        for i in env_indices:
            self.needs_init.add(i)


class ImageObservationSequence(ImageObservationHistory):
    """Stores full trajectory per-environment as lists and returns padded batches with attention masks."""

    def initialize(self, processed_obs: Dict[str, torch.Tensor]):
        self.history = {key: [[] for _ in range(self.num_envs)] for key in self.obs_keys}

    def update(self, processed_obs: Dict[str, torch.Tensor], env_indices: List[int]):
        if self.history is None:
            self.initialize(processed_obs)
        if self.needs_init and self.obs_keys is not None:
            for env_idx in list(self.needs_init):
                if env_idx < self.num_envs:
                    for key in self.obs_keys:
                        self.history[key][env_idx].clear()
                    self.needs_init.remove(env_idx)
        if self.obs_keys is not None:
            for key in self.obs_keys:
                tensor = processed_obs[key]
                for idx, env_idx in enumerate(env_indices):
                    obs = tensor[idx]
                    self.history[key][env_idx].append(obs.detach().clone().to(self.device))

    def get_batch(self, env_indices: List[int]) -> Dict[str, torch.Tensor]:
        if self.history is None or self.obs_keys is None:
            return {}
        lengths = [len(self.history[self.obs_keys[0]][env]) for env in env_indices]
        max_len = max(lengths) if lengths else 0
        lengths_tensor = torch.tensor(lengths, device=self.device)
        attention_mask = (torch.arange(max_len, device=self.device).unsqueeze(0) < lengths_tensor.unsqueeze(1)).long()
        obs_batch = {
            key: pad_sequence(
                [torch.stack(self.history[key][env], dim=0) for env in env_indices],
                batch_first=True,
                padding_value=0.0,
            )
            for key in self.obs_keys
        }
        obs_batch["attention_mask"] = attention_mask
        return obs_batch


class DiffusionPolicyWrapper:
    """Wraps diffusion policy to handle Isaac Lab environment observations and action execution."""

    def __init__(self, policy, device: torch.device, n_obs_steps: int = 2, num_envs: int = 1):
        self.policy = policy
        self.device = device
        self.n_obs_steps = n_obs_steps
        self.num_envs = num_envs

        self.is_image_policy = self._is_image_policy()
        self.is_transformer = self._is_transformer()
        if hasattr(policy.obs_encoder, "keys"):
            obs_keys = policy.obs_encoder.keys
        else:
            obs_keys = policy.obs_encoder.rgb_keys + policy.obs_encoder.low_dim_keys
        if self.is_transformer:
            self.obs_history_manager = ImageObservationSequence(num_envs, obs_keys, n_obs_steps, device)
        elif self.is_image_policy:
            self.obs_history_manager = ImageObservationHistory(num_envs, obs_keys, n_obs_steps, device)
        else:
            self.obs_history_manager = LowDimObservationHistory(num_envs, n_obs_steps, device)

        self.action_queue = [[] for _ in range(num_envs)]
        self.policy.reset()

    def _is_transformer(self) -> bool:
        policy_class_name = self.policy.__class__.__name__.lower()
        return any(
            ind in policy_class_name for ind in ["transformer", "gpt", "bert", "dpt"]
        )

    def _is_image_policy(self) -> bool:
        policy_class_name = self.policy.__class__.__name__.lower()
        return any(ind in policy_class_name for ind in ["image", "hybrid", "video"])

    def reset(self, reset_ids: torch.Tensor):
        reset_indices = reset_ids.tolist() if hasattr(reset_ids, "tolist") else reset_ids
        if not isinstance(reset_indices, list):
            reset_indices = [reset_indices]
        for i in reset_indices:
            self.action_queue[i].clear()
        self.obs_history_manager.reset_envs(reset_indices)
        self.policy.reset()

    def predict_action(self, obs_dict: Dict[str, Any], env_indices: List[int] = None) -> torch.Tensor:
        processed_obs = self._process_obs(obs_dict)

        if self.is_transformer:
            if env_indices is None:
                env_indices = list(range(self.num_envs))
            self.obs_history_manager.update(processed_obs, env_indices)
            need_new_actions = [
                i for i in range(self.num_envs)
                if len(self.action_queue[i]) == 0 and i in env_indices
            ]
        else:
            self.obs_history_manager.update(processed_obs)
            need_new_actions = [i for i in range(self.num_envs) if len(self.action_queue[i]) == 0]

        if need_new_actions:
            new_actions = self._get_action_chunks(need_new_actions)
            for idx, env_idx in enumerate(need_new_actions):
                self.action_queue[env_idx].extend(new_actions[idx])

        if self.is_transformer:
            actions = torch.zeros(
                len(env_indices),
                self.action_queue[env_indices[0]][0].shape[-1],
                device=self.device,
                dtype=torch.float32,
            )
            for idx, env_idx in enumerate(env_indices):
                actions[idx] = self.action_queue[env_idx].pop(0)
            return actions
        else:
            actions = torch.zeros(
                self.num_envs,
                self.action_queue[0][0].shape[-1],
                device=self.device,
                dtype=torch.float32,
            )
            for i in range(self.num_envs):
                actions[i] = self.action_queue[i].pop(0)
            return actions

    def _process_obs(self, obs_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        obs = obs_dict.get("policy", obs_dict) if isinstance(obs_dict, dict) else obs_dict
        if self.is_image_policy:
            return self._process_image_obs(obs)
        return self._process_lowdim_obs(obs)

    def _process_image_obs(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else torch.tensor(value, device=self.device)
            for key, value in obs.items()
        }

    def _process_lowdim_obs(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        obs_components = []
        for key in sorted(obs.keys()):
            v = obs[key]
            obs_components.append(v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device))
        if obs_components:
            obs_tensor = torch.cat(obs_components, dim=-1)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            return {"obs": obs_tensor}
        return {"obs": torch.zeros((self.num_envs, 0), device=self.device)}

    def _get_action_chunks(self, env_indices: List[int]) -> List[torch.Tensor]:
        obs_batch = self.obs_history_manager.get_batch(env_indices)
        with torch.no_grad():
            result = self.policy.predict_action(obs_batch)
        action_chunk = result["action"] if isinstance(result, dict) else result

        if action_chunk.ndim == 3:
            return [action_chunk[i] for i in range(action_chunk.shape[0])]
        return action_chunk.unsqueeze(1)
