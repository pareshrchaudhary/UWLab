# Copyright (c) 2024-2025, The UW Lab Project Developers.
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

"""Extended policy exporter that includes compute_distribution for stochastic sampling."""

import os
import torch
from torch import nn

from isaaclab_rl.rsl_rl.exporter import _TorchPolicyExporter, _OnnxPolicyExporter


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file with compute_distribution support.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporterExtended(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file with compute_distribution support.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporterExtended(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


class _StateDependentPolicyMixin(nn.Module):
    """Mixin that adds compute_distribution to exported policies."""

    def _setup_state_dependent_policy(self, policy):
        self.actor_features = self.actor[:-1]
        self.actor_final = self.actor[-1]
        self.register_buffer('log_std', policy.log_std.clone())
        self.epsilon = 1e-6

    def _setup_regular_policy(self, policy):
        self.actor_features = self.actor[:-1]
        self.actor_final = self.actor[-1]

        if hasattr(policy, 'std'):
            self.register_buffer('std', policy.std.clone())
        if hasattr(policy, 'log_std'):
            self.register_buffer('log_std', policy.log_std.clone())
        if hasattr(policy, 'noise_std_type'):
            self.noise_std_type = policy.noise_std_type
        else:
            self.noise_std_type = "scalar"

        if self.noise_std_type == "gsde":
            self.epsilon = 1e-6

    def _ensure_compatibility_attributes(self, policy):
        """Ensure all attributes exist for TorchScript compatibility."""
        if not hasattr(self, 'std'):
            if hasattr(policy, 'std'):
                self.register_buffer('std', policy.std.clone())
            else:
                default_std = torch.ones(policy.num_actions if hasattr(policy, 'num_actions') else 1)
                self.register_buffer('std', default_std)

        if not hasattr(self, 'log_std'):
            if hasattr(policy, 'log_std'):
                self.register_buffer('log_std', policy.log_std.clone())
            else:
                default_log_std = torch.zeros(policy.num_actions if hasattr(policy, 'num_actions') else 1)
                self.register_buffer('log_std', default_log_std)

        if not hasattr(self, 'epsilon'):
            self.epsilon = 1e-6

        if not hasattr(self, 'noise_std_type'):
            if hasattr(policy, 'noise_std_type'):
                self.noise_std_type = policy.noise_std_type
            else:
                self.noise_std_type = "scalar"

        if self.noise_std_type == "gsde" and not hasattr(self, 'epsilon'):
            self.epsilon = 1e-6

    def _compute_distribution(self, observations):
        """Compute mean and std for the action distribution."""
        if self.is_state_dependent.item():
            features = self.actor_features(observations)
            mean = self.actor_final(features)
            variance = torch.mm(features**2, torch.exp(self.log_std) ** 2)
            std = torch.sqrt(variance + self.epsilon)
            return mean, std
        else:
            mean = self.actor(observations)

            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            elif self.noise_std_type == "gsde":
                features = self.actor_features(observations)
                variance = torch.mm(features**2, torch.exp(self.log_std) ** 2)
                std = torch.sqrt(variance + self.epsilon)
            else:
                std = torch.ones_like(mean)

            return mean, std


class _TorchPolicyExporterExtended(_TorchPolicyExporter, _StateDependentPolicyMixin):
    def __init__(self, policy, normalizer=None):
        super().__init__(policy, normalizer)

        is_state_dependent = hasattr(policy, 'use_state_dependent_noise') and policy.use_state_dependent_noise
        self.register_buffer('is_state_dependent', torch.tensor(is_state_dependent, dtype=torch.bool))

        if is_state_dependent:
            self._setup_state_dependent_policy(policy)
        else:
            self._setup_regular_policy(policy)

        self._ensure_compatibility_attributes(policy)

    @torch.jit.export
    def compute_distribution(self, x):
        observations = self.normalizer(x)
        return self._compute_distribution(observations)


class _OnnxPolicyExporterExtended(_OnnxPolicyExporter, _StateDependentPolicyMixin):
    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__(policy, normalizer, verbose)

        is_state_dependent = hasattr(policy, 'use_state_dependent_noise') and policy.use_state_dependent_noise
        self.register_buffer('is_state_dependent', torch.tensor(is_state_dependent, dtype=torch.bool))

        if is_state_dependent:
            self._setup_state_dependent_policy(policy)
        else:
            self._setup_regular_policy(policy)

    @torch.jit.export
    def compute_distribution(self, x):
        observations = self.normalizer(x)
        return self._compute_distribution(observations)
