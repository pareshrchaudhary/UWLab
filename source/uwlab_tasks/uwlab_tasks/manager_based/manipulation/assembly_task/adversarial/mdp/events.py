# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import tempfile

import torch

from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.version import get_isaac_sim_version

from uwlab_tasks.manager_based.manipulation.assembly_task.common.mdp import utils
from uwlab_tasks.manager_based.manipulation.assembly_task.common.mdp.success_monitor_cfg import SuccessMonitorCfg
from uwlab_tasks.manager_based.manipulation.assembly_task.common.mdp.utils import sample_from_nested_dict, sample_state_data_set


def _get_action_term_raw_actions(env: ManagerBasedEnv, action_name: str) -> torch.Tensor:
    """Fetch raw action tensor from the action manager."""

    action_term = env.action_manager._terms.get(action_name)
    if action_term is None:
        raise ValueError(f"Action term '{action_name}' not found in action manager.")
    if not hasattr(action_term, "raw_actions"):
        raise ValueError(f"Action term '{action_name}' does not expose 'raw_actions'.")
    raw_actions = action_term.raw_actions
    if raw_actions is None:
        raise ValueError(f"Action term '{action_name}' has no 'raw_actions' set.")
    return raw_actions


def adversary_operational_space_controller_gains_from_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    adversary_action_name: str,
    osc_action_name: str,
    stiffness_scale_range: tuple[float, float],
    damping_scale_range: tuple[float, float],
    action_stiffness_index: int = 7,
    action_damping_index: int = 8,
) -> None:
    """Set OSC stiffness/damping gains from adversary action (reset-only).

    Uses two scalars from the adversary action vector:
        - stiffness_scale at action_stiffness_index
        - damping_scale at action_damping_index
    Both are clamped to the provided ranges and applied uniformly to xyz and rpy blocks.
    """

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    # Get OSC action term (the one that owns the controller instance).
    osc_term = env.action_manager._terms.get(osc_action_name)
    if osc_term is None:
        raise ValueError(f"Action term '{osc_action_name}' not found in action manager.")
    if not hasattr(osc_term, "_osc") or not hasattr(osc_term._osc, "cfg"):
        raise ValueError(f"Action term '{osc_action_name}' does not appear to be an operational space controller.")
    controller = osc_term._osc

    # Read adversary outputs.
    adv_raw = _get_action_term_raw_actions(env, adversary_action_name)
    a = adv_raw[env_ids]

    original_stiffness = torch.tensor(controller.cfg.motion_stiffness_task, device=env.device)
    original_damping = torch.tensor(controller.cfg.motion_damping_ratio_task, device=env.device)

    for i in range(len(env_ids)):
        env_id = env_ids[i : i + 1]
        stiff_scale = torch.clamp(a[i, action_stiffness_index], stiffness_scale_range[0], stiffness_scale_range[1])
        damp_scale = torch.clamp(a[i, action_damping_index], damping_scale_range[0], damping_scale_range[1])

        new_stiffness = torch.zeros((1, 6), device=env.device)
        new_stiffness[:, 0:3] = original_stiffness[0:3] * stiff_scale  # xyz
        new_stiffness[:, 3:6] = original_stiffness[3:6] * stiff_scale  # rpy

        controller._motion_p_gains_task[env_id] = torch.diag_embed(new_stiffness)
        controller._motion_p_gains_task[env_id] = (
            controller._selection_matrix_motion_task[env_id] @ controller._motion_p_gains_task[env_id]
        )

        new_damping_ratios = torch.zeros((1, 6), device=env.device)
        new_damping_ratios[:, 0:3] = original_damping[0:3] * damp_scale  # xyz
        new_damping_ratios[:, 3:6] = original_damping[3:6] * damp_scale  # rpy

        controller._motion_d_gains_task[env_id] = torch.diag_embed(
            2 * torch.diagonal(controller._motion_p_gains_task[env_id], dim1=-2, dim2=-1).sqrt() * new_damping_ratios
        )


def adversary_robot_material_from_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str,
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    num_buckets: int = 256,
    make_consistent: bool = True,
) -> None:
    """Set robot material params from adversary action (reset-only).

    Expected action layout (dim=9):
        [0] static_friction (absolute)
        [1] dynamic_friction (absolute)
        [2] robot_mass_scale
        [3] robot_joint_friction_scale
        [4] robot_joint_armature_scale
        [5] gripper_stiffness_scale
        [6] gripper_damping_scale
        [7] osc_stiffness_scale (used by a different reset event)
        [8] osc_damping_scale (used by a different reset event)
    """

    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_cpu = env_ids.cpu()
    env_ids_dev = env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]

    # Get the asset from the scene
    asset = env.scene[asset_cfg.name]

    # Clamp adversary-chosen absolute values (IsaacLab uses CPU tensors for materials).
    static_friction = torch.clamp(a[:, 0], static_friction_range[0], static_friction_range[1]).to("cpu")
    dynamic_friction = torch.clamp(a[:, 1], dynamic_friction_range[0], dynamic_friction_range[1]).to("cpu")
    if make_consistent:
        dynamic_friction = torch.minimum(dynamic_friction, static_friction)

    # Retrieve material buffer from the physics simulation and update.
    materials = asset.root_physx_view.get_material_properties()
    materials[env_ids_cpu, :, 0] = static_friction.view(-1, 1)
    materials[env_ids_cpu, :, 1] = dynamic_friction.view(-1, 1)
    materials[env_ids_cpu, :, 2] = 0.0  # restitution

    # Apply to simulation
    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)


def adversary_robot_mass_from_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str,
    asset_cfg: SceneEntityCfg,
    mass_scale_range: tuple[float, float],
    recompute_inertia: bool = True,
) -> None:
    """Set robot mass scaling from adversary action (reset-only).

    Expected action layout (dim=9):
        [2] robot_mass_scale
    """

    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_cpu = env_ids.cpu()
    env_ids_dev = env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]

    # Get the asset from the scene
    asset = env.scene[asset_cfg.name]

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current masses of the bodies (num_envs, num_bodies)
    masses = asset.root_physx_view.get_masses()

    # Reset to defaults, then apply per-env scaling (mirror IsaacLab randomize_rigid_body_mass).
    masses[env_ids_cpu[:, None], body_ids] = asset.data.default_mass[env_ids_cpu[:, None], body_ids].clone()
    mass_scale = torch.clamp(a[:, 2], mass_scale_range[0], mass_scale_range[1]).to("cpu")
    masses[env_ids_cpu[:, None], body_ids] *= mass_scale.view(-1, 1)
    masses = torch.clamp(masses, min=1e-6)

    # Set the mass into the physics simulation
    asset.root_physx_view.set_masses(masses, env_ids_cpu)

    # Recompute inertia tensors if needed
    if recompute_inertia:
        # Compute the ratios of the new masses to the default masses.
        ratios = masses[env_ids_cpu[:, None], body_ids] / asset.data.default_mass[env_ids_cpu[:, None], body_ids]

        # Scale the inertia tensors by the ratios (mirror IsaacLab).
        inertias = asset.root_physx_view.get_inertias()
        if isinstance(asset, Articulation):
            inertias[env_ids_cpu[:, None], body_ids] = asset.data.default_inertia[env_ids_cpu[:, None], body_ids] * ratios[
                ..., None
            ]
        else:
            # For rigid objects, inertia tensor is (num_envs, 9). Expect a single body.
            if ratios.ndim == 2 and ratios.shape[1] == 1:
                ratios = ratios[:, 0]
            inertias[env_ids_cpu] = asset.data.default_inertia[env_ids_cpu] * ratios.view(-1, 1)

        asset.root_physx_view.set_inertias(inertias, env_ids_cpu)


def adversary_robot_joint_parameters_from_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str,
    asset_cfg: SceneEntityCfg,
    friction_scale_range: tuple[float, float],
    armature_scale_range: tuple[float, float],
) -> None:
    """Set robot joint friction/armature scaling from adversary action (reset-only).

    Expected action layout (dim=9):
        [3] robot_joint_friction_scale
        [4] robot_joint_armature_scale
    """

    # Get the asset from the scene
    asset = env.scene[asset_cfg.name]
    env_ids = env_ids.to(asset.device)

    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_dev = env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]

    # Resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if env_ids != slice(None) and joint_ids != slice(None):
        env_ids_for_slice = env_ids[:, None]
    else:
        env_ids_for_slice = env_ids

    # Clamp adversary-chosen scale factors.
    friction_scale = torch.clamp(a[:, 3], friction_scale_range[0], friction_scale_range[1]).to(asset.device)
    armature_scale = torch.clamp(a[:, 4], armature_scale_range[0], armature_scale_range[1]).to(asset.device)

    # Mirror IsaacLab's randomize_joint_parameters write-paths.
    # -- joint friction coefficient
    friction_coeff = asset.data.default_joint_friction_coeff.clone()
    friction_coeff[env_ids_for_slice, joint_ids] *= friction_scale.view(-1, 1)
    friction_coeff = torch.clamp(friction_coeff, min=0.0)
    static_friction_coeff = friction_coeff[env_ids_for_slice, joint_ids]

    if get_isaac_sim_version().major >= 5:
        dynamic_friction_coeff = asset.data.default_joint_dynamic_friction_coeff.clone()
        viscous_friction_coeff = asset.data.default_joint_viscous_friction_coeff.clone()

        dynamic_friction_coeff[env_ids_for_slice, joint_ids] *= friction_scale.view(-1, 1)
        viscous_friction_coeff[env_ids_for_slice, joint_ids] *= friction_scale.view(-1, 1)

        dynamic_friction_coeff = torch.clamp(dynamic_friction_coeff, min=0.0)
        viscous_friction_coeff = torch.clamp(viscous_friction_coeff, min=0.0)

        # Ensure dynamic <= static (same shape before indexing).
        dynamic_friction_coeff = torch.minimum(dynamic_friction_coeff, friction_coeff)

        dynamic_friction_coeff = dynamic_friction_coeff[env_ids_for_slice, joint_ids]
        viscous_friction_coeff = viscous_friction_coeff[env_ids_for_slice, joint_ids]
    else:
        dynamic_friction_coeff = None
        viscous_friction_coeff = None

    asset.write_joint_friction_coefficient_to_sim(
        joint_friction_coeff=static_friction_coeff,
        joint_dynamic_friction_coeff=dynamic_friction_coeff,
        joint_viscous_friction_coeff=viscous_friction_coeff,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )

    # -- joint armature
    armature = asset.data.default_joint_armature.clone()
    armature[env_ids_for_slice, joint_ids] *= armature_scale.view(-1, 1)
    armature = torch.clamp(armature, min=0.0)
    asset.write_joint_armature_to_sim(armature[env_ids_for_slice, joint_ids], joint_ids=joint_ids, env_ids=env_ids)


def adversary_gripper_actuator_gains_from_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str,
    asset_cfg: SceneEntityCfg,
    stiffness_scale_range: tuple[float, float],
    damping_scale_range: tuple[float, float],
) -> None:
    """Set gripper actuator stiffness/damping scaling from adversary action (reset-only).

    Expected action layout (dim=9):
        [5] gripper_stiffness_scale
        [6] gripper_damping_scale
    """

    # Get the asset from the scene
    asset = env.scene[asset_cfg.name]
    env_ids = env_ids.to(asset.device)

    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_dev = env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]

    stiffness_scale = torch.clamp(a[:, 5], stiffness_scale_range[0], stiffness_scale_range[1]).to(asset.device)
    damping_scale = torch.clamp(a[:, 6], damping_scale_range[0], damping_scale_range[1]).to(asset.device)

    # Copy IsaacLab's actuator-loop structure from randomize_actuator_gains, but apply adversary scales.
    for actuator in asset.actuators.values():
        if isinstance(asset_cfg.joint_ids, slice):
            # We take all the joints of the actuator.
            actuator_indices = slice(None)
            if isinstance(actuator.joint_indices, slice):
                global_indices = slice(None)
            elif isinstance(actuator.joint_indices, torch.Tensor):
                global_indices = actuator.joint_indices.to(asset.device)
            else:
                raise TypeError("Actuator joint indices must be a slice or a torch.Tensor.")
        elif isinstance(actuator.joint_indices, slice):
            # We take the joints defined in the asset config.
            global_indices = actuator_indices = torch.tensor(asset_cfg.joint_ids, device=asset.device)
        else:
            # Take intersection of actuator joints and asset config joints.
            actuator_joint_indices = actuator.joint_indices
            asset_joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)
            actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
            if len(actuator_indices) == 0:
                continue
            global_indices = actuator_joint_indices[actuator_indices]

        # Stiffness: reset to defaults then scale per-env.
        stiffness = actuator.stiffness[env_ids].clone()
        stiffness[:, actuator_indices] = asset.data.default_joint_stiffness[env_ids][:, global_indices].clone()
        stiffness[:, actuator_indices] *= stiffness_scale.view(-1, 1)
        actuator.stiffness[env_ids] = stiffness
        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)

        # Damping: reset to defaults then scale per-env.
        damping = actuator.damping[env_ids].clone()
        damping[:, actuator_indices] = asset.data.default_joint_damping[env_ids][:, global_indices].clone()
        damping[:, actuator_indices] *= damping_scale.view(-1, 1)
        actuator.damping[env_ids] = damping
        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)


def adversary_insertive_object_material_from_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str,
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    num_buckets: int = 256,
    make_consistent: bool = True,
) -> None:
    """Set insertive object material params from adversary action (reset-only).

    Expected action layout (dim=15):
        [9] insertive_object static_friction (absolute)
        [10] insertive_object dynamic_friction (absolute)
    """

    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_cpu = env_ids.cpu()
    env_ids_dev = env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]

    # Get the asset from the scene
    asset: RigidObject = env.scene[asset_cfg.name]

    # Clamp adversary-chosen absolute values (IsaacLab uses CPU tensors for materials).
    static_friction = torch.clamp(a[:, 9], static_friction_range[0], static_friction_range[1]).to("cpu")
    dynamic_friction = torch.clamp(a[:, 10], dynamic_friction_range[0], dynamic_friction_range[1]).to("cpu")
    if make_consistent:
        dynamic_friction = torch.minimum(dynamic_friction, static_friction)

    # Retrieve material buffer from the physics simulation and update.
    materials = asset.root_physx_view.get_material_properties()
    materials[env_ids_cpu, :, 0] = static_friction.view(-1, 1)
    materials[env_ids_cpu, :, 1] = dynamic_friction.view(-1, 1)
    materials[env_ids_cpu, :, 2] = 0.0  # restitution

    # Apply to simulation
    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)


def adversary_receptive_object_material_from_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str,
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    num_buckets: int = 256,
    make_consistent: bool = True,
) -> None:
    """Set receptive object material params from adversary action (reset-only).

    Expected action layout (dim=15):
        [12] receptive_object static_friction (absolute)
        [13] receptive_object dynamic_friction (absolute)
    """

    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_cpu = env_ids.cpu()
    env_ids_dev = env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]

    # Get the asset from the scene
    asset: RigidObject = env.scene[asset_cfg.name]

    # Clamp adversary-chosen absolute values (IsaacLab uses CPU tensors for materials).
    static_friction = torch.clamp(a[:, 12], static_friction_range[0], static_friction_range[1]).to("cpu")
    dynamic_friction = torch.clamp(a[:, 13], dynamic_friction_range[0], dynamic_friction_range[1]).to("cpu")
    if make_consistent:
        dynamic_friction = torch.minimum(dynamic_friction, static_friction)

    # Retrieve material buffer from the physics simulation and update.
    materials = asset.root_physx_view.get_material_properties()
    materials[env_ids_cpu, :, 0] = static_friction.view(-1, 1)
    materials[env_ids_cpu, :, 1] = dynamic_friction.view(-1, 1)
    materials[env_ids_cpu, :, 2] = 0.0  # restitution

    # Apply to simulation
    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)


def adversary_insertive_object_mass_from_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str,
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float],
    recompute_inertia: bool = True,
) -> None:
    """Set insertive object mass from adversary action (reset-only).

    Uses absolute mass value (not a scale factor) since insertive objects
    can range from 20g to 200g.

    Expected action layout (dim=15):
        [11] insertive_object mass (absolute value in kg)
    """

    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_cpu = env_ids.cpu()
    env_ids_dev = env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]

    # Get the asset from the scene
    asset: RigidObject = env.scene[asset_cfg.name]

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current masses of the bodies (num_envs, num_bodies)
    masses = asset.root_physx_view.get_masses()

    # Apply absolute mass value from adversary action
    mass_value = torch.clamp(a[:, 11], mass_range[0], mass_range[1]).to("cpu")
    masses[env_ids_cpu[:, None], body_ids] = mass_value.view(-1, 1)
    masses = torch.clamp(masses, min=1e-6)

    # Set the mass into the physics simulation
    asset.root_physx_view.set_masses(masses, env_ids_cpu)

    # Recompute inertia tensors if needed
    if recompute_inertia:
        # Compute the ratios of the new masses to the default masses.
        ratios = masses[env_ids_cpu[:, None], body_ids] / asset.data.default_mass[env_ids_cpu[:, None], body_ids]

        # Scale the inertia tensors by the ratios.
        inertias = asset.root_physx_view.get_inertias()
        # For rigid objects, inertia tensor is (num_envs, 9). Expect a single body.
        if ratios.ndim == 2 and ratios.shape[1] == 1:
            ratios = ratios[:, 0]
        inertias[env_ids_cpu] = asset.data.default_inertia[env_ids_cpu] * ratios.view(-1, 1)

        asset.root_physx_view.set_inertias(inertias, env_ids_cpu)


def adversary_receptive_object_mass_from_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str,
    asset_cfg: SceneEntityCfg,
    mass_scale_range: tuple[float, float],
    recompute_inertia: bool = True,
) -> None:
    """Set receptive object mass scaling from adversary action (reset-only).

    Uses scale factor (not absolute value) to multiply default mass.

    Expected action layout (dim=15):
        [14] receptive_object mass scale factor
    """

    raw_actions = _get_action_term_raw_actions(env, action_name)
    env_ids_cpu = env_ids.cpu()
    env_ids_dev = env_ids.to(raw_actions.device)
    a = raw_actions[env_ids_dev]

    # Get the asset from the scene
    asset: RigidObject = env.scene[asset_cfg.name]

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current masses of the bodies (num_envs, num_bodies)
    masses = asset.root_physx_view.get_masses()

    # Reset to defaults, then apply per-env scaling.
    masses[env_ids_cpu[:, None], body_ids] = asset.data.default_mass[env_ids_cpu[:, None], body_ids].clone()
    mass_scale = torch.clamp(a[:, 14], mass_scale_range[0], mass_scale_range[1]).to("cpu")
    masses[env_ids_cpu[:, None], body_ids] *= mass_scale.view(-1, 1)
    masses = torch.clamp(masses, min=1e-6)

    # Set the mass into the physics simulation
    asset.root_physx_view.set_masses(masses, env_ids_cpu)

    # Recompute inertia tensors if needed
    if recompute_inertia:
        # Compute the ratios of the new masses to the default masses.
        ratios = masses[env_ids_cpu[:, None], body_ids] / asset.data.default_mass[env_ids_cpu[:, None], body_ids]

        # Scale the inertia tensors by the ratios.
        inertias = asset.root_physx_view.get_inertias()
        # For rigid objects, inertia tensor is (num_envs, 9). Expect a single body.
        if ratios.ndim == 2 and ratios.shape[1] == 1:
            ratios = ratios[:, 0]
        inertias[env_ids_cpu] = asset.data.default_inertia[env_ids_cpu] * ratios.view(-1, 1)

        asset.root_physx_view.set_inertias(inertias, env_ids_cpu)


class MultiResetManager(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        base_paths: list[str] = cfg.params.get("base_paths", [])
        probabilities: list[float] = cfg.params.get("probs", [])
        self.action_name: str | None = cfg.params.get("action_name", None)
        self.action_prob_start_idx: int = cfg.params.get("action_prob_start_idx", 9)
        self.min_prob: float = cfg.params.get("min_prob", 0.05)

        if not base_paths:
            raise ValueError("No base paths provided")
        if not probabilities and self.action_name is not None:
            probabilities = [1.0 / len(base_paths)] * len(base_paths)
        elif len(base_paths) != len(probabilities):
            raise ValueError("Number of base paths must match number of probabilities")

        insertive_usd_path = env.scene["insertive_object"].cfg.spawn.usd_path
        receptive_usd_path = env.scene["receptive_object"].cfg.spawn.usd_path
        reset_state_hash = utils.compute_assembly_hash(insertive_usd_path, receptive_usd_path)

        dataset_files = []
        for base_path in base_paths:
            dataset_files.append(f"{base_path}/{reset_state_hash}.pt")

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

        self.probs = torch.tensor(probabilities, device=env.device) / sum(probabilities)
        self.num_states = torch.tensor(num_states, device=env.device)
        self.num_tasks = len(self.datasets)

        if cfg.params.get("success") is not None:
            success_monitor_cfg = SuccessMonitorCfg(
                monitored_history_len=100, num_monitored_data=self.num_tasks, device=env.device
            )
            self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

        self.task_id = torch.randint(0, self.num_tasks, (self.num_envs,), device=self.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        base_paths: list[str],
        probs: list[float] = [],
        success: str | None = None,
        action_name: str | None = None,
        action_prob_start_idx: int = 9,
        min_prob: float = 0.05,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._env.device)

        if self.action_name is not None:
            raw_actions = _get_action_term_raw_actions(env, self.action_name)
            prob_actions = raw_actions[:, self.action_prob_start_idx:self.action_prob_start_idx + self.num_tasks]
            prob_actions_mean = prob_actions.mean(dim=0)
            if prob_actions_mean.abs().sum() < 1e-6:
                current_probs = self.probs.clone()
            else:
                current_probs = torch.softmax(prob_actions_mean, dim=0)
                self.probs = current_probs
            current_probs = torch.clamp(current_probs, min=self.min_prob)
            current_probs = current_probs / current_probs.sum()
        else:
            current_probs = torch.clamp(self.probs.clone(), min=self.min_prob)
            current_probs = current_probs / current_probs.sum()

        if success is not None:
            success_mask = torch.where(eval(success)[env_ids], 1.0, 0.0)
            self.success_monitor.success_update(self.task_id[env_ids], success_mask)

            success_rates = self.success_monitor.get_success_rate()
            if "log" not in self._env.extras:
                self._env.extras["log"] = {}
            for task_idx in range(self.num_tasks):
                self._env.extras["log"].update({
                    f"Metrics/task_{task_idx}_success_rate": success_rates[task_idx].item(),
                    f"Metrics/task_{task_idx}_prob": current_probs[task_idx].item(),
                    f"Metrics/task_{task_idx}_normalized_prob": current_probs[task_idx].item(),
                })

        dataset_indices = torch.multinomial(current_probs, len(env_ids), replacement=True)
        self.task_id[env_ids] = dataset_indices

        for dataset_idx in range(self.num_tasks):
            mask = dataset_indices == dataset_idx
            if not mask.any():
                continue

            current_env_ids = env_ids[mask]
            state_indices = torch.randint(
                0, self.num_states[dataset_idx], (len(current_env_ids),), device=self._env.device
            )  # type: ignore
            states_to_reset_from = sample_from_nested_dict(self.datasets[dataset_idx], state_indices)
            self._env.scene.reset_to(states_to_reset_from["initial_state"], env_ids=current_env_ids, is_relative=True)

        robot: Articulation = self._env.scene["robot"]
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel[env_ids]), env_ids=env_ids)


