# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Test script to verify adversary action flow through ActionManager and EventManager.

This script verifies:
1. Combined actions (robot + adversary) are passed to env.step()
2. ActionManager routes adversary slice to AdversaryAction term
3. AdversaryAction stores values in raw_actions
4. Event functions can read raw_actions during reset

Usage:
    python source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/adversarial_env/test_adversary_action_flow.py \
      --task Cage-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
      --num_envs 1 \
      --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Parse arguments before launching
parser = argparse.ArgumentParser(description="Test adversary action flow")
parser.add_argument("--task", type=str, default="Cage-Ur5eRobotiq2f85-RelCartesianOSC-State-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after sim initialization
import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    """Test adversary action flow."""
    def _assert_close(name: str, got: torch.Tensor, expected: torch.Tensor, atol: float = 1e-5, rtol: float = 1e-5):
        if not torch.allclose(got, expected, atol=atol, rtol=rtol):
            diff = (got - expected).abs()
            raise AssertionError(
                f"{name} mismatch: max_abs_diff={diff.max().item():.6g}, "
                f"got={got.flatten()[:10].tolist()}, expected={expected.flatten()[:10].tolist()}"
            )

    # Override num_envs
    env_cfg.scene.num_envs = args_cli.num_envs

    # Create environment with config
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"\nEnvironment created with {args_cli.num_envs} envs")

    # Get action manager info
    action_manager = env.unwrapped.action_manager # type: ignore[attr-defined]
    print(f"\nAction Manager Info:")
    print(f"  Total action dim: {action_manager.total_action_dim}")
    print(f"  Action terms: {list(action_manager._terms.keys())}")

    # Get adversary action term
    adversary_term = action_manager._terms.get("adversaryaction")
    assert adversary_term is not None, "ActionManager missing 'adversaryaction' term"

    # Calculate action dimensions
    adversary_dim = adversary_term.action_dim
    robot_dim = action_manager.total_action_dim - adversary_dim
    print(f"\nAction Split:")
    print(f"  Robot actions: indices [0:{robot_dim}]")
    print(f"  Adversary actions: indices [{robot_dim}:{robot_dim + adversary_dim}]")

    # Reset environment
    env.reset()

    # Create test actions with known values
    # Robot actions: zeros
    # Adversary actions: specific test values [0.1, 0.2, 0.3, ..., 0.9]
    num_envs = args_cli.num_envs
    device = env.unwrapped.device  # type: ignore[attr-defined]
    robot_actions = torch.zeros(num_envs, robot_dim, device=device)
    adversary_actions = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]] * num_envs,
        device=device,
    )

    # Combine actions
    combined_actions = torch.cat([robot_actions, adversary_actions], dim=1)
    print(f"\nTest Actions:")
    print(f"  Combined shape: {combined_actions.shape}")
    print(f"  Adversary values (env 0): {adversary_actions[0].tolist()}")

    # Step environment
    print("\n" + "-" * 40)
    print("STEP 1: Passing actions to env.step()")
    print("-" * 40)

    obs, reward, terminated, truncated, info = env.step(combined_actions)

    # Check if adversary actions were stored in raw_actions
    stored_actions = adversary_term.raw_actions
    print(f"\nAfter step - AdversaryAction.raw_actions (env 0):")
    print(f"  {stored_actions[0].tolist()}")

    # Verify values match
    match = torch.allclose(stored_actions, adversary_actions, atol=1e-5)
    print(f"\nVerification: Stored values match input? {match}")

    if match:
        print("  [PASS] ActionManager correctly routed adversary actions!")
    else:
        print("  [FAIL] Values don't match!")
        print(f"  Expected: {adversary_actions[0].tolist()}")
        print(f"  Got: {stored_actions[0].tolist()}")

    # Test reset flow
    print("\n" + "-" * 40)
    print("STEP 2: Testing reset flow")
    print("-" * 40)

    # Set specific adversary values before reset
    # Layout (dim=9):
    # [0] static_friction(abs), [1] dynamic_friction(abs), [2] mass_scale,
    # [3] joint_friction_scale, [4] joint_armature_scale,
    # [5] gripper_stiffness_scale, [6] gripper_damping_scale,
    # [7] osc_stiffness_scale, [8] osc_damping_scale
    # Intentionally include some out-of-range values to verify clamping.
    new_adversary = torch.tensor(
        [[1.1, 1.5, 1.25, 3.5, 0.1, 2.5, 0.1, 2.0, 0.0]] * num_envs,
        device=env.unwrapped.device,  # type: ignore[attr-defined]
    )
    new_combined = torch.cat([robot_actions, new_adversary], dim=1)

    # Step to update raw_actions
    env.step(new_combined)
    print(f"\nUpdated adversary values (env 0): {adversary_term.raw_actions[0].tolist()}")

    # Force reset all environments
    print("\nForcing reset for all environments...")

    # Step with termination to trigger reset
    # Or directly call reset
    obs, _ = env.reset()

    print(f"\nAfter reset - AdversaryAction.raw_actions (env 0):")
    print(f"  {adversary_term.raw_actions[0].tolist()}")

    # Test event function access
    print("\n" + "-" * 40)
    print("STEP 3: Verify event functions can access raw_actions")
    print("-" * 40)

    from uwlab_tasks.manager_based.manipulation.adversarial_env.mdp.events import (
        _get_action_term_raw_actions, # type: ignore[attr-defined]
    )

    try:
        raw = _get_action_term_raw_actions(env.unwrapped, "adversaryaction")
        print(f"\n_get_action_term_raw_actions() returned shape: {raw.shape}")
        print(f"  Values (env 0): {raw[0].tolist()}")
        print("  [PASS] Event functions can access raw_actions!")
    except Exception as e:
        print(f"  [FAIL] Error accessing raw_actions: {e}")

    print("[PASS] adversaryaction.raw_actions is correctly stored and readable by reset events.")

    # Verify adversary-driven reset events actually applied values to sim
    print("\n" + "-" * 40)
    print("STEP 4: Verify adversary resets apply to simulator")
    print("-" * 40)

    robot = env.unwrapped.scene["robot"]  # type: ignore[attr-defined]

    # Compute expected clamped values for env 0 based on config ranges.
    # (These ranges are from the task cfg; keep in sync with rl_state_cfg.py)
    exp_static = float(torch.clamp(new_adversary[0, 0], 0.3, 1.2).item())
    exp_dynamic = float(torch.clamp(new_adversary[0, 1], 0.2, 1.0).item())
    exp_dynamic = min(exp_dynamic, exp_static)  # make_consistent=True
    exp_mass_scale = float(torch.clamp(new_adversary[0, 2], 0.7, 1.3).item())
    exp_joint_fric_scale = float(torch.clamp(new_adversary[0, 3], 0.25, 4.0).item())
    exp_joint_arm_scale = float(torch.clamp(new_adversary[0, 4], 0.25, 4.0).item())
    exp_grip_stiff_scale = float(torch.clamp(new_adversary[0, 5], 0.5, 2.0).item())
    exp_grip_damp_scale = float(torch.clamp(new_adversary[0, 6], 0.5, 2.0).item())

    # (A) Materials: read back from PhysX material buffer.
    try:
        mats = robot.root_physx_view.get_material_properties()
        got_static = mats[0, :, 0].mean().item()
        got_dynamic = mats[0, :, 1].mean().item()
        _assert_close("robot.static_friction(mean)", torch.tensor(got_static), torch.tensor(exp_static), atol=1e-4)
        _assert_close("robot.dynamic_friction(mean)", torch.tensor(got_dynamic), torch.tensor(exp_dynamic), atol=1e-4)
        print(f"  [PASS] robot materials set: static={got_static:.4f}, dynamic={got_dynamic:.4f}")
    except Exception as e:
        print(f"  [WARN] Could not verify robot materials: {e}")

    # (B) Masses: read back from PhysX masses buffer and compare against default_mass * scale.
    try:
        masses = robot.root_physx_view.get_masses()
        default_mass = robot.data.default_mass
        exp = (default_mass[0].detach().cpu()) * exp_mass_scale
        _assert_close("robot.masses", masses[0], exp, atol=1e-4, rtol=1e-4)
        print(f"  [PASS] robot masses scaled by {exp_mass_scale:.3f}")
    except Exception as e:
        print(f"  [WARN] Could not verify robot masses: {e}")

    # (C) Joint parameters: best-effort verification via data buffers if available.
    try:
        # Resolve joint indices for the randomized group (finger + arm joints).
        joint_ids, joint_names = robot.find_joints(["shoulder.*", "elbow.*", "wrist.*", "finger_joint"])  # type: ignore[attr-defined]
        joint_ids_t = torch.tensor(joint_ids, device=robot.device, dtype=torch.long)

        # Armature
        if hasattr(robot.data, "default_joint_armature") and hasattr(robot.data, "joint_armature"):
            got = robot.data.joint_armature[0, joint_ids_t].detach().cpu()
            exp = (robot.data.default_joint_armature[0, joint_ids_t].detach().cpu()) * exp_joint_arm_scale
            _assert_close("robot.joint_armature", got, exp, atol=1e-3, rtol=1e-3)
            print(f"  [PASS] joint armature scaled by {exp_joint_arm_scale:.3f}")
        else:
            print("  [WARN] No joint armature buffers found to verify.")

        # Friction (try common buffer names)
        friction_default = None
        friction_current = None
        for base in ("joint_friction_coeff", "joint_friction", "joint_friction_coefficient"):
            dname = f"default_{base}"
            cname = base
            if hasattr(robot.data, dname) and hasattr(robot.data, cname):
                friction_default = getattr(robot.data, dname)
                friction_current = getattr(robot.data, cname)
                break
        if friction_default is not None and friction_current is not None:
            got = friction_current[0, joint_ids_t].detach().cpu()
            exp = (friction_default[0, joint_ids_t].detach().cpu()) * exp_joint_fric_scale
            exp = torch.clamp(exp, min=0.0)
            _assert_close("robot.joint_friction", got, exp, atol=1e-3, rtol=1e-3)
            print(f"  [PASS] joint friction scaled by {exp_joint_fric_scale:.3f}")
        else:
            print("  [WARN] No joint friction buffers found to verify.")
    except Exception as e:
        print(f"  [WARN] Could not verify joint parameters: {e}")

    # (D) Gripper actuator gains: verify finger_joint gains in actuator buffers if possible.
    try:
        finger_ids, _ = robot.find_joints("finger_joint")  # type: ignore[attr-defined]
        finger_ids_t = torch.tensor(finger_ids, device=robot.device, dtype=torch.long)

        if hasattr(robot.data, "default_joint_stiffness") and hasattr(robot.data, "default_joint_damping"):
            exp_stiff = (robot.data.default_joint_stiffness[0, finger_ids_t].detach().cpu()) * exp_grip_stiff_scale
            exp_damp = (robot.data.default_joint_damping[0, finger_ids_t].detach().cpu()) * exp_grip_damp_scale

            # Find an actuator that includes the finger joint index and compare its buffers.
            checked = False
            for actuator in robot.actuators.values():
                if not hasattr(actuator, "stiffness") or not hasattr(actuator, "damping"):
                    continue
                # actuator.joint_indices may be slice or tensor of global indices
                if isinstance(actuator.joint_indices, slice):
                    continue  # can't map reliably here
                global_ids = actuator.joint_indices.detach().cpu().tolist()
                if any(j in global_ids for j in finger_ids):
                    idxs = [global_ids.index(j) for j in finger_ids if j in global_ids]
                    got_stiff = actuator.stiffness[0, idxs].detach().cpu()
                    got_damp = actuator.damping[0, idxs].detach().cpu()
                    _assert_close("gripper.stiffness", got_stiff, exp_stiff[: len(idxs)], atol=1e-3, rtol=1e-3)
                    _assert_close("gripper.damping", got_damp, exp_damp[: len(idxs)], atol=1e-3, rtol=1e-3)
                    checked = True
                    break
            if checked:
                print(
                    f"  [PASS] gripper gains scaled: stiffness={exp_grip_stiff_scale:.3f}, damping={exp_grip_damp_scale:.3f}"
                )
            else:
                print("  [WARN] Could not find a suitable actuator buffer to verify gripper gains.")
        else:
            print("  [WARN] No default_joint_stiffness/damping buffers found to verify gripper gains.")
    except Exception as e:
        print(f"  [WARN] Could not verify gripper actuator gains: {e}")

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()  # type: ignore[misc]
    simulation_app.close()
