# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset states tasks for IsaacLab."""

import gymnasium as gym

from . import agents


# Register SysID env
gym.register(
    id="Cage-Ur5eRobotiq2f85-Sysid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sysid_cfg:SysidEnvCfg"},
)

# Register Camera Alignment env
gym.register(
    id="Cage-Ur5eRobotiq2f85-CameraAlign-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.camera_align_cfg:CameraAlignEnvCfg"},
)

# Register RL state environments

# =============================================================================
# Adversary Base
# =============================================================================

_omnireset = f"{__name__}.rl_state_omnireset"

gym.register(
    id="Cage-Ur5eRobotiq2f85-AdversaryBase",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85AdversaryTrainCfg",
        "eval_ObjectAnywhereEEAnywhere_cfg_entry_point": f"{_omnireset}:Ur5eRobotiq2f85RelCartesianOSCEvalCfg",
        "eval_ObjectRestingEEGrasped_cfg_entry_point": f"{_omnireset}:Ur5eRobotiq2f85RelCartesianOSCEvalObjectRestingEEGraspedCfg",
        "eval_ObjectAnywhereEEGrasped_cfg_entry_point": f"{_omnireset}:Ur5eRobotiq2f85RelCartesianOSCEvalObjectAnywhereEEGraspedCfg",
        "eval_ObjectPartiallyAssembledEEGrasped_cfg_entry_point": f"{_omnireset}:Ur5eRobotiq2f85RelCartesianOSCEvalObjectPartiallyAssembledEEGraspedCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AdversaryBaseRunner",
    },
)


# =============================================================================
# CAGE AdversaryAdvancedEventCfg environments
# =============================================================================

gym.register(
    id="Cage-Ur5eRobotiq2f85-AdversaryAdvanced",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85AdversaryAdvancedTrainCfg",
        "eval_ObjectAnywhereEEAnywhere_cfg_entry_point": f"{_omnireset}:Ur5eRobotiq2f85RelCartesianOSCEvalCfg",
        "eval_ObjectRestingEEGrasped_cfg_entry_point": f"{_omnireset}:Ur5eRobotiq2f85RelCartesianOSCEvalObjectRestingEEGraspedCfg",
        "eval_ObjectAnywhereEEGrasped_cfg_entry_point": f"{_omnireset}:Ur5eRobotiq2f85RelCartesianOSCEvalObjectAnywhereEEGraspedCfg",
        "eval_ObjectPartiallyAssembledEEGrasped_cfg_entry_point": f"{_omnireset}:Ur5eRobotiq2f85RelCartesianOSCEvalObjectPartiallyAssembledEEGraspedCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AdversaryAdvancedRunner",
    },
)

gym.register(
    id="Cage-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_omnireset:Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg",
        "eval_env_cfg_entry_point": f"{__name__}.rl_state_omnireset:Ur5eRobotiq2f85RelCartesianOSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="Cage-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_omnireset:Ur5eRobotiq2f85RelCartesianOSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Single-variant state eval configs (multi-variant bench uses State-Play-v0 + apply_multi_eval_events)
gym.register(
    id="Cage-Ur5eRobotiq2f85-RelCartesianOSC-State-EvalCfgV0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eval_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalCfgV0",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)
gym.register(
    id="Cage-Ur5eRobotiq2f85-RelCartesianOSC-State-EvalCfgV1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eval_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalCfgV1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)
gym.register(
    id="Cage-Ur5eRobotiq2f85-RelCartesianOSC-State-EvalCfgV2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eval_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalCfgV2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)
gym.register(
    id="Cage-Ur5eRobotiq2f85-RelCartesianOSC-State-EvalCfgV3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eval_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalCfgV3",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="Cage-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_omnireset:Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)


# RGB environments for data collection and evaluation
gym.register(
    id="Cage-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DataCollection-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="Cage-Ur5eRobotiq2f85-RelCartesianOSC-RGB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

# Aliases and OOD physics eval RGB (cage-clean registry)
gym.register(
    id="Cage-Eval-RGB",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)
gym.register(
    id="Cage-Eval-RGB-OOD-OSC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eval_rgb_cfg:RGBOODOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)
gym.register(
    id="Cage-Eval-RGB-OOD-Robot",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eval_rgb_cfg:RGBOODRobotCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)
gym.register(
    id="Cage-Eval-RGB-OOD-Object",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eval_rgb_cfg:RGBOODObjectCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)
gym.register(
    id="Cage-Eval-RGB-OOD-All",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eval_rgb_cfg:RGBOODAllCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

# OOD (out-of-distribution) RGB environments
gym.register(
    id="Cage-Ur5eRobotiq2f85-RelCartesianOSC-RGB-OOD-DataCollection-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCOODCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="Cage-Ur5eRobotiq2f85-RelCartesianOSC-RGB-OOD-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85EvalRGBRelCartesianOSCOODCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="Cage-Eval-OOD-RGB",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85EvalRGBRelCartesianOSCOODCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)
gym.register(
    id="Cage-Eval-OOD-RGB-OOD-All",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eval_rgb_cfg:OODRGBOODAllCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)





