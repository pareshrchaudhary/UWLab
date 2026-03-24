# BC-PPO RGB Fine-Tuning: Current State

## Overview

Pure PPO fine-tuning of a behavior-cloning (BC) trained `MLPImagePolicy`. The BC policy acts as the actor (using RGB images), while a fresh MLP critic uses privileged simulation state. No expert model, no BC regularization loss.

**Training command:**

```bash
conda activate uwlab
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f85-BCPPO-RGB-v0 \
    --num_envs 8 \
    --max_iterations 1000 \
    --headless --enable_cameras
```

---

## Architecture

```
Observations (from Isaac Lab env)
  |
  +-- obs["policy"]  (TensorDict, not concatenated)
  |     |-- front_rgb:           (B, 4, 3, 224, 224)
  |     |-- side_rgb:            (B, 4, 3, 224, 224)
  |     |-- wrist_rgb:           (B, 4, 3, 224, 224)
  |     |-- arm_joint_pos:       (B, 4, 6)
  |     |-- end_effector_pose:   (B, 4, 6)
  |     |-- last_arm_action:     (B, 4, 6)
  |     +-- last_gripper_action: (B, 4, 1)
  |
  +-- obs["critic"]  (Tensor, concatenated)
        shape: (B, 243)

          BCImageActorCritic
         /                  \
   Actor (from BC ckpt)    Critic (fresh MLP)
   +-----------------+     +------------------+
   | obs_encoder     |     | MLP: 243 ->      |
   |  (ResNet18 x3)  |     |   512 -> 256 ->  |
   | trunk MLP       |     |   128 -> 1       |
   | mean_head       |     +------------------+
   | log_std_head    |
   +-----------------+
   Output: Normal(mean, std) over 7 actions
```

**Actor data flow:**
1. Extract `obs["policy"]` from full obs TensorDict.
2. Normalize with BC `LinearNormalizer` (scale+offset per key).
3. Reshape `(B, T=4, ...)` to `(B*T, ...)`, pass through `MultiImageObsEncoder` (ResNet18 per camera + low-dim concat).
4. Reshape encoder output to `(B, T * 1555)` = `(B, 6220)`.
5. Pass through `trunk` MLP -> `mean_head` / `log_std_head`.
6. Unnormalize action distribution from BC norm space to env space: `mean_env = (mean_norm - offset) / scale`.
7. PPO samples from `Normal(mean_env, std_env)`.

**Critic data flow:**
1. Extract `obs["critic"]` -- flat 243-dim vector of privileged state.
2. Pass through fresh MLP `243 -> [512, 256, 128] -> 1`.

---

## File Reference

### 1. `rsl_rl/rsl_rl/modules/bc_image_actor_critic.py`

The core actor-critic module.

| Setting | Value | Notes |
|---|---|---|
| `freeze_encoder` | `True` (default) | Only ResNet18 encoder frozen. Trunk + heads are trainable. |
| `n_obs_steps` | 4 | From BC checkpoint. Matches `history_length=4` in env config. |
| `obs_feature_dim` | 1555 | Per-timestep encoder output dim. |
| `trunk` input dim | 6220 | `4 * 1555` |
| `action_dim` | 7 | 6 arm + 1 gripper |
| `log_std_limits` | `(-5.0, 2.0)` | From BC checkpoint |
| Augmentations | Stripped | Only `Resize` + `Normalize` kept from BC encoder transforms. Isaac Lab handles domain randomization. |
| `bc_normalizer` | Frozen (`requires_grad=False`) | `LinearNormalizer` from BC checkpoint. Used for obs normalization and action unnormalization. |

**Key methods:**
- `act(obs)` -- extracts `obs["policy"]`, encodes, samples action. Called during both rollout and PPO update.
- `evaluate(obs)` -- extracts `obs["critic"]`, returns value estimate.
- `act_inference(obs)` -- deterministic (mean only), for evaluation.
- `train(mode)` -- overridden to keep encoder in `eval()` when `freeze_encoder=True`.

**When `freeze_encoder=True`:**
- Encoder parameters: `requires_grad=False`, permanently in `eval()` mode.
- Encoder forward runs under `torch.no_grad()`, output is `.detach()`ed.
- Trunk, `mean_head`, `log_std_head`: fully trainable with gradients.

**When `freeze_encoder=False`:**
- Everything is trainable. Requires `num_mini_batches >= 32` to fit in GPU memory due to image backward pass.

### 2. `source/uwlab_tasks/.../bc_ppo_rgb_cfg.py`

Environment observation configuration.

| Group | Type | Settings |
|---|---|---|
| `policy` | `BCPPOPolicyCfg` (inherits `RGBPolicyCfg`) | `concatenate_terms=False`, `history_length=4`, `flatten_history_dim=False`, `enable_corruption=True` |
| `critic` | `CriticCfg` | `concatenate_terms=True`, `history_length=1`, `enable_corruption=False` |

**Policy obs terms** (from `RGBPolicyCfg`): `front_rgb`, `side_rgb`, `wrist_rgb` (224x224), `arm_joint_pos`, `end_effector_pose`, `last_arm_action`, `last_gripper_action`.

**Critic obs terms** (243 dims total): `prev_actions(7)`, `joint_pos(14)`, `end_effector_pose(6)`, `insertive_asset_pose(6)`, `receptive_asset_pose(6)`, `insertive_in_receptive(6)`, `time_left(1)`, `joint_vel(14)`, `ee_vel(6)`, material properties (48+6+27+21), masses (16+1+1+1), joint params (14*4=56).

**Environment class:** `Ur5eRobotiq2f85BCPPORGBCfg` -- inherits full RGB scene (cameras, curtains, visual randomizations) from `Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg`.

### 3. `source/uwlab_tasks/.../agents/rsl_rl_cfg.py`

Training configuration (`Base_BCPPORunnerCfg`).

| Setting | Value |
|---|---|
| `num_steps_per_env` | 16 |
| `max_iterations` | 1000 |
| `save_interval` | 50 |
| `experiment_name` | `omnireset_bc_ppo_rgb` |
| `obs_groups` | `{"policy": ["policy"], "critic": ["critic"]}` |

**Policy config** (`RslRlBCImageActorCriticCfg`):

| Setting | Value |
|---|---|
| `bc_checkpoint_path` | `diffusion_policy/data/outputs/2026.03.20/18.10.35_.../step_0300002.ckpt` |
| `critic_hidden_dims` | `[512, 256, 128]` |
| `critic_activation` | `elu` |
| `freeze_encoder` | `True` |

**Algorithm config** (`RslRlPpoAlgorithmCfg`, standard PPO):

| Setting | Value |
|---|---|
| `clip_param` | 0.1 |
| `entropy_coef` | 0.01 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 16 |
| `learning_rate` | 1e-5 |
| `schedule` | adaptive |
| `desired_kl` | 0.005 |
| `gamma` | 0.99 |
| `lam` | 0.95 |
| `max_grad_norm` | 1.0 |

### 4. `source/uwlab_rl/uwlab_rl/rsl_rl/rl_cfg.py`

Defines `RslRlBCImageActorCriticCfg` dataclass with fields: `class_name="BCImageActorCritic"`, `bc_checkpoint_path`, `critic_hidden_dims`, `critic_activation`, `freeze_encoder`.

### 5. `rsl_rl/rsl_rl/storage/rollout_storage.py`

Standard `rl` mode rollout storage. Stores the full obs TensorDict (including nested `"policy"` sub-TensorDict with images). Added `_create_obs_buffer()` static method to recursively handle nested TensorDicts during buffer allocation.

### 6. `rsl_rl/rsl_rl/algorithms/__init__.py`

Exports `PPO`, `Distillation`, `SimplePPO`. No `BCPPO` (deleted).

### 7. `rsl_rl/rsl_rl/modules/__init__.py`

Exports `BCImageActorCritic` alongside `ActorCritic`, `ActorCriticRecurrent`, `AsymmetricActorCritic`, etc.

### 8. Gym registration (`__init__.py`)

```
id: OmniReset-Ur5eRobotiq2f85-BCPPO-RGB-v0
env_cfg: bc_ppo_rgb_cfg:Ur5eRobotiq2f85BCPPORGBCfg
agent_cfg: agents.rsl_rl_cfg:Base_BCPPORunnerCfg
```

---

## BC Checkpoint Details

| Property | Value |
|---|---|
| Path | `diffusion_policy/data/outputs/2026.03.20/18.10.35_omnireset_train_mlp_image_sim2real_image/checkpoints/step_0300002.ckpt` |
| Policy type | `MLPImagePolicy` |
| Encoder | `MultiImageObsEncoder` with ResNet18 backbone |
| Image keys | `front_rgb`, `side_rgb`, `wrist_rgb` (3x224x224) |
| Low-dim keys | `end_effector_pose(6)`, `arm_joint_pos(6)`, `last_arm_action(6)`, `last_gripper_action(1)` |
| `n_obs_steps` | 4 |
| `n_action_steps` | 1 |
| `obs_feature_dim` | 1555 per timestep |
| Trunk input | 6220 (4 * 1555) |
| Action dim | 7 |
| Normalizer | `LinearNormalizer` (`x_norm = x * scale + offset`) |
| Original augmentations | `ColorJitter`, `GaussianBlur`, `RandomGrayscale`, `GaussianNoise` (stripped at load time) |

---

## What is Trainable

| Component | `freeze_encoder=True` | `freeze_encoder=False` |
|---|---|---|
| `obs_encoder` (ResNet18 x3 + projection) | Frozen | Trainable |
| `trunk` (MLP 6220 -> hidden) | Trainable | Trainable |
| `mean_head` (hidden -> 7) | Trainable | Trainable |
| `log_std_head` (hidden -> 7) | Trainable | Trainable |
| `bc_normalizer` | Always frozen | Always frozen |
| `critic` MLP | Trainable | Trainable |

---

## GPU Memory Notes

- With `freeze_encoder=True` and `num_mini_batches=16`: fits on a single RTX 4090 (24GB) with 8 envs.
- With `freeze_encoder=False`: need `num_mini_batches >= 32` due to backward pass through image encoder.
- Image storage: each env step stores 3 images x 4 timesteps x 3x224x224 floats per env. With 8 envs x 16 steps = 128 transitions, this is ~18.5 GB for images alone in the rollout buffer.
