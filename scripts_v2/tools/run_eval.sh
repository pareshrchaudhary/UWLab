#!/usr/bin/env bash
# Run OmniReset distilled-policy evals sequentially; each task runs NUM_REPEATS times.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# CHECKPOINT="/home/paresh/UWLab/diffusion_policy/data/outputs/2026.03.20/18.10.35_omnireset_train_mlp_image_sim2real_image/checkpoints/step_0300002.ckpt"
CHECKPOINT="/home/paresh/UWLab/diffusion_policy/data/outputs/2026.03.19/20.20.45_cage_train_mlp_image_sim2real_image/checkpoints/step_0120001.ckpt"
NUM_ENVS=32
NUM_TRAJ=100
NUM_REPEATS=5

LOG_DIR="$(dirname "$(dirname "$CHECKPOINT")")"
STEP="$(basename "$CHECKPOINT" .ckpt)"
LOG_FILE="${LOG_DIR}/eval_${STEP}.txt"
mkdir -p "$LOG_DIR"
: >"$LOG_FILE"

TASKS=(
  OmniReset-Eval-RGB
  OmniReset-Eval-RGB-OOD-OSC
  OmniReset-Eval-RGB-OOD-Robot
  OmniReset-Eval-RGB-OOD-Object
  OmniReset-Eval-RGB-OOD-All
  OmniReset-Eval-OOD-RGB
  OmniReset-Eval-OOD-RGB-OOD-All
)

for task in "${TASKS[@]}"; do
  for i in $(seq 1 "$NUM_REPEATS"); do
    echo ""
    echo "================================================================================"
    echo "Task: $task  (run $i/$NUM_REPEATS)"
    echo "================================================================================"
    python scripts_v2/tools/eval_distilled_policy.py \
      --task "$task" \
      --checkpoint "$CHECKPOINT" \
      --num_envs "$NUM_ENVS" \
      --num_trajectories "$NUM_TRAJ" \
      --headless \
      --enable_cameras \
      --eval_summary_log "$LOG_FILE" \
      > /dev/null 2>&1
    sleep 5
  done
done

echo ""
echo "All eval runs finished."
