#!/usr/bin/env bash
# Run OmniReset distilled-policy evals sequentially; each task runs NUM_REPEATS times.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CHECKPOINT="/home/paresh/cage/policies/leg_distilled_rgb.ckpt"
NUM_ENVS=32
NUM_TRAJ=100
NUM_REPEATS=1

CKPT_DIR="$(dirname "$CHECKPOINT")"
LOG_FILE="$CKPT_DIR/eval_$(basename "$CHECKPOINT" .ckpt).txt"
mkdir -p "$CKPT_DIR"
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
      env.scene.insertive_object=fbleg \
      env.scene.receptive_object=fbtabletop \
      # > /dev/null 2>&1
    sleep 5
  done
done

echo ""
echo "All eval runs finished."
