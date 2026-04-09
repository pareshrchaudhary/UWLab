#!/bin/bash
# Get the script and project root directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
UWLAB_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PROJECT_ROOT="$( cd "$UWLAB_ROOT/.." && pwd )"

# uwlab_docker contains volumes/caches
UW_BASE="$UWLAB_ROOT/uwlab_docker"
SCRUBBED_BASE="$PROJECT_ROOT"
mkdir -p ${UW_BASE}/isaac-cache-kit \
         ${UW_BASE}/isaac-cache-ov \
         ${UW_BASE}/isaac-cache-pip \
         ${UW_BASE}/isaac-cache-gl \
         ${UW_BASE}/isaac-cache-compute \
         ${UW_BASE}/isaac-logs \
         ${UW_BASE}/isaac-carb-logs \
         ${UW_BASE}/isaac-data \
         ${UW_BASE}/isaac-docs \
         ${UW_BASE}/uw-lab-docs \
         ${UW_BASE}/uw-lab-logs \
         ${UW_BASE}/uw-lab-data \
         ${UW_BASE}/outputs \
         ${PROJECT_ROOT}/datasets

# Create bash history file if it doesn't exist
touch ${SCRIPT_DIR}/.uw-lab-docker-history

# rsl_rl: monorepo layout (sibling of UWLab) or checkout under UWLab
if [[ -d "${PROJECT_ROOT}/rsl_rl" ]]; then
  RSL_RL_HOST="${PROJECT_ROOT}/rsl_rl"
elif [[ -d "${UWLAB_ROOT}/rsl_rl" ]]; then
  RSL_RL_HOST="${UWLAB_ROOT}/rsl_rl"
else
  echo "ERROR: rsl_rl not found. Clone it to ${PROJECT_ROOT}/rsl_rl (next to UWLab)" >&2
  echo "       or ${UWLAB_ROOT}/rsl_rl" >&2
  exit 1
fi

# apptainer exec --nv  --writable-tmpfs \
#   --bind ${UW_BASE}/isaac-cache-kit:/isaac-sim/kit/cache \
#   --bind ${UW_BASE}/isaac-cache-ov:/root/.cache/ov \
#   --bind ${UW_BASE}/isaac-cache-pip:/root/.cache/pip \
#   --bind ${UW_BASE}/isaac-cache-gl:/root/.cache/nvidia/GLCache \
#   --bind ${UW_BASE}/isaac-cache-compute:/root/.nv/ComputeCache \
#   --bind ${UW_BASE}/logs:/workspace/uwlab/logs \
#   --bind ${UW_BASE}/outputs:/workspace/uwlab/outputs \
#   --bind ${UW_BASE}/data_storage:/workspace/uwlab/data_storage \
#   --bind ${SCRUBBED_BASE}:/workspace/uwlab/scrubbed \
#   --bind /etc/pki/ca-trust:/etc/pki/ca-trust:ro \
#   --pwd /workspace/uwlab \
#   ${UW_BASE}/uw-lab_latest.sif \
#   bash --noprofile --norc


NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0}" apptainer exec --nv --writable-tmpfs \
  --bind ${UW_BASE}/isaac-cache-kit:/isaac-sim/kit/cache \
  --bind ${UW_BASE}/isaac-cache-ov:/root/.cache/ov \
  --bind ${UW_BASE}/isaac-cache-pip:/root/.cache/pip \
  --bind ${UW_BASE}/isaac-cache-gl:/root/.cache/nvidia/GLCache \
  --bind ${UW_BASE}/isaac-cache-compute:/root/.nv/ComputeCache \
  --bind ${UW_BASE}/isaac-logs:/root/.nvidia-omniverse/logs \
  --bind ${UW_BASE}/isaac-carb-logs:/isaac-sim/kit/logs/Kit/Isaac-Sim \
  --bind ${UW_BASE}/isaac-data:/root/.local/share/ov/data \
  --bind ${UW_BASE}/isaac-docs:/root/Documents \
  --bind ${UWLAB_ROOT}/source:/workspace/uwlab/source \
  --bind ${UWLAB_ROOT}/scripts:/workspace/uwlab/scripts \
  --bind ${UWLAB_ROOT}/scripts_v2:/workspace/uwlab/scripts_v2 \
  --bind ${UWLAB_ROOT}/docs:/workspace/uwlab/docs \
  --bind ${UWLAB_ROOT}/tools:/workspace/uwlab/tools \
  --bind ${RSL_RL_HOST}:/workspace/uwlab/rsl_rl \
  --bind ${PROJECT_ROOT}/datasets:/workspace/uwlab/datasets \
  --bind ${UW_BASE}/uw-lab-docs:/workspace/uwlab/docs/_build \
  --bind ${UW_BASE}/uw-lab-logs:/workspace/uwlab/logs \
  --bind ${UW_BASE}/uw-lab-data:/workspace/uwlab/data_storage \
  --bind ${UW_BASE}/outputs:/workspace/uwlab/outputs \
  --bind ${SCRUBBED_BASE}:/workspace/uwlab/scrubbed \
  --bind ${SCRIPT_DIR}/.uw-lab-docker-history:/root/.bash_history \
  --bind /etc/pki/ca-trust:/etc/pki/ca-trust:ro \
  --pwd /workspace/uwlab \
  ${UW_BASE}/uw-lab_latest.sif \
  bash --noprofile --norc