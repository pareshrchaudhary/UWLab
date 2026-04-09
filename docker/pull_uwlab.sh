#!/bin/bash
# Helper script to pull uwlab image with correct cache directory

# Get the project root directory (two levels up from this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Layer cache stays on persistent storage (survives node changes, re-pulls)
export APPTAINER_CACHEDIR="$PROJECT_ROOT/apptainer/cache"

# TMPDIR is where Apptainer extracts OCI layers and builds the SquashFS.
# This is small-file-IO-heavy, so it MUST be on fast local storage.
# GPFS (/mmfs1) is a network FS and makes extraction 10-100x slower.
# Prefer /dev/shm (RAM disk, fastest); fall back to /tmp (local NVMe).
if [[ -d /dev/shm ]] && [[ $(df -B1 /dev/shm | awk 'NR==2 {print $4}') -gt 30000000000 ]]; then
  export APPTAINER_TMPDIR="/dev/shm/apptainer-tmp-$USER"
elif [[ -d /tmp ]]; then
  export APPTAINER_TMPDIR="/tmp/apptainer-tmp-$USER"
else
  export APPTAINER_TMPDIR="$PROJECT_ROOT/apptainer/tmp"
fi

mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
mkdir -p "$PROJECT_ROOT/UWLab/uwlab_docker"

echo "Using cache directory: $APPTAINER_CACHEDIR"
echo "Using tmp directory:   $APPTAINER_TMPDIR"
apptainer pull --force "$PROJECT_ROOT/UWLab/uwlab_docker/uw-lab_latest.sif" docker://pareshrchaudhary/uwlab:latest

# Clean up the temp dir — it can be 2-3x the final .sif size
rm -rf "$APPTAINER_TMPDIR"

