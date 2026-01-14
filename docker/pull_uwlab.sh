#!/bin/bash
# Helper script to pull uwlab image with correct cache directory

# Get the project root directory (two levels up from this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

export APPTAINER_CACHEDIR="$PROJECT_ROOT/apptainer/cache"
export APPTAINER_TMPDIR="$PROJECT_ROOT/apptainer/tmp"

mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
mkdir -p "$PROJECT_ROOT/UWLab/uwlab_docker"

echo "Using cache directory: $APPTAINER_CACHEDIR"
apptainer pull "$PROJECT_ROOT/UWLab/uwlab_docker/uw-lab_latest.sif" docker://pareshrchaudhary/uwlab:latest

