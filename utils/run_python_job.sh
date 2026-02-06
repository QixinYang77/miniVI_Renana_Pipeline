#!/bin/bash
# Generic SLURM runner for Python scripts.
#
# Usage:
#   sbatch run_python_job.sh <python_script> [args...]
#   sbatch run_python_job.sh <workdir> <python_script> [args...]
#
# Notes:
# - Pass job resources (name/output/etc.) via sbatch flags when submitting if needed:
#     sbatch --job-name myjob --output ./%x_%j.out --error ./%x_%j.err run_python_job.sh ...

#SBATCH -p elscn.q
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH --mem=128GB
#SBATCH --job-name generic_python_job
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Error: Missing arguments."
    echo "Usage:"
    echo "  sbatch $0 <python_script> [args...]"
    echo "  sbatch $0 <workdir> <python_script> [args...]"
    exit 1
fi

DEFAULT_WORKDIR="/ems/elsc-labs/adam-y/qixin.yang/ClusterCode/miniVI_Renana_Pipeline/Pipeline_Final"
WORKDIR="${PIPELINE_WORKDIR:-$DEFAULT_WORKDIR}"

if [ $# -ge 2 ] && [ -d "$1" ]; then
    WORKDIR="$1"
    shift
fi

PY_SCRIPT="$1"
shift || true

echo "Starting Python: ${PY_SCRIPT}"
echo "Workdir: ${WORKDIR}"
echo "Args: $*"

cd "$WORKDIR"
export PYTHONPATH="$WORKDIR${PYTHONPATH:+:$PYTHONPATH}"

if [ -f "$HOME/.bashrc" ]; then
    # Some login scripts reference unset variables; temporarily disable nounset.
    set +u
    . "$HOME/.bashrc"
    set -u
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-caiman}"
if [ -f "$HOME/anaconda3/bin/activate" ]; then
    # Conda activation scripts can reference unset variables (e.g., binutils activate.d);
    # with `set -u` this would abort the job. Disable nounset just for activation.
    set +u
    . "$HOME/anaconda3/bin/activate" "$CONDA_ENV_NAME"
    set -u
else
    echo "Warning: conda activate script not found at $HOME/anaconda3/bin/activate"
fi

echo "Current working directory: $(pwd)"
echo "Python executable: $(command -v python)"
python -V

python -u "$PY_SCRIPT" "$@"

echo "Job finished."
