# Pipeline for running VolPy on a remote SLURM cluster via SSH

Finalized/cleaned pipeline workspace for running a local notebook that submits jobs to a remote SLURM cluster over SSH.

Contents:
- `notebooks/`: notebooks used from the local machine
- `utils/`: lightweight helper utilities for SSH + SLURM submission (new, pipeline-specific)

SLURM runner:
- `Pipeline_Final/utils/run_python_job.sh`: generic `sbatch` wrapper that forwards arbitrary python args
  - `sbatch run_python_job.sh my_step.py --flag value`
  - `sbatch run_python_job.sh /path/to/workdir my_step.py --flag value`
  - Override defaults with env vars: `PIPELINE_WORKDIR`, `CONDA_ENV_NAME`
