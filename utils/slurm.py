import re
import time
from typing import Iterable, List, Optional

from .ssh import run_ssh


_SBATCH_JOB_ID_RE = re.compile(r"Submitted batch job\\s+(\\d+)")


def sbatch(ssh_client, command: str) -> int:
    """
    Submits an `sbatch ...` command and returns the numeric job id.
    """
    code, out, err = run_ssh(ssh_client, command)
    if code != 0:
        raise RuntimeError(f"sbatch failed (exit {code}): {err.strip() or out.strip()}")
    m = _SBATCH_JOB_ID_RE.search(out)
    if not m:
        raise RuntimeError(f"Could not parse job id from sbatch output: {out.strip()}")
    return int(m.group(1))


def wait_for_jobs(
    ssh_client,
    job_ids: Iterable[int],
    *,
    poll_interval_s: int = 10,
    timeout_s: Optional[int] = None,
) -> None:
    """
    Polls SLURM until all jobs in `job_ids` are no longer present in squeue.
    """
    job_ids = [int(j) for j in job_ids]
    remaining = set(job_ids)
    start = time.time()

    while remaining:
        if timeout_s is not None and (time.time() - start) > timeout_s:
            raise TimeoutError(f"Timed out waiting for jobs: {sorted(remaining)}")

        ids_str = ",".join(str(j) for j in sorted(remaining))
        code, out, err = run_ssh(ssh_client, f"squeue -h -j {ids_str} -o '%i'")
        if code != 0:
            raise RuntimeError(f"squeue failed (exit {code}): {err.strip() or out.strip()}")

        present = {int(line.strip()) for line in out.splitlines() if line.strip().isdigit()}
        remaining = remaining.intersection(present)

        if remaining:
            time.sleep(poll_interval_s)
