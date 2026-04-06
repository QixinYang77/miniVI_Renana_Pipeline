"""Helpers for discovering FLIR AVI files and submitting cluster conversion jobs."""

from __future__ import annotations

import os
import re
import shlex
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

try:
    from IPython.display import clear_output
except ImportError:  # pragma: no cover
    def clear_output(*args, **kwargs) -> None:
        return None

if TYPE_CHECKING:
    import paramiko


LOCAL_CLUSTER_MOUNT = "/Volumes/adam-lab"
REMOTE_CLUSTER_MOUNT = "/ems/elsc-labs/adam-y"
DEFAULT_CLUSTER_HOST = "loginserver.elsc.huji.ac.il"
DEFAULT_USERNAME = "qixin.yang"
DEFAULT_SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_rsa")
DEFAULT_PIPELINE_WORKDIR = "/ems/elsc-labs/adam-y/qixin.yang/ClusterCode/miniVI_Renana_Pipeline"
DEFAULT_CONDA_ENV = "caiman"
DEFAULT_OUTPUT_DIR = (
    "/Volumes/adam-lab/Adam-Lab-Shared/Data/renana_malka/All_behavior_videos_mp4"
)
DEFAULT_LOG_DIR = f"{DEFAULT_PIPELINE_WORKDIR}/miscellaneous/logs_ds_behav"
DEFAULT_RUNNER_SCRIPT = f"{DEFAULT_PIPELINE_WORKDIR}/utils/run_python_job.sh"
DEFAULT_CONVERSION_SCRIPT = "miscellaneous/convert_flir_avi_to_mp4.py"
ANIMAL_PATTERN = re.compile(r"(PRL|PR|PX)")
CAGE_PATTERN = re.compile(r"(pAce\d+)")
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class VideoJob:
    local_avi_path: str
    cluster_avi_path: str
    local_output_path: str
    cluster_output_path: str
    cage: str
    animal: str
    date: str
    session: str

    @property
    def output_basename(self) -> str:
        return Path(self.local_output_path).name

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def local_to_cluster_path(path: str | Path) -> str:
    return str(path).replace(LOCAL_CLUSTER_MOUNT, REMOTE_CLUSTER_MOUNT, 1)


def cluster_to_local_path(path: str | Path) -> str:
    return str(path).replace(REMOTE_CLUSTER_MOUNT, LOCAL_CLUSTER_MOUNT, 1)


def _extract_cage(parts: Sequence[str]) -> str:
    for part in parts:
        match = CAGE_PATTERN.search(part)
        if match:
            return match.group(1)
    raise ValueError(f"Could not infer cage from path parts: {parts}")


def _extract_animal(parts: Sequence[str]) -> str:
    for part in reversed(parts):
        match = ANIMAL_PATTERN.fullmatch(part) or ANIMAL_PATTERN.search(part)
        if match:
            return match.group(1)
    raise ValueError(f"Could not infer animal from path parts: {parts}")


def _extract_date(parts: Sequence[str]) -> str:
    for part in parts:
        match = DATE_PATTERN.search(part)
        if match:
            return match.group(1)
    raise ValueError(f"Could not infer date from path parts: {parts}")


def build_output_name(avi_path: str | Path) -> str:
    avi = Path(avi_path)
    parts = avi.parts
    cage = _extract_cage(parts)
    animal = _extract_animal(parts)
    date = _extract_date(parts)
    session = avi.parent.name.replace("-", "_")
    return f"{cage}_{animal}_{date}_{session}.mp4"


def make_video_job(
    avi_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> VideoJob:
    avi = Path(avi_path)
    output_name = build_output_name(avi)
    local_output_path = str(Path(output_dir) / output_name)
    return VideoJob(
        local_avi_path=str(avi),
        cluster_avi_path=local_to_cluster_path(avi),
        local_output_path=local_output_path,
        cluster_output_path=local_to_cluster_path(local_output_path),
        cage=_extract_cage(avi.parts),
        animal=_extract_animal(avi.parts),
        date=_extract_date(avi.parts),
        session=avi.parent.name.replace("-", "_"),
    )


def discover_flir_avi_jobs(
    roots: Iterable[str | Path],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[VideoJob]:
    jobs: list[VideoJob] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Data folder does not exist: {root_path}")
        good_dirs = sorted(
            path for path in root_path.iterdir() if path.is_dir() and "-good" in path.name
        )
        for good_dir in good_dirs:
            avi_paths = sorted(good_dir.glob("Flir*.avi"))
            for avi_path in avi_paths:
                jobs.append(make_video_job(avi_path, output_dir=output_dir))
    return jobs


def print_job_summary(jobs: Sequence[VideoJob]) -> None:
    print(f"Discovered {len(jobs)} AVI file(s).")
    for idx, job in enumerate(jobs, start=1):
        print(
            f"{idx:02d}. {job.local_avi_path}\n"
            f"    -> {job.local_output_path}"
        )


def connect_ssh(
    cluster_host: str = DEFAULT_CLUSTER_HOST,
    username: str = DEFAULT_USERNAME,
    ssh_key_path: str | None = DEFAULT_SSH_KEY_PATH,
) -> "paramiko.SSHClient":
    import paramiko

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs = {}
    if ssh_key_path and os.path.exists(ssh_key_path):
        kwargs["key_filename"] = ssh_key_path
    ssh.connect(cluster_host, username=username, **kwargs)
    return ssh


def run_remote_command(ssh: "paramiko.SSHClient", command: str) -> tuple[str, str]:
    stdin, stdout, stderr = ssh.exec_command(f"bash -l -c {shlex.quote(command)}")
    del stdin
    output = stdout.read().decode().strip()
    error = stderr.read().decode().strip()
    return output, error


def ensure_remote_directories(
    ssh: "paramiko.SSHClient",
    directories: Iterable[str | Path],
) -> None:
    quoted = " ".join(shlex.quote(str(directory)) for directory in directories)
    _, error = run_remote_command(ssh, f"mkdir -p {quoted}")
    if error:
        raise RuntimeError(f"Could not create remote directories: {error}")


def build_sbatch_command(
    job: VideoJob,
    *,
    pipeline_workdir: str = DEFAULT_PIPELINE_WORKDIR,
    conda_env: str = DEFAULT_CONDA_ENV,
    runner_script: str = DEFAULT_RUNNER_SCRIPT,
    conversion_script: str = DEFAULT_CONVERSION_SCRIPT,
    log_dir: str = DEFAULT_LOG_DIR,
    ffmpeg_binary: str = "ffmpeg",
    time_limit: str = "08:00:00",
    cpus: int = 2,
    mem_gb: int = 8,
    overwrite: bool = False,
    crf: int = 17,
    preset: str = "slow",
) -> str:
    overwrite_flag = "--overwrite" if overwrite else ""
    command_parts = [
        "sbatch",
        "--job-name",
        f"flir_mp4_{job.cage}_{job.session}",
        "-c",
        str(cpus),
        "--mem",
        f"{mem_gb}G",
        "-t",
        time_limit,
        "--export",
        f"ALL,CONDA_ENV_NAME={conda_env}",
        "--chdir",
        pipeline_workdir,
        "--output",
        f"{log_dir}/%x_%j.out",
        "--error",
        f"{log_dir}/%x_%j.err",
        runner_script,
        pipeline_workdir,
        conversion_script,
        "--input-video",
        job.cluster_avi_path,
        "--output-video",
        job.cluster_output_path,
        "--ffmpeg-binary",
        ffmpeg_binary,
        "--crf",
        str(crf),
        "--preset",
        preset,
    ]
    if overwrite_flag:
        command_parts.append(overwrite_flag)
    return " ".join(shlex.quote(part) for part in command_parts)


def submit_conversion_jobs(
    ssh: "paramiko.SSHClient",
    jobs: Sequence[VideoJob],
    *,
    pipeline_workdir: str = DEFAULT_PIPELINE_WORKDIR,
    conda_env: str = DEFAULT_CONDA_ENV,
    runner_script: str = DEFAULT_RUNNER_SCRIPT,
    conversion_script: str = DEFAULT_CONVERSION_SCRIPT,
    log_dir: str = DEFAULT_LOG_DIR,
    output_dir: str = local_to_cluster_path(DEFAULT_OUTPUT_DIR),
    ffmpeg_binary: str = "ffmpeg",
    time_limit: str = "08:00:00",
    cpus: int = 2,
    mem_gb: int = 8,
    overwrite: bool = False,
    crf: int = 17,
    preset: str = "slow",
    dry_run: bool = False,
) -> list[dict[str, str | None]]:
    ensure_remote_directories(ssh, [log_dir, output_dir])

    results: list[dict[str, str | None]] = []
    for job in jobs:
        command = build_sbatch_command(
            job,
            pipeline_workdir=pipeline_workdir,
            conda_env=conda_env,
            runner_script=runner_script,
            conversion_script=conversion_script,
            log_dir=log_dir,
            ffmpeg_binary=ffmpeg_binary,
            time_limit=time_limit,
            cpus=cpus,
            mem_gb=mem_gb,
            overwrite=overwrite,
            crf=crf,
            preset=preset,
        )
        if dry_run:
            results.append({"job_id": None, "command": command, "output": "", "error": ""})
            continue

        output, error = run_remote_command(ssh, command)
        match = re.search(r"Submitted batch job (\d+)", output)
        results.append(
            {
                "job_id": match.group(1) if match else None,
                "command": command,
                "output": output,
                "error": error,
            }
        )
    return results


def wait_for_jobs(
    ssh: "paramiko.SSHClient",
    job_ids: Sequence[str],
    poll_interval: int = 60,
) -> bool:
    if not job_ids:
        print("No jobs to wait for.")
        return True

    pending_jobs = set(job_ids)
    failed_jobs: list[tuple[str, str]] = []
    while pending_jobs:
        job_list = ",".join(sorted(pending_jobs))
        output, error = run_remote_command(
            ssh, f"sacct -j {job_list} --format=JobID,State,ExitCode -n -P"
        )
        if error and "Invalid job id" not in error:
            print(f"[WARNING] {error}")

        completed: set[str] = set()
        for line in output.splitlines():
            if not line or "." in line.split("|")[0]:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            job_id, state = parts[0], parts[1]
            if state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}:
                completed.add(job_id)
                if state != "COMPLETED":
                    failed_jobs.append((job_id, state))

        pending_jobs -= completed
        clear_output(wait=True)
        print(f"[{time.strftime('%H:%M:%S')}] Jobs status:")
        print(f"  Completed: {len(job_ids) - len(pending_jobs)}/{len(job_ids)}")
        print(f"  Pending:   {len(pending_jobs)}")
        if failed_jobs:
            print(f"  Failed:    {failed_jobs}")
        if pending_jobs:
            print(f"\nWaiting {poll_interval}s before next check...")
            time.sleep(poll_interval)

    if failed_jobs:
        print(f"WARNING: {len(failed_jobs)} job(s) failed: {failed_jobs}")
        return False
    print("All jobs completed successfully.")
    return True
