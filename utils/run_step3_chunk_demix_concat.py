#!/usr/bin/env python
"""Run Step 3 (chunk -> demix -> concat) sequentially on cluster.

This script is designed to be submitted once via SLURM and orchestrates:
  1) utils/chunk_video_by_ROI.py
  2) utils/demix_chunk.py for each chunk movie
  3) utils/concat_chunk_results.py

It exits non-zero if any dataset fails.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple


class DatasetStatus(object):
    def __init__(self, data_file):
        self.data_file = data_file
        self.chunk_ok = False
        self.n_chunks = 0
        self.n_demix_ok = 0
        self.n_demix_fail = 0
        self.concat_ok = False
        self.failed = False
        self.failure_reason = ""


_SBATCH_JOB_ID_RE = re.compile(r"Submitted batch job\s+(\d+)")
_RUNNING_STATES = {
    "PENDING",
    "RUNNING",
    "COMPLETING",
    "CONFIGURING",
    "SUSPENDED",
    "RESIZING",
    "STAGE_OUT",
    "SIGNALING",
}


def _log(prefix: str, msg: str) -> None:
    if prefix:
        print(f"[{prefix}] {msg}", flush=True)
    else:
        print(msg, flush=True)


def _run_cmd(cmd: List[str], *, cwd: Path, prefix: str) -> None:
    _log(prefix, "RUN: " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _discover_chunk_videos(data_folder: Path) -> List[Path]:
    chunks_dir = data_folder / "chunks"
    chunk_dirs = [p for p in chunks_dir.glob("chunk_*") if p.is_dir()]
    if not chunk_dirs:
        return []

    def _chunk_key(p: Path) -> Tuple[int, str]:
        m = re.search(r"chunk_(\d+)$", p.name)
        if m:
            return int(m.group(1)), p.name
        return 10**9, p.name

    chunk_dirs = sorted(chunk_dirs, key=_chunk_key)
    out = []
    for d in chunk_dirs:
        tif = d / "movReg_denoised.tif"
        if tif.exists():
            out.append(tif)
    return out


def _run_demix_for_chunks(
    chunk_videos: List[Path],
    *,
    pipeline_workdir: Path,
    demix_launch_mode: str,
    demix_mem_gb: int,
    demix_poll_seconds: int,
    max_chunks_parallel: int,
    prefix: str,
) -> Tuple[int, int]:
    if not chunk_videos:
        return 0, 0

    if str(demix_launch_mode).lower() == "sbatch":
        return _run_demix_for_chunks_sbatch(
            chunk_videos,
            pipeline_workdir=pipeline_workdir,
            demix_mem_gb=int(demix_mem_gb),
            demix_poll_seconds=int(max(1, demix_poll_seconds)),
            prefix=prefix,
        )

    return _run_demix_for_chunks_local(
        chunk_videos,
        pipeline_workdir=pipeline_workdir,
        max_chunks_parallel=max(1, int(max_chunks_parallel)),
        prefix=prefix,
    )


def _run_demix_for_chunks_local(
    chunk_videos: List[Path],
    *,
    pipeline_workdir: Path,
    max_chunks_parallel: int,
    prefix: str,
) -> Tuple[int, int]:
    py = sys.executable
    demix_script = pipeline_workdir / "utils" / "demix_chunk.py"
    if max_chunks_parallel <= 1:
        ok = 0
        fail = 0
        for cv in chunk_videos:
            try:
                _run_cmd([py, "-u", str(demix_script), str(cv)], cwd=pipeline_workdir, prefix=prefix)
                ok += 1
            except subprocess.CalledProcessError:
                fail += 1
        return ok, fail

    def _run_one(cv: Path) -> None:
        _run_cmd(
            [py, "-u", str(demix_script), str(cv)],
            cwd=pipeline_workdir,
            prefix=prefix,
        )

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=max_chunks_parallel) as ex:
        futs = {ex.submit(_run_one, cv): cv for cv in chunk_videos}
        for fut in as_completed(futs):
            cv = futs[fut]
            try:
                _ = fut.result()
                _log(prefix, f"[STEP3B] Demix OK: {cv}")
                ok += 1
            except Exception as e:
                _log(prefix, f"[STEP3B] Demix FAIL: {cv} ({e!r})")
                fail += 1
    return ok, fail


def _wait_for_slurm_jobs(job_ids: List[str], *, poll_seconds: int, prefix: str) -> Tuple[int, int]:
    if not job_ids:
        return 0, 0

    remaining = set(str(j) for j in job_ids)
    terminal_states = {}
    poll_seconds = int(max(1, poll_seconds))

    while remaining:
        ids_str = ",".join(sorted(remaining))
        cp = subprocess.run(
            ["sacct", "-j", ids_str, "--format=JobID,State,ExitCode", "-n", "-P"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if cp.returncode != 0:
            _log(prefix, f"[WARN] sacct failed while waiting for demix jobs: {cp.stderr.strip()}")
            time.sleep(poll_seconds)
            continue

        seen_any = False
        for line in cp.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            jid = parts[0].strip()
            if not jid or "." in jid:
                continue
            if jid not in remaining:
                continue
            seen_any = True
            state = parts[1].strip().upper()
            if not state or state in _RUNNING_STATES:
                continue
            terminal_states[jid] = state
            remaining.remove(jid)

        if not seen_any and remaining:
            # Accounting can lag slightly; keep polling.
            time.sleep(poll_seconds)
            continue
        if remaining:
            time.sleep(poll_seconds)

    ok = 0
    fail = 0
    for jid in job_ids:
        st = terminal_states.get(str(jid), "").upper()
        if st.startswith("COMPLETED"):
            ok += 1
        else:
            fail += 1
    return ok, fail


def _run_demix_for_chunks_sbatch(
    chunk_videos: List[Path],
    *,
    pipeline_workdir: Path,
    demix_mem_gb: int,
    demix_poll_seconds: int,
    prefix: str,
) -> Tuple[int, int]:
    logs_dir = pipeline_workdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_script = pipeline_workdir / "utils" / "run_python_job.sh"
    demix_script_rel = "utils/demix_chunk.py"

    if not run_script.exists():
        _log(prefix, f"[STEP3B] FAIL: Missing runner script {run_script}")
        return 0, len(chunk_videos)

    submitted_ids = []
    submit_fail = 0
    for cv in chunk_videos:
        cmd = [
            "sbatch",
            "--job-name",
            "demix_chunk_orch",
            "--mem={}G".format(int(max(1, demix_mem_gb))),
            "--chdir",
            str(pipeline_workdir),
            "--output",
            str(logs_dir / "%x_%j.out"),
            "--error",
            str(logs_dir / "%x_%j.err"),
            str(run_script),
            str(pipeline_workdir),
            demix_script_rel,
            str(cv),
        ]
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        out = (cp.stdout or "").strip()
        err = (cp.stderr or "").strip()
        if cp.returncode != 0:
            submit_fail += 1
            _log(prefix, f"[STEP3B] Demix submit FAIL: {cv} | {err or out}")
            continue
        m = _SBATCH_JOB_ID_RE.search(out)
        if not m:
            submit_fail += 1
            _log(prefix, f"[STEP3B] Demix submit FAIL (no job id): {cv} | {out} {err}")
            continue
        jid = m.group(1)
        submitted_ids.append(jid)
        _log(prefix, f"[STEP3B] Demix submitted: {cv} -> job {jid}")

    if not submitted_ids:
        return 0, submit_fail

    _log(prefix, f"[STEP3B] Waiting for {len(submitted_ids)} demix jobs...")
    wait_ok, wait_fail = _wait_for_slurm_jobs(
        submitted_ids,
        poll_seconds=demix_poll_seconds,
        prefix=prefix,
    )
    return int(wait_ok), int(wait_fail + submit_fail)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Step 3 chunk->demix->concat on cluster.")
    p.add_argument(
        "--data-file",
        action="append",
        required=True,
        help="Path to one denoised movie (repeat for multiple datasets).",
    )
    p.add_argument(
        "--pipeline-workdir",
        default=os.getcwd(),
        help="Pipeline root directory (default: current working directory).",
    )
    p.add_argument("--dilation", type=int, default=10, help="Dilation for chunk_video_by_ROI.py")
    p.add_argument(
        "--reuse-existing-chunks",
        dest="reuse_existing_chunks",
        action="store_true",
        help="If chunk videos already exist, skip chunking and reuse them (default).",
    )
    p.add_argument(
        "--no-reuse-existing-chunks",
        dest="reuse_existing_chunks",
        action="store_false",
        help="Always run chunking even if existing chunks are found.",
    )
    p.set_defaults(reuse_existing_chunks=True)
    p.add_argument(
        "--demix-mem-gb",
        type=int,
        default=64,
        help="Demix job memory in GB when demix-launch-mode=sbatch.",
    )
    p.add_argument(
        "--demix-launch-mode",
        choices=["sbatch", "local"],
        default="sbatch",
        help="How to run demix chunks: sbatch (simultaneous cluster jobs) or local.",
    )
    p.add_argument(
        "--demix-poll-seconds",
        type=int,
        default=10,
        help="Polling interval in seconds while waiting for demix sbatch jobs.",
    )
    p.add_argument(
        "--max-chunks-parallel",
        type=int,
        default=1,
        help="Used only when demix-launch-mode=local.",
    )
    p.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Strict mode: any demix chunk failure skips concat for that dataset.",
    )
    p.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Non-strict mode: continue to concat if at least one demix chunk succeeds.",
    )
    p.set_defaults(strict=True)
    p.add_argument("--log-prefix", default="", help="Optional log prefix.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pipeline_workdir = Path(args.pipeline_workdir).resolve()
    py = sys.executable

    chunk_script = pipeline_workdir / "utils" / "chunk_video_by_ROI.py"
    concat_script = pipeline_workdir / "utils" / "concat_chunk_results.py"

    _log(args.log_prefix, f"[INFO] Python: {py}")
    _log(args.log_prefix, f"[INFO] Workdir: {pipeline_workdir}")
    _log(
        args.log_prefix,
        f"[INFO] Params: dilation={args.dilation}, reuse_existing_chunks={args.reuse_existing_chunks}, "
        f"demix_mem_gb={args.demix_mem_gb}, "
        f"demix_launch_mode={args.demix_launch_mode}, "
        f"max_chunks_parallel={args.max_chunks_parallel}, strict={args.strict}",
    )

    statuses: List[DatasetStatus] = []

    for data_file in args.data_file:
        st = DatasetStatus(data_file=data_file)
        statuses.append(st)
        data_path = Path(data_file)
        data_folder = data_path.parent
        ds_name = data_folder.name
        pref = f"{args.log_prefix}:{ds_name}" if args.log_prefix else ds_name

        _log(pref, f"[DATASET] Start: {data_file}")

        # STEP 3A: chunking (or reuse existing chunks)
        chunk_videos = _discover_chunk_videos(data_folder)
        if args.reuse_existing_chunks and chunk_videos:
            st.chunk_ok = True
            _log(
                pref,
                f"[STEP3A] Reusing existing chunks: found {len(chunk_videos)} chunk videos; skipping chunking.",
            )
        else:
            try:
                _run_cmd(
                    [py, "-u", str(chunk_script), str(data_file), "--dilation", str(int(args.dilation))],
                    cwd=pipeline_workdir,
                    prefix=pref,
                )
                st.chunk_ok = True
            except subprocess.CalledProcessError as e:
                st.failed = True
                st.failure_reason = f"chunk failed ({e.returncode})"
                _log(pref, f"[STEP3A] FAIL: {st.failure_reason}")
                continue
            chunk_videos = _discover_chunk_videos(data_folder)

        st.n_chunks = len(chunk_videos)
        if st.n_chunks <= 0:
            st.failed = True
            st.failure_reason = "no chunk videos found after chunking"
            _log(pref, f"[STEP3B] FAIL: {st.failure_reason}")
            continue
        _log(pref, f"[STEP3B] Found {st.n_chunks} chunks")

        # STEP 3B: demix each chunk
        n_ok, n_fail = _run_demix_for_chunks(
            chunk_videos,
            pipeline_workdir=pipeline_workdir,
            demix_launch_mode=str(args.demix_launch_mode),
            demix_mem_gb=int(args.demix_mem_gb),
            demix_poll_seconds=int(args.demix_poll_seconds),
            max_chunks_parallel=max(1, int(args.max_chunks_parallel)),
            prefix=pref,
        )
        st.n_demix_ok = int(n_ok)
        st.n_demix_fail = int(n_fail)
        _log(pref, f"[STEP3B] Demix summary: ok={n_ok}, fail={n_fail}")

        if args.strict and n_fail > 0:
            st.failed = True
            st.failure_reason = f"demix failures in strict mode (fail={n_fail})"
            _log(pref, f"[STEP3B] FAIL: {st.failure_reason}")
            continue
        if n_ok <= 0:
            st.failed = True
            st.failure_reason = "no successful demix chunks"
            _log(pref, f"[STEP3B] FAIL: {st.failure_reason}")
            continue

        # STEP 3C: concat
        try:
            _run_cmd([py, "-u", str(concat_script), str(data_folder)], cwd=pipeline_workdir, prefix=pref)
            st.concat_ok = True
        except subprocess.CalledProcessError as e:
            st.failed = True
            st.failure_reason = f"concat failed ({e.returncode})"
            _log(pref, f"[STEP3C] FAIL: {st.failure_reason}")
            continue

        _log(pref, "[DATASET] SUCCESS")

    # Final summary
    _log(args.log_prefix, "")
    _log(args.log_prefix, "========== STEP 3 SUMMARY ==========")
    for st in statuses:
        ds_name = Path(st.data_file).parent.name
        _log(
            args.log_prefix,
            (
                f"{ds_name}: chunk_ok={st.chunk_ok}, chunks={st.n_chunks}, "
                f"demix_ok={st.n_demix_ok}, demix_fail={st.n_demix_fail}, "
                f"concat_ok={st.concat_ok}, failed={st.failed}"
                + (f", reason={st.failure_reason}" if st.failure_reason else "")
            ),
        )

    any_failed = any(st.failed for st in statuses)
    if any_failed:
        _log(args.log_prefix, "[RESULT] FAILED")
        return 1
    _log(args.log_prefix, "[RESULT] SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
