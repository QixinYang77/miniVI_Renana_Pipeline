#!/usr/bin/env python3
"""Build the all-animal refined CKII cluster input bundle."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent.parent.resolve()
_top_repo = str(HERE.parent)
_top_repo_norm = os.path.normpath(_top_repo)
sys.path[:] = [
    p for p in sys.path
    if os.path.normpath(os.path.abspath(p or os.getcwd())) != _top_repo_norm
]
sys.path.insert(0, str(HERE))
os.environ["PYTHONPATH"] = str(HERE)

from notebooks_HPC.egocentric_refined_config import (
    ANIMALS,
    CLUSTER_REFINED_BUNDLE_FILENAME,
    MANUAL_REFINED_SIDECAR_FILENAME,
    REFINED_BEHAVIOR_FILENAME,
    REFINED_CLUSTER_INPUT_SCHEMA_VERSION,
    refined_analysis_params,
    refined_parameter_snapshot,
)
from utils.placecell_pipeline import (
    REFINED_ANALYSIS_DATA_KEY,
    build_refined_analysis_data_from_manual_sidecar,
)

SLIM_DROP_KEYS = (
    "spike_heights_interpolated",
    "manual_refined_manual_exclusion_masks",
    "manual_refined_snr_cutoff_masks",
)
SLIM_FLOAT32_KEYS = (
    "x_neural",
    "y_neural",
    "speed",
    "hd_angles_neural",
    "ts_neural",
)
SLIM_EVENT_TRACE_KEYS = (
    "trace",
    "trace_bl_subtracted",
    "trace_mf",
    "trace_lp",
    "fitted_baseline",
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_record(path: Path) -> dict[str, Any]:
    exists = path.exists()
    rec: dict[str, Any] = {
        "path": str(path),
        "exists": bool(exists),
        "is_symlink": bool(path.is_symlink()),
    }
    if path.is_symlink():
        try:
            rec["link_target"] = os.readlink(path)
        except OSError as exc:
            rec["link_target_error"] = str(exc)
    if exists:
        stat = path.stat()
        rec.update(
            {
                "resolved_path": str(path.resolve()),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return rec


def _load_sidecar_refined(animal_dir: Path) -> dict[str, Any]:
    sidecar_path = animal_dir / MANUAL_REFINED_SIDECAR_FILENAME
    with sidecar_path.open("rb") as f:
        sidecar = pickle.load(f)
    if not isinstance(sidecar, dict):
        raise ValueError(f"Manual sidecar is not a dict: {sidecar_path}")
    refined = sidecar.get(REFINED_ANALYSIS_DATA_KEY)
    if not isinstance(refined, dict):
        raise ValueError(f"Missing {REFINED_ANALYSIS_DATA_KEY!r} in {sidecar_path}")
    return refined


def _build_refined_payload(animal_dir: Path, *, rebuild_refined: bool) -> dict[str, Any]:
    if rebuild_refined:
        analysis = refined_analysis_params()
        return build_refined_analysis_data_from_manual_sidecar(
            animal_dir,
            merged_data_filename=REFINED_BEHAVIOR_FILENAME,
            sidecar_filename=MANUAL_REFINED_SIDECAR_FILENAME,
            apply_cb_baseline_removal=analysis.refined_apply_cb_baseline_removal,
            cb_baseline_window_s=analysis.refined_cb_baseline_window_s,
            snr_cb_baseline_window_s=analysis.refined_snr_cb_baseline_window_s,
        )
    return _load_sidecar_refined(animal_dir)


def _array_nbytes_recursive(value: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    oid = id(value)
    if oid in seen:
        return 0
    seen.add(oid)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_array_nbytes_recursive(k, seen) + _array_nbytes_recursive(v, seen) for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(_array_nbytes_recursive(v, seen) for v in value)
    return 0


def _strip_event_trace_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if str(k) not in SLIM_EVENT_TRACE_KEYS}
    if isinstance(value, list):
        return [_strip_event_trace_fields(v) if isinstance(v, dict) else v for v in value]
    if isinstance(value, tuple):
        return tuple(_strip_event_trace_fields(v) if isinstance(v, dict) else v for v in value)
    return value


def slim_refined_payload_for_cluster(refined: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove duplicated/nonessential arrays from a refined payload for cluster transfer."""
    before_bytes = _array_nbytes_recursive(refined)
    slimmed = dict(refined)
    dropped: dict[str, int] = {}

    for key in SLIM_DROP_KEYS:
        if key in slimmed:
            dropped[key] = _array_nbytes_recursive(slimmed[key])
            slimmed.pop(key, None)

    converted_to_float32: list[str] = []
    for key in SLIM_FLOAT32_KEYS:
        if key not in slimmed:
            continue
        arr = np.asarray(slimmed[key], dtype=np.float32).reshape(-1)
        slimmed[key] = arr
        converted_to_float32.append(key)

    stripped_event_dicts: list[str] = []
    for key in ("complex_bursts_dicts", "plateaus_dicts"):
        if key in slimmed:
            slimmed[key] = _strip_event_trace_fields(slimmed[key])
            stripped_event_dicts.append(key)

    after_bytes = _array_nbytes_recursive(slimmed)
    report = {
        "enabled": True,
        "profile": "refined_cluster_slim_v1",
        "array_bytes_before": int(before_bytes),
        "array_bytes_after": int(after_bytes),
        "array_bytes_saved": int(before_bytes - after_bytes),
        "dropped_keys": dropped,
        "float32_keys": converted_to_float32,
        "event_trace_keys_removed": list(SLIM_EVENT_TRACE_KEYS),
        "event_dict_keys_slimmed": stripped_event_dicts,
    }
    slimmed["cluster_export_slimming"] = report
    return slimmed, report


def _validate_refined_payload(animal_id: str, refined: dict[str, Any]) -> dict[str, Any]:
    source_filename = str(refined.get("hydration_source_filename", ""))
    if source_filename != REFINED_BEHAVIOR_FILENAME:
        raise ValueError(
            f"{animal_id}: expected hydration source {REFINED_BEHAVIOR_FILENAME!r}, "
            f"got {source_filename!r}. Use --rebuild-refined to rebuild from the requested source."
        )
    n_frames = int(refined.get("n_frames", 0))
    n_cells = int(refined.get("n_cells", len(refined.get("spikes", []))))
    hd = np.asarray(refined.get("hd_angles_neural", []), dtype=float).reshape(-1)
    if n_frames <= 0:
        raise ValueError(f"{animal_id}: refined payload has no frames")
    if hd.size != n_frames:
        raise ValueError(f"{animal_id}: hd_angles_neural size {hd.size} does not match n_frames {n_frames}")
    if n_cells <= 0:
        raise ValueError(f"{animal_id}: refined payload has no cells")
    return {
        "n_frames": n_frames,
        "n_cells": n_cells,
        "hydration_source_filename": source_filename,
        "hd_nan_frames": int(np.sum(~np.isfinite(hd))),
        "manual_refined_source": bool(refined.get("manual_refined_source", False)),
        "schema_version": int(refined.get("schema_version", 0)),
    }


def build_bundle(
    *,
    data_root: Path,
    animals: list[str],
    rebuild_refined: bool,
    include_data: bool,
    slim: bool,
) -> dict[str, Any]:
    animal_payloads: dict[str, dict[str, Any]] = {}
    source_files: dict[str, dict[str, Any]] = {}
    validation: dict[str, dict[str, Any]] = {}
    slimming: dict[str, dict[str, Any]] = {}

    for animal_id in animals:
        animal_dir = data_root / animal_id
        if not animal_dir.exists():
            raise FileNotFoundError(f"Missing animal directory: {animal_dir}")
        refined = _build_refined_payload(animal_dir, rebuild_refined=rebuild_refined)
        if slim:
            refined, slimming[animal_id] = slim_refined_payload_for_cluster(refined)
        else:
            slimming[animal_id] = {
                "enabled": False,
                "array_bytes_before": int(_array_nbytes_recursive(refined)),
                "array_bytes_after": int(_array_nbytes_recursive(refined)),
                "array_bytes_saved": 0,
            }
        validation[animal_id] = _validate_refined_payload(animal_id, refined)
        validation[animal_id]["cluster_export_slim"] = bool(slim)
        validation[animal_id]["array_bytes_after_slim"] = int(slimming[animal_id]["array_bytes_after"])
        validation[animal_id]["array_bytes_saved_by_slim"] = int(slimming[animal_id]["array_bytes_saved"])
        source_files[animal_id] = {
            MANUAL_REFINED_SIDECAR_FILENAME: _file_record(animal_dir / MANUAL_REFINED_SIDECAR_FILENAME),
            REFINED_BEHAVIOR_FILENAME: _file_record(animal_dir / REFINED_BEHAVIOR_FILENAME),
        }
        if include_data:
            animal_payloads[animal_id] = refined

    return {
        "schema_version": REFINED_CLUSTER_INPUT_SCHEMA_VERSION,
        "generated_at": _utcnow_iso(),
        "generator": {
            "script": str(Path(__file__).resolve()),
            "data_root": str(data_root),
            "rebuild_refined": bool(rebuild_refined),
            "slim": bool(slim),
        },
        "parameter_snapshot": refined_parameter_snapshot(),
        "source_files": source_files,
        "validation": validation,
        "slimming": slimming,
        "animals": animal_payloads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=HERE / "data")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "cluster_exports" / CLUSTER_REFINED_BUNDLE_FILENAME,
    )
    parser.add_argument("--animals", nargs="+", default=list(ANIMALS))
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output bundle")
    parser.add_argument("--dry-run", action="store_true", help="Validate sources without writing the bundle")
    parser.add_argument(
        "--rebuild-refined",
        action="store_true",
        help="Rebuild refined payloads in memory from manual sidecars and merged_aligned_data_new.pkl",
    )
    parser.add_argument(
        "--no-slim",
        action="store_true",
        help="Write the full refined payload without cluster-transfer slimming",
    )
    args = parser.parse_args()

    output = args.output
    if output.exists() and not args.force and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite existing bundle without --force: {output}")

    bundle = build_bundle(
        data_root=args.data_root,
        animals=[str(a) for a in args.animals],
        rebuild_refined=bool(args.rebuild_refined),
        include_data=not bool(args.dry_run),
        slim=not bool(args.no_slim),
    )

    print(json.dumps(bundle["validation"], indent=2, sort_keys=True))
    if args.dry_run:
        print("[DRY RUN] Validated refined inputs; no bundle written.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved refined cluster bundle: {output}")


if __name__ == "__main__":
    main()
