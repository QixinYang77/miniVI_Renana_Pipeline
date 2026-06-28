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
    CLUSTER_REFINED_INPUT_FILENAME,
    CLUSTER_SPATIAL_ANALYSIS_FILENAME,
    MANUAL_REFINED_SIDECAR_FILENAME,
    REFINED_BEHAVIOR_FILENAME,
    REFINED_CLUSTER_INPUT_SCHEMA_VERSION,
    refined_analysis_params,
    refined_parameter_snapshot,
)
from utils.placecell_pipeline import (
    REFINED_ANALYSIS_DATA_KEY,
    build_refined_analysis_data_from_manual_sidecar,
    _compute_bad_masks,
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
EGOCENTRIC_PROFILE = "egocentric"
SLIM_PROFILE = "slim"
FULL_PROFILE = "full"

EGOCENTRIC_KEEP_KEYS = (
    "schema_version",
    "manual_refined_source",
    "hydration_source_filename",
    "hydration_source_path",
    "n_frames",
    "n_cells",
    "frame_rate",
    "session_start_frames",
    "x_neural",
    "y_neural",
    "speed",
    "hd_angles_neural",
    "ts_neural",
    "spikes",
    "all_spikes",
    "refined_SS",
    "all_CS_spikes",
)

EGOCENTRIC_FLOAT32_KEYS = (
    "x_neural",
    "y_neural",
    "speed",
    "hd_angles_neural",
    "ts_neural",
)

SPATIAL_CLUSTER_KEEP_KEYS = (
    "animal_id",
    "cell_idx",
    "is_place_cell",
    "is_place_cell_ss",
    "is_place_cell_cs",
    "si",
    "p_value",
    "ss_si",
    "ss_p_value",
    "cs_si",
    "cs_p_value",
    "peak_rate",
    "ss_peak_rate",
    "cs_peak_rate",
    "n_place_fields",
    "n_ss_place_fields",
    "n_cs_place_fields",
    "pf_sizes",
    "ss_pf_sizes",
    "cs_pf_sizes",
    "field_area",
    "ss_field_area",
    "cs_field_area",
    "place_field_components",
    "place_field_mask",
    "ss_place_field_mask",
    "cs_place_field_mask",
    "n_frames_total",
    "n_frames_kept_total",
    "n_removed_frames_total",
    "n_removed_frames_snr_only",
    "n_removed_frames_pos_nan",
    "n_removed_frames_head_direction_nan",
    "pct_removed_frames_total",
    "pct_removed_frames_snr_only",
    "pct_removed_frames_head_direction_nan",
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


def _as_float32_1d(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _as_int_spike_list(value: Any, n_cells: int, n_frames: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seq = value if isinstance(value, (list, tuple)) else []
    for cell_idx in range(int(n_cells)):
        arr = np.asarray(seq[cell_idx] if cell_idx < len(seq) else [], dtype=np.int64).reshape(-1)
        arr = arr[(arr >= 0) & (arr < int(n_frames))]
        out.append(np.asarray(np.unique(arr), dtype=np.int32))
    return out


def egocentric_refined_payload_for_cluster(refined: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the trace-free payload needed by cluster egocentric shuffling jobs."""
    before_bytes = _array_nbytes_recursive(refined)
    analysis = refined_analysis_params()
    n_frames = int(refined.get("n_frames", len(refined.get("x_neural", []))))
    n_cells = int(refined.get("n_cells", len(refined.get("spikes", []))))
    bad_masks, bad_mask_stats = _compute_bad_masks(
        refined,
        snr_threshold=float(analysis.snr_threshold),
        min_good_minutes=float(analysis.min_good_minutes),
        return_stats=True,
    )
    if bad_masks.shape != (n_cells, n_frames):
        raise RuntimeError(f"Precomputed bad mask shape mismatch: expected {(n_cells, n_frames)}, got {bad_masks.shape}")

    slimmed: dict[str, Any] = {}
    for key in EGOCENTRIC_KEEP_KEYS:
        if key in refined:
            slimmed[key] = refined[key]
    slimmed["n_frames"] = int(n_frames)
    slimmed["n_cells"] = int(n_cells)
    slimmed["manual_refined_source"] = bool(refined.get("manual_refined_source", False))
    slimmed["hydration_source_filename"] = str(refined.get("hydration_source_filename", REFINED_BEHAVIOR_FILENAME))

    for key in EGOCENTRIC_FLOAT32_KEYS:
        if key in slimmed:
            slimmed[key] = _as_float32_1d(slimmed[key])

    slimmed["spikes"] = _as_int_spike_list(refined.get("spikes", []), n_cells, n_frames)
    slimmed["all_spikes"] = _as_int_spike_list(refined.get("all_spikes", refined.get("spikes", [])), n_cells, n_frames)
    slimmed["refined_SS"] = _as_int_spike_list(refined.get("refined_SS", []), n_cells, n_frames)
    slimmed["all_CS_spikes"] = _as_int_spike_list(refined.get("all_CS_spikes", []), n_cells, n_frames)
    slimmed["cluster_precomputed_bad_masks"] = np.asarray(bad_masks, dtype=bool)
    slimmed["cluster_precomputed_bad_mask_stats"] = list(bad_mask_stats)
    slimmed["cluster_precomputed_bad_mask_params"] = {
        "snr_threshold": float(analysis.snr_threshold),
        "min_good_minutes": float(analysis.min_good_minutes),
        "includes_position_nan": True,
        "includes_head_direction_nan": True,
        "includes_manual_refined_masks": True,
        "includes_source_trace_bad_frames": True,
    }

    after_bytes = _array_nbytes_recursive(slimmed)
    dropped_keys = sorted(str(k) for k in refined.keys() if k not in slimmed)
    report = {
        "enabled": True,
        "profile": "refined_cluster_egocentric_only_v1",
        "array_bytes_before": int(before_bytes),
        "array_bytes_after": int(after_bytes),
        "array_bytes_saved": int(before_bytes - after_bytes),
        "dropped_keys": dropped_keys,
        "float32_keys": list(EGOCENTRIC_FLOAT32_KEYS),
        "precomputed_bad_masks": True,
        "runtime_data_filename": CLUSTER_REFINED_INPUT_FILENAME,
    }
    slimmed["cluster_export_slimming"] = report
    return slimmed, report


def _cb_in_pf_count_from_spatial_cell(cell: dict[str, Any]) -> int:
    spike_shapes = cell.get("spike_shapes")
    if isinstance(spike_shapes, dict) and "complex" in spike_shapes:
        complex_shapes = spike_shapes.get("complex") or {}
        shapes = complex_shapes.get("shapes", {}) if isinstance(complex_shapes, dict) else {}
        if isinstance(shapes, dict):
            try:
                return int(len(shapes.get("run_in", [])))
            except Exception:
                return 0
    return int(cell.get("n_cb_in_pf", 0) or 0)


def _minimal_spatial_cell(cell: dict[str, Any]) -> dict[str, Any]:
    out = {key: cell[key] for key in SPATIAL_CLUSTER_KEEP_KEYS if key in cell}
    out["cell_idx"] = int(cell.get("cell_idx", -1))
    out["n_cb_in_pf"] = _cb_in_pf_count_from_spatial_cell(cell)
    return out


def _load_minimal_spatial_analysis(animal_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spatial_path = animal_dir / CLUSTER_SPATIAL_ANALYSIS_FILENAME
    if not spatial_path.exists():
        raise FileNotFoundError(
            f"Missing local {CLUSTER_SPATIAL_ANALYSIS_FILENAME} for {animal_dir.name}. "
            "Run the refined notebook/place-cell cache locally before exporting the cluster bundle."
        )
    with spatial_path.open("rb") as f:
        cells = pickle.load(f)
    if not isinstance(cells, list):
        raise ValueError(f"{spatial_path} is not a list")
    minimal = [_minimal_spatial_cell(cell) for cell in cells if isinstance(cell, dict)]
    return minimal, {
        "source_path": str(spatial_path),
        "source_file": _file_record(spatial_path),
        "n_cells": int(len(minimal)),
        "array_bytes": int(_array_nbytes_recursive(minimal)),
        "profile": "spatial_classification_minimal_v1",
    }


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
    profile: str,
) -> dict[str, Any]:
    animal_payloads: dict[str, dict[str, Any]] = {}
    spatial_payloads: dict[str, list[dict[str, Any]]] = {}
    source_files: dict[str, dict[str, Any]] = {}
    validation: dict[str, dict[str, Any]] = {}
    slimming: dict[str, dict[str, Any]] = {}
    spatial_validation: dict[str, dict[str, Any]] = {}

    for animal_id in animals:
        animal_dir = data_root / animal_id
        if not animal_dir.exists():
            raise FileNotFoundError(f"Missing animal directory: {animal_dir}")
        refined = _build_refined_payload(animal_dir, rebuild_refined=rebuild_refined)
        if profile == EGOCENTRIC_PROFILE:
            refined, slimming[animal_id] = egocentric_refined_payload_for_cluster(refined)
        elif profile == SLIM_PROFILE:
            refined, slimming[animal_id] = slim_refined_payload_for_cluster(refined)
        elif profile == FULL_PROFILE:
            slimming[animal_id] = {
                "enabled": False,
                "profile": "full_refined_payload",
                "array_bytes_before": int(_array_nbytes_recursive(refined)),
                "array_bytes_after": int(_array_nbytes_recursive(refined)),
                "array_bytes_saved": 0,
            }
        else:
            raise ValueError(f"Unknown export profile: {profile!r}")
        spatial_payload, spatial_meta = _load_minimal_spatial_analysis(animal_dir)
        spatial_validation[animal_id] = spatial_meta
        validation[animal_id] = _validate_refined_payload(animal_id, refined)
        validation[animal_id]["cluster_export_profile"] = str(profile)
        validation[animal_id]["array_bytes_after_slim"] = int(slimming[animal_id]["array_bytes_after"])
        validation[animal_id]["array_bytes_saved_by_slim"] = int(slimming[animal_id]["array_bytes_saved"])
        validation[animal_id]["spatial_classification_cells"] = int(spatial_meta["n_cells"])
        source_files[animal_id] = {
            MANUAL_REFINED_SIDECAR_FILENAME: _file_record(animal_dir / MANUAL_REFINED_SIDECAR_FILENAME),
            REFINED_BEHAVIOR_FILENAME: _file_record(animal_dir / REFINED_BEHAVIOR_FILENAME),
            CLUSTER_SPATIAL_ANALYSIS_FILENAME: _file_record(animal_dir / CLUSTER_SPATIAL_ANALYSIS_FILENAME),
        }
        if include_data:
            animal_payloads[animal_id] = refined
            spatial_payloads[animal_id] = spatial_payload

    return {
        "schema_version": REFINED_CLUSTER_INPUT_SCHEMA_VERSION,
        "generated_at": _utcnow_iso(),
        "generator": {
            "script": str(Path(__file__).resolve()),
            "data_root": str(data_root),
            "rebuild_refined": bool(rebuild_refined),
            "profile": str(profile),
        },
        "parameter_snapshot": refined_parameter_snapshot(),
        "source_files": source_files,
        "validation": validation,
        "spatial_validation": spatial_validation,
        "slimming": slimming,
        "animals": animal_payloads,
        "spatial_analysis": spatial_payloads,
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
        help="Deprecated alias for --profile full",
    )
    parser.add_argument(
        "--profile",
        choices=[EGOCENTRIC_PROFILE, SLIM_PROFILE, FULL_PROFILE],
        default=EGOCENTRIC_PROFILE,
        help=(
            "Export profile. 'egocentric' writes trace-free Step-2 inputs plus reduced spatial "
            "classification files; 'slim' keeps trace-bearing refined payloads with duplicate arrays "
            "removed; 'full' writes the full refined payload."
        ),
    )
    args = parser.parse_args()
    profile = FULL_PROFILE if args.no_slim else str(args.profile)

    output = args.output
    if output.exists() and not args.force and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite existing bundle without --force: {output}")

    bundle = build_bundle(
        data_root=args.data_root,
        animals=[str(a) for a in args.animals],
        rebuild_refined=bool(args.rebuild_refined),
        include_data=not bool(args.dry_run),
        profile=profile,
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
