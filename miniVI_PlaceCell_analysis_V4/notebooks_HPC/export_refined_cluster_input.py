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
) -> dict[str, Any]:
    animal_payloads: dict[str, dict[str, Any]] = {}
    source_files: dict[str, dict[str, Any]] = {}
    validation: dict[str, dict[str, Any]] = {}

    for animal_id in animals:
        animal_dir = data_root / animal_id
        if not animal_dir.exists():
            raise FileNotFoundError(f"Missing animal directory: {animal_dir}")
        refined = _build_refined_payload(animal_dir, rebuild_refined=rebuild_refined)
        validation[animal_id] = _validate_refined_payload(animal_id, refined)
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
        },
        "parameter_snapshot": refined_parameter_snapshot(),
        "source_files": source_files,
        "validation": validation,
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
    args = parser.parse_args()

    output = args.output
    if output.exists() and not args.force and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite existing bundle without --force: {output}")

    bundle = build_bundle(
        data_root=args.data_root,
        animals=[str(a) for a in args.animals],
        rebuild_refined=bool(args.rebuild_refined),
        include_data=not bool(args.dry_run),
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
