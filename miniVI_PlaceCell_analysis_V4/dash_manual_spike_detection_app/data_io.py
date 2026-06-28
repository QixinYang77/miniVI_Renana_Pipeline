"""Data loading and sidecar persistence for manual spike detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import tempfile
from typing import Any

import numpy as np

SIDE_CAR_FILENAME = "manual_spike_detection_results.pkl"
SOURCE_FILENAME = "merged_aligned_data.pkl"
TRACE_KEY = "traces"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AnimalBundle:
    animal_id: str
    animal_dir: Path
    source_path: Path
    merged: dict[str, Any]
    traces: np.ndarray
    frame_rate: float
    n_cells: int
    n_frames: int


def default_data_root() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def discover_animals(data_root: str | Path) -> list[str]:
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        return []
    out: list[str] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.startswith("CKII") and (child / SOURCE_FILENAME).is_file():
            out.append(child.name)
    return out


def load_animal(data_root: str | Path, animal_id: str) -> AnimalBundle:
    root = Path(data_root).expanduser().resolve()
    animal_dir = root / str(animal_id)
    source_path = animal_dir / SOURCE_FILENAME
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing {SOURCE_FILENAME} for {animal_id}: {source_path}")

    with source_path.open("rb") as f:
        merged = pickle.load(f)

    if TRACE_KEY not in merged:
        raise KeyError(f"{source_path} does not contain key {TRACE_KEY!r}")
    traces = np.asarray(merged[TRACE_KEY], dtype=float)
    if traces.ndim != 2:
        raise ValueError(f"{TRACE_KEY!r} must be 2D, got shape {traces.shape}")
    if traces.shape[0] > traces.shape[1]:
        traces = traces.T

    frame_rate = float(merged.get("frame_rate", np.nan))
    if not np.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(f"Invalid frame_rate in {source_path}: {frame_rate!r}")

    return AnimalBundle(
        animal_id=str(animal_id),
        animal_dir=animal_dir,
        source_path=source_path,
        merged=merged,
        traces=traces,
        frame_rate=frame_rate,
        n_cells=int(traces.shape[0]),
        n_frames=int(traces.shape[1]),
    )


def sidecar_path(animal_dir: str | Path) -> Path:
    return Path(animal_dir).expanduser().resolve() / SIDE_CAR_FILENAME


def load_sidecar(animal_dir: str | Path) -> dict[str, Any]:
    path = sidecar_path(animal_dir)
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "source_file": None,
            "animal_id": Path(animal_dir).name,
            "trace_key": TRACE_KEY,
            "cells": {},
        }
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid sidecar payload: {path}")
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("trace_key", TRACE_KEY)
    payload.setdefault("cells", {})
    if not isinstance(payload["cells"], dict):
        payload["cells"] = {}
    return payload


def get_saved_cell(payload: dict[str, Any], cell_idx: int) -> dict[str, Any] | None:
    cells = payload.get("cells", {})
    if not isinstance(cells, dict):
        return None
    idx = int(cell_idx)
    item = cells.get(str(idx))
    if item is None:
        item = cells.get(idx)
    return item if isinstance(item, dict) else None


def save_cell_result(bundle: AnimalBundle, cell_idx: int, result: dict[str, Any]) -> Path:
    path = sidecar_path(bundle.animal_dir)
    payload = load_sidecar(bundle.animal_dir)
    source_stat = bundle.source_path.stat()
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "source_file": str(bundle.source_path),
            "source_filename": bundle.source_path.name,
            "source_mtime_ns": int(source_stat.st_mtime_ns),
            "animal_id": bundle.animal_id,
            "frame_rate": float(bundle.frame_rate),
            "trace_key": TRACE_KEY,
        }
    )
    payload.setdefault("cells", {})
    payload["cells"][str(int(cell_idx))] = result
    # A manual edit changes the event/trace inputs used to build refined caches.
    # Force an explicit re-hydration before refined analysis uses this sidecar again.
    payload.pop("refined_analysis_data", None)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            pickle.dump(payload, f)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
    return path
