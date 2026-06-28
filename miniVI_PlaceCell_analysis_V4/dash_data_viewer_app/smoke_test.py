"""Smoke tests for the Dash trace viewer data and SNR helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import types

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ANALYSIS_ROOT = _THIS_DIR.parent
_UTILS_DIR = _ANALYSIS_ROOT / "utils"
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_ROOT))
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from data_io import (  # noqa: E402
    bad_mask_for_cell,
    default_data_root,
    dict_intervals,
    discover_animals,
    load_animal,
    mask_to_intervals,
)

if "utils" not in sys.modules or not hasattr(sys.modules["utils"], "__path__"):
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [str(_UTILS_DIR)]
    sys.modules["utils"] = utils_pkg
else:
    sys.modules["utils"].__path__ = [str(_UTILS_DIR)]

from utils.placecell_pipeline import _compute_bad_masks  # noqa: E402


def run_smoke(data_root: str | Path, *, compare_snr: bool = True) -> None:
    animals = discover_animals(data_root)
    if not animals:
        raise RuntimeError(f"No merged animal folders found under {data_root}")

    for animal_id in animals:
        animal = load_animal(data_root, animal_id)
        if animal.n_cells <= 0:
            raise AssertionError(f"{animal_id}: no cells")
        if animal.n_frames <= 0:
            raise AssertionError(f"{animal_id}: no frames")
        if len(animal.all_spikes) != animal.n_cells:
            raise AssertionError(f"{animal_id}: all_spikes length mismatch")
        _ = dict_intervals(animal.complex_bursts[0], animal.n_frames)
        _ = dict_intervals(animal.plateaus[0], animal.n_frames)
        _ = mask_to_intervals(np.zeros(animal.n_frames, dtype=bool))

        if compare_snr:
            expected = _compute_bad_masks(animal.merged, 3.0, 5.0)
            got_rows = []
            for cell_idx in range(animal.n_cells):
                mask, _stats = bad_mask_for_cell(animal, cell_idx, 3.0, 5.0)
                got_rows.append(mask)
            got = np.asarray(got_rows, dtype=bool)
            if got.shape != expected.shape or not np.array_equal(got, expected):
                raise AssertionError(f"{animal_id}: app SNR masks differ from _compute_bad_masks")

        print(f"OK {animal_id}: cells={animal.n_cells}, frames={animal.n_frames}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Dash trace viewer helpers")
    parser.add_argument("--data-root", type=str, default=str(default_data_root()))
    parser.add_argument("--skip-snr-compare", action="store_true")
    args = parser.parse_args()
    run_smoke(args.data_root, compare_snr=not args.skip_snr_compare)


if __name__ == "__main__":
    main()
