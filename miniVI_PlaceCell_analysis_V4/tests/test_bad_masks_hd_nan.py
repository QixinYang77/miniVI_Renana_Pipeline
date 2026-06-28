from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
TOP_REPO = ANALYSIS_ROOT.parent
top_repo_norm = str(TOP_REPO.resolve())
sys.path[:] = [
    p for p in sys.path
    if str(Path(p or ".").resolve()) != top_repo_norm
]
sys.path.insert(0, str(ANALYSIS_ROOT))

from utils import placecell_pipeline as pcp


def test_hd_nan_frames_are_bad_for_every_legacy_cell(monkeypatch):
    n_frames = 8
    n_cells = 2

    def fake_snr(**kwargs):
        return {"snr_time_varying": np.full(n_frames, 10.0)}

    monkeypatch.setattr(pcp, "compute_time_varying_snr_from_trace", fake_snr)
    data = {
        "x_neural": np.arange(n_frames, dtype=float),
        "y_neural": np.arange(n_frames, dtype=float),
        "speed": np.full(n_frames, 5.0),
        "hd_angles_neural": np.array([0.0, 0.1, np.nan, 0.3, 0.4, np.nan, 0.6, 0.7]),
        "frame_rate": 10.0,
        "spikes": [np.array([1]), np.array([2])],
        "all_spikes": [np.array([1]), np.array([2])],
        "traces_SNR_interpolated": [np.ones(n_frames), np.ones(n_frames)],
        "complex_bursts_dicts": [{}, {}],
    }

    masks, stats = pcp._compute_bad_masks(
        data,
        snr_threshold=5.0,
        min_good_minutes=0.0,
        return_stats=True,
    )

    expected = np.zeros((n_cells, n_frames), dtype=bool)
    expected[:, [2, 5]] = True
    assert np.array_equal(masks, expected)
    assert [row["n_removed_frames_head_direction_nan"] for row in stats] == [2, 2]


def test_manual_refined_masks_combine_hd_nan_with_existing_bad_sources():
    n_frames = 8
    stored = np.zeros((1, n_frames), dtype=bool)
    stored[0, 1] = True
    manual_exclusion = np.zeros((1, n_frames), dtype=bool)
    manual_exclusion[0, 3] = True
    manual_snr_cutoff = np.zeros((1, n_frames), dtype=bool)
    manual_snr_cutoff[0, 4] = True
    trace = np.ones(n_frames)
    trace[5] = np.nan
    snr = np.full(n_frames, 10.0)
    snr[6] = 1.0

    data = {
        "manual_refined_source": True,
        "manual_refined_bad_masks": stored,
        "manual_refined_manual_exclusion_masks": manual_exclusion,
        "manual_refined_snr_cutoff_masks": manual_snr_cutoff,
        "manual_refined_bad_mask_stats": [{}],
        "x_neural": np.arange(n_frames, dtype=float),
        "y_neural": np.arange(n_frames, dtype=float),
        "speed": np.full(n_frames, 5.0),
        "hd_angles_neural": np.array([0.0, 0.1, np.nan, 0.3, 0.4, 0.5, 0.6, 0.7]),
        "frame_rate": 10.0,
        "spikes": [np.array([1])],
        "traces": [trace],
        "SNR_interpolated": [snr],
        "spike_heights_interpolated": [np.ones(n_frames)],
    }

    masks, stats = pcp._compute_bad_masks(
        data,
        snr_threshold=5.0,
        min_good_minutes=0.0,
        return_stats=True,
    )

    assert set(np.flatnonzero(masks[0])) == {1, 2, 3, 4, 5, 6}
    assert stats[0]["n_removed_frames_head_direction_nan"] == 1
    assert stats[0]["n_removed_frames_source_trace_bad"] == 1
    assert stats[0]["n_removed_frames_manual_snr_cutoff"] == 1
    assert stats[0]["n_removed_frames_snr_threshold_only"] == 1


def test_cluster_precomputed_masks_allow_trace_free_payload():
    n_frames = 6
    precomputed = np.zeros((2, n_frames), dtype=bool)
    precomputed[0, 1] = True
    precomputed[1, 4] = True
    data = {
        "manual_refined_source": True,
        "x_neural": np.arange(n_frames, dtype=float),
        "y_neural": np.arange(n_frames, dtype=float),
        "speed": np.full(n_frames, 5.0),
        "hd_angles_neural": np.array([0.0, np.nan, 0.2, 0.3, 0.4, 0.5]),
        "frame_rate": 10.0,
        "spikes": [np.array([1]), np.array([2])],
        "cluster_precomputed_bad_masks": precomputed,
        "cluster_precomputed_bad_mask_params": {
            "snr_threshold": 5.0,
            "min_good_minutes": 0.0,
        },
        "cluster_precomputed_bad_mask_stats": [
            {"n_removed_frames_snr_only": 1},
            {"n_removed_frames_snr_only": 1},
        ],
    }

    masks, stats = pcp._compute_bad_masks(
        data,
        snr_threshold=5.0,
        min_good_minutes=0.0,
        return_stats=True,
    )

    assert set(np.flatnonzero(masks[0])) == {1}
    assert set(np.flatnonzero(masks[1])) == {1, 4}
    assert [row["n_removed_frames_head_direction_nan"] for row in stats] == [1, 1]
    assert all(row["cluster_precomputed_bad_masks"] for row in stats)
