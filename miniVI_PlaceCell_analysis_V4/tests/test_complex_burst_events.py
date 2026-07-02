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


def _test_config(tmp_path):
    return pcp.PipelineConfig(
        project_root=tmp_path,
        data_root=tmp_path,
        figures_root=tmp_path,
        notebooks_root=tmp_path,
        animals=["test"],
        analysis=pcp.AnalysisParams(snr_threshold=5.0, min_good_minutes=0.0),
    )


def test_complex_burst_events_use_first_complex_spike_per_burst():
    bursts = [
        {
            "starts": np.array([10, 20, 40, 80]),
            "ends": np.array([15, 25, 45, 90]),
            "locs": np.array([11, 21, 41, 81]),
        }
    ]
    cs_spikes = [np.array([9, 12, 14, 20, 23, 44, 50, 85, 86])]

    events = pcp.derive_complex_burst_event_list(
        bursts,
        cs_spikes,
        n_cells=1,
        n_frames=100,
    )

    assert len(events) == 1
    assert np.array_equal(events[0], np.array([12, 20, 44, 85]))


def test_complex_burst_events_fallback_to_locs_without_complex_spikes():
    bursts = [
        {
            "starts": np.array([10, 20, 40, 80]),
            "ends": np.array([15, 25, 45, 90]),
            "locs": np.array([11, 21, 21, 101, -1]),
        }
    ]

    events = pcp.derive_complex_burst_event_list(
        bursts,
        [None],
        n_cells=1,
        n_frames=100,
    )

    assert np.array_equal(events[0], np.array([11, 21]))


def test_complex_burst_events_do_not_fallback_when_complex_spikes_are_empty():
    bursts = [
        {
            "starts": np.array([10, 20]),
            "ends": np.array([15, 25]),
            "locs": np.array([11, 21]),
        }
    ]

    events = pcp.derive_complex_burst_event_list(
        bursts,
        [np.array([], dtype=int)],
        n_cells=1,
        n_frames=100,
    )

    assert np.array_equal(events[0], np.array([], dtype=np.int64))


def test_prepare_context_filters_complex_burst_events_with_bad_masks(tmp_path):
    n_frames = 8
    precomputed = np.zeros((1, n_frames), dtype=bool)
    precomputed[0, 4] = True
    merged = {
        "manual_refined_source": True,
        "x_neural": np.arange(n_frames, dtype=float),
        "y_neural": np.arange(n_frames, dtype=float),
        "speed": np.full(n_frames, 5.0),
        "hd_angles_neural": np.array([0.0, 0.1, np.nan, 0.3, 0.4, 0.5, 0.6, 0.7]),
        "frame_rate": 10.0,
        "spikes": [np.array([1, 2, 4, 6])],
        "all_spikes": [np.array([1, 2, 4, 6])],
        "refined_SS": [np.array([1])],
        "all_CS_spikes": [np.array([2, 4, 6])],
        "complex_bursts_dicts": [
            {
                "starts": np.array([1, 4, 5]),
                "ends": np.array([3, 4, 7]),
                "locs": np.array([2, 4, 6]),
            }
        ],
        "cluster_precomputed_bad_masks": precomputed,
        "cluster_precomputed_bad_mask_params": {
            "snr_threshold": 5.0,
            "min_good_minutes": 0.0,
        },
    }

    ctx = pcp._prepare_native_analysis_context(
        merged,
        _test_config(tmp_path),
        require_traces=False,
    )

    assert np.array_equal(ctx["complex_burst_events"][0], np.array([6]))
    assert ctx["bad_masks"][0, 2]
    assert ctx["bad_masks"][0, 4]
