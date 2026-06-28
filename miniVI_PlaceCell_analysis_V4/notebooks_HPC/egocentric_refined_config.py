"""Shared refined CKII cluster configuration for egocentric analyses."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from utils.placecell_pipeline import (
    AnalysisParams,
    CachePolicy,
    EgocentricSummaryPlotParams,
    EgocentricTuningParams,
    PFTraversalParams,
    PipelineConfig,
    PlaceCellParams,
    PooledParams,
)

REFINED_CLUSTER_INPUT_SCHEMA_VERSION = 1
CLUSTER_REFINED_BUNDLE_FILENAME = "ckii_refined_cluster_input_v1.pkl"
CLUSTER_REFINED_INPUT_FILENAME = "cluster_refined_analysis_data.pkl"
CLUSTER_REFINED_METADATA_FILENAME = "cluster_refined_analysis_metadata.json"
MANUAL_REFINED_SIDECAR_FILENAME = "manual_spike_detection_results.pkl"
REFINED_BEHAVIOR_FILENAME = "merged_aligned_data_new.pkl"

ANIMALS = [
    "CKII_pAce21_PR_20250806",
    "CKII_pAce38_PX_20251126",
    "CKII_pAce45_PX_20260118",
    "CKII_pAce47_PX_20260128",
    "CKII_pAce46_PR_20260222",
    "CKII_pAce50_PRL_20260317",
    "CKII_pAce54_PR_20260506",
    "CKII_pAce54_PX_20260514",
]

DEFAULT_CATEGORIES = ("CSplus", "CSminus")
DEFAULT_FIRST_N_MINUTES = 10.0
DEFAULT_DIRECTION_MODE = "head"


def refined_analysis_params() -> AnalysisParams:
    return AnalysisParams(
        speed_threshold=2.5,
        speed_threshold_quiet=0.5,
        behavior_speed_outlier_threshold_cm_s=100.0,
        behavior_speed_outlier_cleaning=True,
        min_duration_s=0.2,
        merge_gap_s=0.0,
        kernel_size=51,
        snr_threshold=5.0,
        min_good_minutes=5.0,
        theta_freqs=(4.0, 10.0),
        slow_freqs=2.0,
        refined_apply_cb_baseline_removal=True,
        refined_cb_baseline_window_s=5.0,
        refined_snr_cb_baseline_window_s=1.0,
    )


def refined_place_cell_params() -> PlaceCellParams:
    return PlaceCellParams(
        bin_size=1.5,
        place_field_threshold=0.25,
        min_component_peak_ratio=0.3,
        split_multi_peak_fields=False,
        split_secondary_peak_ratio=0.6,
        split_secondary_peak_min_separation_cm=6.0,
        min_peak_rate=0.8,
        max_field_area_ratio=0.66,
        min_field_bins=10,
        min_pf_firing_traversals=4,
        pf_firing_traversal_distance_window_cm=15.0,
        pf_firing_traversal_detection_window_cm=8.0,
        pf_firing_traversal_distance_bin_cm=1.5,
        pf_firing_traversal_distance_mode="euclidean_to_peak",
        pf_firing_traversal_center_vicinity_min_cm=1,
        pf_firing_traversal_center_vicinity_max_cm=5,
        pf_firing_traversal_resting_speed_threshold=0.5,
        pf_firing_traversal_merge_gap_s=2.0,
        pf_firing_traversal_exclude_trials_with_bad_frames=True,
        pf_reliability_dilation_bins=3,
        pf_reliability_dilation_shape="disk",
        smooth_sigma=1.5,
        min_occupancy_s=0.1,
        occ_smooth_sigma=1.5,
        num_shuffles=1000,
        random_seed=42,
        ss_shape_min_separation_ms=14.0,
        trim_sparse_top_row_for_analysis=True,
        trim_sparse_top_row_for_plotting=True,
        sparse_top_row_nonocc_frac_threshold=0.8,
    )


def refined_traversal_params() -> PFTraversalParams:
    return PFTraversalParams(
        center_by_pf_position=True,
        pf_component_selection="peak_rate",
        min_duration_ms=100.0,
        min_distance_cm=5.0,
        traversal_merge_gap_s=2.0,
        clear_traversal=False,
        session_indices=(0, 1),
        pf_center_window_sec=10.0,
        min_traversals=10,
        firing_rate_bin_ms=100.0,
        firing_rate_smooth_ms=50.0,
        subtract_pre_traversal_baseline=False,
        mask_non_traversal_pf=True,
        max_pf_distance_cm=8.0,
        plateau_min_duration_ms=100.0,
    )


def refined_pooled_params() -> PooledParams:
    return PooledParams(
        cb_num_threshold=5,
        cs_peak_rate_threshold=0.5,
        cs_plc_definition_mode="cs_place_field",
        run_psd_sections=True,
        cs_plc_only=True,
        psd_speed_threshold=3,
        psd_chunk_s=2.0,
        psd_nperseg_s=1.0,
        psd_noverlap_frac=0.5,
        simple_event_window_ms=80.0,
        simple_event_min_gap_ms=50.0,
        min_chunk_valid_fraction=1.0,
        max_freq=100.0,
        normalize_psd=True,
        norm_freq_range=(20.0, 100.0),
    )


def build_refined_config(
    project_root: str | Path,
    data_root: str | Path,
    figures_root: str | Path,
    *,
    force_recompute: bool = False,
) -> PipelineConfig:
    project_root = Path(project_root).resolve()
    return PipelineConfig(
        project_root=project_root,
        data_root=Path(data_root),
        figures_root=Path(figures_root),
        notebooks_root=project_root / "notebooks_PCs_refined",
        animals=list(ANIMALS),
        merged_data_filename=CLUSTER_REFINED_INPUT_FILENAME,
        analysis=refined_analysis_params(),
        place_cell=refined_place_cell_params(),
        traversal=refined_traversal_params(),
        pooled=refined_pooled_params(),
        cache=CachePolicy(
            force_recompute=bool(force_recompute),
            validate_only=False,
            save_executed_notebooks=False,
        ),
    )


def build_refined_egocentric_params(
    *,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    first_n_minutes: float | None = DEFAULT_FIRST_N_MINUTES,
    direction_mode: str = DEFAULT_DIRECTION_MODE,
    n_surrogates: int = 1000,
    n_jobs: int = 1,
    clear_output: bool = False,
    save_null_distributions: bool = False,
    show_progress: bool = False,
) -> EgocentricTuningParams:
    return EgocentricTuningParams(
        categories=tuple(categories),
        first_n_minutes=first_n_minutes,
        direction_mode=direction_mode,
        time_bin_s=0.1,
        arena_size_cm=(35.5, 20.0),
        local_spatial_bin_cm=5.0,
        coarse_spatial_bin_cm=2.0,
        n_angle_bins=10,
        speed_min_cm_s=3.0,
        speed_max_cm_s=60.0,
        occupancy_threshold_s=0.2,
        min_occupied_angle_bins=3,
        min_mean_rate_hz=0.5,
        min_valid_spatial_bins_for_fit=5,
        n_restarts=100,
        optimizer_method="Nelder-Mead",
        n_surrogates=int(n_surrogates),
        n_jobs=int(n_jobs),
        surrogate_chunk_size=20,
        random_seed=42,
        clear_output=bool(clear_output),
        save_null_distributions=bool(save_null_distributions),
        show_progress=bool(show_progress),
    )


def build_refined_summary_plot_params(
    *,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    first_n_minutes: float | None = DEFAULT_FIRST_N_MINUTES,
    direction_mode: str = DEFAULT_DIRECTION_MODE,
    save_formats: tuple[str, ...] = ("svg", "png"),
) -> EgocentricSummaryPlotParams:
    analysis = refined_analysis_params()
    place_cell = refined_place_cell_params()
    return EgocentricSummaryPlotParams(
        categories=tuple(categories),
        first_n_minutes=first_n_minutes,
        direction_mode=direction_mode,
        time_bin_s=0.1,
        arena_size_cm=(35.5, 20.0),
        speed_min_cm_s=3.0,
        speed_max_cm_s=60.0,
        local_spatial_bin_cm=5.0,
        n_angle_bins=10,
        occupancy_threshold_s=0.2,
        min_occupied_angle_bins=3,
        min_mean_rate_hz=0.5,
        only_plot_spikes_in_valid_spatial_bins=False,
        show_empirical_fit_curve=True,
        show_spatial_map_with_fitted_arrows=True,
        curve_polar=False,
        split_maps_placecell_style=True,
        split_map_bin_size_cm=None,
        pc_bin_size_cm=3.0,
        pc_smooth_sigma=place_cell.smooth_sigma,
        pc_occ_smooth_sigma=place_cell.occ_smooth_sigma,
        pc_min_occupancy_s=place_cell.min_occupancy_s,
        pc_use_smoothed_occ_mask=False,
        pc_kernel_size=analysis.kernel_size,
        pc_filter_type="boxcar",
        pc_speed_threshold_cm_s=analysis.speed_threshold,
        pc_min_duration_s=analysis.min_duration_s,
        pc_merge_gap_s=analysis.merge_gap_s,
        travel_smooth_window=5,
        travel_min_step=0.0,
        theta_freqs=analysis.theta_freqs,
        slow_freqs=analysis.slow_freqs,
        theta_slow_speed_threshold=analysis.speed_threshold,
        theta_slow_kernel_size=analysis.kernel_size,
        theta_slow_min_duration_s=analysis.min_duration_s,
        theta_slow_merge_gap_s=analysis.merge_gap_s,
        save_formats=tuple(save_formats),
        clear_output=False,
    )


def refined_parameter_snapshot() -> dict[str, Any]:
    return {
        "animals": list(ANIMALS),
        "notebooks_root": "notebooks_PCs_refined",
        "merged_data_filename": CLUSTER_REFINED_INPUT_FILENAME,
        "manual_refined_sidecar_filename": MANUAL_REFINED_SIDECAR_FILENAME,
        "hydration_behavior_filename": REFINED_BEHAVIOR_FILENAME,
        "analysis": asdict(refined_analysis_params()),
        "place_cell": asdict(refined_place_cell_params()),
        "traversal": asdict(refined_traversal_params()),
        "pooled": asdict(refined_pooled_params()),
        "egocentric": asdict(build_refined_egocentric_params()),
        "egocentric_summary_plot": asdict(build_refined_summary_plot_params()),
    }
