#!/usr/bin/env python3
"""
SLURM job script: run egocentric tuning analysis on the cluster.

Submitted by HPC_egocentric_job_submission.ipynb via sbatch.
"""
import argparse
import os
import sys
from pathlib import Path

# Add miniVI_PlaceCell_analysis_V4 to path so utils can be imported
HERE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(HERE))

from utils.placecell_pipeline import (
    AnalysisParams,
    PlaceCellParams,
    PFTraversalParams,
    PooledParams,
    CachePolicy,
    PipelineConfig,
    EgocentricTuningParams,
    ensure_cache_for_all_animals,
    run_pooled_egocentric_tuning_analysis,
)
from utils.spatial_heatmaps import classify_spatial_cells


def parse_args():
    parser = argparse.ArgumentParser(description='Run pooled egocentric tuning analysis on the cluster.')
    parser.add_argument('--data-root', type=str, default=str(HERE / 'data'),
                        help='Path to data root containing per-animal folders (default: <repo>/data)')
    parser.add_argument('--figures-root', type=str, default=str(HERE / 'figures'),
                        help='Path to figures root (default: <repo>/figures)')
    parser.add_argument('--n-jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help='Number of parallel jobs for surrogate computation')
    parser.add_argument('--direction-mode', type=str, default='head', choices=['head', 'travel'],
                        help='Egocentric direction mode: head or travel')
    parser.add_argument('--n-surrogates', type=int, default=1000,
                        help='Number of surrogates for significance testing')
    parser.add_argument('--first-n-minutes', type=float, default=10.0,
                        help='Only use first N minutes of each session')
    parser.add_argument('--force-recompute', action='store_true',
                        help='Force recompute of all cached artifacts')
    return parser.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    figures_root = Path(args.figures_root)
    figures_root.mkdir(parents=True, exist_ok=True)

    print(f'data_root:      {data_root}')
    print(f'figures_root:   {figures_root}')
    print(f'n_jobs:         {args.n_jobs}')
    print(f'direction_mode: {args.direction_mode}')
    print(f'n_surrogates:   {args.n_surrogates}')

    animals = [
        'CKII_pAce21_PR_20250806',
        'CKII_pAce38_PX_20251126',
        'CKII_pAce45_PX_20260118',
        'CKII_pAce47_PX_20260128',
        'CKII_pAce46_PR_20260222',
        'CKII_pAce50_PRL_20260317',
    ]

    config = PipelineConfig(
        project_root=HERE,
        data_root=data_root,
        figures_root=figures_root,
        notebooks_root=HERE / 'notebooks_PCs',
        animals=animals,
        analysis=AnalysisParams(
            speed_threshold=3.0,
            speed_threshold_quiet=0.5,
            min_duration_s=0.25,
            merge_gap_s=0.0,
            kernel_size=51,
            snr_threshold=3.0,
            min_good_minutes=5.0,
            theta_freqs=(4.0, 8.0),
            slow_freqs=2.0,
        ),
        place_cell=PlaceCellParams(
            bin_size=1.5,
            place_field_threshold=0.35,
            min_component_peak_ratio=0.45,
            split_multi_peak_fields=True,
            split_secondary_peak_ratio=0.6,
            split_secondary_peak_min_separation_cm=6.0,
            min_peak_rate=0.5,
            max_field_area_ratio=0.5,
            min_field_bins=10,
            min_pf_reliability=0.2,
            min_pf_traversals=5,
            pf_reliability_dilation_bins=3,
            pf_reliability_dilation_shape='disk',
            smooth_sigma=1.5,
            min_occupancy_s=0.001,
            occ_smooth_sigma=1.5,
            num_shuffles=1000,
            random_seed=42,
            ss_shape_min_separation_ms=14.0,
            trim_sparse_top_row_for_analysis=True,
            trim_sparse_top_row_for_plotting=True,
            sparse_top_row_nonocc_frac_threshold=0.8,
        ),
        traversal=PFTraversalParams(
            center_by_pf_position=True,
            pf_component_selection='peak_rate',
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
        ),
        pooled=PooledParams(
            cb_num_threshold=5,
            cs_peak_rate_threshold=0.5,
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
        ),
        cache=CachePolicy(
            force_recompute=args.force_recompute,
            validate_only=False,
            save_executed_notebooks=False,
        ),
    )

    # Step 1: Ensure per-animal cache exists
    print('\n--- Ensuring cache for all animals ---')
    statuses = ensure_cache_for_all_animals(config, force=args.force_recompute)
    for st in statuses:
        print(f'  [{st.action}] {st.animal_id}')
        if st.reasons:
            print(f'    reasons: {st.reasons}')

    # Step 2: Load spatial data
    print('\n--- Loading spatial data ---')
    spatial_data = classify_spatial_cells(
        data_folder=str(config.data_root),
        folders=config.animals,
        cb_num_threshold=config.pooled.cb_num_threshold,
        cs_peak_rate_threshold=config.pooled.cs_peak_rate_threshold,
        snr_threshold=config.analysis.snr_threshold,
    )
    print(f'  CS+ PLCs:  {len(spatial_data.plcs_csplus)}')
    print(f'  CS- PLCs:  {len(spatial_data.plcs_csminus)}')
    print(f'  Non-PLCs:  {len(spatial_data.non_plcs)}')

    # Step 3: Run egocentric tuning analysis
    figure_save_folder = str(figures_root / 'CKII_pooled')
    os.makedirs(figure_save_folder, exist_ok=True)

    egocentric_params = EgocentricTuningParams(
        categories=('CSplus', 'CSminus'),
        first_n_minutes=args.first_n_minutes,
        direction_mode=args.direction_mode,
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
        optimizer_method='Nelder-Mead',
        n_surrogates=args.n_surrogates,
        n_jobs=args.n_jobs,
        surrogate_chunk_size=20,
        random_seed=42,
        clear_output=True,
        save_null_distributions=True,
        show_progress=True,
    )

    print(f'\n--- Running egocentric tuning analysis ---')
    print(f'  direction_mode:   {args.direction_mode}')
    print(f'  n_surrogates:     {args.n_surrogates}')
    print(f'  n_jobs:           {args.n_jobs}')
    print(f'  first_n_minutes:  {args.first_n_minutes}')

    egocentric_summary = run_pooled_egocentric_tuning_analysis(
        config=config,
        spatial_data=spatial_data,
        egocentric_params=egocentric_params,
        figure_save_folder=figure_save_folder,
    )

    print(f'\nOutput folder: {egocentric_summary["figure_root"]}')
    print('Attempted cells by category:')
    for cat, n in egocentric_summary['attempted_cells_by_category'].items():
        print(f'  {cat}: {n}')
    print('Successful cells by category:')
    for cat, n in egocentric_summary['successful_cells_by_category'].items():
        print(f'  {cat}: {n}')
    print('Skipped cells by category:')
    for cat, n in egocentric_summary['skipped_cells_by_category'].items():
        print(f'  {cat}: {n}')

    # Save summary CSV
    df = egocentric_summary.get('summary_df')
    if df is not None and len(df) > 0:
        csv_path = Path(egocentric_summary['figure_root']) / 'egocentric_summary.csv'
        df.to_csv(csv_path, index=False)
        print(f'Saved summary CSV: {csv_path}')
    else:
        print('No successful fits — no CSV saved.')

    print('\nDone.')


if __name__ == '__main__':
    main()
