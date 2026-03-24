#!/usr/bin/env python3
"""
Step 5a: Generate per-cell egocentric summary plots on the cluster.

Iterates through all cells in the manifest, extracts plot timeseries
(including theta amplitude and slow Vm from traces), and generates
per-cell summary figures organized by category folder.

Uses animal-level caching so each animal's data is loaded only once.

Submitted by HPC_egocentric_job_submission.ipynb via sbatch.
"""
import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

# --------------- path surgery (same as other HPC scripts) ---------------
HERE = Path(__file__).parent.parent.resolve()
_top_repo = str(HERE.parent)
sys.path[:] = [p for p in sys.path if os.path.normpath(p) != os.path.normpath(_top_repo)]
sys.path.insert(0, str(HERE))
os.environ['PYTHONPATH'] = str(HERE)

import matplotlib
matplotlib.use('Agg')  # headless backend for cluster

from utils.placecell_pipeline import (
    AnalysisParams,
    PlaceCellParams,
    PFTraversalParams,
    PooledParams,
    CachePolicy,
    PipelineConfig,
    EgocentricSummaryPlotParams,
    _load_merged_data,
    _prepare_native_analysis_context,
    _load_spatial_analysis_by_idx,
    _get_spatial_category_cells,
    _normalize_pf_category_name,
    _passes_egocentric_category_gate,
    _extract_egocentric_plot_timeseries,
    _plot_egocentric_per_cell_summary_figure,
    _build_egocentric_summary_lookup,
)
from utils.spatial_heatmaps import classify_spatial_cells


ANIMALS = [
    'CKII_pAce21_PR_20250806',
    'CKII_pAce38_PX_20251126',
    'CKII_pAce45_PX_20260118',
    'CKII_pAce47_PX_20260128',
    'CKII_pAce46_PR_20260222',
    'CKII_pAce50_PRL_20260317',
]


def build_config(data_root, figures_root):
    return PipelineConfig(
        project_root=HERE,
        data_root=Path(data_root),
        figures_root=Path(figures_root),
        notebooks_root=HERE / 'notebooks_PCs',
        animals=ANIMALS,
        analysis=AnalysisParams(
            speed_threshold=3.0, speed_threshold_quiet=0.5, min_duration_s=0.25,
            merge_gap_s=0.0, kernel_size=51, snr_threshold=3.0, min_good_minutes=5.0,
            theta_freqs=(4.0, 8.0), slow_freqs=2.0,
        ),
        place_cell=PlaceCellParams(
            bin_size=1.5, place_field_threshold=0.35, min_component_peak_ratio=0.45,
            split_multi_peak_fields=True, split_secondary_peak_ratio=0.6,
            split_secondary_peak_min_separation_cm=6.0, min_peak_rate=0.5,
            max_field_area_ratio=0.5, min_field_bins=10, min_pf_reliability=0.2,
            min_pf_traversals=5, pf_reliability_dilation_bins=3,
            pf_reliability_dilation_shape='disk', smooth_sigma=1.5, min_occupancy_s=0.001,
            occ_smooth_sigma=1.5, num_shuffles=1000, random_seed=42,
            ss_shape_min_separation_ms=14.0, trim_sparse_top_row_for_analysis=True,
            trim_sparse_top_row_for_plotting=True, sparse_top_row_nonocc_frac_threshold=0.8,
        ),
        traversal=PFTraversalParams(
            center_by_pf_position=True, pf_component_selection='peak_rate',
            min_duration_ms=100.0, min_distance_cm=5.0, traversal_merge_gap_s=2.0,
            clear_traversal=False, session_indices=(0, 1), pf_center_window_sec=10.0,
            min_traversals=10, firing_rate_bin_ms=100.0, firing_rate_smooth_ms=50.0,
            subtract_pre_traversal_baseline=False, mask_non_traversal_pf=True,
            max_pf_distance_cm=8.0, plateau_min_duration_ms=100.0,
        ),
        pooled=PooledParams(
            cb_num_threshold=5, cs_peak_rate_threshold=0.5, run_psd_sections=True,
            cs_plc_only=True, psd_speed_threshold=3, psd_chunk_s=2.0, psd_nperseg_s=1.0,
            psd_noverlap_frac=0.5, simple_event_window_ms=80.0, simple_event_min_gap_ms=50.0,
            min_chunk_valid_fraction=1.0, max_freq=100.0, normalize_psd=True,
            norm_freq_range=(20.0, 100.0),
        ),
        cache=CachePolicy(
            force_recompute=False, validate_only=False, save_executed_notebooks=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description='Generate per-cell egocentric summary plots')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Direction/spike-specific output dir (e.g. .../head_all_spike)')
    parser.add_argument('--manifest-dir', type=str, default=None,
                        help='Directory containing manifest.json (shared). Defaults to --output-dir.')
    parser.add_argument('--data-root', type=str, default=str(HERE / 'data'))
    parser.add_argument('--figures-root', type=str, default=str(HERE / 'figures'))
    parser.add_argument('--direction-mode', type=str, default='head', choices=['head', 'travel'])
    parser.add_argument('--first-n-minutes', type=float, default=10.0)
    parser.add_argument('--save-formats', nargs='+', default=['svg', 'png'])
    args = parser.parse_args()

    import pandas as pd

    output_dir = Path(args.output_dir)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else output_dir

    config = build_config(args.data_root, args.figures_root)

    # Load manifest
    manifest_path = manifest_dir / 'manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f'Loaded manifest: {len(manifest)} cells')

    # Load summary CSV from aggregation step (for fit parameters / pass status)
    summary_csv = output_dir / 'egocentric_tuning_summary.csv'
    summary_lookup = _build_egocentric_summary_lookup(
        summary_csv=summary_csv,
        summary_df=None,
    )
    print(f'Loaded summary lookup: {len(summary_lookup)} entries')

    # Build plot params matching the notebook configuration
    params = EgocentricSummaryPlotParams(
        categories=tuple(sorted(set(m['category'] for m in manifest))),
        first_n_minutes=args.first_n_minutes,
        direction_mode=args.direction_mode,
        time_bin_s=0.1,
        arena_size_cm=(35.5, 20.0),
        speed_min_cm_s=3.0,
        speed_max_cm_s=60.0,
        local_spatial_bin_cm=5.0,
        n_angle_bins=10,
        occupancy_threshold_s=0.2,
        min_occupied_angle_bins=3,
        min_mean_rate_hz=0.5,
        only_plot_spikes_in_valid_spatial_bins=True,
        show_empirical_fit_curve=True,
        show_spatial_map_with_fitted_arrows=True,
        curve_polar=False,
        split_maps_placecell_style=True,
        split_map_bin_size_cm=None,
        pc_bin_size_cm=1.5,
        pc_smooth_sigma=1.5,
        pc_occ_smooth_sigma=1.5,
        pc_min_occupancy_s=0.001,
        pc_use_smoothed_occ_mask=False,
        pc_kernel_size=51,
        pc_filter_type='boxcar',
        pc_speed_threshold_cm_s=3.0,
        pc_min_duration_s=0.25,
        pc_merge_gap_s=0.0,
        travel_smooth_window=5,
        travel_min_step=0.0,
        theta_freqs=(4.0, 8.0),
        slow_freqs=2.0,
        theta_slow_speed_threshold=3.0,
        theta_slow_kernel_size=51,
        theta_slow_min_duration_s=0.25,
        theta_slow_merge_gap_s=0.0,
        save_formats=tuple(args.save_formats),
        clear_output=False,
    )

    # Output structure: per_cell_summary/<category>/...
    per_cell_root = output_dir / 'per_cell_summary'
    per_cell_root.mkdir(parents=True, exist_ok=True)

    # Load spatial data (needed for category gate validation)
    print('\n--- Loading spatial data ---')
    spatial_data = classify_spatial_cells(
        data_folder=str(config.data_root),
        folders=config.animals,
        cb_num_threshold=config.pooled.cb_num_threshold,
        cs_peak_rate_threshold=config.pooled.cs_peak_rate_threshold,
        snr_threshold=config.analysis.snr_threshold,
    )

    # Organize manifest by animal to minimize data reloading
    from collections import defaultdict
    cells_by_animal = defaultdict(list)
    for idx, cell_info in enumerate(manifest):
        cells_by_animal[cell_info['animal_id']].append(cell_info)

    mode = args.direction_mode
    manifest_rows = []
    skip_rows = []
    counts = {'attempted': 0, 'plotted': 0, 'skipped': 0}

    for animal_id, cells in cells_by_animal.items():
        print(f'\n=== {animal_id} ({len(cells)} cells) ===')
        animal_dir = config.data_root / animal_id

        try:
            merged = _load_merged_data(animal_dir)
            ctx = _prepare_native_analysis_context(merged, config)
            spatial_by_idx = _load_spatial_analysis_by_idx(animal_dir)
        except Exception as exc:
            print(f'  [ERROR] Failed to load data: {exc}')
            for cell_info in cells:
                skip_rows.append({
                    'category': cell_info['category'],
                    'animal_id': animal_id,
                    'cell_idx': cell_info['cell_idx'],
                    'reason': f'data_load_failed({exc})',
                })
            counts['skipped'] += len(cells)
            continue

        for cell_info in cells:
            category = cell_info['category']
            cell_idx = int(cell_info['cell_idx'])
            counts['attempted'] += 1

            cat_dir = per_cell_root / str(category)
            cat_dir.mkdir(parents=True, exist_ok=True)

            # Validate cell
            analysis = spatial_by_idx.get(cell_idx)
            if not isinstance(analysis, dict):
                skip_rows.append({
                    'category': category, 'animal_id': animal_id,
                    'cell_idx': cell_idx, 'reason': 'missing_spatial_analysis',
                })
                counts['skipped'] += 1
                continue

            pass_gate, gate_reason = _passes_egocentric_category_gate(
                category=str(category), analysis=analysis,
            )
            if not pass_gate:
                skip_rows.append({
                    'category': category, 'animal_id': animal_id,
                    'cell_idx': cell_idx, 'reason': gate_reason or 'category_gate_failed',
                })
                counts['skipped'] += 1
                continue

            if cell_idx >= int(ctx.get('n_cells', 0)):
                skip_rows.append({
                    'category': category, 'animal_id': animal_id,
                    'cell_idx': cell_idx, 'reason': 'cell_idx_out_of_range',
                })
                counts['skipped'] += 1
                continue

            eligible = np.asarray(ctx.get('eligible_cells', np.array([])), dtype=bool)
            if cell_idx >= eligible.size or not bool(eligible[cell_idx]):
                skip_rows.append({
                    'category': category, 'animal_id': animal_id,
                    'cell_idx': cell_idx, 'reason': 'cell_not_eligible',
                })
                counts['skipped'] += 1
                continue

            bad_mask = np.asarray(ctx['bad_masks'][cell_idx], dtype=bool)
            summary_row = summary_lookup.get((str(category), str(animal_id), int(cell_idx)))

            try:
                plot_data = _extract_egocentric_plot_timeseries(
                    ctx=ctx,
                    cell_idx=int(cell_idx),
                    bad_mask=bad_mask,
                    analysis=analysis,
                    params=params,
                )
                out_base = cat_dir / f'{animal_id}_cell{cell_idx + 1:03d}_egocentric_summary'
                plot_meta = _plot_egocentric_per_cell_summary_figure(
                    category=str(category),
                    animal_id=str(animal_id),
                    cell_idx=int(cell_idx),
                    mode=str(mode),
                    data=plot_data,
                    summary_row=summary_row,
                    params=params,
                    out_base=out_base,
                )
                import matplotlib.pyplot as plt
                plt.close('all')  # free memory

                counts['plotted'] += 1
                manifest_rows.append({
                    'category': category,
                    'animal_id': animal_id,
                    'cell_idx': cell_idx,
                    'cell_num': cell_idx + 1,
                    'has_best_reference': bool(plot_meta.get('has_best_reference', False)),
                    'n_subplots': int(plot_meta.get('n_subplots', 0)),
                    'real_mrl': plot_meta.get('real_mrl', np.nan),
                    'pass_95': str(plot_meta.get('pass_95', '')),
                    'pass_99': str(plot_meta.get('pass_99', '')),
                    'pass_100': str(plot_meta.get('pass_100', '')),
                    'saved_paths': ';'.join(list(plot_meta.get('saved_paths', []))),
                })
                print(f'  [OK] {category} / cell {cell_idx + 1}')

            except Exception as exc:
                skip_rows.append({
                    'category': category, 'animal_id': animal_id,
                    'cell_idx': cell_idx, 'reason': f'plot_failed({type(exc).__name__}: {exc})',
                })
                counts['skipped'] += 1
                print(f'  [SKIP] {category} / cell {cell_idx + 1}: {exc}')
                continue

    # Save manifests
    manifest_df = pd.DataFrame(manifest_rows)
    skip_df = pd.DataFrame(skip_rows)
    manifest_csv = per_cell_root / 'egocentric_per_cell_plot_manifest.csv'
    skip_csv = per_cell_root / 'egocentric_per_cell_plot_skipped.csv'
    manifest_df.to_csv(manifest_csv, index=False)
    skip_df.to_csv(skip_csv, index=False)

    print(f'\n{"=" * 50}')
    print(f'Attempted: {counts["attempted"]}')
    print(f'Plotted:   {counts["plotted"]}')
    print(f'Skipped:   {counts["skipped"]}')
    print(f'Manifest:  {manifest_csv}')
    print(f'Skipped:   {skip_csv}')

    # Per-category breakdown
    for cat in sorted(set(m['category'] for m in manifest)):
        n_plotted = sum(1 for r in manifest_rows if r['category'] == cat)
        n_skipped = sum(1 for r in skip_rows if r['category'] == cat)
        print(f'  {cat}: plotted={n_plotted}, skipped={n_skipped}')


if __name__ == '__main__':
    main()
