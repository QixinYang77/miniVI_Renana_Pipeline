#!/usr/bin/env python3
"""
Generate head-direction per-cell egocentric summary figures for cells tuned at 99% in any spike type.

Uses three completed run folders:
  - head_all_spike
  - head_simple_spike
  - head_complex_spike

Selection rule:
  include a cell if pass_99 is True in at least one of the three runs.

Plot behavior:
  - Row 1 uses all-spike fit (when available)
  - Row 2 uses simple-spike fit for reference/arrows/curve
  - Row 3 uses complex-spike fit for reference/arrows/curve

Output:
  <output-dir>/per_cell_summary/<category>/*.svg
  plus CSV manifests.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# --------------- path surgery (same as other HPC scripts) ---------------
HERE = Path(__file__).parent.parent.resolve()
_top_repo = str(HERE.parent)
sys.path[:] = [p for p in sys.path if os.path.normpath(p) != os.path.normpath(_top_repo)]
sys.path.insert(0, str(HERE))
os.environ['PYTHONPATH'] = str(HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    _passes_egocentric_category_gate,
    _extract_egocentric_plot_timeseries,
    _plot_egocentric_per_cell_summary_figure,
    _filter_egocentric_summary_row_by_valid_bins,
    _filter_egocentric_summary_row_for_tuning_decision,
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


def _coerce_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(v)):
            return False
        return bool(v)
    s = str(v).strip().lower()
    if s in ('nan', 'na', 'none', ''):
        return False
    return s in ('true', '1', 'yes')


def load_npz_summary_lookup(results_dir, manifest):
    """Build summary lookup dict from per-cell .npz files.

    Returns {(category, animal_id, cell_idx): dict} for successful cells.
    """
    lookup = {}
    n_found = 0
    n_missing = 0
    n_skipped = 0

    for manifest_idx, cell_info in enumerate(manifest):
        npz_path = results_dir / f'cell_{manifest_idx:04d}.npz'
        if not npz_path.exists():
            n_missing += 1
            continue

        try:
            data = dict(np.load(npz_path, allow_pickle=True))
        except Exception as exc:
            print(f'  [WARN] Failed to load {npz_path}: {exc}')
            n_missing += 1
            continue

        status = str(data.get('status', ''))
        if status != 'success':
            n_skipped += 1
            continue

        row = {}
        for key, val in data.items():
            if key == 'status':
                continue
            v = val.item() if hasattr(val, 'item') and getattr(val, 'ndim', 1) == 0 else val
            row[key] = v

        cat = str(row.get('category', cell_info['category']))
        animal = str(row.get('animal_id', cell_info['animal_id']))
        cidx = int(row.get('cell_idx', cell_info['cell_idx']))
        lookup[(cat, animal, cidx)] = row
        n_found += 1

    print(f'NPZ lookup ({results_dir.parent.name}): {n_found} success, {n_skipped} skipped, {n_missing} missing')
    return lookup


def main():
    parser = argparse.ArgumentParser(description='Generate per-cell summaries for union(pass_99 across 3 head spike types)')
    parser.add_argument('--base-dir', type=str, required=True,
                        help='Root folder containing head_all_spike/head_simple_spike/head_complex_spike')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output dir for this comparison analysis')
    parser.add_argument('--manifest-dir', type=str, default=None,
                        help='Directory containing manifest.json. Defaults to --base-dir')
    parser.add_argument('--all-run', type=str, default='head_all_spike')
    parser.add_argument('--ss-run', type=str, default='head_simple_spike')
    parser.add_argument('--cs-run', type=str, default='head_complex_spike')
    parser.add_argument('--data-root', type=str, default=str(HERE / 'data'))
    parser.add_argument('--spatial-cache-root', type=str, default=None,
                        help='Root containing per-animal spatial_analysis_full.pkl files. '
                             'Defaults to --data-root')
    parser.add_argument('--figures-root', type=str, default=str(HERE / 'figures'))
    parser.add_argument('--direction-mode', type=str, default='head', choices=['head', 'travel'])
    parser.add_argument('--first-n-minutes', type=float, default=10.0)
    parser.add_argument('--save-formats', nargs='+', default=['svg'])
    args = parser.parse_args()

    import pandas as pd

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else base_dir

    run_all = base_dir / args.all_run
    run_ss = base_dir / args.ss_run
    run_cs = base_dir / args.cs_run
    for p in [run_all, run_ss, run_cs]:
        if not p.exists():
            raise FileNotFoundError(f'Missing run folder: {p}')

    config = build_config(args.data_root, args.figures_root)
    spatial_cache_root = Path(args.spatial_cache_root) if args.spatial_cache_root else Path(args.data_root)

    manifest_path = manifest_dir / 'manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f'Loaded manifest: {len(manifest)} cells')

    lookup_all = load_npz_summary_lookup(run_all / 'per_cell_results', manifest)
    lookup_ss = load_npz_summary_lookup(run_ss / 'per_cell_results', manifest)
    lookup_cs = load_npz_summary_lookup(run_cs / 'per_cell_results', manifest)

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
        only_plot_spikes_in_valid_spatial_bins=False,
        show_empirical_fit_curve=True,
        show_spatial_map_with_fitted_arrows=True,
        curve_polar=False,
        split_maps_placecell_style=True,
        split_map_bin_size_cm=2.5,
        pc_bin_size_cm=1.5,
        pc_smooth_sigma=2.5,
        pc_occ_smooth_sigma=2.5,
        pc_min_occupancy_s=0.2,
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

    per_cell_root = output_dir / 'per_cell_summary'
    if per_cell_root.exists():
        shutil.rmtree(per_cell_root)
        print(f'Cleared previous output folder: {per_cell_root}')
    per_cell_root.mkdir(parents=True, exist_ok=True)

    print('\n--- Loading spatial data ---')
    spatial_data = classify_spatial_cells(
        data_folder=str(spatial_cache_root),
        folders=config.animals,
        cb_num_threshold=config.pooled.cb_num_threshold,
        cs_peak_rate_threshold=config.pooled.cs_peak_rate_threshold,
        snr_threshold=config.analysis.snr_threshold,
    )
    print(f'Using spatial cache root: {spatial_cache_root}')

    cells_by_animal = defaultdict(list)
    for cell_info in manifest:
        cells_by_animal[cell_info['animal_id']].append(cell_info)

    manifest_rows = []
    skip_rows = []
    counts = {
        'attempted': 0,
        'selected_any99': 0,
        'plotted': 0,
        'skipped': 0,
        'filtered_false_positive': 0,
        'filtered_low_real_mrl_all': 0,
    }

    for animal_id, cells in cells_by_animal.items():
        print(f'\n=== {animal_id} ({len(cells)} manifest cells) ===')
        animal_dir = config.data_root / animal_id
        spatial_cache_animal_dir = spatial_cache_root / animal_id

        try:
            merged = _load_merged_data(animal_dir)
            ctx = _prepare_native_analysis_context(merged, config)
            spatial_by_idx = _load_spatial_analysis_by_idx(spatial_cache_animal_dir)
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
            category = str(cell_info['category'])
            cell_idx = int(cell_info['cell_idx'])
            counts['attempted'] += 1

            key = (category, str(animal_id), int(cell_idx))
            row_all = lookup_all.get(key)
            row_ss = lookup_ss.get(key)
            row_cs = lookup_cs.get(key)
            row_all = _filter_egocentric_summary_row_by_valid_bins(row_all, min_valid_bins=5)
            row_ss = _filter_egocentric_summary_row_by_valid_bins(row_ss, min_valid_bins=5)
            row_cs = _filter_egocentric_summary_row_by_valid_bins(row_cs, min_valid_bins=5)

            cat_dir = per_cell_root / category
            cat_dir.mkdir(parents=True, exist_ok=True)

            analysis = spatial_by_idx.get(cell_idx)
            if not isinstance(analysis, dict):
                skip_rows.append({
                    'category': category, 'animal_id': animal_id,
                    'cell_idx': cell_idx, 'reason': 'missing_spatial_analysis',
                })
                counts['skipped'] += 1
                continue
            # Force split-map smoothing parameters from this run config, rather than
            # inheriting precomputed analysis defaults from spatial_analysis_full.pkl.
            analysis_for_plot = dict(analysis)
            analysis_params_for_plot = (
                dict(analysis.get('params', {}))
                if isinstance(analysis.get('params', {}), dict)
                else {}
            )
            analysis_params_for_plot['smooth_sigma'] = float(params.pc_smooth_sigma)
            analysis_params_for_plot['occ_smooth_sigma'] = float(params.pc_occ_smooth_sigma)
            analysis_for_plot['params'] = analysis_params_for_plot

            pass_gate, gate_reason = _passes_egocentric_category_gate(
                category=category, analysis=analysis,
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

            # Primary row drives row-1/all-spike title text and default fit fallback.
            row_primary = None
            if isinstance(row_all, dict):
                row_primary = dict(row_all)
            elif isinstance(row_ss, dict):
                row_primary = dict(row_ss)
            elif isinstance(row_cs, dict):
                row_primary = dict(row_cs)
            else:
                row_primary = {
                    'category': category,
                    'animal_id': animal_id,
                    'cell_idx': int(cell_idx),
                }

            bad_mask = np.asarray(ctx['bad_masks'][cell_idx], dtype=bool)

            try:
                plot_data = _extract_egocentric_plot_timeseries(
                    ctx=ctx,
                    cell_idx=int(cell_idx),
                    bad_mask=bad_mask,
                    analysis=analysis_for_plot,
                    params=params,
                )
                row_all = _filter_egocentric_summary_row_for_tuning_decision(
                    row_all,
                    local_tuning=plot_data.get('local_tuning_all'),
                    min_valid_bins=5,
                )
                row_ss = _filter_egocentric_summary_row_for_tuning_decision(
                    row_ss,
                    local_tuning=plot_data.get('local_tuning_ss'),
                    min_valid_bins=5,
                )
                row_cs = _filter_egocentric_summary_row_for_tuning_decision(
                    row_cs,
                    local_tuning=plot_data.get('local_tuning_cs'),
                    min_valid_bins=5,
                )
                pass99_all = _coerce_bool((row_all or {}).get('pass_99', False))
                pass99_ss = _coerce_bool((row_ss or {}).get('pass_99', False))
                pass99_cs = _coerce_bool((row_cs or {}).get('pass_99', False))
                pass99_any = bool(pass99_all or pass99_ss or pass99_cs)
                if not pass99_any:
                    continue
                counts['selected_any99'] += 1
                real_mrl_all = np.nan
                if isinstance(row_all, dict):
                    try:
                        real_mrl_all = float(row_all.get('real_mrl', np.nan))
                    except Exception:
                        real_mrl_all = np.nan
                if np.isfinite(real_mrl_all) and (real_mrl_all < 0.2):
                    skip_rows.append({
                        'category': category,
                        'animal_id': animal_id,
                        'cell_idx': cell_idx,
                        'reason': f'low_real_mrl_all_lt0p2(real_mrl_all={real_mrl_all:.3f})',
                    })
                    counts['filtered_low_real_mrl_all'] += 1
                    counts['skipped'] += 1
                    print(
                        f'  [SKIP] {category} / cell {cell_idx + 1}: '
                        f'low real_mrl_all ({real_mrl_all:.3f} < 0.200)'
                    )
                    continue
                pass95_any = any(_coerce_bool((r or {}).get('pass_95', False)) for r in (row_all, row_ss, row_cs))
                pass100_any = any(_coerce_bool((r or {}).get('pass_100', False)) for r in (row_all, row_ss, row_cs))
                row_primary['pass_95'] = bool(pass95_any)
                row_primary['pass_99'] = bool(pass99_any)
                row_primary['pass_100'] = bool(pass100_any)

                out_base = cat_dir / f'{animal_id}_cell{cell_idx + 1:03d}_egocentric_summary_any99_3spike'
                plot_meta = _plot_egocentric_per_cell_summary_figure(
                    category=category,
                    animal_id=str(animal_id),
                    cell_idx=int(cell_idx),
                    mode=str(args.direction_mode),
                    data=plot_data,
                    summary_row=row_primary,
                    summary_row_all=row_all,
                    summary_row_ss=row_ss,
                    summary_row_cs=row_cs,
                    params=params,
                    out_base=out_base,
                )
                plt.close('all')

                green_all = int(plot_meta.get('green_ring_count_all', 0))
                green_ss = int(plot_meta.get('green_ring_count_ss', 0))
                green_cs = int(plot_meta.get('green_ring_count_cs', 0))
                if max(green_all, green_ss, green_cs) < 3:
                    for saved in list(plot_meta.get('saved_paths', [])):
                        try:
                            Path(saved).unlink(missing_ok=True)
                        except Exception:
                            pass
                    skip_rows.append({
                        'category': category,
                        'animal_id': animal_id,
                        'cell_idx': cell_idx,
                        'reason': f'false_positive_green_rings_lt3(all={green_all},ss={green_ss},cs={green_cs})',
                    })
                    counts['filtered_false_positive'] += 1
                    counts['skipped'] += 1
                    print(
                        f'  [SKIP] {category} / cell {cell_idx + 1}: '
                        f'false_positive (green rings all/ss/cs={green_all}/{green_ss}/{green_cs})'
                    )
                    continue

                counts['plotted'] += 1
                manifest_rows.append({
                    'category': category,
                    'animal_id': animal_id,
                    'cell_idx': cell_idx,
                    'cell_num': cell_idx + 1,
                    'selected_by_any_pass99': True,
                    'pass99_all': bool(pass99_all),
                    'pass99_ss': bool(pass99_ss),
                    'pass99_cs': bool(pass99_cs),
                    'real_mrl_all': float(real_mrl_all) if np.isfinite(real_mrl_all) else np.nan,
                    'valid_bin_mean_mrl_all': plot_meta.get('valid_bin_mean_mrl_all', np.nan),
                    'valid_bin_mean_mrl_ss': plot_meta.get('valid_bin_mean_mrl_ss', np.nan),
                    'valid_bin_mean_mrl_cs': plot_meta.get('valid_bin_mean_mrl_cs', np.nan),
                    'valid_bin_mrl_n_all': plot_meta.get('valid_bin_mrl_n_all', np.nan),
                    'valid_bin_mrl_n_ss': plot_meta.get('valid_bin_mrl_n_ss', np.nan),
                    'valid_bin_mrl_n_cs': plot_meta.get('valid_bin_mrl_n_cs', np.nan),
                    'has_best_reference': bool(plot_meta.get('has_best_reference', False)),
                    'n_subplots': int(plot_meta.get('n_subplots', 0)),
                    'real_mrl_primary': plot_meta.get('real_mrl', np.nan),
                    'pass_95_any': str(plot_meta.get('pass_95', '')),
                    'pass_99_any': str(plot_meta.get('pass_99', '')),
                    'pass_100_any': str(plot_meta.get('pass_100', '')),
                    'green_rings_all': int(green_all),
                    'green_rings_ss': int(green_ss),
                    'green_rings_cs': int(green_cs),
                    'saved_paths': ';'.join(list(plot_meta.get('saved_paths', []))),
                })
                print(f'  [OK] {category} / cell {cell_idx + 1} (all={pass99_all}, ss={pass99_ss}, cs={pass99_cs})')

            except Exception as exc:
                skip_rows.append({
                    'category': category, 'animal_id': animal_id,
                    'cell_idx': cell_idx,
                    'reason': f'plot_failed({type(exc).__name__}: {exc})',
                })
                counts['skipped'] += 1
                print(f'  [SKIP] {category} / cell {cell_idx + 1}: {exc}')
                continue

    manifest_df = pd.DataFrame(manifest_rows)
    skip_df = pd.DataFrame(skip_rows)
    manifest_csv = per_cell_root / 'egocentric_per_cell_plot_manifest_any99_3spike.csv'
    skip_csv = per_cell_root / 'egocentric_per_cell_plot_skipped_any99_3spike.csv'
    manifest_df.to_csv(manifest_csv, index=False)
    skip_df.to_csv(skip_csv, index=False)

    print(f'\n{"=" * 60}')
    print(f'Attempted manifest cells:          {counts["attempted"]}')
    print(f'Selected by any pass_99 (3 runs): {counts["selected_any99"]}')
    print(f'Filtered low all real_mrl<0.2:    {counts["filtered_low_real_mrl_all"]}')
    print(f'Filtered false positives:         {counts["filtered_false_positive"]}')
    print(f'Plotted:                          {counts["plotted"]}')
    print(f'Skipped after selection:          {counts["skipped"]}')
    print(f'Manifest:                         {manifest_csv}')
    print(f'Skipped:                          {skip_csv}')


if __name__ == '__main__':
    main()
