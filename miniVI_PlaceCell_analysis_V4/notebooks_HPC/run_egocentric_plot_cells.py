#!/usr/bin/env python3
"""
Step 5a: Generate per-cell egocentric summary plots on the cluster.

Reads per-cell .npz results directly from per_cell_results/ (no aggregation
step needed). Iterates through all cells in the manifest, loads the
corresponding .npz to build the summary_row, extracts plot timeseries,
and generates per-cell summary figures organized by category folder.

Uses animal-level caching so each animal's data is loaded only once.

Tolerates missing .npz files (e.g. jobs that haven't finished yet).

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
_top_repo_norm = os.path.normpath(_top_repo)
sys.path[:] = [
    p for p in sys.path
    if os.path.normpath(os.path.abspath(p or os.getcwd())) != _top_repo_norm
]
sys.path.insert(0, str(HERE))
os.environ['PYTHONPATH'] = str(HERE)

import matplotlib
matplotlib.use('Agg')  # headless backend for cluster
import matplotlib.pyplot as plt

from notebooks_HPC.egocentric_refined_config import (
    ANIMALS,
    DEFAULT_DIRECTION_MODE,
    DEFAULT_FIRST_N_MINUTES,
    build_refined_config,
    build_refined_summary_plot_params,
)
from utils.placecell_pipeline import (
    _resolve_merged_data_path,
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


def build_config(data_root, figures_root):
    return build_refined_config(
        HERE,
        data_root,
        figures_root,
        force_recompute=False,
    )


def load_npz_summary_lookup(results_dir, manifest):
    """Build summary_lookup dict from per-cell .npz files.

    Returns {(category, animal_id, cell_idx): dict} for all successful cells.
    Missing or skipped .npz files are silently ignored.
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

        # Build a summary row dict from the .npz fields
        row = {}
        for key, val in data.items():
            if key == 'status':
                continue
            # np.load wraps scalars in 0-d arrays
            v = val.item() if hasattr(val, 'item') and val.ndim == 0 else val
            row[key] = v
        row = _filter_egocentric_summary_row_by_valid_bins(row, min_valid_bins=5)

        cat = str(row.get('category', cell_info['category']))
        animal = str(row.get('animal_id', cell_info['animal_id']))
        cidx = int(row.get('cell_idx', cell_info['cell_idx']))
        lookup[(cat, animal, cidx)] = row
        n_found += 1

    print(f'NPZ lookup: {n_found} success, {n_skipped} skipped, {n_missing} missing')
    return lookup


def main():
    parser = argparse.ArgumentParser(description='Generate per-cell egocentric summary plots')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Direction/spike-specific output dir (e.g. .../head_all_spike)')
    parser.add_argument('--manifest-dir', type=str, default=None,
                        help='Directory containing manifest.json (shared). Defaults to --output-dir.')
    parser.add_argument('--data-root', type=str, default=str(HERE / 'data'))
    parser.add_argument('--figures-root', type=str, default=str(HERE / 'figures'))
    parser.add_argument('--direction-mode', type=str, default=DEFAULT_DIRECTION_MODE, choices=['head', 'travel'])
    parser.add_argument('--spike-type', type=str, default='all_spike',
                        choices=['all_spike', 'simple_spike', 'complex_spike', 'complex_burst'],
                        help='Primary spike source for tuning filters and row-1 summary panels')
    parser.add_argument('--first-n-minutes', type=float, default=DEFAULT_FIRST_N_MINUTES)
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

    # Build summary lookup directly from .npz files (no aggregation step needed)
    results_dir = output_dir / 'per_cell_results'
    summary_lookup = load_npz_summary_lookup(results_dir, manifest)

    # Build plot params matching the refined notebook configuration
    params = build_refined_summary_plot_params(
        categories=tuple(sorted(set(m['category'] for m in manifest))),
        first_n_minutes=args.first_n_minutes,
        direction_mode=args.direction_mode,
        save_formats=tuple(args.save_formats),
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
        cs_plc_definition_mode=config.pooled.cs_plc_definition_mode,
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
            resolved_data_path = _resolve_merged_data_path(animal_dir, config)
            print(f'  Runtime merged data: {resolved_data_path}')
            print(f'  Spatial analysis:    {animal_dir / "spatial_analysis_full.pkl"}')
            merged = _load_merged_data(animal_dir, config)
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

            # Check if .npz result exists for this cell
            summary_row = summary_lookup.get((str(category), str(animal_id), int(cell_idx)))
            if summary_row is None:
                skip_rows.append({
                    'category': category, 'animal_id': animal_id,
                    'cell_idx': cell_idx, 'reason': 'npz_missing_or_skipped',
                })
                counts['skipped'] += 1
                print(f'  [SKIP] {category} / cell {cell_idx + 1}: no .npz result')
                continue

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

            try:
                plot_data = _extract_egocentric_plot_timeseries(
                    ctx=ctx,
                    cell_idx=int(cell_idx),
                    bad_mask=bad_mask,
                    analysis=analysis,
                    params=params,
                    spike_type=args.spike_type,
                )
                summary_row = _filter_egocentric_summary_row_for_tuning_decision(
                    summary_row,
                    local_tuning=plot_data.get('local_tuning'),
                    min_valid_bins=5,
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
