#!/usr/bin/env python3
"""
Step 1 of 3: Prepare for parallel egocentric analysis.

- Ensures per-animal cache exists (spatial_analysis_full.pkl etc.)
- Builds a cell manifest listing every (category, animal_id, cell_idx)
  that will be analyzed.
- Saves the manifest as JSON so array tasks can pick their cell by index.

Submitted by HPC_egocentric_job_submission.ipynb via sbatch.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent.parent.resolve()
# Remove any top-level repo paths that would shadow our utils/ with the top-level utils/
_top_repo = str(HERE.parent)
sys.path[:] = [p for p in sys.path if os.path.normpath(p) != os.path.normpath(_top_repo)]
sys.path.insert(0, str(HERE))
os.environ['PYTHONPATH'] = str(HERE)

from utils.placecell_pipeline import (
    AnalysisParams,
    PlaceCellParams,
    PFTraversalParams,
    PooledParams,
    CachePolicy,
    PipelineConfig,
    ensure_cache_for_all_animals,
    _get_spatial_category_cells,
    _normalize_pf_category_name,
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


def build_config(data_root, figures_root, force_recompute=False):
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
            force_recompute=force_recompute, validate_only=False,
            save_executed_notebooks=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=str(HERE / 'data'))
    parser.add_argument('--figures-root', type=str, default=str(HERE / 'figures'))
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to write manifest.json and per-cell results')
    parser.add_argument('--categories', nargs='+', default=['CSplus', 'CSminus'])
    parser.add_argument('--force-recompute', action='store_true')
    args = parser.parse_args()

    config = build_config(args.data_root, args.figures_root, args.force_recompute)

    # Step 1: Ensure cache
    print('--- Ensuring cache for all animals ---')
    statuses = ensure_cache_for_all_animals(config, force=args.force_recompute)
    for st in statuses:
        print(f'  [{st.action}] {st.animal_id}')

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

    # Step 3: Build cell manifest
    categories = [_normalize_pf_category_name(c) for c in args.categories]
    manifest = []
    for category in categories:
        cells = _get_spatial_category_cells(spatial_data, category)
        for cell_meta in cells:
            manifest.append({
                'category': category,
                'animal_id': str(cell_meta.get('session', '')),
                'cell_idx': int(cell_meta.get('cell_idx', -1)),
            })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'per_cell_results').mkdir(exist_ok=True)

    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\n--- Manifest ---')
    print(f'Total cells: {len(manifest)}')
    for cat in categories:
        n = sum(1 for m in manifest if m['category'] == cat)
        print(f'  {cat}: {n}')
    print(f'Saved: {manifest_path}')
    print(f'Array range: 0-{len(manifest) - 1}')


if __name__ == '__main__':
    main()
