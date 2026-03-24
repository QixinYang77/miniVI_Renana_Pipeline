#!/usr/bin/env python3
"""
Step 2 of 3: Process a single cell's egocentric tuning analysis.

Reads the manifest written by run_egocentric_prepare.py, picks the cell
at --cell-index (typically $SLURM_ARRAY_TASK_ID), runs the analysis,
and saves the result as a .npz file under per_cell_results/.

Submitted as a SLURM array job by HPC_egocentric_job_submission.ipynb.
"""
import argparse
import json
import os
import sys
import pickle
import numpy as np
from dataclasses import asdict
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
    EgocentricTuningParams,
    _run_single_cell_egocentric_tuning_frame_sampled,
    _load_merged_data,
    _prepare_native_analysis_context,
    _load_spatial_analysis_by_idx,
    _passes_egocentric_category_gate,
)


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory for per_cell_results/ (direction/spike-specific)')
    parser.add_argument('--manifest-dir', type=str, default=None,
                        help='Directory containing manifest.json (shared across runs). '
                             'Defaults to --output-dir if not specified.')
    parser.add_argument('--data-root', type=str, default=str(HERE / 'data'))
    parser.add_argument('--figures-root', type=str, default=str(HERE / 'figures'))
    parser.add_argument('--cell-index', type=int, required=True,
                        help='Index into manifest (typically $SLURM_ARRAY_TASK_ID)')
    parser.add_argument('--direction-mode', type=str, default='head', choices=['head', 'travel'])
    parser.add_argument('--spike-type', type=str, default='all_spike',
                        choices=['all_spike', 'simple_spike', 'complex_spike'],
                        help='Which spike type to use for egocentric analysis')
    parser.add_argument('--n-surrogates', type=int, default=1000)
    parser.add_argument('--n-jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--first-n-minutes', type=float, default=10.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else output_dir
    results_dir = output_dir / 'per_cell_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest_path = manifest_dir / 'manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)

    if args.cell_index < 0 or args.cell_index >= len(manifest):
        print(f'[ERROR] cell_index {args.cell_index} out of range (0-{len(manifest)-1})')
        sys.exit(1)

    cell_info = manifest[args.cell_index]
    category = cell_info['category']
    animal_id = cell_info['animal_id']
    cell_idx = int(cell_info['cell_idx'])
    out_file = results_dir / f'cell_{args.cell_index:04d}.npz'

    print(f'Cell index:  {args.cell_index}')
    print(f'Category:    {category}')
    print(f'Animal:      {animal_id}')
    print(f'Cell idx:    {cell_idx}')
    print(f'Spike type:  {args.spike_type}')
    print(f'Output:      {out_file}')

    config = build_config(args.data_root, args.figures_root)

    # Build egocentric params
    params = EgocentricTuningParams(
        categories=(category,),
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
        clear_output=False,
        save_null_distributions=False,
        show_progress=False,
    )

    # Validate cell identity
    if not animal_id or cell_idx < 0:
        _save_skip(out_file, category, animal_id, cell_idx, 'invalid_cell_identity')
        return

    # Load animal data
    animal_dir = config.data_root / animal_id
    print(f'Loading data from {animal_dir} ...')
    try:
        merged = _load_merged_data(animal_dir)
        ctx = _prepare_native_analysis_context(merged, config)
        spatial_by_idx = _load_spatial_analysis_by_idx(animal_dir)
    except Exception as exc:
        _save_skip(out_file, category, animal_id, cell_idx, f'data_load_failed({exc})')
        return

    # Validate cell
    analysis = spatial_by_idx.get(cell_idx)
    if not isinstance(analysis, dict):
        _save_skip(out_file, category, animal_id, cell_idx, 'missing_spatial_analysis')
        return

    pass_gate, gate_reason = _passes_egocentric_category_gate(category=category, analysis=analysis)
    if not pass_gate:
        _save_skip(out_file, category, animal_id, cell_idx, gate_reason or 'category_gate_failed')
        return

    if cell_idx >= int(ctx.get('n_cells', 0)):
        _save_skip(out_file, category, animal_id, cell_idx, 'cell_idx_out_of_range')
        return

    eligible = np.asarray(ctx.get('eligible_cells', np.array([])), dtype=bool)
    if cell_idx >= eligible.size or not bool(eligible[cell_idx]):
        _save_skip(out_file, category, animal_id, cell_idx, 'cell_not_eligible_after_snr_filter')
        return

    # Select spike type
    SPIKE_TYPE_MAP = {
        'all_spike': 'all_spikes',
        'simple_spike': 'refined_ss',
        'complex_spike': 'all_cs_spikes',
    }
    spike_key = SPIKE_TYPE_MAP[args.spike_type]
    spike_frames = np.asarray(ctx[spike_key][cell_idx], dtype=int)
    if spike_frames.size == 0:
        _save_skip(out_file, category, animal_id, cell_idx,
                   f'no_spikes_for_type_{args.spike_type}')
        return

    bad_mask = np.asarray(ctx['bad_masks'][cell_idx], dtype=bool)
    rng = np.random.default_rng(42 + args.cell_index)

    print(f'Running egocentric analysis (spike_type={args.spike_type}, '
          f'n_spikes={spike_frames.size}, n_surrogates={args.n_surrogates}, n_jobs={args.n_jobs}) ...')
    result, null_mrls, fail_reason = _run_single_cell_egocentric_tuning_frame_sampled(
        category=category,
        animal_id=animal_id,
        cell_idx=cell_idx,
        ctx=ctx,
        spike_frames=spike_frames,
        bad_mask=bad_mask,
        params=params,
        rng=rng,
    )

    if result is None:
        _save_skip(out_file, category, animal_id, cell_idx, fail_reason or 'fit_failed')
        return

    # Save success result
    result_dict = asdict(result)
    np.savez(
        out_file,
        status='success',
        **{k: np.array(v) for k, v in result_dict.items()},
        null_mrls=np.asarray(null_mrls, dtype=float),
    )
    print(f'[OK] real_mrl={result.real_mrl:.4f}, pass_95={result.pass_95}, '
          f'pass_99={result.pass_99}, empirical_p={result.empirical_p:.4f}')
    print(f'Saved: {out_file}')


def _save_skip(out_file, category, animal_id, cell_idx, reason):
    """Save a skip/fail result."""
    np.savez(
        out_file,
        status='skipped',
        category=np.array(category),
        animal_id=np.array(animal_id),
        cell_idx=np.array(cell_idx),
        reason=np.array(reason),
    )
    print(f'[SKIP] {reason}')
    print(f'Saved: {out_file}')


if __name__ == '__main__':
    main()
