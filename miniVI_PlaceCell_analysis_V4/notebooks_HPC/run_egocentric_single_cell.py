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
_top_repo_norm = os.path.normpath(_top_repo)
sys.path[:] = [
    p for p in sys.path
    if os.path.normpath(os.path.abspath(p or os.getcwd())) != _top_repo_norm
]
sys.path.insert(0, str(HERE))
os.environ['PYTHONPATH'] = str(HERE)

from notebooks_HPC.egocentric_refined_config import (
    ANIMALS,
    DEFAULT_DIRECTION_MODE,
    DEFAULT_FIRST_N_MINUTES,
    build_refined_config,
    build_refined_egocentric_params,
)
from utils.placecell_pipeline import (
    _run_single_cell_egocentric_tuning_frame_sampled,
    _resolve_merged_data_path,
    _load_merged_data,
    _prepare_native_analysis_context,
    _load_spatial_analysis_by_idx,
    _passes_egocentric_category_gate,
)


def build_config(data_root, figures_root):
    return build_refined_config(
        HERE,
        data_root,
        figures_root,
        force_recompute=False,
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
    parser.add_argument('--cell-index', type=int, default=None,
                        help='Index into manifest. Defaults to $SLURM_ARRAY_TASK_ID env var.')
    parser.add_argument('--direction-mode', type=str, default=DEFAULT_DIRECTION_MODE, choices=['head', 'travel'])
    parser.add_argument('--spike-type', type=str, default='all_spike',
                        choices=['all_spike', 'simple_spike', 'complex_spike'],
                        help='Which spike type to use for egocentric analysis')
    parser.add_argument('--n-surrogates', type=int, default=1000)
    parser.add_argument('--n-jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--first-n-minutes', type=float, default=DEFAULT_FIRST_N_MINUTES)
    args = parser.parse_args()

    # Resolve cell index: CLI arg > $SLURM_ARRAY_TASK_ID
    if args.cell_index is None:
        slurm_task = os.environ.get('SLURM_ARRAY_TASK_ID')
        if slurm_task is None:
            print('[ERROR] --cell-index not provided and $SLURM_ARRAY_TASK_ID not set.')
            sys.exit(1)
        args.cell_index = int(slurm_task)

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
    params = build_refined_egocentric_params(
        categories=(category,),
        first_n_minutes=args.first_n_minutes,
        direction_mode=args.direction_mode,
        n_surrogates=args.n_surrogates,
        n_jobs=args.n_jobs,
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
    if cell_info.get('runtime_merged_data_file'):
        print(f"Manifest runtime merged data: {cell_info['runtime_merged_data_file']}")
    if cell_info.get('cache_merged_data_file'):
        print(f"Manifest cache merged data:   {cell_info['cache_merged_data_file']}")
    try:
        resolved_data_path = _resolve_merged_data_path(animal_dir, config)
        print(f'Resolved runtime merged data: {resolved_data_path}')
        print(f'Spatial analysis file:        {animal_dir / "spatial_analysis_full.pkl"}')
        merged = _load_merged_data(animal_dir, config)
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
