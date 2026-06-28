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
    DEFAULT_CATEGORIES,
    DEFAULT_DIRECTION_MODE,
    DEFAULT_FIRST_N_MINUTES,
    build_refined_config,
    build_refined_egocentric_params,
)
from utils.placecell_pipeline import (
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
    parser.add_argument('--direction-mode', type=str, default=DEFAULT_DIRECTION_MODE, choices=['head', 'travel'],
                        help='Egocentric direction mode: head or travel')
    parser.add_argument('--n-surrogates', type=int, default=1000,
                        help='Number of surrogates for significance testing')
    parser.add_argument('--first-n-minutes', type=float, default=DEFAULT_FIRST_N_MINUTES,
                        help='Only use first N minutes of each session')
    parser.add_argument('--categories', nargs='+', default=list(DEFAULT_CATEGORIES),
                        help='Categories to include in the egocentric analysis')
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

    config = build_refined_config(
        HERE,
        data_root,
        figures_root,
        force_recompute=args.force_recompute,
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
        cs_plc_definition_mode=config.pooled.cs_plc_definition_mode,
        snr_threshold=config.analysis.snr_threshold,
    )
    print(f'  CS+ PLCs:  {len(spatial_data.plcs_csplus)}')
    print(f'  CS- PLCs:  {len(spatial_data.plcs_csminus)}')
    print(f'  Non-PLCs:  {len(spatial_data.non_plcs)}')

    # Step 3: Run egocentric tuning analysis
    figure_save_folder = str(figures_root / 'CKII_pooled')
    os.makedirs(figure_save_folder, exist_ok=True)

    egocentric_params = build_refined_egocentric_params(
        categories=tuple(args.categories),
        first_n_minutes=args.first_n_minutes,
        direction_mode=args.direction_mode,
        n_surrogates=args.n_surrogates,
        n_jobs=args.n_jobs,
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
