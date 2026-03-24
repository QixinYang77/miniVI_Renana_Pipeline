#!/usr/bin/env python3
"""
Step 3 of 3: Aggregate per-cell egocentric results into summary CSVs.

Reads all cell_XXXX.npz files from per_cell_results/, combines them into:
  - egocentric_tuning_summary.csv  (successful fits)
  - egocentric_tuning_skipped.csv  (skipped/failed cells)
  - null_distributions/             (per-cell null MRL .npz files)

Submitted by HPC_egocentric_job_submission.ipynb after the array job completes.
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory containing manifest.json and per_cell_results/')
    parser.add_argument('--save-null-distributions', action='store_true',
                        help='Copy null MRL arrays into null_distributions/ folder')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_dir = output_dir / 'per_cell_results'
    manifest_path = output_dir / 'manifest.json'

    with open(manifest_path) as f:
        manifest = json.load(f)

    n_total = len(manifest)
    print(f'Total cells in manifest: {n_total}')

    # Collect results
    summary_rows = []
    skip_rows = []
    missing = []
    null_dir = output_dir / 'null_distributions'
    if args.save_null_distributions:
        null_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_total):
        npz_path = results_dir / f'cell_{i:04d}.npz'
        if not npz_path.exists():
            missing.append(i)
            cell_info = manifest[i]
            skip_rows.append({
                'category': cell_info['category'],
                'animal_id': cell_info['animal_id'],
                'cell_idx': int(cell_info['cell_idx']),
                'reason': 'result_file_missing',
            })
            continue

        with np.load(npz_path, allow_pickle=True) as data:
            status = str(data['status'])
            if status == 'success':
                row = {}
                for key in data.files:
                    if key in ('status', 'null_mrls'):
                        continue
                    val = data[key]
                    # Convert 0-d arrays to scalars
                    if val.ndim == 0:
                        val = val.item()
                    row[key] = val
                summary_rows.append(row)

                # Save null distribution
                if args.save_null_distributions and 'null_mrls' in data:
                    category = str(row.get('category', ''))
                    animal_id = str(row.get('animal_id', ''))
                    cell_idx = int(row.get('cell_idx', -1))
                    cat_dir = null_dir / category
                    cat_dir.mkdir(parents=True, exist_ok=True)
                    null_path = cat_dir / f'{animal_id}_cell{cell_idx + 1:03d}_null_mrl.npz'
                    np.savez(
                        null_path,
                        category=category,
                        animal_id=animal_id,
                        cell_idx=cell_idx,
                        cell_num=cell_idx + 1,
                        null_mrl=np.asarray(data['null_mrls'], dtype=float),
                        real_mrl=float(row.get('real_mrl', 0.0)),
                    )
            else:
                skip_rows.append({
                    'category': str(data.get('category', '')),
                    'animal_id': str(data.get('animal_id', '')),
                    'cell_idx': int(data.get('cell_idx', -1)),
                    'reason': str(data.get('reason', 'unknown')),
                })

    # Save CSVs
    summary_df = pd.DataFrame(summary_rows)
    skip_df = pd.DataFrame(skip_rows)

    summary_csv = output_dir / 'egocentric_tuning_summary.csv'
    skip_csv = output_dir / 'egocentric_tuning_skipped.csv'
    summary_df.to_csv(summary_csv, index=False)
    skip_df.to_csv(skip_csv, index=False)

    # Print summary
    print(f'\nResults:')
    print(f'  Successful fits: {len(summary_rows)}')
    print(f'  Skipped/failed:  {len(skip_rows)}')
    if missing:
        print(f'  Missing files:   {len(missing)} (indices: {missing})')

    # Per-category breakdown
    if summary_rows:
        cats = set(r.get('category', '') for r in summary_rows)
        for cat in sorted(cats):
            n = sum(1 for r in summary_rows if r.get('category', '') == cat)
            print(f'  {cat} successful: {n}')

    print(f'\nSaved:')
    print(f'  {summary_csv}')
    print(f'  {skip_csv}')
    if args.save_null_distributions:
        print(f'  {null_dir}/')

    print('\nDone.')


if __name__ == '__main__':
    main()
