#!/usr/bin/env python3
"""
Step 1 of 3: Prepare for parallel egocentric analysis.

- Uses compact per-animal cluster_refined_spatial_classification.pkl files
  unpacked from the local refined bundle.
- Builds a cell manifest listing every (category, animal_id, cell_idx)
  that will be analyzed.
- Saves the manifest as JSON so array tasks can pick their cell by index.

Submitted by HPC_egocentric_job_submission.ipynb via sbatch.
"""
import argparse
import json
import os
import pickle
import sys
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
    CLUSTER_REFINED_METADATA_FILENAME,
    CLUSTER_SPATIAL_CLASSIFICATION_FILENAME,
    DEFAULT_CATEGORIES,
    build_refined_config,
)
from utils.placecell_pipeline import (
    ensure_cache_for_all_animals,
    _resolve_merged_data_path,
    _get_spatial_category_cells,
    _normalize_pf_category_name,
)
from utils.spatial_heatmaps import (
    SpatialCategoryData,
    cell_has_cs_place_field,
    classify_spatial_cells,
    is_csplus_place_cell,
    normalize_cs_plc_definition_mode,
)


def _path_record(path: Path) -> dict:
    exists = path.exists()
    rec = {
        'path': str(path),
        'exists': bool(exists),
        'is_symlink': bool(path.is_symlink()),
    }
    if path.is_symlink():
        try:
            rec['link_target'] = os.readlink(path)
        except OSError as exc:
            rec['link_target_error'] = str(exc)
    if exists:
        stat = path.stat()
        rec.update({
            'resolved_path': str(path.resolve()),
            'size_bytes': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns),
        })
    return rec


def collect_data_source_records(config):
    records = []
    for animal_id in config.animals:
        animal_dir = config.data_root / animal_id
        rec = {
            'animal_id': animal_id,
            'animal_dir': str(animal_dir),
            'cache_merged_data': None,
            'runtime_merged_data': None,
            'manual_spike_sidecar': _path_record(animal_dir / 'manual_spike_detection_results.pkl'),
            'cluster_refined_metadata': _path_record(animal_dir / CLUSTER_REFINED_METADATA_FILENAME),
            'cluster_spatial_classification': _path_record(animal_dir / CLUSTER_SPATIAL_CLASSIFICATION_FILENAME),
            'spatial_analysis_full': _path_record(animal_dir / 'spatial_analysis_full.pkl'),
            'animal_cache_bundle': _path_record(animal_dir / 'animal_cache_bundle_v1.pkl'),
        }
        try:
            resolved = _resolve_merged_data_path(animal_dir, config)
            rec['cache_merged_data'] = _path_record(resolved)
            rec['runtime_merged_data'] = _path_record(resolved)
        except Exception as exc:
            rec['cache_merged_data_error'] = str(exc)
            rec['runtime_merged_data_error'] = str(exc)
        records.append(rec)
    return records


def validate_exported_spatial_classification_files(config):
    missing = []
    for animal_id in config.animals:
        spatial_path = config.data_root / animal_id / CLUSTER_SPATIAL_CLASSIFICATION_FILENAME
        if not spatial_path.exists():
            missing.append(spatial_path)
    if missing:
        raise FileNotFoundError(
            'Missing exported cluster spatial classification files. Run Step 0 unpack first, '
            'or rerun this script with --force-recompute when full trace data are available: '
            + ', '.join(str(p) for p in missing)
        )


def load_exported_spatial_category_data(config):
    """Load compact local-derived spatial classification files from Step 0 unpack."""
    valid_spatial_cells = []
    cb_in_pf_counts = {}
    data_folder_abs = os.path.abspath(str(config.data_root))
    mode = normalize_cs_plc_definition_mode(config.pooled.cs_plc_definition_mode)

    for animal_id in config.animals:
        spatial_path = config.data_root / animal_id / CLUSTER_SPATIAL_CLASSIFICATION_FILENAME
        with spatial_path.open('rb') as f:
            cells = pickle.load(f)
        if not isinstance(cells, list):
            raise ValueError(f'{spatial_path} is not a list')

        loaded = 0
        for raw_cell in cells:
            if not isinstance(raw_cell, dict):
                continue
            cell = dict(raw_cell)
            cell_idx = int(cell.get('cell_idx', -1))
            if cell_idx < 0:
                continue
            cell['session'] = animal_id
            cell['data_folder'] = data_folder_abs
            n_cb = int(cell.get('n_cb_in_pf', 0) or 0)
            has_cs_pf = cell_has_cs_place_field(cell)
            cell['n_cb_in_pf'] = n_cb
            cell['has_cs_place_field'] = has_cs_pf
            cell['cs_plc_definition_mode'] = mode
            cell['is_cs_plc'] = is_csplus_place_cell(
                is_place_cell=bool(cell.get('is_place_cell', False)),
                n_cb_in_pf=n_cb,
                cs_peak_rate=float(cell.get('cs_peak_rate', float('nan'))),
                cb_num_threshold=int(config.pooled.cb_num_threshold),
                cs_peak_rate_threshold=float(config.pooled.cs_peak_rate_threshold),
                has_cs_place_field=has_cs_pf,
                cs_plc_definition_mode=mode,
            )
            valid_spatial_cells.append(cell)
            cb_in_pf_counts[(animal_id, cell_idx)] = n_cb
            loaded += 1
        print(f'Loaded {loaded} classified cells from {animal_id}')

    plcs_csplus, plcs_csminus, non_plcs = [], [], []
    for cell in valid_spatial_cells:
        if bool(cell.get('is_place_cell', False)):
            if bool(cell.get('is_cs_plc', False)):
                plcs_csplus.append(cell)
            else:
                plcs_csminus.append(cell)
        else:
            non_plcs.append(cell)

    print(f'\nTotal: {len(valid_spatial_cells)} classified cells loaded')
    return SpatialCategoryData(
        valid_spatial_cells=valid_spatial_cells,
        plcs_csplus=plcs_csplus,
        plcs_csminus=plcs_csminus,
        non_plcs=non_plcs,
        deleted_cells=set(),
        cb_in_pf_counts=cb_in_pf_counts,
    )


def build_config(data_root, figures_root, force_recompute=False):
    return build_refined_config(
        HERE,
        data_root,
        figures_root,
        force_recompute=force_recompute,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=str(HERE / 'data'))
    parser.add_argument('--figures-root', type=str, default=str(HERE / 'figures'))
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to write manifest.json (shared across direction/spike runs)')
    parser.add_argument('--categories', nargs='+', default=list(DEFAULT_CATEGORIES))
    parser.add_argument('--force-recompute', action='store_true')
    args = parser.parse_args()

    config = build_config(args.data_root, args.figures_root, args.force_recompute)

    # Step 1: Ensure spatial classification inputs.
    if args.force_recompute:
        print('--- Ensuring cache for all animals ---')
        statuses = ensure_cache_for_all_animals(config, force=True)
        for st in statuses:
            print(f'  [{st.action}] {st.animal_id}')
    else:
        print('--- Reusing exported cluster spatial classification files; cache recomputation skipped ---')
        validate_exported_spatial_classification_files(config)
        for animal_id in config.animals:
            print(f'  [reused_exported_spatial_classification] {animal_id}')

    data_sources = collect_data_source_records(config)
    source_by_animal = {rec['animal_id']: rec for rec in data_sources}

    # Step 2: Load spatial data
    print('\n--- Loading spatial classification data ---')
    if args.force_recompute:
        spatial_data = classify_spatial_cells(
            data_folder=str(config.data_root),
            folders=config.animals,
            cb_num_threshold=config.pooled.cb_num_threshold,
            cs_peak_rate_threshold=config.pooled.cs_peak_rate_threshold,
            cs_plc_definition_mode=config.pooled.cs_plc_definition_mode,
            snr_threshold=config.analysis.snr_threshold,
        )
    else:
        spatial_data = load_exported_spatial_category_data(config)
    print(f'  CS+ PLCs:  {len(spatial_data.plcs_csplus)}')
    print(f'  CS- PLCs:  {len(spatial_data.plcs_csminus)}')
    print(f'  Non-PLCs:  {len(spatial_data.non_plcs)}')

    # Step 3: Build cell manifest
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = [_normalize_pf_category_name(c) for c in args.categories]
    manifest = []
    for category in categories:
        cells = _get_spatial_category_cells(spatial_data, category)
        for cell_meta in cells:
            animal_id = str(cell_meta.get('session', ''))
            source_meta = source_by_animal.get(animal_id, {})
            manifest.append({
                'category': category,
                'animal_id': animal_id,
                'cell_idx': int(cell_meta.get('cell_idx', -1)),
                'cache_merged_data_file': (
                    source_meta.get('cache_merged_data') or {}
                ).get('path', ''),
                'runtime_merged_data_file': (
                    source_meta.get('runtime_merged_data') or {}
                ).get('path', ''),
                'spatial_classification_file': (
                    source_meta.get('cluster_spatial_classification') or {}
                ).get('path', ''),
                'spatial_analysis_file': (
                    source_meta.get('cluster_spatial_classification')
                    or source_meta.get('spatial_analysis_full')
                    or {}
                ).get('path', ''),
            })

    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    data_sources_path = output_dir / 'data_sources.json'
    with open(data_sources_path, 'w') as f:
        json.dump(data_sources, f, indent=2)

    print(f'\n--- Manifest ---')
    print(f'Total cells: {len(manifest)}')
    for cat in categories:
        n = sum(1 for m in manifest if m['category'] == cat)
        print(f'  {cat}: {n}')
    print(f'Saved: {manifest_path}')
    print(f'Data sources: {data_sources_path}')
    print(f'Array range: 0-{len(manifest) - 1}')


if __name__ == '__main__':
    main()
