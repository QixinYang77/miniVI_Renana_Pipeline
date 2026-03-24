#!/usr/bin/env python3
"""
Run once on the cluster to set up the expected data folder structure.

Creates miniVI_PlaceCell_analysis_V4/data/ANIMAL_NAME/merged_aligned_data.pkl
as a symlink pointing to the actual file on the HPC shared network.

Usage (on the cluster):
    cd /ems/elsc-labs/adam-y/qixin.yang/ClusterCode/miniVI_Renana_Pipeline
    python miniVI_PlaceCell_analysis_V4/notebooks_HPC/setup_data_links.py
"""
from pathlib import Path

# Mapping: animal_name -> folder containing merged_aligned_data.pkl on the HPC
ANIMAL_DATA_MAP = {
    'CKII_pAce21_PR_20250806': '/ems/elsc-labs/adam-y/Adam-Lab-Shared/Data/renana_malka/pAce21/2025-08-06_pAce21_PR/Awake',
    'CKII_pAce38_PX_20251126': '/ems/elsc-labs/adam-y/Adam-Lab-Shared/Data/renana_malka/pAce38/2025-11-26_pAce38/PX/post/Awake',
    'CKII_pAce45_PX_20260118': '/ems/elsc-labs/adam-y/Adam-Lab-Shared/Data/renana_malka/pAce45/2026-01-18_pAce45/PX/post/Awake',
    'CKII_pAce46_PR_20260222': '/ems/elsc-labs/adam-y/Adam-Lab-Shared/Data/renana_malka/pAce46/2026-02-22_pAce46/PR/post/Awake',
    'CKII_pAce47_PX_20260128': '/ems/elsc-labs/adam-y/Adam-Lab-Shared/Data/renana_malka/pAce47/2026-01-28_pAce47/PX/post/Awake',
    'CKII_pAce50_PRL_20260317': '/ems/elsc-labs/adam-y/Adam-Lab-Shared/Data/renana_malka/pAce50/2026-03-17_pAce50_PRL/Awake',
}

DATA_ROOT = Path(__file__).parent.parent / 'data'

print(f'Data root: {DATA_ROOT}')
DATA_ROOT.mkdir(parents=True, exist_ok=True)

for animal_name, src_folder in ANIMAL_DATA_MAP.items():
    src_folder = Path(src_folder)
    src_pkl = src_folder / 'merged_aligned_data.pkl'
    if not src_pkl.exists():
        src_pkl = src_folder / 'merged_aligned_data_CS.pkl'
    if not src_pkl.exists():
        print(f'[WARN] No merged_aligned_data.pkl found for {animal_name} in {src_folder}')
        continue

    animal_dir = DATA_ROOT / animal_name
    animal_dir.mkdir(parents=True, exist_ok=True)

    link_path = animal_dir / src_pkl.name
    if link_path.exists() or link_path.is_symlink():
        print(f'[SKIP] Already exists: {link_path}')
    else:
        link_path.symlink_to(src_pkl)
        print(f'[OK]   {link_path}')
        print(f'    -> {src_pkl}')

print('\nDone. Run ensure_cache_for_all_animals next (handled by the job script).')
