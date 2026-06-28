#!/usr/bin/env python3
"""Unpack the refined all-animal CKII bundle into per-animal cluster inputs."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent.parent.resolve()
_top_repo = str(HERE.parent)
_top_repo_norm = os.path.normpath(_top_repo)
sys.path[:] = [
    p for p in sys.path
    if os.path.normpath(os.path.abspath(p or os.getcwd())) != _top_repo_norm
]
sys.path.insert(0, str(HERE))
os.environ["PYTHONPATH"] = str(HERE)

from notebooks_HPC.egocentric_refined_config import (
    CLUSTER_REFINED_INPUT_FILENAME,
    CLUSTER_REFINED_METADATA_FILENAME,
    REFINED_CLUSTER_INPUT_SCHEMA_VERSION,
)


def _load_bundle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict):
        raise ValueError(f"Bundle is not a dict: {path}")
    schema_version = int(bundle.get("schema_version", -1))
    if schema_version != REFINED_CLUSTER_INPUT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {schema_version}; "
            f"expected {REFINED_CLUSTER_INPUT_SCHEMA_VERSION}"
        )
    animals = bundle.get("animals")
    if not isinstance(animals, dict) or not animals:
        raise ValueError(f"Bundle has no animals payload: {path}")
    return bundle


def _metadata_for_animal(bundle: dict[str, Any], bundle_path: Path, animal_id: str) -> dict[str, Any]:
    return {
        "schema_version": int(bundle.get("schema_version", -1)),
        "bundle_path": str(bundle_path),
        "generated_at": bundle.get("generated_at"),
        "generator": bundle.get("generator", {}),
        "parameter_snapshot": bundle.get("parameter_snapshot", {}),
        "source_files": (bundle.get("source_files", {}) or {}).get(animal_id, {}),
        "validation": (bundle.get("validation", {}) or {}).get(animal_id, {}),
    }


def unpack_bundle(
    *,
    bundle_path: Path,
    data_root: Path,
    animals: list[str] | None,
    force: bool,
    validate_only: bool,
) -> list[Path]:
    bundle = _load_bundle(bundle_path)
    animal_payloads = bundle["animals"]
    selected = list(animals) if animals else list(animal_payloads)
    missing = [animal_id for animal_id in selected if animal_id not in animal_payloads]
    if missing:
        raise KeyError(f"Requested animals missing from bundle: {missing}")

    written: list[Path] = []
    for animal_id in selected:
        animal_dir = data_root / animal_id
        out_pickle = animal_dir / CLUSTER_REFINED_INPUT_FILENAME
        out_meta = animal_dir / CLUSTER_REFINED_METADATA_FILENAME
        if not validate_only and not force:
            existing = [p for p in (out_pickle, out_meta) if p.exists()]
            if existing:
                raise FileExistsError(
                    "Refusing to overwrite existing unpacked files without --force: "
                    + ", ".join(str(p) for p in existing)
                )
        print(f"{animal_id}: {out_pickle}")
        if validate_only:
            continue
        animal_dir.mkdir(parents=True, exist_ok=True)
        with out_pickle.open("wb") as f:
            pickle.dump(animal_payloads[animal_id], f, protocol=pickle.HIGHEST_PROTOCOL)
        with out_meta.open("w") as f:
            json.dump(_metadata_for_animal(bundle, bundle_path, animal_id), f, indent=2, sort_keys=True)
        written.append(out_pickle)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Uploaded all-animal refined cluster bundle")
    parser.add_argument("--data-root", type=Path, default=HERE / "data")
    parser.add_argument("--animals", nargs="+", default=None, help="Optional subset to unpack")
    parser.add_argument("--force", action="store_true", help="Overwrite existing per-animal outputs")
    parser.add_argument("--validate-only", action="store_true", help="Validate bundle and planned paths only")
    args = parser.parse_args()

    written = unpack_bundle(
        bundle_path=args.bundle,
        data_root=args.data_root,
        animals=None if args.animals is None else [str(a) for a in args.animals],
        force=bool(args.force),
        validate_only=bool(args.validate_only),
    )
    if args.validate_only:
        print("[VALIDATE ONLY] Bundle is readable and output paths are resolvable.")
    else:
        print(f"Unpacked {len(written)} animals into {args.data_root}")


if __name__ == "__main__":
    main()
