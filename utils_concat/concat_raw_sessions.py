import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np


def _read_experiment_xml(xml_path: str) -> Tuple[float, int, int]:
    """
    Read ScanImage-style Experiment.xml metadata.

    Returns:
        frame_rate_hz, width, height
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    width = int(root[5].attrib["width"])
    height = int(root[5].attrib["height"])
    lsm = root.find("LSM")
    if lsm is None or "frameRate" not in lsm.attrib:
        raise ValueError("Could not find LSM frameRate in: {}".format(xml_path))
    frame_rate_hz = float(lsm.attrib["frameRate"])
    return frame_rate_hz, width, height


def _raw_frame_count(raw_path: str, *, width: int, height: int) -> int:
    frame_bytes = int(width) * int(height) * 2  # uint16
    size = os.path.getsize(raw_path)
    if size % frame_bytes != 0:
        raise ValueError(
            "Raw size not divisible by frame bytes ({}): {} (size={})".format(frame_bytes, raw_path, size)
        )
    return int(size // frame_bytes)


def _copy_metadata(
    src_folder: str,
    out_dir: str,
    *,
    experiment_xml_name: str = "Experiment.xml",
    rois_xaml_name: str = "ROIs.xaml",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    xml_src = os.path.join(src_folder, experiment_xml_name)
    if not os.path.exists(xml_src):
        raise FileNotFoundError("Missing {} next to raw: {}".format(experiment_xml_name, xml_src))
    shutil.copy(xml_src, os.path.join(out_dir, experiment_xml_name))

    xaml_src = os.path.join(src_folder, rois_xaml_name)
    if os.path.exists(xaml_src):
        shutil.copy(xaml_src, os.path.join(out_dir, rois_xaml_name))
    else:
        print("[WARN] Missing {} next to raw: {} (DS_motion_correction will fail without ROIs.xaml)".format(rois_xaml_name, xaml_src), flush=True)


def concat_raw_sessions(
    raw_paths: List[str],
    out_dir: str,
    *,
    out_raw_name: str,
    frames_to_truncate_per_session: int,
    overwrite: bool,
    copy_metadata_from: str,
    experiment_xml_name: str,
    rois_xaml_name: str,
    copy_chunk_mb: int = 64,
) -> Dict:
    if not raw_paths:
        raise ValueError("Provide at least one --raw path.")

    raw_paths = [os.path.abspath(p) for p in raw_paths]
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Resolve metadata source
    if copy_metadata_from == "first":
        meta_src_raw = raw_paths[0]
    else:
        meta_src_raw = os.path.abspath(copy_metadata_from)
        if os.path.isdir(meta_src_raw):
            # allow passing a folder that contains the Experiment.xml
            meta_src_raw = os.path.join(meta_src_raw, os.path.basename(raw_paths[0]))

    meta_src_folder = os.path.dirname(meta_src_raw)
    xml0 = os.path.join(meta_src_folder, experiment_xml_name)
    fr0, w0, h0 = _read_experiment_xml(xml0)

    # Validate all sessions have consistent dimensions
    session_info = []
    for p in raw_paths:
        sess_dir = os.path.dirname(p)
        xml = os.path.join(sess_dir, experiment_xml_name)
        fr, w, h = _read_experiment_xml(xml)
        if w != w0 or h != h0:
            raise ValueError("Shape mismatch: {} is {}x{}, expected {}x{}".format(p, w, h, w0, h0))
        if abs(float(fr) - float(fr0)) > 1e-6:
            print("[WARN] frameRate differs: {} has {}, first has {}".format(p, fr, fr0), flush=True)

        n_frames = _raw_frame_count(p, width=w0, height=h0)
        if frames_to_truncate_per_session and int(frames_to_truncate_per_session) >= int(n_frames):
            raise ValueError(
                "frames_to_truncate_per_session={} >= n_frames={} for {}".format(
                    frames_to_truncate_per_session, n_frames, p
                )
            )
        n_keep = int(n_frames) - int(max(0, int(frames_to_truncate_per_session)))
        session_info.append(
            {
                "raw_path": p,
                "experiment_xml": xml,
                "n_frames_total": int(n_frames),
                "n_frames_kept": int(n_keep),
            }
        )

    out_raw_path = os.path.join(out_dir, out_raw_name)
    edges = [0]

    if os.path.exists(out_raw_path):
        if overwrite:
            os.remove(out_raw_path)
        else:
            raise FileExistsError("Output raw already exists: {} (pass --overwrite)".format(out_raw_path))

    frame_bytes = int(w0) * int(h0) * 2
    skip_bytes = int(max(0, int(frames_to_truncate_per_session))) * frame_bytes
    chunk_bytes = int(max(1, int(copy_chunk_mb))) * 1024 * 1024

    written_frames = 0
    with open(out_raw_path, "wb") as fout:
        for info in session_info:
            p = info["raw_path"]
            print("[INFO] Append {} (keep {} frames)".format(p, info["n_frames_kept"]), flush=True)
            with open(p, "rb") as fin:
                if skip_bytes:
                    fin.seek(skip_bytes, os.SEEK_SET)
                while True:
                    buf = fin.read(chunk_bytes)
                    if not buf:
                        break
                    fout.write(buf)
            written_frames += int(info["n_frames_kept"])
            edges.append(int(written_frames))

    _copy_metadata(
        meta_src_folder,
        out_dir,
        experiment_xml_name=experiment_xml_name,
        rois_xaml_name=rois_xaml_name,
    )

    edges_path = os.path.join(out_dir, "session_edges_frames.npy")
    np.save(edges_path, np.asarray(edges, dtype=np.int64))

    manifest = {
        "out_dir": out_dir,
        "out_raw_path": out_raw_path,
        "out_raw_name": out_raw_name,
        "frame_rate_hz": float(fr0),
        "width": int(w0),
        "height": int(h0),
        "frames_to_truncate_per_session": int(frames_to_truncate_per_session),
        "session_edges_frames": edges,
        "sessions": session_info,
    }
    manifest_path = os.path.join(out_dir, "concat_raw_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print("[INFO] Wrote concatenated raw: {}".format(out_raw_path), flush=True)
    print("[INFO] Session edges (frames): {}".format(edges), flush=True)
    print("[INFO] Manifest: {}".format(manifest_path), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate multiple session .raw files into a single Image_001_001.raw dataset folder (for motion correction on the concatenated movie)."
    )
    parser.add_argument("--out-dir", required=True, help="Destination folder (e.g., .../Awake/output_concat).")
    parser.add_argument(
        "--raw",
        action="append",
        default=[],
        help="Raw .raw file path for each session. Repeat for each session.",
    )
    parser.add_argument(
        "--out-raw-name",
        default="Image_001_001.raw",
        help="Output raw filename (default: Image_001_001.raw). Keep this to match downstream ROI naming.",
    )
    parser.add_argument(
        "--frames-to-truncate-per-session",
        type=int,
        default=500,
        help="Frames to drop from the start of EACH session before concatenation (default: 500).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output raw + manifest in out-dir.",
    )
    parser.add_argument(
        "--copy-metadata-from",
        default="first",
        help="Where to copy Experiment.xml / ROIs.xaml from: 'first' or a path to a session folder/raw (default: first).",
    )
    parser.add_argument("--experiment-xml-name", default="Experiment.xml")
    parser.add_argument("--rois-xaml-name", default="ROIs.xaml")
    args = parser.parse_args()

    concat_raw_sessions(
        raw_paths=list(args.raw),
        out_dir=str(args.out_dir),
        out_raw_name=str(args.out_raw_name),
        frames_to_truncate_per_session=int(args.frames_to_truncate_per_session),
        overwrite=bool(args.overwrite),
        copy_metadata_from=str(args.copy_metadata_from),
        experiment_xml_name=str(args.experiment_xml_name),
        rois_xaml_name=str(args.rois_xaml_name),
    )


if __name__ == "__main__":
    main()

