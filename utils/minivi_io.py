import glob
import os
import xml.etree.ElementTree as ET
from typing import Tuple

import numpy as np
from matplotlib.path import Path
from skimage.transform import downscale_local_mean


def load_experiment_metadata(raw_file: str) -> Tuple[float, int, int]:
    """
    Load frame rate and image shape from `Experiment.xml` next to a ScanImage-style `.raw`.

    Args:
        raw_file: Path or glob pattern to `.raw` file.

    Returns:
        frame_rate_hz: float
        width: int
        height: int
    """
    raw_matches = glob.glob(raw_file)
    if not raw_matches:
        raise FileNotFoundError(f"Raw file not found: {raw_file}")
    raw_path = raw_matches[0]

    data_folder = os.path.dirname(raw_path)
    xml_path = os.path.join(data_folder, "Experiment.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Experiment.xml not found next to raw file: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    width = int(root[5].attrib["width"])
    height = int(root[5].attrib["height"])
    lsm = root.find("LSM")
    if lsm is None or "frameRate" not in lsm.attrib:
        raise ValueError(f"Could not find LSM frameRate in: {xml_path}")
    frame_rate_hz = float(lsm.attrib["frameRate"])

    return frame_rate_hz, width, height


def load_raw(raw_file: str, *, frames_to_truncate: int = 500) -> Tuple[np.ndarray, float, int, int]:
    """
    Load a ScanImage-style `.raw` movie and metadata from `Experiment.xml` in the same folder.

    Returns:
        movie_3d: (T, H, W) uint16
        frame_rate_hz: float
        width: int
        height: int
    """
    raw_matches = glob.glob(raw_file)
    if not raw_matches:
        raise FileNotFoundError(f"Raw file not found: {raw_file}")
    raw_path = raw_matches[0]

    data_folder = os.path.dirname(raw_path)
    xml_path = os.path.join(data_folder, "Experiment.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Experiment.xml not found next to raw file: {xml_path}")

    rawmovie_1d = np.fromfile(raw_path, dtype=np.uint16)

    frame_rate_hz, width, height = load_experiment_metadata(raw_path)

    movie_3d = np.reshape(rawmovie_1d, (-1, height, width))
    if frames_to_truncate and frames_to_truncate > 0:
        movie_3d = movie_3d[frames_to_truncate:, :, :]

    return movie_3d, frame_rate_hz, width, height


def load_ROI(data_folder: str, width: int, height: int, *, xaml_name: str = "ROIs.xaml") -> np.ndarray:
    """
    Load ROI polygons from `ROIs.xaml` and rasterize into boolean masks.

    Returns:
        ROIs: (N, H, W) bool
    """
    import xmltodict

    path = os.path.join(data_folder, xaml_name)
    xml_data = open(path, "r").read()
    xml_dict = xmltodict.parse(xml_data)
    polygons_struct = xml_dict["ROICapsule"]["ROICapsule.ROIs"]["x:Array"]["ROIPoly"]

    poly_lst = []
    for i in range(len(polygons_struct)):
        temp = polygons_struct[i]["@Points"]
        if temp not in poly_lst:
            poly_lst.append(temp)

    rect_data = xml_dict["ROICapsule"]["ROICapsule.ROIs"]["x:Array"]["ROIRect"]
    bottom_left_x, bottom_left_y = [float(i) for i in rect_data["@BottomLeft"].split(",")]
    top_left_x, top_left_y = [float(i) for i in rect_data["@TopLeft"].split(",")]
    height_rec = float(rect_data["@ROIHeight"])
    width_rec = float(rect_data["@ROIWidth"])

    corrected_polygons = []
    for polygon in poly_lst:
        corrected_points = []
        points = polygon.split(" ")
        for point in points:
            x, y = [float(i) for i in point.split(",")]
            x = min(max(x - bottom_left_x, 1), width_rec)
            y = max(1, min(y - top_left_y, height_rec))
            corrected_points.append((x, y))
        corrected_points.append(corrected_points[0])
        corrected_polygons.append(corrected_points)

    ROIs = []
    for poly in corrected_polygons:
        flipped_poly = [(j, i) for i, j in poly]
        poly_path = Path(flipped_poly)
        x, y = np.mgrid[:height, :width]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        mask = poly_path.contains_points(coors)
        ROIs.append(mask.reshape(height, width))
    return np.stack(ROIs).astype(bool)


def downsample_video_local_mean(video_data: np.ndarray, factors: Tuple[int, int]) -> np.ndarray:
    """
    Spatially downsample a 3D array (T, H, W) using block-mean.
    Works for movies (T=frames) or ROI stacks (T=number of ROIs).
    """
    if not isinstance(video_data, np.ndarray):
        raise TypeError("Input 'video_data' must be a NumPy array.")
    if video_data.ndim != 3:
        raise ValueError(
            f"Input 'video_data' must be 3-dimensional (T, H, W), but got shape {video_data.shape}"
        )
    if not isinstance(factors, tuple) or len(factors) != 2:
        raise TypeError("Input 'factors' must be a tuple of length 2 (factor_h, factor_w).")
    if not all(isinstance(f, int) and f > 0 for f in factors):
        raise ValueError("Factors must be positive integers.")

    factor_h, factor_w = factors
    full_factors = (1, factor_h, factor_w)
    downsampled = downscale_local_mean(video_data, full_factors)
    return downsampled.astype(video_data.dtype, copy=False)
