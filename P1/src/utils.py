import os
import glob
from typing import Dict, List, Tuple

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_color(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_gray_pair(grayL, grayR, scale):
    if scale == 1.0:
        return grayL, grayR
    h, w = grayL.shape[:2]
    size = (int(round(w * scale)), int(round(h * scale)))
    return (cv2.resize(grayL, size, interpolation=cv2.INTER_AREA),
            cv2.resize(grayR, size, interpolation=cv2.INTER_AREA))


def _is_left(name):
    s = name.lower()
    return "left" in s or s.endswith("_l") or s.endswith("-l")


def _is_right(name):
    s = name.lower()
    return "right" in s or s.endswith("_r") or s.endswith("-r")


def _strip_lr(stem):
    s = stem
    for t in ["_left", "-left", ".left", "left", "_l", "-l", ".l"]:
        s = s.replace(t, "")
    for t in ["_right", "-right", ".right", "right", "_r", "-r", ".r"]:
        s = s.replace(t, "")
    return s.strip("_-.")


def discover_pairs(pairs_dir: str) -> List[Dict[str, str]]:
    """Find stereo pairs named like pair01_left.jpg / pair01_right.jpg."""
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(pairs_dir, f"*{ext}")))

    buckets: Dict[str, Dict[str, str]] = {}
    for p in sorted(files):
        stem, _ = os.path.splitext(os.path.basename(p))
        isL, isR = _is_left(stem), _is_right(stem)
        if not (isL or isR):
            continue
        key = _strip_lr(stem)
        if key not in buckets:
            buckets[key] = {}
        if isL:
            buckets[key]["left"] = p
        if isR:
            buckets[key]["right"] = p

    return [{"name": k, "left": d["left"], "right": d["right"]}
            for k, d in buckets.items() if "left" in d and "right" in d]
