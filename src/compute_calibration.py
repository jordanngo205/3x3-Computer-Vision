from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from court_geometry import COURT_LENGTH_FT, COURT_WIDTH_FT, landmark_points_ft


def parse_landmark_pairs(landmarks_px: Dict[str, List[float]]) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    known_landmarks = landmark_points_ft()
    source_points = []
    target_points = []
    pairs_used = []

    for name, pixel in landmarks_px.items():
        if pixel is None:
            continue
        if name not in known_landmarks:
            raise ValueError(f"Unknown landmark: {name}")
        if not isinstance(pixel, list) or len(pixel) != 2:
            raise ValueError(f"Landmark {name} must be a list like [x, y].")

        source_points.append([float(pixel[0]), float(pixel[1])])
        target = known_landmarks[name]
        target_points.append([float(target[0]), float(target[1])])
        pairs_used.append(
            {
                "name": name,
                "pixel": [float(pixel[0]), float(pixel[1])],
                "court_ft": [float(target[0]), float(target[1])],
            }
        )

    if len(source_points) < 4:
        raise ValueError("At least 4 landmark pairs are required.")

    return (
        np.array(source_points, dtype=np.float32),
        np.array(target_points, dtype=np.float32),
        pairs_used,
    )


def compute_reprojection_error(matrix: np.ndarray, source_points: np.ndarray, target_points: np.ndarray) -> float:
    projected = cv2.perspectiveTransform(source_points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - target_points, axis=1)
    return float(errors.mean())
