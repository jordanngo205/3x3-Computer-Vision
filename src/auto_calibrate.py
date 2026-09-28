from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np


def order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("Expected exactly four points.")

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def lane_landmark_names(side: str) -> List[str]:
    prefix = "left" if side == "left" else "right"
    return [
        f"{prefix}_lane_top_left",
        f"{prefix}_lane_top_right",
        f"{prefix}_lane_bottom_right",
        f"{prefix}_lane_bottom_left",
    ]


def build_dark_mask(image: np.ndarray, vmax: int) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, vmax))
    h, _ = mask.shape
    mask[: int(h * 0.18), :] = 0
    mask[int(h * 0.84) :, :] = 0
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def detect_backboard_candidates(image: np.ndarray) -> List[Dict[str, Any]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 150), (180, 65, 255))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]
    candidates: List[Dict[str, Any]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 60 or area > 9000:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if y > int(h * 0.55):
            continue
        if not (12 <= bw <= 120 and 10 <= bh <= 120):
            continue
        aspect = bw / max(1.0, bh)
        if not (0.35 <= aspect <= 3.5):
            continue

        region = mask[y : y + bh, x : x + bw]
        fill_ratio = cv2.countNonZero(region) / max(1.0, bw * bh)
        if fill_ratio < 0.18:
            continue

        side = "left" if (x + bw / 2.0) < (w / 2.0) else "right"
        candidates.append(
            {
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "center": [float(x + bw / 2.0), float(y + bh / 2.0)],
                "side": side,
                "area": float(area),
                "fill_ratio": float(fill_ratio),
            }
        )

    return candidates
