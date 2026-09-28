from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

try:
    from .auto_calibrate import (
        backboard_proximity_score,
        build_dark_mask,
        detect_backboard_candidates,
        lane_landmark_names,
        order_quad,
        red_pixel_score,
    )
    from .court_geometry import landmark_points_ft
except ImportError:
    from auto_calibrate import (
        backboard_proximity_score,
        build_dark_mask,
        detect_backboard_candidates,
        lane_landmark_names,
        order_quad,
        red_pixel_score,
    )
    from court_geometry import landmark_points_ft

COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0
SCAN_STEP_S = 0.25
MIN_CANDIDATE_SCORE = 145.0
SUPPRESSION_WINDOW_S = 2.0
RELEASE_OFFSET_S = 0.75
MAX_CANDIDATES = 20
TEMPORAL_PROVISIONAL_MULTIPLIER = 4
TEMPORAL_MIDPOINTS = (0.25, 0.5, 0.75)
MAX_PEAK_RIM_DISTANCE_PX = 52.0
MIN_RELEASE_RIM_DISTANCE_PX = 70.0
MIN_APPROACH_DELTA_PX = 34.0
TEMPORAL_SIDE_MARGIN_PX = 55.0
TEMPORAL_SEARCH_RADIUS_PX = 195.0
MAX_CLUSTER_CONTACT_DISTANCE_PX = 34.0
MIN_CLUSTER_APPROACH_DELTA_PX = 12.0

CANDIDATE_FIELDNAMES = [
    "timestamp",
    "frame_image",
    "label",
    "pixel_x",
    "pixel_y",
    "result",
    "play_type",
    "notes",
    "score",
    "peak_timestamp",
    "peak_ball_x",
    "peak_ball_y",
    "peak_board_x",
    "peak_board_y",
    "peak_side",
    "court_x_ft",
    "court_y_ft",
    "in_bounds",
]


def load_matrix(calibration_matrix_path: Path | None) -> np.ndarray | None:
    if calibration_matrix_path is None or not calibration_matrix_path.exists():
        return None
    payload = json.loads(calibration_matrix_path.read_text(encoding="utf-8"))
    return np.array(payload["matrix"], dtype=np.float32)


def project_pixel(matrix: np.ndarray, x: float, y: float) -> tuple[float, float, bool]:
    points = np.array([[[float(x), float(y)]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)[0]
    court_x = float(projected[0])
    court_y = float(projected[1])
    in_bounds = 0.0 <= court_x <= COURT_LENGTH_FT and 0.0 <= court_y <= COURT_WIDTH_FT
    return court_x, court_y, in_bounds


def rim_candidates_from_lanes(image: np.ndarray) -> List[Dict[str, Any]]:
    h, w = image.shape[:2]
    backboards = detect_backboard_candidates(image)
    candidates: List[Dict[str, Any]] = []

    for vmax in (55, 65, 75, 90):
        mask = build_dark_mask(image, vmax)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 3200 or area > 95000:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            if not (0.07 * w <= bw <= 0.48 * w):
                continue
            if not (0.08 * h <= bh <= 0.45 * h):
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) < 4:
                continue

            quad = cv2.boxPoints(cv2.minAreaRect(contour)) if len(approx) != 4 else approx.reshape(-1, 2)
            ordered = order_quad(np.asarray(quad, dtype=np.float32))
            top_width = np.linalg.norm(ordered[1] - ordered[0])
            bottom_width = np.linalg.norm(ordered[2] - ordered[3])
            left_height = np.linalg.norm(ordered[3] - ordered[0])
            right_height = np.linalg.norm(ordered[2] - ordered[1])
            width_ratio = max(top_width, bottom_width) / max(1.0, min(top_width, bottom_width))
            height_ratio = max(left_height, right_height) / max(1.0, min(left_height, right_height))
            if width_ratio > 2.8 or height_ratio > 2.8:
                continue

            center_x = x + bw / 2.0
            center_y = y + bh / 2.0
            if not (h * 0.18 <= center_y <= h * 0.72):
                continue

            side = "left" if center_x < w / 2.0 else "right"
            baseline_margin = x if side == "left" else w - (x + bw)
            if baseline_margin > 0.35 * w:
                continue

            lane_candidate = {
                "score": 0.0,
                "side": side,
                "quad": ordered.tolist(),
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "area": float(area),
                "threshold_vmax": vmax,
            }
            extent = area / max(1.0, bw * bh)
            rim_score = red_pixel_score(image, lane_candidate)
            board_score = backboard_proximity_score(lane_candidate, backboards)
            if board_score < 0.42:
                continue

            landmark_names = lane_landmark_names(side)
            dst = np.array([landmark_points_ft()[name] for name in landmark_names], dtype=np.float32)
            homography, _ = cv2.findHomography(ordered, dst, method=0)
            if homography is None:
                continue

            try:
                inv_homography = np.linalg.inv(homography)
            except np.linalg.LinAlgError:
                continue
            rim_name = "left_rim_center" if side == "left" else "right_rim_center"
            rim_ft = np.array([[[*landmark_points_ft()[rim_name]]]], dtype=np.float32)
            rim_px = cv2.perspectiveTransform(rim_ft, inv_homography).reshape(-1, 2)[0]
            rim_x = float(rim_px[0])
            rim_y = float(rim_px[1])
            if not (-120.0 <= rim_x <= w + 120.0 and -80.0 <= rim_y <= h + 120.0):
                continue

            lane_score = area * extent + 0.6 * rim_score + 12000.0 * board_score
            lane_candidate.update(
                {
                    "lane_score": float(lane_score),
                    "extent": float(extent),
                    "rim_score": float(rim_score),
                    "backboard_score": float(board_score),
                    "rim_px": [rim_x, rim_y],
                }
            )
            candidates.append(lane_candidate)

    candidates.sort(key=lambda item: item["lane_score"], reverse=True)
    deduped: List[Dict[str, Any]] = []
    for candidate in candidates:
        rim_x, rim_y = candidate["rim_px"]
        if all(abs(rim_x - keep["rim_px"][0]) > 28.0 or abs(rim_y - keep["rim_px"][1]) > 28.0 for keep in deduped):
            deduped.append(candidate)
        if len(deduped) >= 6:
            break
    return deduped


def orange_candidates(image: np.ndarray) -> List[Dict[str, Any]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_a = cv2.inRange(hsv, (4, 90, 70), (24, 255, 255))
    mask_b = cv2.inRange(hsv, (0, 40, 80), (12, 255, 255))
    mask = cv2.bitwise_or(mask_a, mask_b)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[Dict[str, Any]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 8 or area > 220:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        if not (3 <= bw <= 18 and 3 <= bh <= 18):
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = 0.0 if perimeter <= 0 else 4 * math.pi * area / (perimeter * perimeter)
        fill_ratio = area / max(1.0, bw * bh)
        if circularity < 0.45 and fill_ratio < 0.28:
            continue

        candidates.append(
            {
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "center": [float(x + bw / 2.0), float(y + bh / 2.0)],
                "area": float(area),
                "circularity": float(circularity),
                "fill_ratio": float(fill_ratio),
            }
        )

    return candidates


def read_frame_at(capture: cv2.VideoCapture, timestamp_s: float) -> np.ndarray | None:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps > 1.0:
        frame_index = max(0, int(round(float(timestamp_s) * fps)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    else:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_s) * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    return frame


def pick_temporal_ball_candidate(
    frame: np.ndarray,
    *,
    rim_x: float,
    rim_y: float,
    side: str,
    target_distance: float,
    min_distance: float,
    max_distance: float,
    reference_center: tuple[float, float] | None,
) -> Dict[str, Any] | None:
    best: Dict[str, Any] | None = None
    for candidate in orange_candidates(frame):
        x, y = candidate["center"]
        if abs(x - rim_x) > TEMPORAL_SEARCH_RADIUS_PX or abs(y - rim_y) > TEMPORAL_SEARCH_RADIUS_PX:
            continue
        if side == "right" and x > rim_x + TEMPORAL_SIDE_MARGIN_PX:
            continue
        if side == "left" and x < rim_x - TEMPORAL_SIDE_MARGIN_PX:
            continue

        distance_to_rim = math.hypot(x - rim_x, y - rim_y)
        if distance_to_rim < min_distance or distance_to_rim > max_distance:
            continue

        score = (
            -abs(distance_to_rim - target_distance) * 1.55
            + candidate["circularity"] * 12.0
            + candidate["area"] * 0.25
        )
        if reference_center is not None:
            score -= math.hypot(x - reference_center[0], y - reference_center[1]) * 0.22

        if best is None or score > best["score"]:
            best = {
                "center": (float(x), float(y)),
                "distance_to_rim": float(distance_to_rim),
                "score": float(score),
            }

    return best
