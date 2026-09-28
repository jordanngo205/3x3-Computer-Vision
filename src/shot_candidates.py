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


def score_frame_peak(image: np.ndarray) -> Dict[str, Any] | None:
    rim_candidates = rim_candidates_from_lanes(image)
    balls = orange_candidates(image)
    if not rim_candidates or not balls:
        return None

    best: Dict[str, Any] | None = None
    for ball in balls:
        ball_x, ball_y = ball["center"]
        for lane in rim_candidates:
            rim_x, rim_y = lane["rim_px"]
            dx = ball_x - rim_x
            dy = ball_y - rim_y
            abs_dx = abs(dx)
            abs_dy = abs(dy)
            if abs_dx > 44.0:
                continue
            if dy < -64.0 or dy > 36.0:
                continue

            vertical_bonus = max(0.0, 42.0 - abs(dy - 4.0))
            horizontal_bonus = max(0.0, 54.0 - abs_dx * 1.2)
            contact_bonus = 18.0 if -20.0 <= dy <= 24.0 else 0.0
            score = (
                lane["lane_score"] * 0.006
                + ball["area"] * 3.0
                + ball["circularity"] * 65.0
                + horizontal_bonus
                + max(0.0, 44.0 - abs_dy * 1.05)
                + vertical_bonus
                + contact_bonus
            )
            if best is None or score > best["score"]:
                best = {
                    "score": float(score),
                    "ball": ball,
                    "lane": lane,
                    "dx": float(dx),
                    "dy": float(dy),
                }

    return best


def release_pixel_from_ball(
    image: np.ndarray,
    peak: Dict[str, Any],
    release_ball_x: float,
    release_ball_y: float,
) -> tuple[float, float]:
    h, w = image.shape[:2]
    rim_x, rim_y = peak["lane"]["rim_px"]
    release_x = float(release_ball_x) * 0.84 + rim_x * 0.16
    release_y = float(release_ball_y) + h * 0.11
    release_x = max(0.0, min(float(w - 1), release_x))
    release_y = max(0.0, min(float(h - 1), release_y))
    return release_x, release_y


def backtrack_release_frame(
    capture: cv2.VideoCapture,
    *,
    peak_timestamp: float,
    duration_s: float,
    peak: Dict[str, Any],
    fallback_release_offset_s: float,
) -> tuple[float, np.ndarray, tuple[float, float]]:
    rim_x, rim_y = peak["lane"]["rim_px"]
    peak_ball_x, peak_ball_y = peak["ball"]["center"]
    peak_distance = math.hypot(peak_ball_x - rim_x, peak_ball_y - rim_y)
    side = peak["lane"]["side"]

    best_release: Dict[str, Any] | None = None
    for offset_s in (0.20, 0.35, 0.50, 0.65, 0.80, 0.95):
        release_timestamp = max(0.0, min(duration_s, peak_timestamp - offset_s))
        frame = read_frame_at(capture, release_timestamp)
        if frame is None:
            continue

        for candidate in orange_candidates(frame):
            x, y = candidate["center"]
            if abs(x - rim_x) > 180.0 or abs(y - rim_y) > 180.0:
                continue
            if side == "right" and x > rim_x + 35.0:
                continue
            if side == "left" and x < rim_x - 35.0:
                continue

            distance_to_rim = math.hypot(x - rim_x, y - rim_y)
            if distance_to_rim < peak_distance + 12.0:
                continue

            score = (
                -abs(distance_to_rim - 88.0) * 1.6
                + offset_s * 14.0
                + candidate["circularity"] * 10.0
                + candidate["area"] * 0.3
            )
            if best_release is None or score > best_release["score"]:
                best_release = {
                    "timestamp": release_timestamp,
                    "frame": frame.copy(),
                    "ball_center": (float(x), float(y)),
                    "score": float(score),
                }

    if best_release is not None:
        return best_release["timestamp"], best_release["frame"], best_release["ball_center"]

    release_timestamp = max(0.0, min(duration_s, peak_timestamp - fallback_release_offset_s))
    frame = read_frame_at(capture, release_timestamp)
    if frame is None:
        raise RuntimeError("Could not read the fallback release frame.")

    fallback_ball = peak["ball"]["center"]
    best_distance = None
    for candidate in orange_candidates(frame):
        x, y = candidate["center"]
        distance = abs(x - fallback_ball[0]) + abs(y - fallback_ball[1])
        if best_distance is None or distance < best_distance:
            best_distance = distance
            fallback_ball = (float(x), float(y))

    return release_timestamp, frame, (float(fallback_ball[0]), float(fallback_ball[1]))


def has_shot_trajectory(
    capture: cv2.VideoCapture,
    *,
    peak_timestamp: float,
    duration_s: float,
    peak: Dict[str, Any],
    release_timestamp: float,
    release_ball_center: tuple[float, float],
) -> bool:
    rim_x, rim_y = peak["lane"]["rim_px"]
    side = peak["lane"]["side"]
    peak_ball_x, peak_ball_y = peak["ball"]["center"]
    peak_distance = math.hypot(peak_ball_x - rim_x, peak_ball_y - rim_y)
    release_distance = math.hypot(release_ball_center[0] - rim_x, release_ball_center[1] - rim_y)

    if peak_distance > MAX_PEAK_RIM_DISTANCE_PX:
        return False
    if release_distance < max(MIN_RELEASE_RIM_DISTANCE_PX, peak_distance + MIN_APPROACH_DELTA_PX):
        return False

    delta_t = peak_timestamp - release_timestamp
    if delta_t < 0.18:
        return False

    release_entry = {
        "timestamp": float(release_timestamp),
        "center": (float(release_ball_center[0]), float(release_ball_center[1])),
        "distance_to_rim": float(release_distance),
    }
    peak_entry = {
        "timestamp": float(peak_timestamp),
        "center": (float(peak_ball_x), float(peak_ball_y)),
        "distance_to_rim": float(peak_distance),
    }

    reference_center = peak_entry["center"]
    midpoint_entries: List[Dict[str, Any]] = []
    for progress in sorted(TEMPORAL_MIDPOINTS, reverse=True):
        midpoint_timestamp = release_timestamp + delta_t * progress
        midpoint_timestamp = max(0.0, min(duration_s, midpoint_timestamp))
        frame = read_frame_at(capture, midpoint_timestamp)
        if frame is None:
            continue

        target_distance = release_distance + (peak_distance - release_distance) * progress
        midpoint = pick_temporal_ball_candidate(
            frame,
            rim_x=rim_x,
            rim_y=rim_y,
            side=side,
            target_distance=float(target_distance),
            min_distance=float(peak_distance + 6.0),
            max_distance=float(release_distance + 24.0),
            reference_center=reference_center,
        )
        if midpoint is None:
            continue

        midpoint_entries.append(
            {
                "timestamp": float(midpoint_timestamp),
                "center": midpoint["center"],
                "distance_to_rim": midpoint["distance_to_rim"],
            }
        )
        reference_center = midpoint["center"]

    if len(midpoint_entries) < 2:
        return False

    sequence = [release_entry, *sorted(midpoint_entries, key=lambda item: item["timestamp"]), peak_entry]
    approach_steps = 0
    for previous, current in zip(sequence, sequence[1:]):
        if current["distance_to_rim"] <= previous["distance_to_rim"] - 8.0:
            approach_steps += 1

    if approach_steps < len(sequence) - 1:
        return False

    post_timestamp = min(duration_s, peak_timestamp + min(0.25, delta_t * 0.4))
    post_frame = read_frame_at(capture, post_timestamp)
    if post_frame is None:
        return True

    post_candidate = pick_temporal_ball_candidate(
        post_frame,
        rim_x=rim_x,
        rim_y=rim_y,
        side=side,
        target_distance=float(max(peak_distance + 18.0, 55.0)),
        min_distance=float(peak_distance + 4.0),
        max_distance=float(release_distance + 24.0),
        reference_center=peak_entry["center"],
    )
    return post_candidate is not None or delta_t >= 0.55


def select_temporal_peaks(
    raw_candidates: List[Dict[str, Any]],
    min_score: float,
    suppression_window_s: float,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    eligible = [candidate for candidate in raw_candidates if candidate["score"] >= min_score]
    eligible.sort(key=lambda candidate: candidate["score"], reverse=True)

    selected: List[Dict[str, Any]] = []
    for candidate in eligible:
        candidate_time = candidate["peak_timestamp"]
        if all(abs(candidate_time - chosen["peak_timestamp"]) >= suppression_window_s for chosen in selected):
            selected.append(candidate)
        if len(selected) >= max_candidates:
            break

    selected.sort(key=lambda candidate: candidate["peak_timestamp"])
    return selected


def candidate_rim_distance(candidate: Dict[str, Any]) -> float:
    rim_x, rim_y = candidate["lane"]["rim_px"]
    ball_x, ball_y = candidate["ball"]["center"]
    return float(math.hypot(ball_x - rim_x, ball_y - rim_y))


def same_rim_context(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if a["lane"]["side"] != b["lane"]["side"]:
        return False
    return (
        abs(float(a["lane"]["rim_px"][0]) - float(b["lane"]["rim_px"][0])) <= 70.0
        and abs(float(a["lane"]["rim_px"][1]) - float(b["lane"]["rim_px"][1])) <= 90.0
    )


def cluster_raw_candidates(
    raw_candidates: List[Dict[str, Any]],
    scan_step_s: float,
) -> List[List[Dict[str, Any]]]:
    if not raw_candidates:
        return []

    sorted_candidates = sorted(raw_candidates, key=lambda candidate: float(candidate["peak_timestamp"]))
    max_gap_s = max(scan_step_s * 3.2, 0.78)
    clusters: List[List[Dict[str, Any]]] = []
    current_cluster: List[Dict[str, Any]] = []

    for candidate in sorted_candidates:
        if (
            current_cluster
            and float(candidate["peak_timestamp"]) - float(current_cluster[-1]["peak_timestamp"]) <= max_gap_s
            and same_rim_context(candidate, current_cluster[-1])
        ):
            current_cluster.append(candidate)
            continue

        if current_cluster:
            clusters.append(current_cluster)
        current_cluster = [candidate]

    if current_cluster:
        clusters.append(current_cluster)
    return clusters


def pick_shot_from_cluster(
    cluster: List[Dict[str, Any]],
    *,
    scan_step_s: float,
) -> Dict[str, Any] | None:
    if len(cluster) < 2:
        return None

    distances = [candidate_rim_distance(candidate) for candidate in cluster]
    contact_index = min(
        range(len(cluster)),
        key=lambda index: (distances[index], -float(cluster[index]["score"])),
    )
    contact_candidate = cluster[contact_index]
    contact_distance = distances[contact_index]
    if contact_distance > MAX_CLUSTER_CONTACT_DISTANCE_PX:
        return None

    best_release_index: int | None = None
    best_release_score: float | None = None
    contact_timestamp = float(contact_candidate["peak_timestamp"])
    for release_index in range(contact_index):
        release_distance = distances[release_index]
        if release_distance < contact_distance + MIN_CLUSTER_APPROACH_DELTA_PX:
            continue

        release_timestamp = float(cluster[release_index]["peak_timestamp"])
        if contact_timestamp - release_timestamp < max(scan_step_s * 0.75, 0.18):
            continue

        approach_distances = distances[release_index : contact_index + 1]
        approach_steps = 0
        for previous, current in zip(approach_distances, approach_distances[1:]):
            if current <= previous - 4.0:
                approach_steps += 1
        if approach_steps < max(1, len(approach_distances) - 2):
            continue

        release_score = (release_distance - contact_distance) * 4.0 + (contact_timestamp - release_timestamp) * 10.0
        if best_release_score is None or release_score > best_release_score:
            best_release_index = release_index
            best_release_score = float(release_score)

    if best_release_index is None:
        return None

    release_candidate = cluster[best_release_index]
    release_distance = distances[best_release_index]

    cluster_score = max(float(candidate["score"]) for candidate in cluster) + (release_distance - contact_distance) * 4.0
    return {
        "peak_candidate": contact_candidate,
        "release_candidate": release_candidate,
        "cluster_score": float(cluster_score),
    }


def build_sampled_sequence(
    raw_candidates: List[Dict[str, Any]],
    peak_candidate: Dict[str, Any],
    scan_step_s: float,
) -> List[Dict[str, Any]]:
    peak_timestamp = float(peak_candidate["peak_timestamp"])
    window = [
        candidate
        for candidate in raw_candidates
        if 0.0 <= peak_timestamp - float(candidate["peak_timestamp"]) <= 1.1
        and same_rim_context(candidate, peak_candidate)
    ]
    if not window:
        return []

    window.sort(key=lambda candidate: float(candidate["peak_timestamp"]))
    sequence = [window[-1]]
    max_gap_s = max(scan_step_s * 1.6, 0.36)
    for candidate in reversed(window[:-1]):
        if float(sequence[0]["peak_timestamp"]) - float(candidate["peak_timestamp"]) <= max_gap_s:
            sequence.insert(0, candidate)
        else:
            break
    return sequence


def pick_release_from_sampled_sequence(
    sequence: List[Dict[str, Any]],
    *,
    scan_step_s: float,
) -> Dict[str, Any] | None:
    if len(sequence) < 3:
        return None

    distances = [candidate_rim_distance(candidate) for candidate in sequence]
    peak_distance = distances[-1]
    if peak_distance > MAX_PEAK_RIM_DISTANCE_PX:
        return None

    approach_steps = 0
    for previous, current in zip(distances, distances[1:]):
        if current <= previous - 6.0:
            approach_steps += 1
    if approach_steps < max(2, len(distances) - 2):
        return None

    best_index = max(range(len(sequence) - 1), key=lambda index: distances[index])
    release_candidate = sequence[best_index]
    release_distance = distances[best_index]
    if release_distance < max(MIN_RELEASE_RIM_DISTANCE_PX, peak_distance + MIN_APPROACH_DELTA_PX):
        return None

    peak_timestamp = float(sequence[-1]["peak_timestamp"])
    release_timestamp = float(release_candidate["peak_timestamp"])
    if peak_timestamp - release_timestamp < max(scan_step_s * 0.75, 0.18):
        return None

    return release_candidate
