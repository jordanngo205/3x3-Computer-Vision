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


def red_pixel_score(image: np.ndarray, candidate: Dict[str, Any]) -> float:
    h, w = image.shape[:2]
    quad = np.asarray(candidate["quad"], dtype=np.float32)
    x_min = float(np.min(quad[:, 0]))
    x_max = float(np.max(quad[:, 0]))
    y_min = float(np.min(quad[:, 1]))
    y_max = float(np.max(quad[:, 1]))
    bw = max(1.0, x_max - x_min)
    bh = max(1.0, y_max - y_min)

    if candidate["side"] == "left":
        rx0 = max(0, int(x_min - 0.35 * bw))
        rx1 = min(w, int(x_min + 0.55 * bw))
    else:
        rx0 = max(0, int(x_max - 0.55 * bw))
        rx1 = min(w, int(x_max + 0.35 * bw))

    ry0 = max(0, int(y_min - 0.95 * bh))
    ry1 = min(h, int(y_min + 0.35 * bh))
    if rx1 <= rx0 or ry1 <= ry0:
        return 0.0

    region = image[ry0:ry1, rx0:rx1]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask_a = cv2.inRange(hsv, (0, 70, 70), (12, 255, 255))
    mask_b = cv2.inRange(hsv, (165, 70, 70), (180, 255, 255))
    mask = cv2.bitwise_or(mask_a, mask_b)
    return float(cv2.countNonZero(mask))


def backboard_proximity_score(candidate: Dict[str, Any], backboards: List[Dict[str, Any]]) -> float:
    if not backboards:
        return 0.0

    quad = np.asarray(candidate["quad"], dtype=np.float32)
    candidate_center = np.mean(quad, axis=0)
    best = 0.0
    for backboard in backboards:
        if backboard["side"] != candidate["side"]:
            continue
        bx, by = backboard["center"]
        dx = abs(candidate_center[0] - bx)
        dy = candidate_center[1] - by
        if dy < 0:
            continue
        score = 1.0 / (1.0 + 0.015 * dx + 0.01 * dy)
        best = max(best, score)
    return float(best)


def find_lane_candidate(image: np.ndarray) -> Dict[str, Any] | None:
    h, w = image.shape[:2]
    best: Dict[str, Any] | None = None
    backboards = detect_backboard_candidates(image)

    for vmax in (55, 65, 75, 90):
        mask = build_dark_mask(image, vmax)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 4000 or area > 90000:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            if not (0.08 * w <= bw <= 0.45 * w):
                continue
            if not (0.08 * h <= bh <= 0.45 * h):
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) < 4:
                continue

            if len(approx) != 4:
                rect = cv2.minAreaRect(contour)
                quad = cv2.boxPoints(rect)
            else:
                quad = approx.reshape(-1, 2)

            ordered = order_quad(np.asarray(quad, dtype=np.float32))
            top_width = np.linalg.norm(ordered[1] - ordered[0])
            bottom_width = np.linalg.norm(ordered[2] - ordered[3])
            left_height = np.linalg.norm(ordered[3] - ordered[0])
            right_height = np.linalg.norm(ordered[2] - ordered[1])
            width_ratio = max(top_width, bottom_width) / max(1.0, min(top_width, bottom_width))
            height_ratio = max(left_height, right_height) / max(1.0, min(left_height, right_height))
            if width_ratio > 2.5 or height_ratio > 2.5:
                continue

            extent = area / max(1.0, bw * bh)
            center_x = x + bw / 2.0
            center_y = y + bh / 2.0
            side = "left" if center_x < w / 2.0 else "right"
            baseline_margin = x if side == "left" else w - (x + bw)
            if baseline_margin > 0.22 * w:
                continue

            edge_penalty = 0.0
            if x <= 5 or x + bw >= w - 5:
                edge_penalty += 0.15
            if y <= int(h * 0.2):
                edge_penalty += 0.1

            candidate = {
                "score": 0.0,
                "side": side,
                "quad": ordered.tolist(),
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "area": float(area),
                "threshold_vmax": vmax,
                "baseline_margin": float(baseline_margin),
            }
            rim_score = red_pixel_score(image, candidate)
            board_score = backboard_proximity_score(candidate, backboards)
            score = area * extent * (1.0 - edge_penalty) + 40.0 * rim_score + 25000.0 * board_score
            candidate["rim_score"] = float(rim_score)
            candidate["backboard_score"] = float(board_score)
            candidate["score"] = float(score)
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    return best


def write_debug_overlay(image: np.ndarray, candidate: Dict[str, Any], output_path: Path) -> None:
    overlay = image.copy()
    quad = np.asarray(candidate["quad"], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [quad], True, (0, 255, 0), 3)
    for index, point in enumerate(candidate["quad"]):
        x, y = int(point[0]), int(point[1])
        cv2.circle(overlay, (x, y), 6, (255, 80, 80), -1)
        cv2.putText(
            overlay,
            str(index),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def calibration_payload(video_name: str, frame_image: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
    names = lane_landmark_names(candidate["side"])
    landmarks_px = {
        name: [round(float(point[0]), 2), round(float(point[1]), 2)]
        for name, point in zip(names, candidate["quad"])
    }
    return {
        "video": video_name,
        "frame_image": frame_image,
        "landmarks_px": landmarks_px,
        "auto_calibration": {
            "method": "dark_paint_contour",
            "score": round(float(candidate["score"]), 2),
            "side": candidate["side"],
            "bbox": candidate["bbox"],
            "threshold_vmax": candidate["threshold_vmax"],
        },
    }


def auto_calibrate_frames(
    video_name: str,
    frame_paths: Iterable[Path],
    relative_frame_paths: Iterable[str] | None = None,
    debug_dir: Path | None = None,
) -> Dict[str, Any]:
    relative_lookup = (
        list(relative_frame_paths)
        if relative_frame_paths is not None
        else [str(path) for path in frame_paths]
    )
    best_payload: Dict[str, Any] | None = None
    best_candidate: Dict[str, Any] | None = None
    best_frame_path: Path | None = None

    for frame_path, relative_frame_path in zip(frame_paths, relative_lookup):
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        candidate = find_lane_candidate(image)
        if candidate is None:
            continue
        payload = calibration_payload(video_name, relative_frame_path, candidate)
        if best_candidate is None or candidate["score"] > best_candidate["score"]:
            best_candidate = candidate
            best_payload = payload
            best_frame_path = frame_path

    if best_payload is None or best_candidate is None or best_frame_path is None:
        raise RuntimeError("Auto calibration could not find a usable painted key in the sampled frames.")

    if debug_dir is not None:
        image = cv2.imread(str(best_frame_path))
        if image is not None:
            write_debug_overlay(image, best_candidate, debug_dir / f"{best_frame_path.stem}_auto_calibration.png")

    return best_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-calibrate a video by detecting the painted key.")
    parser.add_argument("--video-name", required=True)
    parser.add_argument("--frames", nargs="+", required=True, help="Absolute or relative paths to frame images.")
    parser.add_argument("--frame-labels", nargs="*", help="Optional relative frame labels to store in calibration.")
    parser.add_argument("--out", required=True, help="Calibration JSON output path.")
    parser.add_argument("--debug-dir", help="Optional directory for debug overlays.")
    args = parser.parse_args()

    frame_paths = [Path(item).expanduser().resolve() for item in args.frames]
    frame_labels = args.frame_labels if args.frame_labels else [str(path) for path in frame_paths]
    debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else None
    payload = auto_calibrate_frames(args.video_name, frame_paths, frame_labels, debug_dir=debug_dir)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
