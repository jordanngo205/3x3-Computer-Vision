from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np


def load_matrix(calibration_path: Path) -> tuple[np.ndarray, dict]:
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    matrix = np.array(payload["matrix"], dtype=np.float32)
    return matrix, payload


def project_points(matrix: np.ndarray, points: List[List[float]]) -> np.ndarray:
    packed = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(packed, matrix).reshape(-1, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project clicked event pixels into court coordinates.")
    parser.add_argument("--events", required=True, help="CSV with pixel_x and pixel_y columns.")
    parser.add_argument("--calibration", required=True, help="Calibration JSON from compute_calibration.py.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args()

    events_path = Path(args.events).expanduser().resolve()
    calibration_path = Path(args.calibration).expanduser().resolve()
    output_path = Path(args.out).expanduser().resolve()

    matrix, calibration = load_matrix(calibration_path)
    bounds = calibration["court_bounds_ft"]

    with events_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError("The events CSV is empty.")

    points = []
    for row in rows:
        if "pixel_x" not in row or "pixel_y" not in row:
            raise ValueError("Events CSV must contain pixel_x and pixel_y columns.")
        points.append([float(row["pixel_x"]), float(row["pixel_y"])])

    projected = project_points(matrix, points)

    fieldnames = list(rows[0].keys()) + ["court_x_ft", "court_y_ft", "in_bounds"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, point in zip(rows, projected):
            x_ft = float(point[0])
            y_ft = float(point[1])
            in_bounds = bounds["x_min"] <= x_ft <= bounds["x_max"] and bounds["y_min"] <= y_ft <= bounds["y_max"]
            row["court_x_ft"] = f"{x_ft:.3f}"
            row["court_y_ft"] = f"{y_ft:.3f}"
            row["in_bounds"] = "true" if in_bounds else "false"
            writer.writerow(row)

    print(output_path)


if __name__ == "__main__":
    main()
