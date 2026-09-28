from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

from PIL import Image, ImageDraw

COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0
FREE_THROW_LINE_FROM_BASELINE_FT = 19.0
LANE_WIDTH_FT = 12.0
CENTER_CIRCLE_RADIUS_FT = 6.0
RIM_OFFSET_FROM_BASELINE_FT = 5.25
RIM_RADIUS_FT = 0.75
BACKBOARD_OFFSET_FROM_BASELINE_FT = 4.0
BACKBOARD_HALF_WIDTH_FT = 3.0


def landmark_points_ft() -> Dict[str, Tuple[float, float]]:
    lane_top = (COURT_WIDTH_FT - LANE_WIDTH_FT) / 2.0
    lane_bottom = lane_top + LANE_WIDTH_FT

    return {
        "left_baseline_top": (0.0, 0.0),
        "left_baseline_bottom": (0.0, COURT_WIDTH_FT),
        "right_baseline_top": (COURT_LENGTH_FT, 0.0),
        "right_baseline_bottom": (COURT_LENGTH_FT, COURT_WIDTH_FT),
        "midcourt_top": (COURT_LENGTH_FT / 2.0, 0.0),
        "midcourt_bottom": (COURT_LENGTH_FT / 2.0, COURT_WIDTH_FT),
        "center_circle_left": (COURT_LENGTH_FT / 2.0 - CENTER_CIRCLE_RADIUS_FT, COURT_WIDTH_FT / 2.0),
        "center_circle_right": (COURT_LENGTH_FT / 2.0 + CENTER_CIRCLE_RADIUS_FT, COURT_WIDTH_FT / 2.0),
        "center_circle_top": (COURT_LENGTH_FT / 2.0, COURT_WIDTH_FT / 2.0 - CENTER_CIRCLE_RADIUS_FT),
        "center_circle_bottom": (COURT_LENGTH_FT / 2.0, COURT_WIDTH_FT / 2.0 + CENTER_CIRCLE_RADIUS_FT),
        "left_lane_top_left": (0.0, lane_top),
        "left_lane_top_right": (FREE_THROW_LINE_FROM_BASELINE_FT, lane_top),
        "left_lane_bottom_left": (0.0, lane_bottom),
        "left_lane_bottom_right": (FREE_THROW_LINE_FROM_BASELINE_FT, lane_bottom),
        "right_lane_top_left": (COURT_LENGTH_FT - FREE_THROW_LINE_FROM_BASELINE_FT, lane_top),
        "right_lane_top_right": (COURT_LENGTH_FT, lane_top),
        "right_lane_bottom_left": (COURT_LENGTH_FT - FREE_THROW_LINE_FROM_BASELINE_FT, lane_bottom),
        "right_lane_bottom_right": (COURT_LENGTH_FT, lane_bottom),
        "left_free_throw_center": (FREE_THROW_LINE_FROM_BASELINE_FT, COURT_WIDTH_FT / 2.0),
        "right_free_throw_center": (COURT_LENGTH_FT - FREE_THROW_LINE_FROM_BASELINE_FT, COURT_WIDTH_FT / 2.0),
        "left_rim_center": (RIM_OFFSET_FROM_BASELINE_FT, COURT_WIDTH_FT / 2.0),
        "right_rim_center": (COURT_LENGTH_FT - RIM_OFFSET_FROM_BASELINE_FT, COURT_WIDTH_FT / 2.0),
    }


def court_image_size(scale_px_per_ft: float, margin_px: int) -> Tuple[int, int]:
    width = int(round(COURT_LENGTH_FT * scale_px_per_ft + 2 * margin_px))
    height = int(round(COURT_WIDTH_FT * scale_px_per_ft + 2 * margin_px))
    return width, height


def feet_to_pixels(point_ft: Tuple[float, float], scale_px_per_ft: float, margin_px: int) -> Tuple[float, float]:
    x_ft, y_ft = point_ft
    return margin_px + x_ft * scale_px_per_ft, margin_px + y_ft * scale_px_per_ft


def draw_court(
    scale_px_per_ft: float = 20.0,
    margin_px: int = 40,
    annotate_landmarks: bool = True,
) -> Image.Image:
    image = Image.new("RGB", court_image_size(scale_px_per_ft, margin_px), "#f7f2e8")
    draw = ImageDraw.Draw(image)

    def rect(a_ft: Tuple[float, float], b_ft: Tuple[float, float], color: str, width: int = 3) -> None:
        ax, ay = feet_to_pixels(a_ft, scale_px_per_ft, margin_px)
        bx, by = feet_to_pixels(b_ft, scale_px_per_ft, margin_px)
        draw.rectangle([ax, ay, bx, by], outline=color, width=width)

    def line(a_ft: Tuple[float, float], b_ft: Tuple[float, float], color: str, width: int = 3) -> None:
        ax, ay = feet_to_pixels(a_ft, scale_px_per_ft, margin_px)
        bx, by = feet_to_pixels(b_ft, scale_px_per_ft, margin_px)
        draw.line([ax, ay, bx, by], fill=color, width=width)

    def circle(center_ft: Tuple[float, float], radius_ft: float, color: str, width: int = 3) -> None:
        cx, cy = feet_to_pixels(center_ft, scale_px_per_ft, margin_px)
        radius_px = radius_ft * scale_px_per_ft
        draw.ellipse(
            [cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px],
            outline=color,
            width=width,
        )

    lane_top = (COURT_WIDTH_FT - LANE_WIDTH_FT) / 2.0
    lane_bottom = lane_top + LANE_WIDTH_FT
    court_line = "#151515"
    accent = "#b42222"

    rect((0.0, 0.0), (COURT_LENGTH_FT, COURT_WIDTH_FT), court_line, width=4)
    line((COURT_LENGTH_FT / 2.0, 0.0), (COURT_LENGTH_FT / 2.0, COURT_WIDTH_FT), court_line)
    circle((COURT_LENGTH_FT / 2.0, COURT_WIDTH_FT / 2.0), CENTER_CIRCLE_RADIUS_FT, court_line)

    rect((0.0, lane_top), (FREE_THROW_LINE_FROM_BASELINE_FT, lane_bottom), court_line)
    rect(
        (COURT_LENGTH_FT - FREE_THROW_LINE_FROM_BASELINE_FT, lane_top),
        (COURT_LENGTH_FT, lane_bottom),
        court_line,
    )
    circle((FREE_THROW_LINE_FROM_BASELINE_FT, COURT_WIDTH_FT / 2.0), CENTER_CIRCLE_RADIUS_FT, court_line)
    circle((COURT_LENGTH_FT - FREE_THROW_LINE_FROM_BASELINE_FT, COURT_WIDTH_FT / 2.0), CENTER_CIRCLE_RADIUS_FT, court_line)

    line(
        (BACKBOARD_OFFSET_FROM_BASELINE_FT, COURT_WIDTH_FT / 2.0 - BACKBOARD_HALF_WIDTH_FT),
        (BACKBOARD_OFFSET_FROM_BASELINE_FT, COURT_WIDTH_FT / 2.0 + BACKBOARD_HALF_WIDTH_FT),
        accent,
    )
    line(
        (COURT_LENGTH_FT - BACKBOARD_OFFSET_FROM_BASELINE_FT, COURT_WIDTH_FT / 2.0 - BACKBOARD_HALF_WIDTH_FT),
        (COURT_LENGTH_FT - BACKBOARD_OFFSET_FROM_BASELINE_FT, COURT_WIDTH_FT / 2.0 + BACKBOARD_HALF_WIDTH_FT),
        accent,
    )
    circle((RIM_OFFSET_FROM_BASELINE_FT, COURT_WIDTH_FT / 2.0), RIM_RADIUS_FT, accent)
    circle((COURT_LENGTH_FT - RIM_OFFSET_FROM_BASELINE_FT, COURT_WIDTH_FT / 2.0), RIM_RADIUS_FT, accent)

    if annotate_landmarks:
        for name, point_ft in landmark_points_ft().items():
            px, py = feet_to_pixels(point_ft, scale_px_per_ft, margin_px)
            dot = 5
            draw.ellipse([px - dot, py - dot, px + dot, py + dot], fill=accent, outline=accent)
            draw.text((px + 8, py - 8), name, fill=accent)

    return image


def write_landmark_file(path: Path) -> None:
    payload = {
        "court_length_ft": COURT_LENGTH_FT,
        "court_width_ft": COURT_WIDTH_FT,
        "landmarks_ft": {name: [point[0], point[1]] for name, point in landmark_points_ft().items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_landmark_names() -> Iterable[str]:
    return landmark_points_ft().keys()
