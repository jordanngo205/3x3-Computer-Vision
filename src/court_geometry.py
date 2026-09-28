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
