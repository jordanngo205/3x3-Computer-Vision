from __future__ import annotations

import csv
import imageio_ffmpeg
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from PIL import Image

try:
    from .auto_calibrate import auto_calibrate_frames
    from .court_geometry import court_image_size, draw_court, feet_to_pixels, load_landmark_names
    from .shot_candidates import candidate_rows_to_events, mine_shot_candidates, read_candidate_rows
    from .video_library import (
        build_video_library,
        list_cached_profiles,
        load_profile,
        profile_paths,
        project_root,
        refresh_profile_status,
    )
except ImportError:
    from auto_calibrate import auto_calibrate_frames
    from court_geometry import court_image_size, draw_court, feet_to_pixels, load_landmark_names
    from shot_candidates import candidate_rows_to_events, mine_shot_candidates, read_candidate_rows
    from video_library import (
        build_video_library,
        list_cached_profiles,
        load_profile,
        profile_paths,
        project_root,
        refresh_profile_status,
    )

ROOT = project_root()
CHART_SCALE = 20.0
CHART_MARGIN = 40
REVIEW_CLIP_BEFORE_S = 2.0
REVIEW_CLIP_AFTER_S = 2.0
REVIEW_FRAME_OFFSETS_S = [-1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 1.5]
REVIEW_GIF_DURATION_MS = 220
PLAY_TYPES = [
    "transition",
    "pnr",
    "spot_up",
    "post_up",
    "isolation",
    "cut",
    "handoff",
    "off_screen",
    "other",
]

app = Flask(__name__, template_folder=str(ROOT / "templates"))
app.secret_key = "local-shot-tracker"


def media_url(relative_path: str | None) -> str | None:
    if not relative_path:
        return None
    return url_for("media", relpath=relative_path)


app.jinja_env.globals["media_url"] = media_url


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def ensure_clean_court_reference() -> str:
    output_path = ROOT / "data" / "reference" / "court_clean.png"
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = draw_court(
            scale_px_per_ft=CHART_SCALE,
            margin_px=CHART_MARGIN,
            annotate_landmarks=False,
        )
        image.save(output_path)
    return str(output_path.relative_to(ROOT))


def bundled_ffmpeg_exe() -> str:
    binaries_dir = Path(imageio_ffmpeg.__file__).resolve().parent / "binaries"
    candidates = sorted(path for path in binaries_dir.glob("ffmpeg-*") if path.is_file())
    if not candidates:
        return imageio_ffmpeg.get_ffmpeg_exe()
    return str(candidates[0])


def extract_frame_fast(video_path: Path, timestamp_s: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        bundled_ffmpeg_exe(),
        "-y",
        "-ss",
        f"{timestamp_s:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-update",
        "1",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "Frame extraction failed.")


def review_clip_info(out_path: Path, clip_start_s: float, clip_end_s: float, timestamp_s: float) -> Dict[str, Any]:
    return {
        "path": str(out_path.relative_to(ROOT)),
        "clip_start_s": clip_start_s,
        "clip_end_s": clip_end_s,
        "event_timestamp_s": timestamp_s,
    }


def is_valid_review_clip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    command = [
        bundled_ffmpeg_exe(),
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0
