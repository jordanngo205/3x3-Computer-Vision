from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import List

import imageio_ffmpeg


def ffmpeg_exe() -> str:
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError:
        binaries_dir = Path(imageio_ffmpeg.__file__).resolve().parent / "binaries"
        candidates = sorted(path for path in binaries_dir.glob("ffmpeg-*") if path.is_file())
        if not candidates:
            raise
        return str(candidates[0])


def parse_duration_seconds(ffmpeg_output: str) -> float | None:
    match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", ffmpeg_output)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_stream_metadata(ffmpeg_output: str) -> dict:
    match = re.search(r"Video: .*?, (\d+)x(\d+).*?, ([0-9.]+)\s*fps", ffmpeg_output)
    if not match:
        return {}
    width, height, fps = match.groups()
    return {
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
    }


def probe_video(video_path: Path) -> dict:
    ffmpeg = ffmpeg_exe()
    command = [
        ffmpeg,
        "-hide_banner",
        "-ss",
        "0",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    metadata = parse_stream_metadata(completed.stderr)
    metadata["duration_seconds"] = parse_duration_seconds(completed.stderr)
    metadata["video_path"] = str(video_path)
    metadata["ffmpeg"] = ffmpeg
    return metadata
