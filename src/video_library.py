from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    from .extract_frames import extract_frame, probe_video
except ImportError:
    from extract_frames import extract_frame, probe_video

VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v"}


def slugify(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def profiles_root(root: Path) -> Path:
    return root / "data" / "profiles"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def relative_to_root(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def default_sample_timestamps(duration_seconds: float | None) -> List[float]:
    if not duration_seconds or duration_seconds <= 0:
        return [0.0]

    fractions = [0.0, 0.25, 0.5, 0.75]
    timestamps = []
    for fraction in fractions:
        timestamp = round(duration_seconds * fraction, 2)
        if not timestamps or abs(timestamp - timestamps[-1]) >= 0.5:
            timestamps.append(timestamp)
    return timestamps


def list_video_files(root: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
    )
