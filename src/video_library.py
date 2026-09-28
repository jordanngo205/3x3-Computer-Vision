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


def metadata_path_for(root: Path, slug: str) -> Path:
    return profiles_root(root) / slug / "metadata.json"


def load_profile(root: Path, slug: str) -> Dict[str, Any]:
    metadata_path = metadata_path_for(root, slug)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Unknown video slug: {slug}")
    return load_json(metadata_path)


def profile_paths(root: Path, slug: str) -> Dict[str, Path]:
    profile_dir = profiles_root(root) / slug
    return {
        "profile_dir": profile_dir,
        "metadata": profile_dir / "metadata.json",
        "frames_dir": profile_dir / "frames",
        "candidate_frames_dir": profile_dir / "candidate_frames",
        "review_clips_dir": profile_dir / "review_clips",
        "review_frames_dir": profile_dir / "review_frames",
        "review_gifs_dir": profile_dir / "review_gifs",
        "calibration": profile_dir / "calibration.json",
        "auto_calibration_debug_dir": profile_dir / "debug",
        "calibration_matrix": profile_dir / "calibration_matrix.json",
        "events": profile_dir / "events.csv",
        "shot_candidates": profile_dir / "shot_candidates.csv",
        "projected_shots": profile_dir / "projected_shots.csv",
        "shot_chart": profile_dir / "shot_chart.png",
    }


def ensure_video_profile(root: Path, video_path: Path, refresh: bool = False) -> Dict[str, Any]:
    slug = slugify(video_path.stem)
    paths = profile_paths(root, slug)
    video_stat = video_path.stat()

    if paths["metadata"].exists() and not refresh:
        cached = load_json(paths["metadata"])
        if (
            cached.get("video_path") == relative_to_root(root, video_path)
            and cached.get("video_size_bytes") == video_stat.st_size
            and cached.get("video_mtime_ns") == video_stat.st_mtime_ns
        ):
            frame_paths = [root / rel for rel in cached.get("sample_frames", [])]
            if frame_paths and all(path.exists() for path in frame_paths):
                cached["calibration_exists"] = paths["calibration"].exists()
                cached["candidate_exists"] = paths["shot_candidates"].exists()
                cached["events_exists"] = paths["events"].exists()
                cached["projected_exists"] = paths["projected_shots"].exists()
                cached["shot_chart_exists"] = paths["shot_chart"].exists()
                save_json(paths["metadata"], cached)
                return cached

    metadata = probe_video(video_path)
    timestamps = default_sample_timestamps(metadata.get("duration_seconds"))
    paths["frames_dir"].mkdir(parents=True, exist_ok=True)

    sample_frames = []
    for index, timestamp in enumerate(timestamps):
        output_path = paths["frames_dir"] / f"frame_{index:03d}_{timestamp:06.2f}s.png"
        if refresh or not output_path.exists():
            extract_frame(video_path, timestamp, output_path)
        sample_frames.append(relative_to_root(root, output_path))

    payload: Dict[str, Any] = {
        "slug": slug,
        "video_name": video_path.name,
        "video_path": relative_to_root(root, video_path),
        "video_size_bytes": video_stat.st_size,
        "video_mtime_ns": video_stat.st_mtime_ns,
        "duration_seconds": metadata.get("duration_seconds"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "fps": metadata.get("fps"),
        "sample_frames": sample_frames,
        "calibration_path": relative_to_root(root, paths["calibration"]),
        "shot_candidates_path": relative_to_root(root, paths["shot_candidates"]),
        "events_path": relative_to_root(root, paths["events"]),
        "projected_shots_path": relative_to_root(root, paths["projected_shots"]),
        "shot_chart_path": relative_to_root(root, paths["shot_chart"]),
        "calibration_exists": paths["calibration"].exists(),
        "candidate_exists": paths["shot_candidates"].exists(),
        "events_exists": paths["events"].exists(),
        "projected_exists": paths["projected_shots"].exists(),
        "shot_chart_exists": paths["shot_chart"].exists(),
    }
    save_json(paths["metadata"], payload)
    return payload


def build_video_library(root: Path, refresh: bool = False) -> List[Dict[str, Any]]:
    profiles = [ensure_video_profile(root, path, refresh=refresh) for path in list_video_files(root)]
    return sorted(profiles, key=lambda item: item["video_name"].lower())


def list_cached_profiles(root: Path) -> List[Dict[str, Any]]:
    profile_dir = profiles_root(root)
    if not profile_dir.exists():
        return []

    profiles = []
    for metadata_path in profile_dir.glob("*/metadata.json"):
        profile = load_json(metadata_path)
        slug = metadata_path.parent.name
        paths = profile_paths(root, slug)
        profile["calibration_exists"] = paths["calibration"].exists()
        profile["candidate_exists"] = paths["shot_candidates"].exists()
        profile["events_exists"] = paths["events"].exists()
        profile["projected_exists"] = paths["projected_shots"].exists()
        profile["shot_chart_exists"] = paths["shot_chart"].exists()
        profiles.append(profile)

    return sorted(profiles, key=lambda item: item["video_name"].lower())


def refresh_profile_status(root: Path, slug: str) -> Dict[str, Any]:
    profile = load_profile(root, slug)
    paths = profile_paths(root, slug)
    profile["calibration_exists"] = paths["calibration"].exists()
    profile["candidate_exists"] = paths["shot_candidates"].exists()
    profile["events_exists"] = paths["events"].exists()
    profile["projected_exists"] = paths["projected_shots"].exists()
    profile["shot_chart_exists"] = paths["shot_chart"].exists()
    save_json(paths["metadata"], profile)
    return profile
