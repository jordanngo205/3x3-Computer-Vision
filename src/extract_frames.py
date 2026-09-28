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


def extract_frame(video_path: Path, timestamp: float, output_path: Path) -> None:
    ffmpeg = ffmpeg_exe()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-update",
        "1",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())


def build_timestamps(
    explicit_timestamps: List[float] | None,
    every: float | None,
    start: float,
    end: float | None,
    duration_seconds: float | None,
) -> List[float]:
    if explicit_timestamps:
        return explicit_timestamps
    if every is None:
        raise ValueError("Provide either --timestamps or --every.")

    hard_end = end if end is not None else duration_seconds
    if hard_end is None:
        raise ValueError("Could not infer clip duration. Pass --end explicitly.")

    timestamps: List[float] = []
    current = start
    while current <= hard_end + 1e-9:
        timestamps.append(round(current, 3))
        current += every
    return timestamps


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract still frames from a video clip.")
    parser.add_argument("--video", required=True, help="Path to the source video.")
    parser.add_argument("--out-dir", required=True, help="Directory for extracted frames.")
    parser.add_argument("--timestamps", nargs="*", type=float, help="Explicit timestamps in seconds.")
    parser.add_argument("--every", type=float, help="Extract every N seconds.")
    parser.add_argument("--start", type=float, default=0.0, help="Start time for --every mode.")
    parser.add_argument("--end", type=float, help="End time for --every mode.")
    parser.add_argument("--metadata-out", help="Optional JSON output for parsed clip metadata.")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    metadata = probe_video(video_path)
    timestamps = build_timestamps(
        explicit_timestamps=args.timestamps,
        every=args.every,
        start=args.start,
        end=args.end,
        duration_seconds=metadata.get("duration_seconds"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    for index, timestamp in enumerate(timestamps):
        file_name = f"frame_{index:03d}_{timestamp:06.2f}s.png"
        output_path = out_dir / file_name
        extract_frame(video_path, timestamp, output_path)
        saved_files.append(str(output_path))

    metadata["timestamps_extracted"] = timestamps
    metadata["files"] = saved_files

    if args.metadata_out:
        metadata_out = Path(args.metadata_out).expanduser().resolve()
        metadata_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
