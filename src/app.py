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


def ensure_review_clip(profile: Dict[str, Any], event_number: int, timestamp_text: str | None) -> Dict[str, Any] | None:
    if not timestamp_text:
        return None

    try:
        timestamp_s = float(timestamp_text)
    except (TypeError, ValueError):
        return None

    video_relpath = profile.get("video_path")
    if not video_relpath:
        return None

    video_path = ROOT / str(video_relpath)
    if not video_path.exists():
        return None

    duration_s = float(profile.get("duration_seconds") or 0.0)
    clip_start_s = max(0.0, timestamp_s - REVIEW_CLIP_BEFORE_S)
    clip_end_s = timestamp_s + REVIEW_CLIP_AFTER_S
    if duration_s > 0:
        clip_end_s = min(duration_s, clip_end_s)
    clip_duration_s = max(0.25, clip_end_s - clip_start_s)

    paths = profile_paths(ROOT, profile["slug"])
    out_path = paths["review_clips_dir"] / f"event_{event_number:03d}_{timestamp_s:06.2f}s.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_out_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")

    if out_path.exists() and is_valid_review_clip(out_path):
        return review_clip_info(out_path, clip_start_s, clip_end_s, timestamp_s)

    if out_path.exists():
        out_path.unlink(missing_ok=True)
    temp_out_path.unlink(missing_ok=True)

    command = [
        bundled_ffmpeg_exe(),
        "-y",
        "-ss",
        f"{clip_start_s:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{clip_duration_s:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp_out_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        temp_out_path.unlink(missing_ok=True)
        return None
    if completed.returncode != 0:
        temp_out_path.unlink(missing_ok=True)
        return None
    os.replace(temp_out_path, out_path)

    if not is_valid_review_clip(out_path):
        out_path.unlink(missing_ok=True)
        return None

    return review_clip_info(out_path, clip_start_s, clip_end_s, timestamp_s)


def parse_event_timestamp(timestamp_text: str | None) -> float | None:
    if not timestamp_text:
        return None
    try:
        return float(timestamp_text)
    except (TypeError, ValueError):
        return None


def ensure_review_frames(profile: Dict[str, Any], event_number: int, timestamp_s: float) -> List[Dict[str, Any]]:
    video_relpath = profile.get("video_path")
    if not video_relpath:
        return []

    video_path = ROOT / str(video_relpath)
    if not video_path.exists():
        return []

    duration_s = float(profile.get("duration_seconds") or 0.0)
    paths = profile_paths(ROOT, profile["slug"])
    frame_dir = paths["review_frames_dir"] / f"event_{event_number:03d}_{timestamp_s:06.2f}s"
    frame_dir.mkdir(parents=True, exist_ok=True)

    review_frames: List[Dict[str, Any]] = []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    for index, offset_s in enumerate(REVIEW_FRAME_OFFSETS_S):
        frame_time_s = timestamp_s + offset_s
        if duration_s > 0:
            frame_time_s = min(max(0.0, frame_time_s), duration_s)
        else:
            frame_time_s = max(0.0, frame_time_s)

        frame_path = frame_dir / f"frame_{index:02d}_{frame_time_s:06.2f}s.png"
        if not frame_path.exists():
            capture.set(cv2.CAP_PROP_POS_MSEC, frame_time_s * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            cv2.imwrite(str(frame_path), frame)

        review_frames.append(
            {
                "path": str(frame_path.relative_to(ROOT)),
                "timestamp_s": frame_time_s,
                "is_event_moment": abs(offset_s) < 1e-9,
            }
        )

    capture.release()
    return review_frames


def ensure_review_gif(profile: Dict[str, Any], event_number: int, timestamp_s: float, review_frames: List[Dict[str, Any]]) -> str | None:
    if not review_frames:
        return None

    paths = profile_paths(ROOT, profile["slug"])
    gif_path = paths["review_gifs_dir"] / f"event_{event_number:03d}_{timestamp_s:06.2f}s.gif"
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    if gif_path.exists():
        return str(gif_path.relative_to(ROOT))

    images: List[Image.Image] = []
    try:
        for frame in review_frames:
            image = Image.open(ROOT / frame["path"]).convert("RGB")
            image.thumbnail((720, 405))
            images.append(image)

        if not images:
            return None

        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=REVIEW_GIF_DURATION_MS,
            loop=0,
            optimize=False,
        )
    finally:
        for image in images:
            image.close()

    return str(gif_path.relative_to(ROOT))


def load_calibration(slug: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    paths = profile_paths(ROOT, slug)
    if paths["calibration"].exists():
        return read_json(paths["calibration"])
    return {
        "video": profile["video_name"],
        "frame_image": profile["sample_frames"][0] if profile["sample_frames"] else "",
        "landmarks_px": {},
    }


def try_auto_calibration(slug: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    paths = profile_paths(ROOT, slug)
    absolute_frames = [ROOT / relative for relative in profile["sample_frames"]]
    payload = auto_calibrate_frames(
        video_name=profile["video_name"],
        frame_paths=absolute_frames,
        relative_frame_paths=profile["sample_frames"],
        debug_dir=paths["auto_calibration_debug_dir"],
    )
    write_json(paths["calibration"], payload)
    return payload


def read_events(slug: str) -> List[Dict[str, str]]:
    path = profile_paths(ROOT, slug)["events"]
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_events(slug: str, rows: List[Dict[str, str]]) -> None:
    path = profile_paths(ROOT, slug)["events"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "frame_image", "label", "pixel_x", "pixel_y", "result", "play_type", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_event(slug: str, row: Dict[str, str]) -> None:
    rows = read_events(slug)
    rows.append(row)
    write_events(slug, rows)


def read_projected_rows(slug: str) -> List[Dict[str, str]]:
    path = profile_paths(ROOT, slug)["projected_shots"]
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_review_rows(profile: Dict[str, Any], projected_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    frame_width = float(profile.get("width") or 1.0)
    frame_height = float(profile.get("height") or 1.0)
    court_width, court_height = court_image_size(CHART_SCALE, CHART_MARGIN)
    review_rows: List[Dict[str, Any]] = []

    for index, row in enumerate(projected_rows, start=1):
        frame_x = float(row.get("pixel_x") or 0.0)
        frame_y = float(row.get("pixel_y") or 0.0)
        court_x = float(row.get("court_x_ft") or 0.0)
        court_y = float(row.get("court_y_ft") or 0.0)
        court_px_x, court_px_y = feet_to_pixels((court_x, court_y), CHART_SCALE, CHART_MARGIN)
        result = (row.get("result") or "").strip().lower()

        review_row: Dict[str, Any] = dict(row)
        review_row["event_number"] = index
        review_row["has_projection"] = bool(row.get("court_x_ft")) and bool(row.get("court_y_ft"))
        review_row["frame_left_pct"] = (frame_x / frame_width) * 100.0
        review_row["frame_top_pct"] = (frame_y / frame_height) * 100.0
        review_row["court_left_pct"] = (court_px_x / court_width) * 100.0
        review_row["court_top_pct"] = (court_px_y / court_height) * 100.0
        review_row["marker_class"] = "make" if result in {"make", "made", "1", "true", "yes"} else "miss"
        timestamp_s = parse_event_timestamp(row.get("timestamp"))
        if timestamp_s is not None:
            clip_start_s = max(0.0, timestamp_s - REVIEW_CLIP_BEFORE_S)
            clip_end_s = timestamp_s + REVIEW_CLIP_AFTER_S
            duration_s = float(profile.get("duration_seconds") or 0.0)
            if duration_s > 0:
                clip_end_s = min(duration_s, clip_end_s)
            review_row["clip_start_s"] = clip_start_s
            review_row["clip_end_s"] = clip_end_s
            review_row["event_timestamp_s"] = timestamp_s

            try:
                review_frames = ensure_review_frames(profile, index, timestamp_s)
            except RuntimeError:
                review_frames = []
            if review_frames:
                review_row["review_frames"] = review_frames
                review_gif_path = ensure_review_gif(profile, index, timestamp_s, review_frames)
                if review_gif_path:
                    review_row["review_gif_path"] = review_gif_path
        review_rows.append(review_row)

    return review_rows


def run_pipeline(slug: str) -> None:
    paths = profile_paths(ROOT, slug)
    calibration_path = paths["calibration"]
    events_path = paths["events"]
    if not calibration_path.exists():
        raise RuntimeError("Save a calibration first.")
    if not events_path.exists():
        raise RuntimeError("Add at least one event first.")

    commands = [
        [
            sys.executable,
            str(ROOT / "src" / "compute_calibration.py"),
            "--config",
            str(calibration_path),
            "--out",
            str(paths["calibration_matrix"]),
        ],
        [
            sys.executable,
            str(ROOT / "src" / "project_events.py"),
            "--events",
            str(events_path),
            "--calibration",
            str(paths["calibration_matrix"]),
            "--out",
            str(paths["projected_shots"]),
        ],
        [
            sys.executable,
            str(ROOT / "src" / "render_shot_chart.py"),
            "--events",
            str(paths["projected_shots"]),
            "--out",
            str(paths["shot_chart"]),
        ],
    ]

    for command in commands:
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Pipeline step timed out. Recheck the calibration points and try a cleaner frame."
            ) from exc
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "Unknown pipeline error."
            raise RuntimeError(message)


@app.route("/")
def index():
    videos = list_cached_profiles(ROOT)
    if not videos:
        videos = build_video_library(ROOT)
    return render_template("index.html", videos=videos)


@app.post("/rescan")
def rescan():
    build_video_library(ROOT, refresh=True)
    flash("Video library refreshed.")
    return redirect(url_for("index"))


@app.route("/media/<path:relpath>")
def media(relpath: str):
    return send_file(ROOT / relpath)


@app.route("/videos/<slug>")
def video_detail(slug: str):
    profile = refresh_profile_status(ROOT, slug)
    return render_template("video_detail.html", profile=profile)


@app.route("/videos/<slug>/calibration", methods=["GET", "POST"])
def calibration(slug: str):
    profile = refresh_profile_status(ROOT, slug)
    paths = profile_paths(ROOT, slug)

    if request.method == "POST":
        landmarks_json = request.form.get("landmarks_json", "").strip()
        try:
            landmarks = json.loads(landmarks_json) if landmarks_json else {}
        except json.JSONDecodeError as exc:
            flash(f"Invalid landmark JSON: {exc}")
            return redirect(url_for("calibration", slug=slug))

        payload = {
            "video": profile["video_name"],
            "frame_image": request.form.get("frame_image", profile["sample_frames"][0] if profile["sample_frames"] else ""),
            "landmarks_px": landmarks,
        }
        write_json(paths["calibration"], payload)
        refresh_profile_status(ROOT, slug)
        flash("Calibration saved.")
        return redirect(url_for("calibration", slug=slug))

    calibration_payload = load_calibration(slug, profile)
    return render_template(
        "calibration.html",
        profile=profile,
        calibration=calibration_payload,
        landmark_names=list(load_landmark_names()),
    )


@app.post("/videos/<slug>/auto-calibration")
def auto_calibration(slug: str):
    profile = refresh_profile_status(ROOT, slug)
    try:
        try_auto_calibration(slug, profile)
        refresh_profile_status(ROOT, slug)
        flash("Auto calibration saved.")
    except RuntimeError as exc:
        flash(str(exc))
    return redirect(url_for("calibration", slug=slug))


@app.route("/videos/<slug>/candidates")
def candidates(slug: str):
    profile = refresh_profile_status(ROOT, slug)
    paths = profile_paths(ROOT, slug)
    candidate_rows = read_candidate_rows(paths["shot_candidates"])
    calibration_matrix = read_json(paths["calibration_matrix"]) if paths["calibration_matrix"].exists() else None
    return render_template(
        "candidates.html",
        profile=profile,
        candidate_rows=candidate_rows,
        review_rows=build_review_rows(profile, candidate_rows),
        calibration_matrix=calibration_matrix,
        court_reference_path=ensure_clean_court_reference(),
    )


@app.post("/videos/<slug>/candidates/mine")
def mine_candidates(slug: str):
    profile = refresh_profile_status(ROOT, slug)
    paths = profile_paths(ROOT, slug)
    try:
        rows = mine_shot_candidates(
            root=ROOT,
            video_path=ROOT / str(profile["video_path"]),
            duration_s=float(profile.get("duration_seconds") or 0.0),
            out_csv=paths["shot_candidates"],
            frame_dir=paths["candidate_frames_dir"],
            calibration_matrix_path=paths["calibration_matrix"] if paths["calibration_matrix"].exists() else None,
        )
        refresh_profile_status(ROOT, slug)
        flash(f"Mined {len(rows)} shot candidates.")
    except RuntimeError as exc:
        flash(str(exc))
    return redirect(url_for("candidates", slug=slug))


@app.post("/videos/<slug>/candidates/promote")
def promote_candidates(slug: str):
    paths = profile_paths(ROOT, slug)
    candidate_rows = read_candidate_rows(paths["shot_candidates"])
    if not candidate_rows:
        flash("Mine candidates first.")
        return redirect(url_for("candidates", slug=slug))

    write_events(slug, candidate_rows_to_events(candidate_rows))
    refresh_profile_status(ROOT, slug)

    if paths["calibration"].exists():
        try:
            run_pipeline(slug)
            refresh_profile_status(ROOT, slug)
            flash("Candidates replaced the manual events and outputs were rebuilt.")
            return redirect(url_for("results", slug=slug))
        except RuntimeError as exc:
            flash(f"Candidates saved as events, but the outputs build failed: {exc}")
            return redirect(url_for("events", slug=slug))

    flash("Candidates replaced the manual events.")
    return redirect(url_for("events", slug=slug))


@app.route("/videos/<slug>/events", methods=["GET", "POST"])
def events(slug: str):
    profile = refresh_profile_status(ROOT, slug)

    if request.method == "POST":
        row = {
            "timestamp": request.form.get("timestamp", "").strip(),
            "frame_image": request.form.get("frame_image", "").strip(),
            "label": request.form.get("label", "").strip(),
            "pixel_x": request.form.get("pixel_x", "").strip(),
            "pixel_y": request.form.get("pixel_y", "").strip(),
            "result": request.form.get("result", "").strip(),
            "play_type": request.form.get("play_type", "").strip(),
            "notes": request.form.get("notes", "").strip(),
        }
        if not row["pixel_x"] or not row["pixel_y"]:
            flash("Click the frame to set pixel coordinates before saving.")
            return redirect(url_for("events", slug=slug))
        append_event(slug, row)
        refresh_profile_status(ROOT, slug)
        flash("Event saved.")
        return redirect(url_for("events", slug=slug))

    return render_template(
        "events.html",
        profile=profile,
        events=read_events(slug),
        play_types=PLAY_TYPES,
    )


@app.post("/videos/<slug>/events/reset")
def reset_events(slug: str):
    write_events(slug, [])
    refresh_profile_status(ROOT, slug)
    flash("Events cleared.")
    return redirect(url_for("events", slug=slug))


@app.post("/videos/<slug>/build")
def build_outputs(slug: str):
    try:
        run_pipeline(slug)
        refresh_profile_status(ROOT, slug)
        flash("Shot coordinates and chart generated.")
    except RuntimeError as exc:
        flash(str(exc))
    return redirect(url_for("results", slug=slug))


@app.route("/videos/<slug>/results")
def results(slug: str):
    profile = refresh_profile_status(ROOT, slug)
    paths = profile_paths(ROOT, slug)
    calibration_matrix = read_json(paths["calibration_matrix"]) if paths["calibration_matrix"].exists() else None
    projected_rows = read_projected_rows(slug)
    return render_template(
        "results.html",
        profile=profile,
        projected_rows=projected_rows,
        review_rows=build_review_rows(profile, projected_rows),
        calibration_matrix=calibration_matrix,
        court_reference_path=ensure_clean_court_reference(),
    )
