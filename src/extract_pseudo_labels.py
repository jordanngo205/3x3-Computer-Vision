"""
Bootstrap training data from our OWN pipeline's confident output (no external
dataset needed). For each already-tracked clip, pull every REAL (non-interpolated)
detection box per track, crop the player from the source frame, then build:

  - positive pairs: same track_id, nearby in time (the pipeline was confident these
    are the same person)
  - negative pairs: different track_ids visible in the same frame (they are
    definitely different people)

These pairs become training data for a small appearance/ReID embedding, meant to
replace the hand-tuned HSV-histogram matching in track_clip7.py.

`real test.mov` (output_tracks_realtest.json) is intentionally NOT included here -
it is held out to check afterward whether the trained model actually helps on the
one clip that matters most.
"""
import json
import os
import random
from collections import defaultdict

import cv2

PROJECT_DIR = "/Users/jordanngo/Projects/AI live tracking"
CROPS_DIR = os.path.join(PROJECT_DIR, "pseudo_labels", "crops")
MANIFEST_PATH = os.path.join(PROJECT_DIR, "pseudo_labels", "pairs.json")

SOURCES = [
    {"tag": "v7", "video": "video.mp4", "json": "output_tracks_v7.json"},
    {"tag": "test1", "video": "Test1.mp4", "json": "output_tracks_test1.json"},
    {"tag": "test2", "video": "Test2.mp4", "json": "output_tracks_test2.json"},
] + [
    {"tag": f"sameGame{i:02d}", "video": f"same_game_clips/clip{i:02d}.mp4", "json": f"output_tracks_sameGame{i:02d}.json"}
    for i in range(1, 12)
]

MAX_POSITIVE_GAP_FRAMES = 45   # ~1.5s at 30fps: still "nearby in time"
MAX_POSITIVES_PER_TRACK = 200  # cap combinatorial blowup on long tracks
MAX_NEGATIVES_PER_FRAME = 4    # cap pairs drawn from one frame's other tracks
RANDOM_SEED = 42


def load_real_detections(track_json_path):
    """Returns frame_idx -> [(track_id, box, team), ...] for non-interpolated boxes only."""
    with open(track_json_path) as f:
        d = json.load(f)

    predicted = d.get("predicted", {})
    track_team = d.get("track_team", {})
    by_frame = defaultdict(list)
    by_track = defaultdict(list)

    for tid, frames in d["tracks"].items():
        pred_set = set(int(x) for x in predicted.get(tid, []))
        team = track_team.get(tid)
        for fkey, box in frames.items():
            fidx = int(fkey)
            if fidx in pred_set:
                continue
            by_frame[fidx].append((tid, box, team))
            by_track[tid].append((fidx, box, team))

    for tid in by_track:
        by_track[tid].sort(key=lambda x: x[0])

    return by_frame, by_track


def extract_crops(tag, video_path, by_frame, out_dir):
    """Single sequential pass over the video, grabbing every needed frame once."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")

    crop_path_by_track_frame = {}
    needed_frames = set(by_frame.keys())
    fidx = 0
    n_saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx in needed_frames:
            h, w = frame.shape[:2]
            for tid, box, team in by_frame[fidx]:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w - 1), min(y2, h - 1)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                track_dir = os.path.join(out_dir, f"track{tid}")
                os.makedirs(track_dir, exist_ok=True)
                crop_path = os.path.join(track_dir, f"frame{fidx:06d}.jpg")
                cv2.imwrite(crop_path, crop)
                crop_path_by_track_frame[(tid, fidx)] = {"path": crop_path, "team": team}
                n_saved += 1
        fidx += 1
    cap.release()
    print(f"  [{tag}] scanned {fidx} frames, saved {n_saved} crops")
    return crop_path_by_track_frame


def build_positive_pairs(by_track, crop_lookup, tag):
    rng = random.Random(RANDOM_SEED)
    pairs = []
    for tid, entries in by_track.items():
        candidates = []
        for i in range(len(entries)):
            fi, _, _ = entries[i]
            for j in range(i + 1, len(entries)):
                fj, _, _ = entries[j]
                gap = fj - fi
                if gap > MAX_POSITIVE_GAP_FRAMES:
                    break
                key_i = (tid, fi)
                key_j = (tid, fj)
                if key_i in crop_lookup and key_j in crop_lookup:
                    candidates.append((crop_lookup[key_i]["path"], crop_lookup[key_j]["path"]))
        if len(candidates) > MAX_POSITIVES_PER_TRACK:
            candidates = rng.sample(candidates, MAX_POSITIVES_PER_TRACK)
        pairs.extend(candidates)
    print(f"  [{tag}] positive pairs: {len(pairs)}")
    return pairs


def build_negative_pairs(by_frame, crop_lookup, tag):
    rng = random.Random(RANDOM_SEED)
    pairs = []
    for fidx, entries in by_frame.items():
        tids_here = [tid for tid, _, _ in entries]
        if len(tids_here) < 2:
            continue
        possible = []
        for i in range(len(tids_here)):
            for j in range(i + 1, len(tids_here)):
                key_i = (tids_here[i], fidx)
                key_j = (tids_here[j], fidx)
                if key_i in crop_lookup and key_j in crop_lookup:
                    possible.append((crop_lookup[key_i]["path"], crop_lookup[key_j]["path"]))
        if len(possible) > MAX_NEGATIVES_PER_FRAME:
            possible = rng.sample(possible, MAX_NEGATIVES_PER_FRAME)
        pairs.extend(possible)
    print(f"  [{tag}] negative pairs: {len(pairs)}")
    return pairs


def main():
    all_positive = []
    all_negative = []

    for src in SOURCES:
        tag = src["tag"]
        video_path = os.path.join(PROJECT_DIR, src["video"])
        json_path = os.path.join(PROJECT_DIR, src["json"])
        print(f"processing {tag} ({src['video']})...")

        by_frame, by_track = load_real_detections(json_path)
        out_dir = os.path.join(CROPS_DIR, tag)
        crop_lookup = extract_crops(tag, video_path, by_frame, out_dir)

        pos = build_positive_pairs(by_track, crop_lookup, tag)
        neg = build_negative_pairs(by_frame, crop_lookup, tag)
        all_positive.extend(pos)
        all_negative.extend(neg)

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump({"positive": all_positive, "negative": all_negative}, f)

    print()
    print(f"TOTAL positive pairs: {len(all_positive)}")
    print(f"TOTAL negative pairs: {len(all_negative)}")
    print(f"manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
