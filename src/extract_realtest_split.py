"""
The v1 embedding (trained only on video.mp4/Test1.mp4/Test2.mp4) failed to
generalize to real test.mov - it collapsed to near-identical embeddings for
everyone on that clip. Diagnosis: those 3 training clips are a different visual
domain (short clips, different lighting/camera) than the actual target broadcast
clip.

Fix: split real test.mov itself in half by time. First half's confident
detections get added to the training pool (so the model actually sees this
domain). Second half is kept as a true, never-trained-on held-out set for
validation - temporally separated, so it's still a fair test of generalization
within the clip.
"""
import json
import os

from extract_pseudo_labels import load_real_detections, build_positive_pairs, build_negative_pairs

PROJECT_DIR = "/Users/jordanngo/Projects/AI live tracking"
JSON_PATH = os.path.join(PROJECT_DIR, "output_tracks_realtest.json")
CROPS_DIR = os.path.join(PROJECT_DIR, "pseudo_labels", "crops", "realtest_heldout")
ORIGINAL_MANIFEST = os.path.join(PROJECT_DIR, "pseudo_labels", "pairs.json")
COMBINED_MANIFEST = os.path.join(PROJECT_DIR, "pseudo_labels", "pairs_with_realtest_half.json")
EVAL_MANIFEST = os.path.join(PROJECT_DIR, "pseudo_labels", "realtest_second_half_eval_pairs.json")


def crop_path(tid, fidx):
    return os.path.join(CROPS_DIR, f"track{tid}", f"frame{fidx:06d}.jpg")


def build_crop_lookup(by_frame):
    """Crops for real test.mov were already extracted once by
    validate_embedding_on_realtest.py - reuse those files instead of re-decoding
    the video."""
    lookup = {}
    for fidx, entries in by_frame.items():
        for tid, box, team in entries:
            p = crop_path(tid, fidx)
            if os.path.exists(p):
                lookup[(tid, fidx)] = {"path": p, "team": team}
    return lookup


def split_by_frame(by_frame, by_track, split_frame):
    first_by_frame = {f: v for f, v in by_frame.items() if f < split_frame}
    second_by_frame = {f: v for f, v in by_frame.items() if f >= split_frame}
    first_by_track = {}
    second_by_track = {}
    for tid, entries in by_track.items():
        first = [e for e in entries if e[0] < split_frame]
        second = [e for e in entries if e[0] >= split_frame]
        if first:
            first_by_track[tid] = first
        if second:
            second_by_track[tid] = second
    return first_by_frame, first_by_track, second_by_frame, second_by_track


def main():
    with open(JSON_PATH) as f:
        n_frames = json.load(f)["n_frames"]
    split_frame = n_frames // 2
    print(f"n_frames={n_frames}, splitting at frame {split_frame}")

    by_frame, by_track = load_real_detections(JSON_PATH)
    first_frame, first_track, second_frame, second_track = split_by_frame(by_frame, by_track, split_frame)
    print(f"first half: {len(first_track)} tracks, {len(first_frame)} frames with detections")
    print(f"second half: {len(second_track)} tracks, {len(second_frame)} frames with detections")

    first_lookup = build_crop_lookup(first_frame)
    second_lookup = build_crop_lookup(second_frame)
    print(f"first half crops found on disk: {len(first_lookup)}")
    print(f"second half crops found on disk: {len(second_lookup)}")

    train_pos = build_positive_pairs(first_track, first_lookup, "realtest_train_half")
    train_neg = build_negative_pairs(first_frame, first_lookup, "realtest_train_half")
    eval_pos = build_positive_pairs(second_track, second_lookup, "realtest_eval_half")
    eval_neg = build_negative_pairs(second_frame, second_lookup, "realtest_eval_half")

    with open(ORIGINAL_MANIFEST) as f:
        original = json.load(f)

    combined = {
        "positive": original["positive"] + train_pos,
        "negative": original["negative"] + train_neg,
    }
    with open(COMBINED_MANIFEST, "w") as f:
        json.dump(combined, f)
    print(f"combined training manifest: {len(combined['positive'])} positive, {len(combined['negative'])} negative")
    print(f"  -> {COMBINED_MANIFEST}")

    eval_manifest = {"positive": eval_pos, "negative": eval_neg}
    with open(EVAL_MANIFEST, "w") as f:
        json.dump(eval_manifest, f)
    print(f"held-out eval manifest (second half only): {len(eval_pos)} positive, {len(eval_neg)} negative")
    print(f"  -> {EVAL_MANIFEST}")


if __name__ == "__main__":
    main()
