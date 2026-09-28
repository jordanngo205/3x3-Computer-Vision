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
