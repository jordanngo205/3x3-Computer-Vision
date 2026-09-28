"""
Minimal person detect + track, built up in deliberate steps:
  1. YOLO finds people
  2. drop anyone not standing inside the court (refs, bench, crowd, staff)
  3. match the rest to existing tracks by box overlap (IoU) - cheap and
     reliable when someone is continuously visible
  4. NEW: for anyone left unmatched after step 3 (occluded briefly, moved
     fast, camera jerked - the exact cases that caused ID switching before),
     fall back to the trained appearance model: does this look like the same
     person even though the box moved too far to overlap?

A track that stays unmatched for too many frames in a row (person actually
left the frame) is dropped.
"""
import argparse
import os
import pickle

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from train_embedding import EmbeddingNet, IMG_H, IMG_W

IOU_MATCH_THRESHOLD = 0.3       # below this, box overlap alone doesn't count as a match
APPEARANCE_MATCH_THRESHOLD = 0.6  # below this embedding distance, treat as the same person
                                    # (picked from validate_embedding_on_realtest.py's held-out
                                    # best-threshold search, which found ~0.56 on real footage)
APPEARANCE_RECONNECT_MAX = 0.95 # looser appearance bar, only allowed when the detection is
                                    # also in a plausible position for that track (see stage 2)
MAX_HOLD_SECONDS = 1.0          # keep an unmatched track alive this long before dropping it;
                                    # converted to frames from the video's real fps at runtime
                                    # (10 hard-coded frames was only 0.17s on 57fps footage - tracks
                                    # died before the appearance rescue could ever reconnect them)


def load_embedding_model(checkpoint_path):
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = EmbeddingNet(embed_dim=ckpt["embed_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def compute_embedding(model, device, frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1] - 1), min(y2, frame.shape[0] - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = cv2.resize(frame[y1:y2, x1:x2], (IMG_W, IMG_H))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    crop = (crop - 0.5) / 0.5
    tensor = torch.from_numpy(np.transpose(crop, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tensor).cpu().numpy()[0]


def appearance_distance(gallery, emb):
    """Distance from a detection's look to a track's gallery of recent clean
    looks - min over the gallery, so matching works even when the player's
    pose/angle changed since most snapshots were taken."""
    if not gallery or emb is None:
        return float("inf")
    return min(float(np.linalg.norm(g - emb)) for g in gallery)


GALLERY_SIZE = 10       # clean appearance snapshots kept per track
COAST_DECAY = 0.92      # per-frame damping of a hidden player's predicted motion
COAST_MAX_SPEED = 30.0  # px/frame cap so a bad velocity estimate can't run away

# Default court boundary for 960x540 clips (same framing as video.mp4/the
# same_game_clips). Override with --court-poly for a different camera framing.
DEFAULT_COURT_POLY = np.array([
    (105, 300), (30, 355), (0, 400), (0, 540), (650, 540),
    (760, 430), (830, 330), (830, 260), (350, 248),
], dtype=np.int32)


def foot_on_court(court_poly, fx, fy):
    return cv2.pointPolygonTest(court_poly, (float(fx), float(fy)), False) >= 0


def containment(outer, inner):
    """Fraction of `inner`'s area that sits inside `outer`."""
    x1 = max(outer[0], inner[0]); y1 = max(outer[1], inner[1])
    x2 = min(outer[2], inner[2]); y2 = min(outer[3], inner[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_inner = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area_inner
