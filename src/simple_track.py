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


def reject_two_person_boxes(dets, min_aspect, contain_frac, contain_area_ratio):
    """Drop detections that are actually TWO players wrapped in one box.

    During contact the detector regularly emits a single box around a pair -
    tracking those as if they were people inflates the identity count, steals
    matches from the real players, and feeds two-player crops to the
    appearance model. Two independent tells:

      1. shape - an upright player's box is much taller than it is wide; a box
         around two players standing side by side is far squarer.
      2. containment - a box that entirely swallows two other detections is a
         wrapper around them, not a person of its own.
    """
    kept, dropped_shape, dropped_wrapper = [], 0, 0
    for i, a in enumerate(dets):
        aw, ah = a[2] - a[0], a[3] - a[1]
        if aw <= 0 or ah <= 0:
            continue
        if ah / aw < min_aspect:
            dropped_shape += 1
            continue
        area_a = aw * ah
        swallowed = 0
        for j, b in enumerate(dets):
            if i == j:
                continue
            area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
            if containment(a, b) >= contain_frac and area_a >= contain_area_ratio * area_b:
                swallowed += 1
        if swallowed >= 2:
            dropped_wrapper += 1
            continue
        kept.append(a)
    return kept, dropped_shape, dropped_wrapper


def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def merge_tracklets(history, galleries, max_overlap_frames, appearance_max, gap_reach_per_frame):
    """Offline pass: stitch fragments of the same player back together.

    Adapted from the ID Switch Detection & Merging step in TrackID3x3
    (arXiv:2503.18282), whose key insight is a hard physical constraint:
    two tracklets that are on court AT THE SAME TIME cannot be the same
    person. A player who picks up a fresh ID coming out of a screen never
    coexists with their old ID, so the pair is free to merge.

    We add two checks the paper's version doesn't need (it merges only into
    tracks present from frame 0; we merge any pair, so we must be stricter):
    the two fragments have to look alike, and the jump between where one
    ended and the other began has to be physically plausible for the gap.

    Returns {track_id: merged_id}.
    """
    ids = sorted(history)
    spans = {i: (min(history[i]), max(history[i])) for i in ids}
    frames = {i: set(history[i]) for i in ids}
    parent = {i: i for i in ids}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # try the closest-in-time pairs first: a fragment is most likely to belong
    # to whoever just disappeared, not to someone who vanished a minute ago
    candidates = []
    for a in ids:
        for b in ids:
            if a == b or spans[a][1] >= spans[b][0]:
                continue  # b must start after a ends
            gap = spans[b][0] - spans[a][1]
            candidates.append((gap, a, b))
    candidates.sort()

    merged = 0
    for gap, a, b in candidates:
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        # the physical constraint: never merge two tracklets that were on
        # court simultaneously - collapse the merged groups, not just the pair
        group_a = [i for i in ids if root(i) == ra]
        group_b = [i for i in ids if root(i) == rb]
        overlap = 0
        for i in group_a:
            for j in group_b:
                overlap += len(frames[i] & frames[j])
        if overlap > max_overlap_frames:
            continue

        ga, gb = galleries.get(a), galleries.get(b)
        if not ga or not gb:
            continue
        look_alike = min(float(np.linalg.norm(x - y)) for x in ga for y in gb)
        if look_alike > appearance_max:
            continue

        last_box = history[a][spans[a][1]]
        first_box = history[b][spans[b][0]]
        lcx, lcy = (last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2
        fcx, fcy = (first_box[0] + first_box[2]) / 2, (first_box[1] + first_box[3]) / 2
        jump = ((fcx - lcx) ** 2 + (fcy - lcy) ** 2) ** 0.5
        height = max(1.0, last_box[3] - last_box[1])
        if jump > height * gap_reach_per_frame * max(1, gap):
            continue  # they'd have had to teleport

        parent[rb] = ra
        merged += 1

    mapping = {i: root(i) for i in ids}
    return mapping, merged
