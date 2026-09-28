"""
Skeleton render with tiled pose and temporal gap-filling.

Two additions over the first version, both aimed at skeletons that vanish for a
few frames and pop back:

1. Tiled inference. The wide broadcast makes far-side players small, and a
   single whole-frame pass misses them. Running pose on overlapping vertical
   strips is effectively zooming in on each part of the court, then the results
   are merged. (Same idea as pose_hq in the reference project, which took their
   density from ~2.2 skeletons/frame to a ~70% median.)

2. Gap filling. Bodies are linked across frames by simple overlap - continuity
   only, no attempt at identity - and a body that disappears for a few frames
   has its skeleton interpolated rather than dropped. This is the easy half of
   tracking: holding a person for a second, not naming them for a game.
"""
import argparse
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

from pose_detect import EDGES, detect_court, foot_on_court

SKELETON = (60, 240, 60)
FILLED = (40, 190, 120)   # interpolated frames, slightly different shade
BALL = (40, 150, 245)
COURT = (70, 78, 70)


def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def tiled_pose(model, frame, imgsz, conf, iou, tiles=3, overlap=0.2):
    """Run pose on the whole frame plus overlapping vertical strips, merge."""
    h, w = frame.shape[:2]
    found = []

    def collect(res, ox=0):
        if res.boxes is None or res.keypoints is None:
            return
        for b, k in zip(res.boxes.xyxy.cpu().numpy(), res.keypoints.data.cpu().numpy()):
            b = b.copy(); k = k.copy()
            b[0] += ox; b[2] += ox
            k[:, 0] = np.where(k[:, 0] > 0, k[:, 0] + ox, k[:, 0])
            found.append((b, k))

    collect(model(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0])

    step = int(w / (tiles - (tiles - 1) * overlap))
    stride = int(step * (1 - overlap))
    for i in range(tiles):
        x0 = min(i * stride, max(0, w - step))
        x1 = min(x0 + step, w)
        if x1 - x0 < 40:
            continue
        collect(model(frame[:, x0:x1], conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0], ox=x0)

    # merge duplicates from overlapping tiles. Box overlap alone is not enough:
    # the same player seen in two tiles gets slightly different boxes that can
    # fall under any IoU bar, and both skeletons then get drawn on one body.
    def torso_centre(k):
        pts = k[[5, 6, 11, 12], :2]
        pts = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
        return pts.mean(axis=0) if len(pts) else None

    found.sort(key=lambda bk: -((bk[0][2]-bk[0][0]) * (bk[0][3]-bk[0][1])))
    kept = []
    for b, k in found:
        c = torso_centre(k)
        h = max(1.0, b[3] - b[1])
        dup = False
        for kb, kk in kept:
            if box_iou(b, kb) >= 0.4:
                dup = True
                break
            ck = torso_centre(kk)
            if c is not None and ck is not None and np.linalg.norm(c - ck) < 0.35 * h:
                dup = True   # same body, offset boxes from two tiles
                break
        if not dup:
            kept.append((b, k))
    return kept
