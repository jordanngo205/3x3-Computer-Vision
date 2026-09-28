"""
Post-process a tracking result to remove boxes that wrap TWO players.

These are what actually look like "opponents swapping": a single box covers a
red player and a white player at once, so whichever one dominates the crop
decides the label, and it flips as they move. It is not the tracker losing
identity - the box genuinely contains two people.

Two independent tells, required together so we don't delete real players:

  1. too wide for where it is. Player width scales with position on court
     (further away = smaller), so we fit width ~ f(foot_y) from the clip's own
     boxes and flag the outliers.
  2. two teams inside it. Split the torso window down the middle; if the left
     half reads as one kit and the right half as the other, and both reads are
     confident, there are two players in there.

Runs off a saved tracks JSON, so it needs no re-detection.
"""
import argparse
import json

import cv2
import numpy as np


def half_teams(frame, box):
    """Team read for the left and right halves of the torso window."""
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if w <= 4 or h <= 4:
        return None
    ty1, ty2 = max(0, y1 + int(h * 0.15)), max(1, y1 + int(h * 0.60))
    reads = []
    for a, b in [(0.02, 0.5), (0.5, 0.98)]:
        patch = frame[ty1:ty2, max(0, x1 + int(w * a)):max(1, x1 + int(w * b))]
        if patch.size == 0:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        S, V = hsv[:, :, 1], hsv[:, :, 2]
        reads.append((((S > 90) & (V > 60)).mean(), ((S < 60) & (V > 120)).mean()))
    return reads


def fit_width_model(tracks):
    """Expected player box width as a function of foot position."""
    ws, ys = [], []
    for fb in tracks.values():
        for box in fb.values():
            ws.append(box[2] - box[0])
            ys.append(box[3])
    A = np.vstack([np.array(ys), np.ones(len(ys))]).T
    coef, *_ = np.linalg.lstsq(A, np.array(ws), rcond=None)
    return coef
