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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out-tracks", required=True)
    p.add_argument("--width-ratio", type=float, default=1.5,
                   help="flag boxes this many times wider than expected for their court position")
    p.add_argument("--min-team-conf", type=float, default=0.18,
                   help="how strongly each half must read as a kit before we trust the disagreement")
    args = p.parse_args()

    data = json.load(open(args.tracks))
    tracks = data["tracks"]
    coef = fit_width_model(tracks)
    print(f"width model: w = {coef[0]:.4f}*foot_y + {coef[1]:.1f}")

    by_frame = {}
    for tid, fb in tracks.items():
        for f, box in fb.items():
            by_frame.setdefault(int(f), []).append((tid, box))

    cap = cv2.VideoCapture(args.video)
    removed, kept, fi = 0, 0, 0
    drop = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, box in by_frame.get(fi, []):
            expected = coef[0] * box[3] + coef[1]
            too_wide = (box[2] - box[0]) > args.width_ratio * max(1.0, expected)
            if not too_wide:
                kept += 1
                continue
            reads = half_teams(frame, box)
            if not reads:
                kept += 1
                continue
            (lc, lw), (rc, rw) = reads
            left = "C" if lc > lw else "W"
            right = "C" if rc > rw else "W"
            confident = min(max(lc, lw), max(rc, rw)) > args.min_team_conf
            if left != right and confident:
                drop.setdefault(tid, set()).add(str(fi))
                removed += 1
            else:
                kept += 1
        fi += 1
    cap.release()

    for tid, frames in drop.items():
        for f in frames:
            tracks[tid].pop(f, None)
    tracks = {t: fb for t, fb in tracks.items() if fb}
    data["tracks"] = tracks
    data["teams"] = {t: v for t, v in data["teams"].items() if t in tracks}
    json.dump(data, open(args.out_tracks, "w"))
    print(f"removed {removed} two-player boxes, kept {kept} "
          f"({100 * removed / max(1, removed + kept):.1f}% removed)")
    print(f"wrote {args.out_tracks}")


if __name__ == "__main__":
    main()
