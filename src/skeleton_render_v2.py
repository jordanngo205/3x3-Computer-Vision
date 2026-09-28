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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--model", default="yolo11m-pose.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.85)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--no-tiles", action="store_true")
    p.add_argument("--fill-gaps", action="store_true",
                   help="reconstruct skeletons across short dropouts. Off by default: a wrong fill "
                        "slides a body across the court, which reads as broken, whereas a missing "
                        "skeleton just reads as 'not detected'.")
    p.add_argument("--max-gap-seconds", type=float, default=0.4,
                   help="how long a player may vanish and still be reconstructed. Specified in "
                        "seconds, not frames: 12 frames is 0.4s at 30fps but only 0.21s at 57fps, "
                        "and a screen lasts about a second.")
    p.add_argument("--keep-off-court", action="store_true")
    p.add_argument("--overlay", action="store_true", help="draw on the video instead of black")
    args = p.parse_args()

    out_path = args.out or args.video.rsplit(".", 1)[0] + "_skel2.mp4"
    court = None if args.keep_off_court else detect_court(args.video)
    model = YOLO(args.model)

    # pass 1: detect, and link bodies frame to frame by overlap (continuity only)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))
    max_gap = max(1, int(round(args.max_gap_seconds * fps)))
    print(f"{fps:.0f} fps -> reconstructing gaps up to {max_gap} frames ({args.max_gap_seconds}s)")
    tracks = {}          # id -> {"box":..., "missed":int}
    poses = defaultdict(dict)   # id -> {frame: keypoints}
    next_id, fi, raw_total = 0, 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        people = (tiled_pose(model, frame, args.imgsz, args.conf, args.iou)
                  if not args.no_tiles else
                  [(b, k) for b, k in zip(
                      model(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0].boxes.xyxy.cpu().numpy(),
                      model(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0].keypoints.data.cpu().numpy())])
        if court is not None:
            people = [(b, k) for b, k in people
                      if foot_on_court(court, (b[0]+b[2])/2, b[3], 15)]
        raw_total += len(people)

        used = set()
        for tid in list(tracks):
            best, best_iou = None, 0.25
            for j, (b, _) in enumerate(people):
                if j in used:
                    continue
                v = box_iou(tracks[tid]["box"], b)
                if v > best_iou:
                    best, best_iou = j, v
            if best is not None:
                b, k = people[best]
                tracks[tid] = {"box": b, "missed": 0}
                poses[tid][fi] = k
                used.add(best)
            else:
                tracks[tid]["missed"] += 1
        tracks = {t: v for t, v in tracks.items() if v["missed"] <= max_gap}
        for j, (b, k) in enumerate(people):
            if j not in used:
                tracks[next_id] = {"box": b, "missed": 0}
                poses[next_id][fi] = k
                next_id += 1
        fi += 1
        if fi % 60 == 0:
            print(f"frame {fi}: {len(people)} skeletons", flush=True)
    cap.release()
    n_frames = fi

    # fill short gaps by interpolating the joints
    filled = defaultdict(dict)
    n_filled = 0
    for tid, byframe in (poses.items() if args.fill_gaps else []):
        fs = sorted(byframe)
        for a, b in zip(fs, fs[1:]):
            gap = b - a
            if 1 < gap <= max_gap:
                ka, kb = byframe[a], byframe[b]
                # refuse fills that would slide a body across the court: either
                # the link was wrong, or the player went too far to interpolate
                ca = ka[[5, 6, 11, 12], :2]; ca = ca[(ca[:, 0] > 0) & (ca[:, 1] > 0)]
                cb = kb[[5, 6, 11, 12], :2]; cb = cb[(cb[:, 0] > 0) & (cb[:, 1] > 0)]
                if len(ca) and len(cb):
                    hgt = max(1.0, np.ptp(ka[:, 1][ka[:, 1] > 0]) if (ka[:, 1] > 0).any() else 1.0)
                    if np.linalg.norm(ca.mean(0) - cb.mean(0)) > 0.9 * hgt:
                        continue
                for g in range(1, gap):
                    t = g / gap
                    k = ka * (1 - t) + kb * t
                    k[(ka[:, 0] <= 0) | (kb[:, 0] <= 0)] = 0
                    filled[tid][a + g] = k
                    n_filled += 1

    by_frame = defaultdict(list)
    for tid, byframe in poses.items():
        for f, k in byframe.items():
            by_frame[f].append((k, False))
    for tid, byframe in filled.items():
        for f, k in byframe.items():
            by_frame[f].append((k, True))

    # pass 2: render
    cap = cv2.VideoCapture(args.video)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = frame if args.overlay else np.zeros_like(frame)
        if court is not None and not args.overlay:
            cv2.polylines(canvas, [court.astype(np.int32)], True, COURT, 1, cv2.LINE_AA)
        for k, was_filled in by_frame.get(fi, []):
            col = FILLED if was_filled else SKELETON
            for a, b in EDGES:
                xa, ya = k[a][:2]; xb, yb = k[b][:2]
                if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                    cv2.line(canvas, (int(xa), int(ya)), (int(xb), int(yb)), col, 2, cv2.LINE_AA)
            for x, y in k[:, :2]:
                if x > 0 and y > 0:
                    cv2.circle(canvas, (int(x), int(y)), 2, col, -1, cv2.LINE_AA)
        writer.write(canvas)
        fi += 1
    cap.release()
    writer.release()

    shown = sum(len(v) for v in by_frame.values())
    print(f"\ndone. {n_frames} frames -> {out_path}")
    print(f"detected {raw_total/max(1,n_frames):.1f}/frame, "
          f"drawn {shown/max(1,n_frames):.1f}/frame ({n_filled} interpolated)")


if __name__ == "__main__":
    main()
