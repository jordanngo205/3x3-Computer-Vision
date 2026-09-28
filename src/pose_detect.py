"""
Detection with skeletons, so detection quality can be judged by eye.

A box only tells you "something is here". A skeleton tells you whether the
model actually understood the person: if the limbs land on the real limbs and
follow them frame to frame, the detection is genuinely locked onto that
player. If a box is really two players merged, the skeleton gives it away
immediately - the joints scatter across both bodies.

Also writes the detections + keypoints to JSON so they can be fed to a
separate association model (CAMELTrack) instead of our own tracking logic.
"""
import argparse
import json

import cv2
import numpy as np
from ultralytics import YOLO

# COCO-17 skeleton: which joints connect to which
EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),        # arms
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (5, 6), (11, 12), (5, 11), (6, 12),      # torso
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
]

DEFAULT_COURT_POLY = np.array([
    (105, 300), (30, 355), (0, 400), (0, 540), (650, 540),
    (760, 430), (830, 330), (830, 260), (350, 248),
], dtype=np.int32)


def detect_court(video, sample_every=10):
    """Find the playing surface automatically, instead of drawing it by hand.

    A hand-drawn polygon is wrong the moment the framing changes, and mine was
    demonstrably inconsistent - it reached past the left sideline (catching
    staff standing off court) while cutting inside the right sideline (dropping
    a referee who was genuinely on court).

    Method: take the median over many frames, which removes the moving players
    and leaves an empty court, then keep the desaturated mid-grey region - the
    playing surface is duller than the white lines and brighter than the dark
    LED boards behind it. The largest such region is the court.
    """
    cap = cv2.VideoCapture(video)
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % sample_every == 0:
            frames.append(f)
        i += 1
    cap.release()
    if not frames:
        return None

    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    surface = ((S < 55) & (V > 78) & (V < 185)).astype(np.uint8)
    surface = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    surface = cv2.morphologyEx(surface, cv2.MORPH_OPEN, np.ones((13, 13), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(surface, 8)
    if n < 2:
        return None
    biggest = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    court = (lab == biggest).astype(np.uint8)

    h = court.shape[0]
    rows, lefts, rights = [], [], []
    for y in range(0, h, 6):
        xs = np.where(court[y] > 0)[0]
        if len(xs) > 60:
            rows.append(y); lefts.append(xs.min()); rights.append(xs.max())
    if len(rows) < 5:
        return None
    pts = [[lefts[i], rows[i]] for i in range(len(rows))] + \
          [[rights[i], rows[i]] for i in range(len(rows))][::-1]
    poly = cv2.approxPolyDP(np.array(pts, dtype=np.int32), 6, True)
    return poly.reshape(-1, 2).astype(np.int32)


def foot_on_court(poly, fx, fy, margin=0.0):
    """margin>0 keeps detections slightly outside the line - players step over
    the sideline constantly, and a hand-drawn boundary is never exact."""
    return cv2.pointPolygonTest(poly, (float(fx), float(fy)), True) >= -margin


def torso_span(kps):
    """Rough size of the person's torso from shoulders/hips, or None."""
    if kps is None:
        return None
    pts = kps[[5, 6, 11, 12], :2]
    pts = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
    if len(pts) < 2:
        return None
    return pts.mean(axis=0)


def is_two_people(box, kps, all_kps):
    """A box holds two players if two different skeletons' torsos sit inside it.

    This replaces judging by box shape, which cannot tell a lunging player
    (wide box, one person) from two players standing together - and was
    deleting real players.
    """
    x1, y1, x2, y2 = box
    inside = 0
    for k in all_kps:
        c = torso_span(k)
        if c is None:
            continue
        if x1 <= c[0] <= x2 and y1 <= c[1] <= y2:
            inside += 1
    return inside >= 2


def draw_person(frame, box, kps, colour, label=None):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    if label:
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    if kps is None:
        return
    for a, b in EDGES:
        xa, ya = kps[a][:2]
        xb, yb = kps[b][:2]
        if xa > 0 and ya > 0 and xb > 0 and yb > 0:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), colour, 2)
    for x, y in kps[:, :2]:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--save-json", default=None)
    p.add_argument("--model", default="yolo11m-pose.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280,
                   help="inference resolution. The default 640 shrinks a 960x540 frame and loses the "
                        "smaller/further players - 3.7 people per frame at 640 vs 8.0 at 1280.")
    p.add_argument("--court-poly", default=None)
    p.add_argument("--no-court-filter", action="store_true")
    p.add_argument("--reject-crowded", action="store_true",
                   help="drop boxes containing two skeletons. Off by default: a neighbour standing "
                        "close puts their torso inside an otherwise valid box, so this deleted real "
                        "players. The pose model already gives one skeleton per person.")
    p.add_argument("--manual-court", action="store_true",
                   help="use the old hand-drawn polygon instead of detecting the court")
    p.add_argument("--court-margin", type=float, default=15.0,
                   help="pixels of slack outside the drawn court line, since players step over it")
    p.add_argument("--iou", type=float, default=0.85,
                   help="NMS overlap threshold. The 0.7 default suppresses a player standing behind "
                        "another as a duplicate; 0.85 keeps them.")
    args = p.parse_args()

    out_path = args.out or args.video.rsplit(".", 1)[0] + "_pose.mp4"
    if args.court_poly:
        nums = [float(v) for v in args.court_poly.split(",")]
        court = np.array(list(zip(nums[0::2], nums[1::2])), dtype=np.int32)
    elif args.manual_court:
        court = DEFAULT_COURT_POLY
    else:
        court = detect_court(args.video)
        if court is None:
            print("court auto-detection failed, falling back to the hand-drawn polygon")
            court = DEFAULT_COURT_POLY
        else:
            print(f"court detected automatically ({len(court)} points)")

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    records = {}
    fi = 0
    kept_total = rejected_court = rejected_shape = 0
    per_frame_counts = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        kps_all = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None

        frame_people = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if not args.no_court_filter and not foot_on_court(court, (x1 + x2) / 2, y2, args.court_margin):
                rejected_court += 1
                continue
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            kps = kps_all[i] if kps_all is not None and i < len(kps_all) else None
            if args.reject_crowded and kps_all is not None and is_two_people(box, kps, kps_all):
                rejected_shape += 1
                continue
            frame_people.append((box.tolist(), kps))

        per_frame_counts.append(len(frame_people))
        kept_total += len(frame_people)

        rng = np.random.RandomState(0)
        for j, (box, kps) in enumerate(frame_people):
            colour = tuple(int(v) for v in np.random.RandomState(j * 41 + 3).randint(70, 255, 3))
            draw_person(frame, box, kps, colour)
        if not args.no_court_filter:
            cv2.polylines(frame, [court], True, (0, 200, 0), 1)
        banner = f"players detected: {len(frame_people)}"
        cv2.rectangle(frame, (12, 12), (12 + 22 * len(banner), 52), (0, 0, 0), -1)
        cv2.putText(frame, banner, (22, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"frame {fi}", (22, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        writer.write(frame)

        if args.save_json:
            records[fi] = [
                {"box": b, "keypoints": (k[:, :3].tolist() if k is not None else None)}
                for b, k in frame_people
            ]
        fi += 1
        if fi % 60 == 0:
            print(f"frame {fi}: {len(frame_people)} people", flush=True)

    cap.release()
    writer.release()
    counts = np.array(per_frame_counts) if per_frame_counts else np.array([0])
    print(f"\ndone. {fi} frames -> {out_path}")
    print(f"people per frame: mean {counts.mean():.1f}, min {counts.min()}, max {counts.max()}")
    print(f"rejected: {rejected_court} off court, {rejected_shape} boxes holding two skeletons")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump({"fps": fps, "width": w, "height": h, "n_frames": fi, "frames": records}, f)
        print(f"detections + keypoints -> {args.save_json}")


if __name__ == "__main__":
    main()
