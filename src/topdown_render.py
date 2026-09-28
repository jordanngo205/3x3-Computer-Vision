"""
Broadcast view beside a top-down court, with every player placed in real metres.

Positions come from the ankles rather than the bottom of the box: the ankles are
where the player actually touches the floor, and the box bottom drifts with
pose and motion blur. Each ankle point goes through the calibrated homography
into court centimetres, so distances on the right-hand panel are real.

Team is decided by torso colour, voted per detection - kit saturation separates
white from a coloured kit far better than hue, which is useless here because
the court grey and a red kit sit at the same hue.
"""
import argparse
import json

import cv2
import numpy as np
from ultralytics import YOLO

from pose_detect import EDGES

COURT_W, COURT_H = 1500.0, 1100.0
PAD, SCALE = 40, 0.42          # top-down canvas: cm -> px


def court_canvas(h_px):
    w = int(COURT_W * SCALE) + PAD * 2
    h = int(COURT_H * SCALE) + PAD * 2
    img = np.full((h, w, 3), 24, np.uint8)
    def P(x, y):
        return (int(PAD + x * SCALE), int(PAD + y * SCALE))
    line = (90, 100, 95)
    cv2.rectangle(img, P(0, 0), P(COURT_W, COURT_H), line, 2)
    cv2.rectangle(img, P(505, 0), P(995, 580), line, 1)
    # the arc has to run all the way down to the baseline: the basket sits
    # 157 cm in front of it, so the sweep extends past 0 and 180 degrees
    a0 = float(np.arcsin(-157.0 / 675.0))
    arc = [P(750 + 675 * np.cos(a), 157 + 675 * np.sin(a))
           for a in np.linspace(a0, np.pi - a0, 140)]
    cv2.polylines(img, [np.array(arc, np.int32)], False, line, 1)
    cv2.circle(img, P(750, 157), 5, (70, 80, 75), -1)
    if h != h_px:                      # match the video panel height
        img = cv2.resize(img, (int(w * h_px / h), h_px))
    return img, (w, h)


def ankle_point(k, box, min_conf=0.5):
    """Ground contact point, or None if we cannot trust it.

    Keypoint confidence matters here: in a pile the pose model happily returns
    ankles that belong to the neighbouring player, and an unchecked low-score
    ankle drags the player metres across the court. When the ankles are not
    trustworthy we fall back to the bottom centre of the box, which is cruder
    but cannot be stolen from someone else.
    """
    a = k[[15, 16]]
    good = a[a[:, 2] >= min_conf]
    if len(good) == 2:
        return good[:, :2].mean(axis=0)
    if len(good) == 1:
        return good[0, :2]
    return np.array([(box[0] + box[2]) / 2.0, box[3]], dtype=np.float32)


def torso_feature(frame, box):
    """How red versus how white the shirt is, as a single signed score.

    A tight window on the centre of the shirt beats a wide torso patch here:
    Germany play in red trimmed with white and Canada in white trimmed with red,
    so a generous crop catches both kits' trim plus skin and court. Measured on
    244 samples, this window separates 85% of detections cleanly, where a colour
    histogram clustered over the whole torso did not.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if w < 10 or h < 25:
        return None
    p = frame[max(0, y1 + int(h * .26)):max(1, y1 + int(h * .46)),
              max(0, x1 + int(w * .34)):max(1, x1 + int(w * .66))]
    if p.size < 30:
        return None
    hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = (((Hh <= 10) | (Hh >= 170)) & (S > 80) & (V > 50)).mean()
    white = ((S < 50) & (V > 140)).mean()
    return float(red - white)


def team_of(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if w <= 2 or h <= 2:
        return "?"
    p = frame[max(0, y1 + int(h * .20)):max(1, y1 + int(h * .55)),
              max(0, x1 + int(w * .25)):max(1, x1 + int(w * .75))]
    if p.size == 0:
        return "?"
    hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    col = ((S > 90) & (V > 60)).mean()
    wht = ((S < 60) & (V > 120)).mean()
    if max(col, wht) < 0.08:
        return "?"
    return "C" if col > wht else "W"


def team_of_feat(score, split=0.0, band=0.15):
    """Coloured kit or white, from the torso score. The threshold is fixed here;
    it later had to be learned per clip, because lighting moves every score."""
    if score is None:
        return "?"
    return "C" if score > split + band else ("W" if score < split - band else "?")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--calib", default="court_calibration.json")
    p.add_argument("--out", default="topdown.mp4")
    p.add_argument("--model", default="yolo11m-pose.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--save-positions", default=None)
    args = p.parse_args()

    H = np.array(json.load(open(args.calib))["H_image_to_court"])
    model = YOLO(args.model)

    def to_court(pt):
        return cv2.perspectiveTransform(np.array([[pt]], dtype=np.float32), H).reshape(2)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vw, vh = int(cap.get(3)), int(cap.get(4))
    panel, _ = court_canvas(vh)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (vw + panel.shape[1], vh))

    COLS = {"C": (60, 60, 235), "W": (235, 235, 235), "?": (120, 160, 120)}
    positions, fi, on_court_total = {}, 0, 0

    detections = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model(frame, conf=args.conf, iou=0.85, imgsz=args.imgsz, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        kps = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None

        here = []
        for i, box in enumerate(boxes):
            if kps is None or i >= len(kps):
                continue
            pt = ankle_point(kps[i], box)
            if pt is None:
                continue
            cx, cy = to_court(pt)
            here.append({"box": box, "kp": kps[i], "pt": pt, "cx": float(cx), "cy": float(cy),
                         "feat": torso_feature(frame, box)})

        detections[fi] = here
        on_court_total += len(here)
        fi += 1
        if fi % 60 == 0:
            print(f"frame {fi}: {len(here)} on court", flush=True)
    cap.release()

    for f, ds in detections.items():
        for d in ds:
            d["team"] = team_of_feat(d.get("feat"))

    n_c = sum(1 for ds in detections.values() for d in ds if d.get("team") == "C")
    n_w = sum(1 for ds in detections.values() for d in ds if d.get("team") == "W")
    print(f"kit scoring: {n_c} coloured / {n_w} white / "
          f"{sum(len(ds) for ds in detections.values()) - n_c - n_w} unknown")

    for f, ds in detections.items():
        for d in ds:
            d.setdefault("team", "?")

    cap = cv2.VideoCapture(args.video)
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        top = panel.copy()
        left = frame
        frame_pts = []
        for d in detections.get(fi, []):
            t = d.get("team", "?")
            col = COLS[t]
            frame_pts.append({"x": d["cx"], "y": d["cy"], "team": t})
            k = d["kp"]
            for a, b in EDGES:
                xa, ya = k[a][:2]; xb, yb = k[b][:2]
                if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                    cv2.line(left, (int(xa), int(ya)), (int(xb), int(yb)), col, 2, cv2.LINE_AA)
            cv2.circle(left, (int(d["pt"][0]), int(d["pt"][1])), 5, col, -1)
            sx = int((PAD + d["cx"] * SCALE) * top.shape[1] / (COURT_W * SCALE + PAD * 2))
            sy = int((PAD + d["cy"] * SCALE) * top.shape[0] / (COURT_H * SCALE + PAD * 2))
            cv2.circle(top, (sx, sy), 9, col, -1)
            cv2.circle(top, (sx, sy), 9, (20, 20, 20), 1)
        positions[fi] = frame_pts
        cv2.putText(left, f"on court: {len(frame_pts)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(np.hstack([left, top]))
        fi += 1
    cap.release()
    writer.release()
    print(f"done. {fi} frames -> {args.out}   {on_court_total/max(1,fi):.1f} players/frame")
    if args.save_positions:
        json.dump({"fps": fps, "court_cm": [COURT_W, COURT_H], "frames": positions},
                  open(args.save_positions, "w"))
        print(f"positions (cm) -> {args.save_positions}")


if __name__ == "__main__":
    main()
