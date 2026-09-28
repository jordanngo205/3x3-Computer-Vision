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
from track_botsort import detect_shots

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


def looks_like_referee(frame, box, cx, cy):
    """Officials wear a dark, unsaturated kit and work near the lines.

    None of these signals is enough alone - measured on hand labels, kit colour
    alone also caught 12% of players, and the boundary rule alone caught only
    23% of referees over a short clip (it is a whole-game statistic in the
    reference project). Together they separated 16/16 referees from 146 players
    with no player lost.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if w < 10 or h < 25:
        return False
    p = frame[max(0, y1 + int(h * .26)):max(1, y1 + int(h * .46)),
              max(0, x1 + int(w * .34)):max(1, x1 + int(w * .66))]
    if p.size < 30:
        return False
    hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
    sat = float(np.median(hsv[:, :, 1]))
    val = float(np.median(hsv[:, :, 2]))
    edge = min(abs(cx), abs(cx - COURT_W), abs(cy), abs(cy - COURT_H))
    return sat < 50 and val < 160 and edge < 220


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--calib", default="court_calibration.json")
    p.add_argument("--out", default="topdown.mp4")
    p.add_argument("--model", default="yolo11m-pose.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--save-positions", default=None)
    p.add_argument("--smooth-seconds", type=float, default=0.25,
                   help="window for smoothing court positions along a track")
    p.add_argument("--fill-seconds", type=float, default=0.25,
                   help="interpolate a player across dropouts up to this long, so brief detection "
                        "misses do not read as the dot flickering")
    p.add_argument("--no-track-vote", action="store_true",
                   help="keep the per-frame kit call instead of settling each track on one kit")
    p.add_argument("--all-cameras", action="store_true",
                   help="process every frame, including shots from other cameras the calibration "
                        "does not apply to")
    p.add_argument("--device", default="mps",
                   help="torch device. Ultralytics was silently choosing CPU on this machine, "
                        "which is 2.5x slower than MPS for the same result.")
    p.add_argument("--tracker", default="botsort.yaml",
                   help="ultralytics tracker config. BoT-SORT uses motion prediction and global "
                        "motion compensation for the panning camera, so a player keeps one id "
                        "through a possession instead of the frame-to-frame box overlap we used "
                        "before - which broke whenever a player was briefly occluded.")
    p.add_argument("--greedy-link", action="store_true",
                   help="use the old box-overlap linker instead of BoT-SORT")
    p.add_argument("--keep-referees", action="store_true",
                   help="keep officials in the output instead of filtering them out")
    p.add_argument("--max-depth", type=float, default=1000.0,
                   help="drop detections deeper than this many cm from the baseline. From 180 hand "
                        "labels: every bench/substitute detection sat beyond 1000 cm, and only one "
                        "player in 129 ever did, so this removes the bench without losing players.")
    p.add_argument("--bounds-margin", type=float, default=60.0,
                   help="centimetres of slack outside the court lines; players step over them")
    args = p.parse_args()

    H = np.array(json.load(open(args.calib))["H_image_to_court"])
    model = YOLO(args.model)

    # The calibration belongs to one camera. Broadcast clips cut to zoomed and
    # low-angle cameras, where the same homography points at meaningless pixels
    # and every position is wrong. Keep only the shots that match the framing we
    # calibrated on.
    wide_frames = None
    if not args.all_cameras:
        shots, wide = detect_shots(args.video)
        wide_frames = set()
        for a, b in wide:
            wide_frames.update(range(a, b))
        skipped = sum(b - a for a, b in shots) - len(wide_frames)
        print(f"camera shots: {len(shots)} total, {len(wide)} match the calibrated view "
              f"({skipped} frames from other cameras will be skipped)")

    def to_court(pt):
        return cv2.perspectiveTransform(np.array([[pt]], dtype=np.float32), H).reshape(2)

    def box_iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        i = (x2 - x1) * (y2 - y1)
        return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vw, vh = int(cap.get(3)), int(cap.get(4))
    panel, _ = court_canvas(vh)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (vw + panel.shape[1], vh))

    COLS = {"C": (60, 60, 235), "W": (235, 235, 235), "?": (120, 160, 120)}
    positions, fi, on_court_total = {}, 0, 0

    # link detections frame to frame so the kit colour can be voted over a
    # player's whole run - a single frame's torso patch is far too noisy, which
    # is why the labels were flickering between red and white
    tracks, votes, next_id = {}, {}, 0
    n_refs = [0]
    detections = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if wide_frames is not None and fi not in wide_frames:
            detections[fi] = []
            fi += 1
            continue
        if args.greedy_link:
            res = model(frame, conf=args.conf, iou=0.85, imgsz=args.imgsz,
                        device=args.device, verbose=False)[0]
        else:
            res = model.track(frame, conf=args.conf, iou=0.85, imgsz=args.imgsz,
                              device=args.device, tracker=args.tracker, persist=True,
                              verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        kps = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None
        # BoT-SORT gives an id per detection; it is None before the tracker has
        # settled, and for detections it chose not to track
        ids = (res.boxes.id.cpu().numpy().astype(int)
               if (res.boxes is not None and res.boxes.id is not None) else None)

        here = []
        for i, box in enumerate(boxes):
            if kps is None or i >= len(kps):
                continue
            pt = ankle_point(kps[i], box)
            if pt is None:
                continue
            cx, cy = to_court(pt)
            # in/out of bounds is now decided in court metres, not by a pixel
            # polygon: a substitute sitting past the sideline is genuinely off
            # the court, whatever the floor looks like there
            m = args.bounds_margin
            if not (-m <= cx <= COURT_W + m and -m <= cy <= args.max_depth):
                continue
            if not args.keep_referees and looks_like_referee(frame, box, cx, cy):
                n_refs[0] += 1
                continue
            d = {"box": box, "kp": kps[i], "pt": pt, "cx": float(cx), "cy": float(cy),
                 "feat": torso_feature(frame, box)}
            if ids is not None and i < len(ids):
                d["tid"] = int(ids[i])
            here.append(d)

        # link to existing tracks by overlap, purely to accumulate colour votes.
        # Only needed when BoT-SORT is off - it already supplies ids.
        used = set()
        for tid in (list(tracks) if args.greedy_link else []):
            best, bi = 0.25, None
            for j, d in enumerate(here):
                if j in used:
                    continue
                v = box_iou(tracks[tid]["box"], d["box"])
                if v > best:
                    best, bi = v, j
            if bi is None:
                tracks[tid]["missed"] += 1
                if tracks[tid]["missed"] > 20:
                    tracks.pop(tid)
            else:
                tracks[tid] = {"box": here[bi]["box"], "missed": 0}
                here[bi]["tid"] = tid
                used.add(bi)
        # anything still without an id (BoT-SORT declined to track it, or the
        # greedy linker found no overlap) starts its own track. Negative ids
        # keep these clear of BoT-SORT's own numbering.
        for j, d in enumerate(here):
            if "tid" not in d:
                if args.greedy_link:
                    tracks[next_id] = {"box": d["box"], "missed": 0}
                    d["tid"] = next_id
                else:
                    d["tid"] = -1 - next_id
                next_id += 1

        detections[fi] = here
        on_court_total += len(here)
        fi += 1
        if fi % 60 == 0:
            print(f"frame {fi}: {len(here)} on court", flush=True)
    cap.release()

    # smooth each track's court position over a short window. A single frame's
    # ankle keypoints jitter by a few pixels, and the homography turns that into
    # tens of centimetres on the far side of the court, which reads as the dots
    # zigzagging instead of walking.
    win = max(1, int(round(args.smooth_seconds * fps)))
    if win > 1:
        by_track = {}
        for f, ds in detections.items():
            for d in ds:
                by_track.setdefault(d.get("tid"), []).append((f, d))
        for tid, seq in by_track.items():
            seq.sort(key=lambda t: t[0])
            xs = np.array([d["cx"] for _, d in seq], dtype=float)
            ys = np.array([d["cy"] for _, d in seq], dtype=float)
            k = np.ones(win) / win
            pad = win // 2
            sx = np.convolve(np.pad(xs, pad, mode="edge"), k, mode="valid")[:len(xs)]
            sy = np.convolve(np.pad(ys, pad, mode="edge"), k, mode="valid")[:len(ys)]
            for (f, d), a, b in zip(seq, sx, sy):
                d["cx"], d["cy"] = float(a), float(b)

    # learn the two kits from every torso patch in the clip at once, then label
    # each detection by which kit it is closer to. Because the clusters are fit
    # on thousands of patches, a short or broken track no longer causes a colour
    # to flip - which per-track voting could not fix.
    # Find where the two kits actually split in THIS clip rather than using a
    # fixed number. Thresholds tuned on one 22s clip failed on a 54s clip from
    # the same game - lighting moved the scores enough to flip white to red.
    # A 1-D two-means on the scores adapts to whatever the footage looks like.
    scores = [d["feat"] for ds in detections.values() for d in ds if d.get("feat") is not None]
    if len(scores) >= 30:
        a = np.array(sorted(scores))
        lo, hi = a[len(a) // 10], a[-len(a) // 10]      # robust starting points
        for _ in range(40):
            mid = (lo + hi) / 2
            left, right = a[a < mid], a[a >= mid]
            if not len(left) or not len(right):
                break
            lo, hi = left.mean(), right.mean()
        split = (lo + hi) / 2
        band = 0.06                                     # scores this close to the split are unclear
        print(f"kit split learned from this clip: {split:+.2f}")
    else:
        split, band = 0.0, 0.15

    # Fill brief dropouts. A player missed for a frame or two reads as the dot
    # flickering; interpolating between two real observations is safe because it
    # is anchored at both ends. Long gaps are left empty rather than invented.
    if args.fill_seconds > 0:
        max_gap = max(1, int(round(args.fill_seconds * fps)))
        seqs = {}
        for f, ds in detections.items():
            for d in ds:
                seqs.setdefault(d.get("tid"), []).append((f, d))
        for tid, seq in seqs.items():
            seq.sort(key=lambda t: t[0])
            for (fa, da), (fb, db) in zip(seq, seq[1:]):
                gap = fb - fa
                if not (1 < gap <= max_gap):
                    continue
                dist = ((db["cx"] - da["cx"]) ** 2 + (db["cy"] - da["cy"]) ** 2) ** 0.5
                if dist > 40 * gap:          # implausible travel: do not invent it
                    continue
                for g in range(1, gap):
                    t = g / gap
                    detections.setdefault(fa + g, []).append({
                        "cx": da["cx"] + (db["cx"] - da["cx"]) * t,
                        "cy": da["cy"] + (db["cy"] - da["cy"]) * t,
                        "kp": da["kp"], "pt": da["pt"], "box": da["box"],
                        "feat": da.get("feat"), "tid": tid, "filled": True})

    # Smooth each track's redness score before deciding anything. Smoothing the
    # score keeps the per-frame 3/3 constraint intact, whereas voting on the
    # labels afterwards overrode it and put the counts back out (46% vs 100%).
    by_track = {}
    for f, ds in detections.items():
        for d in ds:
            if d.get("feat") is not None:
                by_track.setdefault(d.get("tid"), []).append((f, d))
    for tid, seq in by_track.items():
        seq.sort(key=lambda t: t[0])
        vals = np.array([d["feat"] for _, d in seq], dtype=float)
        w = min(len(vals), 9)
        if w >= 3:
            pad = w // 2
            sm = np.convolve(np.pad(vals, pad, mode="edge"), np.ones(w) / w, mode="valid")[:len(vals)]
            for (_, d), v in zip(seq, sm):
                d["feat"] = float(v)

    # Use the rule of the game: it is 3-on-3, so when six players are on court
    # exactly three of them are in each kit. Ranking the six by how red they are
    # and splitting 3/3 is far stronger than judging each one against a
    # threshold, because it only needs the ORDER to be right, not the absolute
    # score - and the order survives lighting changes that move every score at
    # once.
    per_track = {}
    for f, ds in detections.items():
        scored = [d for d in ds if d.get("feat") is not None]
        if len(scored) == 6:
            order = sorted(scored, key=lambda d: -d["feat"])
            for i, d in enumerate(order):
                d["team"] = "C" if i < 3 else "W"
        else:
            for d in ds:
                sc = d.get("feat")
                if sc is None:
                    d["team"] = "?"
                    continue
                d["team"] = "C" if sc > split + band else ("W" if sc < split - band else "?")
        for d in ds:
            if d.get("team") in ("C", "W"):
                per_track.setdefault(d.get("tid"), []).append(d["team"])
    settled = {}
    for tid, labs in per_track.items():
        c, w = labs.count("C"), labs.count("W")
        settled[tid] = "C" if c > w else ("W" if w > c else "?")
    if not args.no_track_vote:
        for f, ds in detections.items():
            for d in ds:
                if d.get("tid") in settled:
                    d["team"] = settled[d["tid"]]

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
            frame_pts.append({"x": d["cx"], "y": d["cy"], "team": t, "track": d.get("tid")})
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
    print(f"referee detections removed: {n_refs[0]}")
    print(f"done. {fi} frames -> {args.out}   {on_court_total/max(1,fi):.1f} players/frame")
    if args.save_positions:
        json.dump({"fps": fps, "court_cm": [COURT_W, COURT_H], "frames": positions},
                  open(args.save_positions, "w"))
        print(f"positions (cm) -> {args.save_positions}")


if __name__ == "__main__":
    main()
