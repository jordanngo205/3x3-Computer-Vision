"""One homography per frame, because the broadcast camera pans.

A clicked calibration is only true for the frame it was clicked on. Measured on
Canada v Ger Long.mov, a fixed patch of floor moves up to 415 px across the
"same" camera shot, which is 180-300 cm of court error - players land two to
three metres from where they are.

So the clicked homography is treated as an anchor at one reference frame and
carried to every other frame:

  H_image_to_court(f) = H_ref_to_court @ H_f_to_ref

H_f_to_ref is built by tracking floor points with KLT from frame to frame. KLT
is used rather than descriptor matching on purpose: the floor is a large plain
grey surface, and ORB descriptors on it are so repetitive that matching to a
distant frame confidently returns the identity transform. KLT only ever has to
explain a few pixels of motion between consecutive frames, which is unambiguous.

Chaining alone drifts, so the result is re-anchored onto the painted lines: the
court model is snapped to the line pixels by optimising the image positions of
four court corners (well scaled, in pixels) rather than the raw matrix entries,
which are badly conditioned.
"""
import argparse
import json
import pickle

import cv2
import numpy as np

COURT_W, COURT_H = 1500.0, 1100.0
BASKET = (750.0, 157.0)
ARC_R = 675.0
KEY = (505.0, 995.0, 0.0, 580.0)        # x0, x1, y0, y1

# the four court points whose image positions parametrise the homography
ANCHORS = np.array([[0.0, 0.0], [COURT_W, 0.0], [COURT_W, COURT_H], [0.0, COURT_H]],
                   dtype=np.float32)


def court_model(step=0.012):
    """Points along every painted line we can rely on, in court centimetres."""
    pts = []
    x0, x1, y0, y1 = KEY
    segs = [((0, 0), (COURT_W, 0)), ((COURT_W, 0), (COURT_W, COURT_H)),
            ((COURT_W, COURT_H), (0, COURT_H)), ((0, COURT_H), (0, 0)),
            ((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    for a, b in segs:
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) * step))
        for t in np.linspace(0, 1, n):
            pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    for t in np.linspace(0, np.pi, 240):
        x, y = BASKET[0] + ARC_R * np.cos(t), BASKET[1] + ARC_R * np.sin(t)
        if 0 <= x <= COURT_W and 0 <= y <= COURT_H:
            pts.append((x, y))
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)


MODEL = court_model()


def line_mask(frame):
    """Painted lines: desaturated and brighter than the surrounding floor."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 1] < 70) & (g > np.percentile(g, 85))).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def floor_mask(frame):
    """The playing surface: the largest desaturated mid-grey region."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    m = ((S < 60) & (V > 70) & (V < 200)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return (m * 255).astype(np.uint8)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return ((lab == big).astype(np.uint8)) * 255


def alignment(frame_line_mask, H_img2court, min_cover=0.35):
    """Fraction of the projected court model that lands on a painted line.

    Scored only over the model points that fall inside the frame, so a
    homography that squeezes the court into a corner could otherwise score
    near 1.0 on a handful of pixels - that artefact had me briefly believe a
    degenerate fit was the best one. Below min_cover the score is refused.
    """
    h, w = frame_line_mask.shape
    p = cv2.perspectiveTransform(MODEL, np.linalg.inv(H_img2court)).reshape(-1, 2)
    inside = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    if inside.sum() < 100 or inside.mean() < min_cover:
        return 0.0
    fat = cv2.dilate(frame_line_mask, np.ones((7, 7), np.uint8))
    return float(fat[p[inside, 1].astype(int), p[inside, 0].astype(int)].mean())


def h_from_anchors(img_pts):
    """Homography image->court from where the four court corners sit in the image."""
    H, _ = cv2.findHomography(np.asarray(img_pts, np.float32), ANCHORS)
    return H


def snap_to_lines(frame, H_init, iters=30, step0=6.0):
    """Re-anchor on the painted lines.

    Optimising the raw matrix entries does not converge - they differ by orders
    of magnitude. Moving the four court corners around in the image is the same
    eight degrees of freedom in units the optimiser can handle.
    """
    lm = line_mask(frame)
    dt = cv2.distanceTransform((1 - lm).astype(np.uint8), cv2.DIST_L2, 3)
    h, w = dt.shape

    def cost(H):
        p = cv2.perspectiveTransform(MODEL, np.linalg.inv(H)).reshape(-1, 2)
        xi = np.clip(p[:, 0], 0, w - 1).astype(int)
        yi = np.clip(p[:, 1], 0, h - 1).astype(int)
        d = dt[yi, xi].copy()
        off = (p[:, 0] < 0) | (p[:, 0] >= w) | (p[:, 1] < 0) | (p[:, 1] >= h)
        d[off] = 80.0
        return float(np.mean(np.minimum(d, 80.0)))

    cur = cv2.perspectiveTransform(ANCHORS.reshape(-1, 1, 2),
                                   np.linalg.inv(H_init)).reshape(-1, 2)
    best_H, best_c, step = H_init, cost(H_init), step0
    for _ in range(iters):
        improved = False
        for i in range(4):
            for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                trial = cur.copy()
                trial[i] += (dx, dy)
                H = h_from_anchors(trial)
                if H is None:
                    continue
                c = cost(H)
                if c < best_c - 1e-4:
                    best_c, best_H, cur, improved = c, H, trial, True
        if not improved:
            step /= 2.0
            if step < 0.4:
                break
    return best_H, best_c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--calib", required=True, help="clicked calibration for the reference frame")
    p.add_argument("--ref-frame", type=int, required=True,
                   help="the frame the calibration was clicked on")
    p.add_argument("--out", required=True, help="pickle of per-frame homographies")
    p.add_argument("--segments", default=None,
                   help="json list of [start,end] frame ranges on the calibrated camera")
    p.add_argument("--scale", type=float, default=0.5, help="tracking resolution")
    p.add_argument("--min-points", type=int, default=120,
                   help="re-detect floor features when fewer than this survive")
    p.add_argument("--reanchor-every", type=int, default=25,
                   help="snap to the painted lines this often, to kill chain drift")
    p.add_argument("--no-reanchor", action="store_true")
    args = p.parse_args()

    H_ref = np.array(json.load(open(args.calib))["H_image_to_court"])
    S = np.diag([args.scale, args.scale, 1.0])
    Sinv = np.linalg.inv(S)

    segs = json.load(open(args.segments)) if args.segments else None

    # Keep only what tracking needs in memory. Holding every frame at full
    # colour is 2305 x 1984x1112x3 = about 15 GB, which drove the machine into
    # swap; the half-resolution grey frames KLT actually uses are ~1.3 GB, and
    # the few colour frames the line re-anchoring needs are re-read on demand.
    cap = cv2.VideoCapture(args.video)
    grey = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        grey.append(cv2.cvtColor(cv2.resize(f, None, fx=args.scale, fy=args.scale),
                                 cv2.COLOR_BGR2GRAY))
    cap.release()
    n = len(grey)
    mb = sum(g.nbytes for g in grey) / 1e6
    print(f"{n} frames, {mb:.0f} MB of half-resolution grey held for tracking")

    _colour_cap = cv2.VideoCapture(args.video)

    def colour(i):
        """Re-read one frame in colour, for the masks that need it."""
        _colour_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = _colour_cap.read()
        return f if ok else None

    def in_seg(i):
        return segs is None or any(a <= i < b for a, b in segs)

    lk = dict(winSize=(21, 21), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def small(i):
        return grey[i]

    def detect(i):
        g = small(i)
        c = colour(i)
        if c is None:
            return g, np.empty((0, 1, 2), np.float32)
        m = cv2.resize(floor_mask(c), (g.shape[1], g.shape[0]))
        pts = cv2.goodFeaturesToTrack(g, maxCorners=1200, qualityLevel=0.01,
                                      minDistance=7, mask=(m > 0).astype(np.uint8))
        return g, (pts if pts is not None else np.empty((0, 1, 2), np.float32))

    out = {}
    for direction in (1, -1):
        H_c2r = np.eye(3)                      # current frame -> reference frame, full res
        prev_g, prev_pts = detect(args.ref_frame)
        i = args.ref_frame
        if direction == 1:
            out[i] = H_ref.copy()
        since_anchor = 0
        while True:
            j = i + direction
            if not (0 <= j < n):
                break
            if not in_seg(j):
                i = j
                continue
            g = small(j)
            if len(prev_pts) < args.min_points:
                prev_g, prev_pts = detect(i)
            if len(prev_pts) >= 12:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, g, prev_pts, None, **lk)
                back, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev_g, nxt, None, **lk)
                fb = np.linalg.norm(prev_pts - back, axis=2).ravel()
                keep = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.5)
                a, b = prev_pts[keep], nxt[keep]
                if len(a) >= 12:
                    Hs, inl = cv2.findHomography(b, a, cv2.RANSAC, 2.0)
                    if Hs is not None and inl is not None and inl.sum() >= 10:
                        H_step = Sinv @ Hs @ S        # frame j -> frame i, full res
                        H_c2r = H_c2r @ H_step
                        prev_g, prev_pts = g, b[inl.ravel() == 1]
                    else:
                        prev_g, prev_pts = detect(j)
                else:
                    prev_g, prev_pts = detect(j)
            else:
                prev_g, prev_pts = detect(j)

            H = H_ref @ H_c2r
            since_anchor += 1
            if not args.no_reanchor and since_anchor >= args.reanchor_every:
                cj = colour(j)
                snapped, _ = (snap_to_lines(cj, H) if cj is not None else (None, None))
                if snapped is not None:
                    H = snapped
                    H_c2r = np.linalg.inv(H_ref) @ H   # fold the correction back in
                since_anchor = 0
                prev_g, prev_pts = detect(j)
            out[j] = H
            i = j
            if len(out) % 200 == 0:
                print(f"  {len(out)} frames registered", flush=True)

    pickle.dump({k: v.tolist() for k, v in out.items()}, open(args.out, "wb"))
    print(f"wrote {len(out)} per-frame homographies -> {args.out}")

    check = sorted(out)[::max(1, len(out) // 15)]
    print(f"\n{'frame':>6} {'fixed':>7} {'per-frame':>10}")
    fa, pa = [], []
    for f in check:
        cf = colour(f)
        if cf is None:
            continue
        lm = line_mask(cf)
        a, b = alignment(lm, H_ref), alignment(lm, np.array(out[f]))
        fa.append(a); pa.append(b)
        print(f"{f:6d} {a:7.3f} {b:10.3f}")
    print(f"\nmean alignment  fixed {np.mean(fa):.3f}  per-frame {np.mean(pa):.3f}")


if __name__ == "__main__":
    main()
