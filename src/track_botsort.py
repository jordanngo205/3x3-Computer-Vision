"""
Player tracking following the TrackID3x3 baseline structure (arXiv:2503.18282)
instead of our own hand-rolled association.

Why the rewrite: we self-trained an appearance embedding on pseudo-labels from
our own clips and it turned out to be worthless for identity - measured on real
footage, its same-team distance (0.616) and cross-team distance (0.640) were
effectively identical, so red and white jerseys were swapping freely. The
training pairs were the problem: almost every source clip contained only one
team, so the model was never asked to learn jersey colour, and "same track
within 45 frames" positives are near-duplicate images it could match on pose
and background instead of identity.

TrackID3x3's baseline gets its identity signal from a tracker with a ReID model
pretrained on large sports datasets (CAMELTrack / BoT-SORT-ReID), plus a
tracklet-level colour histogram for team assignment. This script does the same
with the tools we already have:

  1. detect + associate with BoT-SORT + ReID (Ultralytics), with global motion
     compensation so camera pans don't break association
  2. drop detections off court, and boxes that are really two players in one
     (the small detector merged players in contact; yolo11m separates them)
  3. assign each tracklet a team by voting torso colour over all its frames -
     per-frame colour is ambiguous ~20% of the time, but a vote over hundreds
     of frames is not
  4. merge fragmented tracklets, refusing to merge across teams or across
     tracklets that were on court simultaneously
  5. render
"""
import argparse
import os

import cv2
import numpy as np
from ultralytics import YOLO


def detect_shots(video, cut_threshold=28.0, burst=8):
    """Find camera cuts, and decide which shots are the calibrated wide view.

    TrackID3x3 assumes a single fixed camera showing the whole court. This is
    a TV broadcast: it cuts to replays and close-ups, where the court polygon
    points at the wrong pixels, every track legitimately dies, and the merge
    rule ("two tracks never on screen together are the same player") is
    meaningless. So we segment the video and only track inside continuous
    shots that match the calibrated framing.
    """
    cap = cv2.VideoCapture(video)
    prev, cuts, thumbs, fi = None, [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        small = cv2.resize(f, (160, 90))
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None and np.abs(g - prev).mean() > cut_threshold:
            if not cuts or fi - cuts[-1] > burst:
                cuts.append(fi)
        thumbs.append(g)
        prev = g
        fi += 1
    cap.release()

    bounds = [0] + cuts + [fi]
    shots = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    # the calibrated view is whatever the longest shot looks like; other shots
    # are kept only if they look like it
    longest = max(shots, key=lambda s: s[1] - s[0])
    ref = np.median(np.stack(thumbs[longest[0]:longest[1]:5]), axis=0)
    keep = []
    for a, b in shots:
        med = np.median(np.stack(thumbs[a:b:max(1, (b - a) // 20 or 1)]), axis=0)
        if np.abs(med - ref).mean() < 22.0:
            keep.append((a, b))
    return shots, keep


def foot_on_court(court_poly, fx, fy):
    return cv2.pointPolygonTest(court_poly, (float(fx), float(fy)), False) >= 0


def containment(outer, inner):
    x1 = max(outer[0], inner[0]); y1 = max(outer[1], inner[1])
    x2 = min(outer[2], inner[2]); y2 = min(outer[3], inner[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))


def is_two_person_box(box, others, min_aspect, contain_frac, contain_area_ratio):
    """A box around two players standing together is much squarer than an
    upright player, and often swallows the two individual boxes whole."""
    w, h = box[2] - box[0], box[3] - box[1]
    if w <= 0 or h <= 0:
        return True
    if h / w < min_aspect:
        return True
    area = w * h
    swallowed = 0
    for o in others:
        if o is box:
            continue
        area_o = max(1e-6, (o[2] - o[0]) * (o[3] - o[1]))
        if containment(box, o) >= contain_frac and area >= contain_area_ratio * area_o:
            swallowed += 1
    return swallowed >= 2


def torso_colour_vote(frame, box):
    """Return (opponent_colour_fraction, white_fraction) over the torso window.

    Deliberately looks only at the torso: the jersey is there, while the full
    box is dominated by legs, court and background that carry no team signal.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    ty1, ty2 = max(0, y1 + int(h * 0.20)), max(1, y1 + int(h * 0.55))
    tx1, tx2 = max(0, x1 + int(w * 0.25)), max(1, x1 + int(w * 0.75))
    patch = frame[ty1:ty2, tx1:tx2]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    coloured = ((S > 90) & (V > 60))
    white = ((S < 60) & (V > 120))
    hue_of_coloured = H[coloured]
    return coloured.mean(), white.mean(), (np.median(hue_of_coloured) if hue_of_coloured.size else None)


def split_on_team_flip(history, track_votes, window=7, min_run=12):
    """Cut a tracklet wherever its jersey colour changes.

    A single track whose torso colour goes red -> white has demonstrably
    jumped between two different players; that is exactly the swap that
    survived the tracker. Per-frame colour is noisy (~20% ambiguous), so the
    label is smoothed over a window and a change only counts when the new
    colour holds for `min_run` frames. Cutting here is safe: the merge step
    afterwards reassembles the pieces, and it can only rejoin same-team
    fragments.
    """
    new_history, new_votes, splits = {}, {}, 0
    next_suffix = 0
    for tid, frames_boxes in history.items():
        votes = track_votes.get(tid, [])
        frames_sorted = sorted(frames_boxes)
        labels = {}
        for f, v in votes:
            labels[f] = "COLOUR" if v[0] > v[1] else "WHITE"
        seq = [labels.get(f) for f in frames_sorted]
        known = [s for s in seq if s]
        if len(known) < min_run * 2:
            new_history[tid] = frames_boxes
            new_votes[tid] = votes
            continue

        smoothed = []
        for i in range(len(seq)):
            w = [s for s in seq[max(0, i - window):i + window + 1] if s]
            smoothed.append(max(set(w), key=w.count) if w else (smoothed[-1] if smoothed else "WHITE"))

        # only accept a change of colour that persists, so noise can't cut a track
        segments, start = [], 0
        for i in range(1, len(smoothed)):
            if smoothed[i] != smoothed[i - 1]:
                run = 1
                while i + run < len(smoothed) and smoothed[i + run] == smoothed[i]:
                    run += 1
                if run >= min_run and i - start >= min_run:
                    segments.append((start, i))
                    start = i
        segments.append((start, len(frames_sorted)))

        if len(segments) == 1:
            new_history[tid] = frames_boxes
            new_votes[tid] = votes
            continue

        splits += len(segments) - 1
        for a, b in segments:
            piece_frames = frames_sorted[a:b]
            key = f"{tid}s{next_suffix}"
            next_suffix += 1
            new_history[key] = {f: frames_boxes[f] for f in piece_frames}
            keep = set(piece_frames)
            new_votes[key] = [(f, v) for f, v in votes if f in keep]
    return new_history, new_votes, splits


def assign_teams(track_votes):
    """Team per tracklet from its accumulated torso-colour votes.

    Two teams: one in white/light kit, one in a saturated colour. We don't
    hardcode which colour - the coloured team's hue is whatever the clip's
    coloured detections cluster around.
    """
    teams = {}
    for tid, votes in track_votes.items():
        if not votes:
            teams[tid] = "?"
            continue
        coloured = np.mean([v[0] for _, v in votes])
        white = np.mean([v[1] for _, v in votes])
        teams[tid] = "COLOUR" if coloured > white else "WHITE"
    return teams


def merge_tracklets(history, teams, max_overlap_frames, gap_reach_per_frame):
    """Stitch fragments of the same player together (TrackID3x3 Algorithm 1).

    Core constraint from the paper: two tracklets on court at the same time
    cannot be one person. We add a team constraint - a fragment can only join
    a tracklet wearing the same kit - which is the check that would have
    prevented the red/white identity swaps outright.
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

    candidates = sorted(
        (spans[b][0] - spans[a][1], a, b)
        for a in ids for b in ids
        if a != b and spans[a][1] < spans[b][0]
    )

    merged = 0
    for gap, a, b in candidates:
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        if teams.get(a, "?") != teams.get(b, "?"):
            continue  # different kit - cannot be the same player
        group_a = [i for i in ids if root(i) == ra]
        group_b = [i for i in ids if root(i) == rb]
        overlap = sum(len(frames[i] & frames[j]) for i in group_a for j in group_b)
        if overlap > max_overlap_frames:
            continue
        last, first = history[a][spans[a][1]], history[b][spans[b][0]]
        jump = ((first[0] + first[2] - last[0] - last[2]) / 2) ** 2 + \
               ((first[1] + first[3] - last[1] - last[3]) / 2) ** 2
        height = max(1.0, last[3] - last[1])
        if jump ** 0.5 > height * gap_reach_per_frame * max(1, gap):
            continue
        parent[rb] = ra
        merged += 1
    return {i: root(i) for i in ids}, merged


def id_color(track_id):
    rng = np.random.RandomState(abs(hash(str(track_id))) % (2 ** 31))
    return tuple(int(v) for v in rng.randint(60, 255, size=3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--model", default="yolo11m.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--court-poly", default=None)
    p.add_argument("--min-aspect", type=float, default=1.15)
    p.add_argument("--contain-frac", type=float, default=0.75)
    p.add_argument("--contain-area-ratio", type=float, default=1.4)
    p.add_argument("--merge-max-overlap", type=int, default=10)
    p.add_argument("--merge-gap-reach", type=float, default=0.15)
    p.add_argument("--no-merge", action="store_true")
    p.add_argument("--no-shot-detect", action="store_true",
                   help="treat the clip as one continuous fixed-camera shot")
    p.add_argument("--save-json", default=None,
                   help="write final identities/boxes so re-rendering needs no re-detection")
    p.add_argument("--no-split", action="store_true",
                   help="skip cutting tracks where the jersey colour flips")
    p.add_argument("--min-track-frames", type=int, default=15,
                   help="drop tracklets seen fewer times than this (detector noise)")
    p.add_argument("--tracker", default=None, help="tracker yaml; defaults to the bundled reid config")
    args = p.parse_args()

    out_path = args.out or args.video.rsplit(".", 1)[0] + "_botsort.mp4"
    tracker_cfg = args.tracker or os.path.join(os.path.dirname(__file__), "botsort_reid.yaml")

    court_poly = None
    if args.court_poly:
        nums = [float(v) for v in args.court_poly.split(",")]
        court_poly = np.array(list(zip(nums[0::2], nums[1::2])), dtype=np.int32)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()

    shots, wide = ([], [])
    if not args.no_shot_detect:
        shots, wide = detect_shots(args.video)
        dropped = sum(b - a for a, b in shots) - sum(b - a for a, b in wide)
        print(f"shot detection: {len(shots)} shots, {len(wide)} match the calibrated wide view "
              f"({dropped} frames of replay/close-up will be skipped)")
    shot_of = {}
    for si, (a, b) in enumerate(wide):
        for f in range(a, b):
            shot_of[f] = si

    model = YOLO(args.model)
    history = {}      # tid -> {frame: box}
    track_votes = {}  # tid -> [(coloured_frac, white_frac, hue), ...]

    for fi, res in enumerate(model.track(source=args.video, stream=True, classes=[0],
                                         conf=args.conf, tracker=tracker_cfg,
                                         persist=True, verbose=False)):
        if res.boxes is None or res.boxes.id is None:
            continue
        if shot_of and fi not in shot_of:
            continue  # replay / close-up: our calibration does not apply here
        frame = res.orig_img
        boxes = res.boxes.xyxy.cpu().numpy().tolist()
        ids = res.boxes.id.cpu().numpy().astype(int).tolist()

        on_court = []
        for box, tid in zip(boxes, ids):
            if court_poly is not None and not foot_on_court(court_poly, (box[0] + box[2]) / 2, box[3]):
                continue
            on_court.append((box, tid))

        raw_boxes = [b for b, _ in on_court]
        for box, tid in on_court:
            if is_two_person_box(box, raw_boxes, args.min_aspect,
                                 args.contain_frac, args.contain_area_ratio):
                continue
            key = f"{shot_of.get(fi, 0)}_{tid}" if shot_of else tid
            history.setdefault(key, {})[fi] = box
            vote = torso_colour_vote(frame, box)
            if vote:
                track_votes.setdefault(key, []).append((fi, vote))

        if fi % 100 == 0:
            print(f"frame {fi}, live tracklets so far: {len(history)}", flush=True)

    n_frames = fi + 1
    print(f"tracking done: {n_frames} frames, {len(history)} raw tracklets")

    # ids become strings once a track is split, so normalise up front and keep
    # every downstream dict keyed the same way
    history = {str(t): fb for t, fb in history.items() if len(fb) >= args.min_track_frames}
    track_votes = {str(t): v for t, v in track_votes.items()}
    print(f"after dropping tracklets under {args.min_track_frames} frames: {len(history)}")

    if not args.no_split:
        history, track_votes, splits = split_on_team_flip(history, track_votes)
        print(f"team-flip split: {splits} cuts -> {len(history)} fragments "
              f"(each cut is a track that provably changed player)")

    teams = assign_teams({t: track_votes.get(t, []) for t in history})
    counts = {}
    for t in teams.values():
        counts[t] = counts.get(t, 0) + 1
    print(f"team assignment: {counts}")

    if args.no_merge:
        mapping, merged = {t: t for t in history}, 0
    else:
        mapping, merged = merge_tracklets(history, teams, args.merge_max_overlap, args.merge_gap_reach)
    print(f"merge: {merged} merges, {len(history)} -> {len(set(mapping.values()))} identities")

    # verification the count alone can't give: a correct identity wears one kit
    # for its whole life, so per-identity colour purity measures leftover swaps
    purity_num, purity_den = 0.0, 0
    impure = []
    for final_id in set(mapping.values()):
        labels = []
        for tid in history:
            if mapping[tid] != final_id:
                continue
            for _, v in track_votes.get(tid, []):
                labels.append("C" if v[0] > v[1] else "W")
        if len(labels) < 20:
            continue
        frac_c = labels.count("C") / len(labels)
        purity = max(frac_c, 1 - frac_c)
        purity_num += purity * len(labels)
        purity_den += len(labels)
        if purity < 0.9:
            impure.append((final_id, len(labels), round(frac_c, 2)))
    if purity_den:
        print(f"identity colour purity (1.0 = never changes team): {purity_num/purity_den:.3f}")
        print(f"impure identities (id, frames, colour-fraction): {impure}")

    # internal ids carry split suffixes like "4s11"; renumber to plain
    # sequential ids so the rendered video is readable
    order = sorted(set(mapping.values()),
                   key=lambda fid: -sum(len(history[t]) for t in history if mapping[t] == fid))
    display = {fid: i for i, fid in enumerate(order)}

    by_frame = {}
    for tid, fb in history.items():
        for f, box in fb.items():
            by_frame.setdefault(f, []).append((display[mapping[tid]], box, teams.get(tid, "?")))

    if args.save_json:
        import json
        out = {"fps": fps, "width": w, "height": h, "n_frames": n_frames,
               "tracks": {}, "teams": {}}
        for tid, fb in history.items():
            did = str(display[mapping[tid]])
            out["tracks"].setdefault(did, {}).update({str(f): [float(x) for x in b] for f, b in fb.items()})
            out["teams"][did] = teams.get(tid, "?")
        with open(args.save_json, "w") as fh:
            json.dump(out, fh)
        print(f"saved tracks -> {args.save_json}")

    cap = cv2.VideoCapture(args.video)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, box, team in by_frame.get(fi, []):
            x1, y1, x2, y2 = [int(v) for v in box]
            colour = id_color(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"#{tid} {team[:1]}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        writer.write(frame)
        fi += 1
    cap.release()
    writer.release()
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
