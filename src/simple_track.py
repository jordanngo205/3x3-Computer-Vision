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


def id_color(track_id):
    rng = np.random.RandomState(track_id * 37 + 7)
    return tuple(int(v) for v in rng.randint(60, 255, size=3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default=None, help="output video path (default: <video>_tracked.mp4)")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--court-poly", default=None,
                         help="comma-separated x,y pairs overriding the default court polygon "
                              "(needed when frame resolution/camera framing differs from video.mp4)")
    parser.add_argument("--embedding-checkpoint", default="pseudo_labels/embedding_model_v2.pt",
                         help="trained appearance model used to re-identify people after a gap; "
                              "pass --no-appearance to disable and fall back to pure box-overlap tracking")
    parser.add_argument("--no-appearance", action="store_true")
    parser.add_argument("--reconnect-max", type=float, default=APPEARANCE_RECONNECT_MAX,
                         help="looser appearance bar for position-plausible reconnections (stage 2)")
    parser.add_argument("--model", default="yolo11n.pt",
                         help="YOLO weights. yolo11n is fast but merges players in contact into one box; "
                              "yolo11m separates them correctly and is the better choice for quality")
    parser.add_argument("--min-aspect", type=float, default=1.15,
                         help="reject detections shorter/wider than this height:width ratio - a box around "
                              "two players side by side is far squarer than an upright player")
    parser.add_argument("--contain-frac", type=float, default=0.75)
    parser.add_argument("--contain-area-ratio", type=float, default=1.4)
    parser.add_argument("--cache-dets", default=None,
                         help="path to cache raw detections. Detection is by far the slowest stage, so "
                              "caching it lets tracking/merge/render params be retuned in seconds "
                              "instead of re-running the detector over the whole clip.")
    parser.add_argument("--max-identities", type=int, default=0,
                         help="keep only the N best-supported identities (3x3 has 6 players; use ~8 to "
                              "allow for the two referees). 0 disables the cap.")
    parser.add_argument("--no-merge", action="store_true",
                         help="skip the offline tracklet-merging pass")
    parser.add_argument("--merge-max-overlap", type=int, default=10,
                         help="frames two tracklets may coexist and still be considered the same player "
                              "(TrackID3x3 used 10 indoor / 50 outdoor)")
    parser.add_argument("--merge-appearance-max", type=float, default=0.85,
                         help="how alike two fragments must look to be merged")
    parser.add_argument("--merge-gap-reach", type=float, default=0.15,
                         help="plausible travel per missing frame, in box-heights")
    parser.add_argument("--no-coast", action="store_true",
                         help="freeze a hidden player's box instead of coasting it along their last motion")
    parser.add_argument("--gallery-size", type=int, default=GALLERY_SIZE,
                         help="how many clean appearance snapshots to keep per track (1 = single snapshot)")
    args = parser.parse_args()
    gallery_size = max(1, args.gallery_size)

    embedding_model, embedding_device = (None, None)
    if not args.no_appearance:
        embedding_model, embedding_device = load_embedding_model(args.embedding_checkpoint)

    if args.court_poly:
        nums = [float(v) for v in args.court_poly.split(",")]
        court_poly = np.array(list(zip(nums[0::2], nums[1::2])), dtype=np.int32)
    else:
        court_poly = DEFAULT_COURT_POLY

    out_path = args.out or args.video.rsplit(".", 1)[0] + "_tracked.mp4"

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_missed_frames = max(1, int(round(MAX_HOLD_SECONDS * fps)))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracks = {}  # id -> {'box': [x1,y1,x2,y2], 'missed': int}
    history = {}        # id -> {frame_idx: box} for every frame the player was really seen
    final_gallery = {}  # id -> appearance snapshots, kept after the track itself dies
    next_id = 0
    frame_idx = 0
    births_in_contact = 0
    deaths_in_contact = 0
    dropped_shape = 0
    dropped_wrapper = 0

    det_cache = {}
    cache_hit = False
    if args.cache_dets and os.path.exists(args.cache_dets):
        with open(args.cache_dets, "rb") as f:
            det_cache = pickle.load(f)
        cache_hit = True
        print(f"loaded cached detections for {len(det_cache)} frames from {args.cache_dets}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if cache_hit:
            all_dets = det_cache.get(frame_idx, [])
        else:
            results = model(frame, classes=[0], conf=args.conf, verbose=False)[0]
            all_dets = [box.tolist() for box in results.boxes.xyxy.cpu().numpy()]
            if args.cache_dets:
                det_cache[frame_idx] = all_dets
        dets = []
        for det in all_dets:
            x1, y1, x2, y2 = det
            fx, fy = (x1 + x2) / 2, y2  # foot point: bottom-center of the box
            if foot_on_court(court_poly, fx, fy):
                dets.append(det)
        dets, n_shape, n_wrapper = reject_two_person_boxes(
            dets, args.min_aspect, args.contain_frac, args.contain_area_ratio)
        dropped_shape += n_shape
        dropped_wrapper += n_wrapper

        det_embeddings = [None] * len(dets)
        if embedding_model is not None:
            det_embeddings = [compute_embedding(embedding_model, embedding_device, frame, d) for d in dets]

        # crowding check: a detection whose box heavily overlaps another
        # detection is mid-screen/contact - its crop shows two players mashed
        # together, so don't let it overwrite anyone's appearance memory.
        crowded = [False] * len(dets)
        for j in range(len(dets)):
            for k in range(len(dets)):
                if j != k and box_iou(dets[j], dets[k]) > 0.25:
                    crowded[j] = True
                    break

        def update_track(tid, j):
            t = tracks[tid]
            old_box = t["box"]
            new_box = dets[j]
            # velocity: only learn it from consecutive visible frames - a
            # reconnection after a gap says nothing about per-frame speed
            if t["missed"] == 0:
                dx = ((new_box[0] + new_box[2]) - (old_box[0] + old_box[2])) / 2
                dy = ((new_box[1] + new_box[3]) - (old_box[1] + old_box[3])) / 2
                ovx, ovy = t.get("vel", (0.0, 0.0))
                t["vel"] = (0.7 * ovx + 0.3 * dx, 0.7 * ovy + 0.3 * dy)
            t["box"] = new_box
            t["missed"] = 0
            if det_embeddings[j] is not None and not crowded[j]:
                gallery = t.setdefault("gallery", [])
                gallery.append(det_embeddings[j])
                if len(gallery) > gallery_size:
                    gallery.pop(0)

        track_ids = list(tracks.keys())
        matched_track_ids = set()
        matched_det_idxs = set()

        # stage 1: position + appearance together. Position (IoU) does the
        # heavy lifting for continuously-visible players; the appearance term
        # breaks ties during screens/contact so overlapping players don't
        # swap IDs, and an appearance veto rejects a positionally-convenient
        # match that clearly looks like the wrong person.
        if track_ids and dets:
            cost = np.zeros((len(track_ids), len(dets)))
            ious = np.zeros((len(track_ids), len(dets)))
            for i, tid in enumerate(track_ids):
                for j, det in enumerate(dets):
                    ious[i, j] = box_iou(tracks[tid]["box"], det)
                    iou_cost = 1 - ious[i, j]
                    app_dist = appearance_distance(tracks[tid].get("gallery"), det_embeddings[j])
                    if app_dist == float("inf"):
                        cost[i, j] = iou_cost  # no embeddings available - position only
                    else:
                        cost[i, j] = 0.6 * iou_cost + 0.4 * min(app_dist / 1.2, 1.0)
                        if app_dist > 1.0 and iou_cost > 0.3:
                            cost[i, j] = 10.0  # looks wrong AND isn't a near-perfect overlap: veto
            row_idx, col_idx = linear_sum_assignment(cost)
            for i, j in zip(row_idx, col_idx):
                # require real overlap - zero-overlap reconnections belong to
                # stage 2, which gates purely on the stricter appearance cut
                if ious[i, j] >= IOU_MATCH_THRESHOLD * 0.5 and cost[i, j] <= 0.82:
                    tid = track_ids[i]
                    update_track(tid, j)
                    matched_track_ids.add(tid)
                    matched_det_idxs.add(j)

        # stage 2: reconnection rescue for whoever's left - the player who was
        # screened, went behind someone, and re-emerged somewhere their box no
        # longer overlaps. Two ways in, because these cases split:
        #   - looks unmistakably like them  -> reconnect regardless of distance
        #   - looks plausibly like them AND is plausibly where they'd have got
        #     to by now -> reconnect too. This second gate is the one that
        #     catches screen exits, which land in the dead zone between "clearly
        #     the same person" and "clearly somewhere else".
        if embedding_model is not None:
            leftover_track_ids = [tid for tid in track_ids if tid not in matched_track_ids]
            leftover_det_idxs = [j for j in range(len(dets)) if j not in matched_det_idxs]
            if leftover_track_ids and leftover_det_idxs:
                cost = np.zeros((len(leftover_track_ids), len(leftover_det_idxs)))
                allowed = np.zeros_like(cost, dtype=bool)
                for i, tid in enumerate(leftover_track_ids):
                    t = tracks[tid]
                    tb = t["box"]
                    tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
                    # how far could they plausibly have travelled while hidden?
                    # scaled by their own box height so it works at any distance
                    # from the camera, and grows the longer they've been missing.
                    reach = (tb[3] - tb[1]) * (0.6 + 0.5 * t["missed"])
                    for k, j in enumerate(leftover_det_idxs):
                        d = appearance_distance(t.get("gallery"), det_embeddings[j])
                        if d == float("inf"):
                            cost[i, k] = 1e6
                            continue
                        db = dets[j]
                        dcx, dcy = (db[0] + db[2]) / 2, (db[1] + db[3]) / 2
                        dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
                        near = dist <= reach
                        cost[i, k] = d + (0.0 if near else 0.5)  # prefer plausible positions
                        allowed[i, k] = (d <= APPEARANCE_MATCH_THRESHOLD) or \
                                        (near and d <= args.reconnect_max)
                row_idx, col_idx = linear_sum_assignment(cost)
                for i, k in zip(row_idx, col_idx):
                    if allowed[i, k]:
                        tid = leftover_track_ids[i]
                        j = leftover_det_idxs[k]
                        update_track(tid, j)
                        matched_track_ids.add(tid)
                        matched_det_idxs.add(j)

        for tid in track_ids:
            if tid not in matched_track_ids:
                t = tracks[tid]
                t["missed"] += 1
                # coast a hidden player's box along their last known motion
                # (damped + capped) - a screened player keeps moving, and a
                # frozen box ends up nowhere near them when they re-emerge
                vx, vy = (0.0, 0.0) if args.no_coast else t.get("vel", (0.0, 0.0))
                speed = (vx * vx + vy * vy) ** 0.5
                if speed > COAST_MAX_SPEED:
                    vx, vy = vx * COAST_MAX_SPEED / speed, vy * COAST_MAX_SPEED / speed
                damp = COAST_DECAY ** t["missed"]
                bx = t["box"]
                t["box"] = [bx[0] + vx * damp, bx[1] + vy * damp,
                            bx[2] + vx * damp, bx[3] + vy * damp]

        # diagnostic: a track that dies while sitting on top of another track
        # is almost certainly a screen/contact break, not someone leaving the
        # court. Total-ID count can't see these; this can.
        for tid, t in tracks.items():
            if t["missed"] > max_missed_frames:
                if any(o is not t and box_iou(t["box"], o["box"]) > 0.2 for o in tracks.values()):
                    deaths_in_contact += 1

        tracks = {tid: t for tid, t in tracks.items() if t["missed"] <= max_missed_frames}

        for j, det in enumerate(dets):
            if j not in matched_det_idxs:
                # same diagnostic from the other side: a brand-new ID appearing
                # right on top of an existing player is a screen-induced split
                if any(box_iou(det, t["box"]) > 0.2 for t in tracks.values()):
                    births_in_contact += 1
                gallery = [det_embeddings[j]] if (det_embeddings[j] is not None and not crowded[j]) else []
                tracks[next_id] = {"box": det, "missed": 0, "gallery": gallery}
                next_id += 1

        # record instead of drawing: the merge pass below needs every track's
        # full timeline before anything can be rendered
        for tid, t in tracks.items():
            if t["missed"] > 0:
                continue  # only record frames where the player was actually seen
            history.setdefault(tid, {})[frame_idx] = list(t["box"])
            if t.get("gallery"):
                final_gallery[tid] = list(t["gallery"])

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"frame {frame_idx}, active tracks: {sum(1 for t in tracks.values() if t['missed']==0)}, total ids assigned so far: {next_id}")

    cap.release()
    if args.cache_dets and not cache_hit:
        with open(args.cache_dets, "wb") as f:
            pickle.dump(det_cache, f)
        print(f"cached detections for {len(det_cache)} frames -> {args.cache_dets}")
    print(f"tracking pass done. {frame_idx} frames, {next_id} raw ids assigned.")
    print(f"rejected two-person boxes: {dropped_shape} by shape, {dropped_wrapper} as wrappers")
    print(f"contact-related breaks (pre-merge): {births_in_contact} new ids born in contact, "
          f"{deaths_in_contact} tracks died in contact")

    if args.no_merge:
        mapping = {i: i for i in history}
        merged = 0
    else:
        mapping, merged = merge_tracklets(
            history, final_gallery,
            max_overlap_frames=args.merge_max_overlap,
            appearance_max=args.merge_appearance_max,
            gap_reach_per_frame=args.merge_gap_reach,
        )
    final_ids = sorted(set(mapping.values()))
    print(f"merge pass: {merged} merges, {len(history)} -> {len(final_ids)} identities")

    # roster cap (TrackID3x3 calls this the frame-level detection limitation):
    # we know how many people can actually be out there, so keep the best-
    # supported identities and drop the stragglers - sideline staff, subs
    # standing just inside the line, and one-off detector noise.
    if args.max_identities:
        support = {}
        for tid, frames_boxes in history.items():
            support[mapping[tid]] = support.get(mapping[tid], 0) + len(frames_boxes)
        ranked = sorted(support, key=lambda i: -support[i])
        keep = set(ranked[:args.max_identities])
        dropped = [i for i in ranked[args.max_identities:]]
        print(f"roster cap: keeping {len(keep)} identities, dropping {len(dropped)} "
              f"weakest (frames seen: {[support[i] for i in dropped][:10]}...)")
        history = {tid: fb for tid, fb in history.items() if mapping[tid] in keep}
        final_ids = sorted(keep)

    # render pass: second read of the video, drawing the merged identities
    merged_history = {}
    for tid, frames_boxes in history.items():
        dest = merged_history.setdefault(mapping[tid], {})
        for f, box in frames_boxes.items():
            dest.setdefault(f, box)
    by_frame = {}
    for tid, frames_boxes in merged_history.items():
        for f, box in frames_boxes.items():
            by_frame.setdefault(f, []).append((tid, box))

    cap = cv2.VideoCapture(args.video)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = frame
        for tid, box in by_frame.get(fi, []):
            x1, y1, x2, y2 = [int(v) for v in box]
            color = id_color(tid)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"#{tid}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        writer.write(out)
        fi += 1
    cap.release()
    writer.release()
    print(f"done. saved -> {out_path}")


if __name__ == "__main__":
    main()
