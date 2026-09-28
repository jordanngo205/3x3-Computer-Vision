"""
v7 - offline gap interpolation + outlier-gated assignment, replacing v3-v6's
forward-only velocity extrapolation.

WHY THIS CHANGE: every established online tracker (SORT/DeepSORT/ByteTrack/BoT-SORT)
extrapolates forward from a Kalman-filtered velocity, because in a REAL broadcast or
live feed, future frames don't exist yet - forward extrapolation is the only option.
That's not our situation. We already have the entire clip in memory before we track
a single frame. When Canada #2 went missing at frame 49 and reappeared at frame 56,
v3-v6 had to guess forward from a (possibly noisy) velocity with no way to check
itself - and one noisy pair of real detections (a brief misassignment) sent it
running off past the edge of the frame before real detections dragged it back.

We already know where frame 56 is. So instead of guessing forward, INTERPOLATE
between the confirmed position before the gap and the confirmed position after it.
This can't run off to nowhere - it's mathematically anchored at both ends. This is
the same idea sports-tracking research calls "global tracklet association" (using
information from both directions, not just the past, to fix gaps) - the natural
fit for offline/batch analysis instead of live tracking.

Second fix: outlier-gated assignment. A single-frame misassignment (grabbing a
nearby wrong detection) is what poisoned the velocity estimate in v6. Here, a
candidate match is only accepted if the implied per-frame speed is physically
plausible (scaled by how many frames since the track's last confirmed sighting) -
an implausible "jump" is rejected outright and the track is left as a gap to be
interpolated later, rather than accepting bad data now.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import json

VIDEO = '/Users/jordanngo/Projects/AI live tracking/Test1.mp4'
OUT_VIDEO = '/Users/jordanngo/Projects/AI live tracking/output_annotated_test1.mp4'
OUT_JSON = '/Users/jordanngo/Projects/AI live tracking/output_tracks_test1.json'

COURT_POLY = np.array([
    (105, 300), (30, 355), (0, 400), (0, 540), (650, 540),
    (760, 430), (830, 330), (830, 260), (350, 248),
], dtype=np.int32)

def foot_on_court(fx, fy):
    return cv2.pointPolygonTest(COURT_POLY, (float(fx), float(fy)), False) >= 0

def classify_team(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return None
    w, h = x2 - x1, y2 - y1
    jx1, jx2 = x1 + int(w*0.25), x1 + int(w*0.75)
    jy1, jy2 = y1 + int(h*0.18), y1 + int(h*0.5)
    if jy2 <= jy1 or jx2 <= jx1:
        return None
    patch = frame[jy1:jy2, jx1:jx2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    total = h_.size
    white_mask = (s_ < 15) & (v_ > 110)
    blue_mask = (h_ >= 100) & (h_ <= 140) & (s_ > 50)
    white_frac = white_mask.sum() / total
    blue_frac = blue_mask.sum() / total
    if white_frac < 0.10 and blue_frac < 0.10:
        return None
    return 'Canada' if white_frac >= blue_frac else 'Romania'

def appearance_signature(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def appearance_distance(sig_a, sig_b):
    if sig_a is None or sig_b is None:
        return 0.5
    return cv2.compareHist(sig_a, sig_b, cv2.HISTCMP_BHATTACHARYYA)

def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2-x1) * (y2-y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1]); area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)

model = YOLO('/Users/jordanngo/Projects/AI live tracking/yolo11n.pt')

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
print('loaded', len(frames), 'frames')

# ---- detection: NMS raised to stop collapsing two contested players into one box,
# then de-dup same-team overlapping boxes (verified fix from v6) ----
per_frame_dets = []
results = model(frames, classes=[0], conf=0.25, iou=0.85, verbose=False)
for fi, r in enumerate(results):
    raw = []
    for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = box
        fx, fy = (x1+x2)/2, y2
        if not foot_on_court(fx, fy):
            continue
        team = classify_team(frames[fi], box)
        if team is None:
            continue
        raw.append({'box': box, 'conf': float(conf), 'team': team})
    raw.sort(key=lambda d: -d['conf'])
    keep = []
    for d in raw:
        dup = False
        for k in keep:
            if k['team'] == d['team'] and box_iou(k['box'], d['box']) > 0.55:
                dup = True
                break
        if not dup:
            keep.append(d)
    dets = []
    for d in keep:
        x1, y1, x2, y2 = d['box']
        cx, cy = (x1+x2)/2, (y1+y2)/2
        sig = appearance_signature(frames[fi], d['box'])
        dets.append({'box': d['box'], 'team': d['team'], 'center': (cx, cy), 'sig': sig})
    per_frame_dets.append(dets)
print('detections after on-court+team+dedup filter, frame 0:', len(per_frame_dets[0]))

APPEARANCE_WEIGHT = 150.0
APPEARANCE_EMA = 0.3
JUMP_REJECT_COST = 1e6       # effectively "never accept this match"

# Two-tier reacquisition gate. Distance alone is only trustworthy for a SHORT gap (a
# player can't have gone far); for a LONG gap it stops meaning much (a real player can
# easily be 300px away after 2+ seconds missing) while appearance stays just as valid
# regardless of how long they were gone - a blue jersey is still blue. So: short gaps
# are gated on position alone (cheap, reliable at short range); long gaps get a much
# bigger search radius but ONLY if the appearance signature is a strong, specific match
# - not just "closest available", which is what let v7's flat 100px cap either wrongly
# grab a distant stationary player (too loose) or permanently freeze a track that had
# genuinely moved on (too strict). This replaces both single-number failure modes.
SHORT_GAP_FRAMES = 8
SHORT_RANGE_DIST = 80.0          # px - position-only gate for a recently-lost track
# BUG (found by direct physics check): a flat "long-range" distance cap is the same
# unbounded-reach mistake as before, just moved to the long-gap tier - it let a track
# accept a 284px jump after only a 9-frame gap, which implies ~14 m/s, faster than any
# human sprints. The allowance MUST keep scaling with elapsed time even in the
# appearance-gated tier; only the per-frame rate and the requirement of a strong
# appearance match change for long gaps, not "distance stops mattering."
LONG_RANGE_SPEED_PER_FRAME = 18.0  # px/frame - realistic sprint pace at this court scale
LONG_RANGE_DIST_CAP = 350.0        # absolute ceiling for genuinely long gaps (~1s+)
LONG_RANGE_APPEARANCE_MAX = 0.35   # Bhattacharyya distance - must look convincingly similar

class Track:
    def __init__(self, tid, team, frame_idx, d):
        self.id = tid
        self.team = team
        self.real_boxes = {frame_idx: d['box']}   # only CONFIRMED detections - gaps filled later
        self.last_real_frame = frame_idx
        self.last_real_center = d['center']
        self.signature = d['sig']

    def update(self, frame_idx, d):
        self.real_boxes[frame_idx] = d['box']
        self.last_real_frame = frame_idx
        self.last_real_center = d['center']
        if d['sig'] is not None:
            if self.signature is None:
                self.signature = d['sig']
            else:
                self.signature = APPEARANCE_EMA * d['sig'] + (1 - APPEARANCE_EMA) * self.signature

PLAYERS_PER_TEAM = 3
tracks_by_team = {'Canada': [], 'Romania': []}
next_id = [0]
seed_frame = {'Canada': None, 'Romania': None}
for fi, dets in enumerate(per_frame_dets):
    for team in ('Canada', 'Romania'):
        if seed_frame[team] is None:
            team_dets = [d for d in dets if d['team'] == team]
            if len(team_dets) >= PLAYERS_PER_TEAM:
                seed_frame[team] = fi
                for d in team_dets[:PLAYERS_PER_TEAM]:
                    nt = Track(next_id[0], team, fi, d); next_id[0] += 1
                    tracks_by_team[team].append(nt)

for fi, dets in enumerate(per_frame_dets):
    for team in ('Canada', 'Romania'):
        if seed_frame[team] is None or fi <= seed_frame[team]:
            continue
        team_dets = [d for d in dets if d['team'] == team]
        tracks = tracks_by_team[team]
        if not team_dets:
            continue  # no candidates at all this frame - leave every track as a gap
        cost = np.zeros((len(tracks), len(team_dets)))
        for i, t in enumerate(tracks):
            gap = fi - t.last_real_frame
            for j, d in enumerate(team_dets):
                dx = t.last_real_center[0]-d['center'][0]; dy = t.last_real_center[1]-d['center'][1]
                pos_dist = (dx*dx+dy*dy) ** 0.5
                app_dist = appearance_distance(t.signature, d['sig'])
                base_cost = pos_dist + APPEARANCE_WEIGHT * app_dist
                if gap <= SHORT_GAP_FRAMES:
                    plausible = pos_dist <= SHORT_RANGE_DIST
                else:
                    allowed_long = min(LONG_RANGE_SPEED_PER_FRAME * gap, LONG_RANGE_DIST_CAP)
                    plausible = pos_dist <= allowed_long and app_dist <= LONG_RANGE_APPEARANCE_MAX
                cost[i, j] = base_cost if plausible else JUMP_REJECT_COST
        row, col = linear_sum_assignment(cost)
        for r_, c_ in zip(row, col):
            if cost[r_, c_] < JUMP_REJECT_COST:
                tracks[r_].update(fi, team_dets[c_])
            # else: rejected - track stays a gap this frame, filled by interpolation later

all_tracks = tracks_by_team['Canada'] + tracks_by_team['Romania']
print('tracks:', len(all_tracks))
for t in all_tracks:
    print(f'  #{t.id} {t.team}: {len(t.real_boxes)} real detections, span {min(t.real_boxes)}-{max(t.real_boxes)}')

# ---- gap recovery pass ----
# WHY: gaps happen because conf=0.25 is a global threshold tuned to avoid noise across
# EVERY frame. But for a frame we already know is inside a confirmed gap (real detection
# before AND after it), we have strong priors a blanket threshold can't use: roughly
# where the player should be (interpolated from the confirmed endpoints) and what they
# look like (the track's appearance signature). Re-searching just those frames at a much
# lower confidence, but only accepting a hit that's both near the expected spot AND a
# strong appearance match, recovers real signal a global threshold left on the table -
# without adding noise anywhere else, since this only ever runs on already-known gaps.
RECOVERY_CONF = 0.08
RECOVERY_POS_TOL = 70.0
RECOVERY_APPEARANCE_MAX = 0.3
recovered_total = 0
for t in all_tracks:
    real_frames = sorted(t.real_boxes.keys())
    for i in range(len(real_frames)-1):
        fa, fb = real_frames[i], real_frames[i+1]
        if fb - fa <= 1:
            continue
        ba, bb = np.array(t.real_boxes[fa]), np.array(t.real_boxes[fb])
        for f in range(fa+1, fb):
            frac = (f - fa) / (fb - fa)
            expected_box = ba + (bb - ba) * frac
            expected_center = ((expected_box[0]+expected_box[2])/2, (expected_box[1]+expected_box[3])/2)
            r = model(frames[f], classes=[0], conf=RECOVERY_CONF, iou=0.85, verbose=False)[0]
            best = None
            for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                x1,y1,x2,y2 = box
                cx,cy = (x1+x2)/2, (y1+y2)/2
                pos_dist = ((cx-expected_center[0])**2 + (cy-expected_center[1])**2) ** 0.5
                if pos_dist > RECOVERY_POS_TOL:
                    continue
                team = classify_team(frames[f], box)
                if team != t.team:
                    continue
                sig = appearance_signature(frames[f], box)
                app_dist = appearance_distance(t.signature, sig)
                if app_dist > RECOVERY_APPEARANCE_MAX:
                    continue
                score = pos_dist + APPEARANCE_WEIGHT * app_dist
                if best is None or score < best[0]:
                    best = (score, box, sig)
            if best is not None:
                t.real_boxes[f] = best[1]
                recovered_total += 1
print('gap frames recovered (low-confidence YOLO pass):', recovered_total)

# ---- CAMELTrack cross-reference: tried, measured, reverted ----
# Tested three integration strategies (independent per-frame position check, trust a
# CamelTrack identity for a whole gap once validated at both edges, edge-validated with
# a widened per-frame check). Measured against the full pixel-level audit, EVERY version
# made total flagged issues WORSE, not better (94 with none of this -> 97 -> 113 -> 113).
# CamelTrack is real signal in isolated spot checks (it correctly separated two tangled
# players at our single hardest frame), but its own internal identity can also silently
# drift during occlusion - proven directly: a candidate that matched our confirmed
# detection at BOTH edges of a 41-frame gap (10-12px position match, high confidence)
# turned out to be sitting on a different, wrong-colored player by the middle of that
# same span. No tolerance tuning fixed this without readmitting the original problem.
# Conclusion: our own pipeline alone, on this clip, outperforms every tested way of
# blending in CamelTrack's output. Not pursuing further without a fundamentally
# different integration idea.

# ---- offline gap-filling: interpolate between CONFIRMED real detections instead of
# extrapolating forward - this is the actual fix for the "ran off to x=1351" bug ----
def fill_track(t, n_frames):
    real_frames = sorted(t.real_boxes.keys())
    filled = {}
    predicted = set()
    for f in real_frames:
        filled[f] = t.real_boxes[f]
    # before the first real detection and after the last: hold steady, don't guess a direction
    for f in range(0, real_frames[0]):
        filled[f] = t.real_boxes[real_frames[0]]
        predicted.add(f)
    for f in range(real_frames[-1]+1, n_frames):
        filled[f] = t.real_boxes[real_frames[-1]]
        predicted.add(f)
    # between two confirmed detections: linear interpolation, anchored at both ends
    for i in range(len(real_frames)-1):
        fa, fb = real_frames[i], real_frames[i+1]
        if fb - fa <= 1:
            continue
        ba, bb = np.array(t.real_boxes[fa]), np.array(t.real_boxes[fb])
        for f in range(fa+1, fb):
            frac = (f - fa) / (fb - fa)
            filled[f] = tuple(ba + (bb - ba) * frac)
            predicted.add(f)
    return filled, predicted

def smooth_track(filled, n_frames, window=5):
    """Light moving-average smoothing on the final box sequence. WHY: diagnosed the
    'erratic gliding' complaint down to real per-frame detector jitter - two near-
    duplicate candidate boxes for the SAME real player, just under the dedup IoU
    threshold, alternately winning the assignment each frame. That's real noise in
    the raw boxes, not a wrong match, so it needs smoothing, not a matching fix."""
    arr = np.array([filled[f] for f in range(n_frames)])
    smoothed = arr.copy()
    half = window // 2
    for f in range(n_frames):
        lo, hi = max(0, f-half), min(n_frames, f+half+1)
        smoothed[f] = arr[lo:hi].mean(axis=0)
    return {f: tuple(smoothed[f]) for f in range(n_frames)}

MIN_REAL_DETECTIONS = 15
MIN_MOVEMENT = 40
kept = []
for t in all_tracks:
    if len(t.real_boxes) < MIN_REAL_DETECTIONS:
        continue
    feet = [((b[0]+b[2])/2, b[3]) for b in t.real_boxes.values()]
    xs = [p[0] for p in feet]; ys = [p[1] for p in feet]
    spread = ((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2) ** 0.5
    if spread < MIN_MOVEMENT:
        print(f'  dropping #{t.id} {t.team}: spread={spread:.0f}px (likely stationary ref/staff)')
        continue
    filled, predicted = fill_track(t, len(frames))
    filled = smooth_track(filled, len(frames))
    kept.append((t, filled, predicted))
print('kept tracks:', len(kept))
for t, filled, predicted in kept:
    print(f'  #{t.id} {t.team}: {len(filled)} frames ({len(predicted)} interpolated/held)')

TEAM_COLORS = {'Canada': (255,255,255), 'Romania': (255,140,0)}
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))
for fi, frame in enumerate(frames):
    out = frame.copy()
    for t, filled, predicted in kept:
        color = TEAM_COLORS[t.team]
        x1,y1,x2,y2 = [int(v) for v in filled[fi]]
        if fi in predicted:
            for xx in range(x1, x2, 12):
                cv2.line(out, (xx,y1), (min(xx+6,x2),y1), color, 2)
                cv2.line(out, (xx,y2), (min(xx+6,x2),y2), color, 2)
            for yy in range(y1, y2, 12):
                cv2.line(out, (x1,yy), (x1,min(yy+6,y2)), color, 2)
                cv2.line(out, (x2,yy), (x2,min(yy+6,y2)), color, 2)
        else:
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, f'{t.team} #{t.id}', (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    writer.write(out)
writer.release()
print('saved', OUT_VIDEO)

with open(OUT_JSON, 'w') as f:
    json.dump({
        'fps': fps, 'width': w, 'height': h, 'n_frames': n_frames,
        'tracks': {str(t.id): {str(k): [float(x) for x in v] for k,v in filled.items()} for t,filled,pred in kept},
        'predicted': {str(t.id): sorted(int(x) for x in pred) for t,filled,pred in kept},
        'track_team': {str(t.id): t.team for t,filled,pred in kept},
    }, f)
print('saved', OUT_JSON)
