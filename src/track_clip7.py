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
from sklearn.cluster import KMeans
import json
import sys
import argparse

BASE = '/Users/jordanngo/Projects/AI live tracking'

parser = argparse.ArgumentParser()
parser.add_argument('--video', default=f'{BASE}/video.mp4')
parser.add_argument('--tag', default='v7', help='suffix for output files, e.g. test1 -> output_annotated_test1.mp4')
parser.add_argument('--court-poly', default=None,
                     help='comma-separated x,y pairs overriding the default court polygon '
                          '(needed when frame resolution/camera framing differs from video.mp4)')
parser.add_argument('--team-white', default='Canada', help='name for the white-jersey team')
parser.add_argument('--team-color', default='Romania', help='name for the colored-jersey team (any color, not just blue)')
parser.add_argument('--max-width', type=int, default=None,
                     help='downscale frames to this width before processing (preserves aspect ratio). '
                          'Needed for high-res/long clips - holding thousands of full-res frames plus '
                          'YOLO results in RAM at once caused a silent OOM kill on a 4216-frame, '
                          '1986x1118 clip (would have needed ~28GB just for raw pixels).')
args = parser.parse_args()

VIDEO = args.video
OUT_VIDEO = f'{BASE}/output_annotated_{args.tag}.mp4'
OUT_JSON = f'{BASE}/output_tracks_{args.tag}.json'
TEAM_WHITE = args.team_white
TEAM_COLOR = args.team_color

if args.court_poly:
    nums = [float(v) for v in args.court_poly.split(',')]
    COURT_POLY = np.array(list(zip(nums[0::2], nums[1::2])), dtype=np.int32)
else:
    COURT_POLY = np.array([
        (105, 300), (30, 355), (0, 400), (0, 540), (650, 540),
        (760, 430), (830, 330), (830, 260), (350, 248),
    ], dtype=np.int32)

def foot_on_court(fx, fy):
    return cv2.pointPolygonTest(COURT_POLY, (float(fx), float(fy)), False) >= 0

def classify_team_hsv(frame, box):
    """Old fixed-rule classifier. Too fragile to use per-detection (a bent-over/angled
    pose can shift the sampled patch enough to drop under the white threshold - this is
    exactly what caused a real player to go untracked for 48 straight frames earlier).
    Kept only as a bootstrap: used to NAME the two clusters KMeans finds (see
    color_histogram/cluster_team below), never to classify an individual detection.

    Colored-team mask is hue-agnostic (any saturated, reasonably bright fabric that isn't
    white) rather than a fixed blue range - this rule was originally written for Canada
    (white) vs Romania (blue), but the same two-cluster bootstrap needs to work for any
    single-colored opponent jersey (e.g. Japan's red), since only ONE color needs telling
    apart from white in any given clip."""
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
    # circular hue distance from the auto-detected opponent color (set by
    # detect_opponent_hue below, before this function is ever called) - NOT "any
    # saturated pixel": that first attempt also fired on the white team's own trim/logo
    # colors (e.g. Canada's red maple leaf), which silently merged both clusters into one
    # team. Restricting to a hue band around the one color that's actually dominant across
    # the whole clip's "colored" crops avoids that.
    hue_dist = np.minimum(np.abs(h_.astype(int) - OPPONENT_HUE_CENTER[0]), 180 - np.abs(h_.astype(int) - OPPONENT_HUE_CENTER[0]))
    color_mask = (hue_dist <= HUE_HALF_WIDTH) & (s_ > 50) & (v_ > 60)
    white_frac = white_mask.sum() / total
    color_frac = color_mask.sum() / total
    if white_frac < 0.10 and color_frac < 0.10:
        return None
    return TEAM_WHITE if white_frac >= color_frac else TEAM_COLOR

HUE_HALF_WIDTH = 20
OPPONENT_HUE_CENTER = [110]  # mutable holder, overwritten by detect_opponent_hue() below

def detect_opponent_hue(dets, frames, sample=400, video_path=None, resize_scale=1.0):
    """Auto-detect the opponent team's dominant jersey hue for THIS clip, instead of
    assuming blue - generalizes classify_team_hsv to any single opponent color (tested
    against Romania/blue and Japan/red) without hand-tuning a hue range per video.

    BUG #1 (found via direct regression check on video.mp4): sampling the wide torso
    window (25-75%/18-50%) pulled in bare skin, which outvoted the real jersey color.
    Tightening to the 35-65%/15-42% jersey-only window (matching color_histogram) didn't
    fully fix it either.

    BUG #2 (found by printing raw hue/saturation per box): counting every "saturated,
    non-white" PIXEL regardless of source means a small but vivid accent - Canada's red
    maple-leaf logo, a few percent of the crop - adds just as many hue votes as an actual
    solid-blue jersey, because pixel-counting has no notion of "which color this garment
    actually is." Fix: vote per DETECTION, not per pixel - only count a hue if it's the
    MAJORITY color of that specific crop (>35% of the crop's total area). A small logo
    can never win that vote; a real jersey filling most of the torso crop always can.

    BUG #3 (found on real test.mov, downscaled to 640px wide to fit in memory): at that
    scale a player's torso crop is only a few hundred pixels, too small and noisy to
    reliably clear the 35% single-hue-dominance bar even for a real solid jersey - every
    sampled detection got skipped, hue_votes stayed empty, and it silently fell back to
    the blue default even though the actual opponent (Japan) wears red. Fix: if given the
    original video path, re-read just these ~400 sampled detections directly from disk at
    FULL original resolution (cheap - it's a one-time sample, not the whole clip) instead
    of using the downscaled in-memory frames used everywhere else in the pipeline."""
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(dets), size=min(sample, len(dets)), replace=False)
    hue_votes = np.zeros(180, dtype=np.int64)
    full_res_cap = None
    if video_path is not None and resize_scale != 1.0:
        full_res_cap = cv2.VideoCapture(video_path)
    for i in idxs:
        d = dets[i]
        box = d['box']
        frame = frames[d['fi']]
        if full_res_cap is not None:
            full_res_cap.set(cv2.CAP_PROP_POS_FRAMES, d['fi'])
            ok, full_frame = full_res_cap.read()
            if ok:
                frame = full_frame
                box = [v / resize_scale for v in box]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
        if x2 <= x1 or y2 <= y1:
            continue
        w, h = x2 - x1, y2 - y1
        jx1, jx2 = x1 + int(w*0.35), x1 + int(w*0.65)
        jy1, jy2 = y1 + int(h*0.15), y1 + int(h*0.42)
        if jy2 <= jy1 or jx2 <= jx1:
            continue
        patch = frame[jy1:jy2, jx1:jx2]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h_, s_, v_ = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        total_px = h_.size
        colored = (s_ > 50) & (v_ > 60)
        if colored.sum() / total_px < 0.35:
            continue  # no single color dominates this crop - not a useful vote
        crop_hist = np.bincount(h_[colored].ravel(), minlength=180)
        # circularly smooth BEFORE picking the peak - found via direct debugging on real
        # test.mov (Japan/red): a red garment's hue straddles the 0/180 wraparound, so its
        # pixels split across bins near 179 AND bins near 0-5. Taking the raw single-bin
        # argmax saw ~28% in one bin and ~20% in the other, neither clearing 35%
        # dominance, even though together they're clearly ~48% of the crop, i.e. one real
        # dominant color. Blue (this check's only tested case before) doesn't straddle the
        # boundary, which is why this passed regression testing before but broke on red.
        crop_kernel = 5
        crop_padded = np.concatenate([crop_hist[-crop_kernel:], crop_hist, crop_hist[:crop_kernel]])
        crop_smoothed = np.convolve(crop_padded, np.ones(2*crop_kernel+1), mode='valid')
        peak_bin = int(np.argmax(crop_smoothed))
        if crop_smoothed[peak_bin] / total_px < 0.35:
            continue  # the colored pixels themselves are hue-scattered, not one solid color
        hue_votes[peak_bin] += 1
    if full_res_cap is not None:
        full_res_cap.release()
    if hue_votes.sum() < 10:
        return 110  # fallback: blue, the original default
    # smooth circularly (hue wraps at 0/180) before taking the peak, so a color like red
    # (which straddles the 0/180 boundary) isn't split into two weaker peaks
    kernel = 5
    padded = np.concatenate([hue_votes[-kernel:], hue_votes, hue_votes[:kernel]])
    smoothed = np.convolve(padded, np.ones(2*kernel+1)/(2*kernel+1), mode='valid')
    return int(np.argmax(smoothed))

def color_histogram(frame, box):
    """Direct HSV color histogram of the jersey crop - matches TrackID3x3's own
    attribute-identification approach (calculate_color_hist.py). Validated at 90.1%
    agreement with the HSV ground truth on this clip, vs 67.4% for a general-purpose
    SigLIP embedding + clustering - a general vision model doesn't prioritize color the
    way a plain histogram forces the clustering to.
    Sampling window tightened (35-65% width, 15-42% height, vs the original 25-75%/
    18-50%) after finding that during player-on-player occlusion, the wider window
    picked up contaminating jersey color from the overlapping opponent, causing real
    misclassifications on otherwise-clean detections. Verified this fixes both known
    occlusion-contaminated frames and raises overall agreement to 91.4%."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return None
    w, h = x2 - x1, y2 - y1
    jx1, jx2 = x1 + int(w*0.35), x1 + int(w*0.65)
    jy1, jy2 = y1 + int(h*0.15), y1 + int(h*0.42)
    if jy2 <= jy1 or jx2 <= jx1:
        return None
    patch = frame[jy1:jy2, jx1:jx2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()

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

RESIZE_SCALE = 1.0
if args.max_width and w > args.max_width:
    RESIZE_SCALE = args.max_width / w
    w, h = args.max_width, round(h * RESIZE_SCALE)
    if args.court_poly:
        COURT_POLY = np.round(COURT_POLY * RESIZE_SCALE).astype(np.int32)
    print(f'downscaling frames by {RESIZE_SCALE:.3f} -> {w}x{h}')

frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    if RESIZE_SCALE != 1.0:
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
    frames.append(f)
cap.release()
print('loaded', len(frames), 'frames')

# ---- detection: NMS raised to stop collapsing two contested players into one box.
# Batched in chunks rather than one call over the whole clip - ultralytics accumulates
# all results in RAM for a single batched call, which combined with holding every frame
# in memory is what silently OOM-killed the process on a 4216-frame clip. ----
DETECT_BATCH = 200
results = []
for bstart in range(0, len(frames), DETECT_BATCH):
    results.extend(model(frames[bstart:bstart+DETECT_BATCH], classes=[0], conf=0.25, iou=0.85, verbose=False))

# ---- Pass 1: collect every on-court detection's color histogram across the WHOLE
# clip, with no team decision yet. Team assignment now happens once, globally, via
# clustering - not per-detection with one fixed rule - so a single bent-over/angled
# pose can no longer silently drop a real player the way classify_team_hsv did. ----
on_court_dets = []  # list of {fi, box, conf, hist}
for fi, r in enumerate(results):
    for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = box
        fx, fy = (x1+x2)/2, y2
        if not foot_on_court(fx, fy):
            continue
        hist = color_histogram(frames[fi], box)
        if hist is None:
            continue
        on_court_dets.append({'fi': fi, 'box': box, 'conf': float(conf), 'hist': hist})
print('on-court detections collected for clustering:', len(on_court_dets))

OPPONENT_HUE_CENTER[0] = detect_opponent_hue(on_court_dets, frames, video_path=VIDEO, resize_scale=RESIZE_SCALE)
print('auto-detected opponent hue center:', OPPONENT_HUE_CENTER[0], '(0=red/180, 60=green, 120=blue)')

# Fit KMeans ONLY on detections the old HSV rule is confident about - a clean subset
# (matches what validate_color_clustering.py actually tested at 90.1% agreement). If we
# fit on EVERY on-court detection instead, the many ambiguous/occlusion-contaminated
# crops (exactly the hard cases we're trying to fix) drag the cluster boundary itself to
# a worse place, which is what happened on the first attempt: 5-6 clear misclassifications
# out of 8 spot-checked, most with no occlusion excuse at all. Bootstrap-confident crops
# get used to LEARN the boundary; every on-court detection (confident or not) still gets
# CLASSIFIED with it afterwards via predict() - that's the whole point.
bootstrap_hists, bootstrap_labels = [], []
for d in on_court_dets:
    hsv_label = classify_team_hsv(frames[d['fi']], d['box'])
    if hsv_label is not None:
        bootstrap_hists.append(d['hist'])
        bootstrap_labels.append(hsv_label)
print(f'fitting clustering on {len(bootstrap_hists)} HSV-confident crops (of {len(on_court_dets)} total on-court)')

Xb = np.array(bootstrap_hists, dtype=np.float64)
km = KMeans(n_clusters=2, random_state=42, n_init=10)
bootstrap_cluster_ids = km.fit_predict(Xb)

cluster_to_team = {}
for c in (0, 1):
    votes = [bootstrap_labels[i] for i in range(len(bootstrap_labels)) if bootstrap_cluster_ids[i] == c]
    cluster_to_team[c] = max(set(votes), key=votes.count) if votes else (TEAM_WHITE if c == 0 else TEAM_COLOR)
print('cluster -> team:', cluster_to_team)

X_all = np.array([d['hist'] for d in on_court_dets], dtype=np.float64)
cluster_ids = km.predict(X_all)
for i, d in enumerate(on_court_dets):
    d['team'] = cluster_to_team[cluster_ids[i]]

def classify_team_cluster(frame, box):
    """Classify a single box via the already-fitted KMeans model - used later for the
    low-confidence gap-recovery pass, which needs a team check on newly re-detected
    candidates that weren't part of the original clustering fit."""
    hist = color_histogram(frame, box)
    if hist is None:
        return None
    cluster = km.predict(hist.reshape(1, -1).astype(np.float64))[0]
    return cluster_to_team[cluster]

# ---- Pass 2: de-dup same-team overlapping boxes (verified fix from v6), now using the
# cluster-assigned team ----
per_frame_dets = []
dets_by_frame = {}
for d in on_court_dets:
    dets_by_frame.setdefault(d['fi'], []).append(d)
for fi in range(len(frames)):
    raw = dets_by_frame.get(fi, [])
    raw = sorted(raw, key=lambda d: -d['conf'])
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
tracks_by_team = {TEAM_WHITE: [], TEAM_COLOR: []}
next_id = [0]
seed_frame = {TEAM_WHITE: None, TEAM_COLOR: None}
for fi, dets in enumerate(per_frame_dets):
    for team in (TEAM_WHITE, TEAM_COLOR):
        if seed_frame[team] is None:
            team_dets = [d for d in dets if d['team'] == team]
            if len(team_dets) >= PLAYERS_PER_TEAM:
                seed_frame[team] = fi
                for d in team_dets[:PLAYERS_PER_TEAM]:
                    nt = Track(next_id[0], team, fi, d); next_id[0] += 1
                    tracks_by_team[team].append(nt)

for fi, dets in enumerate(per_frame_dets):
    for team in (TEAM_WHITE, TEAM_COLOR):
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

all_tracks = tracks_by_team[TEAM_WHITE] + tracks_by_team[TEAM_COLOR]
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
MAX_HOLD_SECONDS = 1.5  # also used by fill_track() further below
recovered_total = 0
_recovery_max_gap = max(1, round(MAX_HOLD_SECONDS * fps))
for t in all_tracks:
    real_frames = sorted(t.real_boxes.keys())
    for i in range(len(real_frames)-1):
        fa, fb = real_frames[i], real_frames[i+1]
        # gaps longer than fill_track's own hold cap get dropped later anyway (and on a
        # long clip with scene cuts/replays, a huge gap here would mean thousands of
        # wasted extra YOLO calls searching frames that don't even show the real game feed)
        if fb - fa <= 1 or fb - fa > _recovery_max_gap:
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
                team = classify_team_cluster(frames[f], box)
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
# BUG (found by watching Test1's output, not just the aggregate audit number): a track
# that only ever got real detections in a short early window (e.g. a false-positive that
# briefly chased a cameraman/scorer's table) was being held FROZEN at that last real
# position for the rest of the entire clip - one track sat on a scorer's table as a
# labeled "player" for 313 of 330 frames. The per-track aggregate audit didn't catch this
# because a busy, textured background (people, equipment) doesn't trip the "floating over
# empty court" check, and it's outside the drift check's 20-frame radius. Root cause: gap
# filling had NO cap on gap length - a 2-frame gap and a 300-frame gap were bridged the
# same way. Fix: cap how long any gap (before the first real detection, after the last,
# or between two reals) can be bridged. Beyond the cap, the track simply isn't drawn for
# those frames - an honest "we lost her" is better than a confident-looking wrong guess.

def fill_track(t, n_frames, fps):
    max_hold = max(1, round(MAX_HOLD_SECONDS * fps))
    real_frames = sorted(t.real_boxes.keys())
    filled = {}
    predicted = set()
    for f in real_frames:
        filled[f] = t.real_boxes[f]
    # before the first real detection: hold steady, but only up to max_hold frames back
    for f in range(max(0, real_frames[0] - max_hold), real_frames[0]):
        filled[f] = t.real_boxes[real_frames[0]]
        predicted.add(f)
    # after the last real detection: same cap, don't freeze indefinitely
    for f in range(real_frames[-1] + 1, min(n_frames, real_frames[-1] + 1 + max_hold)):
        filled[f] = t.real_boxes[real_frames[-1]]
        predicted.add(f)
    # between two confirmed detections: linear interpolation, anchored at both ends -
    # but only if the gap is within the cap; a longer gap means we genuinely don't know
    # where she was, so leave those middle frames absent rather than inventing a path
    for i in range(len(real_frames)-1):
        fa, fb = real_frames[i], real_frames[i+1]
        if fb - fa <= 1 or fb - fa > max_hold:
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
    the raw boxes, not a wrong match, so it needs smoothing, not a matching fix.
    Only smooths over frames the track is actually present for - fill_track no longer
    guarantees every frame index exists (see the gap-cap fix above)."""
    present = sorted(filled.keys())
    boxes = np.array([filled[f] for f in present])
    smoothed = {}
    half = window // 2
    lo = 0
    for i, f in enumerate(present):
        while present[lo] < f - half:
            lo += 1
        hi = i
        while hi + 1 < len(present) and present[hi+1] <= f + half:
            hi += 1
        smoothed[f] = tuple(boxes[lo:hi+1].mean(axis=0))
    return smoothed

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
    filled, predicted = fill_track(t, len(frames), fps)
    filled = smooth_track(filled, len(frames))
    kept.append((t, filled, predicted))
print('kept tracks:', len(kept))
for t, filled, predicted in kept:
    print(f'  #{t.id} {t.team}: {len(filled)}/{len(frames)} frames shown ({len(predicted)} interpolated/held)')

TEAM_COLORS = {TEAM_WHITE: (255,255,255), TEAM_COLOR: (255,140,0)}
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))
for fi, frame in enumerate(frames):
    out = frame.copy()
    for t, filled, predicted in kept:
        if fi not in filled:
            continue
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

import pickle
CLASSIFIER_OUT = OUT_JSON.replace('output_tracks_', 'output_team_classifier_').replace('.json', '.pkl')
with open(CLASSIFIER_OUT, 'wb') as f:
    pickle.dump({'kmeans': km, 'cluster_to_team': cluster_to_team}, f)
print('saved', CLASSIFIER_OUT)
