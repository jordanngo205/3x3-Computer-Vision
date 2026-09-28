"""
v5 - adds appearance-based Re-ID on top of v3's team-color-first tracking.

WHY: v3 matches each frame's detections to existing tracks using only PREDICTED
POSITION (constant-velocity extrapolation). That's fine for brief gaps, but when a
player is fully hidden behind teammates for ~20+ frames, the prediction is just a
guess with no way to say "that's clearly not who I'm tracking" - it grabs whatever
detection is nearest, even if it's a different, wrong-colored player. That's the
exact "Romania #3 drifts onto a white jersey" bug from v3/v4.

THE FIX (this is the standard technique - DeepSORT/BoT-SORT/TrackID3x3's BoT-SORT-ReID
all do a version of this): give the tracker a SECOND, independent signal alongside
position - "does this candidate detection actually LOOK like the player I'm tracking?"
Each track keeps a running visual fingerprint (here: an HSV color histogram of the
whole person crop, not just the jersey - it also captures hair color, skin tone, shoe
color, which is exactly the leftover signal that distinguishes two same-team players
once team color alone can't). Matching cost becomes position distance blended with
appearance distance, not position alone, so a wrong-colored player loses the match
even if it happens to be spatially closest.

Why a color histogram instead of a deep CNN embedding (what BoT-SORT-ReID actually
uses): our jerseys already separate players by team; the leftover per-player signal
(hair/skin/shoe) is exactly the kind of thing a cheap histogram captures well, with
no model download or GPU inference needed. A deep embedding would be more robust to
lighting/pose changes - the natural next upgrade once this simpler version is proven
to help, not the first thing to reach for.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import json

VIDEO = '/Users/jordanngo/Projects/AI live tracking/video.mp4'
OUT_VIDEO = '/Users/jordanngo/Projects/AI live tracking/output_annotated_v5.mp4'
OUT_JSON = '/Users/jordanngo/Projects/AI live tracking/output_tracks_v5.json'

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
    """HSV hue+saturation histogram over the WHOLE person crop (not just the torso
    band used for team color) - hair/skin/shoe pixels included on purpose, since
    those are exactly what still differs between two players on the same team."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # H:0-180, S:0-256 - ignore V (brightness) so shadow/highlight doesn't count as
    # a different "appearance"
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def appearance_distance(sig_a, sig_b):
    """Bhattacharyya distance: 0 = identical-looking, 1 = totally different. Bounded
    and comparable across frames, unlike a raw pixel difference."""
    if sig_a is None or sig_b is None:
        return 0.5  # unknown - neutral, let position decide
    return cv2.compareHist(sig_a, sig_b, cv2.HISTCMP_BHATTACHARYYA)

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

per_frame_dets = []
results = model(frames, classes=[0], conf=0.25, verbose=False)
for fi, r in enumerate(results):
    dets = []
    for box in r.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        fx, fy = (x1+x2)/2, y2
        if not foot_on_court(fx, fy):
            continue
        team = classify_team(frames[fi], box)
        if team is None:
            continue
        cx, cy = (x1+x2)/2, (y1+y2)/2
        sig = appearance_signature(frames[fi], box)
        dets.append({'box': box, 'team': team, 'foot': (fx, fy), 'center': (cx, cy), 'sig': sig})
    per_frame_dets.append(dets)
print('detections after on-court+team filter, frame 0:', len(per_frame_dets[0]))

# position distance (px) and appearance distance (0-1) are different units - this is
# "how many pixels of position-cost is one full unit of appearance-mismatch worth."
# Set high enough that a clearly-wrong-looking player loses even if it's the closest
# detection, but not so high that a bit of lighting noise overrides real position info.
APPEARANCE_WEIGHT = 150.0
APPEARANCE_EMA = 0.3  # how fast a track's fingerprint adapts to new real detections

class Track:
    def __init__(self, tid, team, frame_idx, d):
        self.id = tid
        self.team = team
        self.boxes = {frame_idx: d['box']}
        self.predicted = set()
        self.last_frame = frame_idx
        self.last_center = d['center']
        self.velocity = (0.0, 0.0)
        self.signature = d['sig']

    def predict_center(self):
        return (self.last_center[0] + self.velocity[0], self.last_center[1] + self.velocity[1])

    def update(self, frame_idx, d):
        center = d['center']
        dt = frame_idx - self.last_frame
        if dt > 0:
            self.velocity = ((center[0]-self.last_center[0])/dt, (center[1]-self.last_center[1])/dt)
        self.boxes[frame_idx] = d['box']
        self.last_center = center
        self.last_frame = frame_idx
        if d['sig'] is not None:
            if self.signature is None:
                self.signature = d['sig']
            else:
                # exponential moving average - the fingerprint drifts slowly with the
                # player (different angle/lighting) but isn't overwritten by one noisy frame
                self.signature = APPEARANCE_EMA * d['sig'] + (1 - APPEARANCE_EMA) * self.signature

    def mark_missed(self, frame_idx):
        pc = self.predict_center()
        w_prev = self.boxes[self.last_frame]
        bw, bh = w_prev[2]-w_prev[0], w_prev[3]-w_prev[1]
        box = (pc[0]-bw/2, pc[1]-bh/2, pc[0]+bw/2, pc[1]+bh/2)
        self.boxes[frame_idx] = box
        self.predicted.add(frame_idx)
        self.last_center = pc
        self.velocity = (self.velocity[0]*0.8, self.velocity[1]*0.8)
        # signature is left untouched during a miss - we have no real pixels to learn
        # from, and updating it from a possibly-wrong guess would poison the fingerprint

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
            for t in tracks:
                t.mark_missed(fi)
            continue
        cost = np.zeros((len(tracks), len(team_dets)))
        for i, t in enumerate(tracks):
            pc = t.predict_center()
            for j, d in enumerate(team_dets):
                dx = pc[0]-d['center'][0]; dy = pc[1]-d['center'][1]
                pos_dist = (dx*dx+dy*dy) ** 0.5
                app_dist = appearance_distance(t.signature, d['sig'])
                cost[i, j] = pos_dist + APPEARANCE_WEIGHT * app_dist
        row, col = linear_sum_assignment(cost)
        matched_tracks = set()
        for r_, c_ in zip(row, col):
            tracks[r_].update(fi, team_dets[c_])
            matched_tracks.add(r_)
        for i, t in enumerate(tracks):
            if i not in matched_tracks:
                t.mark_missed(fi)

all_tracks = tracks_by_team['Canada'] + tracks_by_team['Romania']
print('raw tracks built:', len(all_tracks))

MIN_FRAMES = 20
MIN_MOVEMENT = 40
kept = []
for t in all_tracks:
    if len(t.boxes) < MIN_FRAMES:
        continue
    feet = [((b[0]+b[2])/2, b[3]) for b in t.boxes.values()]
    xs = [p[0] for p in feet]; ys = [p[1] for p in feet]
    spread = ((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2) ** 0.5
    if spread < MIN_MOVEMENT:
        print(f'  dropping #{t.id} {t.team}: spread={spread:.0f}px (likely stationary ref/staff)')
        continue
    kept.append(t)
print('kept tracks:', len(kept))
for t in kept:
    print(f'  #{t.id} {t.team}: {len(t.boxes)} frames ({len(t.predicted)} predicted), span {min(t.boxes)}-{max(t.boxes)}')

TEAM_COLORS = {'Canada': (255,255,255), 'Romania': (255,140,0)}
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))
for fi, frame in enumerate(frames):
    out = frame.copy()
    for t in kept:
        if fi not in t.boxes:
            continue
        color = TEAM_COLORS[t.team]
        x1,y1,x2,y2 = [int(v) for v in t.boxes[fi]]
        style_gap = 6 if fi in t.predicted else 0
        if style_gap:
            for xx in range(x1, x2, style_gap*2):
                cv2.line(out, (xx,y1), (min(xx+style_gap,x2),y1), color, 2)
                cv2.line(out, (xx,y2), (min(xx+style_gap,x2),y2), color, 2)
            for yy in range(y1, y2, style_gap*2):
                cv2.line(out, (x1,yy), (x1,min(yy+style_gap,y2)), color, 2)
                cv2.line(out, (x2,yy), (x2,min(yy+style_gap,y2)), color, 2)
        else:
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, f'{t.team} #{t.id}', (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    writer.write(out)
writer.release()
print('saved', OUT_VIDEO)

with open(OUT_JSON, 'w') as f:
    json.dump({
        'fps': fps, 'width': w, 'height': h, 'n_frames': n_frames,
        'tracks': {str(t.id): {str(k): [float(x) for x in v] for k,v in t.boxes.items()} for t in kept},
        'predicted': {str(t.id): sorted(int(x) for x in t.predicted) for t in kept},
        'track_team': {str(t.id): t.team for t in kept},
    }, f)
print('saved', OUT_JSON)
