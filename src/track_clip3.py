"""
v3 - custom team-color-first tracking, replacing generic ByteTrack association.
Root cause of the v2 problems: ByteTrack matches boxes by shape/position/IoU only,
so it has no idea two overlapping boxes are different-colored jerseys - that's exactly
when it drifts or drops a track. Since a player's team color is constant for the whole
clip, matching each frame's detections to existing tracks WITHIN the same team color
group first (nearest position, Hungarian assignment) is a much stronger identity signal
than generic IoU during contact.

Pipeline:
1. Plain YOLO detection per frame (no tracking) -> boxes, on-court filter, team color.
2. Per team, greedy/optimal frame-to-frame assignment by feet-position distance.
   A track can go a few frames with no matching detection (occlusion) - it stays alive
   at its predicted (extrapolated) position instead of disappearing.
3. Drop tracks that never move much (catches stationary ref/staff regardless of color).
4. Render: box + team label every frame, solid for real detections, dashed for predicted.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import json

VIDEO = '/Users/jordanngo/Projects/AI live tracking/video.mp4'
OUT_VIDEO = '/Users/jordanngo/Projects/AI live tracking/output_annotated_v3.mp4'
OUT_JSON = '/Users/jordanngo/Projects/AI live tracking/output_tracks_v3.json'

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
    # require real confidence, not just "edges out the other" - a grayish ref shirt
    # sits right at the margin, a genuine jersey clears it comfortably
    if white_frac < 0.10 and blue_frac < 0.10:
        return None
    return 'Canada' if white_frac >= blue_frac else 'Romania'

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

# ---- per-frame detection + on-court + team filter ----
per_frame_dets = []  # list over frames of list of dict(box, team, foot)
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
        dets.append({'box': box, 'team': team, 'foot': (fx, fy), 'center': (cx, cy)})
    per_frame_dets.append(dets)
print('detections after on-court+team filter, frame 0:', len(per_frame_dets[0]))

# ---- per-team greedy/optimal tracking by feet-position distance ----
MAX_GAP = 20          # frames a track can go unmatched before being retired
MAX_MATCH_DIST = 70   # px - a real player shouldn't teleport farther than this in 1 frame

class Track:
    def __init__(self, tid, team, frame_idx, box):
        self.id = tid
        self.team = team
        self.boxes = {frame_idx: box}      # frame -> (x1,y1,x2,y2)
        self.predicted = set()             # frames filled by extrapolation, not real detection
        self.last_frame = frame_idx
        self.last_center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
        self.velocity = (0.0, 0.0)
        self.gap = 0

    def predict_center(self):
        return (self.last_center[0] + self.velocity[0], self.last_center[1] + self.velocity[1])

    def update(self, frame_idx, box):
        center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
        dt = frame_idx - self.last_frame
        if dt > 0:
            self.velocity = ((center[0]-self.last_center[0])/dt, (center[1]-self.last_center[1])/dt)
        self.boxes[frame_idx] = box
        self.last_center = center
        self.last_frame = frame_idx
        self.gap = 0

    def mark_missed(self, frame_idx):
        gap_len = frame_idx - self.last_frame
        self.gap = gap_len
        pc = self.predict_center()
        w_prev = self.boxes[self.last_frame]
        bw, bh = w_prev[2]-w_prev[0], w_prev[3]-w_prev[1]
        box = (pc[0]-bw/2, pc[1]-bh/2, pc[0]+bw/2, pc[1]+bh/2)
        self.boxes[frame_idx] = box
        self.predicted.add(frame_idx)
        self.last_center = pc
        # a player who's stayed missing has usually stopped/cut, not kept running in a
        # straight line - decay the extrapolation so a long gap settles near its last
        # known spot instead of compounding further and further into the wrong player
        self.velocity = (self.velocity[0]*0.8, self.velocity[1]*0.8)

PLAYERS_PER_TEAM = 3  # fixed 3x3 roster on court, no subs expected in a clip this short

tracks_by_team = {'Canada': [], 'Romania': []}
next_id = [0]

# seed exactly 3 tracks per team from the first frame that has >=3 detections for that team,
# so a stray false detection in frame 0 alone can't warp the whole track count
seed_frame = {'Canada': None, 'Romania': None}
for fi, dets in enumerate(per_frame_dets):
    for team in ('Canada', 'Romania'):
        if seed_frame[team] is None:
            team_dets = [d for d in dets if d['team'] == team]
            if len(team_dets) >= PLAYERS_PER_TEAM:
                seed_frame[team] = fi
                for d in team_dets[:PLAYERS_PER_TEAM]:
                    nt = Track(next_id[0], team, fi, d['box']); next_id[0] += 1
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
        # always assign into the fixed 3 tracks - never spawn a new one - so a player who
        # drifted farther than a fixed cutoff during a contact/occlusion gap still gets
        # reattached to their own track instead of becoming a phantom 4th/5th/6th "Canada"
        cost = np.zeros((len(tracks), len(team_dets)))
        for i, t in enumerate(tracks):
            pc = t.predict_center()
            for j, d in enumerate(team_dets):
                dx = pc[0]-d['center'][0]; dy = pc[1]-d['center'][1]
                cost[i,j] = (dx*dx+dy*dy) ** 0.5
        row, col = linear_sum_assignment(cost)
        matched_tracks = set()
        for r_, c_ in zip(row, col):
            tracks[r_].update(fi, team_dets[c_]['box'])
            matched_tracks.add(r_)
        for i, t in enumerate(tracks):
            if i not in matched_tracks:
                t.mark_missed(fi)

all_tracks = tracks_by_team['Canada'] + tracks_by_team['Romania']
print('raw tracks built:', len(all_tracks))

# ---- drop tracks that are too short or barely move (catches stationary ref/staff) ----
MIN_FRAMES = 20
MIN_MOVEMENT = 40  # px, total bounding-box diagonal of the track's foot positions
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

# ---- render ----
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
