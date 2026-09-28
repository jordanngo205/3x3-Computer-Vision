"""
v2 - tighter scope per feedback:
- only detections whose feet are on the court (excludes stands/crowd/staff on the concourse)
- exactly two team colors: white=Canada, blue=Romania; anything else (refs, photographers, bench) is dropped entirely
- occlusion no longer removes the box: gaps in a track are linearly interpolated so a box is drawn every frame
- no long drag trails; just a clean box + team label per frame
"""
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import json

VIDEO = '/Users/jordanngo/Projects/AI live tracking/video.mp4'
OUT_VIDEO = '/Users/jordanngo/Projects/AI live tracking/output_annotated_v2.mp4'
OUT_JSON = '/Users/jordanngo/Projects/AI live tracking/output_tracks_v2.json'

import numpy as np
# traced by eye from the temporal-median background plate (players removed) - the visible
# playing surface only, deliberately tight on the right side where the photographer/bench
# chairs sit just outside the boundary line
COURT_POLY = np.array([
    (85, 285), (0, 340), (0, 540), (650, 540),
    (760, 430), (830, 330), (830, 260), (350, 248),
], dtype=np.int32)

def foot_on_court(fx, fy):
    return cv2.pointPolygonTest(COURT_POLY, (float(fx), float(fy)), False) >= 0

TEAM_COLORS = {
    'Canada':  (255, 255, 255),  # white box
    'Romania': (255, 140, 0),    # blue box (BGR)
}

model = YOLO('/Users/jordanngo/Projects/AI live tracking/yolo11n.pt')

def classify_team(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return None
    w, h = x2 - x1, y2 - y1
    # center half of the box width, upper-torso band -> avoids background bleeding
    # in at the box edges and avoids head/hair and shorts/skin at top and bottom
    jx1, jx2 = x1 + int(w*0.25), x1 + int(w*0.75)
    jy1, jy2 = y1 + int(h*0.18), y1 + int(h*0.5)
    if jy2 <= jy1 or jx2 <= jx1:
        return None
    patch = frame[jy1:jy2, jx1:jx2]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    total = h.size
    # white must be near-zero saturation - a loose "low saturation" band also catches
    # specular highlights/glare on the blue jersey, which still carry some saturation
    white_mask = (s < 15) & (v > 110)
    blue_mask = (h >= 100) & (h <= 140) & (s > 50)
    white_frac = white_mask.sum() / total
    blue_frac = blue_mask.sum() / total
    if white_frac < 0.03 and blue_frac < 0.03:
        return None
    return 'Canada' if white_frac >= blue_frac else 'Romania'

# ---- pass 1: detect + track every frame, keep raw per-frame team votes ----
trails = defaultdict(dict)          # track_id -> {frame_idx: (x1,y1,x2,y2)}
team_votes = defaultdict(lambda: defaultdict(int))

results_gen = model.track(VIDEO, classes=[0], persist=True,
                           tracker='/Users/jordanngo/Projects/AI live tracking/src/bytetrack_custom.yaml',
                           conf=0.25, iou=0.5, verbose=False, stream=True)

frame_idx = 0
frames_cache = []
for r in results_gen:
    frame = r.orig_img
    frames_cache.append(frame.copy())
    boxes = r.boxes
    if boxes is not None and boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        for tid, box in zip(ids, xyxy):
            x1, y1, x2, y2 = box
            foot_x, foot_y = (x1+x2)/2, y2
            if not foot_on_court(foot_x, foot_y):
                continue  # in the stands / bench / photographer, not a player
            team = classify_team(frame, box)
            if team is None:
                continue  # ref / bench / unclear -> drop
            team_votes[int(tid)][team] += 1
            trails[int(tid)][frame_idx] = (float(x1), float(y1), float(x2), float(y2))
    frame_idx += 1

n_frames = frame_idx
print('frames:', n_frames, 'raw tracks with any on-court+team-colored detection:', len(trails))

# majority-vote team per track; drop tracks that never got a confident team-colored hit
track_team = {}
for tid, votes in team_votes.items():
    if not votes:
        continue
    track_team[tid] = max(votes.items(), key=lambda kv: kv[1])[0]

# keep only tracks seen on enough distinct frames to be a real player, not a flicker
MIN_FRAMES = 15
kept = {tid: boxes for tid, boxes in trails.items() if tid in track_team and len(boxes) >= MIN_FRAMES}
print('kept tracks after team+min-frame filter:', len(kept))
for tid, boxes in sorted(kept.items(), key=lambda kv: -len(kv[1])):
    print(f'  #{tid} {track_team[tid]}: {len(boxes)} raw detections, span {min(boxes)}-{max(boxes)}')

# ---- interpolate gaps so the box never disappears within a track's span ----
def interpolate_track(boxes):
    frames_sorted = sorted(boxes.keys())
    filled = {}
    for i in range(len(frames_sorted)-1):
        f0, f1 = frames_sorted[i], frames_sorted[i+1]
        b0, b1 = boxes[f0], boxes[f1]
        filled[f0] = b0
        gap = f1 - f0
        if gap > 1:
            for k in range(1, gap):
                t = k / gap
                interp = tuple(b0[j] + (b1[j]-b0[j])*t for j in range(4))
                filled[f0+k] = interp
    filled[frames_sorted[-1]] = boxes[frames_sorted[-1]]
    return filled

kept_filled = {tid: interpolate_track(boxes) for tid, boxes in kept.items()}

# ---- render annotated video: box + label every frame, no trails ----
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))

for fi, frame in enumerate(frames_cache):
    out = frame.copy()
    for tid, boxes in kept_filled.items():
        if fi not in boxes:
            continue
        team = track_team[tid]
        color = TEAM_COLORS[team]
        x1, y1, x2, y2 = [int(v) for v in boxes[fi]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f'{team} #{tid}', (x1, max(0, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    writer.write(out)
writer.release()
print('saved', OUT_VIDEO)

with open(OUT_JSON, 'w') as f:
    json.dump({
        'fps': fps, 'width': w, 'height': h, 'n_frames': n_frames,
        'tracks': {str(k): v for k, v in kept_filled.items()},
        'track_team': {str(k): v for k, v in track_team.items() if k in kept_filled},
    }, f)
print('saved', OUT_JSON)
