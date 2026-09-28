"""
v4 - pose-estimation based visualization (skeleton instead of a filled box), same
tracking pipeline (team color, court filter, fixed 3-per-team assignment, gap
prediction) as v3. Keypoints also make jersey-color sampling more precise: instead
of a crude box-fraction guess, sample the actual torso band between the shoulder
and hip keypoints.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import json

VIDEO = '/Users/jordanngo/Projects/AI live tracking/video.mp4'
OUT_VIDEO = '/Users/jordanngo/Projects/AI live tracking/output_annotated_v4.mp4'
OUT_JSON = '/Users/jordanngo/Projects/AI live tracking/output_tracks_v4.json'

COURT_POLY = np.array([
    (105, 300), (30, 355), (0, 400), (0, 540), (650, 540),
    (760, 430), (830, 330), (830, 260), (350, 248),
], dtype=np.int32)

def foot_on_court(fx, fy):
    return cv2.pointPolygonTest(COURT_POLY, (float(fx), float(fy)), False) >= 0

# COCO-17 keypoint indices
L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 5, 6, 11, 12, 15, 16
SKELETON = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16),(0,5),(0,6)]

def classify_team(frame, kpts_xy, kpts_conf):
    if kpts_conf[L_SH] < 0.3 or kpts_conf[R_SH] < 0.3 or kpts_conf[L_HIP] < 0.3 or kpts_conf[R_HIP] < 0.3:
        return None, None  # not enough confident torso keypoints to trust a color read
    sh_mid = (kpts_xy[L_SH] + kpts_xy[R_SH]) / 2
    hip_mid = (kpts_xy[L_HIP] + kpts_xy[R_HIP]) / 2
    shoulder_w = np.linalg.norm(kpts_xy[L_SH] - kpts_xy[R_SH])
    half_w = max(shoulder_w * 0.6, 6)
    x1 = int(min(sh_mid[0], hip_mid[0]) - half_w); x2 = int(max(sh_mid[0], hip_mid[0]) + half_w)
    y1 = int(min(sh_mid[1], hip_mid[1])); y2 = int(max(sh_mid[1], hip_mid[1]))
    x1, y1 = max(x1,0), max(y1,0)
    x2, y2 = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return None, (x1,y1,x2,y2)
    patch = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    total = h_.size
    white_mask = (s_ < 15) & (v_ > 110)
    blue_mask = (h_ >= 100) & (h_ <= 140) & (s_ > 50)
    white_frac = white_mask.sum() / total
    blue_frac = blue_mask.sum() / total
    if white_frac < 0.10 and blue_frac < 0.10:
        return None, (x1,y1,x2,y2)
    team = 'Canada' if white_frac >= blue_frac else 'Romania'
    return team, (x1,y1,x2,y2)

model = YOLO('/Users/jordanngo/Projects/AI live tracking/yolo11n-pose.pt')

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
    if r.keypoints is None:
        per_frame_dets.append(dets); continue
    kxy_all = r.keypoints.xy.cpu().numpy()
    kconf_all = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy_all.shape[:2])
    boxes_all = r.boxes.xyxy.cpu().numpy()
    for kxy, kconf, box in zip(kxy_all, kconf_all, boxes_all):
        x1,y1,x2,y2 = box
        # foot position: ankle midpoint if confident, else box bottom-center
        if kconf[L_ANK] > 0.3 and kconf[R_ANK] > 0.3:
            fx, fy = (kxy[L_ANK][0]+kxy[R_ANK][0])/2, (kxy[L_ANK][1]+kxy[R_ANK][1])/2
        else:
            fx, fy = (x1+x2)/2, y2
        if not foot_on_court(fx, fy):
            continue
        team, torso_box = classify_team(frames[fi], kxy, kconf)
        if team is None:
            continue
        if kconf[L_HIP] > 0.3 and kconf[R_HIP] > 0.3:
            cx, cy = (kxy[L_HIP][0]+kxy[R_HIP][0])/2, (kxy[L_HIP][1]+kxy[R_HIP][1])/2
        else:
            cx, cy = (x1+x2)/2, (y1+y2)/2
        dets.append({'team': team, 'center': (cx,cy), 'kxy': kxy, 'kconf': kconf, 'box': box})
    per_frame_dets.append(dets)
print('detections after on-court+team filter, frame 0:', len(per_frame_dets[0]))

class Track:
    def __init__(self, tid, team, frame_idx, d):
        self.id = tid; self.team = team
        self.frames = {frame_idx: d}
        self.predicted = set()
        self.last_frame = frame_idx
        self.last_center = d['center']
        self.velocity = (0.0, 0.0)

    def predict_center(self):
        return (self.last_center[0]+self.velocity[0], self.last_center[1]+self.velocity[1])

    def update(self, frame_idx, d):
        dt = frame_idx - self.last_frame
        if dt > 0:
            self.velocity = ((d['center'][0]-self.last_center[0])/dt, (d['center'][1]-self.last_center[1])/dt)
        self.frames[frame_idx] = d
        self.last_center = d['center']
        self.last_frame = frame_idx

    def mark_missed(self, frame_idx):
        pc = self.predict_center()
        prev = self.frames[max(k for k in self.frames if k <= frame_idx-1)] if any(k <= frame_idx-1 for k in self.frames) else self.frames[self.last_frame]
        shift = (pc[0]-self.last_center[0], pc[1]-self.last_center[1])
        kxy = prev['kxy'] + np.array(shift)
        d = {'team': self.team, 'center': pc, 'kxy': kxy, 'kconf': prev['kconf'], 'box': prev['box'] + np.array([shift[0],shift[1],shift[0],shift[1]])}
        self.frames[frame_idx] = d
        self.predicted.add(frame_idx)
        self.last_center = pc
        self.velocity = (self.velocity[0]*0.8, self.velocity[1]*0.8)

PLAYERS_PER_TEAM = 3
tracks_by_team = {'Canada': [], 'Romania': []}
next_id = [0]
seed_frame = {'Canada': None, 'Romania': None}
for fi, dets in enumerate(per_frame_dets):
    for team in ('Canada','Romania'):
        if seed_frame[team] is None:
            team_dets = [d for d in dets if d['team']==team]
            if len(team_dets) >= PLAYERS_PER_TEAM:
                seed_frame[team] = fi
                for d in team_dets[:PLAYERS_PER_TEAM]:
                    nt = Track(next_id[0], team, fi, d); next_id[0]+=1
                    tracks_by_team[team].append(nt)

for fi, dets in enumerate(per_frame_dets):
    for team in ('Canada','Romania'):
        if seed_frame[team] is None or fi <= seed_frame[team]:
            continue
        team_dets = [d for d in dets if d['team']==team]
        tracks = tracks_by_team[team]
        if not team_dets:
            for t in tracks: t.mark_missed(fi)
            continue
        cost = np.zeros((len(tracks), len(team_dets)))
        for i,t in enumerate(tracks):
            pc = t.predict_center()
            for j,d in enumerate(team_dets):
                dx=pc[0]-d['center'][0]; dy=pc[1]-d['center'][1]
                cost[i,j]=(dx*dx+dy*dy)**0.5
        row,col = linear_sum_assignment(cost)
        matched=set()
        for r_,c_ in zip(row,col):
            tracks[r_].update(fi, team_dets[c_]); matched.add(r_)
        for i,t in enumerate(tracks):
            if i not in matched: t.mark_missed(fi)

all_tracks = tracks_by_team['Canada']+tracks_by_team['Romania']
print('tracks:', len(all_tracks))
for t in all_tracks:
    print(f'  #{t.id} {t.team}: {len(t.frames)} frames ({len(t.predicted)} predicted)')

TEAM_COLORS = {'Canada': (255,255,255), 'Romania': (255,140,0)}
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w,h))
for fi, frame in enumerate(frames):
    out = frame.copy()
    for t in all_tracks:
        if fi not in t.frames: continue
        d = t.frames[fi]
        color = TEAM_COLORS[t.team]
        kxy, kconf = d['kxy'], d['kconf']
        is_pred = fi in t.predicted
        for a,b in SKELETON:
            if kconf[a] > 0.3 and kconf[b] > 0.3:
                p1 = tuple(kxy[a].astype(int)); p2 = tuple(kxy[b].astype(int))
                cv2.line(out, p1, p2, color, 2 if not is_pred else 1, cv2.LINE_AA)
        for k in range(len(kxy)):
            if kconf[k] > 0.3:
                cv2.circle(out, tuple(kxy[k].astype(int)), 3, color, -1)
        head = kxy[0] if kconf[0]>0.3 else (kxy[5]+kxy[6])/2
        cv2.putText(out, f'{t.team} #{t.id}', (int(head[0])-20, int(head[1])-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    writer.write(out)
writer.release()
print('saved', OUT_VIDEO)

with open(OUT_JSON,'w') as f:
    json.dump({
        'fps': fps, 'width': w, 'height': h, 'n_frames': n_frames,
        'track_team': {str(t.id): t.team for t in all_tracks},
        'predicted': {str(t.id): sorted(int(x) for x in t.predicted) for t in all_tracks},
        'centers': {str(t.id): {str(k): [float(v['center'][0]), float(v['center'][1])] for k,v in t.frames.items()} for t in all_tracks},
    }, f)
print('saved', OUT_JSON)
