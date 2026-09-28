"""
Fully automatic pipeline, no manual calibration required:
1. Run YOLO detection+tracking across every frame of video.mp4 (person + sports ball).
2. Classify each tracked person into a team by jersey color (Canada=white, Romania=blue/yellow, other=ref/bench).
3. Record each track's foot position (bottom-center of box) per frame -> pixel-space trajectory.
4. Render an annotated output video with track trails, team-colored boxes, and ball marker.
5. Render a static "coverage" image: every track's trail overlaid on one frame.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import json

VIDEO = '/Users/jordanngo/Projects/AI live tracking/video.mp4'
OUT_VIDEO = '/Users/jordanngo/Projects/AI live tracking/output_annotated.mp4'
OUT_COVERAGE = '/Users/jordanngo/Projects/AI live tracking/output_coverage.png'
OUT_JSON = '/Users/jordanngo/Projects/AI live tracking/output_tracks.json'

model = YOLO('/Users/jordanngo/Projects/AI live tracking/yolo11n.pt')

def classify_team(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1,0), max(y1,0)
    x2, y2 = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return 'unknown', (128,128,128)
    # sample jersey area: upper-middle 40% of the box (torso), avoid skin/shorts extremes
    h = y2 - y1
    jy1 = y1 + int(h*0.15)
    jy2 = y1 + int(h*0.55)
    if jy2 <= jy1:
        jy2 = jy1 + 1
    patch = frame[jy1:jy2, x1:x2]
    if patch.size == 0:
        return 'unknown', (128,128,128)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    avg_s = hsv[:,:,1].mean()
    avg_v = hsv[:,:,2].mean()
    avg_h = hsv[:,:,0].mean()
    # low saturation + high value = white jersey (Canada)
    if avg_s < 60 and avg_v > 120:
        return 'Canada', (255,255,255)
    # blue/yellow mix (Romania) -> higher saturation
    if avg_s >= 60:
        return 'Romania', (0,215,255)
    return 'other', (160,160,160)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))

trails = defaultdict(list)       # track_id -> list of (frame_idx, x, y)
team_votes = defaultdict(lambda: defaultdict(int))  # track_id -> team -> count
ball_positions = []               # (frame_idx, x, y, conf)

results_gen = model.track(VIDEO, classes=[0, 32], persist=True, tracker='bytetrack.yaml',
                           conf=0.25, iou=0.5, verbose=False, stream=True)

frame_idx = 0
for r in results_gen:
    frame = r.orig_img.copy()
    boxes = r.boxes
    if boxes is not None and boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int)
        clses = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for tid, cls, box, conf in zip(ids, clses, xyxy, confs):
            x1, y1, x2, y2 = box
            if cls == 0:  # person
                foot_x, foot_y = (x1+x2)/2, y2
                team, color = classify_team(frame, box)
                team_votes[int(tid)][team] += 1
                trails[int(tid)].append((frame_idx, float(foot_x), float(foot_y)))
                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
                cv2.putText(frame, f'#{tid} {team}', (int(x1), int(y1)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            elif cls == 32:  # sports ball
                bx, by = (x1+x2)/2, (y1+y2)/2
                ball_positions.append((frame_idx, float(bx), float(by), float(conf)))
                cv2.circle(frame, (int(bx),int(by)), 8, (0,140,255), 2)
                cv2.putText(frame, 'ball', (int(bx)+10,int(by)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,140,255), 1)

    # draw trails so far
    for tid, pts in trails.items():
        for i in range(1, len(pts)):
            if pts[i][0] < frame_idx - 90:  # keep trail to last ~3s
                continue
            p1 = (int(pts[i-1][1]), int(pts[i-1][2]))
            p2 = (int(pts[i][1]), int(pts[i][2]))
            cv2.line(frame, p1, p2, (60,220,60), 2)

    writer.write(frame)
    frame_idx += 1

writer.release()
print('frames processed:', frame_idx)

# team label per track = majority vote
track_team = {}
for tid, votes in team_votes.items():
    track_team[tid] = max(votes.items(), key=lambda kv: kv[1])[0]
print('tracks found:', len(trails))
for tid in sorted(trails.keys()):
    print(f'  track {tid}: team={track_team[tid]}, frames_seen={len(trails[tid])}')
print('ball detections:', len(ball_positions))

with open(OUT_JSON, 'w') as f:
    json.dump({
        'fps': fps, 'width': w, 'height': h, 'n_frames': n_frames,
        'trails': {str(k): v for k, v in trails.items()},
        'track_team': {str(k): v for k, v in track_team.items()},
        'ball_positions': ball_positions,
    }, f)
print('saved', OUT_JSON)

# coverage image: overlay every trail on the last frame
cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ok, base = cap.read()
cap.release()
canvas = base.copy()
palette = [(60,220,60),(220,80,220),(80,160,255),(255,180,40),(40,220,220),(220,60,60),(180,220,40),(220,140,220)]
for i, (tid, pts) in enumerate(sorted(trails.items())):
    color = palette[i % len(palette)]
    for j in range(1, len(pts)):
        p1 = (int(pts[j-1][1]), int(pts[j-1][2]))
        p2 = (int(pts[j][1]), int(pts[j][2]))
        cv2.line(canvas, p1, p2, color, 2)
    if pts:
        lx, ly = int(pts[-1][1]), int(pts[-1][2])
        cv2.putText(canvas, f'#{tid} {track_team[tid]}', (lx+5, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
cv2.imwrite(OUT_COVERAGE, canvas)
print('saved', OUT_COVERAGE)
