"""
Systematic frame-by-frame audit of output_tracks_test2.json against the actual video.
For every frame and every kept track, checks two failure signatures:
1. "floating" - the track's box (real or held/interpolated) doesn't overlap ANY
   actual person-shaped, correctly-colored region of the frame at all - i.e. it's
   sitting over empty court.
2. "wrong-color" - re-samples the ACTUAL pixels currently under the box and checks
   whether they still read as the track's assigned team color. If a held/interpolated
   box has drifted onto a different-colored player, this catches it directly instead
   of inferring from position logic alone.
"""
import cv2
import numpy as np
import json

VIDEO = '/Users/jordanngo/Projects/AI live tracking/Test2.mp4'
TRACKS = '/Users/jordanngo/Projects/AI live tracking/output_tracks_test2.json'

def classify_patch(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return None, 0.0, 0.0
    w, h = x2 - x1, y2 - y1
    jx1, jx2 = x1 + int(w*0.25), x1 + int(w*0.75)
    jy1, jy2 = y1 + int(h*0.18), y1 + int(h*0.5)
    if jy2 <= jy1 or jx2 <= jx1:
        return None, 0.0, 0.0
    patch = frame[jy1:jy2, jx1:jx2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    total = h_.size
    white_mask = (s_ < 15) & (v_ > 110)
    blue_mask = (h_ >= 100) & (h_ <= 140) & (s_ > 50)
    wf = white_mask.sum() / total
    bf = blue_mask.sum() / total
    if wf < 0.10 and bf < 0.10:
        return None, wf, bf
    return ('Canada' if wf >= bf else 'Romania'), wf, bf

def is_floating(frame, box):
    """No real person-shaped content at all under this box - background/floor color
    dominates (low saturation AND not particularly 'jersey' bright, or just flat gray)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return True
    patch = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # court floor is a fairly uniform mid-gray; a real person crop has much higher
    # local variance (limbs, shadow, jersey edges, skin) than an empty floor patch
    return float(gray.std()) < 12.0

data = json.load(open(TRACKS))
tracks = data['tracks']
predicted = data['predicted']
track_team = data['track_team']
n_frames = data['n_frames']

cap = cv2.VideoCapture(VIDEO)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()

issues = []
for tid, boxes in tracks.items():
    team = track_team[tid]
    for f in range(n_frames):
        box = boxes[str(f)]
        frame = frames[f]
        floating = is_floating(frame, box)
        actual_team, wf, bf = classify_patch(frame, box)
        wrong_color = actual_team is not None and actual_team != team
        if floating or wrong_color:
            issues.append({
                'frame': f, 'track': tid, 'team': team,
                'floating': floating, 'wrong_color': wrong_color,
                'actual_team': actual_team, 'white_frac': round(wf,2), 'blue_frac': round(bf,2),
                'is_held': f in predicted.get(tid, []),
            })

print(f'Total frame-track issue instances: {len(issues)} out of {n_frames*len(tracks)} checked')
print()
# summarize as contiguous runs per track for readability
from itertools import groupby
issues.sort(key=lambda x: (x['track'], x['frame']))
for tid, group in groupby(issues, key=lambda x: x['track']):
    group = list(group)
    frames_list = [g['frame'] for g in group]
    # contiguous runs
    runs = []
    start = frames_list[0]
    prev = frames_list[0]
    for fr in frames_list[1:]:
        if fr - prev > 1:
            runs.append((start, prev))
            start = fr
        prev = fr
    runs.append((start, prev))
    n_float = sum(1 for g in group if g['floating'])
    n_wrong = sum(1 for g in group if g['wrong_color'])
    print(f"track #{tid} ({track_team[tid]}): {len(frames_list)} bad frames, floating={n_float} wrong_color={n_wrong}")
    print(f"  runs: {runs}")
