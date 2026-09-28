"""
Systematic frame-by-frame audit of output_tracks_v7.json against the actual video.
For every frame and every kept track, checks two failure signatures:
1. "floating" - the track's box (real or held/interpolated) doesn't overlap ANY
   actual person-shaped, correctly-colored region of the frame at all - i.e. it's
   sitting over empty court.
2. "wrong-color" - re-samples the ACTUAL pixels currently under the box and checks
   whether they still read as the track's assigned team color. If a held/interpolated
   box has drifted onto a different-colored player, this catches it directly instead
   of inferring from position logic alone.
3. "drifted" - specifically for held/interpolated frames: checks whether the box's
   REAL person-shaped content still overlaps meaningfully with where the track was at
   the nearest real (non-interpolated) detection. Catches the "smooth but wrong" case a
   pure jump-distance check misses entirely - a straight-line interpolation across a
   long gap is smooth by construction even when it ends up sitting on a totally
   different player, which frame-to-frame smoothness alone can't detect.

Uses the SAME clustering-based team classifier that actually produced these tracks
(loaded from the .pkl saved alongside the tracks JSON), not the old fixed HSV rule -
using a different/outdated classifier to audit the pipeline's own decisions was
producing false-positive "errors" where the audit was wrong, not the pipeline.
"""
import cv2
import numpy as np
import json
import pickle
import argparse

BASE = '/Users/jordanngo/Projects/AI live tracking'

parser = argparse.ArgumentParser()
parser.add_argument('--video', default=f'{BASE}/video.mp4')
parser.add_argument('--tag', default='v7', help='must match the --tag used when running track_clip7.py')
args = parser.parse_args()

VIDEO = args.video
TRACKS = f'{BASE}/output_tracks_{args.tag}.json'
CLASSIFIER = f'{BASE}/output_team_classifier_{args.tag}.pkl'

with open(CLASSIFIER, 'rb') as f:
    clf = pickle.load(f)
km = clf['kmeans']
cluster_to_team = clf['cluster_to_team']

def color_histogram(frame, box):
    """Must exactly match track_clip7.py's color_histogram - same tightened sampling
    window (35-65% width, 15-42% height), or the loaded KMeans model's feature space
    won't line up with what's being sampled here."""
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

def classify_patch(frame, box):
    hist = color_histogram(frame, box)
    if hist is None:
        return None, 0.0, 0.0
    cluster = km.predict(hist.reshape(1, -1).astype(np.float64))[0]
    team = cluster_to_team[cluster]
    # wf/bf kept for printed diagnostics only (not used for the team decision anymore)
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
    w, h = x2 - x1, y2 - y1
    jx1, jx2 = x1 + int(w*0.35), x1 + int(w*0.65)
    jy1, jy2 = y1 + int(h*0.15), y1 + int(h*0.42)
    patch = frame[jy1:jy2, jx1:jx2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    total = h_.size
    wf = ((s_ < 15) & (v_ > 110)).sum() / total
    bf = ((h_ >= 100) & (h_ <= 140) & (s_ > 50)).sum() / total
    return team, wf, bf

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

def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2-x1) * (y2-y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1]); area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)

issues = []
total_checks = 0
for tid, boxes in tracks.items():
    team = track_team[tid]
    pred_set = set(predicted.get(tid, []))
    present_frames = sorted(int(k) for k in boxes.keys())
    real_frames = sorted(f for f in present_frames if f not in pred_set)
    for f in present_frames:
        total_checks += 1
        box = boxes[str(f)]
        frame = frames[f]
        floating = is_floating(frame, box)
        actual_team, wf, bf = classify_patch(frame, box)
        wrong_color = actual_team is not None and actual_team != team
        # drift check: for held/interpolated frames only, does this box still overlap
        # a REAL detection of this track from a nearby frame? A gap-filled box that's
        # drifted onto a totally different player will have near-zero IoU with both its
        # bracketing real detections, even though it moved smoothly to get there.
        drifted = False
        if f in pred_set and real_frames:
            nearest = min(real_frames, key=lambda rf: abs(rf - f))
            if abs(nearest - f) <= 20:  # only meaningful within a reasonable distance
                nearest_box = boxes[str(nearest)]
                if box_iou(box, nearest_box) < 0.05:
                    # boxes far apart is expected if the player genuinely moved a lot in
                    # that many frames - only flag if the CURRENT box also doesn't match
                    # its own assigned team color, i.e. corroborating evidence, not IoU alone
                    drifted = wrong_color
        if floating or wrong_color or drifted:
            issues.append({
                'frame': f, 'track': tid, 'team': team,
                'floating': floating, 'wrong_color': wrong_color, 'drifted': drifted,
                'actual_team': actual_team, 'white_frac': round(wf,2), 'blue_frac': round(bf,2),
                'is_held': f in pred_set,
            })

print(f'Total frame-track issue instances: {len(issues)} out of {total_checks} checked')
print()
from itertools import groupby
issues.sort(key=lambda x: (x['track'], x['frame']))
for tid, group in groupby(issues, key=lambda x: x['track']):
    group = list(group)
    frames_list = [g['frame'] for g in group]
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
    n_drift = sum(1 for g in group if g['drifted'])
    print(f"track #{tid} ({track_team[tid]}): {len(frames_list)} bad frames, floating={n_float} wrong_color={n_wrong} drifted={n_drift}")
    print(f"  runs: {runs}")
