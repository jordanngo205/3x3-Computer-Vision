"""
Validation-only script: does DIRECT color-histogram clustering (K-means on actual HSV
color histograms of the jersey crop - the same approach TrackID3x3's own
calculate_color_hist.py / add_attributes.ipynb uses) agree with our trusted HSV
classify_team() better than the SigLIP general-embedding approach did (67.4%)?

Same crop region as classify_team() on purpose - isolates "does the METHOD (clustering
on real color vs. one fixed rule) fix this" as the one variable being tested, not crop
placement. If this alone doesn't close the gap, crop-region improvements are next.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

VIDEO = '/Users/jordanngo/Projects/AI live tracking/video.mp4'

COURT_POLY = np.array([
    (105, 300), (30, 355), (0, 400), (0, 540), (650, 540),
    (760, 430), (830, 330), (830, 260), (350, 248),
], dtype=np.int32)

def foot_on_court(fx, fy):
    return cv2.pointPolygonTest(COURT_POLY, (float(fx), float(fy)), False) >= 0

def classify_team(frame, box):
    """Our trusted, hand-tuned ground-truth rule for this validation only."""
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

def color_histogram(frame, box):
    """Direct HSV color histogram of the SAME crop region - no deep-learning embedding,
    just the actual color distribution, matching TrackID3x3's own approach."""
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
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()

print('loading video...')
model = YOLO('/Users/jordanngo/Projects/AI live tracking/yolo11n.pt')
cap = cv2.VideoCapture(VIDEO)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
print('loaded', len(frames), 'frames')

print('running detection + collecting crops...')
results = model(frames, classes=[0], conf=0.25, iou=0.85, verbose=False)

hists = []
hsv_labels = []
for fi, r in enumerate(results):
    frame = frames[fi]
    for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = box
        fx, fy = (x1+x2)/2, y2
        if not foot_on_court(fx, fy):
            continue
        label = classify_team(frame, box)
        if label is None:
            continue
        hist = color_histogram(frame, box)
        if hist is None:
            continue
        hists.append(hist)
        hsv_labels.append(label)

print('collected', len(hists), 'labeled crops (Canada:', hsv_labels.count('Canada'), 'Romania:', hsv_labels.count('Romania'), ')')

X = np.array(hists)
print('running KMeans k=2 directly on color histograms...')
km = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_ids = km.fit_predict(X)

cluster_to_team = {}
for c in (0, 1):
    labels_in_c = [hsv_labels[i] for i in range(len(hsv_labels)) if cluster_ids[i] == c]
    if not labels_in_c:
        continue
    majority = max(set(labels_in_c), key=labels_in_c.count)
    cluster_to_team[c] = majority

pred_labels = [cluster_to_team[c] for c in cluster_ids]
agree = sum(1 for p, g in zip(pred_labels, hsv_labels) if p == g)
print(f'agreement: {agree}/{len(hsv_labels)} = {100*agree/len(hsv_labels):.1f}%')

for c in (0, 1):
    idxs = [i for i in range(len(hsv_labels)) if cluster_ids[i] == c]
    canada_n = sum(1 for i in idxs if hsv_labels[i] == 'Canada')
    romania_n = sum(1 for i in idxs if hsv_labels[i] == 'Romania')
    print(f'cluster {c} -> {cluster_to_team.get(c)}: {len(idxs)} items ({canada_n} Canada-HSV, {romania_n} Romania-HSV)')
