"""
Validation-only script: does unsupervised SigLIP+UMAP+KMeans team clustering agree
with our trusted HSV classify_team() on video.mp4? We treat classify_team's output as
ground truth here (it was manually tuned and root-caused earlier in this project to be
accurate on this clip) and just measure agreement, BEFORE swapping anything into the
main pipeline.
"""
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from transformers import AutoProcessor, AutoModel
import umap
from sklearn.cluster import KMeans

VIDEO = '/Users/jordanngo/Projects/AI live tracking/video.mp4'

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

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print('loading SigLIP on', device, '...')
proc = AutoProcessor.from_pretrained('google/siglip-base-patch16-224')
siglip = AutoModel.from_pretrained('google/siglip-base-patch16-224').to(device).eval()

print('running detection + collecting crops...')
results = model(frames, classes=[0], conf=0.25, iou=0.85, verbose=False)

crops = []       # RGB numpy patches
hsv_labels = []   # ground truth from classify_team
meta = []         # (frame_idx, box)
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
        xi1, yi1, xi2, yi2 = [int(v) for v in box]
        xi1, yi1 = max(xi1, 0), max(yi1, 0)
        xi2, yi2 = min(xi2, frame.shape[1]-1), min(yi2, frame.shape[0]-1)
        if xi2 <= xi1 or yi2 <= yi1:
            continue
        patch = cv2.cvtColor(frame[yi1:yi2, xi1:xi2], cv2.COLOR_BGR2RGB)
        crops.append(patch)
        hsv_labels.append(label)
        meta.append((fi, box))

print('collected', len(crops), 'labeled crops (Canada:', hsv_labels.count('Canada'), 'Romania:', hsv_labels.count('Romania'), ')')

print('computing SigLIP embeddings...')
embeddings = []
batch_size = 32
with torch.no_grad():
    for i in range(0, len(crops), batch_size):
        batch = crops[i:i+batch_size]
        inputs = proc(images=batch, return_tensors='pt').to(device)
        feats = siglip.get_image_features(**inputs)
        embeddings.append(feats.cpu().numpy())
embeddings = np.concatenate(embeddings, axis=0)
print('embeddings shape:', embeddings.shape)

print('running UMAP...')
reducer = umap.UMAP(n_components=5, random_state=42)
reduced = reducer.fit_transform(embeddings)

print('running KMeans k=2...')
km = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_ids = km.fit_predict(reduced)

# Map cluster ids -> team names by majority vote against HSV labels (clustering has no
# inherent notion of "Canada" vs "Romania", just group A vs group B)
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

# per-cluster breakdown
for c in (0, 1):
    idxs = [i for i in range(len(hsv_labels)) if cluster_ids[i] == c]
    canada_n = sum(1 for i in idxs if hsv_labels[i] == 'Canada')
    romania_n = sum(1 for i in idxs if hsv_labels[i] == 'Romania')
    print(f'cluster {c} -> {cluster_to_team.get(c)}: {len(idxs)} items ({canada_n} Canada-HSV, {romania_n} Romania-HSV)')
