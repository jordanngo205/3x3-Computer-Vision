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
