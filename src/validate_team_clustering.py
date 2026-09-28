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
