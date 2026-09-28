"""
Held-out check (step 5 of the bootstrap plan): does the trained embedding
separate same-identity vs. different-identity crops on the SECOND HALF of
real test.mov (never trained on - see extract_realtest_split.py) better than
the current hand-tuned HSV-histogram + Bhattacharyya method already in
track_clip7.py?

v1 of this check trained only on video.mp4/Test1.mp4/Test2.mp4 and completely
failed to generalize to real test.mov (domain gap). v2 adds the first half of
real test.mov to training and validates on the second half instead, so the
model has actually seen this visual domain while the eval pairs are still
temporally held out.
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from train_embedding import EmbeddingNet, load_image, IMG_H, IMG_W

PROJECT_DIR = "/Users/jordanngo/Projects/AI live tracking"
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "pseudo_labels", "embedding_model.pt")
EVAL_MANIFEST_PATH = os.path.join(PROJECT_DIR, "pseudo_labels", "realtest_second_half_eval_pairs.json")


def old_hsv_signature(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def old_hsv_distance(path_a, path_b):
    a, b = cv2.imread(path_a), cv2.imread(path_b)
    return cv2.compareHist(old_hsv_signature(a), old_hsv_signature(b), cv2.HISTCMP_BHATTACHARYYA)
