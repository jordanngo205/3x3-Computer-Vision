"""
Train a small appearance/ReID embedding on the pseudo-labeled pairs produced by
extract_pseudo_labels.py. Positive pairs = same track_id, nearby in time (our own
pipeline was already confident these are the same person). Negative pairs =
different track_ids, same frame (definitely different people).

This is meant to eventually replace the hand-tuned HSV-histogram + Bhattacharyya
distance appearance matching in track_clip7.py with something more robust through
occlusion/lighting changes. It can only be as good as the frames the pipeline was
already confident about - it won't fix v7's existing low-confidence failure cases,
just make the appearance-matching step more consistent on the frames it does see.

`real test.mov` is NOT in this training data - it's held out to check afterward
whether this actually helps.
"""
import argparse
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_DIR = "/Users/jordanngo/Projects/AI live tracking"
MANIFEST_PATH = os.path.join(PROJECT_DIR, "pseudo_labels", "pairs.json")
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "pseudo_labels", "embedding_model.pt")

IMG_H, IMG_W = 128, 64  # person-shaped crops
EMBED_DIM = 128
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
VAL_FRACTION = 0.1
MARGIN = 1.0
RANDOM_SEED = 42


def load_pairs(manifest_path):
    with open(manifest_path) as f:
        m = json.load(f)
    pairs = [(a, b, 1) for a, b in m["positive"]] + [(a, b, 0) for a, b in m["negative"]]
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pairs)
    return pairs


_raw_image_cache = {}


def _load_raw_cached(path):
    """Decode+resize each crop from disk only once - every epoch re-reads the
    same ~10k tiny images otherwise, which made disk I/O the bottleneck instead
    of the (tiny) model itself."""
    cached = _raw_image_cache.get(path)
    if cached is None:
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_W, IMG_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cached = img
        _raw_image_cache[path] = cached
    return cached


def load_image(path, train):
    img = _load_raw_cached(path)
    if train and random.random() < 0.5:
        img = img[:, ::-1, :]  # horizontal flip augmentation
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return np.transpose(img, (2, 0, 1))  # CHW


class PairDataset(Dataset):
    def __init__(self, pairs, train):
        self.pairs = pairs
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path_a, path_b, label = self.pairs[idx]
        a = load_image(path_a, self.train)
        b = load_image(path_b, self.train)
        return torch.from_numpy(a), torch.from_numpy(b), torch.tensor(label, dtype=torch.float32)
