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


class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x8
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, embed_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)


def contrastive_loss(emb_a, emb_b, label, margin=MARGIN):
    dist = torch.nn.functional.pairwise_distance(emb_a, emb_b)
    same_loss = label * dist.pow(2)
    diff_loss = (1 - label) * torch.clamp(margin - dist, min=0).pow(2)
    return (same_loss + diff_loss).mean(), dist


def run_epoch(model, loader, optimizer, device, train):
    model.train(train)
    total_loss, n_batches = 0.0, 0
    all_dist, all_label = [], []
    for a, b, label in loader:
        a, b, label = a.to(device), b.to(device), label.to(device)
        with torch.set_grad_enabled(train):
            emb_a, emb_b = model(a), model(b)
            loss, dist = contrastive_loss(emb_a, emb_b, label)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        all_dist.append(dist.detach().cpu())
        all_label.append(label.detach().cpu())

    all_dist = torch.cat(all_dist)
    all_label = torch.cat(all_label)
    same_mean = all_dist[all_label == 1].mean().item() if (all_label == 1).any() else float("nan")
    diff_mean = all_dist[all_label == 0].mean().item() if (all_label == 0).any() else float("nan")
    return total_loss / n_batches, same_mean, diff_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("manifest:", args.manifest)

    pairs = load_pairs(args.manifest)
    n_val = int(len(pairs) * VAL_FRACTION)
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"train pairs: {len(train_pairs)}, val pairs: {len(val_pairs)}")

    train_loader = DataLoader(PairDataset(train_pairs, train=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(PairDataset(val_pairs, train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = EmbeddingNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_same, train_diff = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_same, val_diff = run_epoch(model, val_loader, optimizer, device, train=False)
        print(
            f"epoch {epoch:2d}  train_loss {train_loss:.4f} (same_dist {train_same:.3f}, diff_dist {train_diff:.3f})"
            f"  val_loss {val_loss:.4f} (same_dist {val_same:.3f}, diff_dist {val_diff:.3f})"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "embed_dim": EMBED_DIM, "img_size": (IMG_H, IMG_W)}, args.checkpoint)

    print(f"best val_loss {best_val_loss:.4f}, checkpoint saved to {args.checkpoint}")
