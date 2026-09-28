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


def embedding_distance_fn(model, device):
    cache = {}

    def get_emb(path):
        if path not in cache:
            img = load_image(path, train=False)
            t = torch.from_numpy(img).unsqueeze(0).to(device)
            with torch.no_grad():
                cache[path] = model(t).cpu().numpy()[0]
        return cache[path]

    def dist(path_a, path_b):
        ea, eb = get_emb(path_a), get_emb(path_b)
        return float(np.linalg.norm(ea - eb))

    return dist


def summarize(name, pos_dists, neg_dists):
    pos = np.array(pos_dists)
    neg = np.array(neg_dists)
    midpoint = (pos.mean() + neg.mean()) / 2
    midpoint_acc = ((pos < midpoint).sum() + (neg >= midpoint).sum()) / (len(pos) + len(neg))

    # best-possible threshold (scan candidates) - fairer when the two methods'
    # distances have very different spread, since the naive midpoint can be
    # suboptimal for a high-variance method even if it's more separable overall.
    candidates = np.unique(np.concatenate([pos, neg]))
    best_acc, best_t = 0.0, midpoint
    for t in candidates:
        acc = ((pos < t).sum() + (neg >= t).sum()) / (len(pos) + len(neg))
        if acc > best_acc:
            best_acc, best_t = acc, t

    d_prime = (neg.mean() - pos.mean()) / np.sqrt((pos.std()**2 + neg.std()**2) / 2)

    print(f"[{name}]")
    print(f"  same-identity dist:  mean {pos.mean():.3f}  std {pos.std():.3f}")
    print(f"  diff-identity dist:  mean {neg.mean():.3f}  std {neg.std():.3f}")
    print(f"  separation (d-prime, higher=better): {d_prime:.3f}")
    print(f"  midpoint-threshold accuracy: {midpoint_acc*100:.1f}%")
    print(f"  best-possible-threshold accuracy: {best_acc*100:.1f}% (at threshold {best_t:.3f})")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--eval-manifest", default=EVAL_MANIFEST_PATH)
    args = parser.parse_args()

    print(f"loading held-out eval pairs from {args.eval_manifest}")
    with open(args.eval_manifest) as f:
        eval_manifest = json.load(f)
    pos_pairs = eval_manifest["positive"]
    neg_pairs = eval_manifest["negative"]
    print(f"eval pairs: {len(pos_pairs)} positive, {len(neg_pairs)} negative")
    print()

    old_pos = [old_hsv_distance(a, b) for a, b in pos_pairs]
    old_neg = [old_hsv_distance(a, b) for a, b in neg_pairs]
    summarize("current HSV histogram (track_clip7.py's method)", old_pos, old_neg)

    device = torch.device("cpu")
    print(f"checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = EmbeddingNet(embed_dim=ckpt["embed_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dist_fn = embedding_distance_fn(model, device)
    new_pos = [dist_fn(a, b) for a, b in pos_pairs]
    new_neg = [dist_fn(a, b) for a, b in neg_pairs]
    summarize("trained embedding (held out, never trained on this half)", new_pos, new_neg)
