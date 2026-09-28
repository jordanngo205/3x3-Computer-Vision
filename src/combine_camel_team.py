"""
Combine CAMELTrack's association with our team-colour reasoning.

The two are complementary, and neither is sufficient alone:

  - CAMELTrack is a transformer trained on SportsMOT that associates players
    using motion, appearance and body keypoints. It is strong at following a
    player, but has no notion of teams at all - nothing stops it handing a
    white player's identity to a red player after a collision.

  - Our colour logic knows nothing about motion, but it can prove a swap
    happened: if one track's jersey colour changes partway through, that track
    is now following a different person. That is a hard physical fact, not a
    similarity score.

So we take CAMELTrack's tracks and apply three corrections:
  1. assign each track a team by voting torso colour over its whole life
  2. cut any track whose colour flips - it demonstrably changed player
  3. rejoin the fragments, refusing to merge across teams, or across tracks
     that were on court simultaneously (they cannot be one person)

Input is CAMELTrack's MOT-format output, so this does not re-run detection or
tracking.
"""
import argparse
from collections import defaultdict

import cv2
import numpy as np

from track_botsort import (
    torso_colour_vote, split_on_team_flip, assign_teams,
    merge_tracklets, id_color,
)


def read_mot(path):
    """MOT format: frame,id,left,top,width,height,conf,... (1-based frames)"""
    tracks = defaultdict(dict)
    for line in open(path):
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        f, i = int(p[0]) - 1, str(int(p[1]))
        x, y, w, h = (float(v) for v in p[2:6])
        tracks[i][f] = [x, y, x + w, y + h]
    return dict(tracks)


def colour_purity(history, mapping, votes):
    num = den = 0.0
    impure = []
    for final_id in set(mapping.values()):
        labels = []
        for tid in history:
            if mapping[tid] != final_id:
                continue
            for _, v in votes.get(tid, []):
                labels.append("C" if v[0] > v[1] else "W")
        if len(labels) < 20:
            continue
        frac_c = labels.count("C") / len(labels)
        pur = max(frac_c, 1 - frac_c)
        num += pur * len(labels)
        den += len(labels)
        if pur < 0.9:
            impure.append((final_id, len(labels), round(frac_c, 2)))
    return (num / den if den else float("nan")), impure
