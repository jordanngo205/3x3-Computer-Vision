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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mot", required=True, help="CAMELTrack MOT output")
    p.add_argument("--video", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--min-track-frames", type=int, default=15)
    p.add_argument("--merge-max-overlap", type=int, default=10)
    p.add_argument("--merge-gap-reach", type=float, default=0.15)
    args = p.parse_args()

    history = {t: fb for t, fb in read_mot(args.mot).items() if len(fb) >= args.min_track_frames}
    print(f"CAMELTrack gave {len(history)} tracks (after dropping very short ones)")

    # colour vote per detection, once, straight off the video
    by_frame = defaultdict(list)
    for tid, fb in history.items():
        for f, box in fb.items():
            by_frame[f].append((tid, box))
    votes = defaultdict(list)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, box in by_frame.get(fi, []):
            v = torso_colour_vote(frame, box)
            if v:
                votes[tid].append((fi, v))
        fi += 1
    cap.release()

    teams_before = assign_teams({t: votes.get(t, []) for t in history})
    pur_before, imp_before = colour_purity(history, {t: t for t in history}, votes)
    print(f"BEFORE our corrections: {len(history)} identities, purity {pur_before:.3f}, "
          f"{len(imp_before)} identities change team mid-life")

    history, votes, splits = split_on_team_flip(history, votes)
    print(f"  split {splits} tracks that provably changed player -> {len(history)} fragments")

    teams = assign_teams({t: votes.get(t, []) for t in history})
    mapping, merged = merge_tracklets(history, teams, args.merge_max_overlap, args.merge_gap_reach)
    pur_after, imp_after = colour_purity(history, mapping, votes)
    n_after = len(set(mapping.values()))
    print(f"  merged {merged} fragments back together")
    print(f"AFTER our corrections:  {n_after} identities, purity {pur_after:.3f}, "
          f"{len(imp_after)} identities change team mid-life")

    if not args.out:
        return

    order = sorted(set(mapping.values()),
                   key=lambda fid: -sum(len(history[t]) for t in history if mapping[t] == fid))
    display = {fid: i for i, fid in enumerate(order)}
    render = defaultdict(list)
    for tid, fb in history.items():
        for f, box in fb.items():
            render[f].append((display[mapping[tid]], box, teams.get(tid, "?")))

    cap = cv2.VideoCapture(args.video)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, box, team in render.get(fi, []):
            x1, y1, x2, y2 = [int(v) for v in box]
            c = id_color(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f"#{tid} {team[:1]}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        writer.write(frame)
        fi += 1
    cap.release()
    writer.release()
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
