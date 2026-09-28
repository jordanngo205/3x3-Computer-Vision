"""
Defensive attention ("gravity") from court positions.

For every attacker in every frame we measure, in metres, how close the nearest
defender is and how many defenders are within a given radius. This is the
measurement the court calibration exists for - in pixels the same numbers are
meaningless, because a player near the camera is several times larger than one
at the far side.

No player identity is needed: gravity is a property of a position and the
opposing team's positions, not of a name.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

TIGHT_CM = 150.0     # a defender this close is guarding you
OPEN_CM = 350.0      # nobody within this is an open look
NEAR_CM = 200.0      # radius used for "how many defenders nearby"


def load(path):
    d = json.load(open(path))
    fr = {int(f): ps for f, ps in d["frames"].items() if ps}
    return d.get("fps", 30.0), fr


def gravity_frame(attackers, defenders):
    out = []
    if not defenders:
        return out
    D = np.array([[p["x"], p["y"]] for p in defenders])
    for a in attackers:
        v = D - np.array([a["x"], a["y"]])
        dist = np.hypot(v[:, 0], v[:, 1])
        out.append({
            "x": a["x"], "y": a["y"], "team": a["team"],
            "nearest_cm": float(dist.min()),
            "n_within": int((dist < NEAR_CM).sum()),
        })
    return out


def summarise(rows, label):
    if not rows:
        print(f"{label}: no data")
        return
    near = np.array([r["nearest_cm"] for r in rows])
    within = np.array([r["n_within"] for r in rows])
    paint = np.array([1 if (505 <= r["x"] <= 995 and r["y"] <= 580) else 0 for r in rows])
    print(f"\n{label}   ({len(rows)} player-frames)")
    print(f"  nearest defender      : median {near.mean()/100:5.2f} m   "
          f"(25th {np.percentile(near,25)/100:.2f}, 75th {np.percentile(near,75)/100:.2f})")
    print(f"  tightly guarded <1.5 m: {100*(near<TIGHT_CM).mean():5.1f}%")
    print(f"  open >3.5 m           : {100*(near>OPEN_CM).mean():5.1f}%")
    print(f"  defenders within 2 m  : {within.mean():5.2f}")
    print(f"  time in the paint     : {100*paint.mean():5.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--positions", required=True)
    p.add_argument("--out", default=None, help="write per-frame gravity to JSON")
    args = p.parse_args()

    fps, fr = load(args.positions)
    per_team = defaultdict(list)
    per_frame = {}
    for f, ps in sorted(fr.items()):
        C = [q for q in ps if q["team"] == "C"]
        W = [q for q in ps if q["team"] == "W"]
        rows = gravity_frame(C, W) + gravity_frame(W, C)
        per_frame[f] = rows
        for r in rows:
            per_team[r["team"]].append(r)

    print(f"frames analysed: {len(per_frame)}  ({len(per_frame)/fps:.0f}s of play)")
    summarise(per_team.get("C", []), "COLOURED KIT attacking")
    summarise(per_team.get("W", []), "WHITE KIT attacking")

    # the most-guarded moments: where does the defence collapse?
    allrows = [r for rows in per_frame.values() for r in rows]
    crowded = [r for r in allrows if r["n_within"] >= 2]
    if crowded:
        xs = np.array([r["x"] for r in crowded]); ys = np.array([r["y"] for r in crowded])
        print(f"\ndouble-teams (2+ defenders within 2 m): {len(crowded)} player-frames "
              f"({100*len(crowded)/len(allrows):.1f}%)")
        print(f"  typically at x {np.median(xs):.0f} cm, y {np.median(ys):.0f} cm from the baseline")

    if args.out:
        json.dump({"fps": fps, "frames": {str(k): v for k, v in per_frame.items()}},
                  open(args.out, "w"))
        print(f"\nper-frame gravity -> {args.out}")


if __name__ == "__main__":
    main()
