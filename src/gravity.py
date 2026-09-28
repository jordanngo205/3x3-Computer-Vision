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
