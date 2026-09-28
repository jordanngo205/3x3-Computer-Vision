"""
Project tracked player pixel positions into top-down court coordinates using the
calibrated homography, and render a coverage map on a schematic FIBA 3x3 half-court.
Court convention: 150x110 units = 15m x 11m (matches the WS Tracker app), baseline y=0.
"""
import json
import numpy as np
import cv2
import sys
sys.path.insert(0, '/Users/jordanngo/Projects/AI live tracking/src')
from calibrate import compute

H = compute()

def pixel_to_court(px, py):
    p = H @ np.array([px, py, 1.0])
    p = p / p[2]
    return p[0], p[1]

data = json.load(open('/Users/jordanngo/Projects/AI live tracking/output_tracks.json'))
trails = data['trails']
track_team = data['track_team']

MIN_FRAMES = 30
main_tracks = {tid: pts for tid, pts in trails.items() if len(pts) >= MIN_FRAMES}

court_trails = {}
for tid, pts in main_tracks.items():
    ct = [(f, *pixel_to_court(x, y)) for f, x, y in pts]
    court_trails[tid] = ct

# --- render schematic top-down court, scaled up for visibility ---
SCALE = 6  # px per court-unit (unit=decimeter) -> 150*6=900, 110*6=660
W, H_ = 150*SCALE, 110*SCALE
canvas = np.full((H_, W, 3), (45, 87, 46), dtype=np.uint8)  # green-ish floor like the tracker app

def cu(x, y):
    return int(x*SCALE), int(y*SCALE)

def draw_line(p1, p2, color=(255,255,255), th=2):
    cv2.line(canvas, cu(*p1), cu(*p2), color, th)

def draw_arc(cx, cy, r, a0, a1, color=(255,255,255), th=2, n=80):
    pts = [(cx+r*np.cos(a0+(a1-a0)*i/n), cy+r*np.sin(a0+(a1-a0)*i/n)) for i in range(n+1)]
    for i in range(len(pts)-1):
        draw_line(pts[i], pts[i+1], color, th)

# court boundary
draw_line((0,0),(150,0)); draw_line((0,0),(0,110)); draw_line((150,0),(150,110)); draw_line((0,110),(150,110))
# key
draw_line((50.5,0),(50.5,58)); draw_line((99.5,0),(99.5,58)); draw_line((50.5,58),(99.5,58))
# free-throw circle
draw_arc(75,58,18,0,2*np.pi)
# three-point arc
import math
cx,cy,r = 75,15.75,67.5
a_l = math.atan2(0-cy, 9.36-cx); a_r = math.atan2(0-cy, 140.64-cx)
draw_arc(cx,cy,r,a_l,a_r)
# rim marker
cv2.circle(canvas, cu(75,15.75), 5, (0,140,255), -1)

palette = {
    'Canada': (255,255,255),
    'Romania': (0,215,255),
    'other': (150,150,150),
}
count_used = 0
for tid, ct in sorted(court_trails.items(), key=lambda kv: -len(kv[1])):
    team = track_team[tid]
    color = palette.get(team, (200,200,200))
    pts = [(x,y) for f,x,y in ct]
    # clip to a reasonable extended court-area to avoid wild extrapolated points dominating the plot
    pts = [(x,y) for x,y in pts if -10 <= x <= 160 and -10 <= y <= 120]
    if len(pts) < 5:
        continue
    count_used += 1
    for i in range(1, len(pts)):
        draw_line(pts[i-1], pts[i], color, 2)
    lx, ly = pts[-1]
    cv2.putText(canvas, f'#{tid} {team}', cu(lx+1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

print('tracks plotted:', count_used)
cv2.imwrite('/Users/jordanngo/Projects/AI live tracking/output_topdown_coverage.png', canvas)
print('saved output_topdown_coverage.png')
