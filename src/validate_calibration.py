"""Project the known FIBA 3x3 court geometry onto the original frame using the
inverse of the computed homography, so calibration accuracy can be checked visually."""
import cv2
import numpy as np
import sys
sys.path.insert(0, '/Users/jordanngo/Projects/AI live tracking/src')
from calibrate import compute, POINTS

H = compute()
H_inv = np.linalg.inv(H)

def court_to_pixel(cx, cy):
    p = H_inv @ np.array([cx, cy, 1.0])
    p = p / p[2]
    return int(round(p[0])), int(round(p[1]))

img = cv2.imread('/private/tmp/claude-501/-Users-jordanngo/68362502-ad51-483c-b3ca-797454131f99/scratchpad/vidframes/background_median.png')
overlay = img.copy()

def draw_court_line(pts_court, color=(0,255,0), thickness=2):
    px = [court_to_pixel(cx, cy) for cx, cy in pts_court]
    for i in range(len(px)-1):
        cv2.line(overlay, px[i], px[i+1], color, thickness)

def draw_circle_arc(cx, cy, r, a0, a1, color=(0,255,255), n=60):
    pts = []
    for i in range(n+1):
        a = a0 + (a1-a0) * i / n
        pts.append((cx + r*np.cos(a), cy + r*np.sin(a)))
    draw_court_line(pts, color)

# baseline
draw_court_line([(0,0),(150,0)], (255,0,0), 2)
# sidelines
draw_court_line([(0,0),(0,110)], (255,0,0), 2)
draw_court_line([(150,0),(150,110)], (255,0,0), 2)
# far boundary
draw_court_line([(0,110),(150,110)], (255,0,0), 2)
# key rectangle
draw_court_line([(50.5,0),(50.5,58),(99.5,58),(99.5,0)], (0,255,0), 2)
# free-throw circle (full)
draw_circle_arc(75, 58, 18, 0, 2*np.pi, (0,200,255))
# three-point arc (baseline to baseline)
import math
cx, cy, r = 75, 15.75, 67.5
a_left = math.atan2(0 - cy, 9.36 - cx)
a_right = math.atan2(0 - cy, 140.64 - cx)
draw_circle_arc(cx, cy, r, a_left, a_right, (255,0,255))

# mark calibration points used
for px, py, cx, cy, label in POINTS:
    cv2.circle(overlay, (px, py), 6, (0,0,255), -1)
    cv2.putText(overlay, label, (px+8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)

cv2.imwrite('/private/tmp/claude-501/-Users-jordanngo/68362502-ad51-483c-b3ca-797454131f99/scratchpad/vidframes/calib_check.png', overlay)
print('saved overlay')
