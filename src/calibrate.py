"""
Homography calibration for the fixed-camera video.mp4 clip.
Court coordinate convention (matches the WS Tracker app): 150x110 units = 15m x 11m,
1 unit = 0.1m (decimeter). Baseline at y=0, key from x=50.5-99.5, rim at (75,15.75),
three-point arc radius 67.5 centered at rim, free-throw circle radius 18 centered at (75,58).
"""
import cv2
import numpy as np
import json

# pixel -> court-unit correspondences, best-estimate from visual inspection of frame 0
POINTS = [
    # (pixel_x, pixel_y, court_x, court_y), label
    (155, 316, 50.5, 58,  'key_ft_left'),
    (483, 254, 50.5, 0,   'key_baseline_left'),
    (611, 392, 75,   76,  'ft_circle_apex'),
    (652, 316, 99.5, 58,  'key_ft_right'),
]

def compute(points_override=None):
    pts = points_override if points_override is not None else POINTS
    if len(pts) < 4:
        raise ValueError('Need at least 4 point correspondences for a homography')
    src = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
    dst = np.array([[p[2], p[3]] for p in pts], dtype=np.float32)
    H, mask = cv2.findHomography(src, dst, method=0)
    return H

if __name__ == '__main__':
    H = compute()
    print(H)
    with open('/Users/jordanngo/Projects/AI live tracking/data_calibration.json', 'w') as f:
        json.dump({'homography': H.tolist(), 'points': POINTS}, f, indent=2)
    print('saved calibration')
