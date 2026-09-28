"""
Skeleton-only render: green stick figures and the ball on a black background,
with the court drawn underneath.

Same idea as the reference render we were shown - stripping the picture back to
skeletons makes it obvious whether the system understood the play, because
there is nothing else to look at. It is also the honest view: where pose fails
(replays, close-ups) the skeletons collapse into nonsense instead of hiding
behind the video.

The ball uses the stock COCO "sports ball" class. Expect it to be intermittent:
a 3x3 ball is small, fast and often occluded, and the reference project had to
train a dedicated detector to get ~70% recall.
"""
import argparse

import cv2
import numpy as np
from ultralytics import YOLO

from pose_detect import EDGES, detect_court, foot_on_court

SKELETON = (60, 240, 60)
BALL = (40, 150, 245)
COURT = (70, 78, 70)


def draw_court(canvas, poly):
    if poly is None:
        return
    cv2.polylines(canvas, [poly.astype(np.int32)], True, COURT, 1, cv2.LINE_AA)
