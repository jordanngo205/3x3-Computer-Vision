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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--pose-model", default="yolo11m-pose.pt")
    p.add_argument("--ball-model", default="yolo11m.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--ball-conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.85)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--no-ball", action="store_true")
    p.add_argument("--keep-off-court", action="store_true",
                   help="draw everyone, including crowd and bench")
    args = p.parse_args()

    out_path = args.out or args.video.rsplit(".", 1)[0] + "_skeleton.mp4"
    court = None if args.keep_off_court else detect_court(args.video)
    if court is not None:
        print(f"court detected ({len(court)} points)")

    pose = YOLO(args.pose_model)
    ball_model = None if args.no_ball else YOLO(args.ball_model)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    fi = people_total = ball_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = np.zeros_like(frame)
        draw_court(canvas, court)

        res = pose(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        kps = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None

        drawn = 0
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if court is not None and not foot_on_court(court, (x1 + x2) / 2, y2, 15):
                continue
            if kps is None or i >= len(kps):
                continue
            k = kps[i]
            for a, b in EDGES:
                xa, ya = k[a][:2]
                xb, yb = k[b][:2]
                if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                    cv2.line(canvas, (int(xa), int(ya)), (int(xb), int(yb)), SKELETON, 2, cv2.LINE_AA)
            for x, y in k[:, :2]:
                if x > 0 and y > 0:
                    cv2.circle(canvas, (int(x), int(y)), 2, SKELETON, -1, cv2.LINE_AA)
            drawn += 1
        people_total += drawn

        if ball_model is not None:
            br = ball_model(frame, classes=[32], conf=args.ball_conf,
                            imgsz=args.imgsz, verbose=False)[0]
            if br.boxes is not None and len(br.boxes):
                bb = br.boxes.xyxy.cpu().numpy()
                areas = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
                x1, y1, x2, y2 = bb[int(np.argmin(areas))]  # the ball is the small one
                cv2.circle(canvas, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                           max(4, int((x2 - x1) / 2)), BALL, -1, cv2.LINE_AA)
                ball_frames += 1

        writer.write(canvas)
        fi += 1
        if fi % 60 == 0:
            print(f"frame {fi}: {drawn} skeletons", flush=True)

    cap.release()
    writer.release()
    print(f"\ndone. {fi} frames -> {out_path}")
    print(f"skeletons per frame: {people_total / max(1, fi):.1f}")
    if ball_model is not None:
        print(f"ball found in {ball_frames}/{fi} frames ({100 * ball_frames / max(1, fi):.0f}%)")


if __name__ == "__main__":
    main()
