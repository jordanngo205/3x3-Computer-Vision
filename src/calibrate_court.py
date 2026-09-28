"""
Click four known court points once; everything downstream gets real metres.

A homography needs four points whose real-world positions we know. The FIBA 3x3
court is 15 m wide by 11 m deep, with the basket centred 1.575 m from the
baseline, so any four identifiable line intersections are enough.

Doing this by eye from the image alone is unreliable, and a wrong calibration
corrupts every distance silently, so it is worth the minute of clicking.
"""
import argparse
import json

import cv2
import numpy as np

# FIBA 3x3 half court, in centimetres, origin at the bottom-left corner of the
# playing surface as seen from the main camera.
COURT_W, COURT_H = 1500.0, 1100.0
# The outer court corners are cut off in this broadcast framing, so we use the
# key instead: 4.9 m wide, centred on the basket, 5.8 m from baseline to the
# free-throw line. Four points anywhere on the court plane are enough.
KEY_HALF_W, FT_LINE = 245.0, 580.0
POINTS = [
    ("KEY corner: BASELINE-LEFT   (key meets baseline, left side)", (750.0 - KEY_HALF_W, 0.0)),
    ("KEY corner: BASELINE-RIGHT  (key meets baseline, right side)", (750.0 + KEY_HALF_W, 0.0)),
    ("KEY corner: FREE-THROW-RIGHT (far end of key, right side)", (750.0 + KEY_HALF_W, FT_LINE)),
    ("KEY corner: FREE-THROW-LEFT  (far end of key, left side)", (750.0 - KEY_HALF_W, FT_LINE)),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--frame", type=int, default=250)
    p.add_argument("--out", default="court_calibration.json")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read that frame")

    clicks = []
    disp = frame.copy()
    win = "click the court corners  (u = undo, q = quit when done)"

    def redraw():
        nonlocal disp
        disp = frame.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(disp, (x, y), 7, (0, 255, 255), -1)
            cv2.putText(disp, str(i + 1), (x + 10, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if len(clicks) < len(POINTS):
            msg = f"{len(clicks)+1}/4  click: {POINTS[len(clicks)][0]}"
        else:
            msg = "all four set - press q to save"
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(disp, msg, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(POINTS):
            clicks.append((x, y))
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1600, frame.shape[1]), min(900, frame.shape[0]))
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    while True:
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u") and clicks:
            clicks.pop(); redraw()
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()

    if len(clicks) < 4:
        raise SystemExit("need all four points")

    src = np.array(clicks, dtype=np.float32)
    dst = np.array([w for _, w in POINTS], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)

    # sanity check: how far off are the clicked points once projected?
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst, axis=1)
    print(f"reprojection error per corner (cm): {np.round(err, 1)}")

    json.dump({"video": args.video, "frame": args.frame,
               "image_points": [list(map(float, c)) for c in clicks],
               "court_points": [list(w) for _, w in POINTS],
               "court_size_cm": [COURT_W, COURT_H],
               "H_image_to_court": H.tolist()}, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
