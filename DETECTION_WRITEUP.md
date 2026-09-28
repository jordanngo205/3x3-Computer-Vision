# Player detection in 3x3 basketball broadcast footage

## What this is

A pipeline that finds every player on court in each frame of a 3x3 basketball
clip, and outputs a body skeleton for each one. Skeletons are shown rather than
just boxes because they make detection quality verifiable by eye: if the limbs
land on the real limbs and follow the player as they move, the model has
genuinely located that person. A box can look plausible while actually
containing two players; a skeleton cannot hide that.

**Output:** `clip01_POSE_v3.mp4` — annotated clip with a live count of players
detected per frame, plus `clip01_pose_dets_v3.json` with every box and the 17
body keypoints per player per frame.

## Pipeline

1. **Detection + pose** — YOLO11m-pose. A single model produces both the
   bounding box and 17 COCO body keypoints per person.
2. **Court filter** — a court boundary polygon; only people whose feet fall
   inside (plus a margin) are kept. This removes the crowd, the bench and the
   camera operators, who are otherwise detected correctly but are not players.
3. **Output** — annotated video and a JSON record of boxes + keypoints, so the
   detections can be fed to a separate tracking/identification stage.

## Three settings that mattered far more than expected

Each of these was found by measuring, not by guessing, and each one roughly
doubled or halved what the system saw.

### 1. Inference resolution (the largest single factor)

The detector resizes each frame to a fixed size before looking at it. The
default is 640px. The clips are 960x540, so at the default the frame is shrunk
and the smaller/more distant players stop being detectable.

| inference size | players found per frame |
|---|---|
| 640 (default) | 3.7 |
| 960 | 7.7 |
| 1280 | 8.0 |

This one setting more than doubled detection. It also explains why an
off-the-shelf research pipeline (CAMELTrack/tracklab, which defaults to 640)
appeared to perform poorly on this footage — it was being starved of
detections, not failing at its actual job.

### 2. Duplicate-removal threshold (NMS IoU)

After detecting, the model discards boxes that overlap heavily, assuming they
are duplicates of the same object. In basketball, players legitimately stand
directly behind one another, so a real player gets discarded as a "duplicate".

| NMS IoU | players found per frame |
|---|---|
| 0.70 (default) | 8.0 |
| 0.85 | 8.6 |
| 0.95 | 11.4 (starts producing genuine duplicates) |

0.85 recovers occluded players without introducing duplicate boxes; verified
visually on crowded frames rather than by the count alone.

### 3. Over-aggressive filtering (a self-inflicted problem)

Two filters were added to remove boxes that wrap *two* players — a real failure
mode of smaller detectors. Both turned out to delete real players:

- **Aspect-ratio rule** (reject boxes wider than 1:1.15): deleted players who
  were lunging or diving, whose boxes are legitimately near-square.
- **Two-skeleton rule** (reject boxes containing two torsos): deleted valid
  detections whenever a neighbouring player stood close enough that their torso
  fell inside the box.

Both were removed. The pose model makes them largely unnecessary: it emits one
skeleton per person, so a genuine two-player box shows up as scattered joints
rather than as two clean skeletons.

**Diagnostic example.** In one frame where two players appeared undetected, the
detector had in fact found all 9 people present. One player was deleted by the
aspect rule (she was lunging), and another by the court boundary (her feet
landed just outside a hand-drawn line). Neither was a detection failure. This
is why the filters were reverted.

## Honest limitations

- **Heavy occlusion.** When a player is almost entirely hidden behind another,
  there is little visual evidence to detect. Counts legitimately dip during
  scrambles and rebounds.
- **Referees.** They stand on and around the court. Whether they are "missed"
  depends on the court boundary, which is a choice, not an error.
- **Court boundary is manual and per-camera.** It is drawn once per camera
  framing and breaks if the camera pans or zooms significantly. The standard
  solution is per-frame camera registration (estimating the court-to-image
  homography every frame), which makes the court model follow the camera.
- **Identity is a separate, harder problem.** Detection answers "where are the
  players". Keeping the *same* ID on the *same* player through contact is
  unsolved here, and is a known open problem: the TrackID3x3 paper
  (arXiv:2503.18282) states its own baseline "cannot handle ID exchanges
  between players" and illustrates the failure during occlusion.

## Reproducing

```bash
python src/pose_detect.py \
  --video same_game_clips/clip01.mp4 \
  --out clip01_POSE_v3.mp4 \
  --save-json clip01_pose_dets_v3.json
```

Defaults: `yolo11m-pose.pt`, confidence 0.25, NMS IoU 0.85, inference size
1280, court margin 40px.
