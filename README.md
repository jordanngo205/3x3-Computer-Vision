# 3x3 Basketball Computer Vision

Player detection, tracking and team assignment on FIBA 3x3 broadcast footage, with
positions projected onto a calibrated top-down court in real centimetres.

`TOPDOWN_LONG11.mp4` is the output: the broadcast on the left, the court model on the
right, one dot per player in their kit colour.

Clip: Canada v Germany, FIBA 3x3 Women's Series quarter-final. 1984x1112, variable
frame rate (~43 fps average), 52.9 seconds.

## The problem

Detecting people is easy. The hard parts, in the order they bit:

1. **Keeping one identity per player.** Players screen and collide constantly, and a
   tracker opens a new id every time one is occluded and reappears.
2. **Telling the teams apart.** Red kit vs white kit sounds trivial, but a single
   frame's torso patch is far too noisy, and the court grey sits at the same hue as
   a red kit. Judging each frame on its own made players flicker between colours.
3. **Excluding everyone who is not playing.** Referees, substitutes on the bench, and
   staff standing past the sideline all get detected as people.

## How it works

**Detection** — YOLO11 pose (`yolo11m-pose`) at `imgsz=1280`, NMS IoU `0.85`. The
default IoU of 0.70 suppresses a player standing behind another as a duplicate.
17 COCO keypoints per player.

**Ground contact** — the ankle keypoints, not the bottom of the box. The box bottom
drifts with pose and motion blur; the ankles are where the player touches the floor.

**Court calibration** — a homography from four clicked key corners maps image pixels
into court centimetres (FIBA 3x3 is a 1500x1100 cm half court with a single basket).
Each clip needs its own calibration: the camera framing differs between clips, and
using the wrong one silently places bench players on court.

**Non-players** — decided in court metres rather than by a pixel polygon. A substitute
sitting past the sideline is genuinely off the court whatever the floor looks like
there. Referees are removed by a low-saturation, low-value torso test.

**Tracking** — BoT-SORT, which uses motion prediction and global motion compensation
for the panning camera.

**Tracklet merging** — BoT-SORT still opened a new id whenever a player was occluded:
89 ids for six players over 52 seconds. Two fragments are rejoined only if they never
appear in the same frame (one person is not in two places), the gap is short, and the
distance across it is coverable at running speed.

**Team assignment** — one kit per *track*, not per frame, chosen by minimising a
colour cost (how far each detection sits on the wrong side of the kit split, over the
track's whole life) plus a constraint cost that pushes every six-player frame towards
a 3/3 split. Solved by coordinate descent. Colour decides the tracks it can see
clearly; the 3-on-3 rule decides the ones it cannot.

## Results

Measured on the clip above:

| | tracks | ID breaks | kit changes mid-track | 3/3 rate |
|---|---|---|---|---|
| greedy box-overlap linking, per-frame kit | 89 | 37 | 9 | 80% |
| BoT-SORT + merging + per-track kit | 57 | 7 | **1** | **90%** |

**On the metrics.** A track that changes kit mid-life is provably wrong — one person
wears one kit — so that number should be zero, and it is the honest measure. The 3/3
rate is reported as a *consequence*, not a target: an earlier version forced three red
and three white in every six-player frame, which made its "100% correct" meaningless.

**Open problems.** All six players are detected in roughly half of frames; the rest
lose someone to occlusion. Telling two team-mates apart is unsolved here and in the
published state of the art — TrackID3x3 (arXiv:2503.18282) states its own baseline
"cannot handle ID exchanges between players". Ball detection is not implemented.

## Measured dead ends

Kept here because they cost real time:

- **Tiled inference** (running the detector on overlapping vertical strips) scored
  5.52 people/frame against plain inference's 5.54 — the tiles find nothing new. Its
  duplicate-merging step, which called two skeletons the same person when their torso
  centres were within 0.35x height, was deleting players standing close together.
- **A self-trained appearance embedding** on pseudo-labels from our own tracker could
  not distinguish a red jersey from a white one (same-team distance 0.616, cross-team
  0.640). Nearly every training clip contained a single team, so "same track within 45
  frames" positives were near-duplicate images the model matched on pose and background.
- **Aspect-ratio and two-skeleton filters** deleted real players — lunging players and
  players standing beside a team-mate.

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Calibrate a clip by clicking four key corners (baseline left, baseline right,
free-throw right, free-throw left):

```bash
.venv/bin/python src/calibrate_court.py --video "clip.mov" --out court_calibration_long.json
```

Then render:

```bash
.venv/bin/python src/topdown_render.py \
  --video "Canada v Ger Long.mov" \
  --calib court_calibration_long.json \
  --cache-detections cache_long.pkl \
  --out TOPDOWN_LONG11.mp4 \
  --save-positions positions_long11.json
```

`--cache-detections` is worth using. Detection and tracking are the whole cost and do
not depend on the team logic, so caching them turns an hour-long experiment on the
labelling into a one-second one. Add `--device mps` on Apple Silicon — ultralytics
otherwise selects CPU, which is about 2.5x slower.

## Files

| | |
|---|---|
| `src/topdown_render.py` | the pipeline — detection, tracking, merging, team assignment, rendering |
| `src/pose_detect.py` | detection with skeletons, automatic court-region detection |
| `src/track_botsort.py` | camera-shot detection, team-colour helpers |
| `src/calibrate_court.py` | click tool for the four calibration points |
| `court_calibration_long.json` | homography for `Canada v Ger Long.mov` |

Source footage and model weights are not in the repo. `yolo11m-pose.pt` is downloaded
automatically by ultralytics on first run.
