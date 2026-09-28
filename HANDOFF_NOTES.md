# Handoff notes — resume here after reboot

## Where we left off (2026-08-08)

**Goal:** the current tracking pipeline (`src/track_clip7.py`, "v7") isn't good enough.
Decided path forward: **don't chase the TrackID3x3 external dataset** — instead,
train a small appearance/ReID model using our OWN already-tracked footage as
self-generated ("pseudo-labeled") training data. See rationale below.

## Why we dropped the TrackID3x3 dataset idea

We downloaded parts of the TrackID3x3 Google Drive (`Indoor videos/`, `Indoor output/`,
`color_histograms/Outdoor/`). Checked all three:

- `Indoor videos/raw/` — just 41 raw `.mp4` clips, no annotations at all.
- `Indoor output/` — **NOT ground truth.** This is the TrackID3x3 baseline's own
  CAMELTrack *predictions* (confirmed via `main.log`: `GT=0`, "no ground truth
  detections"). We already tested integrating CAMELTrack into our own pipeline in
  `track_clip7.py` and it made results measurably worse (94→97→113 flagged issues
  in our audit). Training on this folder = training to imitate a tool we already
  rejected.
- `color_histograms/Outdoor/` — pre-extracted 512-dim embeddings per tracklet,
  split into `ID_merging`/`non_ID_merging` folders (same-identity vs different-identity
  signal). Outdoor only, and it's embeddings not raw crops, so limited standalone use.

Mapped the entire Drive tree (`color_histograms/`, `output/{CAMELTrack_outputs,
jersey-number-pipeline}/`, `videos/{Drone,Indoor,Outdoor}/raw/`) — there is no
separate annotations/ground-truth folder anywhere in it, despite the GitHub repo's
license text claiming "videos, annotations and intermediate files" are all there.

## The actual plan (agreed, not yet started)

Bootstrap training data from our OWN videos + OWN pipeline's confident output:

1. For each existing run (`output_tracks_v7.json` / `_test1` / `_test2` / `_realtest`,
   paired with the matching `output_team_classifier_*.pkl`), pull every **real**
   (non-interpolated — i.e. NOT in the `predicted` set) detection box per track.
2. Crop the player image at that box from the source video frame
   (`video.mp4`, `Test1.mp4`, `Test2.mp4`, `real test.mov` respectively).
3. Same track_id nearby in time = positive pair; different track_ids visible in the
   same frame = hard negative pair.
4. Train a small embedding (triplet/contrastive) model on these crops — this should
   make the appearance-matching step in `track_clip7.py` (currently a hand-tuned HSV
   histogram + Bhattacharyya distance) more robust, especially through occlusion.
5. Deliberately **hold out `real test.mov`'s results from training** — use it only
   to check afterward whether the trained model actually improves tracking on the
   one clip that matters most (the full real broadcast clip).

Caveat already flagged to the user: this can only learn from frames the pipeline is
already confident about, so it makes the model more consistent/robust, but won't by
itself fix the specific low-confidence frames v7 already struggles with.

Frame counts already confirmed available (before the environment broke):
- v7 (`video.mp4`): 3 kept tracks, 625 confident real frames
- realtest (`real test.mov`): 3 kept tracks, 6388 confident real frames
- test1 / test2: blocked (see below), not yet counted

## Current blocker: environment broke mid-task (iCloud eviction)

`~/Documents` is enrolled in iCloud's "Desktop & Documents Folders" sync. Several
files in this project went "dataless" (evicted to iCloud-only) **despite having
44GB+ free disk** — "Optimize Mac Storage" evicts opportunistically, not just when
disk is full. Affected and still broken as of last check:

- `output_tracks_test1.json`, `output_tracks_test2.json` — reads time out (`Errno 60`)
- `.venv`'s `torch` AND `sklearn` — imports time out the same way

`brctl status` also showed 3 iCloud sync containers stuck in
`blocked-app-uninstalled` / `SYNC DISABLED` state — looks like a wedged `bird`/`cloudd`
daemon, not just a settings issue.

Steps taken so far, in order:
1. Tried `brctl download <file>` manually — no effect.
2. Tried Finder right-click → "Download Now" on the whole folder — no effect (user
   confirmed clicking it did nothing).
3. Turned OFF "Optimize Mac Storage" in System Settings → iCloud Drive — stops
   future eviction, but did NOT retroactively restore the already-dataless files.
4. **Next step (about to happen): reboot the Mac** to force-restart the stuck
   `bird`/`cloudd` daemons. This is why the terminal is closing now.

## To resume after reboot

Re-run this check first — if it prints "torch ok" and "test1 ok" we're unblocked:

```bash
cd "/Users/jordanngo/Documents/Project/AI live tracking"
python3 -c "import json; d=json.load(open('output_tracks_test1.json')); print('test1 ok', len(d['tracks']))"
.venv/bin/python -c "import torch; print('torch ok', torch.__version__)"
```

If still timing out after reboot, the daemon-level issue is deeper than expected —
worth checking `brctl status` again for the same `blocked-app-uninstalled` containers,
or as a last resort signing out/into iCloud.

Once unblocked: write the crop-extraction script described in step 1-3 above (nothing
else needed from the user — all inputs already exist locally).
