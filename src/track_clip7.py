"""
v7 - offline gap interpolation + outlier-gated assignment, replacing v3-v6's
forward-only velocity extrapolation.

WHY THIS CHANGE: every established online tracker (SORT/DeepSORT/ByteTrack/BoT-SORT)
extrapolates forward from a Kalman-filtered velocity, because in a REAL broadcast or
live feed, future frames don't exist yet - forward extrapolation is the only option.
That's not our situation. We already have the entire clip in memory before we track
a single frame. When Canada #2 went missing at frame 49 and reappeared at frame 56,
v3-v6 had to guess forward from a (possibly noisy) velocity with no way to check
itself - and one noisy pair of real detections (a brief misassignment) sent it
running off past the edge of the frame before real detections dragged it back.

We already know where frame 56 is. So instead of guessing forward, INTERPOLATE
between the confirmed position before the gap and the confirmed position after it.
This can't run off to nowhere - it's mathematically anchored at both ends. This is
the same idea sports-tracking research calls "global tracklet association" (using
information from both directions, not just the past, to fix gaps) - the natural
fit for offline/batch analysis instead of live tracking.

Second fix: outlier-gated assignment. A single-frame misassignment (grabbing a
nearby wrong detection) is what poisoned the velocity estimate in v6. Here, a
candidate match is only accepted if the implied per-frame speed is physically
plausible (scaled by how many frames since the track's last confirmed sighting) -
an implausible "jump" is rejected outright and the track is left as a gap to be
interpolated later, rather than accepting bad data now.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from sklearn.cluster import KMeans
import json
import sys
import argparse

BASE = '/Users/jordanngo/Projects/AI live tracking'

parser = argparse.ArgumentParser()
parser.add_argument('--video', default=f'{BASE}/video.mp4')
parser.add_argument('--tag', default='v7', help='suffix for output files, e.g. test1 -> output_annotated_test1.mp4')
parser.add_argument('--court-poly', default=None,
                     help='comma-separated x,y pairs overriding the default court polygon '
                          '(needed when frame resolution/camera framing differs from video.mp4)')
parser.add_argument('--team-white', default='Canada', help='name for the white-jersey team')
parser.add_argument('--team-color', default='Romania', help='name for the colored-jersey team (any color, not just blue)')
parser.add_argument('--max-width', type=int, default=None,
                     help='downscale frames to this width before processing (preserves aspect ratio). '
                          'Needed for high-res/long clips - holding thousands of full-res frames plus '
                          'YOLO results in RAM at once caused a silent OOM kill on a 4216-frame, '
                          '1986x1118 clip (would have needed ~28GB just for raw pixels).')
args = parser.parse_args()

VIDEO = args.video
OUT_VIDEO = f'{BASE}/output_annotated_{args.tag}.mp4'
OUT_JSON = f'{BASE}/output_tracks_{args.tag}.json'
TEAM_WHITE = args.team_white
TEAM_COLOR = args.team_color

if args.court_poly:
    nums = [float(v) for v in args.court_poly.split(',')]
    COURT_POLY = np.array(list(zip(nums[0::2], nums[1::2])), dtype=np.int32)
else:
    COURT_POLY = np.array([
        (105, 300), (30, 355), (0, 400), (0, 540), (650, 540),
        (760, 430), (830, 330), (830, 260), (350, 248),
    ], dtype=np.int32)

def foot_on_court(fx, fy):
    return cv2.pointPolygonTest(COURT_POLY, (float(fx), float(fy)), False) >= 0

def classify_team_hsv(frame, box):
    """Old fixed-rule classifier. Too fragile to use per-detection (a bent-over/angled
    pose can shift the sampled patch enough to drop under the white threshold - this is
    exactly what caused a real player to go untracked for 48 straight frames earlier).
    Kept only as a bootstrap: used to NAME the two clusters KMeans finds (see
    color_histogram/cluster_team below), never to classify an individual detection.

    Colored-team mask is hue-agnostic (any saturated, reasonably bright fabric that isn't
    white) rather than a fixed blue range - this rule was originally written for Canada
    (white) vs Romania (blue), but the same two-cluster bootstrap needs to work for any
    single-colored opponent jersey (e.g. Japan's red), since only ONE color needs telling
    apart from white in any given clip."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return None
    w, h = x2 - x1, y2 - y1
    jx1, jx2 = x1 + int(w*0.25), x1 + int(w*0.75)
    jy1, jy2 = y1 + int(h*0.18), y1 + int(h*0.5)
    if jy2 <= jy1 or jx2 <= jx1:
        return None
    patch = frame[jy1:jy2, jx1:jx2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    total = h_.size
    white_mask = (s_ < 15) & (v_ > 110)
    # circular hue distance from the auto-detected opponent color (set by
    # detect_opponent_hue below, before this function is ever called) - NOT "any
    # saturated pixel": that first attempt also fired on the white team's own trim/logo
    # colors (e.g. Canada's red maple leaf), which silently merged both clusters into one
    # team. Restricting to a hue band around the one color that's actually dominant across
    # the whole clip's "colored" crops avoids that.
    hue_dist = np.minimum(np.abs(h_.astype(int) - OPPONENT_HUE_CENTER[0]), 180 - np.abs(h_.astype(int) - OPPONENT_HUE_CENTER[0]))
    color_mask = (hue_dist <= HUE_HALF_WIDTH) & (s_ > 50) & (v_ > 60)
    white_frac = white_mask.sum() / total
    color_frac = color_mask.sum() / total
    if white_frac < 0.10 and color_frac < 0.10:
        return None
    return TEAM_WHITE if white_frac >= color_frac else TEAM_COLOR

HUE_HALF_WIDTH = 20
OPPONENT_HUE_CENTER = [110]  # mutable holder, overwritten by detect_opponent_hue() below

def detect_opponent_hue(dets, frames, sample=400, video_path=None, resize_scale=1.0):
    """Auto-detect the opponent team's dominant jersey hue for THIS clip, instead of
    assuming blue - generalizes classify_team_hsv to any single opponent color (tested
    against Romania/blue and Japan/red) without hand-tuning a hue range per video.

    BUG #1 (found via direct regression check on video.mp4): sampling the wide torso
    window (25-75%/18-50%) pulled in bare skin, which outvoted the real jersey color.
    Tightening to the 35-65%/15-42% jersey-only window (matching color_histogram) didn't
    fully fix it either.

    BUG #2 (found by printing raw hue/saturation per box): counting every "saturated,
    non-white" PIXEL regardless of source means a small but vivid accent - Canada's red
    maple-leaf logo, a few percent of the crop - adds just as many hue votes as an actual
    solid-blue jersey, because pixel-counting has no notion of "which color this garment
    actually is." Fix: vote per DETECTION, not per pixel - only count a hue if it's the
    MAJORITY color of that specific crop (>35% of the crop's total area). A small logo
    can never win that vote; a real jersey filling most of the torso crop always can.

    BUG #3 (found on real test.mov, downscaled to 640px wide to fit in memory): at that
    scale a player's torso crop is only a few hundred pixels, too small and noisy to
    reliably clear the 35% single-hue-dominance bar even for a real solid jersey - every
    sampled detection got skipped, hue_votes stayed empty, and it silently fell back to
    the blue default even though the actual opponent (Japan) wears red. Fix: if given the
    original video path, re-read just these ~400 sampled detections directly from disk at
    FULL original resolution (cheap - it's a one-time sample, not the whole clip) instead
    of using the downscaled in-memory frames used everywhere else in the pipeline."""
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(dets), size=min(sample, len(dets)), replace=False)
    hue_votes = np.zeros(180, dtype=np.int64)
    full_res_cap = None
    if video_path is not None and resize_scale != 1.0:
        full_res_cap = cv2.VideoCapture(video_path)
    for i in idxs:
        d = dets[i]
        box = d['box']
        frame = frames[d['fi']]
        if full_res_cap is not None:
            full_res_cap.set(cv2.CAP_PROP_POS_FRAMES, d['fi'])
            ok, full_frame = full_res_cap.read()
            if ok:
                frame = full_frame
                box = [v / resize_scale for v in box]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
        if x2 <= x1 or y2 <= y1:
            continue
        w, h = x2 - x1, y2 - y1
        jx1, jx2 = x1 + int(w*0.35), x1 + int(w*0.65)
        jy1, jy2 = y1 + int(h*0.15), y1 + int(h*0.42)
        if jy2 <= jy1 or jx2 <= jx1:
            continue
        patch = frame[jy1:jy2, jx1:jx2]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h_, s_, v_ = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        total_px = h_.size
        colored = (s_ > 50) & (v_ > 60)
        if colored.sum() / total_px < 0.35:
            continue  # no single color dominates this crop - not a useful vote
        crop_hist = np.bincount(h_[colored].ravel(), minlength=180)
        # circularly smooth BEFORE picking the peak - found via direct debugging on real
        # test.mov (Japan/red): a red garment's hue straddles the 0/180 wraparound, so its
        # pixels split across bins near 179 AND bins near 0-5. Taking the raw single-bin
        # argmax saw ~28% in one bin and ~20% in the other, neither clearing 35%
        # dominance, even though together they're clearly ~48% of the crop, i.e. one real
        # dominant color. Blue (this check's only tested case before) doesn't straddle the
        # boundary, which is why this passed regression testing before but broke on red.
        crop_kernel = 5
        crop_padded = np.concatenate([crop_hist[-crop_kernel:], crop_hist, crop_hist[:crop_kernel]])
        crop_smoothed = np.convolve(crop_padded, np.ones(2*crop_kernel+1), mode='valid')
        peak_bin = int(np.argmax(crop_smoothed))
        if crop_smoothed[peak_bin] / total_px < 0.35:
            continue  # the colored pixels themselves are hue-scattered, not one solid color
        hue_votes[peak_bin] += 1
    if full_res_cap is not None:
        full_res_cap.release()
    if hue_votes.sum() < 10:
        return 110  # fallback: blue, the original default
    # smooth circularly (hue wraps at 0/180) before taking the peak, so a color like red
    # (which straddles the 0/180 boundary) isn't split into two weaker peaks
    kernel = 5
    padded = np.concatenate([hue_votes[-kernel:], hue_votes, hue_votes[:kernel]])
    smoothed = np.convolve(padded, np.ones(2*kernel+1)/(2*kernel+1), mode='valid')
    return int(np.argmax(smoothed))

def color_histogram(frame, box):
    """Direct HSV color histogram of the jersey crop - matches TrackID3x3's own
    attribute-identification approach (calculate_color_hist.py). Validated at 90.1%
    agreement with the HSV ground truth on this clip, vs 67.4% for a general-purpose
    SigLIP embedding + clustering - a general vision model doesn't prioritize color the
    way a plain histogram forces the clustering to.
    Sampling window tightened (35-65% width, 15-42% height, vs the original 25-75%/
    18-50%) after finding that during player-on-player occlusion, the wider window
    picked up contaminating jersey color from the overlapping opponent, causing real
    misclassifications on otherwise-clean detections. Verified this fixes both known
    occlusion-contaminated frames and raises overall agreement to 91.4%."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return None
    w, h = x2 - x1, y2 - y1
    jx1, jx2 = x1 + int(w*0.35), x1 + int(w*0.65)
    jy1, jy2 = y1 + int(h*0.15), y1 + int(h*0.42)
    if jy2 <= jy1 or jx2 <= jx1:
        return None
    patch = frame[jy1:jy2, jx1:jx2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()
