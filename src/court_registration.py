"""
Camera registration: draw the court once, and it follows the camera.

Replaces the fixed court polygon, which only ever worked for one framing and
silently pointed at the wrong pixels the moment the camera panned or zoomed
(or cut to a replay).

The trick that makes this cheap: a broadcast camera on a tripod only pans,
tilts and zooms. It does not travel. For a camera rotating about its own
centre, the mapping between any two frames of the same scene is a single
homography over the WHOLE image - not just the court plane. So we can register
using every static feature in view (stands, signage, scorer's table, court
lines) rather than trying to find faint court markings on a plain grey floor.

Per frame we estimate the homography from a reference frame, then warp the
court outline through it. The number of RANSAC inliers doubles as a
confidence signal: when the broadcast cuts to a replay or a close-up from a
different camera, the match collapses and we know to skip the frame - which
removes the need for separate shot detection.
"""
import cv2
import numpy as np


class CourtRegistrar:
    def __init__(self, ref_frame, ref_poly, min_inliers=25, downscale=0.5):
        """ref_poly: the court outline drawn once, in ref_frame's pixels."""
        self.min_inliers = min_inliers
        self.scale = downscale
        self.ref_poly = np.asarray(ref_poly, dtype=np.float32).reshape(-1, 1, 2)
        self.detector = cv2.SIFT_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher()

        self.ref_small = self._prep(ref_frame)
        self.ref_kp, self.ref_desc = self.detector.detectAndCompute(self.ref_small, None)

        self.prev_small = None
        self.prev_H = None      # last successful reference -> frame homography
        self.stats = {"direct": 0, "chained": 0, "failed": 0}

    def _prep(self, frame):
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def _match(self, desc_a, kp_a, desc_b, kp_b):
        if desc_a is None or desc_b is None or len(kp_a) < 8 or len(kp_b) < 8:
            return None, 0
        pairs = self.matcher.knnMatch(desc_a, desc_b, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return None, 0
        src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        return H, int(mask.sum()) if mask is not None else 0

    def register(self, frame):
        """Return (court_polygon_in_this_frame, inliers, how) or (None, n, 'failed')."""
        small = self._prep(frame)
        kp, desc = self.detector.detectAndCompute(small, None)

        # preferred: match straight to the reference, so errors never accumulate
        H, inl = self._match(self.ref_desc, self.ref_kp, desc, kp)
        how = "direct"

        # after a long pan the view may barely overlap the reference; step from
        # the previous frame instead and compose with the last good homography
        if (H is None or inl < self.min_inliers) and self.prev_small is not None and self.prev_H is not None:
            prev_kp, prev_desc = self.detector.detectAndCompute(self.prev_small, None)
            H_step, inl_step = self._match(prev_desc, prev_kp, desc, kp)
            if H_step is not None and inl_step >= self.min_inliers:
                H, inl, how = H_step @ self.prev_H, inl_step, "chained"

        self.prev_small = small
        if H is None or inl < self.min_inliers:
            self.stats["failed"] += 1
            return None, inl, "failed"

        self.prev_H = H
        self.stats[how] += 1
        # homography was fitted on downscaled images; lift it back to full res
        S = np.array([[self.scale, 0, 0], [0, self.scale, 0], [0, 0, 1]], dtype=np.float64)
        H_full = np.linalg.inv(S) @ H @ S
        poly = cv2.perspectiveTransform(self.ref_poly, H_full)
        return poly.reshape(-1, 2), inl, how
