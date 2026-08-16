"""The equal-score tie-break in locate.peak_loc.

cv2.minMaxLoc resolves an exact tie by lowest raster index (top-left). These
tests pin the replacement rule -- among equal-scoring positions, take the one
whose patch center is nearest the image center -- and pin that a peak decided by
a strict maximum is left exactly as it was.
"""

import cv2
import numpy as np

from src.locate import peak_loc, _center_dist2


IMG = (100, 100)      # the notional full search channel
M = 20                # template size, so a valid map is 81x81


def _empty_map():
    return np.zeros((IMG[0] - M + 1, IMG[1] - M + 1), dtype=np.float32)


def test_tie_resolves_toward_center_not_top_left():
    res = _empty_map()
    h, w = res.shape
    # Two exactly-equal maxima. In patch-center coordinates (index + M/2) the
    # image center is (49.5, 49.5): (45, 45) sits ~6.4 px away, (5, 5) ~48.8.
    near, far = (45, 45), (5, 5)
    res[near[1], near[0]] = 0.9
    res[far[1], far[0]] = 0.9

    (x, y), score = peak_loc(res, M, img_shape=IMG)
    assert (x, y) == near
    assert score == np.float32(0.9)

    # Document what changed: OpenCV would have taken the top-left one.
    _, _, _, cv_loc = cv2.minMaxLoc(res)
    assert cv_loc == far


def test_strict_maximum_is_untouched():
    rng = np.random.default_rng(0)
    res = rng.random(_empty_map().shape, dtype=np.float32)
    res[7, 63] = 2.0                       # a unique, decidedly off-center max

    (x, y), score = peak_loc(res, M, img_shape=IMG)
    _, cv_score, _, cv_loc = cv2.minMaxLoc(res)
    assert (x, y) == cv_loc
    assert score == cv_score


def test_flat_map_returns_the_center():
    # A degenerate window (zero variance) makes the whole map one value; every
    # position ties, so the most central one must come back rather than (0, 0).
    res = np.zeros(_empty_map().shape, dtype=np.float32)
    (x, y), _ = peak_loc(res, M, img_shape=IMG)

    h, w = res.shape
    best = min(((i, j) for j in range(h) for i in range(w)),
               key=lambda p: (p[0] + M / 2.0 - 49.5) ** 2
               + (p[1] + M / 2.0 - 49.5) ** 2)
    assert (x, y) == best
    assert cv2.minMaxLoc(res)[3] == (0, 0)


def test_origin_shifts_the_frame():
    # The same map cropped out of the image at a different origin must prefer a
    # different tied peak -- the crop offset is part of the distance.
    res = _empty_map()
    res[0, 0] = res[40, 40] = 0.5

    at_zero, _ = peak_loc(res, M, origin=(0, 0), img_shape=IMG)
    shifted, _ = peak_loc(res, M, origin=(40, 40), img_shape=IMG)
    assert at_zero == (40, 40)     # (50,50) in patch coords, nearest the center
    assert shifted == (0, 0)       # already at (50,50) once the origin is added


def test_center_dist2_is_zero_at_the_center():
    assert _center_dist2(49.5, 49.5, 100) == 0.0
    assert _center_dist2(0.0, 49.5, 100) > 0.0
