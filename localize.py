"""
CUDA port of locate.py via CuPy. Same algorithm, same constants, same answers --
the search just runs on the GPU.

Why this is worth doing: the whole solver is FFT cross-correlation, and OpenCV's
matchTemplate re-transforms the *search image* on every single call. Here the
search image is transformed once per channel and reused by every hypothesis, and
hypotheses are evaluated in batches. A full-resolution hypothesis costs ~1 ms on
a GTX 1650 Ti against 20.7 ms on 12 CPU threads.

The pipeline is a faithful port of locate.locate -- half-resolution locate stage,
global full-resolution re-score, windowed refinement -- so the accuracy
characteristics (including the reasons the re-score must stay global) carry over
unchanged. See locate.py for why each stage exists.

Usage mirrors src/locate.py:
  python localize.py --root .dram_synthetic3 --split severe --csv results/severe.csv
  python localize.py --ref a/reference.png --search a/search.png
  python localize.py --check      # verify the GPU NCC against OpenCV, then exit
  python localize.py --cpu ...    # force the CPU fallback
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import locate as cpu
from src.locate import (ZOOM_MIN, ZOOM_MAX, ROT_MAX_DEG, SHEAR_PX, W_LOW, N_PRUNE,
                        N_THETA_GLOBAL, load_gray, make_template, subpixel_peak,
                        unwarp, bandpass, bandpass_coarse, _center_dist2)

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cnd
    cp.arange(1).sum()                      # force context creation
    GPU_OK = True
    GPU_ERROR = None
except Exception as exc:                    # no CUDA, no driver, bad wheel...
    cp = None
    GPU_OK = False
    GPU_ERROR = f"{type(exc).__name__}: {exc}"

MAX_BATCH = 32          # keeps working set well inside a 4 GB card


# ---------------------------------------------------------------- GPU filtering
def gpu_bandpass(d, low, high):
    """`locate.bandpass` on the GPU, numerically equal to the OpenCV version.

    Two settings make it equal rather than merely similar:
      * mode="mirror" -- ndimage's name for OpenCV's BORDER_REFLECT_101. The
        ndimage default ("reflect") duplicates the edge sample and pushed the
        largest deviation to 1.6; "mirror" brings it to 8e-06.
      * truncate=4.0  -- matches cv2.GaussianBlur's kernel radius for float32.
    """
    gf = lambda x, sg: cnd.gaussian_filter(x, sg, mode="mirror", truncate=4.0)
    f = gf(d, low) - gf(d, high)
    energy = gf(f * f, high)
    return f / (cp.sqrt(energy) + 1e-3)


def gpu_halve(c):
    """Exact equivalent of cv2.resize(..., INTER_AREA) at a factor of 2:
    for an integer factor INTER_AREA is precisely a block mean."""
    h, w = c.shape
    return c[:h // 2 * 2, :w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


# ---------------------------------------------------------------- GPU NCC core
def _fft_size(n):
    """Smallest 5-smooth (2^a*3^b*5^c) size >= n -- the sizes cuFFT handles best."""
    n = max(int(n), 1)
    best = 1 << int(np.ceil(np.log2(n)))     # power of two is always admissible
    p5 = 1
    while p5 < best * 5:
        p35 = p5
        while p35 < best * 3:
            v = p35
            while v < n:
                v *= 2
            if n <= v < best:
                best = v
            p35 *= 3
        p5 *= 5
    return best


class SearchPlane:
    """One filtered search channel, resident on the GPU and pre-transformed.

    Holds everything that depends only on the image (not on the template):
    its FFT, and the integral images used for the NCC denominator. Building
    this once per channel is what makes each hypothesis cheap.
    """

    def __init__(self, img):
        self.d = cp.ascontiguousarray(cp.asarray(img, dtype=cp.float32))
        self.h, self.w = self.d.shape
        # ROIs clipped at an image border are not square, so the transform must
        # cover the larger side.
        self.P = _fft_size(max(self.h, self.w))
        pad = cp.zeros((self.P, self.P), cp.float32)
        pad[:self.h, :self.w] = self.d
        self.F = cp.fft.rfft2(pad)
        # integral images (float64: 1e6 pixels of squares needs the headroom)
        d64 = self.d.astype(cp.float64)
        self.I1 = cp.pad(cp.cumsum(cp.cumsum(d64, 0), 1), ((1, 0), (1, 0)))
        self.I2 = cp.pad(cp.cumsum(cp.cumsum(d64 * d64, 0), 1), ((1, 0), (1, 0)))

    def window_stats(self, m):
        """Per-position sum and sum-of-squares over an m x m window."""
        A, B = self.I1, self.I2
        s1 = A[m:, m:] - A[:-m, m:] - A[m:, :-m] + A[:-m, :-m]
        s2 = B[m:, m:] - B[:-m, m:] - B[m:, :-m] + B[:-m, :-m]
        return s1, s2


def ncc_batch(plane, templates):
    """TM_CCOEFF_NORMED for many same-sized templates against one plane.

    Returns (B, H-m+1, W-m+1) float32 on device.

        num = corr(I, T - mean(T))            -- the image-mean cross term
                                                 vanishes because sum(T-Tbar)=0
        den = sqrt( (S2 - S1^2/n) * sum((T-Tbar)^2) )
    """
    B = len(templates)
    m = templates[0].shape[0]
    P = plane.P
    vh, vw = plane.h - m + 1, plane.w - m + 1

    t = cp.asarray(np.stack(templates).astype(np.float32))
    t = t - t.mean(axis=(1, 2), keepdims=True)
    ss_t = (t * t).sum(axis=(1, 2))                       # (B,)

    pad = cp.zeros((B, P, P), cp.float32)
    pad[:, :m, :m] = t
    corr = cp.fft.irfft2(plane.F[None] * cp.conj(cp.fft.rfft2(pad)), s=(P, P))
    num = corr[:, :vh, :vw]

    s1, s2 = plane.window_stats(m)
    var = s2 - s1 * s1 / float(m * m)
    den = cp.sqrt(cp.maximum(var, 0)[None] * ss_t[:, None, None])
    return cp.where(den > 1e-9, num / cp.maximum(den, 1e-9), 0).astype(cp.float32)


def _parabola(l, c, r):
    """One axis of locate.subpixel_peak, on three samples."""
    d = l - 2.0 * c + r
    return float(np.clip(0.5 * (l - r) / d, -1, 1)) if abs(d) > 1e-9 else 0.0


_D2_CACHE = {}


def _center_dist2_map(h, w, m, origin, center):
    """(h, w) of squared distances from each position's patch center to `center`.

    Depends only on the map geometry, which repeats across every hypothesis in a
    batch and across images, so it is built once and cached on the device.
    """
    key = (h, w, m, origin, center)
    d2 = _D2_CACHE.get(key)
    if d2 is None:
        xs = cp.arange(w, dtype=cp.float32) + (origin[0] + m / 2.0 - center[0])
        ys = cp.arange(h, dtype=cp.float32) + (origin[1] + m / 2.0 - center[1])
        d2 = (ys[:, None] ** 2 + xs[None, :] ** 2).ravel()
        _D2_CACHE[key] = d2
    return d2


def _peaks(maps, w_low, m, origin=(0, 0), center=None):
    """Combine the two channel maps and read off each hypothesis' peak.

    Only a 3x3 neighbourhood per hypothesis comes back across the PCIe bus --
    transferring whole correlation maps (megabytes per call) costs far more than
    the correlation itself at these sizes.

    Exact ties are broken toward `center` (the image center in this plane's own
    coordinate frame) rather than by lowest flat index, matching
    `locate.peak_loc`; see that function for why the raster-order default is a
    problem on a periodic lattice.
    """
    res = maps[0] + w_low * maps[1]
    B, h, w = res.shape
    flat = res.reshape(B, -1)
    if center is None:
        idx = cp.argmax(flat, axis=1)
    else:
        # argmin of distance restricted to the tied maxima: one extra pass, and
        # it stays a single vectorised kernel over the whole batch.
        mx = flat.max(axis=1, keepdims=True)
        d2 = _center_dist2_map(h, w, m, tuple(origin), tuple(center))
        idx = cp.argmin(cp.where(flat == mx, d2[None, :], cp.inf), axis=1)
    scores = cp.take_along_axis(flat, idx[:, None], 1)[:, 0] / (1.0 + w_low)

    ys, xs = cp.divmod(idx, w)
    yc = cp.clip(ys, 1, max(h - 2, 1))
    xc = cp.clip(xs, 1, max(w - 2, 1))
    off = cp.arange(-1, 2)
    neigh = res[cp.arange(B)[:, None, None],
                (yc[:, None, None] + off[None, :, None]),
                (xc[:, None, None] + off[None, None, :])]

    # One transfer, not three. Each cp.asnumpy is a device synchronisation, and
    # at these batch sizes the syncs cost far more than the bytes: separate
    # copies of neigh/xs/ys/scores made this function 2.9 ms per call, which was
    # the single largest cost in the barrel search.
    packed = cp.empty((B, 12), cp.float32)
    packed[:, :9] = neigh.reshape(B, 9)
    packed[:, 9] = scores
    packed[:, 10] = xs
    packed[:, 11] = ys
    p_h = cp.asnumpy(packed)

    out = []
    for b in range(B):
        row = p_h[b]
        x, y = int(row[10]), int(row[11])
        p = row[:9].reshape(3, 3)
        sc_h = row[9:10]
        dx = _parabola(p[1, 0], p[1, 1], p[1, 2]) if 0 < x < w - 1 else 0.0
        dy = _parabola(p[0, 1], p[1, 1], p[2, 1]) if 0 < y < h - 1 else 0.0
        out.append((float(sc_h[0]), x + dx + m / 2.0, y + dy + m / 2.0))
    return out


def _templates_for(ref_bp, zoom, thetas):
    """Build (fine, coarse) templates on the CPU for each theta.

    make_template is a 82x82 cv2.warpAffine -- effectively free, and keeping it
    on the CPU preserves INTER_AREA behaviour exactly. The uploads are tiny.
    """
    fine = [make_template(ref_bp[0], zoom, t) for t in thetas]
    coarse = [make_template(ref_bp[1], zoom, t) for t in thetas]
    return fine, coarse


def match_grid(planes, ref_bp, zoom, thetas, w_low=W_LOW, scale=1.0):
    """Evaluate every theta at one zoom, batched. -> [(score, cx, cy)] per theta."""
    fine, coarse = _templates_for(ref_bp, zoom * scale, thetas)
    m = fine[0].shape[0]
    if m >= planes[0].h:
        return [(-1.0, 0.0, 0.0)] * len(thetas)
    # The plane's own center: at half resolution that is half the search image's,
    # so the tie-break is always measured in the frame the correlation ran in.
    center = ((planes[0].w - 1) / 2.0, (planes[0].h - 1) / 2.0)
    out = []
    for i in range(0, len(thetas), MAX_BATCH):
        f = fine[i:i + MAX_BATCH]
        c = coarse[i:i + MAX_BATCH]
        maps = (ncc_batch(planes[0], f), ncc_batch(planes[1], c))
        out.extend(_peaks(maps, w_low, m, center=center))
    return out


def match_roi(search_planes_src, ref_bp, params, cx, cy, half=40, w_low=W_LOW):
    """Batched equivalent of locate.match_local: many (zoom, theta) hypotheses
    scored inside one fixed window around (cx, cy).

    The window is fixed for the whole batch -- the CPU version recentres as it
    improves, which cannot be batched. Refinement moves the peak only a few px,
    so a +/-`half` window still contains it.
    """
    h, w = search_planes_src[0].shape
    # Build every template once. warpAffine is cheap but not free, and this is
    # the hottest loop in the solver (~200 hypotheses x 2 channels per image).
    tpls = [(make_template(ref_bp[0], z, t), make_template(ref_bp[1], z, t))
            for z, t in params]
    m_max = max(f.shape[0] for f, _ in tpls)

    x0 = int(round(cx - m_max / 2.0)) - half
    y0 = int(round(cy - m_max / 2.0)) - half
    x1 = min(x0 + m_max + 2 * half, w)
    y1 = min(y0 + m_max + 2 * half, h)
    x0, y0 = max(x0, 0), max(y0, 0)
    if x1 - x0 < m_max or y1 - y0 < m_max:
        return [(-1.0, cx, cy)] * len(params)

    roi = [SearchPlane(p[y0:y1, x0:x1]) for p in search_planes_src]
    # group by template size so each batch shares one valid-map shape
    by_m = {}
    for k, (f, _) in enumerate(tpls):
        by_m.setdefault(f.shape[0], []).append(k)

    # The ROI is a crop, so the tie-break needs the crop origin and the *full*
    # plane's center, both in source-plane coordinates.
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    results = []
    for m, keys in by_m.items():
        for i in range(0, len(keys), MAX_BATCH):
            ks = keys[i:i + MAX_BATCH]
            maps = (ncc_batch(roi[0], [tpls[k][0] for k in ks]),
                    ncc_batch(roi[1], [tpls[k][1] for k in ks]))
            for k, (s, px, py) in zip(ks, _peaks(maps, w_low, m,
                                                 origin=(x0, y0), center=center)):
                results.append((k, (s, x0 + px, y0 + py)))
    results.sort()
    return [r for _, r in results]


# ---------------------------------------------------------------- barrel (GPU)
# The barrel k-search stays on the CPU. A fully batched GPU version was built
# and measured against it (MultiPlane: all ~19 candidate k windows stacked into
# one transform, correlations gathered per window) and came out a dead heat --
# 143.9 ms vs 144.9 ms across three cases. At 162 px windows there is simply not
# enough arithmetic to cover CuPy launch overhead, so the CPU implementation is
# kept for its exact parity and ~90 fewer lines. Do not re-port it without
# measuring first; the win is in cutting hypotheses, not in moving hardware.


# ---------------------------------------------------------------- main solver
def locate(ref, search, zoom_range=(ZOOM_MIN, ZOOM_MAX), rot_max=ROT_MAX_DEG,
           w_low=W_LOW, n_seeds=4, shear_px=SHEAR_PX, barrel="auto"):
    """GPU port of locate.locate -- see that function for the algorithm."""
    if not GPU_OK:
        return cpu.locate(ref, search, zoom_range, rot_max, w_low, n_seeds,
                          shear_px, barrel)

    z0, z1 = zoom_range
    zc = 0.5 * (z0 + z1)
    # Reference channels stay on the CPU: make_template warps them with
    # cv2.INTER_AREA, and they are cheap (15 ms) because they are decimated.
    ref_bp = (bandpass_coarse(ref, 1.2 * zc, 12.0 * zc),
              bandpass_coarse(ref, 6.0 * zc, 60.0 * zc))

    # The search's fine channel is the single most expensive filter (43 ms on
    # CPU) and runs on the GPU. Its coarse channel stays on the CPU: that path
    # decimates and re-expands with INTER_CUBIC, which ndimage does not
    # reproduce exactly, and it only costs 27 ms.
    search_bp = (gpu_bandpass(cp.asarray(search, dtype=cp.float32), 1.2, 12.0),
                 cp.asarray(bandpass_coarse(search, 6.0, 60.0)))

    planes_half = [SearchPlane(gpu_halve(c)) for c in search_bp]
    planes_full = [SearchPlane(c) for c in search_bp]

    # --- stage 1 (LOCATE): coarse grid at half resolution
    thetas = [rot_max * (2 * i + 1) / (2.0 * N_THETA_GLOBAL)
              for i in range(N_THETA_GLOBAL)]
    grid = []
    for z in np.arange(z0, z1 + 1e-6, 0.5):
        for t, (s, cx, cy) in zip(thetas,
                                  match_grid(planes_half, ref_bp, z, thetas,
                                             w_low, scale=2.0)):
            grid.append((s, 2.0 * cx, 2.0 * cy, z, t))
    size = search.shape[0]
    grid.sort(key=lambda c: (-c[0], _center_dist2(c[1], c[2], size)))

    pruned, seen = [], []
    for c in grid:
        if all(math.hypot(c[1] - k[1], c[2] - k[2]) > 15.0 for k in seen):
            seen.append(c)
            pruned.append(c)
        if len(pruned) >= N_PRUNE:
            break

    # --- stage 2 (RE-SCORE): global, full resolution. Stays global on purpose:
    # it is the escape hatch when half-res missed the true spot (see locate.py).
    cands = []
    for _, _, _, z, t in pruned:
        s, cx, cy = match_grid(planes_full, ref_bp, z, [t], w_low)[0]
        cands.append((s, cx, cy, z, t))
    cands.sort(key=lambda c: (-c[0], _center_dist2(c[1], c[2], size)))

    seeds, keep = [], []
    for c in cands:
        if all(math.hypot(c[1] - k[1], c[2] - k[2]) > 15.0 for k in keep):
            keep.append(c)
            seeds.append(c)
        if len(seeds) >= n_seeds:
            break

    # --- stage 3 (REFINE): batched windowed search per seed
    best = (-2.0, 0.0, 0.0, z0, 0.0)
    for seed in seeds:
        cur = seed
        for dz, dt in ((0.08, 0.5), (0.03, 0.12)):
            _, _, _, bz, bt = cur
            params = [(z, t)
                      for z in np.arange(bz - 2 * dz, bz + 2 * dz + 1e-6, dz)
                      if z0 - 0.5 <= z <= z1 + 0.5
                      for t in np.arange(max(bt - 2 * dt, 0.0),
                                         bt + 2 * dt + 1e-6, dt)]
            if not params:
                continue
            for (z, t), (s, cx, cy) in zip(
                    params, match_roi(search_bp, ref_bp, params,
                                      cur[1], cur[2], w_low=w_low)):
                if s > cur[0] or (s == cur[0]
                                  and _center_dist2(cx, cy, size)
                                  < _center_dist2(cur[1], cur[2], size)):
                    cur = (s, cx, cy, z, t)
        if cur[0] > best[0] or (cur[0] == best[0]
                                and _center_dist2(cur[1], cur[2], size)
                                < _center_dist2(best[1], best[2], size)):
            best = cur

    # --- stage 4: lens distortion (locate.estimate_barrel, on the CPU -- see
    # the note above). The filtered channels come back once here rather than
    # per candidate k.
    k_est, k_gain = 0.0, 0.0
    if barrel:
        sb_np = tuple(cp.asnumpy(c) for c in search_bp)
        (bs, bx, by, bz, bt), k_est, k_gain = cpu.estimate_barrel(
            sb_np, ref_bp, best, w_low=w_low)
        accept = (barrel is True) or (k_gain > cpu.BARREL_MIN_GAIN)
        if accept and bs > -1.0:
            best = (bs, bx, by, bz, bt)
        else:
            k_est = 0.0

    score, cx, cy, z, t = best
    gx, gy = unwarp(cx, cy, t, size=search.shape[0], shear_px=shear_px)
    return {"x": float(gx), "y": float(gy), "score": float(score),
            "zoom": float(z), "theta": float(t), "k": float(k_est),
            "k_gain": float(k_gain),
            "x_observed": float(cx), "y_observed": float(cy)}


# ---------------------------------------------------------------- correctness
def check_ncc(root=".dram_synthetic3", verbose=True):
    """Compare the GPU NCC against cv2.matchTemplate on real data.

    Nothing downstream is meaningful if this fails, so it runs first.
    """
    if not GPU_OK:
        print(f"GPU unavailable ({GPU_ERROR}) -- nothing to check.")
        return False
    worst = 0.0
    for case in [os.path.join(root, "low", "d00"),
                 os.path.join(root, "severe", "d01"),
                 os.path.join(root, "high", "d12")]:
        ref = load_gray(case + "/reference.png")
        s = load_gray(case + "/search.png")
        zc = 10.0
        rb = bandpass_coarse(ref, 1.2 * zc, 12.0 * zc)
        sb = bandpass(s, 1.2, 12.0)
        plane = SearchPlane(sb)
        for zoom, th in ((9.5, 0.4), (10.0, 1.3), (10.9, 2.7)):
            tpl = make_template(rb, zoom, th)
            gpu = cp.asnumpy(ncc_batch(plane, [tpl])[0])
            ref_map = cv2.matchTemplate(sb, tpl, cv2.TM_CCOEFF_NORMED)
            d = float(np.abs(gpu - ref_map).max())
            worst = max(worst, d)
            if verbose:
                label = "/".join(case.replace("\\", "/").split("/")[-2:])
                print(f"  {label:12s} zoom={zoom:5.2f} th={th:4.1f}"
                      f"  max|GPU-OpenCV| = {d:.2e}")
    ok = worst < 1e-4
    print(f"\nworst deviation over all pairs: {worst:.2e}  -> {'PASS' if ok else 'FAIL'}"
          f" (tolerance 1e-4)")
    return ok


# ---------------------------------------------------------------- evaluation
def evaluate(root, split, limit=None, thresh=5.0, csv_path=None, seeds=4,
             shear=SHEAR_PX, locate_fn=None, barrel="auto"):
    locate_fn = locate_fn or locate
    base = os.path.join(root, split) if split else root
    cases = sorted(d for d in os.listdir(base)
                   if os.path.isdir(os.path.join(base, d)))
    if limit:
        cases = cases[:limit]

    rows, t0 = [], time.time()
    for i, name in enumerate(cases):
        d = os.path.join(base, name)
        ref, search = load_gray(d + "/reference.png"), load_gray(d + "/search.png")
        meta = json.load(open(os.path.join(d, "metadata.json")))

        t_img = time.perf_counter()
        r = locate_fn(ref, search, n_seeds=seeds, shear_px=shear, barrel=barrel)
        if GPU_OK:
            cp.cuda.Stream.null.synchronize()
        t_img = time.perf_counter() - t_img

        err = math.hypot(r["x"] - meta["gt_x"], r["y"] - meta["gt_y"])
        rows.append({
            "image": name,
            "gt_x": meta["gt_x"], "gt_y": meta["gt_y"],
            "pred_x": r["x"], "pred_y": r["y"],
            "error_px": err,
            "passed": int(err < thresh),
            "time_s": t_img,
            "score": r["score"],
            "zoom_pred": r["zoom"], "zoom_gt": meta["params"]["zoom"],
            "rot_pred": r["theta"], "rot_gt": meta["rotation_deg"],
        })
        print(f"{name}: pred=({r['x']:7.2f},{r['y']:7.2f}) "
              f"gt=({meta['gt_x']:7.2f},{meta['gt_y']:7.2f}) err={err:6.2f}  "
              f"ncc={r['score']:.3f} t={t_img:5.2f}s "
              f"{'PASS' if err < thresh else 'FAIL'}")

    errs = np.array([r["error_px"] for r in rows])
    times = np.array([r["time_s"] for r in rows])
    worst = max(rows, key=lambda r: r["error_px"])

    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {csv_path}")

    print(f"\n{split or base}: n={len(errs)}  pass@{thresh}px="
          f"{(errs < thresh).mean()*100:.1f}%")
    print(f"  mean   error : {errs.mean():8.2f} px")
    print(f"  median error : {np.median(errs):8.2f} px")
    print(f"  worst  error : {worst['error_px']:8.2f} px  ({worst['image']})")
    print(f"  mean   time  : {times.mean():8.3f} s/image")
    print(f"  median time  : {np.median(times):8.3f} s/image")
    print(f"  total  time  : {times.sum():8.1f} s  (wall {time.time()-t0:.1f}s)")
    return rows


def warmup():
    """Build cuFFT plans once so the first timed image isn't paying for them.

    Runs a whole `locate` on noise rather than a list of guessed sizes: the
    refinement ROIs take sizes that depend on the templates, and missing one
    left ~6 s of plan building inside the first real image.
    """
    if not GPU_OK:
        return 0.0
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (1000, 1000), dtype=np.uint8)
    b = rng.integers(0, 255, (1000, 1000), dtype=np.uint8)
    locate(a, b)
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ref")
    ap.add_argument("--search")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--shear", type=float, default=SHEAR_PX)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--barrel", dest="barrel", action="store_true",
                    default=None, help="always accept the estimated distortion")
    ap.add_argument("--no-barrel", dest="barrel", action="store_false",
                    help="skip lens-distortion estimation entirely (fastest)")
    ap.add_argument("--cpu", action="store_true", help="force the CPU path")
    ap.add_argument("--check", action="store_true",
                    help="verify the GPU NCC against OpenCV, then exit")
    ap.add_argument("--check-root", default=".dram_synthetic3",
                    help="dataset root used by --check")
    args = ap.parse_args()
    barrel = "auto" if args.barrel is None else args.barrel

    if args.check:
        raise SystemExit(0 if check_ncc(args.check_root) else 1)

    fn = cpu.locate if args.cpu else locate
    if args.cpu:
        print("[forced CPU path]")
    elif GPU_OK:
        print(f"[GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}"
              f"] warm-up {warmup():.2f}s")
    else:
        print(f"[GPU unavailable -- {GPU_ERROR}] falling back to CPU")

    if args.ref and args.search:
        print(json.dumps(fn(load_gray(args.ref), load_gray(args.search),
                            n_seeds=args.seeds, shear_px=args.shear,
                            barrel=barrel), indent=2))
    else:
        evaluate(args.root, args.split, args.limit, csv_path=args.csv,
                 seeds=args.seeds, shear=args.shear, locate_fn=fn,
                 barrel=barrel)


if __name__ == "__main__":
    main()
