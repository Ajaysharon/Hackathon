# References

Sources behind the physics of the generator and the algorithms in the solver,
plus pointers to the validation work done in this submission.

## SEM image formation

1. Reimer, L. *Scanning Electron Microscopy: Physics of Image Formation and
   Microanalysis*, 2nd ed., Springer, 1998. — secondary-electron yield, edge
   ("topographic") contrast, and beam-spot-limited resolution; the basis for the
   edge-brightening and Gaussian spot blur in `src/sem_imaging.py`.
2. Goldstein, J. et al. *Scanning Electron Microscopy and X-Ray Microanalysis*,
   4th ed., Springer, 2018. — dose vs. shot noise; the Poisson model behind
   `dose_reference` / `dose_search` and why low dose is the dominant noise term.
3. Postek, M. T. & Vladár, A. E., "Does your SEM really tell the truth?",
   *Proc. SPIE* 8378, 2012. — charging artifacts, drift and raster distortion in
   CD-SEM metrology; motivates `charging_streak_*`, `drift_jitter_px` and
   `shear_amplitude_px`.

## Distortion and detector model

4. Brown, D. C., "Close-Range Camera Calibration", *Photogrammetric Engineering*
   37(8), 1971. — the radial (Brown–Conrady) distortion model used for
   `barrel_distortion_k` and inverted in `src/locate.py:barrel_point`.
5. Healey, G. E. & Kondepudy, R., "Radiometric CCD camera calibration and noise
   estimation", *IEEE TPAMI* 16(3), 1994. — the shot + additive Gaussian read
   noise split reproduced by `detector_noise_sigma_*`.

## Matching algorithms

6. Lewis, J. P., "Fast Normalized Cross-Correlation", *Vision Interface*, 1995.
   — running-sum (integral image) NCC denominator; exactly what
   `localize.SearchPlane.window_stats` computes, and what makes the batched GPU
   NCC equal to `cv2.matchTemplate(TM_CCOEFF_NORMED)`.
7. Briechle, K. & Hanebeck, U. D., "Template matching using fast normalized
   cross correlation", *Proc. SPIE* 4387, 2001. — the coarse-to-fine strategy
   mirrored by the half-resolution LOCATE / full-resolution RE-SCORE stages.
8. Tian, Q. & Huhns, M. N., "Algorithms for subpixel registration", *CVGIP*
   35(2), 1986. — the 2-D parabola fit used for the sub-pixel peak
   (`subpixel_peak` / `_parabola`).
9. Everingham, M. et al., "The PASCAL Visual Object Classes (VOC) Challenge",
   *IJCV* 88(2), 2010. — the precision/recall + average-precision protocol the
   evaluators follow.

## Device layout

10. JEDEC JESD79-4 (DDR4) and public 1x/1y/1z-nm DRAM process disclosures
    (IEDM/ISSCC survey papers) — cell pitch, mat/strip zone organisation and the
    6F² layout the `dram_*` presets in `src/presets.py` approximate. These
    presets are stylised, not a reproduction of any vendor's layout.

## Tooling

11. CuPy — NumPy/SciPy-compatible array library for CUDA. https://cupy.dev
12. OpenCV — `matchTemplate`, `warpAffine`, `GaussianBlur`. https://opencv.org

## In-submission reports

- [`synthetic_realism.md`](synthetic_realism.md) — realism, diversity and
  reproducibility rationale for every generator parameter.
- [`validation_report.md`](validation_report.md) and
  [`validation_report_seed2024.md`](validation_report_seed2024.md) — statistical
  validation of a generated split, and the seed-2024 repeat.
- [`dram_synthetic_dataset.md`](dram_synthetic_dataset.md) — the dataset card for
  the level-structured benchmark.

(These four are copies of the files in `docs/`, kept here so the references
folder is self-contained.)
