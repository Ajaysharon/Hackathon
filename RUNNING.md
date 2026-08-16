# Running the experiment end to end

Every command below is run from the repository root
(`Submission/Data_Generation_code/`), in order, and reproduces the numbers and
plots in `results/` and `baseline_solution/eval_results/` from nothing but a
seed.

Two complete sessions are given:

- **[Session A — CPU only](#session-a--cpu-only)** — no GPU, no CUDA, no CuPy.
- **[Session B — GPU](#session-b--gpu-cuda--cupy)** — the same experiment with the
  CUDA solver.

They differ only in step 0 (what you install) and in one flag on the localizer.
**The results are identical** — same predictions, same accuracy, same AP; the GPU
is ~2× faster per image on the reference machine. Pick one session and run it
straight through.

---

## Session A — CPU only

No GPU required. `localize.py` detects the absence of CuPy and falls back to the
CPU solver in `src/locate.py` on its own; `--cpu` forces that path even when a GPU
is present, which is what makes this session reproducible on any machine.

Expect ~5 minutes total: ~1 min to generate, ~2 min to localize, ~1 min for the
baseline.

### A0. Install

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Do **not** install CuPy for this session.

### A1. Generate the dataset (DRAM only, from the seed)

```bash
python generate_dataset.py \
  --samples-per-level 40 \
  --rotation-max-deg 3.0 \
  --seed 2026 \
  --architectures dram_1x dram_dense dram_loose dram_wide dram_compact dram_legacy \
  --output-dir .dram_synthetic3
```

160 cases — 40 for each of `low`, `medium`, `high`, `severe` — written as
`.dram_synthetic3/<level>/dNN/{reference.png, search.png, metadata.json}`.
Only the six DRAM presets are used; the FinFET presets are excluded by the
`--architectures` list. Seed 2026 fixes everything: each level's RNG is seeded
`2026 + i*9973`, and architectures are cycled `i % 6`, so this command produces
byte-identical images on every run.

Equivalent one-liner, replaying the recorded config:

```bash
python generate_dataset.py --config configs/dataset_dram_synthetic3.yaml
```

### A2. Check the dataset

```bash
python -c "import glob; print(len(glob.glob('.dram_synthetic3/*/d*')), 'cases')"
```
Expect `160 cases`. Spot-check one metadata file:
```bash
python -c "import json; print(json.load(open('.dram_synthetic3/low/d00/metadata.json')))"
```

### A3. Sanity-check the solver on three cases

```bash
python localize.py --cpu --root .dram_synthetic3 --split low --limit 3
```
Expect three PASS lines with errors around 0.5–1.2 px, ~0.8 s/image.

### A4. Localize all 160 cases

```bash
python localize.py --cpu --root .dram_synthetic3 --split low    --csv results/localize_low.csv
python localize.py --cpu --root .dram_synthetic3 --split medium --csv results/localize_medium.csv
python localize.py --cpu --root .dram_synthetic3 --split high   --csv results/localize_high.csv
python localize.py --cpu --root .dram_synthetic3 --split severe --csv results/localize_severe.csv
```
~35 s per level (0.86–0.90 s/image), ~2 min total. Each command prints its own
pass@5px and error/time summary and writes one row per case.

### A5. PR curves and AP-vs-noise plots

```bash
python baseline_solution/evaluate_localize.py \
  --cpu --root .dram_synthetic3 --output-dir results
```
Reuses the CSVs from A4 (add `--rerun` to re-localize instead). Writes
`results/pr_curves.png`, `results/ap_vs_noise.png`, `results/summary.csv`.

If you skip A4, this one command does the localization itself — A4 exists so you
can watch it per level and keep the per-case detail.

### A6. Evaluate the ZNCC baseline on the same images

```bash
python baseline_solution/evaluate_zncc.py \
  --root .dram_synthetic3 --output-dir baseline_solution/eval_results
```
~21 s (0.13 s/image; the baseline has no GPU path and needs none). Writes
`baseline_solution/eval_results/{zncc_<level>.csv, pr_curves.png, ap_vs_noise.png, summary.csv}`.

Add `--quiet` to suppress the per-case lines.

### A7. Tests

```bash
python -m pytest tests/ -q
```
Expect `18 passed` — the generator pipeline plus `tests/test_peak_tiebreak.py`,
which pins the center tie-break (see [What you should get](#what-you-should-get)).

---

## Session B — GPU (CUDA + CuPy)

Identical experiment, CUDA solver. Only A0 and the localizer flag change.
Expect ~3 minutes total.

### B0. Install

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install cupy-cuda11x          # or cupy-cuda12x -- match your CUDA runtime
```

The reported results were produced with `cupy-cuda11x` 13.6.0. There is no
universal wheel: installing the one that does not match your CUDA runtime is the
usual cause of the fallback message in B2.

Confirm the GPU is visible:
```bash
python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)['name'].decode())"
```
If this fails, you are in Session A — `localize.py` will say
`[GPU unavailable -- ...] falling back to CPU` and still produce the same answers.

### B1. Generate the dataset (DRAM only, from the seed)

Exactly as **A1** — generation is CPU-only and identical in both sessions.

```bash
python generate_dataset.py \
  --samples-per-level 40 \
  --rotation-max-deg 3.0 \
  --seed 2026 \
  --architectures dram_1x dram_dense dram_loose dram_wide dram_compact dram_legacy \
  --output-dir .dram_synthetic3
```

### B2. Verify the GPU NCC against OpenCV

```bash
python localize.py --check --check-root .dram_synthetic3
```
Nothing downstream means anything if this fails. Expect a per-pair table and
`worst deviation over all pairs: ~3e-07 -> PASS (tolerance 1e-4)` — the GPU
correlation is numerically equal to `cv2.matchTemplate`, so the GPU is a speed
change, not an accuracy change.

### B3. Sanity-check the solver on three cases

```bash
python localize.py --root .dram_synthetic3 --split low --limit 3
```
Expect the same three predictions as A3, at ~0.33 s/image after a ~0.7 s warm-up.

### B4. Localize all 160 cases

```bash
python localize.py --root .dram_synthetic3 --split low    --csv results/localize_low.csv
python localize.py --root .dram_synthetic3 --split medium --csv results/localize_medium.csv
python localize.py --root .dram_synthetic3 --split high   --csv results/localize_high.csv
python localize.py --root .dram_synthetic3 --split severe --csv results/localize_severe.csv
```
~17 s per level (0.39–0.43 s/image), ~1 min total.

### B5. PR curves and AP-vs-noise plots

```bash
python baseline_solution/evaluate_localize.py \
  --root .dram_synthetic3 --output-dir results
```
or, replaying the recorded config:
```bash
python baseline_solution/evaluate_localize.py --config configs/eval_localize.yaml
```

### B6. Evaluate the ZNCC baseline on the same images

Exactly as **A6** — the baseline is CPU-only in both sessions.

```bash
python baseline_solution/evaluate_zncc.py \
  --root .dram_synthetic3 --output-dir baseline_solution/eval_results
```

### B7. Tests

```bash
python -m pytest tests/ -q
```
Expect `18 passed`, exactly as in A7 — the suite has no GPU-dependent tests.

---

## What you should get

Both sessions, tolerance 5 px, 160 cases:

| level  | pass@5px | AP    | median err | mean err  |
|--------|----------|-------|------------|-----------|
| low    | 95.0%    | 0.937 | 0.85 px    | 18.12 px  |
| medium | 97.5%    | 0.949 | 0.65 px    | 4.27 px   |
| high   | 100.0%   | 1.000 | 1.20 px    | 1.32 px   |
| severe | 67.5%    | 0.534 | 2.46 px    | 80.95 px  |
| **all**| **90.0%**|       |            |           |

ZNCC baseline on the same images: **15.0%** overall (AP 0.03–0.08 per level).

Timing is the only thing that differs between the sessions (measured on an
NVIDIA GTX 1650 Ti):

| stage | Session A (CPU) | Session B (GPU) |
|---|---|---|
| localize, per image | 0.86–0.90 s | 0.39–0.43 s |
| localize, all 160 | ~2 min | ~1 min |
| ZNCC baseline, all 160 | ~21 s | ~21 s (CPU either way) |

### Why the two sessions agree

The solver contains no randomness — no sampling, no learned weights, no
wall-clock-dependent choices — and equal scores are settled by a fixed geometric
rule (`src/locate.py:peak_loc`: among positions scoring exactly the same, take
the one nearest the image center), so neither session can be decided by raster
order or by the order hypotheses happened to be swept. Pass rates, AP and error
statistics are therefore identical in both, to the digit.

Per-case predictions differ between CPU and GPU by at most **~7e-06 px** — the
CuPy and OpenCV filter paths are not bit-identical (see the `gpu_bandpass`
note in `localize.py`) — which is orders of magnitude below the sub-pixel
precision either path claims, and invisible in every reported statistic.

### Outputs

```
.dram_synthetic3/<level>/dNN/    the 160 generated cases
results/localize_<level>.csv     one row per case: gt, pred, error_px, passed, time_s, score
results/summary.csv              per-level n, pass@5px, AP, error stats, mean time
results/pr_curves.png            PR curve per noise level, AP in the legend
results/ap_vs_noise.png          AP and accuracy vs noise level
baseline_solution/eval_results/  the same five artifacts for the ZNCC baseline
```

## Troubleshooting

- **`no cases found under .dram_synthetic3/<level>`** — step 1 has not been run,
  or `--output-dir` and `--root` disagree.
- **`PermissionError: ... summary.csv`** — the CSV is open in Excel; close it.
- **`--samples-per-level ... cannot be combined with --num-samples`** — those are
  two different output modes. The experiment uses `--samples-per-level`;
  `--num-samples` is the older flat-split mode described in the README.
- **An evaluator prints `Reusing results/localize_<level>.csv` when you wanted a
  fresh run** — pass `--rerun`.
- **GPU present but unused** — `localize.py` prints the reason it fell back
  (`[GPU unavailable -- ImportError: ...]`), usually a CuPy wheel that does not
  match the installed CUDA runtime.
