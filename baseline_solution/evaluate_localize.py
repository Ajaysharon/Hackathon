#!/usr/bin/env python3
"""
Evaluate the GPU solver (`localize.py`) across noise levels and produce
Precision-Recall curves -- the counterpart of `evaluate_zncc.py`, which runs the
same protocol on the ZNCC baseline over the same images.

Two differences from `evaluate.py`, both deliberate:
  - It scores the *saved* dataset (`<root>/<level>/dNN/`) rather than synthesizing
    samples on the fly, so the solver and the baseline are judged on exactly the
    same images and a run is reproducible.
  - "score" is the refined NCC peak `localize.locate` settles on.

Methodology is otherwise identical, and the PR definitions come from evaluate.py
via `dataset_eval` rather than being reimplemented:
  - For each level, run the matcher on every case; record (score, correct) where
    correct = (distance to ground truth <= --tolerance-px).
  - Sweep the score as an acceptance threshold: accepted = score >= threshold.
      TP = accepted & correct;  FP = accepted & incorrect
      FN = rejected -- every case has exactly one true match, so total
           positives = N regardless of threshold.
    precision = TP / (TP + FP); recall = TP / N
  - AP = area under the PR curve (trapezoidal).

By default it reuses `results/localize_<level>.csv` if `localize.py --csv` already
wrote one (those rows carry score and error_px, which is all a PR curve needs);
pass --rerun to force the localisations again.

Usage:
    python baseline_solution/evaluate_localize.py --root .dram_synthetic3 \
        --output-dir results --tolerance-px 5
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import localize
from src.noise_levels import LEVEL_LABELS
from baseline_solution import dataset_eval as de

MATCHER_NAME = "localize.py (GPU)"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="YAML/JSON file supplying defaults for the flags below")
    p.add_argument("--root", default=".dram_synthetic3")
    p.add_argument("--levels", nargs="+", default=LEVEL_LABELS)
    p.add_argument("--tolerance-px", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--shear", type=float, default=localize.SHEAR_PX)
    p.add_argument("--barrel", dest="barrel", action="store_true", default=None)
    p.add_argument("--no-barrel", dest="barrel", action="store_false")
    p.add_argument("--cpu", action="store_true", help="force the CPU path")
    p.add_argument("--rerun", action="store_true",
                   help="re-localise even when results/localize_<level>.csv exists")
    p.add_argument("--output-dir", default="results")

    pre, _ = p.parse_known_args()
    if pre.config:
        with open(pre.config) as f:
            text = f.read()
        try:
            import yaml
            cfg = yaml.safe_load(text)
        except ImportError:
            cfg = json.loads(text)
        p.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    barrel = "auto" if args.barrel is None else args.barrel
    locate_fn = localize.cpu.locate if args.cpu else localize.locate

    need_run = args.rerun or any(
        not os.path.exists(de.csv_path_for(args.output_dir, lb, "localize"))
        for lb in args.levels)
    if need_run and not args.cpu and localize.GPU_OK:
        # Build the cuFFT plans once so case 0 is not charged for them.
        print(f"[GPU warm-up {localize.warmup():.2f}s]")

    results = []
    for label in args.levels:
        path = de.csv_path_for(args.output_dir, label, "localize")
        if args.rerun or not os.path.exists(path):
            print(f"Localising noise level '{label}'...")
            # localize.evaluate already writes exactly the per-case CSV the
            # shared harness reads, so use it rather than re-driving the solver.
            rows = localize.evaluate(args.root, label, limit=args.limit,
                                     thresh=args.tolerance_px, csv_path=path,
                                     seeds=args.seeds, shear=args.shear,
                                     locate_fn=locate_fn, barrel=barrel)
        else:
            print(f"Reusing {path}")
            rows = de.load_rows(path)
        res = de.summarize(label, rows, args.tolerance_px)
        results.append(res)
        print(f"  AP={res['ap']:.3f}  accuracy@{args.tolerance_px}px="
              f"{res['accuracy']:.3f}  median_err={res['median_error_px']:.2f}px")

    de.report(results, args.tolerance_px, args.output_dir, MATCHER_NAME)


if __name__ == "__main__":
    main()
