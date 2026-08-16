#!/usr/bin/env python3
"""
Evaluate the ZNCC baseline matcher (`baseline_solution/zncc.py`) against the
saved benchmark dataset, and produce Precision-Recall curves.

This is `evaluate.py`'s methodology run on `evaluate_localize.py`'s inputs: the
same `.dram_synthetic3/<level>/dNN/` images the GPU solver is scored on, rather
than samples synthesized on the fly. Scoring both matchers on identical images is
what makes the baseline-vs-solver comparison in results/ meaningful -- any gap is
the matcher, not the draw.

  - For each level, run `zncc_match` on every case; record (score, correct) where
    correct = (distance to ground truth <= --tolerance-px). "score" is the peak
    ZNCC confidence at the scale that won the multi-scale sweep.
  - Sweep the score as an acceptance threshold; AP = area under the PR curve.
    Both definitions are imported from evaluate.py, not restated.

Usage:
    python baseline_solution/evaluate_zncc.py --root .dram_synthetic3 \
        --output-dir baseline_solution/eval_results
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.noise_levels import LEVEL_LABELS
from baseline_solution import dataset_eval as de
from baseline_solution.zncc import zncc_match

HERE = os.path.dirname(os.path.abspath(__file__))
MATCHER_NAME = "ZNCC baseline"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="YAML/JSON file supplying defaults for the flags below")
    p.add_argument("--root", default=".dram_synthetic3")
    p.add_argument("--levels", nargs="+", default=LEVEL_LABELS)
    p.add_argument("--tolerance-px", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--scales", nargs="+", type=float,
                   default=[9.0, 9.5, 10.0, 10.5, 11.0],
                   help="reference:search pixel-size ratios swept by the matcher")
    p.add_argument("--rerun", action="store_true",
                   help="re-match even when eval_results/zncc_<level>.csv exists")
    p.add_argument("--output-dir", default=os.path.join(HERE, "eval_results"))
    p.add_argument("--quiet", action="store_true", help="suppress per-case lines")

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
    matcher = lambda ref, search: zncc_match(ref, search, scales=tuple(args.scales))

    results = []
    for label in args.levels:
        path = de.csv_path_for(args.output_dir, label, "zncc")
        if args.rerun or not os.path.exists(path):
            print(f"Matching noise level '{label}' with the ZNCC baseline...")
            rows = de.run_matcher(args.root, label, matcher, args.tolerance_px,
                                  limit=args.limit, verbose=not args.quiet)
            de.write_rows(rows, path)
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
