#!/usr/bin/env python3
"""
Generate and save the best_v6 evaluation set to disk, one folder per noise
level, one subfolder per sample -- so runs are reproducible/inspectable
instead of using throwaway temp files (see evaluate_v6.py).

Each sample folder contains:
    reference.png, search.png, metadata.json (gt_x, gt_y, rotation_deg,
    architecture, and the full GenerationParams used)

Layout:
    <output-dir>/<level>/d00/reference.png
    <output-dir>/<level>/d00/search.png
    <output-dir>/<level>/d00/metadata.json
    ...

This directly matches the folder shape best_v6.main() expects (glob "d*"
under a base dir with reference*.png / search*.png / metadata*.json), so
`python baseline_solution/best_v6.py` can also be pointed at
`<output-dir>/<level>` directly by running it from that directory.

Usage:
    python baseline_solution/generate_eval_dataset.py --samples-per-level 20 \\
        --rotation-max-deg 3.0 --output-dir ./baseline_solution/eval_data
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.presets import PRESETS, DRAM_PRESET_NAMES
from src.noise_levels import NOISE_LEVELS
from src.dataset_levels import generate_level  # shared with generate_dataset.py


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples-per-level", type=int, default=20)
    p.add_argument("--rotation-max-deg", type=float, default=3.0,
                    help="each sample draws its search-image rotation uniformly from [0, this]")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--architectures", nargs="+", default=DRAM_PRESET_NAMES,
                    choices=list(PRESETS.keys()),
                    help="presets to cycle through per sample (default: all DRAM presets)")
    p.add_argument("--output-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "eval_data"))
    args = p.parse_args()

    for i, level in enumerate(NOISE_LEVELS):
        level_dir = os.path.join(args.output_dir, level["label"])
        os.makedirs(level_dir, exist_ok=True)
        generate_level(level, args.samples_per_level, args.rotation_max_deg,
                        args.seed + i * 9973, level_dir, args.architectures)


if __name__ == "__main__":
    main()
