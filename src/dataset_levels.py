"""Write a noise-level-structured dataset split to disk.

Layout (one folder per level, one folder per sample):

    <output-dir>/<level>/d00/reference.png
    <output-dir>/<level>/d00/search.png
    <output-dir>/<level>/d00/metadata.json

This is the shape the solver (`localize.py`) and the PR-curve evaluators read:
`metadata.json` carries `gt_x`, `gt_y`, `rotation_deg` and the full
`GenerationParams` used, so a run is reproducible and inspectable instead of
living in throwaway temp files.
"""

import json
import os

import cv2
import numpy as np

from src.pipeline import GenerationParams, generate_sample
from src.noise_levels import NOISE_LEVELS


def generate_level(level, n_samples, rotation_max_deg, base_seed, level_dir,
                   architectures, verbose=True):
    """Write `n_samples` cases for one noise level into `level_dir`."""
    os.makedirs(level_dir, exist_ok=True)
    rng = np.random.default_rng(base_seed)
    overrides = {k: v for k, v in level.items() if k != "label"}

    for i in range(n_samples):
        # Cycle architectures deterministically (not randomly) so every level
        # gets the exact same architecture distribution -- otherwise mix luck
        # can swamp the noise-level signal at moderate sample counts.
        arch = architectures[i % len(architectures)]
        params = GenerationParams(rotation_max_deg=rotation_max_deg, **overrides)
        sample = generate_sample(arch, rng, params)

        sample_dir = os.path.join(level_dir, f"d{i:02d}")
        os.makedirs(sample_dir, exist_ok=True)

        cv2.imwrite(os.path.join(sample_dir, "reference.png"), sample["reference_img"])
        cv2.imwrite(os.path.join(sample_dir, "search.png"), sample["search_img"])

        gx0, gy0, gw, gh = sample["gt_box"]
        metadata = {
            "gt_x": sample["gt_x"],
            "gt_y": sample["gt_y"],
            "gt_box": {"x": gx0, "y": gy0, "w": gw, "h": gh},
            "architecture": arch,
            "rotation_deg": sample["rotation_deg"],
            "noise_level": level["label"],
            "params": sample["params"],
        }
        with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    if verbose:
        print(f"wrote {n_samples} samples to {level_dir}")


def generate_levels(output_dir, samples_per_level, rotation_max_deg, seed,
                    architectures, levels=None, verbose=True):
    """Write every noise level. Per-level seeds are `seed + i * 9973`."""
    levels = levels if levels is not None else NOISE_LEVELS
    for i, level in enumerate(levels):
        generate_level(level, samples_per_level, rotation_max_deg,
                       seed + i * 9973, os.path.join(output_dir, level["label"]),
                       architectures, verbose=verbose)
    return output_dir
