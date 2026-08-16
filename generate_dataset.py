#!/usr/bin/env python3
"""CLI to generate a Drift-Sense synthetic dataset.

Two output modes:

1. Flat split (default) -- one folder of references, one of searches, plus a
   manifest.csv row per sample:

    python generate_dataset.py --num-samples 20 --split train \
        --architectures dram_1x finfet_10nm --output-dir ./output --seed 42

2. Noise-level benchmark (`--samples-per-level`) -- one folder per acquisition
   regime (low/medium/high/severe), one folder per sample, each with
   reference.png / search.png / metadata.json. This is the layout `localize.py`
   and the PR-curve evaluators read:

    python generate_dataset.py --samples-per-level 40 --rotation-max-deg 3.0 \
        --seed 2026 --architectures dram_1x dram_dense dram_loose dram_wide \
        dram_compact dram_legacy --output-dir .dram_synthetic3
"""

import argparse
import csv
import os
import sys
from dataclasses import replace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import GenerationParams, generate_sample
from src.presets import PRESETS
from src.noise_levels import NOISE_LEVELS
from src.dataset_levels import generate_levels


def _uniform_nonneg(rng, hi, lo_frac=0.0):
    """Draw a per-sample value in [hi * lo_frac, hi]. hi<=0 is a no-op passthrough
    (keeps a disabled artifact, e.g. vignette_strength=0.0, disabled for every sample)."""
    if hi <= 0:
        return hi
    return float(rng.uniform(hi * lo_frac, hi))


def _uniform_signed(rng, hi):
    """Draw in [-|hi|, |hi|]; hi==0 is a no-op passthrough."""
    if hi == 0:
        return 0.0
    mag = abs(hi)
    return float(rng.uniform(-mag, mag))


def _jitter_around_one(rng, val):
    """Jitter a multiplicative factor (gamma, astigmatism_ratio) around 1.0 by
    +/- the distance the CLI value already sits from 1.0; val==1.0 (no effect)
    is a no-op passthrough."""
    if val == 1.0:
        return 1.0
    delta = abs(val - 1.0)
    return float(max(1.0 + rng.uniform(-delta, delta), 0.05))


def randomize_imaging_params(base: GenerationParams, rng: np.random.Generator) -> GenerationParams:
    """Redraw the per-*acquisition* fields of GenerationParams for a single
    sample, using the CLI-supplied values as the upper bound / no-op default.
    Structural/geometry fields (collapse_threshold_nm, mat_size_nm,
    strip_width_nm, boundary_bias, linewidth_bias_nm, corner_rounding_px,
    beam_spot_size_nm) are left untouched -- those describe the device/tooling,
    not the per-capture acquisition condition, per the structural-vs-imaging
    split documented in src/sem_imaging.py and src/structural_defects.py.
    """
    return replace(
        base,
        dose_reference=_uniform_nonneg(rng, base.dose_reference, lo_frac=0.4),
        dose_search=_uniform_nonneg(rng, base.dose_search, lo_frac=0.25),
        shear_amplitude_px=_uniform_nonneg(rng, base.shear_amplitude_px),
        drift_jitter_px=_uniform_nonneg(rng, base.drift_jitter_px),
        rotation_max_deg=_uniform_nonneg(rng, base.rotation_max_deg),
        detector_noise_sigma_ref=_uniform_nonneg(rng, base.detector_noise_sigma_ref),
        detector_noise_sigma_search=_uniform_nonneg(rng, base.detector_noise_sigma_search),
        astigmatism_ratio=_jitter_around_one(rng, base.astigmatism_ratio),
        vignette_strength=_uniform_nonneg(rng, base.vignette_strength),
        gamma=_jitter_around_one(rng, base.gamma),
        barrel_distortion_k=_uniform_signed(rng, base.barrel_distortion_k),
        charging_streak_prob=_uniform_nonneg(rng, base.charging_streak_prob),
        charging_streak_intensity=_uniform_nonneg(rng, base.charging_streak_intensity),
        speckle_sigma=_uniform_nonneg(rng, base.speckle_sigma),
        salt_pepper_prob=_uniform_nonneg(rng, base.salt_pepper_prob),
    )


def _load_config(path):
    """Read a YAML/JSON run config into a dict of argparse defaults.

    Keys may use either hyphens or underscores; both map to the argparse dest.
    """
    import json
    with open(path) as f:
        text = f.read()
    try:
        import yaml
        cfg = yaml.safe_load(text)
    except ImportError:                      # pyyaml missing -- JSON still works
        cfg = json.loads(text)
    if not isinstance(cfg, dict):
        raise SystemExit(f"config {path} must contain a mapping")
    return {k.replace("-", "_"): v for k, v in cfg.items()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="YAML/JSON file supplying defaults for any flag below "
                        "(explicit command-line flags still win)")
    p.add_argument("--samples-per-level", type=int, default=None,
                   help="switch to noise-level mode: write this many samples for each of "
                        f"{', '.join(lv['label'] for lv in NOISE_LEVELS)} as "
                        "<output-dir>/<level>/dNN/{reference.png,search.png,metadata.json}")
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--architectures", nargs="+", default=list(PRESETS.keys()), choices=list(PRESETS.keys()))
    p.add_argument("--split", default="train")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beam-spot-size-nm", type=float, default=GenerationParams.beam_spot_size_nm)
    p.add_argument("--collapse-threshold-nm", type=float, default=GenerationParams.collapse_threshold_nm)
    p.add_argument("--dose-reference", type=float, default=GenerationParams.dose_reference)
    p.add_argument("--dose-search", type=float, default=GenerationParams.dose_search)
    p.add_argument("--shear-amplitude-px", type=float, default=GenerationParams.shear_amplitude_px)
    p.add_argument("--drift-jitter-px", type=float, default=GenerationParams.drift_jitter_px)
    p.add_argument("--rotation-max-deg", type=float, default=GenerationParams.rotation_max_deg)
    p.add_argument("--astigmatism-ratio", type=float, default=GenerationParams.astigmatism_ratio)
    p.add_argument("--vignette-strength", type=float, default=GenerationParams.vignette_strength)
    p.add_argument("--gamma", type=float, default=GenerationParams.gamma)
    p.add_argument("--barrel-distortion-k", type=float, default=GenerationParams.barrel_distortion_k)
    p.add_argument("--charging-streak-prob", type=float, default=GenerationParams.charging_streak_prob)
    p.add_argument("--charging-streak-intensity", type=float, default=GenerationParams.charging_streak_intensity)
    p.add_argument("--speckle-sigma", type=float, default=GenerationParams.speckle_sigma)
    p.add_argument("--salt-pepper-prob", type=float, default=GenerationParams.salt_pepper_prob)
    p.add_argument("--linewidth-bias-nm", type=float, default=GenerationParams.linewidth_bias_nm)
    p.add_argument("--corner-rounding-px", type=float, default=GenerationParams.corner_rounding_px)
    p.add_argument("--mat-size-nm", type=float, default=GenerationParams.mat_size_nm)
    p.add_argument("--strip-width-nm", type=float, default=GenerationParams.strip_width_nm)
    p.add_argument("--boundary-bias", type=float, default=GenerationParams.boundary_bias)
    p.add_argument(
        "--zoom-min", type=float, default=GenerationParams.zoom_min,
        help="Lower bound of the per-sample Reference:Search pixel-size ratio "
             "(default 9.0, i.e. 9:1). Set equal to --zoom-max for a fixed ratio.",
    )
    p.add_argument(
        "--zoom-max", type=float, default=GenerationParams.zoom_max,
        help="Upper bound of the per-sample Reference:Search pixel-size ratio "
             "(default 11.0, i.e. 11:1).",
    )
    p.add_argument(
        "--randomize-imaging", dest="randomize_imaging", action="store_true", default=True,
        help="Redraw acquisition-condition params (dose, drift, noise, etc.) per sample, "
             "using the CLI values as upper bounds, instead of using one fixed setting "
             "for the whole split (default: on).",
    )
    p.add_argument(
        "--no-randomize-imaging", dest="randomize_imaging", action="store_false",
        help="Use the exact CLI-supplied GenerationParams for every sample (old fixed behavior).",
    )

    pre, _ = p.parse_known_args()
    if pre.config:
        p.set_defaults(**_load_config(pre.config))
    args = p.parse_args()

    if args.samples_per_level is not None:
        clash = [f for f in ("--num-samples", "--split") if f in sys.argv]
        if clash:
            p.error(f"--samples-per-level is a different output mode and cannot be "
                    f"combined with {', '.join(clash)}")
    return args


def main():
    args = parse_args()

    if args.samples_per_level is not None:
        generate_levels(args.output_dir, args.samples_per_level,
                        args.rotation_max_deg, args.seed, args.architectures)
        total = args.samples_per_level * len(NOISE_LEVELS)
        print(f"Wrote {total} samples ({args.samples_per_level} x "
              f"{len(NOISE_LEVELS)} levels) to {args.output_dir}")
        return

    rng = np.random.default_rng(args.seed)

    params = GenerationParams(
        beam_spot_size_nm=args.beam_spot_size_nm,
        collapse_threshold_nm=args.collapse_threshold_nm,
        dose_reference=args.dose_reference,
        dose_search=args.dose_search,
        shear_amplitude_px=args.shear_amplitude_px,
        drift_jitter_px=args.drift_jitter_px,
        rotation_max_deg=args.rotation_max_deg,
        astigmatism_ratio=args.astigmatism_ratio,
        vignette_strength=args.vignette_strength,
        gamma=args.gamma,
        barrel_distortion_k=args.barrel_distortion_k,
        charging_streak_prob=args.charging_streak_prob,
        charging_streak_intensity=args.charging_streak_intensity,
        speckle_sigma=args.speckle_sigma,
        salt_pepper_prob=args.salt_pepper_prob,
        linewidth_bias_nm=args.linewidth_bias_nm,
        corner_rounding_px=args.corner_rounding_px,
        mat_size_nm=args.mat_size_nm,
        strip_width_nm=args.strip_width_nm,
        boundary_bias=args.boundary_bias,
        zoom_min=args.zoom_min,
        zoom_max=args.zoom_max,
    )

    split_dir = os.path.join(args.output_dir, args.split)
    ref_dir = os.path.join(split_dir, "reference")
    search_dir = os.path.join(split_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    manifest_path = os.path.join(split_dir, "manifest.csv")
    fieldnames = [
        "id", "reference_path", "search_path", "gt_x", "gt_y",
        "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h", "architecture",
        "beam_spot_size_nm", "collapse_threshold_nm", "dose_reference",
        "dose_search", "shear_amplitude_px", "drift_jitter_px",
        "astigmatism_ratio", "vignette_strength", "gamma", "barrel_distortion_k",
        "rotation_deg", "rotation_max_deg",
        "detector_noise_sigma_ref", "detector_noise_sigma_search",
        "charging_streak_prob", "charging_streak_intensity",
        "speckle_sigma", "salt_pepper_prob",
        "linewidth_bias_nm", "corner_rounding_px",
        "mat_size_nm", "strip_width_nm", "boundary_bias", "zoom", "zoom_min", "zoom_max", "seed",
    ]

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for i in range(args.num_samples):
            architecture = args.architectures[int(rng.integers(0, len(args.architectures)))]
            sample_params = randomize_imaging_params(params, rng) if args.randomize_imaging else params
            sample = generate_sample(architecture, rng, sample_params)

            ref_path = os.path.join(ref_dir, f"{i:05d}.png")
            search_path = os.path.join(search_dir, f"{i:05d}.png")
            cv2.imwrite(ref_path, sample["reference_img"])
            cv2.imwrite(search_path, sample["search_img"])

            gx0, gy0, gw, gh = sample["gt_box"]
            writer.writerow({
                "id": i,
                "reference_path": ref_path,
                "search_path": search_path,
                "gt_x": sample["gt_x"],
                "gt_y": sample["gt_y"],
                "gt_box_x": gx0, "gt_box_y": gy0, "gt_box_w": gw, "gt_box_h": gh,
                "architecture": architecture,
                **sample["params"],
                "seed": args.seed,
            })
            print(f"[{i + 1}/{args.num_samples}] {architecture} -> gt=({sample['gt_x']:.1f}, {sample['gt_y']:.1f})")

    print(f"Wrote {args.num_samples} samples to {split_dir}")


if __name__ == "__main__":
    main()
