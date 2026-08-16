"""Acquisition-noise levels shared by the dataset generator and the evaluators.

Each entry is a `label` plus a set of GenerationParams overrides describing one
acquisition regime, from a clean high-dose capture ("low") down to a fast,
starved, drifting one ("severe"). Keeping them in one place means the dataset
written by `generate_dataset.py --samples-per-level` and the on-the-fly samples
used by `baseline_solution/evaluate.py` describe the same four regimes.
"""

NOISE_LEVELS = [
    {"label": "low", "dose_search": 800.0, "detector_noise_sigma_search": 2.0,
     "shear_amplitude_px": 0.5, "drift_jitter_px": 0.2},
    {"label": "medium", "dose_search": 200.0, "detector_noise_sigma_search": 5.0,
     "shear_amplitude_px": 1.5, "drift_jitter_px": 0.5},
    {"label": "high", "dose_search": 60.0, "detector_noise_sigma_search": 8.0,
     "shear_amplitude_px": 2.5, "drift_jitter_px": 1.0, "speckle_sigma": 0.15},
    {"label": "severe", "dose_search": 25.0, "detector_noise_sigma_search": 12.0,
     "shear_amplitude_px": 4.0, "drift_jitter_px": 1.8, "speckle_sigma": 0.3,
     "salt_pepper_prob": 0.01},
]

LEVEL_LABELS = [lv["label"] for lv in NOISE_LEVELS]
