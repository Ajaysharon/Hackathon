#!/usr/bin/env python3
"""Shared harness for scoring a matcher against a saved level-structured dataset.

Both `evaluate_localize.py` (GPU FFT-NCC solver) and `evaluate_zncc.py` (the ZNCC
baseline) do the same thing to the same images and differ only in which matcher
they call, so the case loading, per-case CSV, PR summary, plots and summary table
live here once. The precision/recall definitions themselves come from
`evaluate.py` -- this module imports them rather than restating them.

A matcher is any callable `(reference_img, search_img) -> {"x", "y", "score", ...}`;
extra keys are recorded when present ("zoom"/"theta" for the GPU solver, "scale"
for ZNCC).
"""

import csv
import glob
import json
import math
import os
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline_solution.evaluate import pr_curve, average_precision

# Categorical slots 1/3/4/8 of the reference palette, validated for CVD
# separation on a light surface. Marker shape varies with them so the four
# levels stay distinguishable without relying on colour alone.
LEVEL_STYLE = {
    "low":    ("#2a78d6", "o"),
    "medium": ("#1baf7a", "s"),
    "high":   ("#eda100", "^"),
    "severe": ("#e34948", "D"),
}
GRID = "#d8d8d4"
INK = "#26261f"

CSV_FIELDS = ["image", "gt_x", "gt_y", "pred_x", "pred_y", "error_px", "passed",
              "time_s", "score", "zoom_pred", "zoom_gt", "rot_pred", "rot_gt"]


def csv_path_for(output_dir, label, prefix):
    return os.path.join(output_dir, f"{prefix}_{label}.csv")


def load_cases(root, label, limit=None):
    """Every `<root>/<label>/dNN/` case, with its ground truth."""
    base = os.path.join(root, label)
    cases = []
    for d in sorted(glob.glob(os.path.join(base, "d*"))):
        meta_path = os.path.join(d, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        cases.append({"name": os.path.basename(d), "dir": d, "meta": meta})
    if not cases:
        raise SystemExit(f"no cases found under {base} -- generate the dataset first")
    return cases[:limit] if limit else cases


def run_matcher(root, label, matcher, tolerance_px, limit=None, verbose=True):
    """Score every case of one level. Returns the per-case rows."""
    rows = []
    for case in load_cases(root, label, limit):
        meta = case["meta"]
        ref = cv2.imread(os.path.join(case["dir"], "reference.png"), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(os.path.join(case["dir"], "search.png"), cv2.IMREAD_GRAYSCALE)

        t0 = time.perf_counter()
        m = matcher(ref, search)
        dt = time.perf_counter() - t0

        err = math.hypot(m["x"] - meta["gt_x"], m["y"] - meta["gt_y"])
        rows.append({
            "image": case["name"],
            "gt_x": meta["gt_x"], "gt_y": meta["gt_y"],
            "pred_x": m["x"], "pred_y": m["y"],
            "error_px": err,
            "passed": int(err < tolerance_px),
            "time_s": dt,
            "score": m["score"],
            # ZNCC reports the scale it matched at, not a refined zoom; either
            # way this column is the matcher's estimate of the same quantity.
            "zoom_pred": m.get("zoom", m.get("scale", "")),
            "zoom_gt": meta["params"]["zoom"],
            "rot_pred": m.get("theta", ""),
            "rot_gt": meta["rotation_deg"],
        })
        if verbose:
            r = rows[-1]
            print(f"{label}/{r['image']}: pred=({r['pred_x']:7.2f},{r['pred_y']:7.2f}) "
                  f"gt=({r['gt_x']:7.2f},{r['gt_y']:7.2f}) err={err:7.2f}  "
                  f"score={r['score']:.3f} t={dt:5.2f}s "
                  f"{'PASS' if r['passed'] else 'FAIL'}")
    return rows


def write_rows(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def load_rows(path):
    """Read back per-case rows written by `write_rows` (or by localize.py)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k != "image" and v != "":
                r[k] = float(v)
    return rows


def summarize(label, rows, tolerance_px):
    errs = np.array([float(r["error_px"]) for r in rows])
    times = np.array([float(r["time_s"]) for r in rows])
    scores = [float(r["score"]) for r in rows]
    corrects = errs <= tolerance_px

    precision, recall = pr_curve(scores, corrects, len(rows))
    return {
        "label": label,
        "n": len(rows),
        "precision": precision,
        "recall": recall,
        "ap": average_precision(precision, recall),
        "accuracy": float(corrects.mean()),
        "mean_error_px": float(errs.mean()),
        "median_error_px": float(np.median(errs)),
        "worst_error_px": float(errs.max()),
        "mean_time_s": float(times.mean()),
    }


def plot_pr(results, tolerance_px, path, matcher_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    for res in results:
        color, marker = LEVEL_STYLE.get(res["label"], ("#4a3aa7", "o"))
        ax.plot(res["recall"], res["precision"], marker=marker, markersize=4,
                linewidth=2, color=color, markeredgecolor="white",
                markeredgewidth=0.6,
                label=f"{res['label']} (AP={res['ap']:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{matcher_name}: Precision-Recall by noise level "
                 f"(tol={tolerance_px}px)", color=INK)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, labelcolor=INK)
    ax.grid(alpha=0.35, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot_trend(results, tolerance_px, path, matcher_name):
    labels = [r["label"] for r in results]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, [r["ap"] for r in results], marker="o", markersize=8, linewidth=2,
            color="#2a78d6", markeredgecolor="white", markeredgewidth=0.8,
            label="Average Precision")
    ax.plot(x, [r["accuracy"] for r in results], marker="s", markersize=8,
            linewidth=2, color="#eb6834", markeredgecolor="white",
            markeredgewidth=0.8, label=f"Accuracy (<= {tolerance_px}px)")
    # Direct labels on the accuracy series -- the operating number people quote.
    for xi, r in zip(x, results):
        ax.annotate(f"{r['accuracy']*100:.1f}%", (xi, r["accuracy"]),
                    textcoords="offset points", xytext=(0, 11),
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(f"{matcher_name} quality vs noise level", color=INK)
    ax.legend(frameon=False, labelcolor=INK)
    ax.grid(alpha=0.35, color=GRID, linewidth=0.8, axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def write_summary(results, tolerance_px, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "n", f"pass@{tolerance_px}px", "ap",
                    "mean_error_px", "median_error_px", "worst_error_px",
                    "mean_time_s"])
        for r in results:
            w.writerow([r["label"], r["n"], f"{r['accuracy']:.4f}", f"{r['ap']:.4f}",
                        f"{r['mean_error_px']:.3f}", f"{r['median_error_px']:.3f}",
                        f"{r['worst_error_px']:.3f}", f"{r['mean_time_s']:.4f}"])
    print("wrote", path)


def report(results, tolerance_px, output_dir, matcher_name):
    """Write both plots and the summary table for a finished run."""
    os.makedirs(output_dir, exist_ok=True)
    plot_pr(results, tolerance_px, os.path.join(output_dir, "pr_curves.png"), matcher_name)
    plot_trend(results, tolerance_px, os.path.join(output_dir, "ap_vs_noise.png"), matcher_name)
    write_summary(results, tolerance_px, os.path.join(output_dir, "summary.csv"))

    n = sum(r["n"] for r in results)
    overall = sum(r["accuracy"] * r["n"] for r in results) / max(n, 1)
    print(f"\noverall pass@{tolerance_px}px across {n} cases: {overall*100:.1f}%")
