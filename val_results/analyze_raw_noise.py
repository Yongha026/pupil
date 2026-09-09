#!/usr/bin/env python3
"""
analyze_raw_noise.py  –  Offline diagnostic tool for raw pupil/gaze export CSVs.

Usage:
    python val_results/analyze_raw_noise.py                       # latest run
    python val_results/analyze_raw_noise.py <path_to_run_dir>     # specific run
    python val_results/analyze_raw_noise.py --list                # list all runs

Reads from val_results/raw_data/<session_type>_<model>_<timestamp>/
and reports:
  • Pupil Pixel Jitter     – median / 95th / 99th percentile
  • Gaze Angular Error     – distribution buckets (<1°, 1–2°, 2–5°, 5–10°, >10°)
  • Outlier Analysis       – high-error samples vs. jitter / confidence
  • Summary statistics
"""
import argparse
import os
import sys
import csv
import math
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("numpy is required: pip install numpy")
    sys.exit(1)

# ─── Path helpers ─────────────────────────────────────────────────────────────

def find_raw_data_dir() -> Path:
    """Return the val_results/raw_data directory relative to this script."""
    script_dir = Path(__file__).resolve().parent
    return script_dir / "raw_data"


def list_run_dirs(raw_dir: Path) -> list[Path]:
    """Return sorted list of run directories (newest last)."""
    if not raw_dir.exists():
        return []
    dirs = sorted(
        [d for d in raw_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    return dirs


def pick_run(raw_dir: Path, explicit: str | None) -> Path | None:
    dirs = list_run_dirs(raw_dir)
    if explicit:
        p = Path(explicit)
        if p.exists() and p.is_dir():
            return p
        # try relative to raw_dir
        p2 = raw_dir / explicit
        if p2.exists():
            return p2
        print(f"  [!] Directory not found: {explicit}")
        return None
    if not dirs:
        print("  [!] No run directories found in:", raw_dir)
        return None
    return dirs[-1]  # newest


# ─── CSV loaders ──────────────────────────────────────────────────────────────

def _float(val):
    try:
        v = float(val)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def load_pupil_positions(run_dir: Path) -> list[dict]:
    path = run_dir / "pupil_positions.csv"
    if not path.exists():
        print(f"  [!] pupil_positions.csv not found in {run_dir}")
        return []
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "timestamp": _float(row.get("pupil_timestamp")),
                "eye_id": row.get("eye_id"),
                "confidence": _float(row.get("confidence")),
                "raw_confidence": _float(row.get("raw_confidence")),
                "ellipse_cx": _float(row.get("ellipse_center_x")),
                "ellipse_cy": _float(row.get("ellipse_center_y")),
                "raw_cx": _float(row.get("raw_center_x")),
                "raw_cy": _float(row.get("raw_center_y")),
                "pixel_jitter": _float(row.get("pixel_jitter")),
            })
    return rows


def load_evaluation(run_dir: Path) -> list[dict]:
    path = run_dir / "evaluation.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "sample_idx": row.get("sample_idx"),
                "model": row.get("model"),
                "session_type": row.get("session_type"),
                "ref_norm_x": _float(row.get("ref_norm_x")),
                "ref_norm_y": _float(row.get("ref_norm_y")),
                "gaze_norm_x": _float(row.get("gaze_norm_x")),
                "gaze_norm_y": _float(row.get("gaze_norm_y")),
                "angular_error_deg": _float(row.get("angular_error_deg")),
                "is_outlier": row.get("is_outlier", "").lower() in ("true", "1", "yes"),
                "pupil_confidence": _float(row.get("pupil_confidence")),
                "raw_cx": _float(row.get("raw_cx")),
                "raw_cy": _float(row.get("raw_cy")),
                "pixel_jitter": _float(row.get("pixel_jitter")),
            })
    return rows


def load_export_info(run_dir: Path) -> dict:
    path = run_dir / "export_info.csv"
    if not path.exists():
        return {}
    info = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                info[row[0]] = row[1]
    return info


# --- Analysis routines --------------------------------------------------------

def analyze_pupil_jitter(pupils: list[dict]):
    print("\n===  Pupil Pixel Jitter  ===")
    jitters = [p["pixel_jitter"] for p in pupils if p["pixel_jitter"] is not None]
    if not jitters:
        print("  No pixel_jitter data found (was detector_2d_nn_plugin.py patched?)")
        return

    arr = np.array(jitters)
    print(f"  Samples with jitter data : {len(arr):,}")
    print(f"  Mean                     : {arr.mean():.2f} px")
    print(f"  Median (p50)             : {np.median(arr):.2f} px")
    print(f"  p75                      : {np.percentile(arr, 75):.2f} px")
    print(f"  p95                      : {np.percentile(arr, 95):.2f} px")
    print(f"  p99                      : {np.percentile(arr, 99):.2f} px")
    print(f"  Max                      : {arr.max():.2f} px")

    # Histogram buckets
    buckets = [0, 2, 5, 10, 20, 40, float("inf")]
    labels = ["0-2 px", "2-5 px", "5-10 px", "10-20 px", "20-40 px", ">40 px"]
    print("\n  Jitter distribution:")
    for lo, hi, label in zip(buckets, buckets[1:], labels):
        count = np.sum((arr >= lo) & (arr < hi))
        pct = 100.0 * count / len(arr) if len(arr) > 0 else 0.0
        bar = "#" * int(pct / 2)
        print(f"    {label:12s}: {count:5,}  ({pct:5.1f}%)  {bar}")


def analyze_gaze_accuracy(evals: list[dict]):
    print("\n===  Gaze Angular Error  ===")
    errors = [e["angular_error_deg"] for e in evals if e["angular_error_deg"] is not None]
    if not errors:
        print("  No evaluation.csv data found (validation-only).")
        return

    arr = np.array(errors)
    print(f"  Matched samples          : {len(arr):,}")
    print(f"  Mean angular error       : {arr.mean():.3f} deg")
    print(f"  Median (p50)             : {np.median(arr):.3f} deg")
    print(f"  p75                      : {np.percentile(arr, 75):.3f} deg")
    print(f"  p95                      : {np.percentile(arr, 95):.3f} deg")
    print(f"  p99                      : {np.percentile(arr, 99):.3f} deg")
    print(f"  Max                      : {arr.max():.3f} deg")

    buckets = [0, 1, 2, 5, 10, float("inf")]
    labels = ["< 1 deg", "1-2 deg", "2-5 deg", "5-10 deg", ">= 10 deg"]
    print("\n  Error distribution:")
    for lo, hi, label in zip(buckets, buckets[1:], labels):
        count = np.sum((arr >= lo) & (arr < hi))
        pct = 100.0 * count / len(arr) if len(arr) > 0 else 0.0
        bar = "#" * int(pct / 2)
        print(f"    {label:12s}: {count:5,}  ({pct:5.1f}%)  {bar}")


def analyze_outliers(evals: list[dict], pupils: list[dict]):
    print("\n===  Outlier Analysis  ===")
    errors = [(e["angular_error_deg"], e["pixel_jitter"], e["pupil_confidence"])
              for e in evals
              if e["angular_error_deg"] is not None]
    if not errors:
        print("  No data.")
        return

    ang_arr = np.array([x[0] for x in errors])
    jit_arr = np.array([x[1] if x[1] is not None else float("nan") for x in errors])
    conf_arr = np.array([x[2] if x[2] is not None else float("nan") for x in errors])

    outlier_mask = ang_arr > 5.0  # > 5 deg as "bad" samples
    normal_mask = ~outlier_mask

    n_total = len(ang_arr)
    n_out = int(outlier_mask.sum())
    print(f"  Outliers (>5 deg)        : {n_out:,} / {n_total:,}  ({100*n_out/n_total:.1f}%)")
    print(f"  Outliers (>10 deg)       : {int((ang_arr>10).sum()):,} / {n_total:,}  ({100*(ang_arr>10).mean():.1f}%)")

    if n_out > 0 and not np.all(np.isnan(jit_arr[outlier_mask])):
        print(f"\n  Pixel jitter - outlier samples:")
        out_j = jit_arr[outlier_mask]
        out_j = out_j[~np.isnan(out_j)]
        if len(out_j):
            print(f"    median = {np.median(out_j):.2f} px,  p95 = {np.percentile(out_j, 95):.2f} px")

    if normal_mask.sum() > 0 and not np.all(np.isnan(jit_arr[normal_mask])):
        print(f"  Pixel jitter - normal samples:")
        ok_j = jit_arr[normal_mask]
        ok_j = ok_j[~np.isnan(ok_j)]
        if len(ok_j):
            print(f"    median = {np.median(ok_j):.2f} px,  p95 = {np.percentile(ok_j, 95):.2f} px")

    if n_out > 0 and not np.all(np.isnan(conf_arr[outlier_mask])):
        out_c = conf_arr[outlier_mask]
        out_c = out_c[~np.isnan(out_c)]
        if len(out_c):
            print(f"\n  Confidence - outlier samples : mean = {out_c.mean():.3f},  min = {out_c.min():.3f}")

    if normal_mask.sum() > 0 and not np.all(np.isnan(conf_arr[normal_mask])):
        ok_c = conf_arr[normal_mask]
        ok_c = ok_c[~np.isnan(ok_c)]
        if len(ok_c):
            print(f"  Confidence - normal samples  : mean = {ok_c.mean():.3f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze raw pupil/gaze export data")
    parser.add_argument("run_dir", nargs="?", default=None, help="Run directory to analyze")
    parser.add_argument("--list", action="store_true", help="List available run directories")
    args = parser.parse_args()

    raw_dir = find_raw_data_dir()

    if args.list:
        dirs = list_run_dirs(raw_dir)
        if not dirs:
            print("No run directories found in:", raw_dir)
        else:
            print(f"Run directories in {raw_dir}:")
            for d in dirs:
                tag = "  <-- latest" if d == dirs[-1] else ""
                print(f"  {d.name}{tag}")
        return

    run_dir = pick_run(raw_dir, args.run_dir)
    if run_dir is None:
        return

    print(f"\n{'='*60}")
    print(f"  Run: {run_dir.name}")
    print(f"{'='*60}")

    info = load_export_info(run_dir)
    if info:
        print("\n===  Export Info  ===")
        for k, v in info.items():
            print(f"  {k:<24s}: {v}")

    pupils = load_pupil_positions(run_dir)
    evals = load_evaluation(run_dir)

    if not pupils and not evals:
        print("\n[!] No data found. Check that the run directory contains CSVs.")
        return

    if pupils:
        analyze_pupil_jitter(pupils)

    if evals:
        analyze_gaze_accuracy(evals)
        analyze_outliers(evals, pupils)
    else:
        print("\n===  Gaze Accuracy  ===")
        print("  evaluation.csv not found - only available after a Validation (Testing) run.")

    print(f"\n{'='*60}")
    print("  Analysis complete.")
    print(f"  Run directory: {run_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
