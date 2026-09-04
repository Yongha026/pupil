#!/usr/bin/env python3
"""
Pupil Labs Process Cycle Latency Profiler
=========================================
Visualizes the execution cycle (Detection -> Display) of Pupil Labs processes
(world, eye0, eye1), explicitly distinguishing:
  - Initial Booting Latency: Evaluated using the RAW datum (cold-start / initialization).
  - Steady-State Looping Latency: Evaluated using the AVERAGE of the subsequent looping data.

Visualizations:
  1. Process Cycle Overview: Initial Boot (Raw) vs. Looping (Average) across processes.
  2. Cycle Stage Breakdown: Comparison of Detection vs. Display in Boot vs. Loop.
  3. Steady-State Looping Zoom-In: High-resolution breakdown of recurring frame latencies (Mean ± Std).
  4. Cycle Progression Curve: Cold-start spike (raw) transitioning into steady-state average baseline.

Usage:
  python latency_plot_cycle.py [CSV_PATH] [--output OUTPUT_PATH] [--mode {all,overview,breakdown,progression}]
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

# Use non-interactive Agg backend by default for remote/headless environments
if "--show" not in sys.argv:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def find_latest_latency_csv(base_dir: Optional[str] = None) -> Optional[str]:
    """Find the most recently modified latency CSV file in logged_latencies or root."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    candidates: List[str] = []
    log_dir = os.path.join(base_dir, "logged_latencies")
    if os.path.isdir(log_dir):
        candidates.extend(glob.glob(os.path.join(log_dir, "latency_*.csv")))

    candidates.extend(glob.glob(os.path.join(base_dir, "latency_*.csv")))
    fallback = os.path.join(base_dir, "latency_logs.csv")
    if os.path.exists(fallback):
        candidates.append(fallback)

    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load CSV, validate required fields, and ensure sorted chronological order."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    expected_cols = {"process", "plugin", "processing_latency_ms", "t_start", "t_end"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")

    for col in [
        "processing_latency_ms",
        "std_latency_ms",
        "min_latency_ms",
        "max_latency_ms",
        "p95_latency_ms",
        "e2e_latency_ms",
        "sample_count",
        "t_start",
        "t_end",
        "frame_timestamp",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "stage" not in df.columns:
        df["stage"] = df.get("phase", "loop")
    if "phase" not in df.columns:
        df["phase"] = "loop"

    if "model" not in df.columns:
        df["model"] = "unknown"
    else:
        df["model"] = df["model"].fillna("unknown")

    df = df.dropna(subset=["t_start", "t_end", "processing_latency_ms"])
    df = df[df["t_end"] >= df["t_start"]]

    if df.empty:
        raise ValueError(f"CSV '{csv_path}' contains no valid latency rows.")

    df = df.sort_values(by="t_start").reset_index(drop=True)
    return df


def extract_cycle_statistics(df: pd.DataFrame, metric: str = "processing_latency_ms") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separates Initial Booting (first raw datum) from Looping (average of subsequent data).
    Returns:
      1. detailed_df: By (process, plugin, stage)
      2. process_summary_df: Aggregated by process cycle
    """
    detailed_records: List[Dict] = []

    for (proc, plugin, stage), group in df.groupby(["process", "plugin", "stage"], sort=False):
        group_sorted = group.sort_values(by="t_start").reset_index(drop=True)
        if group_sorted.empty:
            continue

        model_name = group_sorted.iloc[0].get("model", "unknown")

        # Check if phase column explicitly identifies boot vs loop
        has_phase_boot = "boot" in group_sorted["phase"].values

        if has_phase_boot:
            boot_group = group_sorted[group_sorted["phase"] == "boot"]
            loop_group = group_sorted[group_sorted["phase"] == "loop"]
        else:
            boot_group = group_sorted.iloc[:1]
            loop_group = group_sorted.iloc[1:]

        # 1. Initial Booting: exact raw occurrence
        if not boot_group.empty:
            boot_row = boot_group.iloc[0]
            boot_val = float(boot_row[metric])
            detailed_records.append({
                "process": proc,
                "plugin": plugin,
                "model": model_name,
                "stage": stage,
                "phase": "Initial Booting",
                "val_type": "Raw Datum",
                "latency": boot_val,
                "std": 0.0,
                "p95": boot_val,
                "count": 1,
                "t_start": boot_row["t_start"],
            })

        # 2. Looping Data: subsequent occurrences / window averages
        if not loop_group.empty:
            loop_mean = float(loop_group[metric].mean())
            if "std_latency_ms" in loop_group and loop_group["std_latency_ms"].max() > 0:
                loop_std = float(loop_group["std_latency_ms"].mean())
            else:
                loop_std = float(loop_group[metric].std(ddof=1)) if len(loop_group) > 1 else 0.0

            if "p95_latency_ms" in loop_group and loop_group["p95_latency_ms"].max() > 0:
                loop_p95 = float(loop_group["p95_latency_ms"].mean())
            else:
                loop_p95 = float(np.percentile(loop_group[metric], 95))

            sample_cnt = int(loop_group["sample_count"].sum()) if "sample_count" in loop_group else len(loop_group)

            detailed_records.append({
                "process": proc,
                "plugin": plugin,
                "model": model_name,
                "stage": stage,
                "phase": "Looping Average",
                "val_type": "Average of Datum",
                "latency": loop_mean,
                "std": loop_std,
                "p95": loop_p95,
                "count": sample_cnt,
                "t_start": loop_group["t_start"].min(),
            })

    detailed_df = pd.DataFrame(detailed_records)

    # Aggregate by process for overall cycle: Detection + Display
    process_cycle_records: List[Dict] = []
    for proc in detailed_df["process"].unique():
        sub = detailed_df[detailed_df["process"] == proc]

        for phase, val_type in [("Initial Booting", "Raw Datum"), ("Looping Average", "Average of Datum")]:
            phase_sub = sub[sub["phase"] == phase]
            if phase_sub.empty:
                continue

            detect_val = phase_sub[phase_sub["stage"] == "detect"]["latency"].sum()
            detect_std = phase_sub[phase_sub["stage"] == "detect"]["std"].sum()
            # If no explicit detect stage exists (e.g. world capture process), fallback to recent_events
            if detect_val == 0.0:
                detect_val = phase_sub[phase_sub["stage"] == "recent_events"]["latency"].sum()
                detect_std = phase_sub[phase_sub["stage"] == "recent_events"]["std"].sum()

            display_val = phase_sub[phase_sub["stage"] == "gl_display"]["latency"].sum()
            display_std = phase_sub[phase_sub["stage"] == "gl_display"]["std"].sum()
            total_cycle_val = detect_val + display_val

            # Compute combined standard deviation for looping cycle
            total_std = np.sqrt(detect_std**2 + display_std**2)

            process_cycle_records.append({
                "process": proc,
                "phase": phase,
                "val_type": val_type,
                "detect_ms": detect_val,
                "display_ms": display_val,
                "total_cycle_ms": total_cycle_val,
                "total_std": total_std if phase == "Looping Average" else 0.0,
            })

    process_summary_df = pd.DataFrame(process_cycle_records)
    return detailed_df, process_summary_df


def plot_process_overview(ax: plt.Axes, summary_df: pd.DataFrame):
    """Subplot 1: Total Process Cycle Latency (Booting Raw vs Looping Average)."""
    processes = sorted(summary_df["process"].unique())
    x = np.arange(len(processes))
    width = 0.35

    boot_vals = []
    loop_vals = []
    loop_errs = []

    for proc in processes:
        b = summary_df[(summary_df["process"] == proc) & (summary_df["phase"] == "Initial Booting")]
        l = summary_df[(summary_df["process"] == proc) & (summary_df["phase"] == "Looping Average")]
        boot_vals.append(b["total_cycle_ms"].values[0] if not b.empty else 0.0)
        loop_vals.append(l["total_cycle_ms"].values[0] if not l.empty else 0.0)
        loop_errs.append(l["total_std"].values[0] if not l.empty else 0.0)

    rects1 = ax.bar(
        x - width / 2,
        boot_vals,
        width,
        label="Initial Booting (Raw Datum)",
        color="#e74c3c",
        alpha=0.88,
        edgecolor="#c0392b",
        linewidth=1.2,
    )
    rects2 = ax.bar(
        x + width / 2,
        loop_vals,
        width,
        yerr=loop_errs,
        capsize=4,
        label="Looping Cycle (Average Datum)",
        color="#2ecc71",
        alpha=0.88,
        edgecolor="#27ae60",
        linewidth=1.2,
    )

    # Add text labels on top of bars
    for rect in rects1:
        height = rect.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f} ms\n(Raw)",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
                color="#962d22",
            )

    for i, rect in enumerate(rects2):
        height = rect.get_height()
        if height > 0:
            speedup = (boot_vals[i] / height) if height > 0 and boot_vals[i] > 0 else 1.0
            ax.annotate(
                f"{height:.2f} ms\n(Avg) [{speedup:.1f}x]",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
                color="#1e8449",
            )

    ax.set_ylabel("Total Cycle Latency (ms)", fontsize=11, fontweight="bold")
    ax.set_title("Total Process Cycle Latency: Initial Booting (Raw) vs. Steady-State Looping (Average)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(processes, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    max_h = max(max(boot_vals, default=1.0), max(loop_vals, default=1.0))
    ax.set_ylim(bottom=0, top=max_h * 1.25)


def plot_cycle_breakdown(ax: plt.Axes, summary_df: pd.DataFrame):
    """Subplot 2: Cycle Stages Breakdown (Detection vs. Display)."""
    processes = sorted(summary_df["process"].unique())
    n_proc = len(processes)
    indices = np.arange(n_proc * 2)

    bar_labels = []
    detect_vals = []
    display_vals = []

    for proc in processes:
        for phase, tag in [("Initial Booting", "Boot\n(Raw)"), ("Looping Average", "Loop\n(Avg)")]:
            row = summary_df[(summary_df["process"] == proc) & (summary_df["phase"] == phase)]
            bar_labels.append(f"{proc}\n{tag}")
            if not row.empty:
                detect_vals.append(row["detect_ms"].values[0])
                display_vals.append(row["display_ms"].values[0])
            else:
                detect_vals.append(0.0)
                display_vals.append(0.0)

    p1 = ax.bar(
        indices,
        detect_vals,
        0.55,
        label="Detection (detect)",
        color="#3498db",
        alpha=0.9,
        edgecolor="#2980b9",
        linewidth=1.1,
    )
    p2 = ax.bar(
        indices,
        display_vals,
        0.55,
        bottom=detect_vals,
        label="Display (gl_display)",
        color="#f39c12",
        alpha=0.9,
        edgecolor="#d68910",
        linewidth=1.1,
    )

    # Annotate total on stack top
    for i in range(len(indices)):
        tot = detect_vals[i] + display_vals[i]
        if tot > 0:
            ax.annotate(
                f"{tot:.2f} ms",
                xy=(indices[i], tot),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=8.5,
            )

    ax.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
    ax.set_title("Cycle Stage Decomposition: Detection vs. Display per Process Phase", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(indices)
    ax.set_xticklabels(bar_labels, fontsize=9.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    max_tot = max([d + g for d, g in zip(detect_vals, display_vals)], default=1.0)
    ax.set_ylim(bottom=0, top=max_tot * 1.25)


def plot_looping_zoom(ax: plt.Axes, summary_df: pd.DataFrame):
    """Subplot 3: High-Resolution View of Steady-State Looping Cycle (Mean ± Std)."""
    loop_df = summary_df[summary_df["phase"] == "Looping Average"].copy()
    if loop_df.empty:
        ax.text(0.5, 0.5, "No Looping Data Available", ha="center", va="center")
        return

    x = np.arange(len(loop_df))
    width = 0.35

    ax.bar(
        x - width / 2,
        loop_df["detect_ms"],
        width,
        label="Looping Detection (Mean)",
        color="#2980b9",
        alpha=0.88,
        edgecolor="#1f618d",
    )
    ax.bar(
        x + width / 2,
        loop_df["display_ms"],
        width,
        label="Looping Display (Mean)",
        color="#e67e22",
        alpha=0.88,
        edgecolor="#b96517",
    )

    for i, (_, row) in enumerate(loop_df.iterrows()):
        d_val = row["detect_ms"]
        g_val = row["display_ms"]
        ax.annotate(
            f"{d_val:.2f} ms",
            xy=(i - width / 2, d_val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
            color="#1b4f72",
        )
        ax.annotate(
            f"{g_val:.3f} ms",
            xy=(i + width / 2, g_val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
            color="#7e430b",
        )

    ax.set_ylabel("Looping Latency (ms)", fontsize=11, fontweight="bold")
    ax.set_title("Steady-State Looping Cycle Detail (Detection vs. Display Averages)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p} (Loop)" for p in loop_df["process"]], fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    max_val = max(loop_df["detect_ms"].max(), loop_df["display_ms"].max(), 0.1)
    ax.set_ylim(bottom=0, top=max_val * 1.3)


def plot_cycle_progression(ax: plt.Axes, raw_df: pd.DataFrame, metric: str = "processing_latency_ms", max_cycles: int = 50):
    """Subplot 4: Cycle Progression (First N cycles showing Boot spike -> stable average baseline)."""
    detect_df = raw_df[raw_df["stage"] == "detect"].copy()
    if detect_df.empty:
        detect_df = raw_df

    processes = sorted(detect_df["process"].unique())
    palette = sns.color_palette("tab10", n_colors=max(len(processes), 3))

    for idx, proc in enumerate(processes):
        sub = detect_df[detect_df["process"] == proc].sort_values(by="t_start").reset_index(drop=True)
        if sub.empty:
            continue

        plot_sub = sub.iloc[:max_cycles].copy()
        cycles = np.arange(len(plot_sub))
        vals = plot_sub[metric].values

        # Plot cycle values
        color = palette[idx % len(palette)]
        ax.plot(
            cycles,
            vals,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=f"{proc}: Detection Cycles",
            color=color,
            alpha=0.85,
        )

        # Draw steady-state looping average line
        if len(sub) > 1:
            loop_mean = sub.iloc[1:][metric].mean()
            ax.axhline(
                y=loop_mean,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                label=f"{proc}: Looping Avg ({loop_mean:.2f} ms)",
            )

        # Highlight Initial Boot Raw Datum (Cycle 0)
        boot_val = vals[0]
        ax.scatter([0], [boot_val], color="#c0392b", s=65, zorder=5)
        ax.annotate(
            f"Boot Raw:\n{boot_val:.1f}ms",
            xy=(0, boot_val),
            xytext=(10, -5),
            textcoords="offset points",
            fontweight="bold",
            fontsize=8.5,
            color="#900c3f",
        )

    ax.set_xlabel("Detection Cycle Number (0 = Initial Booting)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Latency ({metric.replace('_', ' ')})", fontsize=11, fontweight="bold")
    ax.set_title(f"Cycle Progression: Booting Cold-Start Spike vs. Looping Average (First {max_cycles} Cycles)", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)


def print_cycle_summary_table(detailed_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Print structured summary to terminal."""
    print("\n" + "=" * 110)
    print("                           PROCESS CYCLE LATENCY REPORT: BOOTING vs. LOOPING                            ")
    print("=" * 110)
    print(
        f"{'Process':<8} {'Phase':<17} {'Plugin':<24} {'Stage':<13} "
        f"{'Metric Type':<18} {'Latency (ms)':>13} {'Std (ms)':>10} {'Count':>7}"
    )
    print("-" * 110)

    for _, r in detailed_df.iterrows():
        lat_str = f"{r['latency']:.2f}"
        std_str = f"{r['std']:.2f}" if r["val_type"] == "Average of Datum" else "-"
        print(
            f"{str(r['process']):<8} "
            f"{str(r['phase']):<17} "
            f"{str(r['plugin'])[:23]:<24} "
            f"{str(r['stage'])[:12]:<13} "
            f"{str(r['val_type']):<18} "
            f"{lat_str:>13} "
            f"{std_str:>10} "
            f"{int(r['count']):>7}"
        )

    print("-" * 110)
    print("CYCLE SUMMARY (Detection + Display):")
    for _, s in summary_df.iterrows():
        proc = s["process"]
        phase = s["phase"]
        vtype = s["val_type"]
        tot = s["total_cycle_ms"]
        det = s["detect_ms"]
        disp = s["display_ms"]
        std_str = f"± {s['total_std']:.2f}" if phase == "Looping Average" else "(exact raw)"
        print(f"  * [{proc}] {phase:<16} ({vtype}): Total = {tot:.2f} ms {std_str:<12} [Detect: {det:.2f} ms | Display: {disp:.3f} ms]")
    print("=" * 110 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot average process cycle latencies (Detection -> Display) distinguishing Initial Booting (raw datum) from Looping (average datum)."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help="Path to latency CSV file. Defaults to latest in 'logged_latencies/'.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to save output chart image (PNG).",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "overview", "breakdown", "progression"],
        default="all",
        help="Visualization mode: 'all' (4-panel dashboard), 'overview' (process total), 'breakdown' (detect vs display), or 'progression' (cycles 0..N).",
    )
    parser.add_argument(
        "--metric",
        choices=["processing_latency_ms", "e2e_latency_ms"],
        default="processing_latency_ms",
        help="Latency metric to plot. Default: processing_latency_ms.",
    )
    parser.add_argument(
        "--max-progression-cycles",
        type=int,
        default=40,
        help="Number of initial cycles to display in the cycle progression panel. Default: 40.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive GUI window (requires graphical display/X11).",
    )

    args = parser.parse_args()

    # Locate CSV file
    csv_file = args.csv_path
    if not csv_file or not os.path.exists(csv_file):
        csv_file = find_latest_latency_csv()
        if not csv_file:
            print("Error: No CSV file provided and none found in 'logged_latencies/'.")
            sys.exit(1)
        print(f"Using latest latency CSV: {csv_file}")
    else:
        print(f"Loading latency CSV: {csv_file}")

    # Set output image path
    output_path = args.output
    if not output_path:
        csv_stem = os.path.splitext(os.path.basename(csv_file))[0]
        csv_dir = os.path.dirname(os.path.abspath(csv_file))
        output_path = os.path.join(csv_dir, f"{csv_stem}_cycle_plot.png")

    # Load and process data
    try:
        raw_df = load_and_clean_data(csv_file)
        detailed_df, summary_df = extract_cycle_statistics(raw_df, metric=args.metric)
    except Exception as e:
        print(f"Error processing cycle latency data: {e}")
        sys.exit(1)

    # Print summary table
    print_cycle_summary_table(detailed_df, summary_df)

    # Style configuration
    sns.set_theme(style="whitegrid", font_scale=0.95)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # Render plots based on requested mode
    if args.mode == "overview":
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        plot_process_overview(ax, summary_df)
        plt.tight_layout()

    elif args.mode == "breakdown":
        fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
        plot_cycle_breakdown(ax, summary_df)
        plt.tight_layout()

    elif args.mode == "progression":
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        plot_cycle_progression(ax, raw_df, metric=args.metric, max_cycles=args.max_progression_cycles)
        plt.tight_layout()

    else:  # mode == "all"
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2,
            2,
            figsize=(18, 12),
            dpi=150,
            gridspec_kw={"height_ratios": [1.1, 1.0]},
        )
        plot_process_overview(ax1, summary_df)
        plot_cycle_breakdown(ax2, summary_df)
        plot_looping_zoom(ax3, summary_df)
        plot_cycle_progression(ax4, raw_df, metric=args.metric, max_cycles=args.max_progression_cycles)
        plt.tight_layout(pad=3.0)

    # Save output plot
    try:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Successfully generated and saved cycle plot to: {output_path}")
    except Exception as e:
        print(f"Failed to save cycle plot: {e}")

    if args.show:
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display interactive window: {e}")


if __name__ == "__main__":
    main()
