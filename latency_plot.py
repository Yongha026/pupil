#!/usr/bin/env python3
"""
Pupil Labs Latency & Execution Timeline Profiler
================================================
Visualizes end-to-end and processing latencies recorded across Pupil Labs processes
(world, eye0, eye1) and plugins (Detector2DPlugin, Pye3DPlugin, Roi, UVC_Source, etc.).

Visualizations:
  1. Timeline (Gantt Chart): X-axis is linear elapsed time (start to end);
     Y-axis lists each plugin/process, plotting each invocation interval [t_start, t_end].
  2. Latency vs. Time: Real-time latency trajectory and spikes over linear time.
  3. Latency Breakdown: Mean ± std and distribution bar plot highlighting bottlenecks.

Usage:
  python latency_plot.py [CSV_PATH] [--output OUTPUT_PATH] [--mode {all,timeline,latency,summary}]
"""

import argparse
import glob
import os
import sys
from typing import List, Optional, Tuple

# Use non-interactive Agg backend by default for headless remote environments
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


def load_latency_data(
    csv_path: str,
    warmup_skip_sec: float = 0.0,
    max_points: Optional[int] = None,
    group_by: str = "process_plugin",
) -> pd.DataFrame:
    """Load, clean, and enrich latency CSV data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Latency CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    expected_cols = {"process", "plugin", "stage", "processing_latency_ms", "t_start", "t_end"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    # Coerce numeric columns
    numeric_cols = [
        "processing_latency_ms",
        "e2e_latency_ms",
        "t_start",
        "t_end",
        "frame_timestamp",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill default strings
    if "model" not in df.columns:
        df["model"] = "unknown"
    else:
        df["model"] = df["model"].fillna("unknown")

    # Drop records with invalid timestamps or negative latencies
    df = df.dropna(subset=["t_start", "t_end", "processing_latency_ms"])
    df = df[df["t_end"] >= df["t_start"]]

    if df.empty:
        raise ValueError(f"CSV file '{csv_path}' has no valid latency records.")

    # Sort strictly chronologically
    df = df.sort_values(by="t_start").reset_index(drop=True)

    # Normalize to linear time (0.0 at session start)
    t0 = df["t_start"].min()
    df["t_rel_start"] = df["t_start"] - t0
    df["t_rel_end"] = df["t_end"] - t0
    df["duration_s"] = np.maximum(0.0, df["t_rel_end"] - df["t_rel_start"])

    # Define minimal bar width for display so sub-millisecond events remain visible
    total_time = max(0.1, df["t_rel_end"].max())
    min_visible_width = max(0.0002, total_time * 0.0005)
    df["duration_s_display"] = np.maximum(df["duration_s"], min_visible_width)

    # Define label for Y axis
    if group_by == "plugin":
        df["plugin_label"] = df["plugin"]
    elif group_by == "stage":
        df["plugin_label"] = df["process"] + ": " + df["plugin"] + " [" + df["stage"] + "]"
    else:  # default: process_plugin
        df["plugin_label"] = df["process"] + ": " + df["plugin"]

    # Filter warmup period if requested
    if warmup_skip_sec > 0:
        df = df[df["t_rel_start"] >= warmup_skip_sec].reset_index(drop=True)

    # Limit rows if requested
    if max_points and len(df) > max_points:
        df = df.iloc[:max_points].copy()

    return df


def plot_timeline(
    ax: plt.Axes,
    df: pd.DataFrame,
    group_col: str = "plugin_label",
    color_by: str = "stage",
    time_range: Optional[Tuple[float, float]] = None,
):
    """Plot execution timeline Gantt chart (X=linear time, Y=plugins)."""
    def sort_key(label: str) -> Tuple[int, str]:
        if label.startswith("world"):
            return (0, label)
        elif label.startswith("eye0"):
            return (1, label)
        elif label.startswith("eye1"):
            return (2, label)
        return (3, label)

    labels = sorted(df[group_col].unique(), key=sort_key, reverse=True)
    y_map = {label: i for i, label in enumerate(labels)}
    bar_height = 0.65

    # Unique colors per stage or model
    categories = sorted(df[color_by].unique())
    palette = sns.color_palette("tab10", n_colors=max(len(categories), 3))
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

    # Render spans using broken_barh for high performance
    for cat in categories:
        cat_df = df[df[color_by] == cat]
        for label in labels:
            sub = cat_df[cat_df[group_col] == label]
            if sub.empty:
                continue
            y_pos = y_map[label]
            xranges = list(zip(sub["t_rel_start"], sub["duration_s_display"]))
            ax.broken_barh(
                xranges,
                (y_pos - bar_height / 2, bar_height),
                facecolors=color_map[cat],
                edgecolor="none",
                alpha=0.85,
            )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontweight="bold", fontsize=10)
    ax.set_xlabel("Linear Time (seconds from start)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Plugins / Processes", fontsize=11, fontweight="bold")
    ax.set_title("Plugin Execution Timeline (Start to End)", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)

    if time_range:
        ax.set_xlim(time_range[0], time_range[1])
    else:
        ax.set_xlim(0, max(df["t_rel_end"].max() * 1.02, 0.1))

    legend_patches = [
        mpatches.Patch(color=color_map[cat], label=f"{cat}")
        for cat in categories
    ]
    ax.legend(
        handles=legend_patches,
        title=color_by.capitalize(),
        loc="upper right",
        framealpha=0.9,
        fontsize=9,
    )


def plot_latency_over_time(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str = "processing_latency_ms",
    time_range: Optional[Tuple[float, float]] = None,
):
    """Plot latency trajectories over linear elapsed time."""
    metric_label = "Processing Latency (ms)" if metric == "processing_latency_ms" else "E2E Latency (ms)"

    plot_df = df[df["stage"].isin(["detect", "recent_events", "gl_display"])]
    if plot_df.empty:
        plot_df = df

    sns.scatterplot(
        data=plot_df,
        x="t_rel_start",
        y=metric,
        hue="plugin_label",
        style="stage",
        alpha=0.75,
        s=30,
        ax=ax,
    )

    ax.set_xlabel("Linear Time (seconds from start)", fontsize=11, fontweight="bold")
    ax.set_ylabel(metric_label, fontsize=11, fontweight="bold")
    ax.set_title(f"{metric_label} vs. Linear Time", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    if time_range:
        ax.set_xlim(time_range[0], time_range[1])
    else:
        ax.set_xlim(0, max(df["t_rel_end"].max() * 1.02, 0.1))

    # Clamp top Y to 99.5th percentile to prevent extreme one-off warmup outliers from squishing scale
    y_max = np.percentile(plot_df[metric], 99.5) * 1.25 if len(plot_df) > 10 else plot_df[metric].max() * 1.1
    ax.set_ylim(bottom=0, top=max(y_max, 5.0))

    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=8, framealpha=0.9)


def plot_latency_breakdown(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str = "processing_latency_ms",
):
    """Plot latency breakdown by component and stage (mean ± std)."""
    metric_label = "Processing Latency (ms)" if metric == "processing_latency_ms" else "E2E Latency (ms)"

    sns.barplot(
        data=df,
        x=metric,
        y="plugin_label",
        hue="stage",
        ci="sd",
        capsize=0.1,
        errwidth=1.2,
        palette="muted",
        ax=ax,
    )

    ax.set_xlabel(f"Mean {metric_label} (± std)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Plugins / Processes", fontsize=11, fontweight="bold")
    ax.set_title("Average Latency Breakdown by Component & Stage", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    ax.legend(title="Stage", loc="lower right", fontsize=9, framealpha=0.9)


def print_terminal_summary(df: pd.DataFrame, metric: str = "processing_latency_ms"):
    """Print clean summary statistics to terminal."""
    metric_label = "Processing Latency" if metric == "processing_latency_ms" else "E2E Latency"
    print("\n" + "=" * 100)
    print(f"                     LATENCY SUMMARY REPORT ({metric_label.upper()})                     ")
    print("=" * 100)
    print(
        f"{'Process':<8} {'Plugin':<24} {'Model':<14} {'Stage':<14} "
        f"{'Count':>6} {'Mean(ms)':>9} {'Std(ms)':>8} {'P95(ms)':>8} {'Max(ms)':>8}"
    )
    print("-" * 100)

    stats = (
        df.groupby(["process", "plugin", "model", "stage"])[metric]
        .agg(
            count="count",
            mean="mean",
            std="std",
            p95=lambda s: np.percentile(s, 95) if len(s) > 0 else 0.0,
            max="max",
        )
        .reset_index()
    )

    stats = stats.sort_values(by="mean", ascending=False)
    max_mean = stats["mean"].max() if not stats.empty else 0.0

    for _, row in stats.iterrows():
        bottleneck_flag = " [!]" if row["mean"] == max_mean and row["mean"] > 10.0 else ""
        print(
            f"{str(row['process']):<8} "
            f"{str(row['plugin'])[:23]:<24} "
            f"{str(row['model'])[:13]:<14} "
            f"{str(row['stage'])[:13]:<14} "
            f"{int(row['count']):>6} "
            f"{float(row['mean']):>9.2f} "
            f"{float(row['std']):>8.2f} "
            f"{float(row['p95']):>8.2f} "
            f"{float(row['max']):>8.2f}"
            f"{bottleneck_flag}"
        )
    print("=" * 100)
    if any(stats["mean"] > 10.0):
        print("Note: Rows marked with [!] indicate primary pipeline execution bottlenecks (>10ms mean).")
    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot execution timelines and latency graphs from Pupil Labs CSV logs."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help="Path to latency CSV file. If omitted, automatically finds the latest in logged_latencies/.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to save the generated plot image (e.g., latency_timeline.png).",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "timeline", "latency", "summary"],
        default="all",
        help="Plot mode: 'timeline' (Gantt start-to-end), 'latency' (latency vs time), "
        "'summary' (bar breakdown), or 'all' (multi-panel dashboard). Default: all.",
    )
    parser.add_argument(
        "--group-by",
        choices=["process_plugin", "plugin", "stage"],
        default="process_plugin",
        help="Y-axis grouping: 'process_plugin' (default), 'plugin', or 'stage'.",
    )
    parser.add_argument(
        "--color-by",
        choices=["stage", "model", "process"],
        default="stage",
        help="Color categorization on timeline chart. Default: stage.",
    )
    parser.add_argument(
        "--metric",
        choices=["processing_latency_ms", "e2e_latency_ms"],
        default="processing_latency_ms",
        help="Latency metric to visualize. Default: processing_latency_ms.",
    )
    parser.add_argument(
        "--warmup-skip",
        type=float,
        default=1.0,
        help="Number of initial seconds to skip (ignores initial model load/CUDA warmup).",
    )
    parser.add_argument(
        "--time-range",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        default=None,
        help="Zoom in on a specific linear time interval in seconds (e.g. --time-range 5.0 15.0).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Maximum rows to load/plot from CSV.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive matplotlib window (requires graphical desktop/X11).",
    )

    args = parser.parse_args()

    # Resolve CSV file
    csv_file = args.csv_path
    if not csv_file or not os.path.exists(csv_file):
        csv_file = find_latest_latency_csv()
        if not csv_file:
            print("Error: No latency CSV file specified and none found in 'logged_latencies/'.")
            sys.exit(1)
        print(f"Using latest latency CSV: {csv_file}")
    else:
        print(f"Loading latency CSV: {csv_file}")

    # Resolve output path
    output_path = args.output
    if not output_path:
        csv_stem = os.path.splitext(os.path.basename(csv_file))[0]
        csv_dir = os.path.dirname(os.path.abspath(csv_file))
        output_path = os.path.join(csv_dir, f"{csv_stem}_plot.png")

    # Load and process data
    try:
        df = load_latency_data(
            csv_path=csv_file,
            warmup_skip_sec=args.warmup_skip,
            max_points=args.max_points,
            group_by=args.group_by,
        )
    except Exception as e:
        print(f"Error reading latency data: {e}")
        sys.exit(1)

    # Print summary statistics
    print_terminal_summary(df, metric=args.metric)

    # Style configuration
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    time_range = tuple(args.time_range) if args.time_range else None

    # Render plots based on mode
    if args.mode == "timeline":
        fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
        plot_timeline(ax, df, group_col="plugin_label", color_by=args.color_by, time_range=time_range)
        plt.tight_layout()

    elif args.mode == "latency":
        fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
        plot_latency_over_time(ax, df, metric=args.metric, time_range=time_range)
        plt.tight_layout()

    elif args.mode == "summary":
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        plot_latency_breakdown(ax, df, metric=args.metric)
        plt.tight_layout()

    else:  # mode == "all"
        fig, (ax1, ax2, ax3) = plt.subplots(
            3,
            1,
            figsize=(16, 14),
            dpi=150,
            gridspec_kw={"height_ratios": [1.4, 1.0, 1.0]},
        )
        plot_timeline(ax1, df, group_col="plugin_label", color_by=args.color_by, time_range=time_range)
        plot_latency_over_time(ax2, df, metric=args.metric, time_range=time_range)
        plot_latency_breakdown(ax3, df, metric=args.metric)
        plt.tight_layout()

    # Save output
    try:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Successfully saved plot to: {output_path}")
    except Exception as e:
        print(f"Failed to save plot image: {e}")

    if args.show:
        try:
            plt.show()
        except Exception as e:
            print(f"Could not open interactive display: {e}")


if __name__ == "__main__":
    main()
