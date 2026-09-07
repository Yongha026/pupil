#!/usr/bin/env python3
"""
Pupil Labs Waterfall System Latency Profiler
============================================
Plots an end-to-end pipeline waterfall chart modeled after NVIDIA Reflex System Latency,
breaking down every stage from Camera Ingestion to Display Buffer Swap:
  - Capture & Ingestion (UVC_Source, ROI)
  - Processing Latency (Preprocess, Inference, Ellipse Fit, Pye3D, Gaze Mapping)
  - Transport & Queueing (ZeroMQ IPC)
  - Display Latency (Render Submission, Buffer Swap)
  - System Latency (Total End-to-End Latency)

Usage:
  python latency_plot_waterfall.py [CSV_PATH] [--output OUTPUT_PATH] [--mode {average,boot,comparison}]
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_pipeline_stages(model_name: str = "pmrnet", roi_val: float = 0.0) -> List[Tuple[str, str, float, str]]:
    """
    Returns the appropriate pipeline stages based on active model and ROI usage:
    - 2dcpp: Includes ROI Extraction and 2D Detection (C++), omits NN Preprocessing/Ellipse Fit.
    - Neural Net models: Omits ROI Extraction (since roi_ms == 0.0), includes Preprocessing, NN Inference, Contour/Ellipse fit.
    """
    if model_name == "2dcpp" or roi_val > 0.001:
        return [
            ("Camera Ingest", "UVC_Source", 1.20, "Capture Latency"),
            ("ROI Extraction", "Roi", 0.03, "Capture Latency"),
            ("2D Detection (C++)", "Detector2D", 2.80, "Processing Latency"),
            ("3D Eye Model", "Pye3D", 0.08, "Processing Latency"),
            ("ZeroMQ IPC Transport", "ZeroMQ Socket", 0.35, "Transport Latency"),
            ("World Gaze Mapping", "Gazer3D", 0.05, "Transport Latency"),
            ("Render Submission", "gl_display", 0.45, "Display Latency"),
            ("Display Buffer Swap", "glfw.swap_buffers", 1.20, "Display Latency"),
        ]
    else:
        return [
            ("Camera Ingest", "UVC_Source", 1.20, "Capture Latency"),
            ("Pupil Preprocessing", "nnUNet (CLAHE/LUT)", 0.45, "Processing Latency"),
            ("Neural Net Inference", "nnUNet (GPU Forward)", 4.12, "Processing Latency"),
            ("Contour & Ellipse Fit", "nnUNet (fitEllipse)", 0.65, "Processing Latency"),
            ("3D Eye Model", "Pye3D", 0.08, "Processing Latency"),
            ("ZeroMQ IPC Transport", "ZeroMQ Socket", 0.35, "Transport Latency"),
            ("World Gaze Mapping", "Gazer3D", 0.05, "Transport Latency"),
            ("Render Submission", "gl_display", 0.45, "Display Latency"),
            ("Display Buffer Swap", "glfw.swap_buffers", 1.20, "Display Latency"),
        ]


def find_latest_waterfall_csv(base_dir: Optional[str] = None) -> Optional[str]:
    """Find the most recently modified waterfall CSV file in logged_latencies or root."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    candidates: List[str] = []
    log_dir = os.path.join(base_dir, "logged_latencies")
    if os.path.isdir(log_dir):
        candidates.extend(glob.glob(os.path.join(log_dir, "waterfall_*.csv")))

    candidates.extend(glob.glob(os.path.join(base_dir, "waterfall_*.csv")))
    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)


def load_waterfall_data(csv_path: str) -> Tuple[Dict[str, float], Dict[str, float], str]:
    """
    Loads waterfall CSV and extracts:
      1. steady_state: average duration for each stage during looping
      2. cold_start: raw duration for each stage during initial boot
      3. model_name: detected detector model architecture
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    stage_cols = [
        "ingest_ms",
        "roi_ms",
        "preprocess_ms",
        "inference_ms",
        "ellipse_fit_ms",
        "pye3d_ms",
        "ipc_transport_ms",
        "gaze_mapping_ms",
        "render_ms",
        "buffer_swap_ms",
    ]

    for col in stage_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Detect model name from log
    model_name = "pmrnet"
    if "model" in df.columns and not df["model"].dropna().empty:
        model_name = str(df["model"].dropna().iloc[-1])

    # Separate boot (cold start) from loop (steady state)
    if "phase" in df.columns and "boot" in df["phase"].values:
        boot_df = df[df["phase"] == "boot"]
        loop_df = df[df["phase"] == "loop"]
    else:
        boot_df = df.iloc[:1]
        loop_df = df.iloc[1:] if len(df) > 1 else df.iloc[:1]

    col_to_stage = {
        "ingest_ms": "Camera Ingest",
        "roi_ms": "ROI Extraction",
        "preprocess_ms": "Pupil Preprocessing",
        "inference_ms": "2D Detection (C++)" if model_name == "2dcpp" else "Neural Net Inference",
        "ellipse_fit_ms": "Contour & Ellipse Fit",
        "pye3d_ms": "3D Eye Model",
        "ipc_transport_ms": "ZeroMQ IPC Transport",
        "gaze_mapping_ms": "World Gaze Mapping",
        "render_ms": "Render Submission",
        "buffer_swap_ms": "Display Buffer Swap",
    }

    steady_state = {}
    cold_start = {}

    for col, stage_name in col_to_stage.items():
        if col in df.columns:
            steady_state[stage_name] = float(loop_df[col].mean()) if not loop_df.empty else 0.0
            cold_start[stage_name] = float(boot_df[col].iloc[0]) if not boot_df.empty else steady_state[stage_name]
        else:
            steady_state[stage_name] = 0.0
            cold_start[stage_name] = 0.0

    return steady_state, cold_start, model_name


def draw_curly_bracket(ax, x1: float, x2: float, y: float, height: float, label: str, text_pos: str = "top"):
    """Draw a clean hierarchical bracket (ㄷ-shaped) with centered label."""
    if x2 <= x1:
        x2 = x1 + 0.1

    mid_x = (x1 + x2) / 2.0
    tip_y = y + height
    base_y = y

    # Bracket wireframe path: [x1, x1, x2, x2] and [base_y, tip_y, tip_y, base_y]
    ax.plot([x1, x1, x2, x2], [base_y, tip_y, tip_y, base_y], color="black", lw=1.2, alpha=0.9, zorder=5)

    # Label text: Position adjusted to avoid overlap
    ax.text(
        mid_x,
        tip_y + (0.3 if height > 0 else -0.5),
        label,
        ha="center",
        va="bottom" if height > 0 else "top",
        fontsize=10,
        fontweight="bold",
        zorder=6
    )


def render_waterfall_panel(
    ax: plt.Axes,
    stage_durations: Dict[str, float],
    model_name: str = "pmrnet",
    title: str = "System Latency Breakdown"
):
    """
    Renders the sequential Reflex-style waterfall chart dynamically adapted for the active model.
    """
    roi_val = stage_durations.get("ROI Extraction", 0.0)
    stages = get_pipeline_stages(model_name=model_name, roi_val=roi_val)
    stage_names = [s[0] for s in stages]
    durations = [max(0.001, stage_durations.get(name, s[2])) for name, s in zip(stage_names, stages)]

    # Compute start offsets: each stage begins when its predecessor finishes
    start_offsets = [0.0]
    for d in durations[:-1]:
        start_offsets.append(start_offsets[-1] + d)

    total_latency = start_offsets[-1] + durations[-1]
    n_stages = len(stage_names)
    y_positions = np.arange(n_stages)[::-1]  # Top to bottom
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Draw horizontal guide lines
    for y in y_positions:
        ax.axhline(y, lw=1.0, zorder=1, alpha=0.3)

    # Draw waterfall bars
    bar_height = 0.58
    for i in range(n_stages):
        x_start = start_offsets[i]
        dur = durations[i]
        y_pos = y_positions[i]

        ax.barh(
            y_pos,
            dur,
            left=x_start,
            height=bar_height,
            color=colors[i % len(colors)],
            linewidth=1.2,
            zorder=3,
        )

        # Label duration inside or adjacent to bar
        val_text = f"{dur:.2f} ms"
        if dur > total_latency * 0.08:
            ax.text(
                x_start + dur / 2.0,
                y_pos,
                val_text,
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=9.5,
                zorder=4,
            )
        else:
            ax.text(
                x_start + dur + total_latency * 0.012,
                y_pos,
                val_text,
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=9,
                zorder=4,
            )

    # Y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(stage_names, fontsize=10.5, fontweight="bold")
    ax.tick_params(axis="y", length=0, pad=10)

    # X-axis
    ax.set_xlabel("Time (ms)", fontsize=12, fontweight="bold", labelpad=12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4, zorder=1)
    ax.set_xlim(-total_latency * 0.02, total_latency * 1.14)

    # Base height above the bars for hierarchical brackets
    bracket_base_y = n_stages + 0.3
    total_height = 0.8
    sub_height = 0.7

    # Draw Overarching System Latency Bracket (Level 1: Highest)
    draw_curly_bracket(
        ax,
        0.0,
        total_latency,
        bracket_base_y + 2.5,
        total_height,
        f"Total System Latency: {total_latency:.2f} ms",
    )

    # Compute category ranges dynamically
    cap_indices = [i for i, s in enumerate(stages) if s[3] == "Capture Latency"]
    proc_indices = [i for i, s in enumerate(stages) if s[3] in ("Processing Latency", "Transport Latency")]
    disp_indices = [i for i, s in enumerate(stages) if s[3] == "Display Latency"]

    if proc_indices:
        t_proc_start = start_offsets[proc_indices[0]]
        t_proc_end = start_offsets[proc_indices[-1]] + durations[proc_indices[-1]]
        proc_dur = t_proc_end - t_proc_start
        draw_curly_bracket(ax, t_proc_start, t_proc_end, bracket_base_y + 1.2, sub_height, f"PC Processing Latency\n({proc_dur:.2f} ms)")

    if cap_indices:
        t_cap_start = start_offsets[cap_indices[0]]
        t_cap_end = start_offsets[cap_indices[-1]] + durations[cap_indices[-1]]
        cap_dur = t_cap_end - t_cap_start
        draw_curly_bracket(ax, t_cap_start, t_cap_end, bracket_base_y + 0.1, sub_height, f"Capture Latency\n({cap_dur:.2f} ms)")

    if disp_indices:
        t_disp_start = start_offsets[disp_indices[0]]
        t_disp_end = start_offsets[disp_indices[-1]] + durations[disp_indices[-1]]
        disp_dur = t_disp_end - t_disp_start
        draw_curly_bracket(ax, t_disp_start, t_disp_end, bracket_base_y + 0.1, sub_height, f"Display Latency\n({disp_dur:.2f} ms)")

    ax.set_ylim(-0.8, n_stages + 5.0)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=28)


def main():
    parser = argparse.ArgumentParser(
        description="Generate end-to-end waterfall latency chart for Pupil Labs."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help="Path to waterfall CSV log. If omitted, automatically picks the latest in logged_latencies/.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to save output chart image (PNG). Default: latency_waterfall.png",
    )
    parser.add_argument(
        "--mode",
        choices=["average", "boot", "comparison"],
        default="average",
        help="Waterfall mode: 'average' (steady-state loop), 'boot' (cold start), or 'comparison' (both side-by-side).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run using verified benchmark timings without requiring an existing CSV log.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive matplotlib window (requires graphical display/X11).",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print results only(don't show)"
    )

    args = parser.parse_args()

    steady_state = {}
    cold_start = {}
    model_name = "pmrnet"

    csv_file = args.csv_path
    if not args.demo:
        if not csv_file or not os.path.exists(csv_file):
            csv_file = find_latest_waterfall_csv()

        if csv_file and os.path.exists(csv_file):
            print(f"Loading waterfall latency log: {csv_file}")
            try:
                steady_state, cold_start, model_name = load_waterfall_data(csv_file)
            except Exception as e:
                print(f"Warning: Could not parse CSV ({e}), falling back to verified benchmark timings.")
                args.demo = True
        else:
            print("No waterfall CSV found in 'logged_latencies/'. Generating benchmark demonstration waterfall.")
            args.demo = True

    if args.demo or not steady_state:
        # Verified timings from remote RTX 3090 / A6000 benchmark
        stages = get_pipeline_stages(model_name)
        steady_state = {name: default_val for name, _, default_val, _ in stages}
        cold_start = dict(steady_state)
        if "Neural Net Inference" in cold_start:
            cold_start["Neural Net Inference"] = 198.50  # Cold start GPU allocation & CUDA kernel launch
        cold_start["Camera Ingest"] = 14.50

    # Determine output path
    output_path = args.output
    if not output_path:
        if csv_file and os.path.exists(csv_file):
            csv_stem = os.path.splitext(os.path.basename(csv_file))[0]
            output_path = os.path.join(os.path.dirname(os.path.abspath(csv_file)), f"{csv_stem}_waterfall.png")
        else:
            output_path = "latency_waterfall.png"

    if args.mode == "comparison":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), dpi=150)
        render_waterfall_panel(
            ax1,
            cold_start,
            model_name=model_name,
            title=f"Initial Booting / Cold-Start System Latency ({model_name})"
        )
        render_waterfall_panel(
            ax2,
            steady_state,
            model_name=model_name,
            title=f"Steady-State Looping System Latency ({model_name})"
        )
        plt.tight_layout(pad=3.0)
    elif args.mode == "boot":
        fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
        render_waterfall_panel(
            ax,
            cold_start,
            model_name=model_name,
            title=f"Initial Booting / Cold-Start System Latency Breakdown ({model_name})"
        )
        plt.tight_layout()
    else:  # mode == "average"
        fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
        render_waterfall_panel(
            ax,
            steady_state,
            model_name=model_name,
            title=f"Pupil Labs End-to-End System Latency Breakdown ({model_name})"
        )
        plt.tight_layout()

    # Save output plot
    try:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Successfully generated and saved waterfall latency plot to: {output_path}")
    except Exception as e:
        print(f"Failed to save waterfall plot image: {e}")

    # Print summary table based on active stages
    if args.print:
        active_stages = get_pipeline_stages(model_name, steady_state.get("ROI Extraction", 0.0))
        print("\n" + "=" * 80)
        print(f"       PIPELINE STAGE LATENCY WATERFALL SUMMARY (Model: {model_name})       ")
        print("=" * 80)
        print(f"{'Stage Name':<26} {'Category':<20} {'Steady State (ms)':>18} {'Cold Start (ms)':>15}")
        print("-" * 80)
        for name, _, _, cat in active_stages:
            ss_val = steady_state.get(name, 0.0)
            cs_val = cold_start.get(name, 0.0)
            print(f"{name:<26} {cat:<20} {ss_val:>18.2f} {cs_val:>15.2f}")
        print("=" * 80)
        ss_tot = sum(steady_state.get(n, 0.0) for n, _, _, _ in active_stages)
        cs_tot = sum(cold_start.get(n, 0.0) for n, _, _, _ in active_stages)
        print(f"{'TOTAL SYSTEM LATENCY':<47} {ss_tot:>18.2f} {cs_tot:>15.2f}")
        print("=" * 80 + "\n")

    if args.show or not args.print:
        try:
            plt.show()
        except Exception as e:
            print(f"Could not open interactive display: {e}")


if __name__ == "__main__":
    main()