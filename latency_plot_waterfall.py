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
  python latency_plot_waterfall.py [CSV_PATH] [--output OUTPUT_PATH] [--mode {average,boot,comparison,modelcomp}]
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


def get_pipeline_stages(model_name: str = "pmrnet", roi_val: float = 0.0) -> List[Tuple[str, str, str]]:
    """
    Returns the appropriate pipeline stages based on active model and ROI usage:
    - 2dcpp: Includes ROI Extraction and 2D Detection (C++), omits NN Preprocessing/Ellipse Fit.
    - Neural Net models: Omits ROI Extraction (since roi_ms == 0.0), includes Preprocessing, NN Inference, Contour/Ellipse fit.
    """
    if model_name == "2dcpp" or roi_val > 0.001:
        return [
            ("Camera Ingest", "UVC_Source", "Capture Latency"),
            ("ROI Extraction", "Roi", "Capture Latency"),
            ("2D Detection (C++)", "Detector2D", "Processing Latency"),
            ("3D Eye Model", "Pye3D", "Processing Latency"),
            ("ZeroMQ IPC Transport", "ZeroMQ Socket", "Transport Latency"),
            ("World Gaze Mapping", "Gazer3D", "Transport Latency"),
            ("Render Submission", "gl_display", "Display Latency"),
            ("Display Buffer Swap", "glfw.swap_buffers", "Display Latency"),
        ]
    else:
        return [
            ("Camera Ingest", "UVC_Source", "Capture Latency"),
            ("Pupil Preprocessing", "nnUNet (CLAHE/LUT)", "Processing Latency"),
            ("Neural Net Inference", "nnUNet (GPU Forward)", "Processing Latency"),
            ("Contour & Ellipse Fit", "nnUNet (fitEllipse)", "Processing Latency"),
            ("3D Eye Model", "Pye3D", "Processing Latency"),
            ("ZeroMQ IPC Transport", "ZeroMQ Socket", "Transport Latency"),
            ("World Gaze Mapping", "Gazer3D", "Transport Latency"),
            ("Render Submission", "gl_display", "Display Latency"),
            ("Display Buffer Swap", "glfw.swap_buffers", "Display Latency"),
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

    # Filter out frame_id == 0 rows if valid non-zero frame records exist
    if "frame_id" in loop_df.columns and (loop_df["frame_id"] > 0).any():
        loop_calc = loop_df[loop_df["frame_id"] > 0]
    else:
        loop_calc = loop_df

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
            series = loop_calc[col]
            pos = series[series > 0]
            steady_state[stage_name] = float(pos.mean()) if not pos.empty else 0.0

            boot_val = float(boot_df[col].iloc[0]) if not boot_df.empty else 0.0
            cold_start[stage_name] = boot_val if boot_val > 0.0 else steady_state[stage_name]
        else:
            steady_state[stage_name] = 0.0
            cold_start[stage_name] = 0.0

    return steady_state, cold_start, model_name


def load_waterfall_models_data(csv_path: str) -> Dict[str, Dict[str, float]]:
    """
    Loads waterfall CSV and extracts steady-state (phase == 'loop') average durations
    for every model present in the CSV.
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

    # Separate loop phase records
    if "phase" in df.columns and "loop" in df["phase"].values:
        loop_df = df[df["phase"] == "loop"]
    else:
        loop_df = df.iloc[1:] if len(df) > 1 else df

    if "model" not in loop_df.columns or loop_df["model"].dropna().empty:
        steady, _, single_model = load_waterfall_data(csv_path)
        return {single_model: steady}

    models_data: Dict[str, Dict[str, float]] = {}

    for raw_model, group in loop_df.groupby("model"):
        model_name = str(raw_model).strip()
        if not model_name:
            continue

        # Filter out zero frame_id rows if non-zero frames exist
        if "frame_id" in group.columns and (group["frame_id"] > 0).any():
            calc_df = group[group["frame_id"] > 0]
        else:
            calc_df = group

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

        steady_state: Dict[str, float] = {}
        for col, stage_name in col_to_stage.items():
            if col in calc_df.columns:
                series = calc_df[col]
                pos = series[series > 0]
                steady_state[stage_name] = float(pos.mean()) if not pos.empty else 0.0
            else:
                steady_state[stage_name] = 0.0

        models_data[model_name] = steady_state

    return models_data


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
    title: str = "System Latency Breakdown",
    max_x: Optional[float] = None,
):
    """
    Renders the sequential Reflex-style waterfall chart dynamically adapted for the active model.
    """
    roi_val = stage_durations.get("ROI Extraction", 0.0)
    stages = get_pipeline_stages(model_name=model_name, roi_val=roi_val)
    stage_names = [s[0] for s in stages]
    durations = [max(0.001, stage_durations.get(name, 0.0)) for name in stage_names]

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
        actual_val = stage_durations.get(stage_names[i], dur)
        val_text = f"{actual_val:.2f} ms"
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

    chart_max = max(total_latency, max_x) if max_x is not None else total_latency
    ax.set_xlim(-chart_max * 0.02, chart_max * 1.14)

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
    cap_indices = [i for i, s in enumerate(stages) if s[2] == "Capture Latency"]
    proc_indices = [i for i, s in enumerate(stages) if s[2] in ("Processing Latency", "Transport Latency")]
    disp_indices = [i for i, s in enumerate(stages) if s[2] == "Display Latency"]

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
        choices=["average", "boot", "comparison", "modelcomp"],
        default="average",
        help="Waterfall mode: 'average' (steady-state loop), 'boot' (cold start), 'comparison' (both side-by-side), or 'modelcomp' (compare all models in steady-state loop).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive matplotlib window (requires graphical display/X11).",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print results only(don't show)",
    )

    args = parser.parse_args()

    csv_file = args.csv_path
    if not csv_file or not os.path.exists(csv_file):
        csv_file = find_latest_waterfall_csv()

    if not csv_file or not os.path.exists(csv_file):
        print("No CSV file found.")
        return

    if args.mode == "modelcomp":
        print(f"Loading waterfall latency log for model comparison: {csv_file}")
        try:
            models_data = load_waterfall_models_data(csv_file)
        except Exception as e:
            print(f"Error parsing CSV file ({csv_file}): {e}")
            return

        if not models_data:
            print("No model latency data found in CSV.")
            return

        # Determine output path
        output_path = args.output
        if not output_path:
            csv_stem = os.path.splitext(os.path.basename(csv_file))[0]
            output_path = os.path.join(
                os.path.dirname(os.path.abspath(csv_file)),
                f"{csv_stem}_waterfall_modelcomp.png",
            )

        # Compute total latency for each model
        model_totals: Dict[str, float] = {}
        for m_name, m_stages in models_data.items():
            roi_val = m_stages.get("ROI Extraction", 0.0)
            stgs = get_pipeline_stages(m_name, roi_val)
            tot = sum(m_stages.get(s[0], 0.0) for s in stgs)
            model_totals[m_name] = tot

        models_list = list(models_data.keys())
        M = len(models_list)

        min_tot = min(model_totals.values()) if model_totals else 1.0
        max_tot = max(model_totals.values()) if model_totals else 1.0
        shared_max_x = max_tot if (max_tot / max(min_tot, 0.001) <= 4.0) else None

        if M == 1:
            fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
            axes = [ax]
        elif M <= 3:
            fig, axes_arr = plt.subplots(1, M, figsize=(11 * M, 9), dpi=150)
            axes = list(axes_arr)
        else:
            ncols = 2
            nrows = (M + ncols - 1) // ncols
            fig, axes_arr = plt.subplots(nrows, ncols, figsize=(11 * ncols, 8 * nrows), dpi=150)
            axes = list(axes_arr.flatten())

        for idx, m_name in enumerate(models_list):
            ax = axes[idx]
            render_waterfall_panel(
                ax,
                models_data[m_name],
                model_name=m_name,
                title=f"Steady-State Latency: {m_name.upper()} ({model_totals[m_name]:.2f} ms)",
                max_x=shared_max_x,
            )

        for idx in range(M, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout(pad=3.0)

        # Print comparative summary table if requested
        if args.print:
            STAGE_ORDER = [
                "Camera Ingest",
                "ROI Extraction",
                "Pupil Preprocessing",
                "2D Detection (C++)",
                "Neural Net Inference",
                "Contour & Ellipse Fit",
                "3D Eye Model",
                "ZeroMQ IPC Transport",
                "World Gaze Mapping",
                "Render Submission",
                "Display Buffer Swap",
            ]
            all_stage_names = []
            stage_category = {}
            for m_name in models_list:
                roi_val = models_data[m_name].get("ROI Extraction", 0.0)
                for s_name, _, cat in get_pipeline_stages(m_name, roi_val):
                    if s_name not in all_stage_names:
                        all_stage_names.append(s_name)
                        stage_category[s_name] = cat

            all_stage_names.sort(key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 999)

            col_w = 16
            header_str = f"{'Stage Name':<25} {'Category':<20}" + "".join(f"{m + ' (ms)':>{col_w}}" for m in models_list)
            total_w = len(header_str)
            print("\n" + "=" * total_w)
            print(" " * max(0, (total_w - 50) // 2) + "STEADY-STATE LOOP LATENCY MODEL COMPARISON SUMMARY")
            print("=" * total_w)
            print(header_str)
            print("-" * total_w)
            for s_name in all_stage_names:
                cat = stage_category.get(s_name, "")
                row_vals = []
                for m_name in models_list:
                    roi_val = models_data[m_name].get("ROI Extraction", 0.0)
                    m_stage_names = [s[0] for s in get_pipeline_stages(m_name, roi_val)]
                    if s_name in m_stage_names:
                        val = models_data[m_name].get(s_name, 0.0)
                        row_vals.append(f"{val:>{col_w}.2f}")
                    else:
                        row_vals.append(f"{'-':>{col_w}}")
                print(f"{s_name:<25} {cat:<20}" + "".join(row_vals))
            print("=" * total_w)
            tot_vals = "".join(f"{model_totals[m]:>{col_w}.2f}" for m in models_list)
            print(f"{'TOTAL SYSTEM LATENCY':<45}" + tot_vals)
            print("=" * total_w + "\n")

    else:
        print(f"Loading waterfall latency log: {csv_file}")
        try:
            steady_state, cold_start, model_name = load_waterfall_data(csv_file)
        except Exception as e:
            print(f"Error parsing CSV file ({csv_file}): {e}")
            return

        if not steady_state:
            print("No latency data found in CSV.")
            return

        # Determine output path
        output_path = args.output
        if not output_path:
            csv_stem = os.path.splitext(os.path.basename(csv_file))[0]
            suffix = f"_{args.mode}" if args.mode != "average" else ""
            output_path = os.path.join(
                os.path.dirname(os.path.abspath(csv_file)),
                f"{csv_stem}_waterfall{suffix}.png",
            )

        if args.mode == "comparison":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), dpi=150)
            render_waterfall_panel(
                ax1,
                cold_start,
                model_name=model_name,
                title=f"Initial Booting / Cold-Start System Latency ({model_name})",
            )
            render_waterfall_panel(
                ax2,
                steady_state,
                model_name=model_name,
                title=f"Steady-State Looping System Latency ({model_name})",
            )
            plt.tight_layout(pad=3.0)
        elif args.mode == "boot":
            fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
            render_waterfall_panel(
                ax,
                cold_start,
                model_name=model_name,
                title=f"Initial Booting / Cold-Start System Latency Breakdown ({model_name})",
            )
            plt.tight_layout()
        else:  # mode == "average"
            fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
            render_waterfall_panel(
                ax,
                steady_state,
                model_name=model_name,
                title=f"Pupil Labs End-to-End System Latency Breakdown ({model_name})",
            )
            plt.tight_layout()

        # Print summary table based on active stages
        if args.print:
            active_stages = get_pipeline_stages(model_name, steady_state.get("ROI Extraction", 0.0))
            print("\n" + "=" * 80)
            print(f"       PIPELINE STAGE LATENCY WATERFALL SUMMARY (Model: {model_name})       ")
            print("=" * 80)
            print(f"{'Stage Name':<26} {'Category':<20} {'Steady State (ms)':>18} {'Cold Start (ms)':>15}")
            print("-" * 80)
            for name, _, cat in active_stages:
                ss_val = steady_state.get(name, 0.0)
                cs_val = cold_start.get(name, 0.0)
                print(f"{name:<26} {cat:<20} {ss_val:>18.2f} {cs_val:>15.2f}")
            print("=" * 80)
            ss_tot = sum(steady_state.get(n, 0.0) for n, _, _ in active_stages)
            cs_tot = sum(cold_start.get(n, 0.0) for n, _, _ in active_stages)
            print(f"{'TOTAL SYSTEM LATENCY':<47} {ss_tot:>18.2f} {cs_tot:>15.2f}")
            print("=" * 80 + "\n")

    # Save output plot
    if not args.print:
        try:
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f"Successfully generated and saved waterfall latency plot to: {output_path}")
        except Exception as e:
            print(f"Failed to save waterfall plot image: {e}")
    else: print("No images saved")

    if args.show or not args.print:
        try:
            plt.show()
        except Exception as e:
            print(f"Could not open interactive display: {e}")


if __name__ == "__main__":
    main()