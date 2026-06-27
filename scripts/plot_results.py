#!/usr/bin/env python3
"""Plot experiment log CSV metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def elapsed_minutes(df, pd):
    """Return elapsed experiment time in minutes, falling back to frame_id."""
    if "timestamp" in df.columns:
        timestamps = pd.to_numeric(df["timestamp"], errors="coerce")
        if timestamps.notna().any():
            first = timestamps[timestamps.notna()].iloc[0]
            return (timestamps - first) / 60.0, "elapsed time (min)"
    return pd.to_numeric(df["frame_id"], errors="coerce"), "frame_id"


def full_inference_rows(df):
    """Return rows where a real inference was executed."""
    if "did_infer" in df.columns:
        mask = df["did_infer"].astype(str).str.lower().isin({"true", "1"})
        return df[mask].copy()
    if "latency_ms" in df.columns:
        return df[pd_to_numeric(df["latency_ms"]) > 0].copy()
    return df.copy()


def pd_to_numeric(series):
    import pandas as pd

    return pd.to_numeric(series, errors="coerce")


def smooth(series, window: int):
    values = pd_to_numeric(series)
    if window <= 1:
        return values
    return values.rolling(window=window, min_periods=1).mean()


def effective_inference_fps(df):
    if "effective_inference_fps" in df.columns:
        return pd_to_numeric(df["effective_inference_fps"])
    fps_col = "loop_fps" if "loop_fps" in df.columns else "fps"
    if fps_col not in df.columns or "inference_interval" not in df.columns:
        return None
    interval = pd_to_numeric(df["inference_interval"]).clip(lower=1)
    return pd_to_numeric(df[fps_col]) / interval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot scene-runtime experiment logs")
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV log from experiments/logs/",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "results",
    )
    p.add_argument(
        "--max-frame-id",
        type=int,
        default=None,
        help="Only plot rows with frame_id < this value (e.g. 260 keeps frames 0-259)",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Rolling window in rows for smoother trend lines; use 1 for raw values",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}")
        sys.exit(1)

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("Install plotting deps: pip install matplotlib pandas")
        sys.exit(1)

    df = pd.read_csv(args.input)
    if "loop_fps" not in df.columns and "fps" in df.columns:
        df["loop_fps"] = df["fps"]
    if args.max_frame_id is not None:
        df = df[df["frame_id"] < args.max_frame_id].copy()
        if df.empty:
            print(f"No rows with frame_id < {args.max_frame_id}")
            sys.exit(1)

    plot_df = full_inference_rows(df)
    if plot_df.empty:
        print("No full inference rows found. Expected did_infer=True or latency_ms > 0.")
        sys.exit(1)

    x, xlabel = elapsed_minutes(plot_df, pd)
    x_all, xlabel_all = elapsed_minutes(df, pd)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    if args.max_frame_id is not None:
        stem = f"{stem}_frames0-{args.max_frame_id - 1}"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Experiment: {stem}")

    if "latency_ms" in df.columns:
        axes[0, 0].plot(
            x,
            smooth(plot_df["latency_ms"], args.smooth_window),
            linewidth=0.9,
            label="full inference latency",
        )
        axes[0, 0].set_title("Full inference latency (ms)")
        axes[0, 0].set_xlabel(xlabel)
        axes[0, 0].legend(loc="best", fontsize=8)

    if "temp_c" in df.columns and df["temp_c"].notna().any():
        axes[0, 1].plot(
            x_all,
            smooth(df["temp_c"], args.smooth_window),
            color="tomato",
            linewidth=0.8,
            label="temperature",
        )
        axes[0, 1].legend(loc="best", fontsize=8)
    else:
        axes[0, 1].text(0.5, 0.5, "No temperature data", ha="center", va="center")
    axes[0, 1].set_title("CPU temperature (C)")
    axes[0, 1].set_xlabel(xlabel_all)

    if "workload" in df.columns:
        wl_map = {"light": 0, "medium": 1, "heavy": 2}
        axes[1, 0].plot(
            x_all,
            df["workload"].map(wl_map),
            drawstyle="steps-post",
            linewidth=0.8,
        )
        axes[1, 0].set_yticks([0, 1, 2])
        axes[1, 0].set_yticklabels(["light", "medium", "heavy"])
    axes[1, 0].set_title("Scene workload")
    axes[1, 0].set_xlabel(xlabel_all)

    eff_fps = effective_inference_fps(df)
    if eff_fps is not None:
        axes[1, 1].plot(
            x_all,
            smooth(eff_fps, args.smooth_window),
            color="purple",
            linewidth=0.8,
            label="effective inference FPS",
        )
    if "actual_inference_fps" in df.columns:
        axes[1, 1].plot(
            x_all,
            smooth(df["actual_inference_fps"], args.smooth_window),
            color="darkgreen",
            linewidth=0.8,
            label="actual inference FPS",
        )
    axes[1, 1].set_title("Effective inference FPS")
    axes[1, 1].set_xlabel(xlabel_all)
    axes[1, 1].legend(loc="best", fontsize=8)

    out = args.output_dir / f"{stem}_summary.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved plot: {out}")

    freq_col = None
    freq_label = "CPU frequency (MHz)"
    if "arm_clock_mhz" in plot_df.columns and plot_df["arm_clock_mhz"].notna().any():
        freq_col = "arm_clock_mhz"
        freq_label = "ARM clock (MHz, actual)"
    elif "freq_mhz_avg" in plot_df.columns and plot_df["freq_mhz_avg"].notna().any():
        freq_col = "freq_mhz_avg"
        freq_label = "CPU frequency sysfs (MHz, governor)"

    if freq_col is not None:
        fig_freq, ax_freq = plt.subplots(figsize=(12, 4))
        ax_freq.plot(
            x,
            smooth(plot_df[freq_col], args.smooth_window),
            color="steelblue",
            linewidth=0.8,
            label=freq_col,
        )
        if (
            freq_col == "arm_clock_mhz"
            and "freq_mhz_avg" in plot_df.columns
            and plot_df["freq_mhz_avg"].notna().any()
        ):
            ax_freq.plot(
                x,
                smooth(plot_df["freq_mhz_avg"], args.smooth_window),
                color="lightgray",
                linewidth=0.8,
                alpha=0.8,
                label="sysfs scaling_cur_freq",
            )
            ax_freq.legend(loc="best")
        ax_freq.set_title(freq_label)
        ax_freq.set_xlabel(xlabel)
        ax_freq.set_ylabel("MHz")
        ax_freq.grid(True, alpha=0.3)
        freq_out = args.output_dir / f"{stem}_cpu_freq.png"
        fig_freq.tight_layout()
        fig_freq.savefig(freq_out, dpi=120)
        plt.close(fig_freq)
        print(f"Saved plot: {freq_out}")

    if (
        "thermal_state" in df.columns
        or "control_thermal_state" in df.columns
        or "action_mode" in df.columns
    ):
        fig_thermal, axes_thermal = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        if "temp_c" in df.columns and df["temp_c"].notna().any():
            axes_thermal[0].plot(
                x_all,
                smooth(df["temp_c"], args.smooth_window),
                color="tomato",
                linewidth=0.8,
                label="temperature",
            )
            axes_thermal[0].set_ylabel("C")
            axes_thermal[0].legend(loc="best", fontsize=8)

        state_map = {"unknown": -1, "normal": 0, "warm": 1, "hot": 2, "critical": 3}
        for col, color in (
            ("raw_thermal_state", "lightgray"),
            ("control_thermal_state", "crimson"),
            ("thermal_state", "darkred"),
        ):
            if col in df.columns:
                axes_thermal[1].plot(
                    x_all,
                    df[col].fillna("unknown").map(state_map),
                    drawstyle="steps-post",
                    linewidth=0.9,
                    label=col,
                    color=color,
                )
        axes_thermal[1].set_yticks([-1, 0, 1, 2, 3])
        axes_thermal[1].set_yticklabels(["unknown", "normal", "warm", "hot", "critical"])
        axes_thermal[1].legend(loc="best", fontsize=8)

        if "action_mode" in df.columns:
            actions = sorted(str(v) for v in df["action_mode"].dropna().unique())
            action_map = {name: idx for idx, name in enumerate(actions)}
            axes_thermal[2].plot(
                x_all,
                df["action_mode"].map(action_map),
                drawstyle="steps-post",
                linewidth=0.8,
                label="action_mode",
            )
            if "did_infer" in df.columns:
                infer = df["did_infer"].astype(str).str.lower().isin({"true", "1"})
                axes_thermal[2].scatter(
                    x_all[infer],
                    [-0.5] * int(infer.sum()),
                    s=8,
                    color="black",
                    alpha=0.5,
                    label="did_infer",
                )
            axes_thermal[2].set_yticks(list(action_map.values()))
            axes_thermal[2].set_yticklabels(list(action_map.keys()), fontsize=7)
            axes_thermal[2].legend(loc="best", fontsize=8)
        axes_thermal[2].set_xlabel(xlabel_all)
        fig_thermal.suptitle(f"Thermal control timeline: {stem}")
        thermal_out = args.output_dir / f"{stem}_thermal_control.png"
        fig_thermal.tight_layout()
        fig_thermal.savefig(thermal_out, dpi=120)
        plt.close(fig_thermal)
        print(f"Saved plot: {thermal_out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
