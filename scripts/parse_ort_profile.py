#!/usr/bin/env python3
"""Parse ONNX Runtime profiling JSON into CSV summaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def load_events(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("ORT profile JSON should be a list of trace events.")

    return data


def extract_node_events(events: list[dict]) -> pd.DataFrame:
    rows = []

    for e in events:
        if e.get("cat") != "Node":
            continue

        args = e.get("args", {}) or {}

        name = str(e.get("name", ""))
        op_name = str(args.get("op_name", ""))
        provider = str(args.get("provider", ""))
        dur_us = float(e.get("dur", 0.0))
        ts_us = float(e.get("ts", 0.0))
        tid = e.get("tid", "")

        rows.append(
            {
                "name": name,
                "op_name": op_name,
                "provider": provider,
                "dur_us": dur_us,
                "dur_ms": dur_us / 1000.0,
                "ts_us": ts_us,
                "tid": tid,
            }
        )

    if not rows:
        raise ValueError(
            "No Node events found. The profile may not contain operator-level events."
        )

    return pd.DataFrame(rows)


def normalize_node_name(name: str) -> str:
    """Remove common ORT suffixes to make grouping cleaner."""
    name = re.sub(r"_kernel_time$", "", name)
    name = re.sub(r"_fence_before$", "", name)
    name = re.sub(r"_fence_after$", "", name)
    return name


def infer_module(name: str, op_name: str) -> str:
    """Best-effort RT-DETR module grouping from node names.

    This depends on whether ONNX export preserved meaningful node names.
    Adjust patterns after inspecting your actual CSV.
    """
    text = f"{name} {op_name}".lower()

    if "backbone" in text or "resnet" in text or "hgnet" in text:
        return "backbone"

    if "encoder" in text:
        return "encoder"

    if "decoder" in text:
        return "decoder"

    if "head" in text or "class" in text or "bbox" in text or "box" in text:
        return "detection_head"

    if "nms" in text or "nonmax" in text:
        return "nms"

    return "unknown"


def save_summaries(df: pd.DataFrame, out_prefix: Path) -> None:
    node_csv = out_prefix.with_suffix(".node_events.csv")
    op_csv = out_prefix.with_suffix(".by_op.csv")
    module_csv = out_prefix.with_suffix(".by_module.csv")
    top_csv = out_prefix.with_suffix(".top_nodes.csv")

    df.to_csv(node_csv, index=False)

    by_op = (
        df.groupby("op_name", dropna=False)
        .agg(
            total_ms=("dur_ms", "sum"),
            mean_ms=("dur_ms", "mean"),
            count=("dur_ms", "count"),
        )
        .reset_index()
        .sort_values("total_ms", ascending=False)
    )
    by_op["percent"] = by_op["total_ms"] / by_op["total_ms"].sum() * 100.0
    by_op.to_csv(op_csv, index=False)

    by_module = (
        df.groupby("module", dropna=False)
        .agg(
            total_ms=("dur_ms", "sum"),
            mean_ms=("dur_ms", "mean"),
            count=("dur_ms", "count"),
        )
        .reset_index()
        .sort_values("total_ms", ascending=False)
    )
    by_module["percent"] = by_module["total_ms"] / by_module["total_ms"].sum() * 100.0
    by_module.to_csv(module_csv, index=False)

    top_nodes = (
        df.groupby(["clean_name", "op_name", "module"], dropna=False)
        .agg(
            total_ms=("dur_ms", "sum"),
            mean_ms=("dur_ms", "mean"),
            count=("dur_ms", "count"),
        )
        .reset_index()
        .sort_values("total_ms", ascending=False)
        .head(50)
    )
    top_nodes["percent"] = top_nodes["total_ms"] / top_nodes["total_ms"].sum() * 100.0
    top_nodes.to_csv(top_csv, index=False)

    print(f"Saved node events: {node_csv}")
    print(f"Saved by-op summary: {op_csv}")
    print(f"Saved by-module summary: {module_csv}")
    print(f"Saved top-node summary: {top_csv}")

    print("\nTop operator types:")
    print(by_op.head(20).to_string(index=False))

    print("\nModule summary:")
    print(by_module.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=Path, help="ONNX Runtime profile JSON")
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="Output prefix. Default: same path without .json",
    )
    args = parser.parse_args()

    events = load_events(args.json)
    df = extract_node_events(events)

    df["clean_name"] = df["name"].map(normalize_node_name)
    df["module"] = [
        infer_module(name, op)
        for name, op in zip(df["clean_name"], df["op_name"])
    ]

    out_prefix = args.out_prefix or args.json.with_suffix("")
    save_summaries(df, out_prefix)


if __name__ == "__main__":
    main()
