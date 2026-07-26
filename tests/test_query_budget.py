"""Tests for thermal and ONNX query-budget control."""

from __future__ import annotations

import pytest

from scene_runtime.controller.actions import RuntimeAction
from scene_runtime.controller.query_budget import ThermalQueryBudgetController
from scene_runtime.inference.onnx_engine import ONNXRTDETREngine


def action(budget: int | None) -> RuntimeAction:
    return RuntimeAction(
        mode="test",
        input_resolution=640,
        inference_interval=1,
        cpu_threads=4,
        query_budget=budget,
    )


def test_temperature_query_budget_uses_hysteresis() -> None:
    controller = ThermalQueryBudgetController(
        {
            "inference": {"max_query_budget": 300},
            "thermal": {
                "normal_max_c": 58,
                "warm_max_c": 66,
                "critical_c": 76,
            },
            "query_budget_control": {
                "enabled": True,
                "normal_budget": 300,
                "warm_budget": 240,
                "hot_budget": 160,
                "critical_budget": 100,
                "hysteresis_c": 5,
            },
        }
    )
    assert controller.update("normal", 50) == (300, "normal")
    assert controller.update("warm", 60) == (240, "warm")
    assert controller.update("normal", 54) == (240, "warm")
    assert controller.update("normal", 52) == (300, "normal")
    assert controller.update("critical", 78) == (100, "critical")


def test_graph_query_budget_is_clamped_and_applied() -> None:
    engine = ONNXRTDETREngine(
        dry_run=True,
        query_budget_mode="strict",
        max_query_budget=300,
    )
    engine._input_names = ["images", "orig_target_sizes", "query_budget"]
    assert engine._resolve_query_budget(action(120)) == (
        120,
        120,
        "graph_input",
        True,
    )
    assert engine._resolve_query_budget(action(999))[1] == 300


def test_strict_mode_rejects_static_onnx() -> None:
    engine = ONNXRTDETREngine(
        dry_run=True,
        query_budget_mode="strict",
    )
    engine._input_names = ["images", "orig_target_sizes"]
    with pytest.raises(RuntimeError, match="make_dynamic_query_onnx"):
        engine._resolve_query_budget(action(100))


def test_postprocess_mode_is_explicitly_not_graph_compute() -> None:
    engine = ONNXRTDETREngine(
        dry_run=True,
        query_budget_mode="postprocess",
    )
    engine._input_names = ["images", "orig_target_sizes"]
    assert engine._resolve_query_budget(action(80)) == (
        80,
        80,
        "postprocess_only",
        False,
    )
