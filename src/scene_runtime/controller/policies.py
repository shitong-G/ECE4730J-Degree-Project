"""Strategy-specific policy wrappers."""

from __future__ import annotations

from typing import Any

from scene_runtime.controller.actions import RuntimeAction


def apply_fixed_policy(config: dict[str, Any]) -> RuntimeAction | None:
    """
    Return a fixed RuntimeAction when strategy sets explicit fixed_* overrides.

    Returns None when adaptive rules should run (thermal/scene/co-adaptive).
    """
    policy = config.get("policy", {})
    has_fixed = any(
        policy.get(k) is not None
        for k in (
            "fixed_inference_interval",
            "fixed_input_resolution",
            "fixed_cpu_threads",
            "fixed_cpu_affinity",
            "fixed_governor",
        )
    )
    if not has_fixed:
        return None

    runtime = config.get("runtime", {})
    return RuntimeAction(
        mode="fixed",
        input_resolution=int(
            policy.get("fixed_input_resolution")
            or runtime.get("default_input_resolution", 640)
        ),
        inference_interval=int(
            policy.get("fixed_inference_interval")
            or runtime.get("default_inference_interval", 1)
        ),
        cpu_threads=int(
            policy.get("fixed_cpu_threads")
            or runtime.get("default_cpu_threads", 4)
        ),
        cpu_affinity=policy.get("fixed_cpu_affinity"),
        governor=policy.get("fixed_governor"),
        decoder_layers=None,
        query_budget=None,
    )
