"""Machine-inference facade.

The API layer calls this module rather than the ML packages directly, so model
loading is centralised and an untrained model produces a single, consistent,
explicit error instead of a crash inside a route handler.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.core.logging_config import get_logger
from app.ml.anomaly_detection.inference import AnomalyInference
from app.ml.failure_prediction.inference import FailureInference, ModelNotTrained
from app.ml.rul_prediction.inference import RULInference

logger = get_logger(__name__)

_lock = threading.Lock()
_failure: FailureInference | None = None
_anomaly: AnomalyInference | None = None
_rul: dict[str, RULInference] = {}


def failure_model() -> FailureInference:
    global _failure
    with _lock:
        if _failure is None:
            _failure = FailureInference()
        return _failure


def anomaly_model() -> AnomalyInference:
    global _anomaly
    with _lock:
        if _anomaly is None:
            _anomaly = AnomalyInference()
        return _anomaly


def rul_model(family: str | None = None) -> RULInference:
    key = family or "default"
    with _lock:
        if key not in _rul:
            _rul[key] = RULInference(family=family)
        return _rul[key]


def invalidate() -> None:
    """Drop cached models. Used after training and by tests."""
    global _failure, _anomaly
    from app.ml import registry

    registry.clear_cache()
    with _lock:
        _failure, _anomaly, _rul = None, None, {}


def status() -> dict[str, Any]:
    """Which models are actually loaded, from which dataset, at which metrics."""
    from app.ml import registry

    out: dict[str, Any] = {
        "failure_prediction": _describe(registry.TASK_FAILURE, failure_model()),
        "anomaly_detection": _describe(registry.TASK_ANOMALY, anomaly_model()),
        "rul_prediction": _describe(registry.TASK_RUL, rul_model()),
    }
    out["any_trained"] = any(v["loaded"] for v in out.values() if isinstance(v, dict))
    return out


def _describe(task: str, inference: Any) -> dict[str, Any]:
    artifact = getattr(inference, "artifact", None)
    if artifact is None:
        return {
            "task": task,
            "loaded": False,
            "version": None,
            "dataset": None,
            "metrics": {},
            "detail": f"No trained {task} model found. Train it before serving predictions.",
        }
    return {
        "task": task,
        "loaded": True,
        "version": artifact.version_id,
        "family": artifact.preprocessing.get("family") or artifact.name,
        "dataset": artifact.dataset,
        "dataset_sha256": artifact.dataset_sha256,
        "training_samples": artifact.training_samples,
        "trained_at": artifact.trained_at,
        "metrics": artifact.metrics,
        "feature_count": len(artifact.feature_names),
        "detail": "Loaded",
    }


def vision_status() -> dict[str, Any]:
    from app.ml.computer_vision.inference import VisionInference

    try:
        inference = VisionInference()
    except Exception as exc:  # noqa: BLE001
        logger.error("Vision model inspection failed: %s", exc)
        return {"task": "visual_inspection", "loaded": False, "trained_categories": [],
                "detail": f"Vision model could not be loaded: {exc}"}
    manifest = inference.manifest or {}
    return {
        "task": "visual_inspection",
        "loaded": inference.ready,
        "trained_categories": inference.trained_categories,
        "active_category": inference.category,
        "version": manifest.get("version"),
        "dataset": manifest.get("dataset"),
        "metrics": manifest.get("metrics"),
        "detail": (
            f"Memory bank trained for {inference.trained_categories}"
            if inference.ready else
            "No PatchCore memory bank trained. Install MVTec AD and run "
            "python backend/app/ml/computer_vision/train.py --all"
        ),
    }


def health() -> dict[str, Any]:
    models = status()
    vision = vision_status()
    return {
        "models": models,
        "vision": vision,
        "llm": {
            "enabled": settings.llm_enabled,
            "active": settings.llm_active,
            "model": settings.openai_model if settings.llm_active else None,
            "detail": (
                "LLM reasoning active." if settings.llm_active else
                "No LLM configured. Every agent runs its deterministic rule-based engine "
                "and labels its output accordingly."
            ),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


__all__ = [
    "ModelNotTrained",
    "anomaly_model",
    "failure_model",
    "health",
    "invalidate",
    "rul_model",
    "status",
    "vision_status",
]
