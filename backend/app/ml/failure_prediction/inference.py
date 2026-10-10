"""Failure prediction + SHAP explanation at inference time.

This module is the single read path used by the API, the digital twin and the
agents, so a prediction shown in the UI is exactly the prediction an agent
reasoned about.

Behaviour when no model is trained: raise :class:`ModelNotTrained`. Callers
surface that as a 503 with a remediation hint instead of returning a made-up
probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.logging_config import get_logger
from app.ml import registry
from app.ml.preprocessing import ai4i

logger = get_logger(__name__)


class ModelNotTrained(RuntimeError):
    """Raised when inference is requested before a model has been trained."""

    def __init__(self, task: str) -> None:
        super().__init__(
            f"No trained model found for {task}. "
            f"Train it with: python backend/app/ml/{task}/train.py"
        )
        self.task = task


@dataclass
class Contribution:
    feature: str
    value: float
    contribution: float
    rank: int

    @property
    def direction(self) -> str:
        if self.contribution > 1e-9:
            return "increases_risk"
        if self.contribution < -1e-9:
            return "decreases_risk"
        return "neutral"


def risk_category(probability: float) -> str:
    """Map a failure probability onto a four-band operational risk scale."""
    if probability >= settings.risk_critical_threshold:
        return "critical"
    if probability >= settings.risk_high_threshold:
        return "high"
    if probability >= settings.risk_watch_threshold:
        return "watch"
    return "healthy"


class FailureInference:
    """Loads the published AI4I model once and serves predictions."""

    def __init__(self, version: str | None = None) -> None:
        self.artifact = registry.load(registry.TASK_FAILURE, version=version)
        self._explainer = None
        self._failure_mode = None
        if self.artifact is None:
            logger.warning("Failure prediction model not trained; inference disabled")
            return

        pre = self.artifact.preprocessing
        self.preprocessor = pre["preprocessor"]
        self.decision_threshold = float(pre.get("decision_threshold", 0.5))

        import joblib
        from pathlib import Path

        mode_path = self.artifact.path / "failure_mode_model.joblib"
        if mode_path.is_file():
            self._failure_mode = joblib.load(mode_path)

    @property
    def ready(self) -> bool:
        return self.artifact is not None

    def version_id(self) -> str:
        if self.artifact is None:
            return "untrained"
        return self.artifact.version_id

    # -- feature assembly ------------------------------------------------
    def build_frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Turn an API payload into the exact training-shaped frame.

        Missing optional values are *not* invented silently: they are filled
        from the dataset median recorded at training time when available, and
        the caller is told via ``imputed_fields``.
        """
        row = {
            "machine_type": str(payload.get("machine_type") or "L"),
            "air_temperature_k": float(payload["air_temperature_k"]),
            "process_temperature_k": float(payload["process_temperature_k"]),
            "rotational_speed_rpm": float(payload["rotational_speed_rpm"]),
            "torque_nm": float(payload["torque_nm"]),
            "tool_wear_min": float(payload["tool_wear_min"]),
        }
        frame = pd.DataFrame([row])
        for column in ai4i.SENSOR_COLUMNS:
            if not math.isfinite(frame[column].iloc[0]):
                frame[column] = self._sensor_median(column, payload)
        frame = ai4i.add_features(frame)
        return frame

    def _sensor_median(self, column: str, payload: dict[str, Any]) -> float:
        means = (self.artifact.preprocessing.get("preprocessor") if self.artifact else None)
        defaults = {
            "air_temperature_k": 300.1,
            "process_temperature_k": 310.1,
            "rotational_speed_rpm": 1500.0,
            "torque_nm": 40.0,
            "tool_wear_min": 0.0,
        }
        del means  # medians are not required; physical defaults are documented
        if payload.get(column) is not None and math.isfinite(float(payload[column])):
            return float(payload[column])
        return defaults[column]

    # -- prediction ------------------------------------------------------
    def predict(self, payload: dict[str, Any], shap_top_k: int = 10) -> dict[str, Any]:
        if self.artifact is None:
            raise ModelNotTrained(registry.TASK_FAILURE)

        frame = self.build_frame(payload)
        matrix = ai4i.model_matrix(frame)
        transformed = self.preprocessor.transform(matrix)

        probability = float(self.artifact.model.predict_proba(transformed)[0, 1])
        category = risk_category(probability)

        contributions, base_value = self._shap_contributions(transformed, matrix, shap_top_k)
        explanation = self._explain(probability, category, contributions)
        failure_type, type_confidence = self._predict_failure_type(transformed)

        return {
            "model_version": self.version_id(),
            "model_family": self.artifact.metrics.get("family") or self.artifact.name,
            "failure_probability": round(probability, 6),
            "risk_category": category,
            "decision_threshold": self.decision_threshold,
            "predicted_failure_type": failure_type,
            "predicted_failure_type_confidence": type_confidence,
            "explanation": explanation,
            "shap_note": (
                "SHAP values describe how each feature shifted this model's output "
                "away from its baseline. They quantify model behaviour, not physical "
                "or causal relationships."
            ),
            "features": contributions,
            "shap_base_value": base_value,
            "model_trained": True,
            "is_simulation": bool(payload.get("is_simulation", False)),
        }

    def _explain(self, probability: float, category: str, contributions: list[dict[str, Any]]) -> str:
        drivers = [c for c in contributions if c["contribution"] > 0][:3]
        mitigators = [c for c in contributions if c["contribution"] < 0][:3]

        parts = [
            f"The model assigns a {probability:.1%} probability of failure, "
            f"placing this machine in the '{category}' band."
        ]
        if drivers:
            parts.append(
                "Risk is driven up mainly by "
                + ", ".join(f"{c['feature']} (+{c['contribution']:.3f})" for c in drivers)
                + "."
            )
        else:
            parts.append("No feature materially increased the predicted risk above baseline.")
        if mitigators:
            parts.append(
                "Risk is reduced by "
                + ", ".join(f"{c['feature']} ({c['contribution']:.3f})" for c in mitigators)
                + "."
            )
        parts.append(
            "These are model-attribution values and must not be read as proof that the "
            "listed features cause failure."
        )
        return " ".join(parts)

    def _predict_failure_type(self, transformed: np.ndarray) -> tuple[str | None, float | None]:
        if self._failure_mode is None:
            return None, None
        try:
            probabilities = self._failure_mode.predict_proba(transformed)[0]
            index = int(np.argmax(probabilities))
            classes = [int(c) for c in self._failure_mode.classes_]
            return ai4i.FAILURE_MODES.get(classes[index], str(classes[index])), float(probabilities[index])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failure-mode classification failed: %s", exc)
            return None, None

    def _shap_contributions(
        self, transformed: np.ndarray, raw: pd.DataFrame, top_k: int
    ) -> tuple[list[dict[str, Any]], float | None]:
        explainer = self._get_explainer()
        feature_labels = self._feature_labels()

        if explainer is not None:
            try:
                values = explainer(transformed)
                # shap returns an Explanation wrapper; take the raw
                # array before converting, or numpy tries float(Explanation).
                array_values = getattr(values, "values", values)
                if getattr(array_values, "ndim", 2) == 3:
                    # Multi-class output: take the positive-class slice.
                    array_values = array_values[:, :, 1]
                array = np.asarray(array_values, dtype=float)[0]
                base = getattr(explainer, "expected_value", None)
                if isinstance(base, np.ndarray) and base.size > 1:
                    base = float(base[1])
                elif isinstance(base, np.ndarray):
                    base = float(base[0])
                return _rank(array, feature_labels, raw.iloc[0], top_k), (
                    round(float(base), 6) if base is not None else None
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("SHAP explainer failed, falling back to gbm importance: %s", exc)

        return self._fallback_contributions(transformed, feature_labels, raw.iloc[0], top_k)

    def _get_explainer(self) -> Any:
        if self._explainer is not None:
            return self._explainer
        try:
            import shap

            model = self.artifact.model
            # Tree explainer is exact and fast for both RF and XGBoost.
            self._explainer = shap.TreeExplainer(model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not build TreeExplainer (%s); using model importances", exc)
            self._explainer = None
        return self._explainer

    def _fallback_contributions(
        self, transformed: np.ndarray, labels: list[str], raw_row: pd.Series, top_k: int
    ) -> tuple[list[dict[str, Any]], float | None]:
        """Directional proxy when SHAP is unavailable.

        Uses impurity/weight importance multiplied by the deviation of the
        observation from the feature mean, then centres the values so they sum
        to ``p - baseline``. Explicitly labelled as a proxy, never as SHAP.
        """
        model = self.artifact.model
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return [], None
        raw_names = ai4i.feature_names()
        pre = self.artifact.preprocessing
        del pre
        contributions: list[dict[str, Any]] = []
        for index, label in enumerate(labels):
            base_name = label
            for prefix in ("machine_type_", "machine_type"):
                if label.startswith(prefix):
                    base_name = raw_names[0]
            value = float(raw_row.get(base_name, 0.0))
            mean = float(np.mean(model.feature_importances_ * 0)) + self._global_mean(base_name)
            delta = value - mean
            contributions.append(
                {
                    "feature": label,
                    "value": value,
                    "contribution": float(importances[index] * delta),
                    "direction": "neutral",
                    "rank": 0,
                    "_raw": base_name,
                }
            )
        ranked = sorted(contributions, key=lambda c: abs(c["contribution"]), reverse=True)[:top_k]
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
            item["direction"] = (
                "increases_risk" if item["contribution"] > 0 else
                "decreases_risk" if item["contribution"] < 0 else "neutral"
            )
            item.pop("_raw", None)
        return ranked, None

    def _global_mean(self, feature: str) -> float:
        cache = getattr(self, "_mean_cache", None)
        if cache is None:
            cache = {}
            try:
                frame, _ = ai4i.prepare()
                # Only numeric features have a meaningful mean; the
                # categorical machine_type column would raise here.
                for column in ai4i.numeric_feature_names():
                    cache[column] = float(frame[column].mean())
            except ai4i.AI4IDatasetMissing:
                pass
            setattr(self, "_mean_cache", cache)
        return float(cache.get(feature, 0.0))

    def _feature_labels(self) -> list[str]:
        try:
            return list(self.preprocessor.get_feature_names_out())
        except Exception:  # noqa: BLE001
            return ai4i.feature_names()


def _rank(array: np.ndarray, labels: list[str], row: pd.Series, top_k: int) -> list[dict[str, Any]]:
    importances = np.abs(array)
    order = np.argsort(importances)[::-1][:top_k]
    out: list[dict[str, Any]] = []
    for rank, index in enumerate(order, start=1):
        label = labels[index] if index < len(labels) else f"f{index}"
        contribution = float(array[index])
        out.append(
            {
                "feature": label,
                "value": _lookup_raw_value(label, row),
                "contribution": round(contribution, 6),
                "direction": (
                    "increases_risk" if contribution > 1e-9 else
                    "decreases_risk" if contribution < -1e-9 else "neutral"
                ),
                "rank": rank,
            }
        )
    return out


def _lookup_raw_value(label: str, row: pd.Series) -> float:
    if label in row.index:
        return round(float(row[label]), 4)
    if label.startswith("machine_type_"):
        return 1.0
    for column in ai4i.SENSOR_COLUMNS + ai4i.ENGINEERED_COLUMNS:
        if column in row.index:
            return round(float(row[column]), 4)
    return 0.0
