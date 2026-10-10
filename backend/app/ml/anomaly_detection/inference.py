"""Anomaly detection inference over live sensor readings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.logging_config import get_logger
from app.ml import registry
from app.ml.preprocessing import ai4i

logger = get_logger(__name__)

SEVERITY_BANDS: tuple[tuple[float, str], ...] = (
    (0.90, "critical"),
    (0.80, "high"),
    (0.70, "warning"),
    (0.00, "low"),
)


class AnomalyInference:
    """Isolation Forest scorer with a persisted calibration envelope."""

    def __init__(self, version: str | None = None) -> None:
        self.artifact = registry.load(registry.TASK_ANOMALY, version=version)
        if self.artifact is None:
            logger.warning("Anomaly model not trained; inference disabled")
            return
        pre = self.artifact.preprocessing
        self.feature_names: list[str] = list(pre["feature_names"])
        self.feature_mean: dict[str, float] = dict(pre["feature_mean"])
        self.feature_std: dict[str, float] = dict(pre["feature_std"])
        self.feature_min: dict[str, float] = dict(pre["feature_min"])
        self.feature_max: dict[str, float] = dict(pre["feature_max"])
        self.reference = dict(pre["score_reference"])
        self.default_threshold = float(pre.get("threshold", settings.anomaly_threshold))

    @property
    def ready(self) -> bool:
        return self.artifact is not None

    def version_id(self) -> str:
        return self.artifact.version_id if self.artifact else "untrained"

    def _frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        row = {
            "machine_type": str(payload.get("machine_type") or "L"),
            "air_temperature_k": float(payload["air_temperature_k"]),
            "process_temperature_k": float(payload["process_temperature_k"]),
            "rotational_speed_rpm": float(payload["rotational_speed_rpm"]),
            "torque_nm": float(payload["torque_nm"]),
            "tool_wear_min": float(payload["tool_wear_min"]),
        }
        return ai4i.add_features(pd.DataFrame([row]))

    def _raw_score(self, matrix: np.ndarray) -> np.ndarray:
        # decision_function: positive = normal. Negate so higher = more anomalous.
        return -self.artifact.model.decision_function(matrix)

    def _normalise(self, raw: np.ndarray) -> np.ndarray:
        low = float(self.reference.get("p01", raw.min()))
        high = float(self.reference.get("p99", raw.max()))
        if high - low < 1e-9:
            high = low + 1e-9
        return np.clip((np.asarray(raw, dtype=float) - low) / (high - low), 0.0, 1.0)

    def predict(self, payload: dict[str, Any], threshold: float | None = None) -> dict[str, Any]:
        if self.artifact is None:
            from app.ml.failure_prediction.inference import ModelNotTrained

            raise ModelNotTrained(registry.TASK_ANOMALY)

        frame = self._frame(payload)
        matrix = ai4i.model_matrix(frame, include_categorical=False)
        ordered = matrix[self.feature_names]

        raw = self._raw_score(ordered.to_numpy(dtype=np.float32))
        score = float(self._normalise(raw)[0])
        active_threshold = float(threshold if threshold is not None else self.default_threshold)
        is_anomaly = score >= active_threshold

        deviations: list[dict[str, Any]] = []
        for column in ai4i.SENSOR_COLUMNS:
            value = float(frame[column].iloc[0])
            mean = float(self.feature_mean.get(column, value))
            std = float(self.feature_std.get(column, 1.0)) or 1.0
            z = (value - mean) / std
            low = float(self.feature_min.get(column, value))
            high = float(self.feature_max.get(column, value))
            out_of_range = value < low or value > high
            deviations.append(
                {
                    "feature": column,
                    "value": round(value, 4),
                    "contribution": round(float(np.clip(abs(z) / 6.0, 0.0, 1.0)), 6),
                    "direction": "increases_risk" if abs(z) > 0.5 else "neutral",
                    "rank": 0,
                    "z_score": round(float(z), 3),
                    "expected_mean": round(mean, 4),
                    "expected_range": [round(low, 4), round(high, 4)],
                    "out_of_training_range": bool(out_of_range),
                }
            )
        deviations.sort(key=lambda d: abs(d["z_score"]), reverse=True)
        for rank, item in enumerate(deviations, start=1):
            item["rank"] = rank

        severity = "low"
        for boundary, label in SEVERITY_BANDS:
            if score >= boundary:
                severity = label
                break

        offenders = [d["feature"] for d in deviations[:3] if abs(d["z_score"]) >= 2.0]
        if is_anomaly:
            summary = (
                f"The reading sits at the {score:.0%} percentile of the training score "
                f"distribution, above the {active_threshold:.0%} threshold. "
                + (f"Deviating sensors: {', '.join(offenders)}." if offenders
                   else "No single sensor dominates the deviation; the combination is unusual.")
            )
        else:
            summary = (
                f"The reading scores {score:.0%}, below the {active_threshold:.0%} threshold, "
                "so it falls inside the normal operating envelope learned from AI4I."
            )

        return {
            "model_version": self.version_id(),
            "anomaly_score": round(score, 6),
            "threshold": round(active_threshold, 6),
            "is_anomaly": is_anomaly,
            "severity": severity,
            "expected_value": {
                column: round(float(self.feature_mean.get(column, 0.0)), 4)
                for column in ai4i.SENSOR_COLUMNS
            },
            "feature_deviation": deviations,
            "explanation": summary,
            "model_trained": True,
        }
