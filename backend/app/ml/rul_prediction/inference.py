"""RUL inference for C-MAPSS engine trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logging_config import get_logger
from app.ml import registry

logger = get_logger(__name__)


class ModelNotTrained(RuntimeError):
    def __init__(self, task: str) -> None:
        super().__init__(
            f"No trained model found for {task}. "
            f"Train it with: python backend/app/ml/{task}/train.py"
        )
        self.task = task


class RULInference:
    """Serves RUL estimates from either the XGBoost baseline or the LSTM."""

    def __init__(self, version: str | None = None, family: str | None = None) -> None:
        self.artifact = registry.load(registry.TASK_RUL, version=version, family=family)
        if self.artifact is None:
            logger.warning("RUL model not trained; inference disabled")
            return
        pre = self.artifact.preprocessing
        self.window = int(pre["window"])
        self.sensors: list[str] = list(pre["sensors"])
        self.max_rul = float(pre.get("max_rul", 125))
        self.family = str(pre.get("family", self.artifact.metrics.get("family", "xgboost")))
        self.scaler = pre["scaler"]
        self._torch_model = None
        if self.family == "lstm":
            self._torch_model = self._load_torch()

    @property
    def ready(self) -> bool:
        return self.artifact is not None

    def version_id(self) -> str:
        return self.artifact.version_id if self.artifact else "untrained"

    def _load_torch(self) -> Any:
        import torch
        from torch import nn

        manifest_name = self.artifact.path / "model.pt"
        payload = torch.load(manifest_name, map_location=settings.torch_device, weights_only=False)
        config = payload["config"]
        n_features = int(config["input_dim"])
        hidden = int(config.get("hidden_dim", 64))
        layers = int(config.get("layers", 2))

        class RULLSTM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True, dropout=0.2)
                self.head = nn.Sequential(
                    nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)
                )

            def forward(self, x):  # noqa: ANN001, ANN202
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        model = RULLSTM()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    def predict_window(
        self,
        window: np.ndarray,
        subset: str = "FD001",
        engine_unit: str = "unknown",
        cycle: int | None = None,
        include_curve: bool = True,
    ) -> dict[str, Any]:
        if self.artifact is None:
            raise ModelNotTrained(registry.TASK_RUL)

        window = np.asarray(window, dtype=np.float32)
        if window.ndim != 2:
            raise ValueError(f"window must be 2-D (window, features); got shape {window.shape}")
        if window.shape[1] != len(self.sensors):
            raise ValueError(
                f"window has {window.shape[1]} features but this model expects "
                f"{len(self.sensors)} ({self.sensors[:3]}...)"
            )
        supplied = window.shape[0]
        if window.shape[0] < self.window:
            if window.shape[0] == 0:
                raise ValueError("window is empty")
            # Left-pad by repeating the earliest observation. This is flagged in
            # the response so a short request is never silently trusted.
            pad = np.repeat(window[:1], self.window - window.shape[0], axis=0)
            window = np.concatenate([pad, window], axis=0)
            padded = True
        else:
            window = window[-self.window :]
            padded = False

        if self.family == "lstm":
            predicted = self._predict_lstm(window)
            uncertainty = self._lstm_uncertainty(window)
        else:
            from app.ml.rul_prediction.train import window_summary

            features = self.scaler.transform(window_summary(window).reshape(1, -1))
            predicted = float(np.clip(self.artifact.model.predict(features)[0], 0, self.max_rul))
            uncertainty = self._tabular_uncertainty(window)

        spread = max(1.0, uncertainty)
        lower = max(0.0, predicted - spread)
        upper = min(self.max_rul, predicted + spread)
        uncertainty_pct = round(100.0 * spread / max(predicted, 1.0), 2)

        curve: list[dict[str, float]] = []
        if include_curve:
            curve = self._degradation_curve(window, predicted)

        caveats: list[str] = []
        if padded:
            caveats.append(
                f"Only {supplied} cycles were supplied for a {self.window}-cycle "
                "window; earlier steps were padded by repeating the oldest observation."
            )
        if subset != "FD001":
            caveats.append(
                f"This model was trained on C-MAPSS {subset}. Applying it to a different "
                "operating-condition family is an extrapolation."
            )

        confidence_note = (
            f"Interval from the residual spread of held-out engine trajectories "
            f"({self._split_mae() or float('nan'):.1f} MAE on the engine-wise "
            "split), not a calibrated Bayesian posterior."
        )

        return {
            "model_version": self.version_id(),
            "model_family": self.family,
            "dataset_subset": subset,
            "engine_unit": engine_unit,
            "cycle": int(cycle) if cycle is not None else None,
            "predicted_rul": round(float(predicted), 2),
            "lower_bound": round(float(lower), 2),
            "upper_bound": round(float(upper), 2),
            "uncertainty_pct": uncertainty_pct,
            "confidence_note": confidence_note,
            "window_length": int(self.window),
            "degradation_curve": curve,
            "model_trained": True,
            "caveats": caveats,
        }

    def _predict_lstm(self, window: np.ndarray) -> float:
        import torch

        scaled = self.scaler.transform(window.reshape(-1, window.shape[-1])).reshape(window.shape)
        tensor = torch.from_numpy(scaled.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            raw = self._torch_model(tensor).item()
        return float(np.clip(raw, 0, self.max_rul))

    def _split_mae(self) -> float | None:
        """MAE of the engine-wise split, where training actually records it."""
        engine_split = self.artifact.metrics.get("engine_split")
        if isinstance(engine_split, dict):
            mae = engine_split.get("mae")
            if isinstance(mae, (int, float)):
                return float(mae)
        mae = self.artifact.metrics.get("mae")
        return float(mae) if isinstance(mae, (int, float)) else None

    def _tabular_uncertainty(self, window: np.ndarray) -> float:
        mae = self._split_mae()
        if mae is not None and mae > 0:
            return mae * 1.25
        return 12.5

    def _lstm_uncertainty(self, window: np.ndarray) -> float:
        mae = self._split_mae()
        if mae is not None and mae > 0:
            return mae * 1.5
        return 15.0

    def _degradation_curve(self, window: np.ndarray, final_rul: float) -> list[dict[str, float]]:
        """Per-step estimates across the window, oldest first.

        Each prefix is scored independently with the same model, giving an
        interpretable degradation trajectory instead of a single number.
        """
        curve: list[dict[str, float]] = []
        steps = len(window)
        if steps < 3:
            return curve
        stride = max(1, steps // 12)
        for end in range(1, steps + 1, stride):
            prefix = window[max(0, end - self.window) : end]
            if len(prefix) < 1:
                continue
            try:
                if self.family == "lstm":
                    value = self._predict_lstm(prefix)
                else:
                    from app.ml.rul_prediction.train import window_summary

                    features = self.scaler.transform(window_summary(prefix).reshape(1, -1))
                    value = float(np.clip(self.artifact.model.predict(features)[0], 0, self.max_rul))
            except Exception:  # noqa: BLE001
                continue
            curve.append(
                {
                    "step": end,
                    "cycle_offset": end,
                    "rul": round(float(value), 2),
                    "health_signal": round(float(np.mean(prefix[:, 1:])), 4),
                }
            )
        if curve:
            curve[-1]["rul"] = round(float(final_rul), 2)
        return curve
