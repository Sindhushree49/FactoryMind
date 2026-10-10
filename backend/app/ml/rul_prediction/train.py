"""Remaining Useful Life regression on NASA C-MAPSS.

Two model families are trained and both are published:

  * ``xgboost`` - flat-feature baseline. Each sliding window is summarised into
    window statistics (mean/std/last/slope), which is the conventional strong
    baseline for C-MAPSS tabular RUL.
  * ``lstm``     - sequence model over the raw window, so degradation *dynamics*
    are modelled rather than only window aggregates.

Leakage control
---------------
Windows are built per engine, then split by **engine unit**. Holding out whole
trajectories prevents the model from being evaluated on near-duplicates of its
own training cycles. The official ``test_FD00x`` split is additionally scored on
the capped-RUL ground truth in ``RUL_FD00x.txt``.

Run:
    python backend/app/ml/rul_prediction/train.py --subset FD001
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.config import settings  # noqa: E402
from app.core.logging_config import configure_logging, get_logger  # noqa: E402
from app.ml import registry  # noqa: E402
from app.ml.preprocessing import cmapss  # noqa: E402

logger = get_logger(__name__)

SEED = settings.random_seed


def window_summary(window: np.ndarray) -> np.ndarray:
    """Flatten a ``(window, features)`` block into fixed-length statistics.

    Captures level, variability, current state and trend, which is what the
    tabular baseline needs and is scale/order invariant across the four subsets.
    """
    steps = np.arange(window.shape[0], dtype=np.float32)
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    last = window[-1]
    first = window[0]
    denom = ((steps - steps.mean()) ** 2).sum()
    if window.shape[0] < 2 or denom <= 0.0:
        # A single-step prefix has no observable slope; zero keeps the
        # feature finite instead of emitting NaN into the model.
        slope = np.zeros_like(last)
        linear_slope = np.zeros_like(last)
    else:
        slope = (window[-1] - window[0]) / (window.shape[0] - 1)
        centred = window - mean
        linear_slope = (centred * (steps - steps.mean())[:, None]).sum(axis=0) / denom
    return np.concatenate([mean, std, last, first, slope, linear_slope, np.ptp(window, axis=0)])


def summarise_all(windows: np.ndarray) -> np.ndarray:
    return np.stack([window_summary(w) for w in windows])


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "nasa_score": _nasa_asymmetric_score(y_true, y_pred),
        "support": int(len(y_true)),
    }


def _nasa_asymmetric_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PHM'08 scoring: late predictions are penalised more than early ones."""
    y_pred = np.clip(y_pred, 0, 125)
    diff = y_pred - y_true
    scores = np.where(diff >= 0, 13 * diff, 5 * np.abs(diff))
    return float(scores.mean())


def train_rul_prediction(
    subset: str = "FD001",
    window: int = cmapss.WINDOW_SIZE,
    epochs: int = 40,
    version: str | None = None,
) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    frame = cmapss.load_train(subset)
    sensors = cmapss.usable_sensors()
    X, y, units, cycles = cmapss.build_windows(frame, window=window, sensor_cols=sensors)
    units = np.asarray(units, dtype=str)
    logger.info("%s windows: X=%s sensors=%d units=%d", subset, X.shape, len(sensors), len(np.unique(units)))

    train_mask, test_mask = cmapss.split_by_unit(units, test_fraction=0.2, seed=SEED)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    units_test = units[test_mask]
    logger.info("Engine-wise split: train windows=%d (%d engines), test windows=%d (%d engines)",
                len(X_train), len(np.unique(units[train_mask])), len(X_test), len(np.unique(units_test)))

    # ---- XGBoost tabular baseline -------------------------------------
    features_train = summarise_all(X_train)
    features_test = summarise_all(X_test)
    scaler = StandardScaler().fit(features_train)

    reg = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        min_child_weight=3,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    reg.fit(features_train, y_train)
    val_pred = np.clip(reg.predict(features_test), 0, cmapss.MAX_RUL)
    xgb_metrics = _regression_metrics(y_test, val_pred)
    logger.info("XGBoost RUL  MAE=%.2f RMSE=%.2f R2=%.4f NASA=%.2f",
                xgb_metrics["mae"], xgb_metrics["rmse"], xgb_metrics["r2"], xgb_metrics["nasa_score"])

    # ---- LSTM sequence model -------------------------------------------
    lstm_metrics, lstm_payload = _train_lstm(X_train, y_train, X_test, y_test, sensors, epochs)

    out_dir = settings.models_dir / registry.TASK_RUL / (
        version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    import joblib

    joblib.dump(reg, out_dir / "model.joblib")
    joblib.dump(
        {
            "window": window,
            "sensors": sensors,
            "scaler": scaler,
            "feature_dim": int(features_train.shape[1]),
            "max_rul": cmapss.MAX_RUL,
            "family": "xgboost",
        },
        out_dir / "preprocessing.joblib",
    )

    manifest = {
        "name": f"cmapss_rul_xgboost_{subset.lower()}",
        "family": "xgboost",
        "task": registry.TASK_RUL,
        "version": out_dir.name,
        "dataset": f"NASA C-MAPSS {subset} (Turbofan Engine Degradation Simulation)",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": int(len(X_train)),
        "feature_names": _summary_feature_names(sensors),
        "window": window,
        "dataset_subset": subset,
        "metrics": {
            "engine_split": xgb_metrics,
            "train_windows": int(len(X_train)),
            "test_windows": int(len(X_test)),
            "test_engines": sorted(set(units_test.tolist())),
            "split_strategy": "by engine unit, never by row (leakage control)",
            "rul_cap": cmapss.MAX_RUL,
        },
        "hyperparameters": {
            "n_estimators": 600, "learning_rate": 0.05, "max_depth": 6,
            "subsample": 0.85, "colsample_bytree": 0.8, "reg_lambda": 1.5,
            "random_state": SEED,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry.set_active_version(registry.TASK_RUL, out_dir.name)

    # Publish the LSTM under its own version so both are selectable.
    lstm_dir = settings.models_dir / registry.TASK_RUL / f"{out_dir.name}_lstm"
    lstm_dir.mkdir(parents=True, exist_ok=True)
    torch_save = {"state_dict": lstm_payload["state_dict"], "config": lstm_payload["config"]}
    torch = _torch()
    torch.save(torch_save, lstm_dir / "model.pt")
    joblib.dump(
        {
            "window": window,
            "sensors": sensors,
            "scaler": lstm_payload["scaler"],
            "max_rul": cmapss.MAX_RUL,
            "family": "lstm",
            "history": lstm_payload["history"],
        },
        lstm_dir / "preprocessing.joblib",
    )
    (lstm_dir / "manifest.json").write_text(
        json.dumps(
            {
                "name": f"cmapss_rul_lstm_{subset.lower()}",
                "family": "lstm",
                "task": registry.TASK_RUL,
                "version": lstm_dir.name,
                "model_file": "model.pt",
                "dataset": manifest["dataset"],
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "training_samples": int(len(X_train)),
                "feature_names": sensors,
                "window": window,
                "dataset_subset": subset,
                "metrics": {"engine_split": lstm_metrics, "epochs": len(lstm_payload["history"]["loss"])},
                "hyperparameters": lstm_payload["config"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _evaluate_official_test(manifest, subset, window, sensors)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Published RUL versions %s and %s", out_dir.name, lstm_dir.name)
    return manifest


def _torch() -> Any:
    import torch

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return torch


def _train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensors: list[str],
    epochs: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _torch()
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # Scale across the whole training corpus (per-feature), fitted on train only.
    scaler = StandardScaler()
    flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(flat)
    scaled_train = scaler.transform(flat).reshape(X_train.shape).astype(np.float32)
    scaled_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape).astype(np.float32)

    device = torch.device(settings.torch_device)
    n_features = scaled_train.shape[-1]

    class RULLSTM(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    torch.manual_seed(SEED)
    model = RULLSTM(n_features).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss(beta=5.0)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(scaled_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=128, shuffle=True, num_workers=0,
    )

    history: dict[str, list[float]] = {"loss": [], "val_mae": []}
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")

    X_val_tensor = torch.from_numpy(scaled_test).to(device)
    y_val_np = y_test.astype(np.float32)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()
            optimiser.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += float(loss.item()) * len(batch_x)
        train_loss = epoch_loss / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            predictions = model(X_val_tensor).cpu().numpy()
        clipped = np.clip(predictions, 0, cmapss.MAX_RUL)
        val_mae = float(np.abs(clipped - y_val_np).mean())

        history["loss"].append(train_loss)
        history["val_mae"].append(val_mae)
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info("LSTM epoch %02d/%d loss=%.4f val_MAE=%.2f", epoch + 1, epochs, train_loss, val_mae)

    model.load_state_dict(best_state)
    final_pred = np.clip(_lstm_predict(model, X_val_tensor, len(y_val_np)), 0, cmapss.MAX_RUL)
    metrics = _regression_metrics(y_val_np, final_pred)
    logger.info("LSTM RUL  MAE=%.2f RMSE=%.2f R2=%.4f NASA=%.2f",
                metrics["mae"], metrics["rmse"], metrics["r2"], metrics["nasa_score"])

    payload = {
        "state_dict": best_state,
        "scaler": scaler,
        "history": history,
        "config": {"input_dim": n_features, "hidden_dim": 64, "layers": 2,
                   "window": int(X_train.shape[1]), "dropout": 0.2},
    }
    return metrics, payload


def _lstm_predict(model: Any, x_tensor: Any, count: int) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        return model(x_tensor).cpu().numpy()


def _summary_feature_names(sensors: list[str]) -> list[str]:
    return [
        f"{stat}_{sensor}"
        for stat in ("mean", "std", "last", "first", "delta", "slope", "range")
        for sensor in sensors
    ]


def _evaluate_official_test(manifest: dict[str, Any], subset: str, window: int, sensors: list[str]) -> None:
    """Score the published model on the official PHM test split."""
    from app.ml.rul_prediction.inference import RULInference

    try:
        inference = RULInference(family="xgboost")
        if not inference.ready:
            logger.warning("Skipping official C-MAPSS test evaluation: model not loadable")
            return
        frame, truth = cmapss.load_test(subset)
        predictions = []
        actuals = []
        for unit, group in frame.groupby("unit", sort=True):
            group = group.sort_values("cycle")
            if len(group) < window:
                continue
            actuals.append(float(truth.loc[unit]))
            predictions.append(
                inference.predict_window(group[sensors].to_numpy(dtype=np.float32), subset=subset)[
                    "predicted_rul"
                ]
            )
        if not predictions:
            logger.warning("Official test split too short for window=%d", window)
            return
        metrics = _regression_metrics(np.array(actuals), np.array(predictions))
        logger.info("Official %s test split: MAE=%.2f RMSE=%.2f NASA=%.2f",
                    subset, metrics["mae"], metrics["rmse"], metrics["nasa_score"])
        manifest["metrics"]["official_test_split"] = metrics
    except Exception as exc:  # noqa: BLE001
        logger.warning("Official C-MAPSS test evaluation failed: %s", exc)


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train C-MAPSS RUL models (XGBoost + LSTM).")
    parser.add_argument("--subset", default="FD001", choices=cmapss.SUBSETS)
    parser.add_argument("--window", type=int, default=cmapss.WINDOW_SIZE)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--version", default=None)
    args = parser.parse_args()

    try:
        manifest = train_rul_prediction(
            subset=args.subset, window=args.window, epochs=args.epochs, version=args.version
        )
    except cmapss.CMAPSSMissing as exc:
        logger.error("%s", exc)
        return 2

    metrics = manifest["metrics"]["engine_split"]
    print(f"\n=== C-MAPSS RUL ({args.subset}) ===")
    print(f"published version : {manifest['version']}")
    print(f"XGBoost  MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} R2={metrics['r2']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
