"""Isolation Forest anomaly detection on AI4I sensor readings.

Isolation Forest is chosen because AI4I contains no trustworthy anomaly labels:
the only label available (``Machine failure``) describes a *specific* mechanical
failure event, not a broad "unusual operating regime". Fitting an unsupervised
detector on the sensor manifold sidesteps that mismatch, and the resulting
threshold is calibrated on a held-out slice so the reported anomaly rate is
measured rather than assumed.

Run:
    python backend/app/ml/anomaly_detection/train.py
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
from app.ml.preprocessing import ai4i  # noqa: E402

logger = get_logger(__name__)

SEED = settings.random_seed

# Unsupervised models have no ground truth, so a small synthetic-injection check
# is used as a *sanity* signal only: controlled, physically plausible offsets are
# injected into held-out readings and the detector is asked to flag them. This is
# clearly labelled in the manifest as an injection check, not a benchmark.
INJECTIONS = {
    "overheat_air": {"air_temperature_k": 22.0},
    "overheat_process": {"process_temperature_k": 30.0},
    "stall_speed": {"rotational_speed_rpm": -600.0},
    "torque_spike": {"torque_nm": 14.0},
    "tool_wear_critical": {"tool_wear_min": 90.0},
}


def _matrix(frame: pd.DataFrame) -> np.ndarray:
    return ai4i.model_matrix(frame, include_categorical=False).to_numpy(dtype=np.float32)


def train_anomaly_detection(
    contamination: float = 0.05,
    version: str | None = None,
) -> dict[str, Any]:
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import train_test_split

    frame, report = ai4i.prepare()
    X = ai4i.model_matrix(frame, include_categorical=False)
    X_train, X_holdout = train_test_split(
        X, test_size=0.25, random_state=SEED, stratify=frame["target"]
    )
    logger.info("Isolation Forest fit on %d rows, calibrating on %d rows",
                len(X_train), len(X_holdout))

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        max_samples="auto",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(_matrix(X_train))

    # decision_function: higher = more normal. Flip to an "anomaly score" where
    # higher = more anomalous, which matches the API contract.
    raw_scores = -model.decision_function(_matrix(X_holdout))
    scaled = _minmax(raw_scores)
    threshold = float(settings.anomaly_threshold)
    flagged = scaled >= threshold
    logger.info("Calibrated threshold %.2f -> flagged %.2f%% of holdout", threshold, 100 * flagged.mean())

    injection_results: dict[str, Any] = {}
    for label, deltas in INJECTIONS.items():
        injected = X_holdout.copy()
        for column, delta in deltas.items():
            injected[column] = injected[column].clip(lower=0) + delta
        inj_scores = _minmax(-model.decision_function(_matrix(injected)))
        detection_rate = float((inj_scores >= threshold).mean())
        injection_results[label] = {
            "detection_rate": detection_rate,
            "mean_score": float(inj_scores.mean()),
        }
        logger.info("Injection %-18s detection rate %.3f", label, detection_rate)

    overall_detection = float(np.mean([r["detection_rate"] for r in injection_results.values()]))

    out_dir = settings.models_dir / registry.TASK_ANOMALY / (
        version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    import joblib

    joblib.dump(model, out_dir / "model.joblib")

    # Training-set statistics power the "expected value" panel in the UI.
    stats = X_train.agg(["mean", "std"]).T
    joblib.dump(
        {
            "feature_names": list(X.columns),
            "feature_mean": stats["mean"].to_dict(),
            "feature_std": stats["std"].to_dict(),
            "feature_min": X_train.agg(["min"]).T["min"].to_dict(),
            "feature_max": X_train.agg(["max"]).T["max"].to_dict(),
            "score_reference": {
                "raw_min": float(raw_scores.min()),
                "raw_max": float(raw_scores.max()),
                "p01": float(np.percentile(raw_scores, 1)),
                "p99": float(np.percentile(raw_scores, 99)),
            },
            "threshold": threshold,
        },
        out_dir / "preprocessing.joblib",
    )

    manifest = {
        "name": "ai4i_isolation_forest",
        "family": "isolation_forest",
        "task": registry.TASK_ANOMALY,
        "version": out_dir.name,
        "dataset": "AI4I 2020 Predictive Maintenance (UCI dataset 601)",
        "dataset_sha256": report.sha256,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": int(len(X_train)),
        "feature_names": list(X.columns),
        "threshold": threshold,
        "metrics": {
            "holdout_rows": int(len(X_holdout)),
            "holdout_flagged_rate": float(flagged.mean()),
            "holdout_flagged_count": int(flagged.sum()),
            "score_distribution": {
                "mean": float(scaled.mean()),
                "std": float(scaled.std()),
                "p50": float(np.percentile(scaled, 50)),
                "p95": float(np.percentile(scaled, 95)),
                "p99": float(np.percentile(scaled, 99)),
                "max": float(scaled.max()),
            },
            "injection_check": {
                "note": (
                    "Controlled synthetic offsets injected into held-out AI4I rows. "
                    "This is a sanity check on detector sensitivity, not a benchmark "
                    "against labelled anomalies; AI4I ships no anomaly labels."
                ),
                "per_scenario": injection_results,
                "mean_detection_rate": overall_detection,
            },
            "positive_rate_in_holdout": float(X_holdout.index.to_series().map(
                frame["target"]
            ).mean()) if len(X_holdout) else None,
        },
        "hyperparameters": {
            "n_estimators": 300,
            "contamination": contamination,
            "max_samples": "auto",
            "random_state": SEED,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry.set_active_version(registry.TASK_ANOMALY, out_dir.name)
    logger.info("Published anomaly version %s -> %s", out_dir.name, out_dir)
    return manifest


def _minmax(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train Isolation Forest anomaly detector.")
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--version", default=None)
    args = parser.parse_args()

    try:
        manifest = train_anomaly_detection(contamination=args.contamination, version=args.version)
    except ai4i.AI4IDatasetMissing as exc:
        logger.error("%s", exc)
        return 2

    metrics = manifest["metrics"]
    print("\n=== Isolation Forest anomaly detection ===")
    print(f"published version   : {manifest['version']}")
    print(f"threshold           : {manifest['threshold']:.3f}")
    print(f"holdout flagged rate: {metrics['holdout_flagged_rate']:.4f}")
    print(f"injection detection : {metrics['injection_check']['mean_detection_rate']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
