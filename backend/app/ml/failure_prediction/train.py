"""Train and compare failure-prediction models on AI4I 2020.

Two tasks are trained from the same dataset, with separate models because they
are genuinely different problems:

  1. Binary failure prediction (target = ``Machine failure``)
  2. Failure-mode classification (target = ``TID``, positives only)

Algorithms compared: RandomForest and XGBoost. The better of the two (selected
on the *validation* split, never the test split) is published as the active
model. Test metrics are computed exactly once at the end.

Run:
    python backend/app/ml/failure_prediction/train.py
    python backend/app/ml/failure_prediction/train.py --subsample 0.5
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

if __package__ in {None, ""}:  # allow `python backend/app/ml/.../train.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.config import settings  # noqa: E402
from app.core.logging_config import get_logger, configure_logging  # noqa: E402
from app.ml import registry  # noqa: E402
from app.ml.preprocessing import ai4i  # noqa: E402

logger = get_logger(__name__)

SEED = settings.random_seed


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["average_precision"] = float(average_precision_score(y_true, y_proba))
    except ValueError:
        metrics["roc_auc"] = None
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    metrics["support"] = int(len(y_true))
    metrics["positives"] = int(y_true.sum())
    return metrics


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "support": int(len(y_true)),
    }


def _make_preprocessor() -> Any:
    """OneHot for machine type, passthrough for the numeric block.

    Feature engineering already put every numeric feature on a comparable scale,
    so a scaler would add a fitted artifact without changing model behaviour.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    numeric = ai4i.numeric_feature_names()
    categorical = list(ai4i.CATEGORICAL_COLUMNS)
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _build_estimators(max_depth: int | None) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    n_estimators = 400
    common: dict[str, Any] = {
        "n_estimators": n_estimators,
        "random_state": SEED,
        "n_jobs": -1,
        "class_weight": "balanced",
    }
    xgb: dict[str, Any] = {
        "n_estimators": n_estimators,
        "random_state": SEED,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "scale_pos_weight": None,
        "learning_rate": 0.08,
        "max_depth": max_depth or 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "min_child_weight": 1,
    }
    estimators: dict[str, Any] = {
        "random_forest": RandomForestClassifier(
            max_depth=max_depth or 18,
            min_samples_leaf=1,
            max_features="sqrt",
            **common,
        ),
        "xgboost": XGBClassifier(**xgb),
    }
    return estimators


def _fit_and_score(
    name: str,
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> dict[str, Any]:
    estimator.fit(X_train, y_train)
    val_proba = estimator.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    return {
        "algorithm": name,
        "validation": _classification_metrics(y_val, val_pred, val_proba),
        "estimator": estimator,
    }


def train_failure_prediction(
    subsample: float = 1.0,
    test_fraction: float = 0.2,
    val_fraction: float = 0.15,
    version: str | None = None,
) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split

    frame, report = ai4i.prepare()
    logger.info(
        "AI4I loaded: %d rows, failure rate %.4f, missing=%d, duplicates=%d",
        report.rows, report.failure_rate, report.missing_values, report.duplicate_rows,
    )

    if subsample < 1.0:
        # Stratified subsampling keeps the ~3.4% positive rate intact.
        parts = [
            group.sample(frac=subsample, random_state=SEED)
            for _, group in frame.groupby("target", sort=True)
        ]
        frame = pd.concat(parts).sort_values("udi").reset_index(drop=True)
        logger.info("Subsampled to %d rows for faster iteration", len(frame))

    X = ai4i.model_matrix(frame)
    y = frame["target"].to_numpy(dtype=int)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_fraction, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_fraction / (1 - test_fraction),
        stratify=y_tmp, random_state=SEED,
    )
    logger.info("Split: train=%d val=%d test=%d (positives %d/%d/%d)",
                len(X_train), len(X_val), len(X_test), int(y_train.sum()),
                int(y_val.sum()), int(y_test.sum()))

    preprocessor = _make_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    results = [
        _fit_and_score(name, est, X_train_t, y_train, X_val_t, y_val)
        for name, est in _build_estimators(max_depth=None).items()
    ]
    for item in results:
        logger.info("Validation %-14s f1=%.4f roc_auc=%s",
                    item["algorithm"], item["validation"]["f1"],
                    f"{item['validation']['roc_auc']:.4f}" if item["validation"]["roc_auc"] else "n/a")

    # Winner chosen on validation F1 only.
    best = max(results, key=lambda r: r["validation"]["f1"])
    logger.info("Selected algorithm: %s", best["algorithm"])

    # Threshold tuned on validation, then frozen for the test evaluation.
    val_proba = best["estimator"].predict_proba(X_val_t)[:, 1]
    from sklearn.metrics import f1_score

    grid = np.arange(0.20, 0.81, 0.01)
    f1s = [f1_score(y_val, (val_proba >= t).astype(int), zero_division=0) for t in grid]
    decision_threshold = float(grid[int(np.argmax(f1s))])
    logger.info("Decision threshold tuned on validation: %.2f (val f1=%.4f)",
                decision_threshold, max(f1s))

    test_proba = best["estimator"].predict_proba(X_test_t)[:, 1]
    test_pred = (test_proba >= decision_threshold).astype(int)
    test_metrics = _classification_metrics(y_test, test_pred, test_proba)
    logger.info("TEST %s: acc=%.4f prec=%.4f rec=%.4f f1=%.4f roc_auc=%.4f",
                best["algorithm"], test_metrics["accuracy"], test_metrics["precision"],
                test_metrics["recall"], test_metrics["f1"], test_metrics["roc_auc"] or 0.0)

    out_dir = settings.models_dir / registry.TASK_FAILURE / (version or _timestamp_version())
    out_dir.mkdir(parents=True, exist_ok=True)

    import joblib

    joblib.dump(best["estimator"], out_dir / "model.joblib")
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "decision_threshold": decision_threshold,
            "sensor_columns": ai4i.SENSOR_COLUMNS,
            "engineered_columns": ai4i.ENGINEERED_COLUMNS,
            "categorical_columns": ai4i.CATEGORICAL_COLUMNS,
        },
        out_dir / "preprocessing.joblib",
    )

    manifest = {
        "name": "ai4i_failure_xgboost" if best["algorithm"] == "xgboost" else "ai4i_failure_random_forest",
        "family": best["algorithm"],
        "task": registry.TASK_FAILURE,
        "version": out_dir.name,
        "dataset": "AI4I 2020 Predictive Maintenance (UCI dataset 601)",
        "dataset_sha256": report.sha256,
        "dataset_facts": {
            "rows": report.rows,
            "failure_rate": report.failure_rate,
            "failure_mode_distribution": report.failure_mode_distribution,
            "machine_type_distribution": report.machine_type_distribution,
            "missing_values": report.missing_values,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": int(len(X_train)),
        "feature_names": ai4i.feature_names(),
        "decision_threshold": decision_threshold,
        "metrics": {
            "test": test_metrics,
            "validation_selected": best["validation"],
            "validation_f1_by_algorithm": {r["algorithm"]: r["validation"]["f1"] for r in results},
            "validation_roc_auc_by_algorithm": {
                r["algorithm"]: r["validation"]["roc_auc"] for r in results
            },
            "data_split": {
                "train": int(len(X_train)),
                "val": int(len(X_val)),
                "test": int(len(X_test)),
                "strategy": "stratified on target, random_state=%d" % SEED,
            },
        },
        "hyperparameters": {
            k: v for k, v in best["estimator"].get_params().items()
            if isinstance(v, (int, float, str, bool, type(None)))
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry.set_active_version(registry.TASK_FAILURE, out_dir.name)
    logger.info("Published %s version %s -> %s", registry.TASK_FAILURE, out_dir.name, out_dir)

    comparison = _train_failure_mode_model(out_dir, frame, manifest)
    manifest["metrics"]["failure_mode_classification"] = comparison
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _train_failure_mode_model(
    parent_dir: Path, frame: pd.DataFrame, parent_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Multiclass model over the five documented failure modes (positives only)."""
    from sklearn.model_selection import train_test_split

    positives = frame[frame["target"] == 1].copy()
    if positives.empty:
        return {"status": "skipped", "reason": "no positive samples"}

    X = ai4i.model_matrix(positives)
    y = positives["model"].to_numpy(dtype=int)
    logger.info("Failure-mode model: %d positives across %d modes", len(y), len(np.unique(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    preprocessor = _make_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=1,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    classifier.fit(X_train_t, y_train)
    predictions = classifier.predict(X_test_t)
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "report": classification_report(
            y_test, predictions, zero_division=0, output_dict=True,
            target_names=[ai4i.FAILURE_MODES[int(i)] for i in sorted(np.unique(y))],
        ),
        "classes": sorted(int(c) for c in np.unique(y)),
        "support": int(len(y_test)),
    }

    import joblib

    joblib.dump(classifier, parent_dir / "failure_mode_model.joblib")
    (parent_dir / "failure_mode_manifest.json").write_text(
        json.dumps(
            {
                "task": "failure_mode_classification",
                "algorithm": "random_forest",
                "classes": metrics["classes"],
                "labels": {str(k): v for k, v in ai4i.FAILURE_MODES.items()},
                "metrics": metrics,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "dataset": parent_manifest["dataset"],
                "dataset_sha256": parent_manifest.get("dataset_sha256"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Failure-mode model accuracy=%.4f macro_f1=%.4f",
                metrics["accuracy"], metrics["macro_f1"])
    return metrics


def _timestamp_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train AI4I failure prediction models.")
    parser.add_argument("--subsample", type=float, default=1.0,
                        help="Stratified fraction of the dataset to train on (1.0 = all).")
    parser.add_argument("--version", default=None)
    args = parser.parse_args()

    try:
        manifest = train_failure_prediction(subsample=args.subsample, version=args.version)
    except ai4i.AI4IDatasetMissing as exc:
        logger.error("%s", exc)
        return 2

    test = manifest["metrics"]["test"]
    print("\n=== AI4I failure prediction ===")
    print(f"published version : {manifest['version']}")
    print(f"selected model    : {manifest['family']}")
    print(f"decision threshold: {manifest['decision_threshold']:.2f}")
    print(f"test accuracy     : {test['accuracy']:.4f}")
    print(f"test precision    : {test['precision']:.4f}")
    print(f"test recall       : {test['recall']:.4f}")
    print(f"test f1           : {test['f1']:.4f}")
    print(f"test roc_auc      : {test['roc_auc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
