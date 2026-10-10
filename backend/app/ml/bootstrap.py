"""First-time model initialization for FactoryMind.

If a task's model artifact is missing, this module trains it from the project's
bundled historical dataset using the existing training pipeline. Training runs
only once per missing artifact; existing artifacts are never overwritten.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from app.core.logging_config import get_logger
from app.ml import registry
from app.services import ml_service

logger = get_logger(__name__)


def _is_trained(task: str) -> bool:
    return registry.resolve_version_dir(task) is not None


def _train_failure() -> None:
    from app.ml.failure_prediction.train import train_failure_prediction

    logger.info("Training failure prediction model on AI4I 2020...")
    train_failure_prediction(subsample=1.0)
    logger.info("Failure prediction model trained.")


def _train_anomaly() -> None:
    from app.ml.anomaly_detection.train import train_anomaly_detection

    logger.info("Training anomaly detection model on AI4I 2020...")
    train_anomaly_detection(contamination=0.05)
    logger.info("Anomaly detection model trained.")


def _train_rul() -> None:
    from app.ml.rul_prediction.train import train_rul_prediction

    logger.info("Training RUL model on NASA C-MAPSS FD001...")
    train_rul_prediction(subset="FD001", window=30, epochs=40)
    logger.info("RUL model trained.")


def ensure_models() -> None:
    """Train any missing models and load them into memory."""
    ml_service.invalidate()

    tasks: list[tuple[str, Callable[[], None]]] = [
        (registry.TASK_FAILURE, _train_failure),
        (registry.TASK_ANOMALY, _train_anomaly),
        (registry.TASK_RUL, _train_rul),
    ]

    for task, trainer in tasks:
        if _is_trained(task):
            logger.info("Model %s already exists, skipping training.", task)
            continue
        try:
            trainer()
        except Exception as exc:
            logger.error("Model initialization failed for %s: %s", task, exc, exc_info=True)

    ml_service.invalidate()
    logger.info("Model initialization complete.")
