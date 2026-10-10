"""Artifact registry for trained FactoryMind models.

A training script publishes a version directory containing:

    manifest.json      metadata + metrics + feature schema
    model.joblib       estimator (or torch state dict inside an .pt)
    preprocessing.joblib  scaler / encoder / threshold bundle

Nothing in this module invents a model. If no version exists for a task, the
loader returns ``None`` and callers must degrade explicitly rather than emit a
placeholder number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, TypeVar

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

TASK_FAILURE = "failure_prediction"
TASK_ANOMALY = "anomaly_detection"
TASK_RUL = "rul_prediction"
TASK_VISION = "visual_inspection"

ACTIVE_POINTER = "active.json"


@dataclass
class ModelArtifact(Generic[T]):
    """A loaded model version ready for inference."""

    task: str
    name: str
    version: str
    model: T
    preprocessing: dict[str, Any]
    feature_names: list[str]
    metrics: dict[str, Any]
    hyperparameters: dict[str, Any]
    dataset: str
    dataset_sha256: str | None
    training_samples: int | None
    trained_at: str | None
    path: Path

    @property
    def version_id(self) -> str:
        return f"{self.name}:{self.version}"

    def summary(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "name": self.name,
            "version": self.version,
            "version_id": self.version_id,
            "dataset": self.dataset,
            "dataset_sha256": self.dataset_sha256,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "hyperparameters": self.hyperparameters,
            "training_samples": self.training_samples,
            "trained_at": self.trained_at,
        }


def _task_dir(task: str) -> Path:
    return settings.models_dir / task


def task_dir(task: str) -> Path:
    """Public accessor for a task's artifact root."""
    return _task_dir(task)


def categories(task: str) -> list[str]:
    """Sub-category names (used by visual inspection, which stores per-category banks)."""
    root = _task_dir(task)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def list_versions(task: str) -> list[dict[str, Any]]:
    """Every published version for a task, newest first."""
    root = _task_dir(task)
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for child in root.iterdir():
        manifest_file = child / "manifest.json"
        if not manifest_file.is_file():
            continue
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Skipping unreadable manifest: %s", manifest_file)
            continue
        data["path"] = str(child)
        out.append(data)
    out.sort(key=lambda d: d.get("trained_at") or "", reverse=True)
    return out


def resolve_version_dir(task: str, version: str | None = None) -> Path | None:
    if version:
        candidate = _task_dir(task) / version
        return candidate if (candidate / "manifest.json").is_file() else None

    pointer = _task_dir(task) / ACTIVE_POINTER
    if pointer.is_file():
        try:
            active = json.loads(pointer.read_text(encoding="utf-8")).get("version")
            candidate = _task_dir(task) / str(active)
            if (candidate / "manifest.json").is_file():
                return candidate
        except json.JSONDecodeError:
            logger.warning("Corrupt active pointer at %s", pointer)

    versions = list_versions(task)
    return Path(versions[0]["path"]) if versions else None


def set_active_version(task: str, version: str) -> Path:
    root = _task_dir(task)
    if not (root / version / "manifest.json").is_file():
        raise FileNotFoundError(f"Model version {task}/{version} not found")
    (root / ACTIVE_POINTER).write_text(
        json.dumps({"version": version, "activated_at": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )
    return root / version


def load_manifest(task: str, version: str | None = None) -> dict[str, Any] | None:
    directory = resolve_version_dir(task, version)
    if directory is None:
        return None
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


_CACHE: dict[str, ModelArtifact[Any]] = {}


def load(
    task: str,
    version: str | None = None,
    family: str | None = None,
    use_cache: bool = True,
) -> ModelArtifact[Any] | None:
    """Load a trained artifact, or ``None`` when nothing is trained.

    ``family`` lets a task publish several algorithms (RUL ships an XGBoost
    baseline and an LSTM) and lets callers select one.
    """
    import joblib

    directory = resolve_version_dir(task, version)
    if directory is None:
        return None
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if family and manifest.get("family") and manifest["family"] != family:
        candidates = [v for v in list_versions(task) if v.get("family") == family]
        if not candidates:
            return None
        directory = Path(candidates[0]["path"])
        manifest = candidates[0]

    cache_key = f"{task}:{manifest['version']}:{manifest.get('family', '-')}"
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    model_file = directory / manifest.get("model_file", "model.joblib")
    if not model_file.is_file():
        logger.error("Model file missing for %s at %s", task, model_file)
        return None

    if model_file.suffix.lower() in {".pt", ".pth"}:
        # Sequence models are published as torch state-dict bundles.
        import torch

        model = torch.load(model_file, map_location="cpu", weights_only=False)
    else:
        model = joblib.load(model_file)

    preprocessing: dict[str, Any] = {}
    pre_file = directory / manifest.get("preprocessing_file", "preprocessing.joblib")
    if pre_file.is_file():
        preprocessing = joblib.load(pre_file)

    artifact: ModelArtifact[Any] = ModelArtifact(
        task=task,
        name=manifest.get("name", task),
        version=manifest["version"],
        model=model,
        preprocessing=preprocessing,
        feature_names=list(manifest.get("feature_names", [])),
        metrics=dict(manifest.get("metrics", {})),
        hyperparameters=dict(manifest.get("hyperparameters", {})),
        dataset=manifest.get("dataset", "unknown"),
        dataset_sha256=manifest.get("dataset_sha256"),
        training_samples=manifest.get("training_samples"),
        trained_at=manifest.get("trained_at"),
        path=directory,
    )
    if use_cache:
        _CACHE[cache_key] = artifact
    return artifact


def is_trained(task: str) -> bool:
    return resolve_version_dir(task) is not None


def available_families(task: str) -> list[str]:
    return sorted({v.get("family") for v in list_versions(task) if v.get("family")})


def clear_cache() -> None:
    _CACHE.clear()
