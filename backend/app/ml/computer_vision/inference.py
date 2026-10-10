"""Visual inspection inference: localisation, heatmap and report artefacts.

If no memory bank is trained for the requested category this module returns
``model_trained: false`` with remediation instructions. It never returns a
synthesised anomaly score.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logging_config import get_logger
from app.ml import registry
from app.ml.computer_vision.features import InvalidImage, encode_png, load_and_prepare

logger = get_logger(__name__)


@dataclass
class InspectionResult:
    model_trained: bool
    verdict: str
    message: str
    category: str | None = None
    model_version: str | None = None
    is_anomalous: bool | None = None
    anomaly_score: float | None = None
    threshold: float | None = None
    num_anomalous_patches: int | None = None
    localisation_pct: float | None = None
    anomaly_map: np.ndarray | None = None
    heatmap_png: bytes | None = None
    overlay_png: bytes | None = None
    original_png: bytes | None = None
    preprocessing: dict[str, Any] | None = None
    model_metrics: dict[str, Any] | None = None
    caveats: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


class VisionInference:
    """PatchCore scorer with one memory bank per MVTec category."""

    def __init__(self, category: str | None = None) -> None:
        self.category = category
        self.config: dict[str, Any] | None = None
        self.memory_bank: np.ndarray | None = None
        self.manifest: dict[str, Any] | None = None
        self.directory: Path | None = None
        self._backbone = None
        self._resolve()

    # -- model resolution -------------------------------------------------
    def _resolve(self) -> None:
        import joblib

        task_root = settings.models_dir / registry.TASK_VISION
        candidates: list[Path] = []
        if self.category and (task_root / self.category).is_dir():
            candidates = [
                p for p in sorted((task_root / self.category).iterdir(), reverse=True)
                if (p / "manifest.json").is_file() and (p / "memory_bank.npy").is_file()
            ]
        elif task_root.is_dir():
            for category_dir in sorted(task_root.iterdir()):
                if not category_dir.is_dir():
                    continue
                candidates.extend(
                    p for p in sorted(category_dir.iterdir(), reverse=True)
                    if (p / "manifest.json").is_file() and (p / "memory_bank.npy").is_file()
                )
        if not candidates:
            logger.warning("No PatchCore memory bank found under %s", task_root)
            return

        import json

        self.directory = candidates[0]
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.config = joblib.load(self.directory / "config.joblib")
        self.memory_bank = np.load(self.directory / "memory_bank.npy")
        self.category = self.config.get("category") or self.manifest.get("category")
        logger.info("Loaded PatchCore bank for category '%s' (%d vectors) from %s",
                    self.category, self.memory_bank.shape[0], self.directory)

    @property
    def ready(self) -> bool:
        return self.memory_bank is not None

    @property
    def trained_categories(self) -> list[str]:
        task_root = settings.models_dir / registry.TASK_VISION
        if not task_root.is_dir():
            return []
        return sorted(
            p.name for p in task_root.iterdir()
            if p.is_dir() and any((c / "manifest.json").is_file() for c in p.iterdir())
        )

    def _backbone_net(self) -> Any:
        if self._backbone is None:
            import torch
            from torchvision.models import resnet18, ResNet18_Weights

            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self._backbone = torch.nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2
            ).eval()
        return self._backbone

    # -- inspection -------------------------------------------------------
    def inspect(self, data: bytes, filename: str, category: str | None = None) -> InspectionResult:
        started = time.perf_counter()
        target = category or self.category

        if not self.ready:
            return InspectionResult(
                model_trained=False,
                verdict="model_not_trained",
                message=(
                    "No PatchCore memory bank is available, so no visual inspection was "
                    "performed. This is reported as untrained rather than estimated. "
                    "Fix: install MVTec AD under datasets/mvtec_ad/mvtec_anomaly_detection/ "
                    "then run: python backend/app/ml/computer_vision/train.py --all"
                ),
                category=target,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        if target and self.category and target != self.category:
            return InspectionResult(
                model_trained=False,
                verdict="category_mismatch",
                message=(
                    f"The trained bank covers category '{self.category}', not '{target}'. "
                    "A PatchCore bank is specific to one product geometry, so it cannot be "
                    f"applied to '{target}'. Train it with: "
                    f"python backend/app/ml/computer_vision/train.py --categories {target}"
                ),
                category=target,
                model_version=self.manifest.get("version") if self.manifest else None,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        prepared = load_and_prepare(data, filename, image_size=int(self.config["image_size"]))
        from app.ml.computer_vision.train import _min_distance, _patch_features, _upsample

        patches, feature_hw = _patch_features(self._backbone_net(), prepared.rgb)
        distances = _min_distance(self.memory_bank, patches)
        anomaly_map = _upsample(distances, feature_hw, int(self.config["image_size"]))

        threshold = float(self.config["threshold"])
        # Anomaly score normalised against the good-image calibration, so it is
        # comparable across categories with different feature scales.
        score = float(anomaly_map.max())
        normalised = score / threshold if threshold > 0 else 0.0
        is_anomalous = score >= threshold

        mask = anomaly_map >= threshold
        patch_count = int(mask.sum())
        localisation = float(100.0 * patch_count / mask.size)
        # Gaussian blur the raw map so the heatmap reads as a region, not pixels.
        heatmap_png, overlay_png = _render(prepared.rgb, anomaly_map, mask, threshold)
        original_png = encode_png(prepared.rgb)

        caveats = list(prepared.warnings)
        if self.category is None:
            caveats.append(
                "No category was supplied; the single available memory bank was used. "
                "Supply `category` for unambiguous results."
            )
        caveats.append(
            "Scores are distances to the learned good-image memory bank. They indicate "
            "unfamiliarity, not a confirmed defect type."
        )

        verdict = "defect_detected" if is_anomalous else "within_tolerance"
        message = (
            f"Anomalous region covering {localisation:.2f}% of the part was located with a "
            f"peak score of {score:.3f} against a {threshold:.3f} threshold. Review the "
            "heatmap and overlay before disposition."
            if is_anomalous else
            f"No region exceeded the {threshold:.3f} threshold (peak {score:.3f}). "
            "The part matches the learned appearance of this category."
        )

        return InspectionResult(
            model_trained=True,
            verdict=verdict,
            message=message,
            category=self.category,
            model_version=(self.manifest or {}).get("version"),
            is_anomalous=is_anomalous,
            anomaly_score=round(score, 6),
            threshold=round(threshold, 6),
            num_anomalous_patches=patch_count,
            localisation_pct=round(localisation, 4),
            anomaly_map=anomaly_map,
            heatmap_png=heatmap_png,
            overlay_png=overlay_png,
            original_png=original_png,
            preprocessing=prepared.to_metadata(),
            model_metrics=(self.manifest or {}).get("metrics"),
            caveats=caveats,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


def _render(
    rgb: np.ndarray, anomaly_map: np.ndarray, mask: np.ndarray, threshold: float
) -> tuple[bytes, bytes]:
    """Produce a standalone heatmap PNG and an annotated overlay PNG."""
    import cv2

    smoothed = cv2.GaussianBlur(anomaly_map.astype(np.float32), (0, 0), sigmaX=3.0)
    # Normalise against the 99th percentile so a single hot pixel does not wash
    # out the colour scale, then clamp.
    scale = float(np.percentile(smoothed, 99.5)) or 1.0
    normalised = np.clip(smoothed / scale, 0.0, 1.0)
    heat = cv2.applyColorMap((normalised * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    overlay = rgb.astype(np.float32).copy()
    dilated = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    overlay[dilated > 0] = (0.55 * overlay[dilated > 0] + 0.45 * heat[dilated > 0])

    # Contour outline for unambiguous localisation.
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 64, 32), 2)

    banner = np.zeros((26, rgb.shape[1], 3), dtype=np.float32)
    verdict_colour = (255, 80, 40) if mask.any() else (40, 220, 160)
    banner[:] = verdict_colour
    combined = np.concatenate([banner, overlay], axis=0).astype(np.uint8)

    del threshold  # threshold already encoded in the colour decision
    return encode_png(heat), encode_png(combined)


def validate_image_upload(filename: str, data: bytes) -> None:
    load_and_prepare(data, filename, image_size=settings.vision_image_size)


__all__ = ["VisionInference", "InspectionResult", "InvalidImage", "validate_image_upload"]
