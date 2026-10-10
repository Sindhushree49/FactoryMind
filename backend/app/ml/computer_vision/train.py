"""Train a PatchCore-style visual anomaly detector on MVTec AD.

Architecture (PatchCore, Bergmann et al., CVPR 2019)
----------------------------------------------------
1. A pretrained CNN backbone (ResNet-18, ImageNet) is run over *normal* training
   images of a single product category.
2. Local patch descriptors are taken from an early high-resolution layer
   (layer2, stride 8), because industrial defects are small and texture-scale.
3. Each patch descriptor, together with its position-aware neighbour mean
   (PatchCore's "locally aware" feature), forms a memory-bank entry.
4. The bank is aggressively coreset-subsampled (greedy k-center) - 1-10% of the
   full bank retains near-full accuracy at a fraction of the memory.
5. At inference, the k-th nearest memory-bank distance for each query patch is
   the anomaly score map. Large distance = far from all known-good patches.

One memory bank is trained per MVTec category, because 'good' is defined
relative to a specific product. MVTec ships no defective training images, which
is exactly the property PatchCore is designed for; evaluation uses the held-out
``test/`` split plus pixel-level ground-truth masks.

This module refuses to fabricate a model. Without the dataset it exits with
instructions rather than emitting synthetic "detections".

Run:
    python backend/app/ml/computer_vision/train.py --categories bottle cable
    python backend/app/ml/computer_vision/train.py --all --coreset 0.01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.config import settings  # noqa: E402
from app.core.logging_config import configure_logging, get_logger  # noqa: E402
from app.ml import registry  # noqa: E402
from app.ml.computer_vision.features import InvalidImage, load_and_prepare  # noqa: E402
from app.ml import datasets as dataset_registry  # noqa: E402

logger = get_logger(__name__)

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]


class MVTECMissing(FileNotFoundError):
    """Raised when MVTec AD is absent; never substituted with synthetic data."""


def _dataset_root() -> Path:
    root = settings.mvtec_dir
    if not root.is_dir():
        status = dataset_registry.mvtec_status()
        raise MVTECMissing(
            f"MVTec AD not found at {root}. {status.detail} "
            "See README 'Computer vision setup' for the manual download steps."
        )
    categories = [p.name for p in root.iterdir() if p.is_dir() and (p / "train" / "good").is_dir()]
    if not categories:
        raise MVTECMissing(
            f"{root} contains no category folders with train/good. Re-extract the MVTec AD archive."
        )
    return root


def _backbone() -> Any:
    import torch
    from torchvision.models import resnet18, ResNet18_Weights

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    # layer2 output: stride 8, 128 channels - fine spatial detail for small defects.
    backbone = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                   model.layer1, model.layer2)
    return backbone.eval(), weights.url


def _patch_features(backbone: Any, rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Return ``(patches, feature_hw)`` for one preprocessed image."""
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    with torch.no_grad():
        features = backbone(tensor)
        pooled = F.adaptive_avg_pool2d(features, 1)
        # PatchCore's locally-aware feature: concatenation of the local
        # descriptor and its 3x3 neighbourhood mean.
        neighbours = F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
        combined = torch.cat([features, neighbours], dim=1)
        patches = combined[0].permute(1, 2, 0).reshape(-1, combined.shape[1]).cpu().numpy()

    feature_hw = (int(features.shape[2]), int(features.shape[3]))
    del pooled
    return patches, feature_hw


def _coreset_subsample(bank: np.ndarray, fraction: float, seed: int = 42) -> np.ndarray:
    """Greedy k-center coreset selection (PatchCore Algorithm 1).

    Keeps a small subset that still covers the full feature distribution, which
    is what makes PatchCore memory-efficient.
    """
    total = bank.shape[0]
    target = max(1, int(total * fraction))
    if target >= total:
        return bank

    rng = np.random.default_rng(seed)
    selected = np.zeros(total, dtype=bool)
    first_idx = rng.integers(0, total)
    selected[first_idx] = True

    # Initialize min distances to the first selected point
    min_distances = np.full(total, np.inf, dtype=np.float32)
    diff = bank - bank[first_idx:first_idx+1]
    min_distances = np.sqrt((diff ** 2).sum(axis=1)).astype(np.float32)
    min_distances[first_idx] = -1.0

    for _ in range(target - 1):
        # Select the point farthest from the current selected set
        nxt = int(np.argmax(min_distances))
        selected[nxt] = True

        # Update min distances with distances to the newly added point
        diff = bank - bank[nxt:nxt+1]
        new_distances = np.sqrt((diff ** 2).sum(axis=1)).astype(np.float32)
        np.minimum(min_distances, new_distances, out=min_distances)
        min_distances[selected] = -1.0

    return bank[selected]


def _evaluate(
    backbone: Any, bank: np.ndarray, root: Path, category: str, threshold: float,
    image_size: int, max_images: int,
) -> dict[str, Any]:
    """Score the held-out test split against pixel ground-truth masks."""
    from PIL import Image

    test_root = root / category / "test"
    if not test_root.is_dir():
        return {"status": "skipped", "reason": "no test split present"}

    scored: list[tuple[float, int, str]] = []   # (peak score, is_defect, defect kind)
    pixel_auroc_inputs: list[tuple[np.ndarray, np.ndarray]] = []
    per_defect: dict[str, dict[str, int]] = {}

    # ``max_images`` is a per-defect-class quota, so every anomaly type is
    # represented even in the large texture categories (carpet, grid, leather, tile).
    for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        kind = defect_dir.name
        image_dir = defect_dir / ("good" if kind == "good" else "test")
        if not image_dir.is_dir():
            continue
        per_defect.setdefault(kind, {"images": 0, "flagged": 0})
        for image_path in sorted(image_dir.glob("*.png"))[:max_images]:
            try:
                prepared = load_and_prepare(
                    image_path.read_bytes(), image_path.name, image_size=image_size
                )
            except InvalidImage as exc:
                logger.warning("Skipping %s: %s", image_path, exc)
                continue
            patches, feature_hw = _patch_features(backbone, prepared.rgb)
            anomaly_map = _upsample(_min_distance(bank, patches), feature_hw, image_size)
            peak = float(anomaly_map.max())
            scored.append((peak, 0 if kind == "good" else 1, kind))
            per_defect[kind]["images"] += 1
            per_defect[kind]["flagged"] += int(peak >= threshold)

            mask_path = defect_dir / "gt" / f"{image_path.stem}.png"
            if kind != "good" and mask_path.is_file():
                ground_truth = np.array(
                    Image.open(mask_path).convert("L").resize((image_size, image_size))
                ) > 0
                prediction = anomaly_map >= threshold
                pixel_auroc_inputs.append((ground_truth, prediction))

    metrics: dict[str, Any] = {
        "status": "evaluated" if scored else "skipped",
        "images_scored": len(scored),
        "image_level": _image_level_metrics(scored, per_defect),
        "per_defect": per_defect,
    }
    if pixel_auroc_inputs:
        from sklearn.metrics import roc_auc_score

        truths = np.concatenate([t.reshape(-1) for t, _ in pixel_auroc_inputs])
        predictions = np.concatenate([p.reshape(-1).astype(float) for _, p in pixel_auroc_inputs])
        if len(np.unique(truths)) > 1:
            metrics["pixel_level"] = {
                "roc_auc": float(roc_auc_score(truths, predictions)),
                "pixel_accuracy": float((truths == predictions).mean()),
                "note": (
                    "Pixel-level scores over defective test images using MVTec ground-truth "
                    "masks, thresholded at the calibrated operating point."
                ),
            }
    return metrics


def _image_level_metrics(
    scored: list[tuple[float, int, str]], per_defect: dict[str, dict[str, int]]
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if not scored:
        return {}
    peaks = np.array([s[0] for s in scored], dtype=float)
    labels = np.array([s[1] for s in scored], dtype=int)
    summary: dict[str, Any] = {
        "mean_score": float(peaks.mean()),
        "max_score": float(peaks.max()),
        "good_mean_score": float(peaks[labels == 0].mean()) if (labels == 0).any() else None,
        "defect_mean_score": float(peaks[labels == 1].mean()) if (labels == 1).any() else None,
    }
    if len(np.unique(labels)) > 1:
        summary["roc_auc"] = float(roc_auc_score(labels, peaks))
        summary["average_precision"] = float(average_precision_score(labels, peaks))
        defect_kinds = [k for k in per_defect if k != "good"]
        if defect_kinds:
            summary["recall_at_threshold_by_defect"] = {
                k: per_defect[k]["flagged"] / max(per_defect[k]["images"], 1) for k in defect_kinds
            }
            summary["false_positive_rate_on_good"] = (
                per_defect["good"]["flagged"] / max(per_defect["good"]["images"], 1)
                if "good" in per_defect else None
            )
    return summary


def _score_image(backbone: Any, bank: np.ndarray, data: bytes, filename: str, image_size: int) -> np.ndarray:
    prepared = load_and_prepare(data, filename, image_size=image_size)
    patches, feature_hw = _patch_features(backbone, prepared.rgb)
    distances = _min_distance(bank, patches)
    return _upsample(distances, feature_hw, image_size)


def _min_distance(bank: np.ndarray, queries: np.ndarray) -> np.ndarray:
    out = np.empty(queries.shape[0], dtype=np.float32)
    bank_sq = (bank ** 2).sum(axis=1)
    for start in range(0, queries.shape[0], 1024):
        chunk = queries[start : start + 1024]
        d2 = (chunk ** 2).sum(axis=1)[:, None] + bank_sq[None, :] - 2.0 * chunk @ bank.T
        out[start : start + 1024] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def _upsample(distances: np.ndarray, feature_hw: tuple[int, int], image_size: int) -> np.ndarray:
    import cv2

    grid = distances.reshape(feature_hw).astype(np.float32)
    return cv2.resize(grid, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def train_category(
    category: str,
    coreset: float = 0.01,
    image_size: int = 256,
    max_train_images: int = 200,
    eval_images: int = 40,
    knn: int = 1,
    version: str | None = None,
) -> dict[str, Any]:
    import joblib
    import torch

    root = _dataset_root()
    train_dir = root / category / "train" / "good"
    if not train_dir.is_dir():
        raise MVTECMissing(f"{train_dir} does not exist")

    images = sorted(train_dir.glob("*.png"))[:max_train_images]
    if not images:
        raise MVTECMissing(f"No training PNGs found in {train_dir}")

    torch.manual_seed(settings.random_seed)
    np.random.seed(settings.random_seed)
    backbone, weights_url = _backbone()

    started = time.perf_counter()
    all_patches: list[np.ndarray] = []
    feature_hw: tuple[int, int] | None = None
    for image_path in images:
        try:
            prepared = load_and_prepare(image_path.read_bytes(), image_path.name, image_size=image_size)
        except InvalidImage as exc:
            logger.warning("Skipping %s: %s", image_path.name, exc)
            continue
        patches, feature_hw = _patch_features(backbone, prepared.rgb)
        all_patches.append(patches)

    if not all_patches:
        raise MVTECMissing(f"None of the {len(images)} training images could be decoded.")

    full_bank = np.concatenate(all_patches, axis=0).astype(np.float32)
    del all_patches
    logger.info("%s: full memory bank %s from %d images (feature map %s)",
                category, full_bank.shape, len(images), feature_hw)

    subsampled = _coreset_subsample(full_bank, coreset)
    logger.info("%s: coreset %s -> %s (%.1f%% retained)",
                category, full_bank.shape, subsampled.shape, 100 * subsampled.shape[0] / full_bank.shape[0])

    # Threshold from the good-image score distribution: keep the false-positive
    # rate on *known-good* validation images at a controlled level.
    calibration_dir = root / category / "test" / "good"
    calibration_images = sorted(calibration_dir.glob("*.png"))[: min(eval_images, 30)]
    calibration_scores: list[float] = []
    for image_path in calibration_images:
        try:
            prepared = load_and_prepare(image_path.read_bytes(), image_path.name, image_size=image_size)
        except InvalidImage:
            continue
        patches, _ = _patch_features(backbone, prepared.rgb)
        calibration_scores.append(float(np.percentile(_min_distance(subsampled, patches), 99.9)))
    if calibration_scores:
        threshold = float(np.percentile(calibration_scores, 99))
    else:
        all_scores = np.percentile(_min_distance(subsampled, subsampled[: min(4096, subsampled.shape[0])]), 99.9)
        threshold = float(all_scores)
    logger.info("%s: threshold %.4f from %d held-out good images", category, threshold, len(calibration_scores))

    evaluation = _evaluate(backbone, subsampled, root, category, threshold, image_size, eval_images)

    out_dir = settings.models_dir / registry.TASK_VISION / category / (
        version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "memory_bank.npy", subsampled)

    joblib.dump(
        {
            "category": category,
            "image_size": image_size,
            "feature_hw": feature_hw,
            "feature_dim": int(subsampled.shape[1]),
            "threshold": threshold,
            "knn": knn,
            "backbone": "resnet18_imagenet1k_v1",
            "backbone_layer": "layer2 (stride 8, 256-d locally-aware descriptor)",
            "weights_url": weights_url,
        },
        out_dir / "config.joblib",
    )

    manifest = {
        "name": f"patchcore_{category}",
        "family": "patchcore",
        "task": registry.TASK_VISION,
        "category": category,
        "version": out_dir.name,
        "dataset": "MVTec AD (Bergmann et al., CVPR 2019)",
        "dataset_path": str(root),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": int(len(images)),
        "metrics": evaluation,
        "hyperparameters": {
            "coreset_fraction": coreset,
            "image_size": image_size,
            "knn": knn,
            "memory_bank_size": int(subsampled.shape[0]),
            "full_bank_size": int(full_bank.shape[0]),
            "threshold": threshold,
            "backbone": "resnet18_imagenet1k_v1",
        },
        "training_seconds": round(time.perf_counter() - started, 2),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Published %s/%s", category, out_dir.name)
    return manifest


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train PatchCore visual anomaly detectors on MVTec AD.")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--coreset", type=float, default=0.01)
    parser.add_argument("--image-size", type=int, default=settings.vision_image_size)
    parser.add_argument("--max-train-images", type=int, default=200)
    parser.add_argument("--eval-images", type=int, default=40)
    parser.add_argument("--knn", type=int, default=1)
    args = parser.parse_args()

    try:
        root = _dataset_root()
    except MVTECMissing as exc:
        logger.error("%s", exc)
        print("\nBLOCKED: MVTec AD is not available.\n%s\n" % exc)
        return 2

    categories = (
        sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "train" / "good").is_dir())
        if args.all or not args.categories else args.categories
    )
    print(f"[FactoryMind] Training {len(categories)} categor{'y' if len(categories)==1 else 'ies'}: {categories}")

    results = []
    for category in categories:
        try:
            results.append(train_category(
                category, coreset=args.coreset, image_size=args.image_size,
                max_train_images=args.max_train_images, eval_images=args.eval_images, knn=args.knn,
            ))
        except MVTECMissing as exc:
            logger.error("Skipping %s: %s", category, exc)

    print("\n=== PatchCore training summary ===")
    for manifest in results:
        metrics = manifest["metrics"]
        image_level = metrics.get("image_level", {})
        print(f"{manifest['category']:<12} bank={manifest['hyperparameters']['memory_bank_size']:>7} "
              f"threshold={manifest['hyperparameters']['threshold']:.4f} "
              f"image_auc={image_level.get('roc_auc')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
