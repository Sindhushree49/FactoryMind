"""Shared dataset loading and version reporting.

Every dataset is treated as an independent asset with its own loader. Nothing
in this module merges, imputes across, or synthesises data from one dataset
into another.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DatasetStatus:
    """Availability report for one dataset."""

    key: str
    title: str
    path: Path
    available: bool
    detail: str
    files: list[str] = field(default_factory=list)
    bytes_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "path": str(self.path),
            "available": self.available,
            "detail": self.detail,
            "file_count": len(self.files),
            "bytes": self.bytes_total,
            "files": self.files[:40],
        }


def _scan(path: Path, suffixes: tuple[str, ...]) -> tuple[list[str], int]:
    if not path.exists():
        return [], 0
    files, total = [], 0
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix.lower() in suffixes:
            files.append(str(item.relative_to(path)))
            total += item.stat().st_size
    return files, total


def ai4i_status() -> DatasetStatus:
    csvs, size = _scan(settings.ai4i_dir, (".csv",))
    if not csvs:
        return DatasetStatus(
            "ai4i", "AI4I 2020 Predictive Maintenance", settings.ai4i_dir, False,
            "ai4i2020.csv not found. Run backend/scripts/download_datasets.py --dataset ai4i",
        )
    return DatasetStatus(
        "ai4i", "AI4I 2020 Predictive Maintenance", settings.ai4i_dir, True,
        f"{len(csvs)} CSV file(s) present", csvs, size,
    )


def cmapss_status() -> DatasetStatus:
    txts, size = _scan(settings.cmapss_dir, (".txt",))
    subsets = sorted({f.split("_")[1].split(".")[0] for f in txts if "_FD" in f})
    available = bool(subsets)
    return DatasetStatus(
        "cmapss", "NASA C-MAPSS Turbofan", settings.cmapss_dir, available,
        f"subsets: {', '.join(subsets)}" if available else
        "C-MAPSS *.txt not found. Run backend/scripts/download_datasets.py --dataset cmapss",
        txts, size,
    )


def mvtec_status() -> DatasetStatus:
    root = settings.mvtec_dir
    images, size = _scan(root, (".png", ".jpg", ".jpeg", ".bmp"))
    if root.exists() and not images:
        return DatasetStatus(
            "mvtec", "MVTec AD", root, False,
            "MVTec AD folder exists but contains no images. Re-extract the archive.",
        )
    categories = sorted({
        os.path.relpath(p, root).split(os.sep)[0] for p in images
    })
    return DatasetStatus(
        "mvtec", "MVTec AD", root, bool(images),
        f"{len(categories)} categories, {len(images)} images" if images else
        "MVTec AD not present. Licence-gated dataset - see README manual setup steps.",
        images, size,
    )


def all_statuses() -> dict[str, DatasetStatus]:
    return {"ai4i": ai4i_status(), "cmapss": cmapss_status(), "mvtec": mvtec_status()}


def manifest() -> dict[str, Any]:
    """Provenance manifest written by the downloader, if present."""
    path = settings.manifest_path
    if not path.exists():
        return {"available": False, "detail": "datasets/manifests/datasets.json not generated yet"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Corrupt dataset manifest at %s: %s", path, exc)
        return {"available": False, "detail": f"corrupt manifest: {exc}"}
