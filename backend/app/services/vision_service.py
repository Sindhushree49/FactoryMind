"""Visual inspection service: run the model, persist artefacts, build the report."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging_config import get_logger
from app.ml.computer_vision.features import InvalidImage
from app.ml.computer_vision.inference import InspectionResult, VisionInference
from app.models import VisualInspection

logger = get_logger(__name__)


def inspect(
    session: Session,
    filename: str,
    data: bytes,
    category: str | None = None,
    machine_id: str | None = None,
    persist: bool = True,
) -> tuple[InspectionResult, dict[str, Any]]:
    inference = VisionInference(category=category)
    result = inference.inspect(data, filename, category=category)

    # Reject an undecodable upload even when no model is trained: the API must not
    # silently accept garbage.
    if result.model_trained or result.verdict != "model_not_trained":
        from app.ml.computer_vision.features import validate_upload

        validate_upload(filename, data)

    paths = _store_artifacts(result, filename) if result.model_trained else {}

    record: VisualInspection | None = None
    if persist:
        record = VisualInspection(
            id=uuid.uuid4().hex,
            machine_id=machine_id,
            original_filename=filename,
            stored_path=paths.get("original", ""),
            category=result.category,
            model_version=result.model_version,
            model_trained=result.model_trained,
            is_anomalous=result.is_anomalous,
            anomaly_score=result.anomaly_score,
            threshold=result.threshold,
            num_anomalous_patches=result.num_anomalous_patches,
            localisation_pct=result.localisation_pct,
            heatmap_path=paths.get("heatmap"),
            overlay_path=paths.get("overlay"),
            status="completed" if result.model_trained else "skipped_untrained",
            message=result.message,
        )
        session.add(record)
        session.flush()

    payload = {
        "inspection_id": record.id if record else None,
        "model_version": result.model_version,
        "model_trained": result.model_trained,
        "is_anomalous": result.is_anomalous,
        "anomaly_score": result.anomaly_score,
        "threshold": result.threshold,
        "num_anomalous_patches": result.num_anomalous_patches,
        "localisation_pct": result.localisation_pct,
        "category": result.category,
        "verdict": result.verdict,
        "message": result.message,
        "preprocessing": result.preprocessing,
        "caveats": result.caveats,
        "elapsed_ms": round(result.elapsed_ms, 2),
        "original_url": _url(paths.get("original")),
        "heatmap_url": _url(paths.get("heatmap")),
        "overlay_url": _url(paths.get("overlay")),
        "trained_categories": inference.trained_categories,
        "dataset": "MVTec AD",
        "model_metrics": result.model_metrics,
    }
    return result, payload


def _store_artifacts(result: InspectionResult, filename: str) -> dict[str, str]:
    settings.vision_outputs_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:12]
    stem = Path(filename).stem[:40]
    paths: dict[str, str] = {}

    from app.ml.computer_vision.features import encode_png

    if result.original_png:
        original = settings.vision_outputs_dir / f"{stem}_{token}_original.png"
        original.write_bytes(result.original_png)
        paths["original"] = str(original)
    if result.heatmap_png:
        heatmap = settings.vision_outputs_dir / f"{stem}_{token}_heatmap.png"
        heatmap.write_bytes(result.heatmap_png)
        paths["heatmap"] = str(heatmap)
    if result.overlay_png:
        overlay = settings.vision_outputs_dir / f"{stem}_{token}_overlay.png"
        overlay.write_bytes(result.overlay_png)
        paths["overlay"] = str(overlay)
    return paths


def _url(path: str | None) -> str | None:
    if not path:
        return None
    try:
        relative = Path(path).resolve().relative_to(settings.storage_dir.resolve())
    except ValueError:
        return None
    return f"/storage/{relative.as_posix()}"


def report_text(result: InspectionResult, payload: dict[str, Any]) -> str:
    """Human-readable inspection report."""
    lines = [
        "FACTORYMIND - VISUAL INSPECTION REPORT",
        "=" * 60,
        f"Model        : {payload.get('model_version') or 'not trained'}",
        f"Dataset      : {payload.get('dataset')}",
        f"Category     : {payload.get('category') or 'unspecified'}",
        f"Verdict      : {payload.get('verdict')}",
        f"Anomalous    : {payload.get('is_anomalous')}",
        f"Score / limit: {payload.get('anomaly_score')} / {payload.get('threshold')}",
        f"Localisation : {payload.get('localisation_pct')}% of the part",
        "",
        "FINDING",
        "-" * 60,
        payload.get("message", ""),
    ]
    preprocessing = payload.get("preprocessing") or {}
    if preprocessing:
        lines += [
            "",
            "PREPROCESSING",
            "-" * 60,
            f"Original size : {preprocessing.get('original_size')}",
            f"Resized to    : {preprocessing.get('resized_to')}",
            f"Format        : {preprocessing.get('format')}",
        ]
        for warning in preprocessing.get("warnings", []):
            lines.append(f"Note          : {warning}")
    if payload.get("caveats"):
        lines += ["", "CAVEATS", "-" * 60, *[f"- {c}" for c in payload["caveats"]]]
    lines += [
        "",
        "DISPOSITION",
        "-" * 60,
        "Automated inspection is an input to a human decision. Quarantine the part for",
        "physical verification before accepting or rejecting it.",
    ]
    return "\n".join(lines)


__all__ = ["inspect", "report_text", "InvalidImage"]
