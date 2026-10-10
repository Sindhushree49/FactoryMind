"""Computer vision endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status

from app.api.deps import DbSession
from app.core.config import settings
from app.core.logging_config import get_logger
from app.ml.computer_vision.features import InvalidImage
from app.schemas.api import VisualInspectionRequest, VisualInspectionResponse
from app.services import ml_service, vision_service

logger = get_logger(__name__)
router = APIRouter(tags=["vision"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@router.post(
    "/vision/inspect",
    response_model=VisualInspectionResponse,
    summary="Inspect an uploaded image for defects",
)
async def inspect_image(
    session: DbSession,
    file: UploadFile = File(..., description="Product image to inspect"),
    category: str | None = Form(default=None, description="MVTec category, e.g. 'bottle'"),
    machine_id: str | None = Form(default=None),
    persist: bool = Form(default=True),
) -> VisualInspectionResponse:
    """Runs the PatchCore detector and returns a localisation heatmap.

    If no memory bank is trained for the requested category the response is
    ``model_trained: false`` with remediation instructions. No anomaly score is
    fabricated in that case.
    """
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image is {len(data) / 1e6:.1f} MB; the limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB.",
        )

    try:
        result, payload = vision_service.inspect(
            session,
            filename=file.filename or "upload.png",
            data=data,
            category=category,
            machine_id=machine_id,
            persist=persist,
        )
    except InvalidImage as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    from datetime import datetime, timezone

    return VisualInspectionResponse(**payload, created_at=datetime.now(timezone.utc))


@router.get("/vision/categories", summary="Categories with a trained memory bank")
def vision_categories() -> dict[str, Any]:
    status_payload = ml_service.vision_status()
    return {
        "trained_categories": status_payload.get("trained_categories", []),
        "loaded": status_payload.get("loaded", False),
        "detail": status_payload.get("detail"),
        "dataset": "MVTec AD (CC BY-NC-SA 4.0; manual registration required)",
        "dataset_path": str(settings.mvtec_dir),
        "model": "PatchCore (ResNet-18 layer2 patch memory bank, coreset subsampled)",
        "expected_categories": [
            "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
            "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
        ],
    }


@router.get("/vision/report/{inspection_id}", summary="Text inspection report")
def inspection_report(
    inspection_id: str, session: DbSession, response: Response
) -> dict[str, Any]:
    from app.models import VisualInspection

    record = session.get(VisualInspection, inspection_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Inspection '{inspection_id}' not found"
        )
    payload = {
        "inspection_id": record.id,
        "model_version": record.model_version,
        "model_trained": record.model_trained,
        "is_anomalous": record.is_anomalous,
        "anomaly_score": record.anomaly_score,
        "threshold": record.threshold,
        "localisation_pct": record.localisation_pct,
        "category": record.category,
        "verdict": record.status,
        "message": record.message,
        "dataset": "MVTec AD",
    }
    from app.ml.computer_vision.inference import InspectionResult

    result = InspectionResult(
        model_trained=record.model_trained,
        verdict=record.status,
        message=record.message or "",
        category=record.category,
        model_version=record.model_version,
        is_anomalous=record.is_anomalous,
        anomaly_score=record.anomaly_score,
        threshold=record.threshold,
        localisation_pct=record.localisation_pct,
    )
    text = vision_service.report_text(result, payload)
    if not record.model_trained:
        response.status_code = status.HTTP_200_OK
    return {"inspection_id": inspection_id, "format": "text/plain", "report": text}


@router.post("/vision/sample", summary="Fetch one MVTec test image for manual upload")
def sample_image(
    session: DbSession, category: str, kind: str = "test", index: int = 0
) -> dict[str, Any]:
    """Returns a stored sample so the UI can demonstrate inspection without a local file."""
    from pathlib import Path

    root = settings.mvtec_dir / category / "test" / kind
    if not root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MVTec AD sample not found at {root}. Install the dataset first.",
        )
    images = sorted(root.glob("*.png"))
    if not images:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No images in that folder")
    chosen = Path(images[index % len(images)])
    return {
        "category": category,
        "kind": kind,
        "filename": chosen.name,
        "url": f"/storage/mvtec_samples/{chosen.relative_to(settings.mvtec_dir).as_posix()}",
        "note": "Served only when the MVTec AD dataset is installed locally.",
    }


__all__ = ["router", "VisualInspectionRequest"]
