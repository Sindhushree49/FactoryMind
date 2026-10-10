"""ML inference endpoints: failure prediction, anomaly detection, RUL."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.api.deps import DbSession, ModelNotTrained, model_not_trained, ml_service
from app.core.logging_config import get_logger
from app.models import AnomalyEvent, FailurePrediction, RULPrediction
from app.schemas.api import (
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    FailurePredictionRequest,
    FailurePredictionResponse,
    RULRequest,
    RULResponse,
)

logger = get_logger(__name__)
router = APIRouter(tags=["ml"])


@router.post(
    "/predict/failure",
    response_model=FailurePredictionResponse,
    summary="Predict machine failure from sensor readings",
    responses={503: {"description": "The failure model has not been trained."}},
)
def predict_failure(
    session: DbSession, payload: FailurePredictionRequest
) -> FailurePredictionResponse:
    """Runs the AI4I-trained classifier and returns a SHAP explanation.

    Temperatures are in Kelvin, matching the AI4I schema. The 300 K / 310 K
    nominal values in the UI are the dataset's own operating point, not Celsius.
    """
    model = ml_service.failure_model()
    if not model.ready:
        raise model_not_trained(ModelNotTrained("failure_prediction"))

    request = payload.model_dump(exclude={"persist", "shap_top_k", "is_simulation"})
    try:
        result = model.predict(request, shap_top_k=payload.shap_top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failure prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        ) from exc

    prediction_id: str | None = None
    if payload.persist:
        record = FailurePrediction(
            machine_id=payload.machine_id,
            model_version=result["model_version"],
            failure_probability=result["failure_probability"],
            risk_category=result["risk_category"],
            predicted_failure_type=result.get("predicted_failure_type"),
            predicted_failure_type_confidence=result.get("predicted_failure_type_confidence"),
            shap_values={f["feature"]: f["contribution"] for f in result["features"]},
            top_features=result["features"],
            explanation=result["explanation"],
            input_payload=request,
            is_simulated=payload.is_simulation,
        )
        session.add(record)
        session.flush()
        prediction_id = record.id

    # ``result`` carries an ``is_simulation`` placeholder that is always
    # False here (the route excludes it from the inference payload), so the
    # authoritative value from the request is passed explicitly instead.
    return FailurePredictionResponse(
        **{k: v for k, v in result.items() if k != "is_simulation"},
        prediction_id=prediction_id,
        is_simulation=payload.is_simulation,
        created_at=datetime.now(timezone.utc),
    )


@router.post(
    "/detect/anomaly",
    response_model=AnomalyDetectionResponse,
    summary="Score a reading for abnormality",
    responses={503: {"description": "The anomaly model has not been trained."}},
)
def detect_anomaly(session: DbSession, payload: AnomalyDetectionRequest) -> AnomalyDetectionResponse:
    model = ml_service.anomaly_model()
    if not model.ready:
        raise model_not_trained(ModelNotTrained("anomaly_detection"))

    request = payload.model_dump(exclude={"persist", "threshold"})
    try:
        result = model.predict(request, threshold=payload.threshold)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Anomaly detection failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        ) from exc

    event_id: str | None = None
    if payload.persist:
        record = AnomalyEvent(
            machine_id=payload.machine_id,
            model_version=result["model_version"],
            anomaly_score=result["anomaly_score"],
            threshold=result["threshold"],
            is_anomaly=result["is_anomaly"],
            severity=result["severity"],
            contributing_features=result["feature_deviation"],
            raw_reading=request,
        )
        session.add(record)
        session.flush()
        event_id = record.id

    return AnomalyDetectionResponse(
        **result, anomaly_event_id=event_id, created_at=datetime.now(timezone.utc)
    )


@router.post(
    "/predict/rul",
    response_model=RULResponse,
    summary="Estimate remaining useful life",
    responses={503: {"description": "The RUL model has not been trained."}},
)
def predict_rul(session: DbSession, payload: RULRequest) -> RULResponse:
    """Estimates RUL for a C-MAPSS engine trajectory.

    If no ``window`` is supplied, the engine's own cycles from the published
    training split are used, so the endpoint works out of the box.
    """
    import numpy as np

    from app.ml.preprocessing import cmapss

    model = ml_service.rul_model(payload.model_family)
    if not model.ready:
        raise model_not_trained(ModelNotTrained("rul_prediction"))

    try:
        if payload.window is not None:
            window = np.asarray(payload.window, dtype=np.float32)
        else:
            if not cmapss.available_subsets():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error": "dataset_unavailable",
                        "message": "C-MAPSS is not present, so no engine history can be read.",
                        "remediation": "python backend/scripts/download_datasets.py --dataset cmapss",
                    },
                )
            subset = (
                payload.dataset_subset
                if payload.dataset_subset in cmapss.available_subsets()
                else cmapss.available_subsets()[0]
            )
            frame = cmapss.sequence_for_unit(subset, payload.engine_unit, up_to_cycle=payload.cycle)
            window = frame[model.sensors].to_numpy(dtype=np.float32)
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown engine unit: {exc}",
        ) from exc
    except cmapss.CMAPSSMissing as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    try:
        result = model.predict_window(
            window,
            subset=payload.dataset_subset,
            engine_unit=payload.engine_unit,
            cycle=payload.cycle,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid sensor window: {exc}",
        ) from exc

    prediction_id: str | None = None
    if payload.persist:
        record = RULPrediction(
            machine_id=None,
            engine_unit=result["engine_unit"],
            cycle=payload.cycle,
            model_version=result["model_version"],
            model_family=result["model_family"],
            dataset_subset=result["dataset_subset"],
            predicted_rul=result["predicted_rul"],
            lower_bound=result["lower_bound"],
            upper_bound=result["upper_bound"],
            uncertainty_pct=result["uncertainty_pct"],
            confidence_note=result["confidence_note"],
            degradation_curve=result["degradation_curve"],
        )
        session.add(record)
        session.flush()
        prediction_id = record.id

    return RULResponse(
        **{k: v for k, v in result.items() if k not in {"cycle", "caveats"}},
        cycle=payload.cycle,
        rul_prediction_id=prediction_id,
        created_at=datetime.now(timezone.utc),
    )


@router.get("/models/status", summary="Which models are loaded")
def models_status() -> dict[str, Any]:
    return ml_service.health()


@router.get("/models/trained", summary="Trained model versions with real metrics")
def trained_models(session: DbSession) -> dict[str, Any]:
    from app.services.dashboard_service import model_metrics

    return model_metrics(session)


@router.get("/models/metrics", summary="Model evaluation metrics")
def model_metrics_endpoint(session: DbSession) -> dict[str, Any]:
    """Returns metrics exactly as recorded at training time.

    These are computed on held-out splits (by engine unit for C-MAPSS, stratified
    for AI4I) and are never regenerated or estimated at request time.
    """
    from app.services.dashboard_service import model_metrics

    payload = model_metrics(session)
    if not payload["items"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "no_trained_models",
                "message": "No model has been trained yet, so no metrics exist.",
                "remediation": (
                    "python backend/app/ml/failure_prediction/train.py && "
                    "python backend/app/ml/anomaly_detection/train.py && "
                    "python backend/app/ml/rul_prediction/train.py"
                ),
            },
        )
    return payload


@router.post("/models/refresh", summary="Drop cached models and reload from disk")
def refresh_models() -> dict[str, Any]:
    ml_service.invalidate()
    return {"refreshed": True, **ml_service.health()}
