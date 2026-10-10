"""Machine registry and telemetry endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, status
from sqlalchemy import select

from app.api.deps import DbSession
from app.core.logging_config import get_logger
from app.ml.preprocessing import ai4i
from app.models import Machine, SensorReading
from app.schemas.api import (
    MachineCreate,
    MachineList,
    MachineRead,
    MachineUpdate,
    SensorReadingCreate,
    SensorReadingRead,
)
from app.services import dashboard_service

logger = get_logger(__name__)
router = APIRouter(prefix="/machines", tags=["machines"])


def _to_read(session: DbSession, machine: Machine) -> MachineRead:
    reading = dashboard_service.latest_reading(session, machine.id)
    return MachineRead(
        id=machine.id,
        asset_id=machine.asset_id,
        name=machine.name,
        model=machine.model,
        machine_type=machine.machine_type,
        location=machine.location,
        rated_rpm=machine.rated_rpm,
        rated_torque_nm=machine.rated_torque_nm,
        is_active=machine.is_active,
        created_at=machine.created_at,
        latest_reading=(
            SensorReadingRead.model_validate(reading) if reading else None
        ),
        health=dashboard_service.machine_health(session, machine.id),
    )


@router.get("", response_model=MachineList, summary="List machines")
def list_machines(
    session: DbSession,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    active_only: bool = Query(default=False),
) -> MachineList:
    machines, total = dashboard_service.list_machines(
        session, limit=limit, offset=offset, active_only=active_only
    )
    return MachineList(
        items=[_to_read(session, m) for m in machines], total=total, limit=limit, offset=offset
    )


@router.post(
    "", response_model=MachineRead, status_code=status.HTTP_201_CREATED, summary="Register a machine"
)
def create_machine(session: DbSession, payload: MachineCreate) -> MachineRead:
    try:
        machine = dashboard_service.create_machine(session, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return _to_read(session, machine)


@router.get("/{machine_id}", response_model=MachineRead, summary="Get one machine")
def get_machine(machine_id: str, session: DbSession) -> MachineRead:
    machine = dashboard_service.get_machine(session, machine_id)
    if machine is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Machine '{machine_id}' not found")
    return _to_read(session, machine)


@router.patch("/{machine_id}", response_model=MachineRead, summary="Update a machine")
def update_machine(machine_id: str, session: DbSession, payload: MachineUpdate) -> MachineRead:
    machine = dashboard_service.get_machine(session, machine_id)
    if machine is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Machine '{machine_id}' not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(machine, key, value)
    session.add(machine)
    session.flush()
    return _to_read(session, machine)


@router.get(
    "/{machine_id}/readings", response_model=list[SensorReadingRead], summary="Sensor history"
)
def machine_readings(
    machine_id: str,
    session: DbSession,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[SensorReadingRead]:
    if dashboard_service.get_machine(session, machine_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Machine '{machine_id}' not found")
    rows = session.scalars(
        select(SensorReading)
        .where(SensorReading.machine_id == machine_id)
        .order_by(SensorReading.recorded_at.desc())
        .limit(limit)
    )
    return [SensorReadingRead.model_validate(row) for row in rows]


@router.post(
    "/{machine_id}/readings",
    response_model=SensorReadingRead,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a sensor reading",
)
def add_reading(
    machine_id: str, session: DbSession, payload: SensorReadingCreate
) -> SensorReadingRead:
    if dashboard_service.get_machine(session, machine_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Machine '{machine_id}' not found")
    reading = SensorReading(
        machine_id=machine_id,
        air_temperature_k=payload.air_temperature_k,
        process_temperature_k=payload.process_temperature_k,
        rotational_speed_rpm=payload.rotational_speed_rpm,
        torque_nm=payload.torque_nm,
        tool_wear_min=payload.tool_wear_min,
        source="api",
    )
    session.add(reading)
    session.flush()
    logger.info("Stored reading for machine %s", machine_id)
    return SensorReadingRead.model_validate(reading)


@router.post(
    "/seed",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="Register machines from real AI4I rows",
)
def seed(
    session: DbSession, count: int = Body(default=8, embed=True, ge=1, le=40)
) -> dict[str, Any]:
    """Registers machines and replays verbatim AI4I rows as their history.

    No sensor value is synthesised: every reading comes from the published
    dataset. The endpoint is refused if the dataset is absent.
    """
    try:
        created = dashboard_service.seed_machines_from_dataset(session, count=count)
    except ai4i.AI4IDatasetMissing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "dataset_unavailable",
                "message": "AI4I 2020 is not present, so no machines were seeded.",
                "remediation": "python backend/scripts/download_datasets.py --dataset ai4i",
            },
        )
    if not created:
        return {
            "created": [],
            "count": 0,
            "source": "AI4I 2020 (UCI dataset 601)",
            "note": "All requested machines were already registered; nothing was re-seeded.",
        }
    return {
        "created": [m.asset_id for m in created],
        "count": len(created),
        "source": "AI4I 2020 (UCI dataset 601)",
        "note": "Sensor history is verbatim AI4I telemetry; nothing is synthesised.",
    }
