"""Digital twin scenario simulation.

A "what-if" run feeds hypothetical operating parameters through the *real*
trained models. The outputs are genuine model responses to hypothetical inputs -
they are not measurements of any machine, and every payload says so via
``is_simulation: true`` and an explicit disclaimer.
"""

from __future__ import annotations

from typing import Any

from app.core.logging_config import get_logger
from app.ml.preprocessing import ai4i
from app.services import ml_service

logger = get_logger(__name__)

SIMULATION_DISCLAIMER = (
    "SIMULATED RESULT. These parameters were entered by an operator, not measured from a "
    "physical machine. The outputs are real model evaluations of hypothetical inputs and "
    "must not be recorded as machine telemetry."
)

BASELINE = {
    "air_temperature_k": 300.5,
    "process_temperature_k": 310.6,
    "rotational_speed_rpm": 1500.0,
    "torque_nm": 40.0,
    "tool_wear_min": 0.0,
    "machine_type": "L",
}


def simulate(payload: dict[str, Any]) -> dict[str, Any]:
    machine_id = payload.get("machine_id")
    scenario = {
        "air_temperature_k": float(payload.get("air_temperature_k", BASELINE["air_temperature_k"])),
        "process_temperature_k": float(
            payload.get("process_temperature_k", BASELINE["process_temperature_k"])
        ),
        "rotational_speed_rpm": float(
            payload.get("rotational_speed_rpm", BASELINE["rotational_speed_rpm"])
        ),
        "torque_nm": float(payload.get("torque_nm", BASELINE["torque_nm"])),
        "tool_wear_min": float(payload.get("tool_wear_min", BASELINE["tool_wear_min"])),
        "machine_type": payload.get("machine_type") or BASELINE["machine_type"],
        "is_simulation": True,
    }

    frame = ai4i.add_features(
        __import__("pandas").DataFrame([{k: v for k, v in scenario.items()
                                         if k not in {"is_simulation"}}])
    )
    derived = {
        column: round(float(frame[column].iloc[0]), 4)
        for column in ai4i.ENGINEERED_COLUMNS
    }

    deltas = {
        key: round(scenario[key] - BASELINE[key], 4)
        for key in BASELINE
        if key != "machine_type" and key in scenario
    }

    result: dict[str, Any] = {
        "machine_id": machine_id,
        "is_simulation": True,
        "disclaimer": SIMULATION_DISCLAIMER,
        "scenario": scenario,
        "baseline": BASELINE,
        "deltas_from_baseline": deltas,
        "derived_features": derived,
        "failure": None,
        "anomaly": None,
        "comparison": None,
    }

    failure_model = ml_service.failure_model()
    if failure_model.ready:
        prediction = failure_model.predict(scenario)
        result["failure"] = prediction
        baseline_prediction = failure_model.predict({**BASELINE, "is_simulation": True})
        result["baseline_failure"] = baseline_prediction
        result["comparison"] = {
            "failure_probability": {
                "baseline": baseline_prediction["failure_probability"],
                "scenario": prediction["failure_probability"],
                "delta": round(
                    prediction["failure_probability"]
                    - baseline_prediction["failure_probability"], 6
                ),
            },
            "risk_category": {
                "baseline": baseline_prediction["risk_category"],
                "scenario": prediction["risk_category"],
            },
        }
    else:
        result["unavailable_reason"] = (
            "No failure model is trained, so the scenario cannot be evaluated. Train it "
            "with: python backend/app/ml/failure_prediction/train.py"
        )

    anomaly_model = ml_service.anomaly_model()
    if anomaly_model.ready:
        result["anomaly"] = anomaly_model.predict(scenario)
    else:
        result.setdefault(
            "unavailable_reason",
            "No anomaly model is trained, so the scenario cannot be scored. Train it with: "
            "python backend/app/ml/anomaly_detection/train.py",
        )

    return result


def health_snapshot(session: Any, machine_id: str) -> dict[str, Any]:
    """Real measured state for one machine, for the twin's live panel."""
    from app.ml.preprocessing import ai4i
    from app.services.dashboard_service import latest_reading, machine_health

    reading = latest_reading(session, machine_id)
    health = machine_health(session, machine_id)
    snapshot: dict[str, Any] = {
        "machine_id": machine_id,
        "is_simulation": False,
        "measured": reading is not None,
        "health": health,
        "readings": None,
        "note": None,
    }
    if reading is None:
        snapshot["note"] = (
            "No stored sensor readings for this machine. Submit telemetry before using "
            "the twin panel."
        )
        return snapshot

    frame = ai4i.add_features(
        __import__("pandas").DataFrame(
            [
                {
                    "machine_type": "L",
                    "air_temperature_k": reading.air_temperature_k,
                    "process_temperature_k": reading.process_temperature_k,
                    "rotational_speed_rpm": reading.rotational_speed_rpm,
                    "torque_nm": reading.torque_nm,
                    "tool_wear_min": reading.tool_wear_min,
                }
            ]
        )
    )
    snapshot["readings"] = {
        "recorded_at": reading.recorded_at.isoformat() if reading.recorded_at else None,
        "source": reading.source,
        **{c: round(float(frame[c].iloc[0]), 4) for c in ai4i.SENSOR_COLUMNS},
        **{c: round(float(frame[c].iloc[0]), 4) for c in ai4i.ENGINEERED_COLUMNS},
    }
    return snapshot
