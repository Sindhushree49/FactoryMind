"""Diagnostic rule base for the rule-based agent fallback.

These are explicitly-labelled engineering heuristics, not learned models. Each
rule states the sensor pattern it keys on, the mechanism it encodes, the
confidence it carries, and the maintenance response it suggests.

Reference ranges come from the AI4I 2020 data dictionary and the published
statistics of that dataset (air 250-330 K, process 250-360 K, speed 0-30000 rpm,
torque 0-60 Nm, tool wear 0-260 min). They are dataset ranges, not
manufacturer limits for any specific machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# AI4I published operating envelope.
NOMINAL = {
    "air_temperature_k": 300.1,
    "process_temperature_k": 310.1,
    "rotational_speed_rpm": 1500.0,
    "torque_nm": 40.0,
    "tool_wear_min": 0.0,
}

LIMITS = {
    "air_temperature_k": {"min": 250.0, "max": 330.0},
    "process_temperature_k": {"min": 250.0, "max": 360.0},
    "rotational_speed_rpm": {"min": 0.0, "max": 30000.0},
    "torque_nm": {"min": 0.0, "max": 60.0},
    "tool_wear_min": {"min": 0.0, "max": 260.0},
}

# Typical magnitudes at AI4I nominal duty, used only to band the derived proxies.
# torque 40 Nm at 1500 rpm -> ~6.28 kW.
_POWER_REFERENCE = {"power_proxy_kw": 6.28, "wear_rate_proxy": 4.0}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    title: str
    mechanism: str
    pattern: dict[str, str]
    base_confidence: float
    action: str
    action_type: str
    priority: str
    downtime_min: int
    keywords: tuple[str, ...] = field(default_factory=tuple)


RULES: tuple[Rule, ...] = (
    Rule(
        rule_id="R-HEAT-001",
        title="Cooling / heat dissipation degradation",
        mechanism=(
            "A rising process-to-air temperature difference with torque held near nominal "
            "indicates the coolant or heat-rejection path is losing capacity, so the same "
            "mechanical work is dissipating less efficiently."
        ),
        pattern={
            "temp_difference_k": "high",
            "torque_nm": "normal_or_high",
            "rotational_speed_rpm": "normal_or_low",
        },
        base_confidence=0.72,
        action=(
            "Inspect the coolant circuit: verify coolant level and concentration, check "
            "for pump or fan degradation, and clean the heat exchanger and radiator fins."
        ),
        action_type="cooling_system_service",
        priority="high",
        downtime_min=120,
        keywords=("temperature", "overheat", "heat dissipation", "cooling", "coolant"),
    ),
    Rule(
        rule_id="R-TOOL-002",
        title="Tool wear approaching end of life",
        mechanism=(
            "Cumulative tool wear with torque rising at similar speed indicates the cutting "
            "edge has dulled, so the same load now needs more torque to achieve the same "
            "material removal."
        ),
        pattern={"tool_wear_min": "high", "torque_nm": "high", "power_proxy_kw": "high"},
        base_confidence=0.78,
        action=(
            "Replace or re-sharpen the cutting tool at the next planned stop and reset the "
            "tool-wear counter. Verify the tool-change interval against the process spec."
        ),
        action_type="tool_replacement",
        priority="high",
        downtime_min=45,
        keywords=("tool wear", "cutting tool", "tool change", "dull"),
    ),
    Rule(
        rule_id="R-LOAD-003",
        title="Mechanical overstrain",
        mechanism=(
            "Torque and mechanical power well above nominal at reduced rotational speed is "
            "the signature of binding, misalignment, or an over-constrained workpiece."
        ),
        pattern={"torque_nm": "high", "rotational_speed_rpm": "low", "power_proxy_kw": "high"},
        base_confidence=0.70,
        action=(
            "Stop and inspect for binding or a misaligned fixture. Check spindle bearings "
            "and confirm the workpiece clamp is correct before restarting."
        ),
        action_type="mechanical_inspection",
        priority="critical",
        downtime_min=180,
        keywords=("overstrain", "torque", "binding", "mechanical load", "stall"),
    ),
    Rule(
        rule_id="R-POWER-004",
        title="Electrical power supply degradation",
        mechanism=(
            "Speed collapsing while torque holds points to a power-delivery or drive "
            "problem rather than a mechanical load change, because the motor cannot hold "
            "its commanded speed."
        ),
        pattern={"rotational_speed_rpm": "low", "torque_nm": "normal_or_high"},
        base_confidence=0.60,
        action=(
            "Check supply voltage, phase balance and drive current limits. Review the "
            "variable-frequency drive fault log for undervoltage or overcurrent events."
        ),
        action_type="electrical_inspection",
        priority="high",
        downtime_min=90,
        keywords=("power", "voltage", "electrical", "drive", "vfd", "supply"),
    ),
    Rule(
        rule_id="R-SPEED-005",
        title="Speed deficit with cool running temperatures",
        mechanism=(
            "Rotational speed below nominal while temperatures stay low is consistent with "
            "reduced commanded setpoint or an encoder/speed-feedback fault rather than "
            "thermal or mechanical strain."
        ),
        pattern={"rotational_speed_rpm": "low", "temp_difference_k": "low"},
        base_confidence=0.45,
        action=(
            "Confirm the commanded speed setpoint, then verify the speed encoder and its "
            "feedback wiring before adjusting process parameters."
        ),
        action_type="sensor_verification",
        priority="medium",
        downtime_min=60,
        keywords=("rotational speed", "setpoint", "encoder", "speed"),
    ),
    Rule(
        rule_id="R-WEAR-006",
        title="Accelerating tool wear rate",
        mechanism=(
            "High tool wear combined with a high wear-rate proxy means wear is progressing "
            "faster than a fixed replacement interval assumes, so the current interval is "
            "not appropriate for this duty cycle."
        ),
        pattern={"wear_rate_proxy": "high", "tool_wear_min": "medium_or_high"},
        base_confidence=0.62,
        action=(
            "Shorten the tool-change interval for this duty cycle and review cutting "
            "parameters (feed and speed) against the material specification."
        ),
        action_type="process_adjustment",
        priority="medium",
        downtime_min=30,
        keywords=("tool wear", "wear rate", "interval", "cutting parameters"),
    ),
    Rule(
        rule_id="R-RAND-007",
        title="Unclassified / random failure indication",
        mechanism=(
            "AI4I includes a random-failure class that carries no single discriminating "
            "sensor signature. An elevated risk without a dominant driver is consistent "
            "with this class, which is why a physical inspection is still required."
        ),
        pattern={"failure_probability": "high", "dominant_driver": "none"},
        base_confidence=0.30,
        action=(
            "Perform a full physical inspection against the machine checklist. No specific "
            "sensor signature identifies this failure mode."
        ),
        action_type="full_inspection",
        priority="medium",
        downtime_min=240,
        keywords=("random failure", "unclassified", "inspection"),
    ),
)

RULE_BY_ID = {rule.rule_id: rule for rule in RULES}


def bands(value: float, key: str) -> str:
    """Classify a sensor value into low/normal/medium/high against dataset ranges."""
    if key == "tool_wear_min":
        return "high" if value >= 200 else "medium" if value >= 100 else "low"
    if key == "rotational_speed_rpm":
        return "low" if value < 900 else "high" if value > 2600 else "normal"
    if key == "torque_nm":
        return "high" if value > 48 else "low" if value < 25 else "normal"
    if key == "temp_difference_k":
        return "high" if value >= 15 else "low" if value <= 5 else "normal"
    if key in ("air_temperature_k", "process_temperature_k"):
        return "high" if value >= 315 else "low" if value <= 295 else "normal"
    if key in ("power_proxy_kw", "wear_rate_proxy"):
        reference = _POWER_REFERENCE.get(key, 1.0)
        return "high" if value >= reference * 1.5 else "low" if value <= reference * 0.6 else "normal"
    if key == "failure_probability":
        return "high" if value >= 0.6 else "medium" if value >= 0.3 else "low"
    return "normal"


def match(readings: list[dict[str, Any]], failure_probability: float | None = None) -> list[dict[str, Any]]:
    """Return rule matches with an evidence-based confidence adjustment."""
    if not readings:
        return []
    latest = readings[-1]
    derived = {**latest}
    temp_difference = float(latest.get("process_temperature_k", 0)) - float(
        latest.get("air_temperature_k", 0)
    )
    derived["temp_difference_k"] = temp_difference
    speed = max(float(latest.get("rotational_speed_rpm", 1)), 1.0)
    torque = float(latest.get("torque_nm", 0))
    derived["power_proxy_kw"] = torque * (speed * 2 * 3.141592653589793 / 60.0) / 1000.0
    derived["wear_rate_proxy"] = temp_difference * torque / (float(latest.get("tool_wear_min", 0)) + 1.0)
    if failure_probability is not None:
        derived["failure_probability"] = float(failure_probability)

    matched: list[dict[str, Any]] = []
    for rule in RULES:
        evidence: dict[str, Any] = {}
        satisfied = 0
        for feature, expectation in rule.pattern.items():
            if feature == "dominant_driver":
                continue
            if feature not in derived:
                continue
            observed = bands(float(derived[feature]), feature)
            ok = (
                expectation in observed
                if expectation.endswith("or_low") or expectation.endswith("or_high")
                else observed == expectation
            )
            if expectation.endswith("or_low"):
                ok = observed in {"low", "normal"}
            elif expectation.endswith("or_high"):
                ok = observed in {"high", "normal"}
            evidence[feature] = {"value": derived[feature], "band": observed, "expected": expectation}
            satisfied += int(ok)

        applicable = sum(1 for f in rule.pattern if f != "dominant_driver" and f in derived)
        if applicable == 0 or satisfied < max(1, applicable - 1):
            continue

        ratio = satisfied / applicable
        confidence = min(0.95, rule.base_confidence * (0.6 + 0.4 * ratio))
        matched.append(
            {
                "rule_id": rule.rule_id,
                "title": rule.title,
                "mechanism": rule.mechanism,
                "action": rule.action,
                "action_type": rule.action_type,
                "priority": rule.priority,
                "estimated_downtime_min": rule.downtime_min,
                "confidence": round(confidence, 3),
                "matched_conditions": satisfied,
                "total_conditions": applicable,
                "evidence": evidence,
                "source": "rule_based_diagnostic",
            }
        )
    matched.sort(key=lambda m: m["confidence"], reverse=True)
    return matched


def question_for(topic: str, symptom: str) -> str:
    """Build a natural-language retrieval query from a diagnosis."""
    keywords = [k for rule in RULES if k in topic.lower() or any(w in symptom.lower() for w in rule.keywords)]
    terms: list[str] = []
    for rule in keywords[:2]:
        terms.extend(rule.keywords[:3])
    if not terms:
        terms = topic.split()[:6]
    return " ".join(dict.fromkeys(terms))
