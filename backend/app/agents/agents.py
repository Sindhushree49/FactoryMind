"""The six FactoryMind investigation agents.

Every agent:
  * reads measurements from the injected ML layer (never invents a reading),
  * writes a trace entry, a report, and any citations it earned,
  * is deterministic and side-effect free, so an investigation is reproducible.

The Verification agent is the only one that can mark a claim unsupported, and
it always leaves ``requires_human_approval`` true. Nothing in this module
approves maintenance work.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from app.agents import rules
from app.agents.state import AgentReport, Engine, InvestigationState, StepTrace
from app.core.config import settings
from app.core.logging_config import get_logger
from app.rag import retrieval

logger = get_logger(__name__)

MONITORING = "monitoring"
DIAGNOSIS = "diagnosis"
RUL_AGENT = "rul"
DOCUMENTATION = "documentation"
PLANNING = "planning"
VERIFICATION = "verification"

AGENT_ORDER = (MONITORING, DIAGNOSIS, RUL_AGENT, DOCUMENTATION, PLANNING, VERIFICATION)

AGENT_ROLES = {
    MONITORING: "Monitoring Agent",
    DIAGNOSIS: "Diagnosis Agent",
    RUL_AGENT: "RUL Agent",
    DOCUMENTATION: "Documentation Agent",
    PLANNING: "Maintenance Planning Agent",
    VERIFICATION: "Verification Agent",
}


class AgentStepLimitReached(RuntimeError):
    """Raised when the hard step cap is hit; prevents agent loops."""


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _guard(state: InvestigationState) -> None:
    """Hard loop control. The graph is linear, but this is the second line."""
    state["step"] = int(state.get("step", 0)) + 1
    if state["step"] > int(state.get("max_steps", 24)):
        state["max_steps_reached"] = True
        raise AgentStepLimitReached(
            f"Investigation {state['request_id']} exceeded {state['max_steps']} steps."
        )


def _trace(
    state: InvestigationState,
    agent: str,
    action: str,
    summary: str,
    status: str,
    started: float,
    evidence: dict[str, Any] | None = None,
) -> None:
    state["trace"].append(
        StepTrace(
            step=state["step"],
            agent=agent,
            action=action,
            summary=summary,
            status=status,  # type: ignore[arg-type]
            engine=state.get("engine", "rule_based"),
            duration_ms=_elapsed(started),
            evidence=evidence or {},
        ).model_dump()
    )


def _report(
    state: InvestigationState,
    agent: str,
    status: str,
    summary: str,
    findings: dict[str, Any],
    confidence: float | None = None,
    caveats: list[str] | None = None,
    citations: list[dict[str, Any]] | None = None,
) -> None:
    state["reports"].append(
        AgentReport(
            agent=agent,
            role=AGENT_ROLES[agent],
            status=status,  # type: ignore[arg-type]
            engine=state.get("engine", "rule_based"),
            summary=summary,
            findings=findings,
            citations=citations or [],
            confidence=confidence,
            caveats=caveats or [],
        ).model_dump()
    )
    state["agents_ran"].append(agent)


# ---------------------------------------------------------------------------
# Agent 1 - Monitoring
# ---------------------------------------------------------------------------
def monitoring_agent(state: InvestigationState) -> InvestigationState:
    started = time.perf_counter()
    _guard(state)
    measurements = state.get("measurements", {})
    readings = state.get("readings", [])

    if not readings and not measurements:
        state["skipped_agents"].append(MONITORING)
        _trace(state, MONITORING, "evaluate_telemetry",
               "No sensor readings were supplied, so no abnormality could be assessed.",
               "skipped", started)
        _report(state, MONITORING, "skipped",
                "No telemetry supplied.", {"abnormal": None}, caveats=[
                    "Monitoring requires at least one sensor reading."
                ])
        return state

    anomaly = measurements.get("anomaly")
    failure = measurements.get("failure")
    deviations: list[dict[str, Any]] = []
    severity = "info"
    abnormal = False

    if anomaly and anomaly.get("model_trained"):
        abnormal = bool(anomaly.get("is_anomaly"))
        severity = anomaly.get("severity", "info")
        deviations = [
            {
                "feature": item["feature"],
                "value": item["value"],
                "z_score": item.get("z_score"),
                "expected_range": item.get("expected_range"),
                "out_of_training_range": item.get("out_of_training_range"),
            }
            for item in anomaly.get("feature_deviation", [])[:5]
        ]
        summary = (
            f"Isolation Forest scored the reading {anomaly['anomaly_score']:.3f} against a "
            f"{anomaly['threshold']:.3f} threshold: {'abnormal' if abnormal else 'within the normal envelope'}. "
            f"Model {anomaly.get('model_version')}."
        )
    elif readings:
        # No trained anomaly model: fall back to explicit range checks only.
        breaches = _range_breaches(readings[-1])
        abnormal = bool(breaches)
        deviations = breaches
        severity = "high" if len(breaches) >= 2 else "warning" if breaches else "info"
        summary = (
            f"{len(breaches)} reading(s) fall outside the AI4I published operating ranges."
            if breaches else
            "All readings fall inside the AI4I published operating ranges. No trained "
            "anomaly model is available, so this is a range check only, not a modelled "
            "abnormality assessment."
        )
    else:
        summary = "Telemetry exists but no anomaly model output was injected."
        severity = "info"

    probability = (failure or {}).get("failure_probability")
    if probability is not None and probability >= settings.risk_high_threshold:
        abnormal = True
        severity = "high" if probability >= settings.risk_critical_threshold else "warning"

    if abnormal:
        state["alerts"].append(
            {
                "alert_type": "anomaly_detected",
                "severity": severity,
                "title": f"Abnormal telemetry on {state.get('machine_name') or state.get('machine_id') or 'machine'}",
                "message": summary,
                "source": f"agent:{MONITORING}",
                "payload": {
                    "anomaly_score": (anomaly or {}).get("anomaly_score"),
                    "failure_probability": probability,
                    "deviations": deviations[:5],
                },
            }
        )

    _trace(state, MONITORING, "evaluate_telemetry", summary, "ok" if not abnormal else "warning",
           started, {"anomaly_score": (anomaly or {}).get("anomaly_score"),
                     "deviation_count": len(deviations)})
    _report(
        state, MONITORING,
        "warning" if abnormal else "ok",
        summary,
        {
            "abnormal": abnormal,
            "severity": severity,
            "failure_probability": probability,
            "risk_category": (failure or {}).get("risk_category"),
            "deviations": deviations,
            "anomaly_model_used": bool(anomaly and anomaly.get("model_trained")),
        },
        confidence=(anomaly or {}).get("anomaly_score"),
    )
    return state


def _range_breaches(reading: dict[str, Any]) -> list[dict[str, Any]]:
    breaches = []
    for key, limits in rules.LIMITS.items():
        if key not in reading:
            continue
        value = float(reading[key])
        if value < limits["min"] or value > limits["max"]:
            breaches.append(
                {
                    "feature": key,
                    "value": value,
                    "expected_range": [limits["min"], limits["max"]],
                    "out_of_training_range": True,
                }
            )
    return breaches


# ---------------------------------------------------------------------------
# Agent 2 - Diagnosis
# ---------------------------------------------------------------------------
def diagnosis_agent(state: InvestigationState) -> InvestigationState:
    started = time.perf_counter()
    _guard(state)
    measurements = state.get("measurements", {})
    failure = measurements.get("failure") or {}
    readings = state.get("readings", [])

    shap_features = failure.get("features") or []
    drivers = [f for f in shap_features if f.get("contribution", 0) > 0][:5]

    matches = rules.match(readings, failure.get("failure_probability")) if readings else []

    if shap_features:
        basis = (
            "The failure model's SHAP attribution for this reading, combined with the "
            "sensor-pattern rules in the diagnostic rule base."
        )
    else:
        basis = (
            "No failure model output was available, so candidates come only from the "
            "sensor-pattern rules in the diagnostic rule base."
        )

    root_causes: list[dict[str, Any]] = []
    for match in matches[:3]:
        root_causes.append(
            {
                "hypothesis": match["title"],
                "mechanism": match["mechanism"],
                "confidence": match["confidence"],
                "evidence": match["evidence"],
                "evidence_basis": basis,
                "source": "rule_based_diagnostic",
                "rule_id": match["rule_id"],
                "causality_note": (
                    "A sensor-pattern correlation and a model attribution. Neither "
                    "establishes physical causation; the mechanism must be confirmed by "
                    "physical inspection."
                ),
            }
        )

    # A dominant SHAP driver is reported as its own candidate cause.
    if drivers:
        top = drivers[0]
        root_causes.insert(
            0,
            {
                "hypothesis": f"Model attributes elevated risk mainly to {top['feature']}",
                "mechanism": failure.get("explanation"),
                "confidence": round(min(0.9, float(top.get("contribution", 0)) * 5), 3),
                "evidence": {
                    "feature": top["feature"],
                    "observed_value": top.get("value"),
                    "shap_contribution": top.get("contribution"),
                    "direction": top.get("direction"),
                },
                "evidence_basis": "SHAP attribution of the trained failure model",
                "source": "shap_explanation",
                "causality_note": (
                    "SHAP quantifies how the model used this feature. It is not evidence "
                    "that the feature physically causes failure."
                ),
            },
        )

    if not root_causes:
        summary = (
            "No root-cause candidate matched. The reading does not display a recognised "
            "degradation signature, so a physical inspection is the appropriate next step "
            "rather than a speculative repair."
        )
        status = "warning"
    else:
        summary = (
            f"{len(root_causes)} root-cause candidate(s) raised. Leading hypothesis: "
            f"{root_causes[0]['hypothesis']}."
        )
        status = "ok"

    state["root_causes"].extend(root_causes)
    _trace(state, DIAGNOSIS, "attribute_failure", summary, status, started,
           {"candidates": len(root_causes), "rule_matches": [m["rule_id"] for m in matches]})
    _report(
        state, DIAGNOSIS, status, summary,
        {
            "root_causes": root_causes,
            "shap_drivers": drivers,
            "rule_matches": [m["rule_id"] for m in matches],
            "predicted_failure_type": failure.get("predicted_failure_type"),
            "failure_probability": failure.get("failure_probability"),
        },
        confidence=root_causes[0]["confidence"] if root_causes else None,
        caveats=[
            "SHAP values and rule matches are correlates, not proofs of causation.",
            failure.get("shap_note", ""),
        ],
    )
    return state


# ---------------------------------------------------------------------------
# Agent 3 - RUL
# ---------------------------------------------------------------------------
def rul_agent(state: InvestigationState) -> InvestigationState:
    started = time.perf_counter()
    _guard(state)
    measurements = state.get("measurements", {})
    rul = measurements.get("rul") or {}

    if not rul or not rul.get("model_trained"):
        message = (
            "No trained Remaining Useful Life model is available, so no life estimate is "
            "reported. RUL requires a C-MAPSS-trained degradation model; train it with "
            "python backend/app/ml/rul_prediction/train.py --subset FD001"
        )
        state["skipped_agents"].append(RUL_AGENT)
        _trace(state, RUL_AGENT, "estimate_rul", message, "skipped", started)
        _report(state, RUL_AGENT, "skipped", message, {"predicted_rul": None},
                caveats=["RUL is only defined for assets with a trained degradation model."])
        return state

    predicted = float(rul["predicted_rul"])
    lower, upper = float(rul["lower_bound"]), float(rul["upper_bound"])
    span = upper - lower
    confidence = round(max(0.1, min(0.9, 1.0 - span / max(upper, 1.0))), 3)

    summary = (
        f"Estimated remaining useful life is {predicted:.0f} cycles "
        f"(interval {lower:.0f}-{upper:.0f}, +/-{rul.get('uncertainty_pct', 0):.0f}%) "
        f"for {rul.get('engine_unit', 'the engine')} using {rul.get('model_family', 'the')} model "
        f"({rul.get('model_version')})."
    )
    if span / max(predicted, 1.0) > 0.6:
        summary += " The interval is wide relative to the estimate, so plan conservatively."
        status = "warning"
    else:
        status = "ok"

    caveats = [rul.get("confidence_note", "")]
    caveats.extend(rul.get("caveats", []))
    if rul.get("dataset_subset") and rul["dataset_subset"] != "FD001":
        caveats.append(
            "Estimated from a different operating-condition subset; treat as indicative."
        )

    _trace(state, RUL_AGENT, "estimate_rul", summary, status, started,
           {"predicted_rul": predicted, "interval": [lower, upper]})
    _report(
        state, RUL_AGENT, status, summary,
        {
            "predicted_rul": predicted,
            "lower_bound": lower,
            "upper_bound": upper,
            "uncertainty_pct": rul.get("uncertainty_pct"),
            "model_family": rul.get("model_family"),
            "model_version": rul.get("model_version"),
            "dataset_subset": rul.get("dataset_subset"),
            "degradation_curve": (rul.get("degradation_curve") or [])[-12:],
        },
        confidence=confidence,
        caveats=[c for c in caveats if c],
    )
    return state


def self_upper(value: float) -> float:
    return max(value, 1.0)


# ---------------------------------------------------------------------------
# Agent 4 - Documentation (RAG)
# ---------------------------------------------------------------------------
DOC_QUERIES = {
    MONITORING: "abnormal vibration temperature rise warning limits machine",
    DIAGNOSIS: "root cause failure mode diagnosis troubleshooting procedure",
    RUL_AGENT: "degradation trend inspection interval remaining life maintenance planning",
}


def documentation_agent(state: InvestigationState) -> InvestigationState:
    started = time.perf_counter()
    _guard(state)

    topics: list[str] = []
    for cause in state.get("root_causes", [])[:2]:
        topics.append(str(cause.get("hypothesis", "")))
    for report in state.get("reports", []):
        if report.get("agent") == MONITORING and report["findings"].get("abnormal"):
            topics.append("abnormal sensor reading limits")
            break
    if not topics:
        topics = ["general maintenance procedure"]

    store_size = _store_size()
    if store_size == 0:
        message = (
            "The knowledge base is empty, so no documentation could be retrieved. Upload "
            "manufacturer manuals or maintenance procedures on the Knowledge Base page."
        )
        state["skipped_agents"].append(DOCUMENTATION)
        _trace(state, DOCUMENTATION, "search_knowledge_base", message, "skipped", started)
        _report(state, DOCUMENTATION, "skipped", message, {"citations": []},
                caveats=["No documents are indexed; every claim below rests on models alone."])
        return state

    if not retrieval.get_store().embedding_available:
        message = (
            "The embedding model could not be loaded, so semantic retrieval is disabled. "
            "No documentation citations can be produced. Install sentence-transformers and "
            "ensure huggingface.co is reachable to enable retrieval."
        )
        state["skipped_agents"].append(DOCUMENTATION)
        _trace(state, DOCUMENTATION, "search_knowledge_base", message, "blocked", started)
        _report(state, DOCUMENTATION, "blocked", message, {"citations": []},
                caveats=["Retrieval backend unavailable."])
        return state

    answers: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    for topic in topics[:3]:
        query = rules.question_for(topic, topic)
        try:
            answer = retrieval.answer(query, top_k=settings.rag_top_k)
        except Exception as exc:  # noqa: BLE001
            logger.error("Retrieval failed for %r: %s", query, exc)
            continue
        answers.append({"query": query, "topic": topic, **answer.to_dict()})
        citations.extend(answer.citations)

    usable = [c for answer in answers for c in answer["citations"]]
    insufficient = all(a.get("insufficient_evidence") for a in answers) if answers else True

    if usable:
        summary = f"Retrieved {len(usable)} cited passage(s) from the indexed documentation."
        status = "ok"
    else:
        summary = (
            "Documents are indexed but none of them address this condition closely enough "
            "to cite. Recommendations will therefore be flagged as lacking documentary "
            "support."
        )
        status = "warning"

    state["knowledge_answers"].extend(answers)
    state["citations"].extend(citations)
    _trace(state, DOCUMENTATION, "search_knowledge_base", summary, status, started,
           {"queries": [a["query"] for a in answers], "citations": len(usable)})
    _report(
        state, DOCUMENTATION, status, summary,
        {
            "queries": [a["query"] for a in answers],
            "citation_count": len(usable),
            "insufficient_evidence": insufficient,
            "answers": answers,
        },
        confidence=round(max((c["score"] for c in usable), default=0.0), 3),
        caveats=[a["disclaimer"] for a in answers[:1]],
        citations=usable[:10],
    )
    return state


def _store_size() -> int:
    try:
        return retrieval.get_store().size
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------
# Agent 5 - Maintenance Planning
# ---------------------------------------------------------------------------
def planning_agent(state: InvestigationState) -> InvestigationState:
    started = time.perf_counter()
    _guard(state)

    causes = state.get("root_causes", [])
    citations = state.get("citations", [])
    rul = next((r for r in state.get("reports", []) if r.get("agent") == RUL_AGENT), None)

    if not causes:
        message = (
            "No maintenance action is planned: no root-cause candidate was established. "
            "Scheduling a repair without a supported hypothesis would be unfounded."
        )
        state["skipped_agents"].append(PLANNING)
        _trace(state, PLANNING, "plan_maintenance", message, "skipped", started)
        _report(state, PLANNING, "skipped", message, {"recommendations": []},
                caveats=["Actions require at least one supported root-cause hypothesis."])
        return state

    risk = next((r for r in state.get("reports", []) if r.get("agent") == MONITORING), None)
    severity = (risk or {}).get("findings", {}).get("severity", "info")
    probability = (risk or {}).get("findings", {}).get("failure_probability")
    estimated_rul = (rul or {}).get("findings", {}).get("predicted_rul")

    recommendations: list[dict[str, Any]] = []
    for cause in causes[:3]:
        rule = rules.RULE_BY_ID.get(str(cause.get("rule_id")))
        downtime = rule.downtime_min if rule else 60
        priority = _priority(cause.get("confidence", 0.4), severity, estimated_rul)
        matched_citations = [
            c for c in citations
            if any(word in str(c.get("document_title", "")).lower() for word in _title_words(cause))
        ] or citations[:1]

        recommendations.append(
            {
                "title": f"{cause.get('hypothesis', 'Inspection')} - corrective action",
                "description": rule.action if rule else (
                    f"Physically inspect the machine for the condition described by "
                    f"'{cause.get('hypothesis')}' before deciding on a repair."
                ),
                "priority": priority,
                "action_type": rule.action_type if rule else "inspection",
                "estimated_downtime_min": downtime,
                "rationale": (
                    f"Hypothesis confidence {cause.get('confidence')}, evidence basis "
                    f"{cause.get('evidence_basis')}. Current failure probability "
                    f"{probability if probability is None else round(float(probability), 4)}, "
                    f"estimated remaining life {estimated_rul if estimated_rul is None else f'{estimated_rul:.0f} cycles'}."
                ),
                "confidence": cause.get("confidence"),
                "evidence": [
                    {"type": "root_cause", "hypothesis": cause.get("hypothesis"),
                     "evidence_basis": cause.get("evidence_basis")},
                    {"type": "sensor_evidence", "details": cause.get("evidence")},
                ],
                "citations": matched_citations,
                "requires_human_approval": True,
                "source_agent": PLANNING,
            }
        )

    recommendations.sort(key=lambda r: _PRIORITY_ORDER.get(r["priority"], 3))
    state["recommendations"].extend(recommendations)

    top = recommendations[0]
    summary = (
        f"{len(recommendations)} maintenance action(s) drafted. Highest priority: "
        f"{top['priority']} - {top['title']}. All actions require human approval."
    )
    _trace(state, PLANNING, "plan_maintenance", summary, "ok", started,
           {"recommendations": len(recommendations), "top_priority": top["priority"]})
    _report(
        state, PLANNING, "ok", summary,
        {
            "recommendations": recommendations,
            "priority_basis": {"severity": severity, "failure_probability": probability,
                               "estimated_rul": estimated_rul},
        },
        confidence=top.get("confidence"),
        caveats=[
            "Recommended actions are proposals generated from model and rule evidence; "
            "a qualified technician must confirm the diagnosis on the machine.",
            "No action is executed automatically. Human approval is mandatory.",
        ],
    )
    return state


_PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _priority(confidence: float, severity: str, estimated_rul: float | None) -> str:
    if severity == "critical" or (estimated_rul is not None and estimated_rul < 20):
        return "critical"
    if confidence >= 0.7 or severity == "high":
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def _title_words(cause: dict[str, Any]) -> set[str]:
    words = set(str(cause.get("hypothesis", "")).lower().split())
    words -= {"the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "is"}
    return {w for w in words if len(w) > 4}


# ---------------------------------------------------------------------------
# Agent 6 - Verification
# ---------------------------------------------------------------------------
def verification_agent(state: InvestigationState) -> InvestigationState:
    started = time.perf_counter()
    _guard(state)

    causes = state.get("root_causes", [])
    recommendations = state.get("recommendations", [])
    citations = state.get("citations", [])
    unsupported: list[dict[str, Any]] = []

    for cause in causes:
        basis = str(cause.get("evidence_basis", ""))
        if "SHAP" in basis or "rule" in basis or "sensor" in basis:
            continue
        unsupported.append(
            {
                "claim": cause.get("hypothesis"),
                "reason": "No model attribution, sensor pattern, or retrieved document supports this claim.",
                "required_evidence": "A documented procedure or a measured sensor signature.",
            }
        )

    for recommendation in recommendations:
        if not recommendation.get("citations"):
            unsupported.append(
                {
                    "claim": recommendation.get("title"),
                    "reason": "No documentation citation backs this maintenance action.",
                    "required_evidence": (
                        "A retrieved procedure from the knowledge base, or an explicit "
                        "engineer sign-off recorded against the recommendation."
                    ),
                }
            )
        if (recommendation.get("confidence") or 0) < 0.4:
            unsupported.append(
                {
                    "claim": recommendation.get("title"),
                    "reason": (
                        f"Confidence {recommendation.get('confidence')} is below the 0.4 "
                        "threshold for acting without human confirmation."
                    ),
                    "required_evidence": "Physical inspection confirming the hypothesis.",
                }
            )

    # Verify that no citation points at a document that was not actually retrieved.
    retrieved_ids = {c.get("document_id") for c in citations if c.get("document_id")}
    for recommendation in recommendations:
        for citation in recommendation.get("citations", []):
            if citation.get("document_id") not in retrieved_ids:
                unsupported.append(
                    {
                        "claim": f"citation[{citation.get('index')}] on {recommendation.get('title')}",
                        "reason": "Cited document was not present in the retrieval result set.",
                        "required_evidence": "Re-run retrieval and confirm the document is indexed.",
                    }
                )

    if not causes:
        verdict = "insufficient_evidence"
    elif unsupported:
        verdict = "conditional"
    else:
        verdict = "supported"

    state["unsupported_claims"].extend(unsupported)
    state["requires_human_approval"] = True   # never cleared by an agent

    if verdict == "supported":
        summary = (
            f"Every proposed action traces to model attribution or a retrieved document. "
            f"{len(recommendations)} action(s) are ready for engineer review."
        )
        status = "ok"
    elif verdict == "conditional":
        summary = (
            f"{len(unsupported)} claim(s) lack complete support and are flagged below. "
            "The investigation can proceed only with an engineer's physical confirmation."
        )
        status = "warning"
    else:
        summary = (
            "Evidence is insufficient to recommend any maintenance action. Perform a "
            "physical inspection; this investigation does not support a repair."
        )
        status = "blocked"

    summary += " Human approval is required before any action is executed."
    state["verdict"] = verdict

    _trace(state, VERIFICATION, "verify_evidence", summary, status, started,
           {"verdict": verdict, "unsupported": len(unsupported), "citations": len(citations)})
    _report(
        state, VERIFICATION, status, summary,
        {
            "verdict": verdict,
            "unsupported_claims": unsupported,
            "citations_verified": len(citations),
            "requires_human_approval": True,
        },
        confidence=round(1.0 - min(1.0, len(unsupported) / max(len(recommendations), 1)), 3),
        caveats=[
            "This agent checks evidence traceability only. It cannot and does not approve "
            "maintenance work.",
            "A 'supported' verdict means the recommendation is traceable to evidence, not "
            "that the diagnosis is physically confirmed.",
        ],
    )
    return state


AGENTS: dict[str, Callable[[InvestigationState], InvestigationState]] = {
    MONITORING: monitoring_agent,
    DIAGNOSIS: diagnosis_agent,
    RUL_AGENT: rul_agent,
    DOCUMENTATION: documentation_agent,
    PLANNING: planning_agent,
    VERIFICATION: verification_agent,
}
