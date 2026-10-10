"""LangGraph orchestration for the six-agent investigation workflow.

Graph shape (strictly linear, so no agent can loop):

    monitoring -> diagnosis -> rul -> documentation -> planning -> verification -> END

Loop safety is enforced in three independent places:

1. The graph topology has no cycles, so a cycle is not expressible.
2. Every node calls ``_guard()`` in ``agents.py``, which raises once the
   configured step budget is exceeded.
3. A recursion limit is passed to the LangGraph runner and an
   ``AgentStepLimitReached`` exception is caught by the caller, which records
   ``max_steps_reached`` instead of retrying.

If LangGraph cannot be imported, the identical node sequence runs as a plain
Python pipeline, so the feature degrades in availability but never in behaviour.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from app.agents.agents import (
    AGENTS,
    AGENT_ORDER,
    AGENT_ROLES,
    AgentStepLimitReached,
    diagnosis_agent,
    documentation_agent,
    monitoring_agent,
    planning_agent,
    rul_agent,
    verification_agent,
)
from app.agents.state import Engine, InvestigationState, new_state
from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

GRAPH_NAME = "factorymind_investigation"
DEFAULT_MAX_STEPS = 24


def build_graph() -> tuple[Any, str]:
    """Compile the LangGraph state graph. Returns ``(compiled, engine_used)``."""
    try:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(InvestigationState)
        for name, node in (
            ("monitoring", monitoring_agent),
            ("diagnosis", diagnosis_agent),
            ("rul", rul_agent),
            ("documentation", documentation_agent),
            ("planning", planning_agent),
            ("verification", verification_agent),
        ):
            graph.add_node(name, node)

        graph.add_edge(START, "monitoring")
        for current, nxt in zip(AGENT_ORDER, AGENT_ORDER[1:]):
            graph.add_edge(current, nxt)
        graph.add_edge("verification", END)

        return graph.compile(), "langgraph"
    except Exception as exc:  # noqa: BLE001
        logger.warning("LangGraph unavailable (%s); running the equivalent linear pipeline", exc)
        return None, "linear_pipeline"


_compiled: Any = None
_engine_used: str | None = None


def _get_graph() -> tuple[Any, str]:
    global _compiled, _engine_used
    if _engine_used is None:
        _compiled, _engine_used = build_graph()
    return _compiled, _engine_used


def run_pipeline(
    machine_id: str | None = None,
    machine_name: str | None = None,
    readings: list[dict[str, Any]] | None = None,
    measurements: dict[str, Any] | None = None,
    max_steps: int = 24,
    triggered_alert_id: str | None = None,
) -> InvestigationState:
    """Execute the full investigation and return the final state."""
    engine: Engine = "hybrid" if settings.llm_active else "rule_based"
    state = new_state(
        request_id=uuid.uuid4().hex,
        max_steps=max_steps,
        engine=engine,
        machine_id=machine_id,
        machine_name=machine_name,
        readings=readings,
        measurements=measurements,
        triggered_alert_id=triggered_alert_id,
    )

    compiled, backend = _get_graph()
    logger.info(
        "Investigation %s starting (backend=%s, engine=%s, max_steps=%d)",
        state["request_id"], backend, engine, max_steps,
    )

    try:
        if compiled is not None:
            state = compiled.invoke(
                state,
                config={
                    "recursion_limit": max_steps + len(AGENT_ORDER) + 2,
                    "tags": [GRAPH_NAME],
                },
            )
        else:
            state = _run_linear(state)
    except AgentStepLimitReached as exc:
        logger.error("Investigation %s halted: %s", state["request_id"], exc)
        state["max_steps_reached"] = True
        state["verdict"] = state.get("verdict") or "halted"
        state["trace"].append(
            {
                "step": state.get("step", 0),
                "agent": "orchestrator",
                "action": "abort",
                "summary": str(exc),
                "status": "blocked",
                "engine": engine,
                "duration_ms": None,
                "evidence": {"reason": "step_budget_exhausted"},
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Investigation %s failed", state["request_id"])
        state["verdict"] = "error"
        state["trace"].append(
            {
                "step": state.get("step", 0),
                "agent": "orchestrator",
                "action": "abort",
                "summary": f"{type(exc).__name__}: {exc}",
                "status": "blocked",
                "engine": engine,
                "duration_ms": None,
                "evidence": {"reason": "unhandled_exception"},
            }
        )

    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    state.setdefault("verdict", None)
    if not state.get("verdict"):
        state["verdict"] = "no_action_required" if not state["recommendations"] else "review_required"
    return state


def _run_linear(state: InvestigationState) -> InvestigationState:
    """Fallback runner with identical semantics to the compiled graph."""
    for name in AGENT_ORDER:
        node: Callable[[InvestigationState], InvestigationState] = AGENTS[name]
        state = node(state)
    return state


def agent_catalog() -> list[dict[str, str]]:
    """Static description of the workflow, used by the UI."""
    descriptions = {
        "monitoring": "Evaluates incoming telemetry and raises abnormality alerts.",
        "diagnosis": "Attributes failure risk to candidate root causes using SHAP and diagnostic rules.",
        "rul": "Estimates remaining useful life and states prediction uncertainty.",
        "documentation": "Retrieves maintenance procedures from the indexed knowledge base with citations.",
        "planning": "Converts supported hypotheses into prioritised maintenance proposals.",
        "verification": "Checks evidence traceability, flags unsupported claims, and holds for human approval.",
    }
    return [
        {"key": name, "label": AGENT_ROLES[name], "description": descriptions[name]}
        for name in AGENT_ORDER
    ]


def graph_description() -> dict[str, Any]:
    compiled, backend = _get_graph()
    nodes = list(AGENT_ORDER)
    edges = [[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1)]
    return {
        "name": GRAPH_NAME,
        "backend": backend,
        "langgraph_active": compiled is not None,
        "nodes": nodes,
        "edges": edges,
        "entrypoint": nodes[0],
        "terminal": nodes[-1],
        "max_steps_default": DEFAULT_MAX_STEPS,
        "loop_protection": [
            "acyclic linear topology - no cycle is expressible in the graph",
            f"per-node step budget of {DEFAULT_MAX_STEPS} steps raises AgentStepLimitReached",
            "LangGraph recursion_limit is set above the node count",
            "agents have read-only access to the measurements dictionary, so no agent can "
            "invent a sensor reading",
        ],
        "llm_active": settings.llm_active,
    }
