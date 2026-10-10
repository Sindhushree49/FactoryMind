"""SHAP plotting and report generation.

Plots are written to ``storage/reports/`` and served as static files. Every
figure is labelled as an explanation of model behaviour, not of causation.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

CAUSALITY_DISCLAIMER = (
    "SHAP decomposes a model's output into per-feature contributions. "
    "It explains what the model used, not what physically causes the failure."
)


def _new_figure(width: float = 10.0, height: float = 6.0) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(width, height))
    return plt, figure, axis


def _save(figure: Any, name: str) -> Path:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = settings.reports_dir / f"{name}_{stamp}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight", facecolor="#0b1220")
    return path


def summary_plot(
    model: Any,
    background: np.ndarray,
    filename: str = "shap_summary",
    max_display: int = 15,
    sample: int = 500,
) -> Path | None:
    """Beeswarm summary of feature impact across a sample of the training data."""
    try:
        import shap

        plt, figure, axis = _new_figure()
        rng = np.random.default_rng(settings.random_seed)
        if background.shape[0] > sample:
            background = background[rng.choice(background.shape[0], sample, replace=False)]
        shap.summary_plot(
            shap.TreeExplainer(model).shap_values(background),
            background,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        axis.set_title(
            f"SHAP summary - global feature impact ({CAUSALITY_DISCLAIMER})",
            fontsize=9, color="#e2e8f0", loc="left",
        )
        path = _save(figure, filename)
        plt.close(figure)
        logger.info("Wrote SHAP summary plot %s", path)
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP summary plot failed: %s", exc)
        return None


def bar_plot(feature_names: list[str], importances: np.ndarray, filename: str = "shap_bar") -> Path:
    """Mean absolute SHAP value per feature."""
    plt, figure, axis = _new_figure(10.0, max(4.0, 0.32 * len(feature_names) + 2.0))
    order = np.argsort(importances)
    axis.barh(
        [feature_names[i] for i in order],
        importances[order],
        color="#22d3ee",
        edgecolor="#0e7490",
    )
    axis.set_xlabel("mean |SHAP value|", color="#94a3b8")
    axis.set_title(
        f"Global feature importance ({CAUSALITY_DISCLAIMER})",
        fontsize=9, color="#e2e8f0", loc="left",
    )
    axis.tick_params(colors="#cbd5e1")
    for spine in axis.spines.values():
        spine.set_color("#334155")
    path = _save(figure, filename)
    plt.close(figure)
    return path


def waterfall_plot(
    feature_labels: list[str],
    contributions: list[float],
    value: float,
    base_value: float,
    filename: str = "shap_waterfall",
) -> Path | None:
    """Per-prediction contribution waterfall."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import shap

        plt, figure, axis = _new_figure(10.0, 6.0)
        shap.force_plot(
            base_value=base_value,
            shap_values=np.array(contributions, dtype=float),
            features=np.array([value], dtype=float),
            feature_names=feature_labels,
            show=False,
            matplotlib=True,
            ax=axis,
        )
        axis.set_title(
            "Single-prediction contribution breakdown (model attribution, not causation)",
            fontsize=9, color="#e2e8f0", loc="left",
        )
        path = _save(figure, filename)
        plt.close(figure)
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP waterfall plot failed: %s", exc)
        return None


def dependence_plot(
    background: np.ndarray,
    shap_values: np.ndarray,
    feature_index: int,
    feature_name: str,
    interaction_index: int = 0,
    filename: str = "shap_dependence",
) -> Path | None:
    try:
        import shap

        plt, figure, axis = _new_figure(10.0, 6.0)
        shap.dependence_plot(
            feature_index,
            shap_values,
            background,
            interaction_index=interaction_index,
            show=False,
            ax=axis,
        )
        axis.set_title(
            f"SHAP dependence: {feature_name} ({CAUSALITY_DISCLAIMER})",
            fontsize=9, color="#e2e8f0", loc="left",
        )
        path = _save(figure, filename)
        plt.close(figure)
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP dependence plot failed: %s", exc)
        return None


def data_uri(path: Path) -> str:
    """Inline a PNG for clients that cannot reach the static mount."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def to_base64_png(figure_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(figure_bytes).decode("ascii")


def render_bytes(figure: Any) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight", facecolor="#0b1220")
    return buffer.getvalue()
