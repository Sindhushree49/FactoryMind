"""AI4I 2020 loading, cleaning and feature engineering.

Schema of the file actually published at UCI dataset 601
(``ai4i2020.csv``, 10 000 rows):

    UDI, Product ID, Type,
    Air temperature [K], Process temperature [K], Rotational speed [rpm],
    Torque [Nm], Tool wear [min], Machine failure,
    TWF, HDF, PWF, OSF, RNF

The five trailing flags are one-hot indicators of the failure sub-type and are
present only on failure rows (``Machine failure == 1``). The dataset does *not*
carry a ``TID``/``Model`` column; the categorical failure mode is reconstructed
from those flags by :func:`failure_mode_from_flags`.

Notes on integrity:
  * ``Machine failure`` is heavily imbalanced (~3.4% positives). Class weights
    and stratified splits are used downstream; nothing is resampled away here.
  * No missing values exist in the published file. The validator still checks and
    records this rather than assuming it.
  * Feature engineering is deterministic and stateless so that training and
    inference cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from app.core.config import settings

AI4I_FILENAME: Final = "ai4i2020.csv"

RAW_COLUMNS: Final = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

FAILURE_FLAG_COLUMNS: Final = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Canonical internal sensor names (identical to the AI4I column semantics).
SENSOR_COLUMNS: Final = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

FAILURE_MODES: Final = {
    0: "No failure",
    1: "Tool wear failure",
    2: "Heat dissipation failure",
    3: "Power failure",
    4: "Overstrain failure",
    5: "Random failure",
}

# Flag column -> numeric mode id used as the classification target.
FLAG_TO_MODE: Final = {"TWF": 1, "HDF": 2, "PWF": 3, "OSF": 4, "RNF": 5}

CATEGORICAL_COLUMNS: Final = ["machine_type"]

ENGINEERED_COLUMNS: Final = [
    "temp_difference_k",
    "temp_to_speed_ratio",
    "power_proxy_kw",
    "torque_to_toolwear",
    "wear_rate_proxy",
    "thermal_load_index",
    "speed_deficit_pct",
    "torque_overload_pct",
]


class AI4IDatasetMissing(FileNotFoundError):
    """Raised when the AI4I CSV is absent; never silently substituted."""


@dataclass(frozen=True)
class AI4IReport:
    """Facts recorded at load time so downstream claims stay auditable."""

    rows: int
    columns: int
    failure_rate: float
    failure_mode_distribution: dict[str, int]
    missing_values: int
    duplicate_rows: int
    machine_type_distribution: dict[str, int]
    sha256: str | None


def dataset_path() -> Path:
    return settings.ai4i_dir / AI4I_FILENAME


def find_dataset() -> Path | None:
    """Locate the CSV, tolerating a nested layout inside ``datasets/ai4i/``."""
    direct = dataset_path()
    if direct.is_file():
        return direct
    if settings.ai4i_dir.exists():
        candidates = sorted(settings.ai4i_dir.rglob("*.csv"))
        if candidates:
            return candidates[0]
    return None


def _sha256(path: Path) -> str | None:
    import hashlib

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def failure_mode_from_flags(row: pd.Series) -> int:
    """Reconstruct the numeric failure mode from the one-hot flags.

    ``0`` means no failure, or a failure row whose flags are all zero (the
    dataset contains none, but the branch keeps the mapping total).
    """
    for flag, mode in FLAG_TO_MODE.items():
        if flag in row.index and float(row[flag]) == 1.0:
            return mode
    return 0


def mode_label(mode_id: int) -> str:
    return FAILURE_MODES.get(int(mode_id), "Unclassified failure")


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the AI4I CSV verbatim and normalise column names."""
    path = path or find_dataset()
    if path is None or not path.is_file():
        raise AI4IDatasetMissing(
            "AI4I 2020 dataset not found. Run: "
            "python backend/scripts/download_datasets.py --dataset ai4i"
        )
    frame = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"AI4I CSV is missing expected columns: {missing}")
    frame = frame.rename(
        columns={
            "UDI": "udi",
            "Product ID": "product_id",
            "Type": "machine_type",
            "Air temperature [K]": "air_temperature_k",
            "Process temperature [K]": "process_temperature_k",
            "Rotational speed [rpm]": "rotational_speed_rpm",
            "Torque [Nm]": "torque_nm",
            "Tool wear [min]": "tool_wear_min",
            "Machine failure": "target",
        }
    )
    for flag in FAILURE_FLAG_COLUMNS:
        frame[flag] = pd.to_numeric(frame[flag], errors="coerce").fillna(0).astype(int)
    frame["model"] = frame.apply(failure_mode_from_flags, axis=1)
    return frame


def build_report(frame: pd.DataFrame, path: Path | None = None) -> AI4IReport:
    positives = frame[frame["target"] == 1]
    mode_labels: dict[str, int] = {}
    for mode_id, count in positives["model"].value_counts().items():
        mode_labels[mode_label(int(mode_id))] = int(count)
    return AI4IReport(
        rows=int(len(frame)),
        columns=int(frame.shape[1]),
        failure_rate=float(frame["target"].mean()),
        failure_mode_distribution=mode_labels,
        missing_values=int(frame.isna().sum().sum()),
        duplicate_rows=int(frame.duplicated().sum()),
        machine_type_distribution={str(k): int(v) for k, v in frame["machine_type"].value_counts().items()},
        sha256=_sha256(path) if path else None,
    )


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, coerce dtypes and clip physically impossible values.

    Clipping bounds come from the AI4I data dictionary, not from the observed
    sample, so they stay stable if the dataset is refreshed.
    """
    out = frame.copy()
    out = out.drop_duplicates(subset=SENSOR_COLUMNS + ["machine_type"], keep="first")
    for column in SENSOR_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["target"] = pd.to_numeric(out["target"], errors="coerce").fillna(0).astype(int)
    out["machine_type"] = out["machine_type"].astype(str).str.strip()

    bounds = {
        "air_temperature_k": (250.0, 330.0),
        "process_temperature_k": (250.0, 360.0),
        "rotational_speed_rpm": (0.0, 30000.0),
        "torque_nm": (0.0, 60.0),
        "tool_wear_min": (0.0, 260.0),
    }
    for column, (low, high) in bounds.items():
        out[column] = out[column].clip(lower=low, upper=high)

    # Median imputation on numeric sensors. The published file has no NaNs; this
    # exists so an inference payload with a gap cannot crash the pipeline.
    for column in SENSOR_COLUMNS:
        if out[column].isna().any():
            out[column] = out[column].fillna(out[column].median())
    return out.reset_index(drop=True)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, domain-motivated feature engineering.

    All derived features are pure functions of the five physical sensors plus
    machine type, so they are computable identically offline and at API time.
    """
    out = frame.copy()
    air = out["air_temperature_k"]
    process = out["process_temperature_k"]
    speed = out["rotational_speed_rpm"].clip(lower=1.0)
    torque = out["torque_nm"]
    wear = out["tool_wear_min"]

    # Cooling margin: process running hot relative to incoming air.
    out["temp_difference_k"] = process - air
    # Thrust proxy: rotational speed relative to thermal headroom.
    out["temp_to_speed_ratio"] = process / speed
    # Mechanical power (kW) = torque * angular velocity.
    out["power_proxy_kw"] = torque * (speed * 2.0 * np.pi / 60.0) / 1000.0
    # Tool wear consumed per unit of torque: high values accelerate wear.
    out["torque_to_toolwear"] = torque / (wear + 1.0)
    # Composite degradation pressure from all four degradation drivers.
    out["wear_rate_proxy"] = out["temp_difference_k"] * torque / (wear + 1.0)
    # Combined thermal + mechanical load, normalised by typical AI4I ranges.
    out["thermal_load_index"] = (
        (out["temp_difference_k"] / 30.0)
        + (torque / 40.0)
        + (wear / 200.0)
    )
    # Fractional deviation from nominal operating point.
    out["speed_deficit_pct"] = (2000.0 - out["rotational_speed_rpm"]) / 2000.0
    out["torque_overload_pct"] = (torque - 40.0) / 40.0
    return out


def numeric_feature_names() -> list[str]:
    return SENSOR_COLUMNS + ENGINEERED_COLUMNS


def feature_names(include_categorical: bool = True) -> list[str]:
    names = numeric_feature_names()
    return names + list(CATEGORICAL_COLUMNS) if include_categorical else names


def model_matrix(frame: pd.DataFrame, include_categorical: bool = True) -> pd.DataFrame:
    """Return the exact ordered feature matrix consumed by every model."""
    columns = feature_names(include_categorical)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"Frame is missing engineered columns: {missing}. Call add_features() first.")
    return frame.loc[:, columns].copy()


def prepare(path: Path | None = None, add_derived: bool = True) -> tuple[pd.DataFrame, AI4IReport]:
    """Full load -> clean -> feature pipeline used by every AI4I entry point."""
    resolved = path or find_dataset()
    raw = load_raw(resolved)
    report = build_report(raw, resolved)
    cleaned = clean(raw)
    if add_derived:
        cleaned = add_features(cleaned)
    return cleaned, report


def failure_mode_labels() -> dict[int, str]:
    return dict(FAILURE_MODES)


def mode_distribution(frame: pd.DataFrame) -> dict[str, int]:
    positives = frame[frame["target"] == 1]
    return {mode_label(int(m)): int(c) for m, c in positives["model"].value_counts().items()}
