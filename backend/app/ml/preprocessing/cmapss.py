"""NASA C-MAPSS turbofan loading, windowing and RUL labelling.

C-MAPSS format (whitespace separated, no header):
  train:  unit cycle op1 op2 op3 sensor1..sensor21
  test:   unit cycle op1 op2 op3 sensor1..sensor21   (RUL lives in RUL_*.txt)
  RUL_*.txt: one ground-truth RUL per test unit

The four subsets differ in operating-condition noise and fault magnitude:
  FD001 one condition, one fault mode
  FD002 one condition, six fault modes
  FD003 one condition, one fault mode + noise
  FD004 six conditions, six fault modes + noise

Leakage control: rows are grouped by ``unit`` and every split is performed on
unit groups, never on individual cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from app.core.config import settings

SUBSETS: Final = ["FD001", "FD002", "FD003", "FD004"]

COLUMNS: Final = ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]

# Sensors that are constant or near-constant in C-MAPSS carry no signal; they are
# removed before scaling (standard practice in the prognostics literature).
CONSTANT_SENSORS: Final = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]

WINDOW_SIZE: Final = 30
MAX_RUL: Final = 125


class CMAPSSMissing(FileNotFoundError):
    """Raised when C-MAPSS files are absent; never silently substituted."""


@dataclass(frozen=True)
class CMAPSSSubsetInfo:
    subset: str
    train_units: int
    test_units: int
    train_rows: int
    max_cycle: int


def subset_dir() -> Path:
    return settings.cmapss_dir


def find_files(subset: str) -> dict[str, Path]:
    root = subset_dir()
    files = {
        "train": root / f"train_{subset}.txt",
        "test": root / f"test_{subset}.txt",
        "rul": root / f"RUL_{subset}.txt",
    }
    if not all(p.is_file() for p in files.values()):
        available = sorted(p.name for p in root.glob("*.txt")) if root.exists() else []
        raise CMAPSSMissing(
            f"C-MAPSS subset {subset} not found in {root}. "
            f"Present files: {available or 'none'}. "
            "Run: python backend/scripts/download_datasets.py --dataset cmapss"
        )
    return files


def available_subsets() -> list[str]:
    root = subset_dir()
    if not root.exists():
        return []
    return [s for s in SUBSETS if (root / f"train_{s}.txt").is_file()]


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name}: expected {len(COLUMNS)} columns, found {frame.shape[1]}")
    frame.columns = COLUMNS
    return frame


def load_train(subset: str = "FD001") -> pd.DataFrame:
    """Training data with a piecewise-constant RUL label attached."""
    frame = _read(find_files(subset)["train"])
    frame = frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    frame["rul"] = frame.groupby("unit")["cycle"].transform(lambda c: c.max() - c)
    frame["rul"] = frame["rul"].clip(upper=MAX_RUL)
    return frame


def load_test(subset: str = "FD001") -> tuple[pd.DataFrame, pd.Series]:
    """Official test cycles plus the published ground-truth RUL per unit."""
    frame = _read(find_files(subset)["test"])
    frame = frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    truth = pd.read_csv(find_files(subset)["rul"], sep=r"\s+", header=None)[0]
    truth.index = frame.groupby("unit")["cycle"].max().index
    return frame, truth


def usable_sensors() -> list[str]:
    return [c for c in COLUMNS if c not in CONSTANT_SENSORS and c not in {"unit", "cycle"}]


def build_windows(
    frame: pd.DataFrame,
    window: int = WINDOW_SIZE,
    sensor_cols: list[str] | None = None,
    with_label: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Sliding windows per engine unit.

    Returns ``(X, y, unit_ids, cycles)`` where ``X`` has shape
    ``(n_windows, window, n_features)``.

    A unit contributes ``max(0, len(unit) - window + 1)`` windows. Units shorter
    than the window are skipped rather than zero-padded, because padding would
    invent degradation trajectories that never occurred.
    """
    sensor_cols = sensor_cols or usable_sensors()
    X: list[np.ndarray] = []
    y: list[float] = []
    units: list[str] = []
    cycles: list[int] = []

    for unit, group in frame.groupby("unit", sort=True):
        values = group[sensor_cols].to_numpy(dtype=np.float32)
        if len(values) < window:
            continue
        for start in range(len(values) - window + 1):
            X.append(values[start : start + window])
            units.append(str(unit))
            cycles.append(int(group["cycle"].iloc[start + window - 1]))
            if with_label:
                y.append(float(group["rul"].iloc[start + window - 1]))

    if not X:
        raise ValueError(
            f"No windows generated: window={window} exceeds every engine length."
        )
    return (
        np.stack(X),
        np.asarray(y, dtype=np.float32),
        units,
        np.asarray(cycles, dtype=np.int32),
    )


def split_by_unit(
    unit_ids: np.ndarray, test_fraction: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks splitting *whole engine units* into train and test.

    Splitting rows of one run-to-failure trajectory would leak near-identical
    cycles across the boundary, so the split is by unit.
    """
    unique = np.unique(unit_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_test = max(1, int(round(len(unique) * test_fraction)))
    test_units = set(shuffled[:n_test].tolist())
    test_mask = np.isin(unit_ids, list(test_units))
    return ~test_mask, test_mask


def sequence_for_unit(
    subset: str, unit: str | int, up_to_cycle: int | None = None
) -> pd.DataFrame:
    """Cycles of one engine up to and including ``up_to_cycle``."""
    frame = load_train(subset)
    unit_str = str(unit)
    group = frame[frame["unit"].astype(str) == unit_str]
    if group.empty:
        available = sorted(frame["unit"].astype(str).unique().tolist())
        raise KeyError(f"Engine unit {unit_str} not in {subset}. Available: {available[:20]}")
    if up_to_cycle is not None:
        group = group[group["cycle"] <= int(up_to_cycle)]
    return group.reset_index(drop=True)


def unit_inventory(subset: str = "FD001") -> list[dict[str, object]]:
    frame = load_train(subset)
    return [
        {"unit": str(u), "max_cycle": int(g["cycle"].max()), "final_rul": float(g["rul"].iloc[-1])}
        for u, g in frame.groupby("unit", sort=True)
    ]


def subset_info(subset: str = "FD001") -> CMAPSSSubsetInfo:
    train = load_train(subset)
    test, _ = load_test(subset)
    return CMAPSSSubsetInfo(
        subset=subset,
        train_units=int(train["unit"].nunique()),
        test_units=int(test["unit"].nunique()),
        train_rows=int(len(train)),
        max_cycle=int(train["cycle"].max()),
    )
