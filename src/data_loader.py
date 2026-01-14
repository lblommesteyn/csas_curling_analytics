"""
Reusable data loading helpers for the CSAS mixed doubles curling analysis.

All CSV files are expected to live in the repository root and use UTF-8 encoding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = {
    "competitions": "Competition.csv",
    "competitors": "Competitors.csv",
    "ends": "Ends.csv",
    "games": "Games.csv",
    "stones": "Stones.csv",
    "teams": "Teams.csv",
}


def load_csv(
    key: str,
    *,
    dtype: Optional[Mapping[str, Any]] = None,
    parse_dates: Optional[list[str]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read one of the challenge CSV files into a DataFrame.

    Parameters
    ----------
    key:
        Logical name of the CSV (must exist in DATA_FILES).
    dtype:
        Optional dtype overrides passed to pandas.
    parse_dates:
        Column names that should be parsed as dates.
    **kwargs:
        Additional arguments passed to pd.read_csv.
    """

    if key not in DATA_FILES:
        raise KeyError(f"Unknown dataset '{key}'. Expected one of {sorted(DATA_FILES)}.")

    csv_path = REPO_ROOT / DATA_FILES[key]
    return pd.read_csv(csv_path, dtype=dtype, parse_dates=parse_dates, **kwargs)


def load_all() -> dict[str, pd.DataFrame]:
    """Convenience wrapper returning every dataset keyed by logical name."""

    return {key: load_csv(key) for key in DATA_FILES}


def get_data_path(filename: str) -> Path:
    """Return an absolute path to one of the raw CSV files."""

    csv_path = REPO_ROOT / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"{filename} was not found relative to {REPO_ROOT}")
    return csv_path
