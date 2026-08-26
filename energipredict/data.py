"""Loading ``data.csv``, building the target, and putting the series on a
regular hourly grid.

The last of those matters more than it sounds. Every lag feature downstream is
computed with ``Series.shift(k)``, which shifts by *k rows*. That equals a shift
of *k hours* only if the index has no missing hours. If the raw file skips a
timestamp -- and it does -- then an unrepaired frame silently gives ``shift(24)``
a value from 23 hours ago, or from three days ago across a long outage, and the
model is trained on features that will never occur in production.

So :func:`load_dataset` reindexes onto a complete hourly range before any
feature is built, and reports what it found instead of hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config as cfg

__all__ = ["DataReport", "load_dataset", "load_raw", "build_target"]


@dataclass
class DataReport:
    """What loading the data actually did, for the write-up.

    Carrying this alongside the frame keeps the README honest: the row counts
    and imputation totals quoted there come from here rather than from memory.
    """

    rows_in_file: int
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    expected_hourly_rows: int
    missing_timestamps: int
    duplicate_timestamps: int
    target_interpolated: int
    target_unrecoverable: int
    weather_interpolated: dict[str, int] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """Fraction of the hourly grid present in the raw file."""
        return self.rows_in_file / self.expected_hourly_rows

    def to_dict(self) -> dict:
        return {
            "rows_in_file": self.rows_in_file,
            "first_timestamp": str(self.first_timestamp),
            "last_timestamp": str(self.last_timestamp),
            "expected_hourly_rows": self.expected_hourly_rows,
            "missing_timestamps": self.missing_timestamps,
            "duplicate_timestamps": self.duplicate_timestamps,
            "coverage": round(self.coverage, 4),
            "target_interpolated": self.target_interpolated,
            "target_unrecoverable": self.target_unrecoverable,
            "weather_interpolated": self.weather_interpolated,
        }

    def summary(self) -> str:
        return (
            f"{self.rows_in_file:,} rows, "
            f"{self.first_timestamp:%Y-%m-%d} to {self.last_timestamp:%Y-%m-%d}, "
            f"{self.coverage:.1%} of the hourly grid present "
            f"({self.missing_timestamps:,} hours absent). "
            f"Target: {self.target_interpolated:,} hours interpolated, "
            f"{self.target_unrecoverable:,} left unusable."
        )


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read ``data.csv`` with a parsed, sorted, timezone-naive hourly index.

    Parameters
    ----------
    path
        Defaults to ``data.csv`` beside the repository root. Passing an explicit
        path is only for tests and for scoring a held-out file.
    """
    path = Path(path) if path is not None else cfg.DATA_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The dataset lives at the repository root as "
            f"data.csv; see the README for its provenance."
        )

    df = pd.read_csv(path)
    if cfg.TIMESTAMP_COL not in df.columns:
        raise ValueError(
            f"expected a {cfg.TIMESTAMP_COL!r} column, found: {list(df.columns)}"
        )

    df[cfg.TIMESTAMP_COL] = pd.to_datetime(df[cfg.TIMESTAMP_COL])
    return df.sort_values(cfg.TIMESTAMP_COL).reset_index(drop=True)


def build_target(df: pd.DataFrame, *, drop_meters: bool = True) -> pd.DataFrame:
    """Add ``hvac_total`` as the sum of the north and south HVAC submeters.

    The building meters its two wings separately, so total HVAC energy is
    ``hvac_N + hvac_S``.

    An earlier version of this project used a self-weighted combination,
    ``(S/(S+N))*S + (N/(S+N))*N``, which simplifies to ``(S**2 + N**2)/(S + N)``.
    That is not a total: it is bounded below by the mean of the two meters and
    above by their max, it is non-linear in both, and it carries no physical
    unit. Results computed against it are not comparable to a building's
    metered HVAC consumption, which is what the forecast is for.

    Parameters
    ----------
    drop_meters
        Remove the two component meters afterwards. They are dropped by default
        because keeping them makes the target trivially recoverable -- a model
        given ``hvac_N`` and ``hvac_S`` has been handed the answer.
    """
    missing = [c for c in cfg.HVAC_METER_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing HVAC submeter column(s): {missing}")

    out = df.copy()
    out[cfg.TARGET] = out[list(cfg.HVAC_METER_COLS)].sum(axis=1, min_count=len(cfg.HVAC_METER_COLS))
    if drop_meters:
        out = out.drop(columns=list(cfg.HVAC_METER_COLS))
    return out


def load_dataset(
    path: str | Path | None = None,
    *,
    max_gap_h: int = 3,
) -> tuple[pd.DataFrame, DataReport]:
    """Load, build the target, and regularise onto a complete hourly index.

    Gaps up to ``max_gap_h`` hours are filled by time interpolation, which is
    reasonable for a physical quantity sampled hourly. Longer outages are left
    as ``NaN``: linear interpolation across a multi-day gap manufactures data
    that was never measured, and a model trained on it is partly fitting an
    interpolation artefact. Rows whose target is still missing after this are
    excluded from training and scoring by
    :func:`energipredict.features.build_features`, but they remain in the frame
    so that the hourly grid -- and therefore every lag -- stays correct.

    Returns
    -------
    (frame, report)
        ``frame`` is indexed by timestamp at hourly frequency.
    """
    raw = load_raw(path)
    rows_in_file = len(raw)

    duplicates = int(raw[cfg.TIMESTAMP_COL].duplicated().sum())
    if duplicates:
        # Averaging repeats beats keeping an arbitrary one; either way the count
        # is reported rather than silently absorbed.
        raw = raw.groupby(cfg.TIMESTAMP_COL, as_index=False).mean(numeric_only=True)

    df = build_target(raw)
    df = df.set_index(cfg.TIMESTAMP_COL).sort_index()

    full_index = pd.date_range(df.index.min(), df.index.max(), freq=cfg.FREQ)
    missing_timestamps = int(len(full_index) - len(df.index))
    df = df.reindex(full_index)
    df.index.name = cfg.TIMESTAMP_COL

    target_missing_before = int(df[cfg.TARGET].isna().sum())
    df[cfg.TARGET] = df[cfg.TARGET].interpolate(method="time", limit=max_gap_h)
    target_unrecoverable = int(df[cfg.TARGET].isna().sum())

    weather_interpolated: dict[str, int] = {}
    for col in cfg.WEATHER_COLS:
        if col not in df.columns:
            continue
        before = int(df[col].isna().sum())
        df[col] = df[col].interpolate(method="time", limit=max_gap_h)
        weather_interpolated[col] = before - int(df[col].isna().sum())

    report = DataReport(
        rows_in_file=rows_in_file,
        first_timestamp=full_index[0],
        last_timestamp=full_index[-1],
        expected_hourly_rows=len(full_index),
        missing_timestamps=missing_timestamps,
        duplicate_timestamps=duplicates,
        target_interpolated=target_missing_before - target_unrecoverable,
        target_unrecoverable=target_unrecoverable,
        weather_interpolated=weather_interpolated,
    )
    return df, report
