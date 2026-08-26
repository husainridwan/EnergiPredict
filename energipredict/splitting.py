"""Chronological train/validation/test splitting.

The project's earlier evaluation used ``train_test_split(..., random_state=42)``,
which shuffles. On hourly building data that is not a small methodological
blemish -- it is the whole result. Consecutive hours are strongly autocorrelated,
so a shuffled split places 2pm in training and 3pm the same day in test. The
model is then scored on rows whose neighbours it memorised, and every algorithm
looks good, the flexible ones most of all.

Splitting by time removes that. Training sees the earliest stretch, validation
the next, test the last, and the test window is genuinely a future the model has
never seen -- which is the only condition under which a reported error bar means
anything about next week's electricity bill.

Note that lag features at the start of the test window are drawn from the end of
the validation window. That is not leakage: it is the operational reality that a
forecaster issued on Tuesday knows Monday's meter reading. What must not cross
the boundary is the *target being predicted*, and it does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config as cfg

__all__ = ["Split", "chronological_split"]


@dataclass
class Split:
    """Three contiguous, time-ordered slices of a design matrix."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    def bounds(self) -> dict[str, dict[str, str | int | float]]:
        """Date range, row count and target distribution of each slice.

        The mean and standard deviation are here because they are needed to read
        the results honestly. R-squared is measured against the variance of the
        slice it is computed on, so a test period that happens to be calmer than
        the training period depresses R-squared even when absolute error is
        unchanged -- the denominator shrank, not the model's skill. Reporting the
        per-slice spread alongside the metrics makes that visible instead of
        leaving it to be discovered.
        """
        out: dict[str, dict[str, str | int | float]] = {}
        for name, y in (
            ("train", self.y_train),
            ("val", self.y_val),
            ("test", self.y_test),
        ):
            out[name] = {
                "start": f"{y.index.min():%Y-%m-%d %H:%M}",
                "end": f"{y.index.max():%Y-%m-%d %H:%M}",
                "rows": len(y),
                "target_mean": round(float(y.mean()), 3),
                "target_std": round(float(y.std()), 3),
            }
        return out

    def summary(self) -> str:
        b = self.bounds()
        return " | ".join(
            f"{k}: {v['start'][:10]}..{v['end'][:10]} ({v['rows']:,})"
            for k, v in b.items()
        )


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_frac: float = cfg.TRAIN_FRAC,
    val_frac: float = cfg.VAL_FRAC,
) -> Split:
    """Cut ``X``/``y`` into train/validation/test by position in time.

    Fractions are of the row count, not of the calendar span, so an outage in the
    middle of the record does not silently hand one slice most of the data.

    Raises
    ------
    ValueError
        If the fractions leave nothing for test, or if the index is not sorted --
        an unsorted index would make "chronological" a lie while still returning
        three plausible-looking frames.
    """
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1:
        raise ValueError("train_frac and val_frac must lie in [0, 1)")
    if train_frac + val_frac >= 1:
        raise ValueError(
            f"train_frac + val_frac = {train_frac + val_frac} leaves no test set"
        )
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows, y has {len(y)}")
    if not X.index.equals(y.index):
        raise ValueError("X and y must share an index")
    if not X.index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending for a chronological split")

    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    tr = slice(0, n_train)
    va = slice(n_train, n_train + n_val)
    te = slice(n_train + n_val, n)

    return Split(
        X_train=X.iloc[tr],
        X_val=X.iloc[va],
        X_test=X.iloc[te],
        y_train=y.iloc[tr],
        y_val=y.iloc[va],
        y_test=y.iloc[te],
    )
