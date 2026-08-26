"""Loading a trained artefact and scoring an upload.

This module exists to close the gap that the previous version of the app had.
``main.py`` used to call ``model.predict(df)`` on whatever CSV arrived, while the
notebook built the target, dropped two columns and engineered nothing. Those two
column orders agreed only by luck, and when they disagreed the failure was a
``KeyError`` in a 500 response rather than a message telling the caller what was
wrong with their file.

Here, serving calls the same :func:`energipredict.features.build_features` that
training called, against the :class:`~energipredict.features.FeatureSpec` recorded
inside the artefact. If the columns do not match, that is a bug rather than a
possibility to handle.

What an upload must contain
---------------------------
Because the forecaster uses lagged consumption, scoring is not row-independent:
predicting 3pm needs the meter history leading up to it. So an upload is a
*history*, not a list of independent cases. It needs

* ``date`` -- hourly timestamps, ascending
* ``hvac_N`` and ``hvac_S`` (or ``hvac_total``) -- past consumption
* the weather columns in :data:`~energipredict.config.WEATHER_COLS_KEPT`

and it must be long enough to fill the longest lag window: 168 hours of history
before the first row that can be predicted, plus the horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as cfg
from .data import build_target
from .features import build_features

__all__ = [
    "MIN_HISTORY_HOURS",
    "Artefact",
    "InvalidUpload",
    "load_artefact",
    "predict_frame",
    "required_upload_columns",
]

#: Rows of history needed before the first predictable hour. Set by the longest
#: window in the feature set -- the 168-hour rolling statistics.
MIN_HISTORY_HOURS = max(max(cfg.TARGET_LAGS_H), max(cfg.TARGET_ROLLING_H))


class InvalidUpload(ValueError):
    """The uploaded file cannot be scored, with a reason the caller can act on.

    Carries ``detail`` so the API can return something more useful than
    ``KeyError: 'intTemp'``: which columns are missing, or how many more hours of
    history are needed.
    """

    def __init__(self, message: str, *, detail: dict | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


@dataclass
class Artefact:
    """A trained model together with the feature contract it was trained under."""

    name: str
    estimator: object
    horizon_h: int
    target: str
    feature_spec: dict
    validation: dict
    test: dict

    @property
    def columns(self) -> list[str]:
        return list(self.feature_spec["columns"])

    def describe(self) -> dict:
        """Public metadata for ``GET /api/model``."""
        return {
            "name": self.name,
            "horizon_h": self.horizon_h,
            "horizon_label": f"{self.horizon_h}-hour-ahead forecast",
            "target": self.target,
            "target_units": cfg.TARGET_UNITS_SHORT,
            "n_features": len(self.columns),
            "test_metrics": self.test,
            "min_history_hours": MIN_HISTORY_HOURS,
            "required_upload_columns": required_upload_columns(),
        }


def required_upload_columns() -> dict[str, list[str]]:
    """Columns an upload must provide, grouped by why they are needed."""
    return {
        "timestamp": [cfg.TIMESTAMP_COL],
        "consumption_history": [*cfg.HVAC_METER_COLS],
        "consumption_history_alternative": [cfg.TARGET],
        "weather": list(cfg.WEATHER_COLS_KEPT),
    }


@lru_cache(maxsize=4)
def load_artefact(path: str | None = None) -> Artefact:
    """Load a model bundle, cached so it is read from disk once per process.

    Parameters
    ----------
    path
        Defaults to ``models/deployed.joblib``. A string rather than a ``Path``
        so the result is hashable and therefore cacheable.
    """
    target = Path(path) if path else cfg.MODELS_DIR / "deployed.joblib"
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found. Train the models first:\n"
            f"    python -m energipredict.train"
        )
    bundle = joblib.load(target)
    missing = {"estimator", "feature_spec", "horizon_h"} - set(bundle)
    if missing:
        raise ValueError(
            f"{target} is not an EnergiPredict artefact (missing {sorted(missing)}). "
            "Artefacts from before v2.0 stored a bare estimator with no feature "
            "contract and cannot be served safely; retrain to regenerate them."
        )
    return Artefact(
        name=bundle.get("name", "unknown"),
        estimator=bundle["estimator"],
        horizon_h=bundle["horizon_h"],
        target=bundle.get("target", cfg.TARGET),
        feature_spec=bundle["feature_spec"],
        validation=bundle.get("validation", {}),
        test=bundle.get("test", {}),
    )


def _prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an uploaded frame and put it on a complete hourly index.

    Raises
    ------
    InvalidUpload
        With a ``detail`` payload naming exactly what is wrong.
    """
    if df.empty:
        raise InvalidUpload("The uploaded file has no rows.")

    if cfg.TIMESTAMP_COL not in df.columns:
        raise InvalidUpload(
            f"Missing the {cfg.TIMESTAMP_COL!r} column, so the rows cannot be "
            "placed in time. A forecast needs timestamped history.",
            detail={"missing": [cfg.TIMESTAMP_COL], "found": list(df.columns)},
        )

    out = df.copy()
    try:
        out[cfg.TIMESTAMP_COL] = pd.to_datetime(out[cfg.TIMESTAMP_COL])
    except (ValueError, TypeError) as exc:
        raise InvalidUpload(
            f"Could not read {cfg.TIMESTAMP_COL!r} as timestamps. Expected a "
            "format like '2020-03-01 14:00:00'.",
            detail={"parse_error": str(exc)},
        ) from exc

    has_meters = all(c in out.columns for c in cfg.HVAC_METER_COLS)
    has_total = cfg.TARGET in out.columns
    if not (has_meters or has_total):
        raise InvalidUpload(
            "Missing past consumption. Supply either both HVAC submeters "
            f"({', '.join(cfg.HVAC_METER_COLS)}) or a precomputed "
            f"{cfg.TARGET!r} column -- the forecast is built from recent load, "
            "so it cannot be made from weather alone.",
            detail={
                "need_either": [list(cfg.HVAC_METER_COLS), [cfg.TARGET]],
                "found": list(out.columns),
            },
        )

    missing_weather = [c for c in cfg.WEATHER_COLS_KEPT if c not in out.columns]
    if missing_weather:
        raise InvalidUpload(
            f"Missing weather column(s): {', '.join(missing_weather)}.",
            detail={"missing": missing_weather, "found": list(out.columns)},
        )

    if has_meters:
        out = build_target(out, drop_meters=True)

    out = out.sort_values(cfg.TIMESTAMP_COL)
    duplicates = int(out[cfg.TIMESTAMP_COL].duplicated().sum())
    if duplicates:
        out = out.groupby(cfg.TIMESTAMP_COL, as_index=False).mean(numeric_only=True)

    out = out.set_index(cfg.TIMESTAMP_COL).sort_index()

    full = pd.date_range(out.index.min(), out.index.max(), freq=cfg.FREQ)
    if len(full) < 2:
        raise InvalidUpload(
            "The upload covers a single hour. A forecast needs a run of history; "
            f"at least {MIN_HISTORY_HOURS} hours."
        )
    out = out.reindex(full)
    out.index.name = cfg.TIMESTAMP_COL

    # Short gaps are bridged the same way training bridged them; longer ones are
    # left missing so the affected rows drop out rather than being invented.
    for col in (cfg.TARGET, *cfg.WEATHER_COLS_KEPT):
        if col in out.columns:
            out[col] = out[col].interpolate(method="time", limit=3)

    return out


def predict_frame(
    df: pd.DataFrame,
    artefact: Artefact | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Forecast every hour in an uploaded history that has enough context.

    Returns
    -------
    (predictions, summary)
        ``predictions`` is indexed by timestamp with a ``prediction`` column, and
        an ``actual`` and ``error`` column for hours where the true value is also
        present -- which lets a caller check the model against their own data
        instead of taking the published metrics on trust.

    Notes
    -----
    Early rows are necessarily unpredictable: the first
    :data:`MIN_HISTORY_HOURS` hours have no lag history inside the file. They are
    reported in ``summary["rows_without_history"]`` rather than silently dropped,
    because a caller who uploads 200 rows and gets 32 predictions back deserves to
    know why.
    """
    artefact = artefact or load_artefact()
    history = _prepare_history(df)

    X, y, spec = build_features(history, artefact.horizon_h, dropna=True)

    if X.empty:
        span = len(history)
        raise InvalidUpload(
            f"Not enough history to forecast. The file spans {span} hours; "
            f"the model needs {MIN_HISTORY_HOURS} hours of unbroken history "
            f"before the first hour it can predict, plus the {artefact.horizon_h}-hour "
            "horizon. Upload a longer run, or one with fewer gaps.",
            detail={
                "hours_supplied": span,
                "hours_required": MIN_HISTORY_HOURS + artefact.horizon_h,
                "horizon_h": artefact.horizon_h,
            },
        )

    expected = artefact.columns
    if list(X.columns) != expected:
        raise RuntimeError(
            "Feature mismatch between training and serving. This is a bug, not a "
            f"bad upload.\n  expected: {expected}\n  built:    {list(X.columns)}"
        )

    predictions = np.asarray(artefact.estimator.predict(X), dtype=float)

    # HVAC consumption cannot be negative. A tree ensemble can extrapolate below
    # zero on unusual inputs, and a negative kW figure in the output would be
    # visibly wrong to any facilities engineer reading it.
    clipped = int((predictions < 0).sum())
    predictions = np.clip(predictions, 0.0, None)

    out = pd.DataFrame({"prediction": np.round(predictions, 3)}, index=X.index)
    out.index.name = cfg.TIMESTAMP_COL

    known = y.reindex(out.index)
    if known.notna().any():
        out["actual"] = known.round(3)
        out["error"] = (out["prediction"] - out["actual"]).round(3)

    summary: dict = {
        "model": artefact.name,
        "horizon_h": artefact.horizon_h,
        "target": artefact.target,
        "units": cfg.TARGET_UNITS_SHORT,
        "rows_uploaded": int(len(df)),
        "hours_spanned": int(len(history)),
        "rows_predicted": int(len(out)),
        "rows_without_history": int(spec.rows_dropped),
        "negative_predictions_clipped": clipped,
        "first_prediction": f"{out.index.min():%Y-%m-%d %H:%M}",
        "last_prediction": f"{out.index.max():%Y-%m-%d %H:%M}",
    }
    if "actual" in out:
        from .metrics import regression_metrics

        summary["scored_against_upload"] = regression_metrics(
            out["actual"], out["prediction"]
        )

    return out, summary
