"""Horizon-aware feature construction.

Every row is keyed by the timestamp being predicted, *t*. The forecast is issued
at ``t - horizon``, and a column may only appear in the row if its value is known
by then. That single rule decides the whole feature set:

=========================  ======================================================
Weather at *t*             admissible -- a weather forecast for *t* exists at
                           ``t - horizon``
Weather before *t*         admissible -- already observed
Calendar position of *t*   admissible -- known indefinitely ahead
Target at ``t - k``        admissible **only when k >= horizon**
Rolling target statistics  admissible over windows ending at ``t - horizon``
=========================  ======================================================

The third row is the one that quietly decides whether a result is real. At a
horizon of 24 hours the model cannot see ``hvac_total`` at ``t-1``: standing at
9am on Monday forecasting 9am Tuesday, 8am Tuesday has not happened. Offering
``lag_1`` there would leak 23 hours of the future into the features and produce
an accuracy that evaporates in deployment. So lags below the horizon are removed
rather than merely discouraged, and :func:`feature_columns` is what the API
validates uploads against, so serving cannot drift from training.

One assumption is worth stating plainly, because it flatters the model: weather
columns are the *observed* values at *t*, standing in for a forecast of *t*. Real
day-ahead weather forecasts carry error, so measured skill here is an upper bound
on deployed skill. Quantifying that would need archived forecasts, which this
dataset does not include.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg

__all__ = ["FeatureSpec", "build_features", "feature_columns"]


@dataclass
class FeatureSpec:
    """The feature contract for one horizon.

    Persisted next to each trained model so serving reproduces training exactly,
    and so the column list in the write-up is generated rather than transcribed.
    """

    horizon_h: int
    columns: list[str]
    target: str = cfg.TARGET
    excluded: dict[str, str] = field(default_factory=lambda: dict(cfg.EXCLUDED_COLS))
    lags_used: list[int] = field(default_factory=list)
    lags_withheld: list[int] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0

    @property
    def rows_dropped(self) -> int:
        return self.rows_before - self.rows_after

    def to_dict(self) -> dict:
        return {
            "horizon_h": self.horizon_h,
            "target": self.target,
            "n_features": len(self.columns),
            "columns": self.columns,
            "lags_used": self.lags_used,
            "lags_withheld": self.lags_withheld,
            "excluded_columns": self.excluded,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_dropped": self.rows_dropped,
        }


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar position, plain and cyclically encoded.

    Both encodings are kept. Gradient-boosted trees split happily on the integer
    hour and do not need the sine pair; the linear baseline does, and without it
    that baseline is handicapped by an arbitrary discontinuity between hour 23
    and hour 0 rather than by its own limitations. Comparing against a baseline
    one has quietly crippled proves nothing.
    """
    out = pd.DataFrame(index=index)
    out["hour"] = index.hour
    out["dayofweek"] = index.dayofweek
    out["month"] = index.month
    out["dayofyear"] = index.dayofyear
    out["is_weekend"] = (index.dayofweek >= 5).astype("int8")

    out["hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * index.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * index.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * index.dayofweek / 7)
    out["doy_sin"] = np.sin(2 * np.pi * index.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * index.dayofyear / 365.25)
    return out


def _weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Weather channels, degree hours, and short lags for thermal mass.

    Degree hours are the standard building-energy transform: cooling load
    responds to how far the outdoor temperature sits *above* a balance point,
    not to the temperature itself, and heating to how far below. A tree can
    approximate the hinge with enough splits; giving it directly costs one
    column and spends the model's capacity on something else.

    The temperature lags exist because a building has thermal mass. The load at
    2pm reflects heat that entered the envelope through the morning, so weather
    over the preceding hours predicts it better than weather at 2pm alone.
    """
    out = pd.DataFrame(index=df.index)
    for col in cfg.WEATHER_COLS_KEPT:
        if col not in df.columns:
            raise ValueError(f"weather column {col!r} absent from frame")
        out[col] = df[col]

    temp = df["air_temp_set_1"]
    out["cooling_degree_hours"] = (temp - cfg.BALANCE_POINT_C).clip(lower=0)
    out["heating_degree_hours"] = (cfg.BALANCE_POINT_C - temp).clip(lower=0)

    for lag in cfg.WEATHER_LAGS_H:
        out[f"air_temp_lag_{lag}h"] = temp.shift(lag)

    # Windows end at t: admissible, since weather at t is itself admissible.
    out["air_temp_roll_mean_24h"] = temp.rolling(24, min_periods=24).mean()
    out["solar_roll_mean_24h"] = (
        df["solar_radiation_set_1"].rolling(24, min_periods=24).mean()
    )
    return out


def _target_history_features(y: pd.Series, horizon_h: int) -> tuple[pd.DataFrame, list[int], list[int]]:
    """Lags and rolling statistics of the target, filtered by horizon.

    Returns the frame along with which lags were used and which were withheld,
    so the exclusion is visible in the results rather than buried here.
    """
    out = pd.DataFrame(index=y.index)

    used = [lag for lag in cfg.TARGET_LAGS_H if lag >= horizon_h]
    withheld = [lag for lag in cfg.TARGET_LAGS_H if lag < horizon_h]
    for lag in used:
        out[f"hvac_lag_{lag}h"] = y.shift(lag)

    # Every window ends at t-horizon, the most recent observation available when
    # the forecast is issued.
    known = y.shift(horizon_h)
    for window in cfg.TARGET_ROLLING_H:
        roll = known.rolling(window, min_periods=window)
        out[f"hvac_roll_mean_{window}h"] = roll.mean()
        out[f"hvac_roll_std_{window}h"] = roll.std()
        out[f"hvac_roll_min_{window}h"] = roll.min()
        out[f"hvac_roll_max_{window}h"] = roll.max()

    return out, used, withheld


def _check_hourly(index: pd.DatetimeIndex) -> None:
    """Refuse a frame whose index is not a complete hourly run.

    Lags are row shifts, so one absent hour silently redefines every lag in the
    rows that follow it. Better to fail here than to train on it.
    """
    if len(index) < 2:
        return  # nothing to infer from; the stub in feature_columns lands here
    if index.freq is not None and index.freq.freqstr.lower() in {"h", "1h"}:
        return
    deltas = index.to_series().diff().dropna().unique()
    if len(deltas) == 1 and deltas[0] == pd.Timedelta(hours=1):
        return
    raise ValueError(
        "df must be on a complete hourly index; found gaps or irregular spacing "
        f"({len(deltas)} distinct step sizes). Load via "
        "energipredict.data.load_dataset, which reindexes onto a full hourly range."
    )


def build_features(
    df: pd.DataFrame,
    horizon_h: int = 1,
    *,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series, FeatureSpec]:
    """Build the design matrix and target for a given forecast horizon.

    Parameters
    ----------
    df
        Hourly-indexed frame from :func:`energipredict.data.load_dataset`. The
        index must be complete and gap-free -- lags are row shifts, so a hole in
        the index silently changes what every lag means.
    horizon_h
        Hours ahead. Target lags shorter than this are withheld.
    dropna
        Drop rows with a missing target or any missing feature. The leading
        ``max(lags + windows)`` rows always go, since their history predates the
        record; a handful more go where the source data has an unfilled outage.
        Set ``False`` only to inspect the raw feature frame.

    Returns
    -------
    (X, y, spec)
    """
    if horizon_h < 1:
        raise ValueError(f"horizon_h must be >= 1, got {horizon_h}")
    if cfg.TARGET not in df.columns:
        raise ValueError(
            f"{cfg.TARGET!r} absent; call energipredict.data.build_target first"
        )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must be indexed by timestamp")

    _check_hourly(df.index)

    y_raw = df[cfg.TARGET]
    history, lags_used, lags_withheld = _target_history_features(y_raw, horizon_h)

    X = pd.concat(
        [_calendar_features(df.index), _weather_features(df), history],
        axis=1,
    )

    rows_before = len(X)
    if dropna:
        keep = X.notna().all(axis=1) & y_raw.notna()
        X, y = X.loc[keep], y_raw.loc[keep]
    else:
        y = y_raw

    spec = FeatureSpec(
        horizon_h=horizon_h,
        columns=list(X.columns),
        lags_used=lags_used,
        lags_withheld=lags_withheld,
        rows_before=rows_before,
        rows_after=len(X),
    )
    return X, y, spec


def feature_columns(horizon_h: int = 1) -> list[str]:
    """Feature names for a horizon, without touching the data.

    Used by the API to check an upload before loading a model, and to tell the
    caller which columns are missing.
    """
    stub_index = pd.date_range("2020-01-01", periods=3, freq=cfg.FREQ)
    stub = pd.DataFrame(
        {c: [0.0] * 3 for c in (*cfg.WEATHER_COLS_KEPT, cfg.TARGET)}, index=stub_index
    )
    _, _, spec = build_features(stub, horizon_h, dropna=False)
    return spec.columns
