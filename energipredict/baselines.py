"""Naive baselines, expressed as scikit-learn estimators.

A forecasting result is meaningless without these. Hourly HVAC load is strongly
autocorrelated and strongly diurnal, so "yesterday at this hour" is already a
decent forecast, and a trained ensemble that cannot beat it has not earned its
place -- whatever its R-squared says.

Both persistence and seasonal-naive forecasts turn out to be *feature lookups*:
``hvac_lag_24h`` is exactly what a seasonal-naive forecaster predicts. So rather
than special-casing them in the evaluation loop, they are estimators that echo a
single column of the design matrix. That has a useful side effect -- it makes it
obvious that the baselines and the ensemble see the same information, so a win by
the ensemble is a win on modelling rather than on access to data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

__all__ = ["ColumnEchoRegressor", "baselines_for_horizon"]


class ColumnEchoRegressor(RegressorMixin, BaseEstimator):
    """Predict the target by echoing one feature column unchanged.

    With ``column="hvac_lag_1h"`` this is a persistence forecast; with
    ``hvac_lag_24h`` a seasonal-naive one; with ``hvac_lag_168h`` a
    same-hour-last-week one. It fits nothing, which is the point.

    Parameters
    ----------
    column
        Name of the column to echo. Must exist in the frames passed to
        :meth:`fit` and :meth:`predict`.
    """

    def __init__(self, column: str):
        self.column = column

    def fit(self, X, y=None):  # noqa: D102 - trivial, see class docstring
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"{type(self).__name__} needs a DataFrame to look up "
                f"{self.column!r}; got {type(X).__name__}"
            )
        if self.column not in X.columns:
            raise ValueError(
                f"column {self.column!r} not in the design matrix. Available "
                f"lag columns: {[c for c in X.columns if 'lag' in c]}"
            )
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:  # noqa: D102
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"expected a DataFrame, got {type(X).__name__}")
        if self.column not in X.columns:
            raise ValueError(f"column {self.column!r} not in the design matrix")
        return X[self.column].to_numpy(dtype=float)


def baselines_for_horizon(horizon_h: int, available: list[str]) -> dict[str, BaseEstimator]:
    """Baselines that are legitimate at ``horizon_h``.

    A baseline is only offered if the column it echoes survived horizon
    filtering. At a 24-hour horizon there is no ``hvac_lag_1h``, so a
    next-hour persistence forecast is not available to the baseline either --
    withholding it from the models but granting it to the baseline would be
    just as dishonest in the other direction.

    ``available`` is the design matrix's column list, usually
    ``FeatureSpec.columns``.

    Note that at ``horizon_h=24`` persistence and the 24-hour seasonal naive
    forecast are the same predictor, so only one entry is returned.
    """
    from sklearn.dummy import DummyRegressor

    candidates: dict[str, BaseEstimator] = {
        # Predicts the training-set mean. Anchors R-squared at ~0 and shows how
        # much of the variance is simply the diurnal cycle.
        "Mean predictor": DummyRegressor(strategy="mean"),
    }

    echoed: set[str] = set()

    def offer(label: str, column: str) -> None:
        if column in available and column not in echoed:
            candidates[label] = ColumnEchoRegressor(column)
            echoed.add(column)

    offer(f"Persistence (t-{horizon_h})", f"hvac_lag_{horizon_h}h")
    for period in (24, 168):
        offer(f"Seasonal naive (t-{period})", f"hvac_lag_{period}h")

    return candidates
