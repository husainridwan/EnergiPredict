"""Regression metrics for hourly building-energy forecasts.

Four numbers are reported for every model, because each hides something the
others reveal:

RMSE
    Squares the errors, so it is dominated by the worst hours. Those are the
    hours that matter for a demand charge, so this is the headline.
MAE
    Linear in error, so it describes a typical hour. Much lower than RMSE means
    the error distribution is heavy-tailed -- accurate most of the time, badly
    wrong occasionally.
R-squared
    Variance explained relative to predicting the test-set mean. Useful for
    intuition, and treacherous on time series: a naive forecast scores highly
    here purely because the diurnal cycle is most of the variance. It can also go
    negative, which simply means the model is worse than a flat line.
CV(RMSE)
    RMSE as a fraction of mean load. This is the criterion ASHRAE Guideline 14
    actually uses to decide whether an hourly model is fit for measurement and
    verification work, with a threshold of 30%. It is the only one of the four
    that answers "is this good enough to use".

MAPE is deliberately absent. It divides by the actual value, and HVAC load in an
unoccupied building at 3am approaches zero, so a handful of near-zero hours would
dominate the average and the resulting percentage would describe those hours
rather than the model.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

__all__ = ["ASHRAE_CVRMSE_THRESHOLD", "regression_metrics"]

#: ASHRAE Guideline 14 acceptance threshold for hourly calibrated models, as a
#: fraction. A model above this is not considered fit for M&V purposes.
ASHRAE_CVRMSE_THRESHOLD = 0.30


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute RMSE, MAE, R-squared and CV(RMSE).

    Parameters
    ----------
    y_true, y_pred
        Equal-length arrays. Non-finite predictions are treated as an error
        rather than dropped: a model that emits NaN for some hours has failed on
        those hours, and quietly excluding them would flatter it.

    Returns
    -------
    dict
        ``rmse``, ``mae``, ``r2``, ``cvrmse`` (fraction), ``cvrmse_pct``, and
        ``meets_ashrae_g14``.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("cannot score an empty series")
    if not np.isfinite(y_pred).all():
        n_bad = int((~np.isfinite(y_pred)).sum())
        raise ValueError(
            f"{n_bad} of {y_pred.size} predictions are not finite; the model "
            "failed on those hours rather than merely erring on them"
        )
    if not np.isfinite(y_true).all():
        raise ValueError("y_true contains non-finite values")

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    mean_load = float(y_true.mean())
    cvrmse = rmse / mean_load if mean_load != 0 else float("nan")

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "cvrmse": round(cvrmse, 4),
        "cvrmse_pct": round(100 * cvrmse, 2),
        "meets_ashrae_g14": bool(cvrmse <= ASHRAE_CVRMSE_THRESHOLD),
    }
