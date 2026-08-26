"""EnergiPredict: hourly HVAC energy forecasting for large auditoria.

The pipeline is deliberately split so that the API and the notebook share one
definition of every step, rather than each having its own copy that drifts:

:mod:`~energipredict.config`      paths, column groups, the availability contract
:mod:`~energipredict.data`        loading, target construction, hourly regularisation
:mod:`~energipredict.features`    horizon-aware feature building
:mod:`~energipredict.splitting`   chronological train/val/test split
:mod:`~energipredict.baselines`   naive forecasts the models must beat
:mod:`~energipredict.models`      model zoo, search spaces, time-respecting stacking
:mod:`~energipredict.metrics`     RMSE / MAE / R-squared / CV(RMSE)
:mod:`~energipredict.train`       the training and evaluation entry point
:mod:`~energipredict.serve`       loading an artefact and scoring an upload

Retrain and regenerate every published number with::

    python -m energipredict.train
"""

from __future__ import annotations

import os

# Cap the native thread pools before numpy, LightGBM, XGBoost or CatBoost are
# imported -- libgomp reads these at load time, so setting them later has no
# effect.
#
# This is not a micro-optimisation. On a machine whose visible CPU count exceeds
# its real quota -- a container, or WSL2, where this project is developed -- the
# boosting libraries spawn one OpenMP thread per logical core and then thrash.
# Measured here on a 13,330 x 36 training matrix, one LightGBM fit took 2.9 s
# single-threaded and 50.6 s with ``n_jobs=-1``: seventeen times *slower* for
# asking for more parallelism. Nested parallelism is the trap -- a
# hyperparameter search running k fits at once, each spawning n threads, wants
# k x n cores and gets contention instead.
#
# So each estimator is single-threaded (see :mod:`energipredict.models`) and
# parallelism is taken at the search level, where the unit of work is a whole
# model fit and joblib uses processes. Override by setting these in the
# environment; a value already present is left alone.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

__version__ = "2.0.0"

from . import baselines, config, data, features, metrics, models, splitting

__all__ = [
    "__version__",
    "baselines",
    "config",
    "data",
    "features",
    "metrics",
    "models",
    "splitting",
]
