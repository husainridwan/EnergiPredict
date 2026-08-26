"""The model zoo, hyperparameter spaces, and a time-respecting stacking ensemble.

Three things here differ from the project's first version, each of which changes
the numbers:

**Cross-validation folds respect time.** ``RandomizedSearchCV`` defaults to
``KFold``, which shuffles. Tuning under a shuffled fold selects the
hyperparameters that best exploit autocorrelation -- deep trees that memorise
neighbouring hours -- and those are the wrong hyperparameters for forecasting.
Every search here uses :class:`~sklearn.model_selection.TimeSeriesSplit`, so each
fold trains on a past and validates on its future.

**The scoring function is a regression metric.** The original tuner defaulted to
``scoring="f1"``, a classification metric, so the search was optimising something
undefined for this problem.

**Stacking blends across time, not across shuffled folds.** ``StackingRegressor``
builds its meta-features with ``cross_val_predict``, which needs folds that
partition the data -- something ``TimeSeriesSplit`` deliberately does not do. Its
default ``KFold`` would hand the meta-learner base predictions made on shuffled
folds, reintroducing exactly the leakage the rest of the pipeline removes. So
stacking is done explicitly against a chronological blend window; see
:class:`ChronologicalStackingRegressor`.
**Estimators are single-threaded, and parallelism is taken at the search level.**
Every model here is configured with one thread. That looks wrong and is not: see
the measurement in :mod:`energipredict.__init__`. Nested parallelism -- a search
running several fits at once, each fanning out across every core -- oversubscribes
the CPU badly enough to cost an order of magnitude. The search parallelises over
whole fits instead, using processes.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as cfg

__all__ = [
    "ChronologicalStackingRegressor",
    "STACK_BASE_NAMES",
    "learned_models",
    "search_jobs",
    "time_series_cv",
    "tune",
]

#: Threads per estimator. One, deliberately -- see the module docstring.
MODEL_THREADS = 1


def search_jobs() -> int:
    """How many fits to run concurrently during a hyperparameter search.

    Each fit is single-threaded, so this is the only place parallelism is taken.
    Capped below the visible core count because the visible count is not the
    available count in a container, and because leaving headroom is what stops
    this turning back into the oversubscription it replaced.

    Override with ``ENERGIPREDICT_SEARCH_JOBS``.
    """
    override = os.environ.get("ENERGIPREDICT_SEARCH_JOBS")
    if override:
        return max(1, int(override))
    return max(1, min(4, (os.cpu_count() or 2) // 2))

#: Which tuned models go into the ensemble. The three gradient-boosting
#: implementations differ enough in how they grow trees and handle
#: regularisation that blending them adds something; adding RandomForest as a
#: fourth mostly duplicates what they already capture.
STACK_BASE_NAMES = ("XGBoost", "LightGBM", "CatBoost")


def time_series_cv(n_splits: int = 4) -> TimeSeriesSplit:
    """Expanding-window folds: each validates on the period after its training.

    Chosen over a blocked or sliding window because it mirrors how the model will
    actually be used -- retrained periodically on everything recorded so far.
    """
    return TimeSeriesSplit(n_splits=n_splits)


def learned_models(random_state: int = cfg.RANDOM_STATE) -> dict[str, tuple[BaseEstimator, dict]]:
    """Estimators and their search spaces, keyed by display name.

    The linear models are wrapped in a scaler. Without it, ``solar_radiation``
    (hundreds of W/m^2) and ``is_weekend`` (0 or 1) enter Ridge's penalty on
    wildly different scales, so the regularisation strength means something
    different for each coefficient and the baseline is weaker than the method
    really is. Tree models are scale-invariant and need no such wrapper.
    """
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    return {
        "Linear regression": (
            Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
            {},  # nothing to tune
        ),
        "Ridge regression": (
            Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=random_state, n_jobs=MODEL_THREADS),
            {
                "n_estimators": [200, 400],
                "max_depth": [10, 20, None],
                "min_samples_leaf": [1, 2, 5],
                "max_features": [0.5, 0.8, 1.0],
            },
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=random_state),
            {
                "n_estimators": [200, 400],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [3, 5],
                "subsample": [0.8, 1.0],
            },
        ),
        "XGBoost": (
            XGBRegressor(
                random_state=random_state,
                n_jobs=MODEL_THREADS,
                tree_method="hist",
                objective="reg:squarederror",
            ),
            {
                "n_estimators": [300, 600],
                "learning_rate": [0.02, 0.05, 0.1],
                "max_depth": [4, 6, 8],
                "subsample": [0.7, 0.9],
                "colsample_bytree": [0.7, 0.9],
                "min_child_weight": [1, 5],
                "reg_lambda": [1.0, 5.0],
            },
        ),
        "LightGBM": (
            LGBMRegressor(
                random_state=random_state, n_jobs=MODEL_THREADS, verbosity=-1
            ),
            {
                "n_estimators": [300, 600],
                "learning_rate": [0.02, 0.05, 0.1],
                "num_leaves": [31, 63, 127],
                "min_child_samples": [10, 20, 40],
                "subsample": [0.7, 0.9],
                "subsample_freq": [1],
                "colsample_bytree": [0.7, 0.9],
            },
        ),
        "CatBoost": (
            CatBoostRegressor(
                random_state=random_state,
                verbose=0,
                allow_writing_files=False,
                thread_count=MODEL_THREADS,
            ),
            {
                "iterations": [400, 800],
                "learning_rate": [0.03, 0.05, 0.1],
                "depth": [4, 6, 8],
                "l2_leaf_reg": [1.0, 3.0, 9.0],
            },
        ),
    }


def tune(
    estimator: BaseEstimator,
    param_dist: dict,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int = 10,
    n_splits: int = 4,
    random_state: int = cfg.RANDOM_STATE,
) -> tuple[BaseEstimator, dict, float]:
    """Randomised search with expanding-window folds, scored on RMSE.

    Returns
    -------
    (fitted_estimator, best_params, cv_rmse)
        ``cv_rmse`` is the mean across folds, as a positive number. For an
        estimator with nothing to tune, the search is skipped and the CV score is
        computed directly, so the returned triple has the same meaning either way.
    """
    if not param_dist:
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(
            clone(estimator),
            X,
            y,
            cv=time_series_cv(n_splits),
            scoring="neg_root_mean_squared_error",
            n_jobs=search_jobs(),
        )
        fitted = clone(estimator).fit(X, y)
        return fitted, {}, float(-scores.mean())

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=time_series_cv(n_splits),
        random_state=random_state,
        refit=True,
        error_score="raise",
        n_jobs=search_jobs(),
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(-search.best_score_)


class ChronologicalStackingRegressor(RegressorMixin, BaseEstimator):
    """Stacking whose meta-features come from a held-back future window.

    ``fit`` proceeds in four steps:

    1. Cut the training data in two by time: an earlier *base* window and a later
       *blend* window (the last ``blend_frac`` of rows).
    2. Fit each base estimator on the base window only.
    3. Predict the blend window with those estimators. Because the blend window is
       strictly after everything the bases saw, these are genuine out-of-sample
       predictions -- which is the entire point of a meta-feature. Fit the final
       estimator on them.
    4. Refit the base estimators on all of the training data, so that at predict
       time they use every hour available rather than only the base window.

    Step 4 leaves a small inconsistency: the meta-learner was calibrated against
    bases trained on less data than the bases it will actually be combining, so
    its weights are fitted to slightly noisier inputs than it receives. This is
    the same compromise ``sklearn``'s own stacking makes, and it is the sound
    direction to err -- the alternative, fitting the meta-learner on in-sample
    base predictions, would make it trust whichever base overfits hardest.

    Parameters
    ----------
    estimators
        ``(name, estimator)`` pairs. Cloned, so the originals are untouched.
    final_estimator
        The meta-learner. Defaults to ridge regression, which is the standard
        choice: base predictions are highly correlated with one another, and the
        L2 penalty keeps that collinearity from producing wild blend weights.
    blend_frac
        Fraction of the training rows, taken from the end, used to fit the
        meta-learner.
    """

    def __init__(
        self,
        estimators: list[tuple[str, BaseEstimator]],
        final_estimator: BaseEstimator | None = None,
        blend_frac: float = 0.2,
    ):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.blend_frac = blend_frac

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit bases, blend on a future window, then refit bases on everything."""
        if not 0 < self.blend_frac < 1:
            raise ValueError(f"blend_frac must lie in (0, 1), got {self.blend_frac}")
        if not self.estimators:
            raise ValueError("no base estimators supplied")

        X = pd.DataFrame(X)
        y = pd.Series(y)
        n = len(X)
        n_blend = int(n * self.blend_frac)
        if n_blend < 2:
            raise ValueError(
                f"blend window is {n_blend} rows; need at least 2. "
                f"Raise blend_frac or supply more training data."
            )
        cut = n - n_blend

        X_base, y_base = X.iloc[:cut], y.iloc[:cut]
        X_blend, y_blend = X.iloc[cut:], y.iloc[cut:]

        meta = np.empty((n_blend, len(self.estimators)), dtype=float)
        for j, (_, est) in enumerate(self.estimators):
            provisional = clone(est).fit(X_base, y_base)
            meta[:, j] = provisional.predict(X_blend)

        self.final_estimator_ = clone(self.final_estimator or Ridge(alpha=1.0))
        self.final_estimator_.fit(meta, y_blend)

        self.estimators_ = [(name, clone(est).fit(X, y)) for name, est in self.estimators]
        self.named_estimators_ = dict(self.estimators_)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Combine the base predictions using the fitted blend weights."""
        if not hasattr(self, "estimators_"):
            raise RuntimeError("call fit before predict")
        X = pd.DataFrame(X)
        meta = np.column_stack([est.predict(X) for _, est in self.estimators_])
        return self.final_estimator_.predict(meta)

    @property
    def blend_weights_(self) -> dict[str, float]:
        """Meta-learner coefficient per base model, for interpretation.

        Worth looking at: a near-zero weight means that base model contributes
        nothing the others do not already provide, and could be dropped from the
        deployed ensemble to cut inference cost.
        """
        if not hasattr(self, "final_estimator_"):
            raise RuntimeError("call fit before reading blend weights")
        coefs = getattr(self.final_estimator_, "coef_", None)
        if coefs is None:
            return {}
        return {name: float(w) for (name, _), w in zip(self.estimators_, np.ravel(coefs))}
