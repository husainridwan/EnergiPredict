"""Train, evaluate, and publish every number the README and the app quote.

Run as::

    python -m energipredict.train                 # both horizons, full search
    python -m energipredict.train --quick         # small search, for a smoke test
    python -m energipredict.train --horizons 24   # just the day-ahead model

Outputs
-------
``models/h{H}/{name}.joblib``
    One artefact per model per horizon. Each bundles the fitted estimator with
    the feature spec it was trained against, so serving cannot silently drift
    from training.
``models/deployed.joblib``
    A copy of the model the API loads, chosen on validation RMSE.
``reports/metrics.json``
    Every measured number: dataset provenance, split boundaries, feature list,
    and validation and test metrics for every model and baseline. The README and
    the ``/api/results`` endpoint both read from here, so a stale figure in the
    write-up is not possible -- there is one source.

Nothing in this script prints a number it has not just computed.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import Ridge

from . import config as cfg
from .baselines import baselines_for_horizon
from .data import DataReport, load_dataset
from .features import FeatureSpec, build_features
from .metrics import ASHRAE_CVRMSE_THRESHOLD, regression_metrics
from .models import (
    STACK_BASE_NAMES,
    ChronologicalStackingRegressor,
    learned_models,
    tune,
)
from .splitting import Split, chronological_split

FAMILIES = {
    "Mean predictor": "baseline",
    "Linear regression": "linear",
    "Ridge regression": "linear",
    "RandomForest": "tree",
    "GradientBoosting": "tree",
    "XGBoost": "tree",
    "LightGBM": "tree",
    "CatBoost": "tree",
    "Stacking ensemble": "ensemble",
}


def _family(name: str) -> str:
    if name in FAMILIES:
        return FAMILIES[name]
    if name.startswith(("Persistence", "Seasonal naive")):
        return "baseline"
    return "other"


@dataclass
class Scored:
    """One model, fitted and scored on validation and test."""

    name: str
    estimator: BaseEstimator
    val: dict[str, float]
    test: dict[str, float]
    best_params: dict
    cv_rmse: float | None
    fit_seconds: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "family": _family(self.name),
            "validation": self.val,
            "test": self.test,
            "best_params": _jsonable(self.best_params),
            "cv_rmse": self.cv_rmse,
            "fit_seconds": round(self.fit_seconds, 2),
        }


def _jsonable(obj: Any) -> Any:
    """Coerce numpy scalars and non-serialisable values for ``json.dump``."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _score(
    name: str,
    estimator: BaseEstimator,
    split: Split,
    *,
    best_params: dict | None = None,
    cv_rmse: float | None = None,
    fit_seconds: float = 0.0,
) -> Scored:
    """Score an already-fitted estimator on the validation and test slices."""
    return Scored(
        name=name,
        estimator=estimator,
        val=regression_metrics(split.y_val, estimator.predict(split.X_val)),
        test=regression_metrics(split.y_test, estimator.predict(split.X_test)),
        best_params=best_params or {},
        cv_rmse=cv_rmse,
        fit_seconds=fit_seconds,
    )


def evaluate_horizon(
    df: pd.DataFrame,
    horizon_h: int,
    *,
    n_iter: int,
    n_splits: int,
    skip: tuple[str, ...] = (),
    verbose: bool = True,
) -> tuple[list[Scored], FeatureSpec, Split, dict[str, float]]:
    """Fit and score every baseline and model at one forecast horizon."""

    def say(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    X, y, spec = build_features(df, horizon_h)
    split = chronological_split(X, y)

    say(f"\n{'=' * 78}\nHORIZON: {horizon_h}h ahead\n{'=' * 78}")
    say(f"  features : {len(spec.columns)}")
    say(f"  lags used: {spec.lags_used}  withheld: {spec.lags_withheld}")
    say(f"  rows     : {spec.rows_after:,} (dropped {spec.rows_dropped:,} for missing history)")
    say(f"  split    : {split.summary()}")

    scored: list[Scored] = []

    say("\n-- baselines --")
    for name, est in baselines_for_horizon(horizon_h, spec.columns).items():
        t0 = time.perf_counter()
        fitted = clone(est).fit(split.X_train, split.y_train)
        result = _score(name, fitted, split, fit_seconds=time.perf_counter() - t0)
        scored.append(result)
        say(f"  {name:26s} val RMSE {result.val['rmse']:7.3f}   test RMSE {result.test['rmse']:7.3f}")

    say("\n-- tuned models --")
    tuned: dict[str, BaseEstimator] = {}
    for name, (estimator, param_dist) in learned_models().items():
        if name in skip:
            say(f"  {name:26s} skipped")
            continue
        t0 = time.perf_counter()
        fitted, best_params, cv_rmse = tune(
            estimator,
            param_dist,
            split.X_train,
            split.y_train,
            n_iter=n_iter,
            n_splits=n_splits,
        )
        elapsed = time.perf_counter() - t0
        tuned[name] = fitted
        result = _score(
            name, fitted, split, best_params=best_params, cv_rmse=cv_rmse, fit_seconds=elapsed
        )
        scored.append(result)
        say(
            f"  {name:26s} val RMSE {result.val['rmse']:7.3f}   "
            f"test RMSE {result.test['rmse']:7.3f}   ({elapsed:.0f}s)"
        )

    bases = [(n, tuned[n]) for n in STACK_BASE_NAMES if n in tuned]
    blend_weights: dict[str, float] = {}
    if len(bases) >= 2:
        say("\n-- stacking ensemble --")
        t0 = time.perf_counter()
        stack = ChronologicalStackingRegressor(
            estimators=[(n, clone(e)) for n, e in bases],
            final_estimator=Ridge(alpha=1.0),
            blend_frac=0.2,
        ).fit(split.X_train, split.y_train)
        elapsed = time.perf_counter() - t0
        blend_weights = stack.blend_weights_
        result = _score("Stacking ensemble", stack, split, fit_seconds=elapsed)
        scored.append(result)
        say(
            f"  {'Stacking ensemble':26s} val RMSE {result.val['rmse']:7.3f}   "
            f"test RMSE {result.test['rmse']:7.3f}   ({elapsed:.0f}s)"
        )
        say(f"  blend weights: " + ", ".join(f"{k} {v:+.3f}" for k, v in blend_weights.items()))
    else:
        say("\n-- stacking skipped: fewer than two base models available --")

    return scored, spec, split, blend_weights


def leakage_demonstration(
    df: pd.DataFrame,
    horizon_h: int = 1,
    *,
    n_iter: int = 4,
    n_splits: int = 3,
) -> dict:
    """Quantify what a shuffled split costs, holding everything else fixed.

    One model, one feature set, two splits: the same LightGBM is tuned and scored
    first under a random 80/20 split and then under a chronological one. Any
    difference is attributable to the split, because nothing else varies.

    This is the project's central methodological finding, so it is measured here
    rather than asserted in prose.
    """
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import train_test_split

    X, y, _ = build_features(df, horizon_h)
    estimator, param_dist = learned_models()["LightGBM"]

    Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
        X, y, test_size=0.2, random_state=cfg.RANDOM_STATE, shuffle=True
    )
    shuffled_fit, _, _ = tune(
        estimator, param_dist, Xtr_s, ytr_s, n_iter=n_iter, n_splits=n_splits
    )
    shuffled = regression_metrics(yte_s, shuffled_fit.predict(Xte_s))

    n_train = int(len(X) * 0.8)
    chrono_fit, _, _ = tune(
        estimator,
        param_dist,
        X.iloc[:n_train],
        y.iloc[:n_train],
        n_iter=n_iter,
        n_splits=n_splits,
    )
    chronological = regression_metrics(y.iloc[n_train:], chrono_fit.predict(X.iloc[n_train:]))

    inflation = (
        round(100 * (chronological["rmse"] - shuffled["rmse"]) / chronological["rmse"], 1)
        if chronological["rmse"]
        else None
    )
    return {
        "model": "LightGBM",
        "horizon_h": horizon_h,
        "note": (
            "Identical features and identical hyperparameter search; only the "
            "split differs. The shuffled split places adjacent hours on both "
            "sides of the boundary, so the test rows are neighbours of training "
            "rows and the score reflects interpolation rather than forecasting."
        ),
        "shuffled_random_split": shuffled,
        "chronological_split": chronological,
        "rmse_understated_by_pct": inflation,
    }


def _persist(scored: list[Scored], spec: FeatureSpec, horizon_h: int, deployed: str) -> None:
    """Write one joblib artefact per model, plus the deployed copy."""
    out_dir = cfg.MODELS_DIR / f"h{horizon_h}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for result in scored:
        if _family(result.name) == "baseline":
            continue  # a baseline is a lookup, not a fitted artefact worth 40 MB
        bundle = {
            "name": result.name,
            "estimator": result.estimator,
            "horizon_h": horizon_h,
            "target": cfg.TARGET,
            "feature_spec": spec.to_dict(),
            "validation": result.val,
            "test": result.test,
            "sklearn_version": __import__("sklearn").__version__,
            "python_version": platform.python_version(),
        }
        slug = result.name.lower().replace(" ", "_")
        joblib.dump(bundle, out_dir / f"{slug}.joblib")

    winner = next(r for r in scored if r.name == deployed)
    joblib.dump(
        {
            "name": winner.name,
            "estimator": winner.estimator,
            "horizon_h": horizon_h,
            "target": cfg.TARGET,
            "feature_spec": spec.to_dict(),
            "validation": winner.val,
            "test": winner.test,
            "sklearn_version": __import__("sklearn").__version__,
            "python_version": platform.python_version(),
        },
        cfg.MODELS_DIR / "deployed.joblib",
    )


def _target_stats(df: pd.DataFrame) -> dict:
    y = df[cfg.TARGET].dropna()
    return {
        "name": cfg.TARGET,
        "definition": " + ".join(cfg.HVAC_METER_COLS),
        "units": cfg.TARGET_UNITS,
        "mean": round(float(y.mean()), 3),
        "std": round(float(y.std()), 3),
        "min": round(float(y.min()), 3),
        "max": round(float(y.max()), 3),
        "n": int(y.size),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m energipredict.train",
        description="Train and evaluate the EnergiPredict forecasters.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(cfg.HORIZONS_H),
        help=f"forecast horizons in hours (default: {list(cfg.HORIZONS_H)})",
    )
    parser.add_argument("--n-iter", type=int, default=8, help="randomised search iterations")
    parser.add_argument("--n-splits", type=int, default=3, help="TimeSeriesSplit folds")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="model names to skip, e.g. --skip GradientBoosting",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="tiny search and no GradientBoosting; for checking the pipeline runs",
    )
    parser.add_argument(
        "--no-leakage-demo",
        action="store_true",
        help="skip the shuffled-vs-chronological comparison",
    )
    args = parser.parse_args(argv)

    n_iter = 2 if args.quick else args.n_iter
    n_splits = 2 if args.quick else args.n_splits
    skip = tuple(args.skip) + (("GradientBoosting",) if args.quick else ())

    started = time.perf_counter()
    print("Loading data ...", flush=True)
    df, report = load_dataset()
    print(f"  {report.summary()}", flush=True)

    horizons: dict[str, dict] = {}
    for horizon_h in args.horizons:
        scored, spec, split, blend_weights = evaluate_horizon(
            df, horizon_h, n_iter=n_iter, n_splits=n_splits, skip=skip
        )

        learned = [r for r in scored if _family(r.name) != "baseline"]
        deployed = min(learned, key=lambda r: r.val["rmse"]).name
        best_baseline = min(
            (r for r in scored if _family(r.name) == "baseline" and r.name != "Mean predictor"),
            key=lambda r: r.test["rmse"],
            default=None,
        )
        best_model = min(learned, key=lambda r: r.test["rmse"])

        verdict = None
        if best_baseline is not None:
            improvement = 100 * (best_baseline.test["rmse"] - best_model.test["rmse"]) / best_baseline.test["rmse"]
            verdict = {
                "best_model": best_model.name,
                "best_baseline": best_baseline.name,
                "test_rmse_improvement_pct": round(improvement, 1),
                "beats_baseline": bool(improvement > 0),
            }

        _persist(scored, spec, horizon_h, deployed)

        horizons[str(horizon_h)] = {
            "horizon_h": horizon_h,
            "feature_spec": spec.to_dict(),
            "split": split.bounds(),
            "models": [r.to_dict() for r in scored],
            "deployed": deployed,
            "blend_weights": _jsonable(blend_weights),
            "verdict": verdict,
        }
        print(f"\n  deployed at h={horizon_h}: {deployed} (chosen on validation RMSE)", flush=True)
        if verdict:
            direction = "beats" if verdict["beats_baseline"] else "loses to"
            print(
                f"  {best_model.name} {direction} {best_baseline.name} on test RMSE "
                f"by {abs(verdict['test_rmse_improvement_pct'])}%",
                flush=True,
            )

    payload: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "energipredict_version": __import__("energipredict").__version__,
        "dataset": report.to_dict(),
        "target": _target_stats(df),
        "ashrae_g14_cvrmse_threshold": ASHRAE_CVRMSE_THRESHOLD,
        "search": {"n_iter": n_iter, "n_splits": n_splits, "skipped": list(skip)},
        "horizons": horizons,
        "deployed_horizon_h": cfg.DEPLOYED_HORIZON_H,
    }

    if not args.no_leakage_demo:
        print("\nMeasuring what a shuffled split costs ...", flush=True)
        payload["leakage_demonstration"] = leakage_demonstration(
            df, horizon_h=1, n_iter=max(2, n_iter // 2), n_splits=n_splits
        )
        demo = payload["leakage_demonstration"]
        print(
            f"  shuffled RMSE {demo['shuffled_random_split']['rmse']} vs "
            f"chronological {demo['chronological_split']['rmse']} "
            f"-- shuffling understates error by {demo['rmse_understated_by_pct']}%",
            flush=True,
        )

    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nWrote {cfg.METRICS_JSON.relative_to(cfg.PROJECT_ROOT)} "
        f"in {time.perf_counter() - started:.0f}s total",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
