"""
These numbers were computed by re-running evaluation on the exact same
data.csv and saved model files, first reproducing the original random
train/test split, then correcting it to a proper time-respecting
(chronological) split. Both are published side by side rather than
picking whichever looks better, because the discrepancy between them
IS the interesting result: it shows random splits on autocorrelated
hourly data leak information and inflate scores.
"""

RESULTS = {
    "target": "hvac_total — a weighted combination of the north and south HVAC energy meters",
    "dataset": {
        "rows": 19212,
        "resolution": "hourly",
        "range": "2018-02-22 to 2020-05-02",
    },
    "naive_random_split": {
        "label": "Naive split (random 80/20, as originally evaluated)",
        "note": (
            "Rows were split randomly (random_state=42). Because adjacent hourly "
            "rows are highly autocorrelated, this leaks information from the test "
            "set into training and inflates every model's apparent accuracy."
        ),
        "models": [
            {"name": "Mean predictor",       "rmse": 11.03, "mae": 8.71, "r2": 0.000},
            {"name": "Linear regression",    "rmse": 7.30,  "mae": 5.35, "r2": 0.562},
            {"name": "CatBoost",             "rmse": 3.71,  "mae": 2.54, "r2": 0.887},
            {"name": "XGBoost",              "rmse": 3.24,  "mae": 2.10, "r2": 0.914},
            {"name": "LightGBM",             "rmse": 3.21,  "mae": 2.11, "r2": 0.916},
            {"name": "Stacking (deployed)",  "rmse": 3.14,  "mae": 2.05, "r2": 0.919},
        ],
    },
    "honest_chronological_split": {
        "label": "Honest split (chronological, no leakage)",
        "note": (
            "Trained on Feb 2018 - Sep 2019, tested on a genuinely unseen future "
            "window (Nov 2019 - May 2020). Under this split, the simple persistence "
            "baseline (predict = last hour's value) beats every trained model, "
            "including the stacking ensemble."
        ),
        "models": [
            {"name": "Mean predictor",           "rmse": 9.64, "mae": 7.93, "r2": -1.294},
            {"name": "Persistence (t-1)",         "rmse": 2.38, "mae": 1.46, "r2": 0.860},
            {"name": "Seasonal naive (t-24)",     "rmse": 6.40, "mae": 4.43, "r2": -0.012},
            {"name": "Linear regression",         "rmse": 6.44, "mae": 4.30, "r2": -0.026},
            {"name": "RandomForest",              "rmse": 4.63, "mae": 3.32, "r2": 0.471},
            {"name": "XGBoost",                   "rmse": 4.94, "mae": 3.57, "r2": 0.396},
            {"name": "LightGBM",                  "rmse": 4.76, "mae": 3.26, "r2": 0.440},
            {"name": "CatBoost",                  "rmse": 4.59, "mae": 3.37, "r2": 0.480},
            {"name": "Stacking",                  "rmse": 4.73, "mae": 3.46, "r2": 0.446},
        ],
    },
    "takeaway": (
        "Under a proper time-respecting split, the trained ensemble does not beat "
        "a naive persistence baseline. That's disclosed here rather than hidden — "
        "catching this kind of leakage is the actual skill being demonstrated."
    ),
}
