"""Paths, column groups and split configuration for EnergiPredict.

The important content of this module is the column grouping. EnergiPredict is a
*forecaster*: it predicts HVAC energy for a future hour, using only information
that is actually available when the forecast is made. That rules out most of the
columns in ``data.csv``, and the distinction is easy to lose once the data is a
DataFrame, so it is written down here once and imported everywhere else.

Two groups are admissible:

``WEATHER_COLS``
    Nearby weather-station channels. A facilities team has these ahead of time
    as a forecast, so a model may use them.

``TIME_FEATURES`` (built in :mod:`energipredict.features`)
    Calendar position. Known arbitrarily far ahead.

Everything else is excluded, for one of two reasons:

``BMS_COLS``
    Building-management and rooftop-unit sensors -- supply/return/mixed air
    temperatures, fan state, hot-water temperatures, indoor temperature. These
    are measured at time *t*. Using them to predict *t* means the answer is
    already on the sensor bus; there is nothing to forecast.

``NON_HVAC_METER_COLS``
    Plug-load and lighting meters for the same building. Same problem: they are
    read at time *t*, not known in advance.

Lagged values of the target itself *are* admissible, but only back to the
forecast horizon -- see :func:`energipredict.features.build_features`.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths. Everything is derived from the repository root so the pipeline runs
# from any working directory, and on any machine, without editing a path.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_JSON = REPORTS_DIR / "metrics.json"

# --------------------------------------------------------------------------
# Raw schema
# --------------------------------------------------------------------------
TIMESTAMP_COL = "date"
FREQ = "h"  # data.csv is hourly

TARGET = "hvac_total"

#: Units of the HVAC submeters. The source dataset records these channels as
#: average electrical power over the interval, in kW. Because the interval is one
#: hour, the number is numerically equal to the energy consumed in that hour in
#: kWh -- so "kW" and "kWh/h" are interchangeable labels here, and RMSE in these
#: units reads directly as an error in kWh per hour.
TARGET_UNITS = "kW (hourly mean, numerically equal to kWh consumed in the hour)"
TARGET_UNITS_SHORT = "kW"

#: The two HVAC submeters that sum to the target. The north and south wings of
#: the building are metered separately; total HVAC energy is their sum.
HVAC_METER_COLS = ("hvac_N", "hvac_S")

#: Weather-station channels, in Synoptic/MesoWest naming. Forecastable.
#:
#: ``air_temp_set_2`` and ``dew_point_temperature_set_1d`` are retained here
#: because they are legitimately available in advance; whether they survive
#: correlation pruning is a modelling decision, made in
#: :data:`WEATHER_COLS_KEPT`, not an availability one.
WEATHER_COLS = (
    "air_temp_set_1",
    "air_temp_set_2",
    "dew_point_temperature_set_1d",
    "relative_humidity_set_1",
    "solar_radiation_set_1",
)

#: Weather channels actually fed to the models. ``air_temp_set_2`` is a second
#: thermometer at the same station and tracks ``air_temp_set_1`` almost exactly;
#: dew point is recoverable from temperature and relative humidity. Dropping
#: both removes near-duplicate columns without losing information.
WEATHER_COLS_KEPT = (
    "air_temp_set_1",
    "relative_humidity_set_1",
    "solar_radiation_set_1",
)

#: Building-management / rooftop-unit sensors. Excluded: measured at time t.
BMS_COLS = (
    "intTemp",
    "extTemp",
    "airSpeed",
    "waterHeat",
    "hp_hws_temp",
    "rtuSat",
    "rtuRat",
    "rtuMat",
    "rtuOat",
    "rtuFan",
)

#: Non-HVAC electrical submeters. Excluded: measured at time t.
NON_HVAC_METER_COLS = ("mels_S", "lig_S", "mels_N")

#: Columns dropped from the modelling frame, with the reason recorded so the
#: exclusion survives into the written-up results.
EXCLUDED_COLS: dict[str, str] = {
    **{c: "bms-sensor: measured at time t, not knowable in advance" for c in BMS_COLS},
    **{
        c: "non-hvac meter: measured at time t, not knowable in advance"
        for c in NON_HVAC_METER_COLS
    },
    "air_temp_set_2": "near-duplicate of air_temp_set_1",
    "dew_point_temperature_set_1d": "recoverable from air_temp and relative_humidity",
}

# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
#: Balance-point temperature for heating/cooling degree hours, in degrees C.
#: 18 C is the conventional value in building-energy work.
BALANCE_POINT_C = 18.0

#: Target lags offered to the model, in hours, before horizon filtering.
#: t-1/t-2/t-3 carry short-run thermal state; t-24 and t-48 the diurnal cycle;
#: t-168 the same hour of the previous week, which is what actually encodes
#: weekday-vs-weekend occupancy.
TARGET_LAGS_H = (1, 2, 3, 24, 48, 168)

#: Rolling-window widths, in hours, for mean/std/min/max of the target.
TARGET_ROLLING_H = (24, 168)

#: Lags applied to outdoor air temperature, in hours. Building thermal mass
#: means the load at time t responds to weather over the preceding hours, not
#: only to weather at t.
WEATHER_LAGS_H = (1, 2, 3, 24)

# --------------------------------------------------------------------------
# Forecast horizons
# --------------------------------------------------------------------------
#: Hours ahead. A horizon of h means: standing at time t, predict t+h. Only
#: lags of h or more are legal, because at t you do not yet know t+h-1.
#:
#: 1  -- next-hour forecast. The horizon at which a persistence baseline is
#:       strongest, and so the honest one to be judged against.
#: 24 -- day-ahead forecast. The horizon the scheduling use case needs.
HORIZONS_H = (1, 24)
DEPLOYED_HORIZON_H = 24

# --------------------------------------------------------------------------
# Chronological split
# --------------------------------------------------------------------------
#: Fractions of the timeline, in order. No shuffling: adjacent hourly rows are
#: strongly autocorrelated, so a random split leaks the test set into training
#: and inflates every score. Validation sits between train and test in time,
#: which is also the order in which they are used.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test takes the remainder

RANDOM_STATE = 42
