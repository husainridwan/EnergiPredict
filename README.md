# EnergiPredict — HVAC Energy Consumption Prediction

Predicting HVAC energy consumption in large university auditoriums, so that
the largest controllable line in a campus electricity bill can be forecast
before it is incurred rather than reconciled after.

Final year project, B.Sc. Electrical & Electronics Engineering,
University of Lagos.

---

## The Problem

Electricity is a substantial and rising cost for the University of Lagos, and
large halls consume a disproportionate share of it. HVAC is the biggest
controllable component of that load — lighting and equipment are close to
fixed, but cooling responds to weather, occupancy and setpoint decisions.

If HVAC consumption can be predicted from conditions that are known in
advance, then setpoints and scheduling become decisions with a visible cost
attached, instead of guesses audited by the next bill.

Existing approaches lean on deep reinforcement learning and model-predictive
control. Both work, both are computationally expensive, and both tend to be
tuned to one building. The question here was narrower: how far do standard
gradient-boosted ensembles get, on hardware a facilities team already has?

---

## Data, and an honest limitation up front

**The model is trained on proxy data, not on the auditoriums themselves.**

No instrumented consumption history existed for the University of Lagos
auditoriums, so this study used a public dataset from a building with
comparable characteristics — [DATASET NAME AND SOURCE], covering
[N] observations at [hourly/15-min] resolution.

That substitution is the single largest constraint on this work, and it costs
two things:

1. **Absolute predictions do not transfer.** The reported errors describe the
   source building. Applied to a UNILAG auditorium, the model would need
   recalibration against locally metered data before any number it produces
   should be trusted.
2. **Climate and usage differ.** The source building's weather profile and
   occupancy pattern are not Lagos's. Coefficients learned on one do not
   automatically hold on the other.

What does transfer is the method and the ranking of approaches. Treat this as
a demonstrated pipeline awaiting local data, not as a calibrated model of
UNILAG.

**Features** — interior and exterior zone temperatures, rooftop unit supply,
mixed and return air temperatures, heating load, lighting load,
[plus any temporal features].

**Target** — [NAME], measured in [UNITS].
Mean [X], standard deviation [Y].

---

## Approach

Five models, in increasing complexity, each evaluated on the same
chronological split:

| Model | Why it's here |
|---|---|
| XGBoost | Strong tabular baseline |
| LightGBM | Faster on this feature count; different split strategy |
| CatBoost | Handles the categorical/temporal features without manual encoding |
| Random Forest | Bagged comparison against the boosted family |
| **Stacking** | Combines the above; a meta-learner weights their disagreements |

The split is **chronological, not random** — neighbouring timestamps are
correlated, and a random split leaks future into past and inflates every score.

---

## Results

Error is reported as RMSE, in the target's own units, alongside baselines.
MSE appears only for continuity with the original write-up.

| Model | RMSE | MAE | R² | CV(RMSE) |
|---|---|---|---|---|
| Mean predictor | [ ] | [ ] | 0.00 | [ ] |
| Persistence (t−1) | [ ] | [ ] | [ ] | [ ] |
| Seasonal naive (t−24) | [ ] | [ ] | [ ] | [ ] |
| Linear regression | [ ] | [ ] | [ ] | [ ] |
| XGBoost | [ ] | [ ] | [ ] | [ ] |
| LightGBM | [ ] | [ ] | [ ] | [ ] |
| CatBoost | [ ] | [ ] | [ ] | [ ] |
| **Stacking** | **2.91** | [ ] | [ ] | [ ] |

Stacking was best, at **MSE 8.4568 → RMSE ≈ 2.91 [units]** on validation.

**What that's worth.** Against the mean predictor it explains [ ]% of variance.
Against seasonal naive — the baseline that matters, since HVAC load is strongly
diurnal — it improves RMSE by [ ]%. ASHRAE Guideline 14 accepts hourly models
at CV(RMSE) ≤ 30%; this one is at [ ]%.

[If it does not beat seasonal naive by a clear margin, say so. A model that
ties a one-line heuristic is a finding, not a failure — but only if you report it.]

---

## What this does not support

- **Any claim about UNILAG's actual consumption.** Proxy data. See above.
- **Causal attribution.** The model predicts; it does not establish that
  changing a setpoint produces a proportional saving.
- **Extrapolation beyond observed conditions.** Ensembles do not extrapolate.
  Outside the training range of temperature and load, output is unreliable.
- **Naira savings.** Converting predicted kWh to money needs a tariff model
  this study does not include.

## Improving it

Occupancy counts, building envelope characteristics, and HVAC runtime
schedules are the three features most likely to move accuracy, and all three
are absent here. Metering a single auditorium for one term would replace the
proxy dataset entirely and make the numbers above locally meaningful.

---

## The application

A web interface where a user uploads a dataset, receives predicted
consumption, and downloads the results. [Stack]. Built so a facilities team
can use the model without touching a notebook.
