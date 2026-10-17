# Preliminary results: NSW road crash data 2020–2024

Source: Transport for NSW, *NSW Road Crash Data 2020–2024* (CRASH and TRAFFIC UNIT tables), CC BY.
All numbers come from the scripts in `src/` and can be reproduced with them (see README).

## 1. Data characterisation

| | |
|---|---|
| Crashes (CRASH table) | 92,189 rows, 50 columns; 92,082 after keeping crash years 2020–2024 |
| Traffic units (TRAFFIC UNIT table) | 170,962 rows, 10 columns (vehicles, pedestrians, cyclists), mean 1.85 per crash |
| Join | `Crash ID`, one-to-many; every traffic unit matches a crash and every crash has ≥1 unit |
| Grain | One row per police-reported crash: injury or tow-away |
| Time detail | Year, month, weekday and **two-hour interval** only (no exact date/time) |
| Location detail | Latitude/longitude, LGA (130), suburb, street, urbanisation |
| Outcome | Degree of crash (5 levels) plus counts killed / seriously / moderately / minor injured |

**Severity distribution** (`reports/figures/01_severity_distribution.png`): non-casualty tow-away 32%,
moderate injury 28%, serious injury 21%, minor injury 17%, fatal 1.5% (1,388 crashes, 1,507 deaths).
The modelling target **KSI (killed or seriously injured)** covers 22.8% of crashes, so it is
moderately imbalanced. Fatal-only (1.5%) is too rare for a reliable first model.

**Trend** (`02_yearly_trend.png`): fatal crashes went from about 260 a year in 2020–2022 to about 300 in
2023–2024. Serious-injury crashes dropped in 2021 (COVID lockdowns) and then rose by 37% to 2024.
Moderate/minor and tow-away crashes held flat.

**When** (`03_time_heatmaps.png`): crash volume peaks on weekday afternoons (Friday 16:00–18:00 is the
busiest slot), but severity peaks on **weekend mornings**: 29–31% of Sunday 08:00–12:00 crashes are KSI,
compared with about 20% in the weekday peak. Many crashes happen in congested traffic, but they are
less likely to be severe.

**Where** (`06_crash_map.png`, `eda_tables.xlsx` → `region`, `lga`):

| Region | Share of crashes | Share of deaths | KSI rate | Fatal rate |
|---|---|---|---|---|
| Greater Sydney / Newcastle / Wollongong | 68% | 38% | 21.5% | 0.9% |
| Regional urban | 19% | 22% | 23.3% | 1.7% |
| **Regional rural** | **13%** | **39%** | **29.1%** | **4.5%** |

Regional rural NSW has 13% of crashes but 39% of road deaths.

**Speed limit × region** (`04_severity_by_speed_region.png`): even at the **same speed limit**, a
crash outside the three big cities is much more likely to be fatal. At 100 km/h the fatal share is 5.0%
in the rest of NSW against 1.4% in the metro areas, and at 80 km/h it is 3.4% against 1.6%. Speed limit
alone does not explain the regional gap. Likely contributors include road design, emergency response
distance, crash type and vehicle mix, which gives question 2 a clear direction for further work.

**Crash type** (`05_ksi_by_impact_type.png`): pedestrian (47% KSI), person–object (46%), head-on (39%)
and rollover (32%) crashes are the most severe. Rear-end crashes are the least severe (13%).

## 2. Data quality issues

| Issue | Evidence | Handling |
|---|---|---|
| **Outcome-dependent recording (leakage)** | `Street lighting` is "Unknown / not stated" for 36% of crashes overall, but for **63% of tow-away and 53% of minor-injury crashes against 0.6% of fatal ones**. Smaller "Unknown" levels (natural lighting, key unit type) show the same pattern. Police record more detail when someone is badly hurt. | Including `Street lighting` lifts ROC-AUC from 0.75 to 0.81, but that comes from *how* the crash was recorded, not what caused it. The main model excludes it. The remaining "Unknown" levels are small (<2.5%) and are flagged for further work. |
| Structural missingness | `Primary temporary/hazardous/permanent feature` are 73–99% blank. A blank most likely means "no feature" rather than unknown. | Not used yet. Recode blank to "None" before using. |
| Under-reporting | Only injury and tow-away crashes are included. Minor non-tow crashes are not, and police attendance rules affect which minor crashes get recorded. | Rates describe *reported* crashes. Model severity *given* a reported crash, not crash risk. |
| Late reporting | 107 crashes from 2019 were reported in 2020, and about 70 a year are reported in the following year. | Analysis uses `Year of crash` and drops 2019. The most recent year may be slightly under-counted. |
| Coarse time | Only two-hour intervals, month and weekday; no exact dates | Time is treated as categorical. Holiday periods and daylight saving can't be identified exactly. |
| Definitional overlap | TfNSW's urban/non-urban split is itself based on speed zone, so "regional urban" crashes are almost all ≤80 km/h and "regional rural" ones ≥90 km/h | Compare metro against the rest of NSW within each speed limit, not urban against rural. |
| Privacy generalisation | No driver age, sex, licence, speeding or alcohol fields are included in the open release | Human factors can't be modelled from this data and are listed as a limitation. |
| Coordinates | All within NSW. Five crashes on Lord Howe Island (159°E) look like outliers but are valid. | Kept, and excluded from the mainland map only. |
| Duplicates / referential integrity | 0 duplicate crash IDs, 0 orphan traffic units | None needed |

The full column-level profile is in `reports/data_quality_profile.csv`.

## 3. Data modelling: initial results

**Task:** classify whether a reported crash is KSI from the circumstances recorded at the scene.

**Features:** 16 categorical crash attributes (speed limit, urbanisation, impact type, key vehicle type,
lighting, weather, surface, alignment, location type, road class, time, weekday, school zone, controls)
and 7 crash-level flags built from the TRAFFIC UNIT table (pedestrian, motorcycle, cyclist or heavy
vehicle involved, tree/pole hit, parked vehicle hit, number of units). Injury counts are excluded
because they define the target.

**Validation:** a **temporal split**, training on 2020–2023 (73,143 crashes) and testing on 2024 (18,939).
This mimics the real use case of learning from past years to flag high-risk crash patterns next year,
and it is stricter than a random split. Class weights are balanced.

| Model | ROC-AUC | PR-AUC | Recall (KSI) | Precision (KSI) | Fatal crashes flagged |
|---|---|---|---|---|---|
| Baseline (always "not KSI") | 0.500 | 0.251 | 0.00 | — | 0% |
| Logistic regression | 0.742 | 0.483 | 0.65 | 0.44 | 85% |
| Gradient boosting (HistGB) | **0.749** | **0.498** | 0.67 | 0.43 | 85% |
| *Gradient boosting incl. street lighting (leaky)* | *0.809* | *0.553* | *0.81* | *0.45* | *95%* |

**Risk ranking:** the gradient boosting model's top-risk decile has an actual KSI rate of 61%, against
5.5% in the bottom decile, an 11× spread (`model_summary.json`).

**Drivers** (`08_logreg_odds_ratios.png`, `09_gb_permutation_importance.png`): both models agree.
- Motorcycle involvement is the strongest factor (odds ratio about 5.7; the largest permutation importance).
- Vulnerable road users: pedestrian about 4.4×, cyclist about 4.3×, person-object about 4.4×.
- Hitting a tree or pole: about 2.5×.
- Head-on impact: about 2×.
- Heavy vehicle involved: about 1.8×.
- Speed limit of 100–110 km/h: about 1.5–1.7×.
- Rear-end, animal and 40 km/h crashes are protective.

These are associations, not causal effects. The odds ratios come from a regularised model in which every
category level is encoded, so read them as relative effects rather than comparisons against a single
reference level.

## 4. Suitability for the case study

- **Logistic regression** is the better fit for stakeholders who need to explain decisions (TfNSW,
  councils). It is interpretable, its odds ratios translate directly into policy targets such as
  motorcycle safety, vulnerable road users and roadside trees and poles, and it is only 0.007 AUC behind
  gradient boosting.
- **Gradient boosting** adds very little, which suggests the signal is mostly additive main effects in
  this feature set. Its value is in confirming feature rankings and capturing interactions such as
  speed × region.
- **Ceiling:** an AUC of about 0.75 is moderate. A large part of severity depends on things the open
  data doesn't record: speed travelled, restraint use, impairment, occupant age and vehicle safety rating.
  This is a limitation of the data, not of the model.
- **Calibration:** with balanced class weights the predicted probabilities are inflated (top decile
  predicts 0.79 against an actual 0.61). They are fine for ranking but should be recalibrated
  (Platt/isotonic) before being reported as probabilities.
- **Leakage check:** the street lighting finding is itself an important result. A naive model looks
  about 0.06 AUC better for the wrong reason.

## 5. Stakeholders

TfNSW Centre for Road Safety (programme targeting, safer-roads investment), local councils (local road
treatments and 40 km/h zones), NSW Police (enforcement timing such as weekend mornings and regional
highways), emergency services and NSW Health (regional response times), icare / CTP insurers (injury
burden), motorcycle and cycling advocacy groups, and the public.

## 6. Next steps

1. Recode the structural blanks and handle the remaining "Unknown" levels; recalibrate probabilities.
2. Add ABS LGA population to get per-capita crash and KSI rates by LGA.
3. Run a regional-gap model: metro vs rest of NSW, controlling for speed, impact type and vehicle mix.
4. Spatial hotspot clustering (DBSCAN or KDE) on KSI crashes.
5. Try a fatal-only model with resampling, and a cross-validated check across years.
