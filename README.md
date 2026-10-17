# NSW Road Crash Analytics

INFT6201 project: a data science case study using public NSW road crash data from Transport for NSW.

## Summary of work and results

**What has been done**
- **Data:** loaded and joined the TfNSW CRASH (92,189 crashes) and TRAFFIC UNIT (170,962 vehicles/people) tables for 2020–2024, and profiled all 50 crash fields for quality issues.
- **Exploration:** nine charts covering severity, yearly trends, day/time patterns, speed limit by region, crash type and maps of NSW and Greater Sydney (with ABS boundaries).
- **Modelling:** predicted whether a crash is **fatal or serious (KSI)** from 23 scene predictors with logistic regression and gradient boosting. Models were trained on 2020–2023 and tested once on 2024.
- **Report:** a LaTeX report with APA references and an appendix that explains both models and includes the code ([reports/report/main.pdf](reports/report/main.pdf)).

**Key results**
| | |
|---|---|
| Crashes analysed | 92,082 (2020–2024); 22.8% fatal or serious, 1.5% fatal, 1,507 deaths |
| Regional gap | Regional rural roads: 13% of crashes but **39% of deaths**; at 100 km/h, 5.0% of crashes outside Sydney/Newcastle/Wollongong are fatal against 1.4% inside |
| When | Most crashes happen on weekday afternoons, but the most severe are on **weekend mornings** (31% KSI, Sunday 10:00–12:00) |
| Crash type | Pedestrian (47% KSI) and head-on (39%) crashes are the most severe; rear-end crashes the least (13%) |
| Strongest risk factors | Motorcycle involved (odds ratio ≈ 5.7), pedestrian or cyclist (≈ 4.4), hit tree or pole (≈ 2.5), head-on (≈ 2.0), 100–110 km/h zone (≈ 1.5–1.7) |
| Model performance (2024 test) | Gradient boosting ROC-AUC **0.749**, logistic regression **0.742** (baseline 0.5); catches about 2 in 3 KSI crashes and 85% of fatal crashes; stable across years (0.73–0.75) |
| Data quality finding | *Street lighting* is left blank far more often for minor crashes, which leaks the outcome; including it inflates ROC-AUC to 0.81, so it is excluded |

**Recommended model:** logistic regression. It is almost as accurate as gradient boosting, and its odds ratios are easy for stakeholders to explain.

**Next steps:** add ABS population for per-capita LGA rates, model the regional gap directly, cluster hotspots of serious crashes, and recalibrate predicted probabilities.

## Research questions
1. Which road, environment and vehicle factors make a crash more likely to be fatal or serious?
2. Why are regional crashes deadlier than metro ones, and does that hold once speed limit and crash type are controlled for?
3. Where are the crash hotspots, and which LGAs have the highest crash rates per head of population?
4. How did crash patterns change through COVID (2020–21) and after?

## Data
- **NSW Road Crash Data 2020–2024** (CRASH and TRAFFIC UNIT tables) and its Data Manual, from the
  [TfNSW Open Data Hub](https://opendata.transport.nsw.gov.au/dataset/nsw-crash-data). Licence: CC BY.
- **ABS ASGS Edition 3 boundaries** (state, Greater Sydney, LGAs) for maps, in `data/external/boundaries/` (CC BY 4.0).
- **ABS population by LGA** (for per-capita rates). To be added.

Raw files go in `data/raw/`. They are not tracked by git, so download them from the source above.

## Structure
```
data/raw/          source files (not tracked)
data/processed/    cleaned outputs (not tracked)
data/external/     ABS boundary files for maps
notebooks/         exploration and modelling notebooks
src/               reusable cleaning, feature and modelling code
reports/figures/   charts for the report
reports/report/    LaTeX report (main.tex, references.bib, main.pdf)
docs/              notes, data dictionary, stakeholder analysis
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the results
```bash
cd src
python3 load_data.py      # xlsx -> data/processed/*.csv (about 30 s)
python3 profile_data.py   # data quality profile -> reports/data_quality_profile.csv
python3 eda.py            # charts -> reports/figures/01-06, tables -> reports/eda_tables.xlsx
python3 model.py          # main KSI models -> reports/figures/07-09, reports/model_results.csv
python3 model.py all      # comparison run that includes Street lighting (leakage check)
```

Findings so far: [docs/preliminary_results.md](docs/preliminary_results.md)
