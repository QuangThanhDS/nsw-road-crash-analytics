# NSW Road Crash Analytics

INFT6201 project: a data science case study using public NSW road crash data from Transport for NSW.

## Research questions
1. Which road, environment and vehicle factors make a crash more likely to be fatal or serious?
2. Why are regional crashes deadlier than metro ones, and does that hold once speed limit and crash type are controlled for?
3. Where are the crash hotspots, and which LGAs have the highest crash rates per head of population?
4. How did crash patterns change through COVID (2020–21) and after?

## Data
- **NSW Road Crash Data 2020–2024** (CRASH and TRAFFIC UNIT tables) and its Data Manual, from the
  [TfNSW Open Data Hub](https://opendata.transport.nsw.gov.au/dataset/nsw-crash-data). Licence: CC BY.
- **ABS population by LGA** (for per-capita rates). To be added.

Raw files go in `data/raw/`. They are not tracked by git, so download them from the source above.

## Structure
```
data/raw/          source files (not tracked)
data/processed/    cleaned outputs (not tracked)
notebooks/         exploration and modelling notebooks
src/               reusable cleaning, feature and modelling code
reports/figures/   charts for the report
docs/              notes, data dictionary, stakeholder analysis
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
