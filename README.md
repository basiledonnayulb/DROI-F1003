# EU Migration Data Lab

**A working Python/Streamlit project trained on the attached Eurostat workbooks.**

The app accepts any combination of age group, sex, reporting country and citizenship
(plus a reference year), and displays an **aggregate annual irregular-presence detection
rate expressed as a percentage**, with the recorded rate, population benchmark,
data coverage, historical trend, and a chart.

**This is not the probability that an individual is illegally present.** There is no
individual labeled `legal/illegal` training dataset in the supplied workbooks, and
the denominator is **population on 1 January**, not verified lawful residence.
A percentage output does not make it a statistically valid legal-status classifier.
Do not use these results to assess a person or for immigration enforcement.

## Quick start (Python 3.10+)

```bash
cd DROI-F1003
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. **Pretrained model and processed data are included**.

## Retrain from the included Eurostat workbooks

```bash
python train.py
streamlit run app.py
```

The source Excel files are in `data/irregular.xlsx` and `data/population.xlsx`.
The streaming parser (`src/prepare.py`) reads the exported wide Data sheets and
converts numeric records to SQLite and a compact `data/training.csv`.
The models (`src/model.py`) use histogram gradient-boosting Poisson regression,
with time-based validation on 2025 and refitting on 2018–2025 for the app.
`data/model.joblib` and `data/training_summary.json` are generated artifacts.

## Exactly what is estimated?

For a given group and reference year:

```text
observed detection benchmark (%) =
    number of detected irregular-presence cases DURING THE YEAR
    ------------------------------------------------------------ × 100
    registered resident population AT THE START OF THE YEAR
```

The model predicts the numerator-to-denominator **benchmark rate** in units of
annual detections per 100,000 population-benchmark members, then displays a percentage.
It does *not* produce probabilities of unlawful residence, or classify a person.
Annual detection counts are flow data and can include short-stay visitors and
repeat detections. The denominator is stock data, potentially with different
coverage. These mismatches make a risk or incidence interpretation invalid.

### How missing filters work

Selecting 'All' for sex/country aggregates matching countries and sexes by
population-weighted averaging; no zero or fabricated values are inserted.
'All third-country citizenships' uses Eurostat's **Total** detection series
and the population series **Non-EU27 countries (from 2020) nor reporting country**;
this is an approximate population scope, not proven to match the detection series.
Selecting a specific citizenship uses matching country-of-citizenship data.

### Age harmonisation (approximation)

The irregular-detection file has under-18, 18–34 and 35+ bins.
The population file has under-15, 15–64 and 65+ bins.

Rather than averaging raw counts, the training pipeline **assumes uniform
population at each age from 15–64**:

| Detected-case age band | Approximate population denominator |
|---|---|
| Under 18 | under-15 + (3/50) × ages 15–64 |
| 18–34 | (17/50) × ages 15–64 |
| 35+ | (30/50) × ages 15–64 + ages 65+ |
| All ages | under-15 + ages 15–64 + ages 65+ (no age approximation) |

This assumption is **not verifiable** from these uploads; the age-specific results
are exploratory and can be biased. Download more precise single-year or 5-year
population age bins to improve accuracy. Unmatched, missing, suppressed and
mathematically inconsistent cells are omitted, not assumed to be zeros.

## Data and modeling limitations

- Source 1: [Eurostat migr_eipre](https://ec.europa.eu/eurostat/databrowser/view/migr_eipre/default/table?lang=en), annual third-country nationals **found** illegally present.
- Source 2: [Eurostat migr_pop1ctz](https://ec.europa.eu/eurostat/databrowser/view/migr_pop1ctz/default/table?lang=en), population on 1 January by age, sex and citizenship.
- Observations are rounded and can be missing or suppressed.
- Annual detections may exceed estimated resident stocks in a group. These are
  excluded rather than silently capped as a probability.
- Country/year coverage is incomplete and can change, affecting trends.
- The model predicts a **rate index**; it does not estimate undetected irregular
  migration stocks or identify who has valid immigration documents.
- Data should never be used for individual profiling or enforcement.

## Project layout

```text
eu_migration_rate_lab/
├── app.py                      Streamlit dashboard
├── train.py                    Rebuild and train models
├── requirements.txt
├── .streamlit/config.toml
├── src/
│   ├── prepare.py              XLSX streaming importer and age harmonisation
│   └── model.py                ML training, evaluation, group aggregation
├── tests/test_model.py
└── data/
    ├── irregular.xlsx          Supplied Eurostat migr_eipre
    ├── population.xlsx         Supplied Eurostat migr_pop1ctz
    ├── training.csv            Cleaned merged cohort observations
    ├── source.sqlite           Parsed, indexable observations
    ├── model.joblib             Two pretrained aggregate models
    └── training_summary.json   Model and coverage metadata
```
