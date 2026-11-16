# French Exit — Offline Address Geocoding

**Offline, evidence-aware geocoding of noisy field addresses using address text, historical visits, GPS trajectories, and spatial reference data.**

This project (French Exit, Problem Statement 3) estimates a corrected location for each address, returns an uncertainty radius, and provides conservative directions/evidence labels for field teams. It is designed to operate **without live mapping API queries** when the supplied source data are available.

> **Data notice:** The original challenge dataset is **not** included in this repository. The notebook requires the original dataset to be supplied separately. Review data rights and confidentiality before publishing any input or generated records.

## Reported results

Five-fold **out-of-fold evaluation on 100 surveyed addresses** (reported in the accompanying [technical report](docs/technical_report.pdf)):

| Metric | Commercial geocoder | Architecture 5.2 |
|---|---:|---:|
| Median location error | 376.40 m | **76.63 m** |
| 90th-percentile error | 839.18 m | **488.64 m** |
| Mean location error | 532.67 m | **243.88 m** |

- **79.6% lower median error** than the commercial baseline on the same 100-address benchmark.
- **92% observed coverage** of the issued Confidence 2.0 radii against a 90% design target (100 surveyed addresses).
- Predictions produced for **3,117 addresses** in the reported full batch; source data included 5,578 field visits and 160,406 GPS trail points.

**Evaluation caveat:** These metrics come from the archived challenge evaluation described in the report; they were **not independently reproduced when preparing this repository** because the raw data were not part of the supplied files. Project (x, y) coordinates are local metric coordinates—not GPS latitude/longitude. The 100-case OOF benchmark should not be confused with a strict geographic or temporal holdout.

## How it works

```text
Raw address text, GPS visit logs, baselines, POI/locality data
                            |
                            v
    1. Normalize addresses, parse location cues, audit GPS integrity
                            |
                            v
    2. Build address-level evidence and visit reliability features
                            |
                            v
    3. Architecture 5.2: estimate corrected (x, y) pin
       + Clean visits  -> reliability-weighted geometric median
       + No visits     -> structured/ML experts and OOD-aware routing
                            |
                            v
    4. Confidence 2.0: evidence-conditioned quantile and conformal radius
                            |
                            v
    5. Safe direction/evidence tier + Visit Planner / PS2 feature exports
```

For mathematical details, experimental design, calibration, and robustness checks, see the [report](docs/technical_report.pdf) and [presentation](docs/presentation.pdf).

## Repository layout

```text
french-exit-offline-geocoding/
├── French_Exit_Offline_Geocoding.ipynb   # End-to-end executable notebook
├── cn_stage1_preprocess.py               # Address parsing, GPS integrity
├── cn_build_master.py                    # Feature table assembly
├── cn_architecture52.py                  # Hybrid location estimation
├── cn_confidence20.py                    # Calibrated radius estimation
├── cn_directions_and_integrations.py     # Directions and integration outputs
├── requirements.txt
├── .gitignore
├── docs/
│   ├── technical_report.pdf
│   └── presentation.pdf
└── README.md
```

The standalone Python modules are the same modules that the notebook writes with `%%writefile`. The preprocessing module's introductory line was corrected to a Python comment in this packaged edition; this is a syntax-only fix, not a model change.

## Reproducing the pipeline

1. Use **Python 3.10+** and a Jupyter notebook environment.
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   # macOS / Linux: source .venv/bin/activate
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Obtain the **original, authorized** challenge data. Extract `dataset.rar` into a folder called `dataset/` beside the notebook. At a minimum, this folder must contain all 12 CSV files listed below.
4. Open `French_Exit_Offline_Geocoding.ipynb` in Jupyter and select **Restart Kernel → Run All**. The notebook runs the preprocessing, model fitting, confidence calibration, and export steps in order. Running can be computationally intensive.

Expected raw inputs:

```text
dataset/
├── addresses.csv
├── accounts.csv
├── field_visits.csv
├── visit_gps_points.csv
├── landmarks_poi.csv
├── baseline_geocodes.csv
├── localities.csv
├── surveyed_addresses.csv
├── splits.csv
├── towns.csv
├── agents.csv
└── lenders.csv
```

**Outputs** are written locally to `submission_outputs/` (not tracked in Git):

- `final_predictions.csv`: estimated pin, confidence radius, and evidence-aware directions for each address.
- `named_poi_review_queue.csv`: uncertain named-POI cases requiring review.
- `visit_planner_features.csv`: location and verification-priority features for visit planning.
- `ps2_location_features.csv`: location-quality features for downstream PS2 use; these are **not** PS2 predictions.
- `summary_metrics.json` and manager charts for quality monitoring.

The example visit ordering uses **straight-line distance**, not real street routing or travel-time optimization. The retrospective visit-feedback demonstration is **not** proof of online model updates. Model confidence is a radius with empirical coverage, not a guarantee for an individual address.

## Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Jupyter; `rapidfuzz` is an optional speedup for fuzzy matching (the notebook falls back to `difflib`).

## Documentation

- [Technical report (PDF)](docs/technical_report.pdf)
- [Competition presentation (PDF)](docs/presentation.pdf)

## Privacy and publication

Raw address/customer/account information, field trails, device/agent information, survey coordinates, and generated address-level predictions may be confidential. `.gitignore` excludes the expected raw datasets and generated outputs; **always review staged files manually before a public push**. Only publish material you are authorized to share.

## Reproducibility status

The notebook and extracted modules have been **syntax-checked** for packaging. They have **not** been executed against the original dataset in this repository package, so runtime reproduction and third-party dependency compatibility remain to be verified in your environment.