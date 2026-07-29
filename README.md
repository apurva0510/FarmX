# FarmX

FarmX is a Streamlit app for crop planning. It combines yield estimation with soil nutrient predictions so growers can compare crops, estimate projected yield, and inspect model-backed nitrogen, phosphorus, and potassium recommendations.

## Features

- Projected yield calculator for 22 supported crops
- Side-by-side crop yield comparison
- Model-backed N, P, and K category predictions
- Predicted nitrogen value in kg/ha
- Curated soil testing and agricultural resource links

## Repository Layout

```text
.
├── app.py                 # Streamlit entry point
├── cost.py                # Yield estimation logic
├── functions.py           # Model training and prediction helpers
├── farmx/                 # Shared app metadata
├── data/                  # Training dataset
├── models/                # Committed model/scaler artifacts used by Streamlit
├── assets/                # Logo and favicon assets
└── train-models.py        # Regenerates model pickle files from the dataset
```

The `models/*.pkl` files are intentionally committed because the Streamlit app loads them at runtime.

## Getting Started

Create the locked environment and install its dependencies:

```bash
uv sync
```

Run the app:

```bash
uv run streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Regenerating Models

If the dataset or feature engineering changes, regenerate the model artifacts:

```bash
uv run python train-models.py
```

This updates the pickle files in `models/`.

## Dependency Notes

Dependencies are declared in `pyproject.toml` and reproducibly pinned in `uv.lock`. `requests` and `streamlit` are bounded above the vulnerable ranges reported by Dependabot.
