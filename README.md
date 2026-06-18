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

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Regenerating Models

If the dataset or feature engineering changes, regenerate the model artifacts:

```bash
python train-models.py
```

This updates the pickle files in `models/`.

## Dependency Notes

Dependencies are bounded in `requirements.txt` to keep Dependabot updates manageable while allowing security patches. `requests` and `streamlit` are pinned above the vulnerable ranges reported by Dependabot.
