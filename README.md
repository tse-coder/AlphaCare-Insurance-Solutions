# AlphaCare Insurance Solutions — Task 1 & 2

This repository contains code and examples for Task 1 (Insurance Terminologies & EDA) and Task 2 (A/B Hypothesis Testing) from the AlphaCare Insurance Solutions (ACIS) analytics project.

Contents

- `requirements.txt` — Python dependencies
- `data/generate_synthetic_data.py` — script to generate a synthetic dataset for development and testing
- `src/eda.py` — helper functions and simple CLI for running EDA summaries
- `src/hypothesis_tests.py` — hypothesis testing utilities and a CLI to run the tests described in the brief
- `.gitignore`

Quick start

1. Create a Python environment (recommended using `python -m venv .venv` and `source .venv/bin/activate`).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Generate a synthetic dataset for tests (optional):

```bash
python data/generate_synthetic_data.py --n 50000 --out data/sample_claims.csv
```

4. Run EDA summary:

```bash
python src/eda.py data/sample_claims.csv --out results/eda_summary.csv
```

5. Run hypothesis tests (province, zipcode, margin, gender):

```bash
python src/hypothesis_tests.py data/sample_claims.csv --out results/hypothesis_results.json
```

