# BESTSELLER Data Scientist Interview Case

## Forecasting SKU-Level Demand

Forecasting is an essential part of managing the supply chain and keeping products in stock. In BESTSELLER, planners rely on tools that forecast weekly demand at the SKU level for a four-week horizon.

### What is a SKU?

A SKU is a combination of style, colour, and size.

Examples:
- Round Neck Knitted Pullover, Black, L
- Knitted Pencil Skirt, Natural Melange, XS

---

## Project structure

The repository is organized to separate concerns by responsibility:

```text
bestseller-case/
├── .github/
│   ├── copilot-instructions.md
│   └── skills/
│       └── grilling/
│           └── SKILL.md
├── data/
│   ├── raw/
│   │   └── fashion_boutique_dataset.json
│   └── processed/
│       └── weekly_sku_features.json
├── docs/
│   └── 2026-08-13 AIFP Data Scientist Case.pdf
├── notebooks/
├── results/
├── scripts/
│   └── run_pipeline.py
├── src/
│   ├── __init__.py
│   └── bestseller_forecast/
│       ├── __init__.py
│       ├── data.py
│       ├── features.py
│       ├── models.py
│       └── evaluation.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── image.png
└── .gitkeep
```

## Workflow

1. Load the raw dataset from `data/raw/`.
2. Build rolling and lag features in `src/bestseller_forecast/features.py`.
3. Train the forecasting model in `src/bestseller_forecast/models.py`.
4. Evaluate metrics in `src/bestseller_forecast/evaluation.py`.
5. Run the full workflow with:

```bash
python scripts/run_pipeline.py
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Original case brief

### 1. Model Design

We want to build an ML model to solve this forecasting problem.

What kind of model and predictor variables would you consider for this forecasting problem?

Given a specific style and colour, the model gives forecasts for each size, and we can calculate the proportion each size represents of a total. A priori, there is no mechanism to ensure these proportions are sensible.

### 2. Size Split Consistency

What can be done to ensure SKU forecasts do not combine into a wrong size split?

Stockouts in historical sales data are problematic for the model if not accounted for.

### 3. Handling Stockouts

What options or strategies would you pursue to account for stockouts, so that we use SKU demand rather than SKU sales as target for the forecasting model?

---

## Notes

The purpose of this exercise is to create a foundation for discussion and not a fully detailed production system.
