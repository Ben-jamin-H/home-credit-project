# Home Credit Default Risk

**Author:** Benjamin Hogan

---

## Project Overview

Home Credit Group believes they are not strong enough at predicting the repayment ability of applicants who have insufficient or non-existent credit histories. This problem is inefficient and is missing realized potential in both application and current customer performance, which is costly to both parties.

The purpose of the proposed project is to prevent future inefficiencies and lost revenue by identifying customers with insufficient credit history who have a high probability of repaying their loan. Those customers will not be rejected in the initial application process once identified. The model will further analyze the applicants in order to give them the most mutually beneficial loan terms. The benefit to Home Credit Group of a successful project will be improved revenue through more qualified customers and more consistent payments which would increase revenue and efficiency.

- **Problem Type:** Binary classification (imbalanced — ~8% default rate)
- **Target Variable:** `TARGET` (1 = default, 0 = repaid on time)
- **Primary Metric:** AUC-ROC (robust to class imbalance)
- **Training Data:** 307,511 loan applications × 122 features
- **Test Data:** 48,744 loan applications × 121 features

---

## Modeling Notebook

### `modeling_notebook.qmd`

A full end-to-end modeling workflow covering baselines, candidate model comparison, class imbalance handling, hyperparameter tuning, supplementary feature engineering, and Kaggle submission. Rendered outputs are generated locally from this `.qmd` source file.

### Kaggle Result

| Submission | Public Leaderboard AUC-ROC |
|---|---|
| Final LightGBM + supplementary features | **0.74718** |

---

## Model Card

### `model_card.qmd`

A structured model card documenting the final LightGBM classifier across nine sections. Written as a professional document — code is hidden and outputs are displayed — intended to be readable by both technical and non-technical stakeholders. Rendered HTML is generated locally from the `.qmd` source file.

### What the Model Card Covers

| Section | Summary |
|---|---|
| **1. Model Details** | LightGBM gradient-boosted classifier, version 1.0 (March 2026); 160 features; tuned hyperparameters |
| **2. Intended Use** | First-pass credit screening for Home Credit underwriters; not designed for fraud detection, portfolio stress testing, or fully automated decisions |
| **3. Performance Metrics** | CV AUC-ROC: 0.759 · Kaggle AUC-ROC: 0.747 · Precision: 0.241 · Recall: 0.412 · Approval rate: 86.2% (at threshold 0.63) |
| **4. Decision Threshold Analysis** | Cost assumptions sourced from McKinsey (2020) and Moody's (2019): ~$934 profit per repaid loan, ~$10,500 loss per default (11.2× ratio). Optimal threshold of 0.63 maximizes expected net value at ~$50.5M — a $47.3M improvement over approving all applicants |
| **5. Explainability** | SHAP analysis on a 1,000-row sample. Top predictors: `EXT_SOURCE_MEAN`, `EXT_SOURCE_2x3` (interaction), `LOAN_TO_VALUE_RATIO`, `INST_LATE_RATIO`, `NAME_EDUCATION_TYPE` |
| **6. Adverse Action Mapping** | Top SHAP features translated into ECOA-compliant plain-language denial reasons (e.g., "limited external credit history", "pattern of late installment payments") |
| **7. Fairness Analysis** | Female applicants approved at 88.4% vs 81.9% for male — gap is consistent with a 3.1pp difference in actual default rates. Education approval gap (91.9% for higher education vs 81.7% for lower secondary) exceeds the underlying default rate gap, flagged for disparate impact monitoring |
| **8. Limitations & Risks** | Missing EXT_SOURCE scores for new borrowers; static training snapshot; uncalibrated probabilities; feedback loop risk; regulatory risk from gender and education as direct model inputs |
| **9. Executive Summary** | Business recommendation: deploy as a tiered screening tool (auto-approve below 0.50, human review 0.50–0.75, auto-deny above 0.75). Key caveats: model misses 59% of defaults; financial estimates use industry benchmarks not internal figures; gender use requires legal review |


---

## Modeling Workflow & Results

### Stage 1 — Baselines

Two baselines establish performance floors before any real modeling:

| Model | AUC-ROC |
|---|---|
| Majority class classifier (always predicts 0) | 0.5000 |
| Logistic Regression — EXT_SOURCE features only | 0.7177 |

The majority class classifier achieves ~92% accuracy but 0.50 AUC — equivalent to random guessing — confirming that **accuracy is a misleading metric** for this imbalanced problem. The three external credit scores (`EXT_SOURCE_1/2/3`) alone already provide meaningful lift, consistent with EDA findings.

---

### Stage 2 — Candidate Model Comparison

Four candidates were evaluated using 3-fold stratified CV on the full feature set:

| Model | Mean AUC-ROC | Std |
|---|---|---|
| **LightGBM (default params)** | **0.7664** | 0.0016 |
| Logistic Regression — full features | 0.7454 | 0.0022 |
| Logistic Regression — engineered features only | 0.7304 | 0.0029 |
| Random Forest | 0.7276 | 0.0013 |

**LightGBM was the clear winner**, outperforming all logistic regression variants and random forest by a meaningful margin. Its ability to handle missing values natively, model non-linear interactions, and scale efficiently on tabular data made it the candidate taken forward. Notably, logistic regression on the full feature set outperformed logistic regression on engineered features only, suggesting the raw features carry signal that the derived ratios alone do not fully capture.

---

### Stage 3 — Class Imbalance Handling

Five strategies were tested on a 10,000-row stratified subsample using LightGBM as the base model:

| Strategy | Mean AUC-ROC |
|---|---|
| **Random undersampling** | **0.7097** |
| SMOTE | 0.7061 |
| No adjustment | 0.6956 |
| Random oversampling | 0.6915 |
| Class weights (`scale_pos_weight`) | 0.6903 |

On the subsample, **random undersampling** produced the best AUC. However, because these results were measured on a small subsample (where variance is high), `scale_pos_weight` — which natively integrates into LightGBM without data manipulation — was carried forward into hyperparameter tuning on the full dataset, where it performed competitively.

---

### Stage 4 — Hyperparameter Tuning

Randomized search (20 iterations, 3-fold CV) was run on a 5,000-row subsample to efficiently explore the parameter space. The best parameters were then used to train on the full dataset:

| Feature Set | Mean AUC-ROC | Std |
|---|---|---|
| Tuned LightGBM — application features only | 0.7477 | 0.0026 |
| Tuned LightGBM + supplementary features | 0.7592 | 0.0024 |

---

### Stage 5 — Supplementary Features

Five supplementary tables were aggregated to the applicant level and joined to the main feature matrix, adding 54 new features (107 → 161 columns):

| Table | Features Added | Key Signals |
|---|---|---|
| `bureau.csv` | 15 | Overdue counts, debt/credit ratios, active loan recency |
| `previous_application.csv` | 12 | Approval/refusal rates, prior credit amounts |
| `installments_payments.csv` | 12 | Late payment ratio, underpayment behavior |
| `POS_CASH_balance.csv` | 7 | DPD counts, late payment ratio |
| `credit_card_balance.csv` | 10 | Utilization rate, drawing behavior, CC DPD |

Adding supplementary features improved CV AUC from **0.7477 → 0.7592** (+0.0115), confirming that credit history signals from external tables add meaningful predictive power beyond the application form alone.

---

### Final Model

The final model is a **LightGBM classifier** trained on the full training dataset (307,511 rows × 160 features) with:

- Tuned hyperparameters from randomized search
- `scale_pos_weight` to handle class imbalance (~11:1 ratio)
- All five supplementary tables joined as additional features

**Kaggle Public Leaderboard AUC-ROC: 0.74718**

---

## Data Preparation Script

### `data_preparation.py`

A comprehensive, reusable module for cleaning, transforming, and engineering features from the Home Credit dataset. Designed to ensure **train/test consistency** by computing all parameters (medians, thresholds) from the training data only.

### Installation

Requires Python 3.10+ and Polars:

```bash
pip install polars scikit-learn lightgbm imbalanced-learn
```

### Quick Start

```python
import polars as pl
from data_preparation import (
    fit_params_from_train,
    prepare_application_data,
    aggregate_bureau,
    aggregate_previous_application,
    aggregate_installments,
    join_supplementary_features,
)

# Load raw data
train = pl.read_csv("home-credit-default-risk/application_train.csv")
test  = pl.read_csv("home-credit-default-risk/application_test.csv")

# Step 1: Fit parameters from training data ONLY
params = fit_params_from_train(train)

# Step 2: Apply pipeline to both datasets
train_prepared = prepare_application_data(train, params, is_train=True)
test_prepared  = prepare_application_data(test, params, is_train=False)

# Step 3: Aggregate supplementary tables
bureau_agg = aggregate_bureau(pl.read_csv("home-credit-default-risk/bureau.csv"))
prev_agg   = aggregate_previous_application(pl.read_csv("home-credit-default-risk/previous_application.csv"))
inst_agg   = aggregate_installments(pl.read_csv("home-credit-default-risk/installments_payments.csv"))

# Step 4: Join supplementary features
train_final = join_supplementary_features(train_prepared, bureau_agg, prev_agg, inst_agg)
test_final  = join_supplementary_features(test_prepared,  bureau_agg, prev_agg, inst_agg)
```

---

## Pipeline Stages

### Phase 1 — Cleaning (`clean_application_data`)

| Transformation | Description |
|---|---|
| `DAYS_EMPLOYED` sentinel | Replace 365243 (placeholder) with null; add `DAYS_EMPLOYED_ANOM` flag |
| `AGE_YEARS` | Convert negative `DAYS_BIRTH` to positive years |
| `CODE_GENDER "XNA"` | Replace with null (4 rows) |
| `OWN_CAR_AGE` | Fill with 0 for non-car-owners |
| EXT_SOURCE imputation | Median imputation using **training medians only** |
| `OCCUPATION_TYPE` | Fill nulls with "Unknown" |
| Housing columns | Drop 47 columns with >50% missing |
| Credit bureau inquiries | Fill nulls with 0 |

**Result:** 122 → 77 columns

### Phase 2 — Feature Engineering (`engineer_features`)

| Category | Features |
|---|---|
| **Demographics** | `EMPLOYED_YEARS`, `REGISTRATION_YEARS`, `ID_PUBLISH_YEARS`, `PHONE_CHANGE_YEARS`, `EMPLOYED_TO_AGE_RATIO` |
| 
