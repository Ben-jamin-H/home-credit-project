# Home Credit Default Risk

**Author:** Benjamin Hogan  

---

## Project Overview

Home Credit Group believes they are not strong enough at predicting the repayment ability of applicants who have insufficient or non-existent credit histories. This problem is inefficient and is missing realized potential in both application and current customer performance, which is costly to both parties.

The purpose of the proposed project is to prevent future inefficiencies and lost revenue by identifying customers with insufficient credit history who have a high probability of repaying their loan. Those customers will not be rejected in the initial application process once identified. The model will further analyze the applicants in order to give them the most mutually beneficial loan terms. The benefit to Home Credit Group of a successful project will be improved revenue through more qualified customers and more consistent payments which would increase revenue and efficiency.

- **Problem Type:** Binary classification (imbalanced — ~8% default rate)
- **Target Variable:** `TARGET` (1 = default, 0 = repaid on time)
- **Training Data:** 307,511 loan applications × 122 features
- **Test Data:** 48,744 loan applications × 121 features

---

## Data Preparation Script

### `data_preparation.py`

A comprehensive, reusable module for cleaning, transforming, and engineering features from the Home Credit dataset. Designed to ensure **train/test consistency** by computing all parameters (medians, thresholds) from the training data only.

### Installation

Requires Python 3.10+ and Polars:

```bash
pip install polars
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
bureau_agg = aggregate_bureau(
    pl.read_csv("home-credit-default-risk/bureau.csv")
)
prev_agg = aggregate_previous_application(
    pl.read_csv("home-credit-default-risk/previous_application.csv")
)
inst_agg = aggregate_installments(
    pl.read_csv("home-credit-default-risk/installments_payments.csv")
)

# Step 4: Join supplementary features
train_final = join_supplementary_features(train_prepared, bureau_agg, prev_agg, inst_agg)
test_final  = join_supplementary_features(test_prepared, bureau_agg, prev_agg, inst_agg)

# Result: train_final (307,511 × 146), test_final (48,744 × 145)
# Both have identical 145 feature columns; train has additional TARGET column
```

---

## Pipeline Stages

### Phase 1 — Cleaning (`clean_application_data`)

Handles data quality issues identified during EDA:

| Transformation | Description |
|----------------|-------------|
| DAYS_EMPLOYED sentinel | Replace 365243 (placeholder) with null; add `DAYS_EMPLOYED_ANOM` flag |
| AGE_YEARS | Convert negative DAYS_BIRTH to positive years |
| CODE_GENDER "XNA" | Replace with null (4 rows) |
| OWN_CAR_AGE | Fill with 0 for non-car-owners |
| EXT_SOURCE imputation | Median imputation using **training medians only** |
| OCCUPATION_TYPE | Fill nulls with "Unknown" |
| Housing columns | Drop 47 columns with >50% missing |
| Credit bureau inquiries | Fill nulls with 0 |

**Result:** 122 → 77 columns

### Phase 2 — Feature Engineering (`engineer_features`)

Creates 30 derived features:

| Category | Features |
|----------|----------|
| **Demographics** | `EMPLOYED_YEARS`, `REGISTRATION_YEARS`, `ID_PUBLISH_YEARS`, `PHONE_CHANGE_YEARS`, `EMPLOYED_TO_AGE_RATIO` |
| **Financial Ratios** | `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `LOAN_TO_VALUE_RATIO`, `CREDIT_TERM_MONTHS`, `GOODS_TO_INCOME_RATIO` |
| **Missing Indicators** | 10 `MISS_*` boolean flags |
| **EXT_SOURCE Interactions** | `EXT_SOURCE_MEAN`, 3 pairwise products, 3 squared terms |
| **Binned Variables** | `AGE_BIN`, `CREDIT_INCOME_BIN`, `EXT_SOURCE_MEAN_BIN` |

**Result:** 77 → 107 columns

### Phase 3 — Supplementary Aggregations

Aggregates data from supplementary tables to the applicant level (`SK_ID_CURR`):

| Function | Source File | New Columns | Description |
|----------|-------------|-------------|-------------|
| `aggregate_bureau()` | bureau.csv | 15 | Credit counts, active/closed ratio, overdue amounts, debt ratios |
| `aggregate_previous_application()` | previous_application.csv | 12 | Application counts, approval/refusal rates, amounts |
| `aggregate_installments()` | installments_payments.csv | 12 | Late payment ratio, underpayment ratio, payment behavior |

**Result:** 107 → 146 columns (145 features + TARGET)

---

## Train/Test Consistency

The `FittedParams` dataclass ensures no data leakage:

```python
@dataclass
class FittedParams:
    ext_source_medians: dict[str, float]  # Medians from training data
    columns_to_keep: list[str]            # Column alignment
```

- **`fit_params_from_train()`** — Computes parameters from training data only
- **`prepare_application_data()`** — Applies identical transformations to both datasets
- **`align_train_test()`** — Ensures identical column order (excluding TARGET)

---

## Key Functions

| Function | Purpose |
|----------|---------|
| `fit_params_from_train(train_df)` | Compute medians and thresholds from training data |
| `prepare_application_data(df, params, is_train)` | Full cleaning + feature engineering pipeline |
| `clean_application_data(df, ext_source_medians)` | Phase 1 cleaning only |
| `engineer_features(df)` | Phase 2 feature engineering only |
| `aggregate_bureau(bureau_df)` | Aggregate bureau.csv to applicant level |
| `aggregate_previous_application(prev_df)` | Aggregate previous_application.csv |
| `aggregate_installments(inst_df)` | Aggregate installments_payments.csv |
| `join_supplementary_features(app_df, ...)` | Join all aggregations to application data |
| `align_train_test(train_df, test_df)` | Ensure identical columns (except TARGET) |

---

## Data Files

All data files are in `home-credit-default-risk/` and are **not committed to the repository** (see `.gitignore`).

| File | Description |
|------|-------------|
| application_train.csv | Training data with TARGET |
| application_test.csv | Test data (no TARGET) |
| bureau.csv | Credit Bureau records |
| previous_application.csv | Previous Home Credit applications |
| installments_payments.csv | Payment history |
| bureau_balance.csv | Monthly bureau balance history |
| POS_CASH_balance.csv | POS/Cash loan balances |
| credit_card_balance.csv | Credit card balances |
| HomeCredit_columns_description.csv | Data dictionary |

