"""
data_preparation.py
====================
Reusable data-cleaning and transformation functions for the
Home Credit Default Risk application data.

This module is designed so that every transformation applied to the
training set can be identically applied to the test set, ensuring
consistency between the two splits.

The module is organised into three phases:

**Phase 1 — Cleaning** (``clean_application_data``)
    Fix anomalies, handle missing values, and drop unusable columns.

**Phase 2 — Feature Engineering** (``engineer_features``)
    Create derived demographic, financial, missing-data, interaction,
    and binned features from the cleaned data.

**Phase 3 — Supplementary Aggregations**
    Aggregate bureau, previous_application, and installments data to
    the applicant level and join to the main application data.

Train/Test Consistency
----------------------
All parameters learned from data (medians, thresholds, etc.) are computed
from the **training set only** and stored in a ``FittedParams`` dataclass.
These parameters are then reused when processing the test set to prevent
data leakage.

Usage
-----
    import polars as pl
    from data_preparation import (
        prepare_application_data,
        fit_params_from_train,
        FittedParams,
        aggregate_bureau,
        aggregate_previous_application,
        aggregate_installments,
        join_supplementary_features,
    )

    # Load raw data
    train = pl.read_csv("home-credit-default-risk/application_train.csv")
    test  = pl.read_csv("home-credit-default-risk/application_test.csv")

    # Fit parameters from training data only
    params = fit_params_from_train(train)

    # Apply identical pipeline to both datasets
    train_prepared = prepare_application_data(train, params, is_train=True)
    test_prepared  = prepare_application_data(test, params, is_train=False)

    # Load and aggregate supplementary tables
    bureau_agg = aggregate_bureau(pl.read_csv("...bureau.csv"))
    prev_agg   = aggregate_previous_application(pl.read_csv("...previous_application.csv"))
    inst_agg   = aggregate_installments(pl.read_csv("...installments_payments.csv"))

    # Join supplementary features
    train_final = join_supplementary_features(train_prepared, bureau_agg, prev_agg, inst_agg)
    test_final  = join_supplementary_features(test_prepared, bureau_agg, prev_agg, inst_agg)

"""

from dataclasses import dataclass, field
import polars as pl
import polars.selectors as cs


# ===================================================================
#  FITTED PARAMETERS — Computed from training data only
# ===================================================================

@dataclass
class FittedParams:
    """Container for all parameters fitted from training data.

    This dataclass stores medians, means, and thresholds computed
    from the training set.  Pass the same instance when processing
    the test set to ensure identical transformations.

    Attributes
    ----------
    ext_source_medians : dict[str, float]
        Median values for EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3.
    columns_to_keep : list[str]
        Final list of columns after cleaning (ensures train/test alignment).
    """

    ext_source_medians: dict[str, float] = field(default_factory=dict)
    columns_to_keep: list[str] = field(default_factory=list)


def fit_params_from_train(train_df: pl.DataFrame) -> FittedParams:
    """Compute all fitted parameters from the training data.

    This function should be called **once** on the training data.
    The returned ``FittedParams`` object is then passed to
    ``prepare_application_data`` for both train and test.

    Parameters
    ----------
    train_df : pl.DataFrame
        Raw application_train.csv DataFrame.

    Returns
    -------
    FittedParams
        Container with all fitted parameters.
    """
    params = FittedParams()

    # EXT_SOURCE medians (computed before any cleaning)
    params.ext_source_medians = {
        "EXT_SOURCE_1": train_df.get_column("EXT_SOURCE_1").median(),
        "EXT_SOURCE_2": train_df.get_column("EXT_SOURCE_2").median(),
        "EXT_SOURCE_3": train_df.get_column("EXT_SOURCE_3").median(),
    }

    return params


# ---------------------------------------------------------------------------
# 1. DAYS_EMPLOYED anomaly
# ---------------------------------------------------------------------------
def fix_days_employed_anomaly(df: pl.DataFrame) -> pl.DataFrame:
    """Replace the DAYS_EMPLOYED sentinel value (365243) with null.

    The EDA revealed that 18% of rows contain 365243 (~1,000 years) in
    DAYS_EMPLOYED.  This is a placeholder for applicants who are not
    currently employed (e.g. pensioners).  The sentinel group actually
    has a *lower* default rate than the rest of the data, so the
    information is worth preserving.

    Transformations
    ---------------
    * DAYS_EMPLOYED = 365243  →  null
    * New boolean column ``DAYS_EMPLOYED_ANOM`` = True where the
      sentinel was present, False otherwise.

    Parameters
    ----------
    df : pl.DataFrame
        Application data containing DAYS_EMPLOYED.

    Returns
    -------
    pl.DataFrame
        DataFrame with the anomaly replaced and the flag column added.
    """
    return df.with_columns(
        # Boolean flag: True where the sentinel value was found
        (pl.col("DAYS_EMPLOYED") == 365243)
        .alias("DAYS_EMPLOYED_ANOM"),
        # Replace the sentinel with null so downstream imputation or
        # models handle it correctly
        pl.when(pl.col("DAYS_EMPLOYED") == 365243)
        .then(None)
        .otherwise(pl.col("DAYS_EMPLOYED"))
        .alias("DAYS_EMPLOYED"),
    )


# ---------------------------------------------------------------------------
# 2. Convert DAYS_BIRTH to positive age in years
# ---------------------------------------------------------------------------
def add_age_years(df: pl.DataFrame) -> pl.DataFrame:
    """Create an AGE_YEARS column from DAYS_BIRTH.

    DAYS_BIRTH is stored as a negative number of days relative to the
    application date.  This helper converts it to a positive age in
    years for readability and easier downstream feature engineering.

    Parameters
    ----------
    df : pl.DataFrame
        Application data containing DAYS_BIRTH.

    Returns
    -------
    pl.DataFrame
        DataFrame with the new AGE_YEARS column appended.
    """
    return df.with_columns(
        (pl.col("DAYS_BIRTH").abs() / 365.25)
        .round(2)
        .alias("AGE_YEARS"),
    )


# ---------------------------------------------------------------------------
# 3. Handle CODE_GENDER = "XNA"
# ---------------------------------------------------------------------------
def fix_gender_xna(df: pl.DataFrame) -> pl.DataFrame:
    """Replace the ambiguous 'XNA' gender code with null.

    Only 4 rows in the training data carry this value.  Setting them to
    null is the safest approach—subsequent imputation or model handling
    can decide how to treat them.

    Parameters
    ----------
    df : pl.DataFrame
        Application data containing CODE_GENDER.

    Returns
    -------
    pl.DataFrame
        DataFrame with 'XNA' replaced by null in CODE_GENDER.
    """
    return df.with_columns(
        pl.when(pl.col("CODE_GENDER") == "XNA")
        .then(None)
        .otherwise(pl.col("CODE_GENDER"))
        .alias("CODE_GENDER"),
    )


# ---------------------------------------------------------------------------
# 4. Fill OWN_CAR_AGE for non-car-owners
# ---------------------------------------------------------------------------
def fix_own_car_age(df: pl.DataFrame) -> pl.DataFrame:
    """Set OWN_CAR_AGE to 0 for applicants who do not own a car.

    66% of OWN_CAR_AGE values are missing, and these correspond almost
    exactly to FLAG_OWN_CAR = 'N'.  For non-car-owners a value of 0 is
    semantically correct (they have no car, so the car age is zero).

    Parameters
    ----------
    df : pl.DataFrame
        Application data containing OWN_CAR_AGE and FLAG_OWN_CAR.

    Returns
    -------
    pl.DataFrame
        DataFrame with OWN_CAR_AGE filled to 0 where appropriate.
    """
    return df.with_columns(
        pl.when(pl.col("FLAG_OWN_CAR") == "N")
        .then(0)
        .otherwise(pl.col("OWN_CAR_AGE"))
        .alias("OWN_CAR_AGE"),
    )


# ---------------------------------------------------------------------------
# 5. Impute EXT_SOURCE variables with the median
# ---------------------------------------------------------------------------
def impute_ext_sources(
    df: pl.DataFrame,
    medians: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Fill missing EXT_SOURCE_1 / _2 / _3 values with their medians.

    These three external credit scores are the strongest individual
    predictors of default.  Missing-data rates differ dramatically:

    * EXT_SOURCE_1 — 56 % missing
    * EXT_SOURCE_2 —  0.2 % missing
    * EXT_SOURCE_3 — 19.8 % missing

    Median imputation is a safe baseline that does not distort the
    central tendency of each feature.

    Parameters
    ----------
    df : pl.DataFrame
        Application data with EXT_SOURCE columns.
    medians : dict[str, float] | None
        Pre-computed medians keyed by column name.  When ``None`` the
        medians are calculated from ``df`` itself (appropriate for the
        training set).  Pass the training-set medians when cleaning the
        test set to avoid data leakage.

    Returns
    -------
    pl.DataFrame
        DataFrame with nulls in EXT_SOURCE columns filled.
    """
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

    # Compute medians from the data if none were supplied
    if medians is None:
        medians = {
            col: df.get_column(col).median()
            for col in ext_cols
        }

    return df.with_columns(
        pl.col(col).fill_null(medians[col])
        for col in ext_cols
    )


# ---------------------------------------------------------------------------
# 6. Fill OCCUPATION_TYPE missing values
# ---------------------------------------------------------------------------
def fill_occupation_type(df: pl.DataFrame) -> pl.DataFrame:
    """Replace null OCCUPATION_TYPE values with 'Unknown'.

    31% of applicants have no recorded occupation.  Rather than dropping
    these rows or imputing with the mode, we assign a dedicated
    'Unknown' category so models can learn whether the *absence* of an
    occupation itself is predictive.

    Parameters
    ----------
    df : pl.DataFrame
        Application data containing OCCUPATION_TYPE.

    Returns
    -------
    pl.DataFrame
        DataFrame with null occupation types replaced by 'Unknown'.
    """
    return df.with_columns(
        pl.col("OCCUPATION_TYPE").fill_null("Unknown"),
    )


# ---------------------------------------------------------------------------
# 7. Drop high-missingness housing columns
# ---------------------------------------------------------------------------

# The 45 housing-characteristic columns (AVG / MODE / MEDI variants)
# each exceed 50 % missing values.  They are unlikely to add predictive
# power without a sophisticated imputation strategy and dramatically
# inflate the feature space.
HOUSING_COLUMNS = [
    "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG",
    "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG",
    "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG",
    "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG",
    "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE",
    "YEARS_BUILD_MODE", "COMMONAREA_MODE", "ELEVATORS_MODE",
    "ENTRANCES_MODE", "FLOORSMAX_MODE", "FLOORSMIN_MODE",
    "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE",
    "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE",
    "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI",
    "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI",
    "ENTRANCES_MEDI", "FLOORSMAX_MEDI", "FLOORSMIN_MEDI",
    "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI", "LIVINGAREA_MEDI",
    "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI",
    "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "TOTALAREA_MODE",
    "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
]


def drop_high_missing_housing(df: pl.DataFrame) -> pl.DataFrame:
    """Remove housing-characteristic columns that exceed 50% missing.

    The EDA found 41+ columns in this group with >50% nulls.  These
    features likely reflect applicants who do not own property or for
    whom housing data was simply unavailable.

    Parameters
    ----------
    df : pl.DataFrame
        Application data.

    Returns
    -------
    pl.DataFrame
        DataFrame with high-missingness housing columns removed.
    """
    # Only drop columns that actually exist (safe for both train/test)
    cols_to_drop = [c for c in HOUSING_COLUMNS if c in df.columns]
    return df.drop(cols_to_drop)


# ---------------------------------------------------------------------------
# 8. Fill remaining AMT_REQ_CREDIT_BUREAU_* nulls
# ---------------------------------------------------------------------------
def fill_credit_bureau_inquiries(df: pl.DataFrame) -> pl.DataFrame:
    """Fill missing credit-bureau inquiry counts with 0.

    The six AMT_REQ_CREDIT_BUREAU_* columns record how many enquiries
    were made to the credit bureau in various time windows.  A null
    most likely means no inquiry was made, so 0 is the natural fill
    value.

    Parameters
    ----------
    df : pl.DataFrame
        Application data.

    Returns
    -------
    pl.DataFrame
        DataFrame with bureau inquiry nulls replaced by 0.
    """
    bureau_cols = [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]
    existing = [c for c in bureau_cols if c in df.columns]
    return df.with_columns(
        pl.col(col).fill_null(0) for col in existing
    )


# ---------------------------------------------------------------------------
# 9. Main cleaning pipeline
# ---------------------------------------------------------------------------
def clean_application_data(
    df: pl.DataFrame,
    ext_source_medians: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Apply all cleaning and transformation steps to application data.

    This is the single entry-point that composes every helper function
    above into one reproducible pipeline.  It works identically on both
    the training and test DataFrames.

    Processing order
    ----------------
    1. Fix DAYS_EMPLOYED sentinel (365243 → null + boolean flag)
    2. Add AGE_YEARS derived from DAYS_BIRTH
    3. Replace CODE_GENDER 'XNA' with null
    4. Fill OWN_CAR_AGE for non-car-owners
    5. Median-impute EXT_SOURCE_1 / _2 / _3
    6. Fill OCCUPATION_TYPE nulls with 'Unknown'
    7. Drop high-missingness housing columns (>50 % null)
    8. Fill AMT_REQ_CREDIT_BUREAU_* nulls with 0

    Parameters
    ----------
    df : pl.DataFrame
        Raw application_train or application_test DataFrame.
    ext_source_medians : dict[str, float] | None
        Pre-computed medians for EXT_SOURCE columns.  Pass ``None``
        when cleaning the training set (medians will be computed
        from the data).  Pass the training-set medians when cleaning
        the test set to prevent data leakage.

    Returns
    -------
    pl.DataFrame
        Cleaned DataFrame ready for feature engineering or modeling.

    Examples
    --------
    >>> import polars as pl
    >>> from data_preparation import clean_application_data, get_ext_source_medians
    >>>
    >>> train = pl.read_csv("home-credit-default-risk/application_train.csv")
    >>> test  = pl.read_csv("home-credit-default-risk/application_test.csv")
    >>>
    >>> # Clean training data (medians computed automatically)
    >>> train_clean = clean_application_data(train)
    >>>
    >>> # Compute medians from train for reuse on test
    >>> medians = get_ext_source_medians(train)
    >>> test_clean = clean_application_data(test, ext_source_medians=medians)
    """
    df = fix_days_employed_anomaly(df)
    df = add_age_years(df)
    df = fix_gender_xna(df)
    df = fix_own_car_age(df)
    df = impute_ext_sources(df, medians=ext_source_medians)
    df = fill_occupation_type(df)
    df = drop_high_missing_housing(df)
    df = fill_credit_bureau_inquiries(df)
    return df


# ---------------------------------------------------------------------------
# 10. Helper: extract EXT_SOURCE medians from training data
# ---------------------------------------------------------------------------
def get_ext_source_medians(df: pl.DataFrame) -> dict[str, float]:
    """Compute median values for EXT_SOURCE columns.

    Call this on the *training* data and pass the result as
    ``ext_source_medians`` when cleaning the test data.  This prevents
    data leakage (test statistics never influence imputation values).

    Parameters
    ----------
    df : pl.DataFrame
        Training application data.

    Returns
    -------
    dict[str, float]
        Mapping of column name → median value.
    """
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    return {
        col: df.get_column(col).median()
        for col in ext_cols
    }


# ===================================================================
#  PHASE 2 — FEATURE ENGINEERING
# ===================================================================
# These functions create new columns from the *cleaned* data.
# Every function accepts and returns a pl.DataFrame so they can be
# composed freely, and all work identically on train and test.
# ===================================================================


# ---------------------------------------------------------------------------
# 11. Demographic duration features (positive years)
# ---------------------------------------------------------------------------
def add_demographic_durations(df: pl.DataFrame) -> pl.DataFrame:
    """Convert negative-day columns to positive durations in years.

    The raw data stores DAYS_EMPLOYED, DAYS_REGISTRATION,
    DAYS_ID_PUBLISH, and DAYS_LAST_PHONE_CHANGE as negative integers
    (days before the application).  This function creates human-readable
    *_YEARS counterparts and a derived EMPLOYMENT_TO_AGE_RATIO that
    captures what fraction of an applicant's life they have been
    employed.

    New columns
    -----------
    * EMPLOYED_YEARS       — years employed (null preserved for sentinel rows)
    * REGISTRATION_YEARS   — years since registration
    * ID_PUBLISH_YEARS     — years since ID was published
    * PHONE_CHANGE_YEARS   — years since last phone change
    * EMPLOYED_TO_AGE_RATIO — EMPLOYED_YEARS / AGE_YEARS

    Parameters
    ----------
    df : pl.DataFrame
        Cleaned application data (must already contain AGE_YEARS).

    Returns
    -------
    pl.DataFrame
        DataFrame with new duration columns appended.
    """
    return df.with_columns(
        # Convert each DAYS column: negate and divide by 365.25
        (pl.col("DAYS_EMPLOYED").abs() / 365.25)
        .round(2)
        .alias("EMPLOYED_YEARS"),
        (pl.col("DAYS_REGISTRATION").abs() / 365.25)
        .round(2)
        .alias("REGISTRATION_YEARS"),
        (pl.col("DAYS_ID_PUBLISH").abs() / 365.25)
        .round(2)
        .alias("ID_PUBLISH_YEARS"),
        (pl.col("DAYS_LAST_PHONE_CHANGE").abs() / 365.25)
        .round(2)
        .alias("PHONE_CHANGE_YEARS"),
    ).with_columns(
        # Ratio: fraction of life spent employed
        # Uses a second .with_columns() because it depends on
        # EMPLOYED_YEARS which is created above
        (pl.col("EMPLOYED_YEARS") / pl.col("AGE_YEARS"))
        .round(4)
        .alias("EMPLOYED_TO_AGE_RATIO"),
    )


# ---------------------------------------------------------------------------
# 12. Financial ratio features
# ---------------------------------------------------------------------------
def add_financial_ratios(df: pl.DataFrame) -> pl.DataFrame:
    """Create common financial ratios used in credit-risk modelling.

    New columns
    -----------
    * CREDIT_INCOME_RATIO   — AMT_CREDIT / AMT_INCOME_TOTAL
          How many years of income the credit represents.
    * ANNUITY_INCOME_RATIO  — AMT_ANNUITY / AMT_INCOME_TOTAL
          What share of monthly income goes toward the annuity payment.
    * LOAN_TO_VALUE_RATIO   — AMT_CREDIT / AMT_GOODS_PRICE
          Ratio of credit amount to the price of the goods being
          financed.  Values > 1 mean the loan exceeds the goods price.
    * CREDIT_TERM_MONTHS    — AMT_CREDIT / AMT_ANNUITY
          Estimated loan duration in months.
    * GOODS_TO_INCOME_RATIO — AMT_GOODS_PRICE / AMT_INCOME_TOTAL
          How many years of income the goods price represents.

    Parameters
    ----------
    df : pl.DataFrame
        Cleaned application data.

    Returns
    -------
    pl.DataFrame
        DataFrame with financial ratio columns appended.
    """
    return df.with_columns(
        # Credit-to-income: higher = more leveraged applicant
        (pl.col("AMT_CREDIT") / pl.col("AMT_INCOME_TOTAL"))
        .round(4)
        .alias("CREDIT_INCOME_RATIO"),
        # Annuity-to-income: higher = larger payment burden
        (pl.col("AMT_ANNUITY") / pl.col("AMT_INCOME_TOTAL"))
        .round(4)
        .alias("ANNUITY_INCOME_RATIO"),
        # Loan-to-value: higher = less collateral coverage
        (pl.col("AMT_CREDIT") / pl.col("AMT_GOODS_PRICE"))
        .round(4)
        .alias("LOAN_TO_VALUE_RATIO"),
        # Estimated repayment period in months
        (pl.col("AMT_CREDIT") / pl.col("AMT_ANNUITY"))
        .round(2)
        .alias("CREDIT_TERM_MONTHS"),
        # Goods price relative to income
        (pl.col("AMT_GOODS_PRICE") / pl.col("AMT_INCOME_TOTAL"))
        .round(4)
        .alias("GOODS_TO_INCOME_RATIO"),
    )


# ---------------------------------------------------------------------------
# 13. Missing-data indicator features
# ---------------------------------------------------------------------------

# Columns whose *original* missingness is likely predictive.
# These were identified during EDA: each has a non-trivial missing
# rate and the fact that data is absent may itself signal risk.
_MISSING_INDICATOR_COLS = [
    "DAYS_EMPLOYED",       # 18% — sentinel rows (pensioners / unemployed)
    "OWN_CAR_AGE",         # 66% before fix — residual nulls for car owners
    "OBS_30_CNT_SOCIAL_CIRCLE",  # ~0.3% — absent social-circle data
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "AMT_GOODS_PRICE",     # <0.1% — some revolving loans have no goods price
    "AMT_ANNUITY",         # <0.01%
    "NAME_TYPE_SUITE",     # 0.4%
    "CNT_FAM_MEMBERS",     # tiny
]


def add_missing_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Create boolean flags for columns that had meaningful missing data.

    Missing-data patterns are often informative.  For example, a null
    DAYS_EMPLOYED (after sentinel removal) indicates the applicant is
    not employed — a group with a distinctly lower default rate.  These
    binary flags let tree-based and linear models exploit that signal.

    New columns
    -----------
    One ``MISS_<column_name>`` flag per entry in the internal
    ``_MISSING_INDICATOR_COLS`` list (True = value was null).

    Parameters
    ----------
    df : pl.DataFrame
        Cleaned application data.

    Returns
    -------
    pl.DataFrame
        DataFrame with boolean missing-indicator columns appended.
    """
    existing = [c for c in _MISSING_INDICATOR_COLS if c in df.columns]
    return df.with_columns(
        pl.col(col).is_null().alias(f"MISS_{col}")
        for col in existing
    )


# ---------------------------------------------------------------------------
# 14. Interaction features (EXT_SOURCE combinations)
# ---------------------------------------------------------------------------
def add_ext_source_interactions(df: pl.DataFrame) -> pl.DataFrame:
    """Create interaction and polynomial features from EXT_SOURCE scores.

    The three EXT_SOURCE variables are the strongest individual
    predictors.  Their pairwise products and an aggregate mean capture
    non-linear and combined effects that individual scores miss.

    New columns
    -----------
    * EXT_SOURCE_MEAN      — row-wise mean of the three scores
    * EXT_SOURCE_1x2       — EXT_SOURCE_1 × EXT_SOURCE_2
    * EXT_SOURCE_1x3       — EXT_SOURCE_1 × EXT_SOURCE_3
    * EXT_SOURCE_2x3       — EXT_SOURCE_2 × EXT_SOURCE_3
    * EXT_SOURCE_1_SQ      — EXT_SOURCE_1 squared
    * EXT_SOURCE_2_SQ      — EXT_SOURCE_2 squared
    * EXT_SOURCE_3_SQ      — EXT_SOURCE_3 squared

    Parameters
    ----------
    df : pl.DataFrame
        Cleaned application data (EXT_SOURCE columns already imputed).

    Returns
    -------
    pl.DataFrame
        DataFrame with interaction columns appended.
    """
    return df.with_columns(
        # Row-wise mean of the three external scores
        pl.mean_horizontal(
            "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        )
        .round(6)
        .alias("EXT_SOURCE_MEAN"),
        # Pairwise interaction products
        (pl.col("EXT_SOURCE_1") * pl.col("EXT_SOURCE_2"))
        .round(6)
        .alias("EXT_SOURCE_1x2"),
        (pl.col("EXT_SOURCE_1") * pl.col("EXT_SOURCE_3"))
        .round(6)
        .alias("EXT_SOURCE_1x3"),
        (pl.col("EXT_SOURCE_2") * pl.col("EXT_SOURCE_3"))
        .round(6)
        .alias("EXT_SOURCE_2x3"),
        # Squared terms (captures non-linear effects)
        (pl.col("EXT_SOURCE_1") ** 2)
        .round(6)
        .alias("EXT_SOURCE_1_SQ"),
        (pl.col("EXT_SOURCE_2") ** 2)
        .round(6)
        .alias("EXT_SOURCE_2_SQ"),
        (pl.col("EXT_SOURCE_3") ** 2)
        .round(6)
        .alias("EXT_SOURCE_3_SQ"),
    )


# ---------------------------------------------------------------------------
# 15. Binned variables
# ---------------------------------------------------------------------------
def add_binned_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create discretised (binned) versions of key continuous features.

    Binning helps tree-based models find splits and makes logistic
    regression more robust to outliers and non-linearities.

    New columns
    -----------
    * AGE_BIN              — 5-year age brackets (e.g. "20-24", "25-29", …)
    * CREDIT_INCOME_BIN    — credit-to-income quartile labels
                             ("Low", "Medium", "High", "Very High")
    * EXT_SOURCE_MEAN_BIN  — mean external score quartile labels
                             ("Low", "Medium", "High", "Very High")

    Parameters
    ----------
    df : pl.DataFrame
        Application data that already contains AGE_YEARS,
        CREDIT_INCOME_RATIO, and EXT_SOURCE_MEAN.

    Returns
    -------
    pl.DataFrame
        DataFrame with binned columns appended.
    """
    return df.with_columns(
        # --- Age bins (5-year brackets, matching EDA) ---
        pl.when(pl.col("AGE_YEARS") < 25).then(pl.lit("20-24"))
        .when(pl.col("AGE_YEARS") < 30).then(pl.lit("25-29"))
        .when(pl.col("AGE_YEARS") < 35).then(pl.lit("30-34"))
        .when(pl.col("AGE_YEARS") < 40).then(pl.lit("35-39"))
        .when(pl.col("AGE_YEARS") < 45).then(pl.lit("40-44"))
        .when(pl.col("AGE_YEARS") < 50).then(pl.lit("45-49"))
        .when(pl.col("AGE_YEARS") < 55).then(pl.lit("50-54"))
        .when(pl.col("AGE_YEARS") < 60).then(pl.lit("55-59"))
        .otherwise(pl.lit("60+"))
        .alias("AGE_BIN"),
        # --- Credit-to-income ratio bins (quartile-inspired) ---
        pl.when(pl.col("CREDIT_INCOME_RATIO") < 2.0)
        .then(pl.lit("Low"))
        .when(pl.col("CREDIT_INCOME_RATIO") < 4.0)
        .then(pl.lit("Medium"))
        .when(pl.col("CREDIT_INCOME_RATIO") < 6.0)
        .then(pl.lit("High"))
        .otherwise(pl.lit("Very High"))
        .alias("CREDIT_INCOME_BIN"),
        # --- External source mean bins (quartile-inspired) ---
        pl.when(pl.col("EXT_SOURCE_MEAN") < 0.3)
        .then(pl.lit("Low"))
        .when(pl.col("EXT_SOURCE_MEAN") < 0.5)
        .then(pl.lit("Medium"))
        .when(pl.col("EXT_SOURCE_MEAN") < 0.7)
        .then(pl.lit("High"))
        .otherwise(pl.lit("Very High"))
        .alias("EXT_SOURCE_MEAN_BIN"),
    )


# ---------------------------------------------------------------------------
# 16. Main feature-engineering pipeline
# ---------------------------------------------------------------------------
def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Apply all feature-engineering steps to cleaned application data.

    This is the single entry-point for Phase 2.  It should be called
    *after* ``clean_application_data``.

    Processing order
    ----------------
    1. Demographic durations (positive years + employed-to-age ratio)
    2. Financial ratios (credit-to-income, annuity-to-income,
       loan-to-value, credit term, goods-to-income)
    3. Missing-data indicator flags
    4. EXT_SOURCE interaction and polynomial features
    5. Binned variables (age, credit-to-income, ext-source mean)

    Parameters
    ----------
    df : pl.DataFrame
        Cleaned application data produced by ``clean_application_data``.

    Returns
    -------
    pl.DataFrame
        DataFrame with all engineered features appended.

    Examples
    --------
    >>> train_clean = clean_application_data(train)
    >>> train_feat  = engineer_features(train_clean)
    >>> print(train_feat.shape)
    """
    df = add_demographic_durations(df)
    df = add_financial_ratios(df)
    df = add_missing_indicators(df)
    df = add_ext_source_interactions(df)
    df = add_binned_features(df)
    return df


# ---------------------------------------------------------------------------
# 17. Unified pipeline: prepare_application_data
# ---------------------------------------------------------------------------
def prepare_application_data(
    df: pl.DataFrame,
    params: FittedParams,
    is_train: bool = True,
) -> pl.DataFrame:
    """Apply the complete cleaning and feature engineering pipeline.

    This is the main entry point that combines Phase 1 (cleaning) and
    Phase 2 (feature engineering) into a single function.  It uses
    pre-fitted parameters to ensure train/test consistency.

    Parameters
    ----------
    df : pl.DataFrame
        Raw application_train or application_test DataFrame.
    params : FittedParams
        Parameters fitted from training data via ``fit_params_from_train``.
    is_train : bool, default True
        If True, stores the final column list in ``params.columns_to_keep``.
        If False, aligns columns to match the training set.

    Returns
    -------
    pl.DataFrame
        Fully cleaned and feature-engineered DataFrame.

    Notes
    -----
    Call with ``is_train=True`` first to populate ``params.columns_to_keep``,
    then call with ``is_train=False`` for the test set.
    """
    # Phase 1 — Cleaning
    df = clean_application_data(df, ext_source_medians=params.ext_source_medians)

    # Phase 2 — Feature Engineering
    df = engineer_features(df)

    # Ensure train/test column alignment
    if is_train:
        # Store columns from training (excluding TARGET for comparison)
        params.columns_to_keep = [c for c in df.columns if c != "TARGET"]
    else:
        # Align test columns to match training (excluding TARGET)
        # Add any missing columns as nulls, drop any extra columns
        for col in params.columns_to_keep:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))
        # Keep only columns that are in training set
        df = df.select(params.columns_to_keep)

    return df


def align_train_test(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Ensure train and test DataFrames have identical columns (except TARGET).

    This function aligns the columns of train and test DataFrames after
    all processing is complete.  It ensures that both have the same
    feature columns in the same order.

    Parameters
    ----------
    train_df : pl.DataFrame
        Processed training DataFrame (should contain TARGET).
    test_df : pl.DataFrame
        Processed test DataFrame (should not contain TARGET).

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Aligned (train_df, test_df) with identical columns except TARGET.
    """
    # Get columns excluding TARGET
    train_cols = [c for c in train_df.columns if c != "TARGET"]
    test_cols = test_df.columns

    # Find any mismatches
    missing_in_test = set(train_cols) - set(test_cols)
    extra_in_test = set(test_cols) - set(train_cols)

    # Add missing columns to test as nulls
    for col in missing_in_test:
        # Infer dtype from train
        dtype = train_df.get_column(col).dtype
        test_df = test_df.with_columns(pl.lit(None).cast(dtype).alias(col))

    # Remove extra columns from test
    if extra_in_test:
        test_df = test_df.drop(list(extra_in_test))

    # Reorder test columns to match train order
    test_df = test_df.select(train_cols)

    # Reorder train to have TARGET last (if present)
    if "TARGET" in train_df.columns:
        train_df = train_df.select(train_cols + ["TARGET"])

    return train_df, test_df


# ===================================================================
#  PHASE 3 — SUPPLEMENTARY TABLE AGGREGATIONS
# ===================================================================
# These functions aggregate data from supplementary CSVs to the
# applicant level (SK_ID_CURR) so they can be joined to the main
# application data.
# ===================================================================


# ---------------------------------------------------------------------------
# 17. Bureau aggregations
# ---------------------------------------------------------------------------
def aggregate_bureau(bureau: pl.DataFrame) -> pl.DataFrame:
    """Aggregate bureau.csv (Credit Bureau data) to the applicant level.

    Bureau data contains records of prior credits from other financial
    institutions reported to the Credit Bureau.  Each applicant may have
    0, 1, or many bureau records.

    New columns (all prefixed ``BUREAU_``)
    --------------------------------------
    **Counts:**
    * BUREAU_COUNT              — total number of bureau credits
    * BUREAU_ACTIVE_COUNT       — credits with status "Active"
    * BUREAU_CLOSED_COUNT       — credits with status "Closed"

    **Ratios:**
    * BUREAU_ACTIVE_RATIO       — active / total credits
    * BUREAU_DEBT_CREDIT_RATIO  — total debt / total credit sum

    **Overdue indicators:**
    * BUREAU_OVERDUE_COUNT      — credits with any days overdue > 0
    * BUREAU_OVERDUE_RATIO      — overdue credits / total credits
    * BUREAU_SUM_OVERDUE        — sum of AMT_CREDIT_SUM_OVERDUE
    * BUREAU_MAX_OVERDUE        — max of AMT_CREDIT_MAX_OVERDUE
    * BUREAU_MEAN_DAYS_OVERDUE  — mean of CREDIT_DAY_OVERDUE

    **Amounts (aggregated):**
    * BUREAU_SUM_CREDIT         — sum of AMT_CREDIT_SUM
    * BUREAU_SUM_DEBT           — sum of AMT_CREDIT_SUM_DEBT
    * BUREAU_MEAN_CREDIT        — mean credit amount

    **Recency:**
    * BUREAU_DAYS_CREDIT_MEAN   — mean days since credit was opened
    * BUREAU_DAYS_CREDIT_MIN    — most recent credit (closest to 0)

    Parameters
    ----------
    bureau : pl.DataFrame
        Raw bureau.csv DataFrame.

    Returns
    -------
    pl.DataFrame
        Aggregated DataFrame with one row per SK_ID_CURR.
    """
    return (
        bureau
        .group_by("SK_ID_CURR")
        .agg(
            # --- Counts ---
            pl.len().alias("BUREAU_COUNT"),
            (pl.col("CREDIT_ACTIVE") == "Active")
            .sum()
            .alias("BUREAU_ACTIVE_COUNT"),
            (pl.col("CREDIT_ACTIVE") == "Closed")
            .sum()
            .alias("BUREAU_CLOSED_COUNT"),
            # --- Overdue counts ---
            (pl.col("CREDIT_DAY_OVERDUE") > 0)
            .sum()
            .alias("BUREAU_OVERDUE_COUNT"),
            # --- Amounts ---
            pl.col("AMT_CREDIT_SUM").sum().alias("BUREAU_SUM_CREDIT"),
            pl.col("AMT_CREDIT_SUM_DEBT").sum().alias("BUREAU_SUM_DEBT"),
            pl.col("AMT_CREDIT_SUM").mean().alias("BUREAU_MEAN_CREDIT"),
            pl.col("AMT_CREDIT_SUM_OVERDUE").sum().alias("BUREAU_SUM_OVERDUE"),
            pl.col("AMT_CREDIT_MAX_OVERDUE").max().alias("BUREAU_MAX_OVERDUE"),
            pl.col("CREDIT_DAY_OVERDUE").mean().alias("BUREAU_MEAN_DAYS_OVERDUE"),
            # --- Recency ---
            pl.col("DAYS_CREDIT").mean().alias("BUREAU_DAYS_CREDIT_MEAN"),
            pl.col("DAYS_CREDIT").max().alias("BUREAU_DAYS_CREDIT_MIN"),
        )
        .with_columns(
            # --- Derived ratios (computed after aggregation) ---
            (pl.col("BUREAU_ACTIVE_COUNT") / pl.col("BUREAU_COUNT"))
            .round(4)
            .alias("BUREAU_ACTIVE_RATIO"),
            (pl.col("BUREAU_OVERDUE_COUNT") / pl.col("BUREAU_COUNT"))
            .round(4)
            .alias("BUREAU_OVERDUE_RATIO"),
            (pl.col("BUREAU_SUM_DEBT") / pl.col("BUREAU_SUM_CREDIT"))
            .round(4)
            .alias("BUREAU_DEBT_CREDIT_RATIO"),
        )
    )


# ---------------------------------------------------------------------------
# 18. Previous application aggregations
# ---------------------------------------------------------------------------
def aggregate_previous_application(prev_app: pl.DataFrame) -> pl.DataFrame:
    """Aggregate previous_application.csv to the applicant level.

    Previous application data contains records of all prior loan
    applications made by the applicant at Home Credit.  Each applicant
    may have 0, 1, or many previous applications.

    New columns (all prefixed ``PREV_``)
    ------------------------------------
    **Counts:**
    * PREV_APP_COUNT            — total previous applications
    * PREV_APPROVED_COUNT       — applications that were approved
    * PREV_REFUSED_COUNT        — applications that were refused
    * PREV_CANCELED_COUNT       — applications that were canceled

    **Ratios:**
    * PREV_APPROVAL_RATE        — approved / total applications
    * PREV_REFUSAL_RATE         — refused / total applications

    **Amounts:**
    * PREV_AMT_APPLICATION_MEAN — mean requested amount
    * PREV_AMT_CREDIT_MEAN      — mean granted credit amount
    * PREV_AMT_APPLICATION_SUM  — total requested amount
    * PREV_CREDIT_REQUEST_RATIO — mean(granted) / mean(requested)

    **Recency:**
    * PREV_DAYS_DECISION_MEAN   — mean days since decision
    * PREV_DAYS_DECISION_MIN    — most recent decision (closest to 0)

    Parameters
    ----------
    prev_app : pl.DataFrame
        Raw previous_application.csv DataFrame.

    Returns
    -------
    pl.DataFrame
        Aggregated DataFrame with one row per SK_ID_CURR.
    """
    return (
        prev_app
        .group_by("SK_ID_CURR")
        .agg(
            # --- Counts ---
            pl.len().alias("PREV_APP_COUNT"),
            (pl.col("NAME_CONTRACT_STATUS") == "Approved")
            .sum()
            .alias("PREV_APPROVED_COUNT"),
            (pl.col("NAME_CONTRACT_STATUS") == "Refused")
            .sum()
            .alias("PREV_REFUSED_COUNT"),
            (pl.col("NAME_CONTRACT_STATUS") == "Canceled")
            .sum()
            .alias("PREV_CANCELED_COUNT"),
            # --- Amounts ---
            pl.col("AMT_APPLICATION").mean().alias("PREV_AMT_APPLICATION_MEAN"),
            pl.col("AMT_CREDIT").mean().alias("PREV_AMT_CREDIT_MEAN"),
            pl.col("AMT_APPLICATION").sum().alias("PREV_AMT_APPLICATION_SUM"),
            # --- Recency ---
            pl.col("DAYS_DECISION").mean().alias("PREV_DAYS_DECISION_MEAN"),
            pl.col("DAYS_DECISION").max().alias("PREV_DAYS_DECISION_MIN"),
        )
        .with_columns(
            # --- Derived ratios ---
            (pl.col("PREV_APPROVED_COUNT") / pl.col("PREV_APP_COUNT"))
            .round(4)
            .alias("PREV_APPROVAL_RATE"),
            (pl.col("PREV_REFUSED_COUNT") / pl.col("PREV_APP_COUNT"))
            .round(4)
            .alias("PREV_REFUSAL_RATE"),
            (pl.col("PREV_AMT_CREDIT_MEAN") / pl.col("PREV_AMT_APPLICATION_MEAN"))
            .round(4)
            .alias("PREV_CREDIT_REQUEST_RATIO"),
        )
    )


# ---------------------------------------------------------------------------
# 19. Installments payments aggregations
# ---------------------------------------------------------------------------
def aggregate_installments(installments: pl.DataFrame) -> pl.DataFrame:
    """Aggregate installments_payments.csv to the applicant level.

    Installments data contains payment records for previous loans.
    Late payments are identified where DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT
    (payment made after due date).  Underpayments are identified where
    AMT_PAYMENT < AMT_INSTALMENT.

    New columns (all prefixed ``INST_``)
    ------------------------------------
    **Counts:**
    * INST_COUNT                — total installment records
    * INST_LATE_COUNT           — payments made after due date
    * INST_ONTIME_COUNT         — payments made on or before due date
    * INST_UNDERPAY_COUNT       — payments less than amount due

    **Ratios:**
    * INST_LATE_RATIO           — late payments / total payments
    * INST_UNDERPAY_RATIO       — underpayments / total payments

    **Payment behavior:**
    * INST_DAYS_LATE_MEAN       — mean days late (negative = early)
    * INST_DAYS_LATE_MAX        — maximum days late
    * INST_PAYMENT_DIFF_MEAN    — mean (paid - due), negative = underpaid
    * INST_PAYMENT_RATIO_MEAN   — mean (paid / due)

    **Amounts:**
    * INST_AMT_PAYMENT_SUM      — total amount paid
    * INST_AMT_INSTALMENT_SUM   — total amount due

    Parameters
    ----------
    installments : pl.DataFrame
        Raw installments_payments.csv DataFrame.

    Returns
    -------
    pl.DataFrame
        Aggregated DataFrame with one row per SK_ID_CURR.
    """
    return (
        installments
        # Compute per-record metrics first
        .with_columns(
            # Days late: positive = late, negative = early
            (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT"))
            .alias("DAYS_LATE"),
            # Payment difference: negative = underpaid
            (pl.col("AMT_PAYMENT") - pl.col("AMT_INSTALMENT"))
            .alias("PAYMENT_DIFF"),
            # Payment ratio
            (pl.col("AMT_PAYMENT") / pl.col("AMT_INSTALMENT"))
            .alias("PAYMENT_RATIO"),
        )
        .group_by("SK_ID_CURR")
        .agg(
            # --- Counts ---
            pl.len().alias("INST_COUNT"),
            (pl.col("DAYS_LATE") > 0).sum().alias("INST_LATE_COUNT"),
            (pl.col("DAYS_LATE") <= 0).sum().alias("INST_ONTIME_COUNT"),
            (pl.col("PAYMENT_DIFF") < 0).sum().alias("INST_UNDERPAY_COUNT"),
            # --- Payment timing ---
            pl.col("DAYS_LATE").mean().alias("INST_DAYS_LATE_MEAN"),
            pl.col("DAYS_LATE").max().alias("INST_DAYS_LATE_MAX"),
            # --- Payment amounts ---
            pl.col("PAYMENT_DIFF").mean().alias("INST_PAYMENT_DIFF_MEAN"),
            pl.col("PAYMENT_RATIO").mean().alias("INST_PAYMENT_RATIO_MEAN"),
            pl.col("AMT_PAYMENT").sum().alias("INST_AMT_PAYMENT_SUM"),
            pl.col("AMT_INSTALMENT").sum().alias("INST_AMT_INSTALMENT_SUM"),
        )
        .with_columns(
            # --- Derived ratios ---
            (pl.col("INST_LATE_COUNT") / pl.col("INST_COUNT"))
            .round(4)
            .alias("INST_LATE_RATIO"),
            (pl.col("INST_UNDERPAY_COUNT") / pl.col("INST_COUNT"))
            .round(4)
            .alias("INST_UNDERPAY_RATIO"),
        )
    )


# ---------------------------------------------------------------------------
# 20. Join all supplementary aggregations to application data
# ---------------------------------------------------------------------------
def join_supplementary_features(
    app_df: pl.DataFrame,
    bureau_agg: pl.DataFrame | None = None,
    prev_app_agg: pl.DataFrame | None = None,
    installments_agg: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Join pre-aggregated supplementary tables to application data.

    This convenience function performs left joins of all supplementary
    aggregations to the main application DataFrame.  Applicants without
    records in a supplementary table will have null values for those
    columns.

    Parameters
    ----------
    app_df : pl.DataFrame
        Application data (cleaned and/or feature-engineered).
    bureau_agg : pl.DataFrame | None
        Output of ``aggregate_bureau()``.  Pass ``None`` to skip.
    prev_app_agg : pl.DataFrame | None
        Output of ``aggregate_previous_application()``.  Pass ``None`` to skip.
    installments_agg : pl.DataFrame | None
        Output of ``aggregate_installments()``.  Pass ``None`` to skip.

    Returns
    -------
    pl.DataFrame
        Application data with supplementary features joined.

    Examples
    --------
    >>> bureau_agg = aggregate_bureau(bureau)
    >>> prev_agg = aggregate_previous_application(prev_app)
    >>> inst_agg = aggregate_installments(installments)
    >>>
    >>> train_full = join_supplementary_features(
    ...     train_feat,
    ...     bureau_agg=bureau_agg,
    ...     prev_app_agg=prev_agg,
    ...     installments_agg=inst_agg,
    ... )
    """
    result = app_df

    if bureau_agg is not None:
        result = result.join(bureau_agg, on="SK_ID_CURR", how="left")

    if prev_app_agg is not None:
        result = result.join(prev_app_agg, on="SK_ID_CURR", how="left")

    if installments_agg is not None:
        result = result.join(installments_agg, on="SK_ID_CURR", how="left")

    return result
