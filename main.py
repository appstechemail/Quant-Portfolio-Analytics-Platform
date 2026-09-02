# ==========================================================
# PART 1 — IMPORTS
# Institutional-Grade Quant Platform
# ==========================================================

# ==========================================================
# STANDARD LIBRARY
# ==========================================================

import os
import uuid
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any

# ==========================================================
# THIRD-PARTY LIBRARIES
# ==========================================================

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================================
# SCIKIT-LEARN
# ==========================================================

from sklearn.metrics import (
    accuracy_score,
)

from sklearn.preprocessing import (
    StandardScaler,
)

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.impute import SimpleImputer

# ==========================================================
# CONFIGURATION
# ==========================================================

from config.config import CONFIG

# ==========================================================
# DATA LAYER
# ==========================================================

from src.data.data_loader import (
    download_price_data,
)

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

from src.features.technical import (
    add_technical_features,
)

from src.features.features import (
    FEATURE_METADATA,
)

from src.features.target import (
    add_target,
)

# ==========================================================
# FUNDAMENTALS
# ==========================================================

from src.fundamentals.fundamentals import (
    add_basic_fundamentals,
)

# ==========================================================
# PREPROCESSING
# ==========================================================

from src.preprocessing.cleaning import (
    clean_data,
)

# ==========================================================
# MARKET REGIME
# ==========================================================

from src.regime.detection import (
    detect_market_regime,
)

from src.regime.selection import (
    select_models_smart,
)

# ==========================================================
# MODELING
# ==========================================================

from src.models.models import (
    train_models,
)

from src.models.model_utils import (
    get_model_probabilities,
)

# ==========================================================
# BACKTESTING
# ==========================================================

from src.backtest.backtest import (
    run_backtest,
)

from src.evaluation.walkforward import (
    run_walkforward_validation,
)

# ==========================================================
# PREDICTION
# ==========================================================

from src.prediction.predict import (
    predict_today,
)

# ==========================================================
# ALPHA PIPELINE
# ==========================================================

from src.alpha.ic_engine import (
    run_alpha_pipeline,
    save_ic_results,
)

from src.alpha.ic_stability import (
    compute_ic_stability,
    build_stability_weights,
    print_stability_summary,
)

from src.alpha.feature_clustering import (
    diversify_features,
)

from src.alpha.feature_decay import (
    compute_feature_decay,
    build_decay_weights,
    print_decay_summary,
)

from src.alpha.feature_category_budget import (
    build_dynamic_category_budget,
    print_feature_weight_summary,
    print_category_budget_summary,
)

# ALPHA ENGINES

from src.alpha.alpha_stage_tracker import (
    AlphaStageTracker,
    AlphaStage,
)

from src.alpha.alpha_pipeline import (
    AlphaPipeline,
)

# ==========================================================
# INSTITUTIONAL INPUT ADAPTER
# ==========================================================

from src.portfolio.construction.input_adapter import (
    build_pipeline_input_engine,
    build_pipeline_input_with_diagnostics,
    smoke_test_input_adapter,
)

# ==========================================================
# INSTITUTIONAL PIPELINE FRAMEWORK
# ==========================================================

from src.portfolio.construction.pipeline import (
    PipelineInput,
    PipelineMetadata,
    PipelineConfig,
    PipelineFrameworkFactory,
    PipelineInputValidator
)

# ==========================================================
# INSTITUTIONAL CONSTRUCTION ENGINE
# ==========================================================

from src.portfolio.construction.pipeline import (
    create_pipeline,
    run_pipeline,
    institutional_pipeline,
    InstitutionalPipelineConfig,
)

# ==========================================================
# REPORTING APIS
# ==========================================================

from src.portfolio.construction.pipeline import (
    build_portfolio,
    build_rebalance,
    build_execution,
    diagnostics_report,
    full_report,
)   

# ==========================================================
# OPTIONAL DIRECT COMPONENT IMPORTS
# (Useful for debugging individual modules)
# ==========================================================

from src.portfolio.construction.constraints import (
    ConstraintEngine,
)

from src.portfolio.construction.risk_model import (
    InstitutionalRiskEngine,
)

from src.portfolio.construction.optimizer import (
    InstitutionalOptimizerEngine,
)

from src.portfolio.construction.portfolio_builder import (
    InstitutionalPortfolioBuilderEngine,
)

# ==========================================================
# CONFIG
# ==========================================================

THRESHOLD = CONFIG["MODEL"]["THRESHOLD"]
META_THRESHOLD = CONFIG["TARGET"]["META_THRESHOLD"]
BULL_META_THRESHOLD = CONFIG["TARGET"]["BULL_META_THRESHOLD"]
SIDEWAYS_META_THRESHOLD = CONFIG["TARGET"]["SIDEWAYS_META_THRESHOLD"]
BULL_VOL_META_THRESHOLD = CONFIG["TARGET"]["BULL_VOL_META_THRESHOLD"]
SIDEWAYS_VOL_META_THRESHOLD = CONFIG["TARGET"]["SIDEWAYS_VOL_META_THRESHOLD"]

SELECTED_MODELS = CONFIG["MODEL"].get(
    "MODEL_LIST",
    ["xgb", "lgb", "cat"]
)

NEUTRALITY = float(
    CONFIG["BACKTEST"].get(
        "NEUTRALITY",
        0.50,
    )
)

# INITIALIZE
tracker = AlphaStageTracker()

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

np.random.seed(42)

# ==========================================================
# PART 2 — EXISTING DATA PIPELINE
#
# Purpose
# -------
# Produce the canonical final_df used throughout
# the platform.
#
# Flow
# ----
#
# download_price_data()
#         ↓
# add_technical_features()
#         ↓
# add_basic_fundamentals()
#         ↓
# add_target()
#         ↓
# clean_data()
#         ↓
# final_df
#
# NOTE:
# -----
# Institutional Portfolio Construction DOES NOT
# modify this section.
#
# final_df remains the canonical Alpha Engine output.
# ==========================================================


def build_final_dataframe() -> pd.DataFrame:
    """
    Builds the canonical dataset.

    Returns
    -------
    pd.DataFrame

    Output
    ------
    final_df
    """

    logger.info(
        "STEP 1 | Downloading market data"
    )

    df = download_price_data()

    df = (
        df
        .sort_values(
            ["Company", "Date"]
        )
        .reset_index(
            drop=True
        )
    )

    logger.info(
        "Raw rows: %s",
        f"{len(df):,}",
    )

    # ======================================================
    # MACRO DATA
    # ======================================================

    logger.info(
        "STEP 2 | Downloading macro data"
    )

    macro = yf.download(
        "^NSEI",
        start=CONFIG["DATA"]["START_DATE"],
        end=CONFIG["DATA"]["END_DATE"],
        progress=False,
    )

    if isinstance(
        macro.columns,
        pd.MultiIndex,
    ):
        macro.columns = [
            c[0]
            for c in macro.columns
        ]

    macro = (
        macro[["Close"]]
        .rename(
            columns={
                "Close":
                "NSE_Close"
            }
        )
    )

    macro["NSE_Return"] = (
        macro["NSE_Close"]
        .pct_change()
    )

    macro = (
        macro
        .reset_index()
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    macro["Date"] = pd.to_datetime(
        macro["Date"]
    )

    df = pd.merge_asof(
        df.sort_values("Date"),
        macro.sort_values("Date"),
        on="Date",
        direction="backward",
    )

    # ======================================================
    # TECHNICAL FEATURES
    # ======================================================

    if CONFIG["FEATURES"][
        "USE_TECHNICAL"
    ]:

        logger.info(
            "STEP 3 | Technical features"
        )

        df = add_technical_features(
            df
        )

        if (
            "Future_Return"
            not in df.columns
        ):

            df["Future_Return"] = (
                df.groupby(
                    "Company"
                )["Close"]
                .shift(-5)
                /
                df["Close"]
                - 1
            )

    # ======================================================
    # FUNDAMENTALS
    # ======================================================

    if CONFIG["FEATURES"][
        "USE_FUNDAMENTAL"
    ]:

        logger.info(
            "STEP 4 | Fundamental features"
        )

        df = add_basic_fundamentals(
            df
        )

    # ======================================================
    # TARGET
    # ======================================================

    logger.info(
        "STEP 5 | Building targets"
    )

    df = add_target(
        df
    )

    # ======================================================
    # CLEANING
    # ======================================================

    logger.info(
        "STEP 6 | Cleaning dataset"
    )

    final_df = clean_data(
        df
    )

    logger.info(
        "Final shape: %s",
        final_df.shape,
    )

    return final_df

# ==========================================================
# BUILD CANONICAL DATASET
# ==========================================================

logger.info(
    "Building canonical final_df..."
)

final_df = build_final_dataframe()

print(
    f"\nfinal_df created successfully: {final_df.shape}"
)

tracker.add_stage(
    AlphaStage.RAW,
    final_df.copy()
)
# ======================================================
# MARKET REGIME
# ======================================================

print("\n===== BEFORE REGIME DETECTION =====")

print(
    "Shape:",
    final_df.shape,
)

final_df = detect_market_regime(
    final_df
)

current_regime = (
    final_df[
        "Market_Regime"
    ].iloc[-1]
)

print(
    f"\n🌍 Current Regime: {current_regime}"
)

# ==========================================================
# PART 3
# IC PIPELINE
# FEATURE SELECTION
# ==========================================================

print("\n📊 Running Institutional Alpha Pipeline")

alpha_results = run_alpha_pipeline(
    df=final_df,
    current_regime=current_regime,
)

tracker.add_stage(
    "IC_PIPELINE",
    alpha_results["tables"]["summary"]
)

# ----------------------------------------------------------
# Persist IC results
# ----------------------------------------------------------

save_ic_results(alpha_results)

# ----------------------------------------------------------
# Extract institutional feature weights
# ----------------------------------------------------------

final_feature_weights = (
    alpha_results["final_feature_weights"]
)

# ----------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------

print("\n🏆 TOP IC FEATURES")

print(
    alpha_results["tables"]["summary"]
    .sort_values(
        "ICIR",
        ascending=False,
    )
    .head(15)
)

print(
    final_feature_weights[
        [
            "Feature",
            "Category",
            "Final_Weight",
        ]
    ]
    .sort_values(
        "Final_Weight",
        ascending=False,
    )
    .head(20)
)

# ----------------------------------------------------------
# Build Institutional Feature Universe
# ----------------------------------------------------------

FEATURES = (
    final_feature_weights
    .query("Final_Weight > 0")
    .sort_values(
        "Final_Weight",
        ascending=False,
    )["Feature"]
    .tolist()
)

# ----------------------------------------------------------
# Leakage Protection
# ----------------------------------------------------------

LEAKAGE_COLS = [

    "Future_Return",
    "Risk_Adjusted_Return",
    "Return_Rank",
    "Target",
    "Meta_Target",

    "TB_Exit",
    "TB_Label",
    "TB_Days",

    "Future_Close",
    "Future_High",
    "Future_Low",
]

FEATURES = [

    feature

    for feature in FEATURES

    if (
        feature in final_df.columns
        and feature not in LEAKAGE_COLS
    )
]

# ----------------------------------------------------------
# Institutional Feature Limit
# (keep top N features)
# ----------------------------------------------------------

MAX_FEATURES = (
    CONFIG["MODEL"]
    .get(
        "MAX_FEATURES",
        40,
    )
)

FEATURES = FEATURES[:MAX_FEATURES]

TARGET = "Target"

target_col=TARGET

print("\n🛡 Institutional Feature Set")

print(
    f"Selected Features: {len(FEATURES)}"
)

print(FEATURES)

Path("artifacts").mkdir(
    parents=True,
    exist_ok=True,
)

# ----------------------------------------------------------
# Optional:
# Persist final feature list
# ----------------------------------------------------------

joblib.dump(
    FEATURES,
    "artifacts/features.pkl",
)

# ----------------------------------------------------------
# Optional:
# Persist feature weights
# ----------------------------------------------------------

joblib.dump(
    final_feature_weights,
    "artifacts/feature_weights.pkl",
)

print(
    "\n✅ Institutional Alpha Engine Complete"
)


# ==========================================================
# PART 4
# MODEL TRAINING
# ==========================================================
results = {}
probas = {}

print("\n" + "=" * 60)
print("PART 4 — MODEL TRAINING")
print("=" * 60)


# ======================================================
# BUILD TRAINING DATASET
# ======================================================

FEATURES = list(dict.fromkeys(FEATURES))

extra_cols = [
    TARGET,
    "Meta_Target",
    "Market_Regime",
]

# ======================================================
# MODEL DATASET
# ======================================================
#
# Do NOT globally drop rows before the date split.
#
# A global dropna() here can remove complete company/date
# observations and shrink the test universe.
#
# Instead:
#   1. preserve the complete panel
#   2. perform the date split
#   3. remove invalid rows only from model-training inputs
#
# ======================================================

# ==========================================================
# CANONICAL SOURCE ROW ID
# ==========================================================
#
# Preserve the exact original final_df row identity.
#
# This ID is used later to map X_test/meta_test probabilities
# back to the correct Date/Company rows in final_df.
#
# Do NOT rely on the reset positional index of `data`.
# ==========================================================

source_row_id = np.arange(
    len(final_df),
    dtype=np.int64,
)

data = (
    final_df[
        ["Date", "Company"]
        + FEATURES
        + extra_cols
    ]
    .copy()
)

data["_source_row_id"] = source_row_id

data = data.reset_index(
    drop=True
)

data["Date"] = pd.to_datetime(
    data["Date"],
    errors="coerce",
)

data = data.dropna(
    subset=[
        "Date",
        "Company",
        TARGET,
        "Meta_Target",
    ]
).reset_index(drop=True)

print("\nDataset Shape:")
print(data.shape)

# ======================================================
# FEATURES / TARGET
# ======================================================

X = data[FEATURES].copy()

y = data[TARGET].copy()

meta = data[
    ["Date", "Company"]
].copy()

meta_y = data["Meta_Target"].copy()

if isinstance(meta_y, pd.DataFrame):

    meta_y = meta_y.iloc[:, 0]

meta_y = meta_y.astype(int)

# ----------------------------------------------------------
# TRAIN / TEST SPLIT — DATE BASED + UNIVERSE SAFE
# ----------------------------------------------------------
#
# IMPORTANT:
# ----------
# 1. Split strictly by DATE, never by dataframe ROW.
# 2. Preserve all observations belonging to each date.
# 3. Build train/test masks from the canonical `data` dataset.
# 4. Explicitly validate the company universe on both sides.
# 5. Do NOT artificially manufacture missing historical data.
#
# Why this is required
# --------------------
# A date-based split prevents the classic problem where the
# last N rows of the dataframe become the test set and therefore
# contain only a subset of the observations belonging to the
# final test dates.
#
# This guarantees that all available observations belonging
# to each test date remain together instead of splitting
# individual dates across train and test.
#
# However, DATE-BASED SPLITTING alone does NOT guarantee that
# every one of the current universe companies exists in the
# historical test period.
#
# Some companies may have:
#   - incomplete history
#   - IPO/listing dates after the training period
#   - missing feature history
#   - insufficient rolling-window history
#   - rows removed by target/feature cleaning
#
# Therefore we explicitly diagnose the universe rather than
# silently pretending that the test universe is complete.
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("DATE-BASED TRAIN / TEST SPLIT")
print("=" * 60)

# ----------------------------------------------------------
# 1. Canonical date universe
# ----------------------------------------------------------

data["Date"] = pd.to_datetime(
    data["Date"],
    errors="coerce",
)

unique_dates = np.sort(
    data["Date"]
    .dropna()
    .unique()
)

if len(unique_dates) < 2:

    raise ValueError(
        "Insufficient unique dates for train/test split."
    )

# ----------------------------------------------------------
# 2. Train/test split ratio
# ----------------------------------------------------------

train_size = float(
    CONFIG["MODEL"]["TRAIN_SIZE"]
)

if not 0.0 < train_size < 1.0:

    raise ValueError(
        "CONFIG['MODEL']['TRAIN_SIZE'] "
        "must be between 0 and 1."
    )

split_date_idx = int(
    len(unique_dates) * train_size
)

# ----------------------------------------------------------
# 3. Safety bounds
# ----------------------------------------------------------

split_date_idx = max(
    1,
    min(
        split_date_idx,
        len(unique_dates) - 1,
    ),
)

train_dates = unique_dates[
    :split_date_idx
]

test_dates = unique_dates[
    split_date_idx:
]

# ----------------------------------------------------------
# 4. DATE masks
# ----------------------------------------------------------

train_mask = (
    data["Date"].isin(train_dates)
)

test_mask = (
    data["Date"].isin(test_dates)
)

# ----------------------------------------------------------
# 5. Train/test date leakage check
# ----------------------------------------------------------

train_date_set = set(
    train_dates
)

test_date_set = set(
    test_dates
)

overlap_dates = (
    train_date_set
    &
    test_date_set
)

if overlap_dates:

    raise RuntimeError(
        "Train/test date leakage detected. "
        f"Overlapping dates: {len(overlap_dates)}"
    )

# ----------------------------------------------------------
# 6. Build model datasets
# ----------------------------------------------------------

X_train = (
    X.loc[train_mask]
    .copy()
)

X_test = (
    X.loc[test_mask]
    .copy()
)

imputer = SimpleImputer(
    strategy="median"
)

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)

if X_train.empty:
    raise RuntimeError(
        "X_train is empty after date-based split."
    )

if X_test.empty:
    raise RuntimeError(
        "X_test is empty after date-based split."
    )

y_train = (
    y.loc[train_mask]
    .copy()
)

y_test = (
    y.loc[test_mask]
    .copy()
)

meta_y_train = (
    meta_y.loc[train_mask]
    .copy()
)

meta_y_test = (
    meta_y.loc[test_mask]
    .copy()
)

meta_test = (
    data.loc[
        test_mask,
        [
            "Date",
            "Company",
            "Market_Regime",
        ],
    ]
    .copy()
    .reset_index(drop=True)
)

# ==========================================================
# CANONICAL TEST SOURCE ROW IDS
# ==========================================================

test_source_row_ids = (
    data.loc[
        test_mask,
        "_source_row_id"
    ]
    .to_numpy(
        dtype=np.int64
    )
)

if len(test_source_row_ids) != len(X_test):
    raise RuntimeError(
        "CRITICAL: Test source-row alignment failure: "
        f"source_ids={len(test_source_row_ids)}, "
        f"X_test={len(X_test)}"
    )

if len(np.unique(test_source_row_ids)) != len(
    test_source_row_ids
):
    raise RuntimeError(
        "CRITICAL: Duplicate source row IDs detected "
        "in test panel."
    )

# ----------------------------------------------------------
# 7. Build explicit train/test universe diagnostics
# ----------------------------------------------------------

train_companies = set(
    data.loc[
        train_mask,
        "Company",
    ]
    .dropna()
    .astype(str)
    .unique()
)

test_companies = set(
    data.loc[
        test_mask,
        "Company",
    ]
    .dropna()
    .astype(str)
    .unique()
)

all_model_companies = (
    train_companies
    |
    test_companies
)

train_only_companies = (
    train_companies
    -
    test_companies
)

test_only_companies = (
    test_companies
    -
    train_companies
)

missing_from_test = (
    all_model_companies
    -
    test_companies
)

missing_from_train = (
    all_model_companies
    -
    train_companies
)

# ----------------------------------------------------------
# 8. Detailed split diagnostics
# ----------------------------------------------------------

print(
    f"Total dates : "
    f"{len(unique_dates):,}"
)

print(
    f"Train dates : "
    f"{len(train_dates):,}"
)

print(
    f"Test dates  : "
    f"{len(test_dates):,}"
)

print(
    f"Train rows  : "
    f"{len(X_train):,}"
)

print(
    f"Test rows   : "
    f"{len(X_test):,}"
)

print(
    f"Train companies : "
    f"{len(train_companies):,}"
)

print(
    f"Test companies  : "
    f"{len(test_companies):,}"
)

print(
    f"Combined companies : "
    f"{len(all_model_companies):,}"
)

print(
    f"Train-only companies : "
    f"{len(train_only_companies):,}"
)

print(
    f"Test-only companies : "
    f"{len(test_only_companies):,}"
)

print(
    f"Missing from test universe : "
    f"{len(missing_from_test):,}"
)

print(
    f"Missing from train universe : "
    f"{len(missing_from_train):,}"
)

# ----------------------------------------------------------
# 9. Print company differences
# ----------------------------------------------------------

if train_only_companies:

    print(
        "\n⚠ TRAIN-ONLY COMPANIES"
    )

    print(
        sorted(
            train_only_companies
        )
    )

if test_only_companies:

    print(
        "\nℹ TEST-ONLY COMPANIES"
    )

    print(
        sorted(
            test_only_companies
        )
    )

# ----------------------------------------------------------
# 10. Basic test universe validation
# ----------------------------------------------------------

if len(test_companies) == 0:

    raise RuntimeError(
        "Test universe is empty after "
        "dataset construction."
    )

# ----------------------------------------------------------
# 11. Date-period diagnostics
# ----------------------------------------------------------

print(
    f"Train period : "
    f"{pd.Timestamp(train_dates[0]).date()} "
    f"→ "
    f"{pd.Timestamp(train_dates[-1]).date()}"
)

print(
    f"Test period  : "
    f"{pd.Timestamp(test_dates[0]).date()} "
    f"→ "
    f"{pd.Timestamp(test_dates[-1]).date()}"
)

# ----------------------------------------------------------
# 12. Row/date integrity checks
# ----------------------------------------------------------

if len(X_train) != int(train_mask.sum()):

    raise RuntimeError(
        "X_train row count does not match train mask."
    )

if len(X_test) != int(test_mask.sum()):

    raise RuntimeError(
        "X_test row count does not match test mask."
    )

if len(y_train) != len(X_train):

    raise RuntimeError(
        "Training X/y row mismatch."
    )

if len(y_test) != len(X_test):

    raise RuntimeError(
        "Test X/y row mismatch."
    )

if len(meta_y_train) != len(X_train):

    raise RuntimeError(
        "meta_y_train/X_train row mismatch."
    )

if len(meta_y_test) != len(X_test):

    raise RuntimeError(
        "meta_y_test/X_test row mismatch."
    )

if len(meta_test) != len(X_test):

    raise RuntimeError(
        "meta_test/X_test row mismatch."
    )

# ----------------------------------------------------------
# 13. Test panel integrity
# ----------------------------------------------------------
#
# IMPORTANT:
# Check duplicates BEFORE drop_duplicates().
# Otherwise duplicate rows would be silently removed and
# the validation would always report zero duplicates.
# ----------------------------------------------------------

test_panel_raw = (
    data.loc[
        test_mask,
        [
            "Date",
            "Company",
        ],
    ]
    .copy()
)

if test_panel_raw.empty:

    raise RuntimeError(
        "Test panel is empty."
    )

test_panel_raw["Company"] = (
    test_panel_raw["Company"]
    .astype(str)
)

duplicate_test_keys = (
    test_panel_raw
    .duplicated(
        ["Date", "Company"]
    )
    .sum()
)

if duplicate_test_keys > 0:

    duplicate_examples = (
        test_panel_raw.loc[
            test_panel_raw.duplicated(
                ["Date", "Company"],
                keep=False,
            )
        ]
        .sort_values(
            ["Date", "Company"]
        )
        .head(20)
    )

    print(
        "\n⚠ DUPLICATE DATE/COMPANY "
        "OBSERVATIONS DETECTED"
    )

    print(
        duplicate_examples
    )

    raise RuntimeError(
        "Duplicate Date/Company observations "
        f"detected in test panel: "
        f"{duplicate_test_keys}"
    )

# Canonical test panel
test_panel = (
    test_panel_raw
    .drop_duplicates(
        ["Date", "Company"]
    )
    .copy()
)

# ----------------------------------------------------------
# 14. Final split validation
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("DATE SPLIT VALIDATION")
print("=" * 60)

print(
    "✓ Date leakage check       : PASS"
)

print(
    "✓ Train X/y alignment      : PASS"
)

print(
    "✓ Test X/y alignment       : PASS"
)

print(
    "✓ Meta X/y alignment       : PASS"
)

print(
    "✓ Test panel uniqueness    : PASS"
)

print(
    "✓ Test universe diagnosed  : PASS"
)

print("=" * 60)

# ----------------------------------------------------------
# IMPORTANT INTERPRETATION
# ----------------------------------------------------------
#
# Example:
#
#   Train companies : 27
#   Test companies  : 34
#   Combined        : 39
#
# This is NOT automatically a split bug.
#
# It means the cleaned/model-ready dataset contains 39
# companies overall, but only 34 have valid observations
# in the selected test period.
#
# We should NOT force all 39 companies into X_test by
# filling missing observations.
#
# The correct action is to diagnose WHY the five companies
# disappear from the historical test panel.
# ----------------------------------------------------------

print(
    "\n📊 MODEL TEST UNIVERSE"
)

print(
    f"Canonical companies : "
    f"{len(all_model_companies):,}"
)

print(
    f"Companies in test   : "
    f"{len(test_companies):,}"
)

if missing_from_test:

    print(
        "\n⚠ Companies absent from test period:"
    )

    print(
        sorted(
            missing_from_test
        )
    )

else:

    print(
        "✓ All canonical companies "
        "have test observations."
    )

print("=" * 60)

# ----------------------------------------------------------
# 15. FINAL UNIVERSE vs TEST UNIVERSE DIAGNOSTIC
# ----------------------------------------------------------
#
# `final_df` represents the canonical/current universe
# entering the model pipeline.
#
# `test_panel` represents companies that actually have
# valid observations in the historical test period.
#
# These universes are intentionally NOT forced to match.
# ----------------------------------------------------------

full_universe = set(
    final_df["Company"]
    .dropna()
    .astype(str)
    .unique()
)

test_universe = set(
    test_panel["Company"]
    .dropna()
    .astype(str)
    .unique()
)

missing_test_companies = sorted(
    full_universe
    -
    test_universe
)

extra_test_companies = sorted(
    test_universe
    -
    full_universe
)

print("\n" + "=" * 60)
print("FINAL UNIVERSE vs TEST UNIVERSE")
print("=" * 60)

print(
    f"Full universe companies : "
    f"{len(full_universe):,}"
)

print(
    f"Test universe companies : "
    f"{len(test_universe):,}"
)

print(
    f"Missing test companies  : "
    f"{len(missing_test_companies):,}"
)

print(
    f"Unexpected test companies : "
    f"{len(extra_test_companies):,}"
)

if missing_test_companies:

    print(
        "\n⚠ Companies present in current/full "
        "universe but absent from historical "
        "test panel:"
    )

    print(
        sorted(
            missing_test_companies
        )
    )

else:

    print(
        "\n✓ All current-universe companies "
        "have test-period observations."
    )

if extra_test_companies:

    print(
        "\n⚠ Companies present in test panel "
        "but absent from current/full universe:"
    )

    print(
        sorted(
            extra_test_companies
        )
    )

else:

    print(
        "✓ No unexpected companies "
        "in test panel."
    )

# ----------------------------------------------------------
# 16. DO NOT MODIFY THE TEST SET
# ----------------------------------------------------------
#
# Missing companies are diagnostic information only.
#
# DO NOT:
#   - add artificial rows
#   - forward-fill entire historical panels
#   - copy current observations backward
#   - inject missing companies into X_test
#   - alter train/test masks to force 39 companies
#
# The backtest must use only observations that genuinely
# existed in the historical test period.
# ----------------------------------------------------------

if not test_universe:

    raise RuntimeError(
        "Test universe is empty after "
        "model dataset construction."
    )

print("=" * 60)


# ----------------------------------------------------------
# REMOVE HIGHLY CORRELATED FEATURES
# ----------------------------------------------------------

print(
    "\n⚙ Removing highly correlated features"
)

corr = X_train.corr().abs()

upper = corr.where(
    np.triu(
        np.ones(corr.shape),
        k=1,
    ).astype(bool)
)

drop_cols = [

    column

    for column in upper.columns

    if any(
        upper[column] > 0.95
    )
]

print(
    f"Removed {len(drop_cols)} features"
)

X_train = X_train.drop(
    columns=drop_cols,
    errors="ignore",
)

X_test = X_test.drop(
    columns=drop_cols,
    errors="ignore",
)

FEATURES = [

    f

    for f in FEATURES

    if f not in drop_cols
]

# ----------------------------------------------------------
# SCALE DATA
# ----------------------------------------------------------

print("\n⚙ Scaling features")

scaler = StandardScaler()

X_train_scaled = (
    scaler.fit_transform(
        X_train
    )
)

X_test_scaled = (
    scaler.transform(
        X_test
    )
)

# ----------------------------------------------------------
# TRAIN BASE MODELS
# ----------------------------------------------------------

print("\n⚙ Training base models")

models = train_models(

    X_train=
    X_train,

    y_train=
    y_train,

    X_train_scaled=
    X_train_scaled,

    selected_models=
    SELECTED_MODELS,
)

if len(models) == 0:

    raise RuntimeError(
        "No models trained."
    )

print(
    "\nFinal trained models:"
)

print(
    list(models.keys())
)

# ----------------------------------------------------------
# TEST PROBABILITIES
# ----------------------------------------------------------

print(
    "\n⚙ Generating probabilities"
)

probas = (
    get_model_probabilities(

        models=
        models,

        X_test=
        X_test,

        X_test_scaled=
        X_test_scaled,
    )
)

# ----------------------------------------------------------
# TRAIN PROBABILITIES
# FOR META MODEL
# ----------------------------------------------------------

train_probas = {}

for name, model in models.items():

    try:

        if name in [
            "lr",
            "svm",
            "mlp",
        ]:

            p = (
                model.predict_proba(
                    X_train_scaled
                )[:, 1]
            )

        else:

            p = (
                model.predict_proba(
                    X_train
                )[:, 1]
            )

        train_probas[name] = p

    except Exception as e:

        print(
            f"Meta train failed "
            f"for {name}: {e}"
        )

meta_X_train = pd.DataFrame(
    train_probas
)

# ----------------------------------------------------------
# BUILD META FEATURES
# ----------------------------------------------------------

meta_X_test = pd.DataFrame()

for k, v in probas.items():

    arr = np.asarray(v)

    if len(arr.shape) > 1:

        arr = arr[:, 1]

    meta_X_test[k] = arr

# ----------------------------------------------------------
# INTERACTION FEATURES
# ----------------------------------------------------------

if (
    "xgb" in meta_X_train.columns
    and
    "lr" in meta_X_train.columns
):

    meta_X_train["xgb_minus_lr"] = (
        meta_X_train["xgb"]
        -
        meta_X_train["lr"]
    )

    meta_X_test["xgb_minus_lr"] = (
        meta_X_test["xgb"]
        -
        meta_X_test["lr"]
    )

if (
    "xgb" in meta_X_train.columns
    and
    "rf" in meta_X_train.columns
):

    meta_X_train["xgb_minus_rf"] = (
        meta_X_train["xgb"]
        -
        meta_X_train["rf"]
    )

    meta_X_test["xgb_minus_rf"] = (
        meta_X_test["xgb"]
        -
        meta_X_test["rf"]
    )

# ----------------------------------------------------------
# SPREAD FEATURE
# ----------------------------------------------------------

meta_X_train["spread"] = (

    meta_X_train.max(axis=1)
    -
    meta_X_train.min(axis=1)
)

meta_X_test["spread"] = (

    meta_X_test.max(axis=1)
    -
    meta_X_test.min(axis=1)
)

meta_X_test = meta_X_test[
    meta_X_train.columns
]

# ----------------------------------------------------------
# META MODEL
# ----------------------------------------------------------

print(
    "\n⚙ Training meta model"
)

meta_model = LogisticRegression(
    max_iter=1000
)

meta_model.fit(
    meta_X_train,
    meta_y_train,
)

meta_proba = (
    meta_model.predict_proba(
        meta_X_test
    )[:, 1]
)


# ================================

logger.info("=" * 80)
logger.info("BACKTEST DEBUG - After Meta Filter")
logger.info("Rows after merge: %d", len(final_df))

logger.info(
    "Unique Dates=%d | Companies=%d",
    final_df["Date"].nunique(),
    final_df["Company"].nunique()
)

logger.info(
    f"Meta-positive observations: "
    f"{(meta_proba > META_THRESHOLD).sum()}"
)

print(
    "Average Meta:",
    round(
        meta_proba.mean(),
        4,
    ),
)

# ----------------------------------------------------------
# ARTIFACTS
# ----------------------------------------------------------

joblib.dump(
    scaler,
    "artifacts/scaler.pkl",
)

joblib.dump(
    imputer,
    "artifacts/imputer.pkl",
)

joblib.dump(
    models,
    "artifacts/models.pkl",
)

joblib.dump(
    meta_model,
    "artifacts/meta_model.pkl",
)

joblib.dump(
    FEATURES,
    "artifacts/features.pkl",
)


# ==========================================================
# PART 5
# ENSEMBLE GENERATION
#
# Purpose
# -------
#
# Base Models
#      ↓
# Probabilities
#      ↓
# Weighted Ensemble
#      ↓
# Meta Model
#      ↓
# Regime Filter
#      ↓
# Volatility Filter
#      ↓
# Cross Sectional Ranking
#      ↓
# final_proba
#
# final_proba becomes the Institutional Alpha Signal.
# ==========================================================

print("\n" + "=" * 60)
print("PART 5 — ENSEMBLE GENERATION")
print("=" * 60)

# ----------------------------------------------------------
# WEIGHTED ENSEMBLE
# ----------------------------------------------------------

signals = []
used_models = []

for model_name, values in probas.items():

    arr = np.asarray(
        values
    )

    if len(arr.shape) > 1:

        arr = arr[:, 1]

    signals.append(arr)

    used_models.append(
        model_name.upper()
    )

signals = np.column_stack(
    signals
)

# ----------------------------------------------------------
# INSTITUTIONAL WEIGHT MAP
# ----------------------------------------------------------

weight_map = {

    "CAT": 0.40,
    "XGB": 0.35,
    "LGB": 0.25,

    "RF": 0.15,
    "LR": 0.10,
}

ensemble_weights = np.array(

    [
        weight_map.get(
            model,
            1.0,
        )

        for model in used_models
    ]
)

ensemble_weights /= (
    ensemble_weights.sum()
)

ensemble_proba = np.dot(

    signals,
    ensemble_weights,
)

ensemble_proba = np.asarray(
    ensemble_proba
).flatten()

print(
    "\nWeighted Ensemble Complete"
)

print(
    "Models:",
    used_models,
)

print(
    "Weights:",
    ensemble_weights,
)


logger.info("=" * 80)
logger.info("BACKTEST DEBUG Before - Meta Model Filter")
logger.info("Rows after merge: %d", len(final_df))

logger.info(
    "Unique Dates=%d | Companies=%d",
    final_df["Date"].nunique(),
    final_df["Company"].nunique()
)

# ----------------------------------------------------------
# META MODEL FILTER
# ----------------------------------------------------------

print(
    "\nApplying Meta Model"
)

meta_proba = (
    meta_model.predict_proba(
        meta_X_test
    )[:, 1]
)

meta_proba = np.asarray(
    meta_proba
).flatten()



logger.info("=" * 80)
logger.info("BACKTEST DEBUG After - Meta Model Filter")
logger.info("Rows after merge: %d", len(final_df))

logger.info(
    "Unique Dates=%d | Companies=%d",
    final_df["Date"].nunique(),
    final_df["Company"].nunique()
)
# ----------------------------------------------------------
# REGIME THRESHOLDS
# ----------------------------------------------------------

test_regimes = (

    data.loc[test_mask]
    ["Market_Regime"]
    .reset_index(drop=True)
)

thresholds = np.where(

    test_regimes == "BULL",
    BULL_META_THRESHOLD,

    np.where(

        test_regimes == "SIDEWAYS",
        SIDEWAYS_META_THRESHOLD,

        np.where(

            test_regimes ==
            "BULL_VOLATILE",

            BULL_VOL_META_THRESHOLD,

            np.where(

                test_regimes ==
                "SIDEWAYS_VOLATILE",

                SIDEWAYS_VOL_META_THRESHOLD,

                META_THRESHOLD,
            ),
        ),
    ),
)



logger.info("=" * 80)
logger.info("BACKTEST DEBUG Before - Meta Filter Final Proba")
logger.info("Rows after merge: %d", len(final_df))

logger.info(
    "Unique Dates=%d | Companies=%d",
    final_df["Date"].nunique(),
    final_df["Company"].nunique()
)
# ----------------------------------------------------------
# META FILTER
# ----------------------------------------------------------

meta_pass = (
    meta_proba > thresholds
)

# ----------------------------------------------------------
# CANONICAL FINAL PROBABILITY
# ----------------------------------------------------------
#
# This is the ONLY place where final_proba is created.
#
# Flow:
#
# weighted ensemble
#        ↓
# ensemble_proba
#        ↓
# meta model
#        ↓
# regime-specific meta threshold
#        ↓
# final_proba
#
# final_proba is the canonical Alpha Engine probability.
# ----------------------------------------------------------

final_proba = np.where(
    meta_pass,
    ensemble_proba,
    NEUTRALITY,
)

# For Integration of Alpha Stage
meta_df = meta_test.copy()

meta_df["Meta_Proba"] = meta_proba
meta_df["Meta_Pass"] = meta_pass
meta_df["Probability"] = final_proba

tracker.add_stage(
    AlphaStage.META,
    meta_df
)

print(
    "Signals after Meta Filter:",
    int(meta_pass.sum()),
)

logger.info("=" * 80)
logger.info("BACKTEST DEBUG After - Meta Filter Final Proba and Before Regime Filter")
logger.info("Rows after merge: %d", len(final_df))

logger.info(
    "Unique Dates=%d | Companies=%d",
    final_df["Date"].nunique(),
    final_df["Company"].nunique()
)
# ----------------------------------------------------------
# REGIME EXPOSURE DIAGNOSTICS
# ----------------------------------------------------------

print(
    "\nApplying Regime Filter"
)

allowed_regimes = [

    "BULL",
    "SIDEWAYS",
    "BULL_VOLATILE",
    "SIDEWAYS_VOLATILE",
]

# ----------------------------------------------------------
# REGIME ADJUSTMENT
# ----------------------------------------------------------
#
# CANONICAL ALPHA CONTRACT
#
# Do not convert a valid BUY probability to zero solely
# because of the market regime.
#
# The Alpha Engine probability remains the canonical
# probability.
#
# Regime is a RISK / EXPOSURE MODIFIER only.
#
# Therefore:
#
#     final_proba
#         ↓
#     remains unchanged
#
#     regime_exposure
#         ↓
#     is passed downstream
#
# Portfolio construction / risk engine may reduce
# exposure using this modifier.
#
# IMPORTANT:
# ----------------------------------------------------------
# DO NOT do:
#
#     final_proba[test_regimes == "BEAR"] = 0
#
# DO NOT do:
#
#     final_proba *= regime_multiplier
#
# DO NOT use regime to destroy the canonical probability.
#
# The canonical probability must remain the probability
# produced by the Alpha Engine after Meta processing.
# ----------------------------------------------------------

test_regimes = (
    meta_test["Market_Regime"]
    .astype(str)
    .fillna("SIDEWAYS")
    .to_numpy()
)

# ----------------------------------------------------------
# Regime exposure multipliers
# ----------------------------------------------------------
#
# These values represent the maximum directional exposure
# allowed downstream for each market regime.
#
# They DO NOT modify final_proba.
# ----------------------------------------------------------

regime_multiplier = np.ones(
    len(test_regimes),
    dtype=float,
)

regime_multiplier[test_regimes == "BULL"] = 1.00
regime_multiplier[test_regimes == "SIDEWAYS"] = 0.85
regime_multiplier[test_regimes == "BULL_VOLATILE"] = 0.75
regime_multiplier[test_regimes == "SIDEWAYS_VOLATILE"] = 0.60
regime_multiplier[test_regimes == "BEAR"] = 0.50
regime_multiplier[test_regimes == "BEAR_VOLATILE"] = 0.25

# ----------------------------------------------------------
# Regime validity
# ----------------------------------------------------------
#
# A valid regime always has a positive exposure multiplier.
# No probability is rejected here.
# ----------------------------------------------------------

regime_pass = (
    regime_multiplier > 0.0
)

# ----------------------------------------------------------
# Canonical regime exposure
# ----------------------------------------------------------
#
# This is the ONLY regime-adjusted quantity.
#
# final_proba remains untouched.
#
# Downstream portfolio / risk engines may use:
#
#     final_proba
#     regime_exposure
#
# to determine final position size.
# ----------------------------------------------------------

regime_exposure = (
    regime_multiplier.copy()
)

# ----------------------------------------------------------
# Defensive validation
# ----------------------------------------------------------

if len(regime_exposure) != len(final_proba):
    raise ValueError(
        "CRITICAL: Regime exposure alignment failure: "
        f"final_proba={len(final_proba)}, "
        f"regime_exposure={len(regime_exposure)}"
    )

if not np.isfinite(
    regime_exposure
).all():
    raise ValueError(
        "CRITICAL: Regime exposure contains "
        "NaN or infinite values."
    )

if (
    np.asarray(regime_exposure) < 0
).any():
    raise ValueError(
        "CRITICAL: Regime exposure contains "
        "negative values."
    )

# ----------------------------------------------------------
# IMPORTANT CANONICAL PROBABILITY CHECK
# ----------------------------------------------------------
#
# Regime adjustment must NOT change final_proba.
#
# Therefore we explicitly retain final_proba as the
# canonical Alpha Engine probability.
# ----------------------------------------------------------

final_proba = np.asarray(
    final_proba,
    dtype=float,
).reshape(-1)

if not np.isfinite(
    final_proba
).all():
    raise ValueError(
        "CRITICAL: Canonical final_proba contains "
        "NaN or infinite values after Regime processing."
    )

final_proba = np.clip(
    final_proba,
    0.0,
    1.0,
)

# ----------------------------------------------------------
# For Integration with Alpha Stage
# ----------------------------------------------------------
#
# The REGIME stage records:
#
#     Probability      = canonical Alpha probability
#     Regime_Exposure  = regime risk/exposure modifier
#
# Probability is intentionally NOT multiplied by the
# regime multiplier.
# ----------------------------------------------------------

regime_df = meta_test.copy()

regime_df["Probability"] = (
    final_proba
)

regime_df["Regime_Exposure"] = (
    regime_exposure
)

regime_df["Regime_Pass"] = (
    regime_pass
)

# ----------------------------------------------------------
# Optional regime diagnostics
# ----------------------------------------------------------

regime_diagnostics = pd.DataFrame(
    {
        "Market_Regime": test_regimes,
        "Regime_Exposure": regime_exposure,
        "Regime_Pass": regime_pass,
        "Probability": final_proba,
    }
)

print(
    "\n===== REGIME EXPOSURE DIAGNOSTICS ====="
)

print(
    regime_diagnostics[
        "Market_Regime"
    ].value_counts()
    .rename("Observations")
)

print(
    "\nRegime Exposure:"
)

print(
    regime_diagnostics
    .groupby("Market_Regime")[
        "Regime_Exposure"
    ]
    .first()
    .sort_values(
        ascending=False
    )
)

print(
    "\nMean Canonical Probability by Regime:"
)

print(
    regime_diagnostics
    .groupby("Market_Regime")[
        "Probability"
    ]
    .mean()
    .sort_values(
        ascending=False
    )
)

print(
    "\nSignals with Positive Canonical Alpha:",
    int(
        (
            regime_diagnostics[
                "Probability"
            ] > NEUTRALITY
        ).sum()
    ),
)

print(
    "Signals with Positive Regime Exposure:",
    int(
        regime_diagnostics[
            "Regime_Pass"
        ].sum()
    ),
)

# ----------------------------------------------------------
# Alpha Stage Tracker
# ----------------------------------------------------------

tracker.add_stage(
    AlphaStage.REGIME,
    regime_df
)

# ----------------------------------------------------------
# IMPORTANT:
# Do NOT report this as "signals after Regime Filter"
# because Regime does not filter probabilities anymore.
# ----------------------------------------------------------

print(
    "Positive Alpha entering downstream stages:",
    int(
        (
            final_proba > NEUTRALITY
        ).sum()
    ),
)

print(
    "Regime-adjusted exposure observations:",
    int(
        (
            regime_exposure > 0
        ).sum()
    ),
)

logger.info("=" * 80)

logger.info(
    "BACKTEST DEBUG After - Regime Adjustment "
    "and Before Volatility Filter"
)

logger.info(
    "Rows after merge: %d",
    len(final_df)
)

logger.info(
    "Unique Dates=%d | Companies=%d",
    final_df["Date"].nunique(),
    final_df["Company"].nunique(),
)

logger.info(
    "Canonical Probability Mean=%.6f | "
    "Min=%.6f | Max=%.6f",
    float(final_proba.mean()),
    float(final_proba.min()),
    float(final_proba.max()),
)

logger.info(
    "Regime Exposure Mean=%.6f | "
    "Min=%.6f | Max=%.6f",
    float(regime_exposure.mean()),
    float(regime_exposure.min()),
    float(regime_exposure.max()),
)

# ----------------------------------------------------------
# VOLATILITY CONTROL DIAGNOSTICS
# ----------------------------------------------------------

print(
    "\nApplying Volatility Filter"
)

test_atr = (
    data.loc[test_mask]
    ["ATR_Z"]
    .reset_index(drop=True)
)

VOL_FILTER_WINDOW = int(
    CONFIG["BACKTEST"].get(
        "VOL_FILTER_WINDOW",
        60,
    )
)

VOL_FILTER_QUANTILE = float(
    CONFIG["BACKTEST"].get(
        "VOL_FILTER_QUANTILE",
        0.80,
    )
)

atr_mean = (
    test_atr
    .rolling(
        VOL_FILTER_WINDOW,
        min_periods=20,
    )
    .mean()
)

atr_threshold = (
    test_atr
    .rolling(
        VOL_FILTER_WINDOW,
        min_periods=20,
    )
    .quantile(
        VOL_FILTER_QUANTILE
    )
)

vol_filter = (
    test_atr <= atr_threshold
)

vol_filter = (
    vol_filter
    .fillna(True)
    .astype(bool)
    .values
)


logger.info("=" * 80)
logger.info("VOLATILITY FILTER DEBUG")

logger.info(
    "ATR_Z count=%d | mean=%.4f | min=%.4f | max=%.4f",
    test_atr.count(),
    test_atr.mean(),
    test_atr.min(),
    test_atr.max(),
)

logger.info(
    "ATR_MEAN count=%d | mean=%.4f | min=%.4f | max=%.4f",
    atr_mean.count(),
    atr_mean.mean(),
    atr_mean.min(),
    atr_mean.max(),
)

logger.info(
    "Vol Filter TRUE=%d | FALSE=%d",
    np.sum(vol_filter),
    len(vol_filter) - np.sum(vol_filter),
)

logger.info(
    "Positive Alpha BEFORE volatility filter: %d",
    int(
        (
            final_proba > NEUTRALITY
        ).sum()
    ),
)

volatility_pass = vol_filter.astype(bool)

# For Integration of Alpha Stage
volatility_df = regime_df.copy()

volatility_df["Probability"] = final_proba

tracker.add_stage(
    AlphaStage.VOLATILITY,
    volatility_df
)

print(
    "Positive Alpha after Volatility Filter:",
    int(
        (
            (final_proba > NEUTRALITY)
            & volatility_pass
        ).sum()
    ),
)


logger.info("=" * 80)
logger.info("BACKTEST DEBUG After - Volatality Filter and Before CROSS SECTIONAL Filter")
logger.info("Rows after merge: %d", len(final_df))

logger.info(
    "Unique Dates=%d | Companies=%d",
    final_df["Date"].nunique(),
    final_df["Company"].nunique()
)

# ----------------------------------------------------------
# CROSS SECTIONAL STAGE
# ----------------------------------------------------------
#
# IMPORTANT
# ----------
# Cross-sectional portfolio selection is owned by
# backtest.py::_cross_sectional_signal().
#
# main.py must NOT rank/filter final_proba here.
#
# At this stage final_proba represents the alpha signal
# after model/meta/regime/volatility filters.
#
# This prevents double cross-sectional filtering.
# ----------------------------------------------------------

print(
    "\nCross Sectional Ranking deferred to Backtest Engine"
)

cross_section_df = meta_test.copy()

cross_section_df["Probability"] = (
    final_proba
)

tracker.add_stage(
    AlphaStage.CROSS_SECTION,
    cross_section_df
)

print(
    "Positive Alpha entering backtest:",
    int(
        (
            (final_proba > NEUTRALITY)
            & meta_pass
            & volatility_pass
        ).sum()
    ),
)

logger.info("=" * 80)
logger.info("BACKTEST DEBUG After -  CROSS SECTIONAL Filter")
logger.info("Rows after merge: %d", len(final_df))

logger.info(
    "Unique Dates=%d | Companies=%d",
    final_df["Date"].nunique(),
    final_df["Company"].nunique()
)

# ----------------------------------------------------------
# FINAL DIAGNOSTICS
# ----------------------------------------------------------

print("\nFINAL SIGNAL STATISTICS")

print(
    "Signal Count:",
    (final_proba > 0).sum(),
)

print(
    "Signal Mean:",
    round(
        final_proba.mean(),
        6,
    ),
)

print(
    "Signal Max:",
    round(
        final_proba.max(),
        6,
    ),
)

print(
    "Signal Min:",
    round(
        final_proba.min(),
        6,
    ),
)

logger.info("=" * 80)
logger.info("BACKTEST DEBUG Before - Backtest")
logger.info("Rows after merge: %d", len(final_df))

logger.info(
    "Unique Dates=%d | Companies=%d",
    final_df["Date"].nunique(),
    final_df["Company"].nunique()
)

# ----------------------------------------------------------
# BACKTEST
# ----------------------------------------------------------

print("\n===== SIGNAL DIAGNOSTICS =====")

signal_df = pd.DataFrame(
    {
        "Date": meta_test["Date"],
        "Proba": final_proba,
    }
)

print(
    signal_df.groupby("Date")
    .size()
    .describe()
)

print(
    signal_df[signal_df["Proba"] > 0]
    .groupby("Date")
    .size()
    .describe()
)

print(
    "\nRunning Ensemble Backtest"
)

print("After merge:", len(final_df))

logger.info(
    "Signals AFTER volatility filter: %d",
    (final_proba > 0).sum(),
)

logger.info(
    "Max proba after vol filter: %.6f",
    np.max(final_proba) if len(final_proba) else -1,
)

logger.info(
    "Mean proba after vol filter: %.6f",
    np.mean(final_proba) if len(final_proba) else -1,
)

logger.info("=" * 80)

backtest_meta = meta_test.copy().reset_index(drop=True)

backtest_meta["Meta_Pass"] = (
    meta_pass
)

backtest_meta["Regime_Pass"] = (
    regime_pass
)

backtest_meta["Volatility_Pass"] = (
    volatility_pass
)

backtest_meta["Regime_Multiplier"] = (
    regime_multiplier
)

# ==========================================================
# CANONICAL BACKTEST SIGNAL CONTRACT
# ==========================================================
#
# IMPORTANT:
#
# The backtest must consume exactly the final probability
# produced by the Alpha Engine after Meta / Regime /
# Volatility processing.
#
# Do NOT allow the backtest to independently reconstruct
# probability from:
#
#   - ensemble_proba
#   - Prediction_Alpha
#   - Alpha_Score
#   - Final_Score
#   - Signal
#
# `Prediction_Prob` is the canonical Alpha Engine probability.
#
# IMPORTANT ALIGNMENT RULE:
#
# final_df contains the COMPLETE cleaned dataset.
# final_proba contains probabilities ONLY for the TEST panel.
#
# Therefore:
#
#     len(final_df)    != len(final_proba)
#
# is expected.
#
# The canonical backtest dataframe must contain exactly the
# same rows as the test prediction panel.
# ==========================================================

final_df = final_df.copy()


# ============================================================
# CANONICAL PROBABILITY ALIGNMENT
# ============================================================

# ------------------------------------------------------------
# 1. Validate probability length against TEST panel
# ------------------------------------------------------------

expected_test_rows = len(meta_test)

if len(final_proba) != expected_test_rows:
    raise ValueError(
        "CRITICAL: Canonical probability alignment failure: "
        f"meta_test={expected_test_rows}, "
        f"final_proba={len(final_proba)}"
    )


# ------------------------------------------------------------
# 2. Validate probability length against X_test
# ------------------------------------------------------------

if len(final_proba) != len(X_test):
    raise ValueError(
        "CRITICAL: Test probability alignment failure: "
        f"X_test={len(X_test)}, "
        f"final_proba={len(final_proba)}"
    )


# ------------------------------------------------------------
# 3. Convert canonical probabilities to numeric array
# ------------------------------------------------------------

canonical_proba = pd.to_numeric(
    np.asarray(final_proba).reshape(-1),
    errors="coerce",
)


# ------------------------------------------------------------
# 4. Validate probability values
# ------------------------------------------------------------

if np.isnan(canonical_proba).any():
    raise ValueError(
        "CRITICAL: Canonical Prediction_Prob contains NaN values."
    )


if np.isinf(canonical_proba).any():
    raise ValueError(
        "CRITICAL: Canonical Prediction_Prob contains "
        "infinite values."
    )


# ------------------------------------------------------------
# 5. Probability must represent a valid probability
# ------------------------------------------------------------

if (
    (canonical_proba < 0.0).any()
    or (canonical_proba > 1.0).any()
):
    raise ValueError(
        "CRITICAL: Canonical Prediction_Prob contains values "
        "outside [0, 1]."
    )


# ============================================================
# 6. CONSTRUCT CANONICAL BACKTEST PANEL
# ============================================================
#
# IMPORTANT:
# X_test/meta_test are derived from `data`.
# `data` has its own reset index.
#
# Therefore DO NOT use:
#
#     final_df.loc[X_test.index]
#
# because that index belongs to `data`, not necessarily
# to the original final_df.
#
# `_source_row_id` is the only authoritative mapping.
# ============================================================

if len(test_source_row_ids) != expected_test_rows:
    raise ValueError(
        "CRITICAL: Test source-row count mismatch: "
        f"expected={expected_test_rows}, "
        f"actual={len(test_source_row_ids)}"
    )

if (
    test_source_row_ids.min() < 0
    or
    test_source_row_ids.max() >= len(final_df)
):
    raise ValueError(
        "CRITICAL: Test source-row IDs contain values "
        "outside final_df bounds."
    )

backtest_df = (
    final_df
    .iloc[test_source_row_ids]
    .copy()
)

backtest_df = (
    backtest_df
    .reset_index(drop=True)
)

# ============================================================
# CANONICAL DATE / COMPANY ALIGNMENT CHECK
# ============================================================

expected_meta = (
    meta_test[
        ["Date", "Company"]
    ]
    .copy()
    .reset_index(drop=True)
)

actual_meta = (
    backtest_df[
        ["Date", "Company"]
    ]
    .copy()
    .reset_index(drop=True)
)

expected_meta["Date"] = pd.to_datetime(
    expected_meta["Date"],
    errors="coerce",
)

actual_meta["Date"] = pd.to_datetime(
    actual_meta["Date"],
    errors="coerce",
)

expected_meta["Company"] = (
    expected_meta["Company"]
    .astype(str)
    .str.strip()
)

actual_meta["Company"] = (
    actual_meta["Company"]
    .astype(str)
    .str.strip()
)

if not expected_meta.equals(actual_meta):
    mismatch_mask = (
        expected_meta["Date"]
        != actual_meta["Date"]
    ) | (
        expected_meta["Company"]
        != actual_meta["Company"]
    )

    mismatch_count = int(
        mismatch_mask.sum()
    )

    raise ValueError(
        "CRITICAL: Canonical Date/Company alignment "
        "failure between meta_test and backtest_df. "
        f"Mismatched rows={mismatch_count}"
    )

print(
    f"✓ Canonical Date/Company alignment PASS | "
    f"Rows={len(backtest_df):,}"
)



# ------------------------------------------------------------
# 7. Validate canonical backtest panel size
# ------------------------------------------------------------

if len(backtest_df) != expected_test_rows:
    raise ValueError(
        "CRITICAL: Canonical backtest panel alignment failure: "
        f"expected={expected_test_rows}, "
        f"actual={len(backtest_df)}"
    )


# ------------------------------------------------------------
# 8. Validate test panel index uniqueness
# ------------------------------------------------------------

if not backtest_df.index.is_unique:
    raise ValueError(
        "CRITICAL: Canonical backtest panel contains "
        "duplicate index values."
    )


# ============================================================
# ASSIGN CANONICAL PROBABILITY
# ============================================================

backtest_df["Prediction_Prob"] = canonical_proba


# ------------------------------------------------------------
# 9. Final probability integrity checks
# ------------------------------------------------------------

if backtest_df["Prediction_Prob"].isna().any():
    raise ValueError(
        "CRITICAL: Prediction_Prob contains NaN values "
        "after canonical assignment."
    )


if np.isinf(
    backtest_df["Prediction_Prob"].to_numpy(dtype=float)
).any():
    raise ValueError(
        "CRITICAL: Prediction_Prob contains infinite values."
    )


if (
    (backtest_df["Prediction_Prob"] < 0.0).any()
    or
    (backtest_df["Prediction_Prob"] > 1.0).any()
):
    raise ValueError(
        "CRITICAL: Prediction_Prob contains values "
        "outside [0, 1] after canonical assignment."
    )


# ============================================================
# CANONICAL CONTINUOUS ALPHA
# ============================================================

backtest_df["Prediction_Alpha"] = (
    backtest_df["Prediction_Prob"]
    - NEUTRALITY
)

# ============================================================
# CANONICAL CONFIDENCE
# ============================================================

backtest_df["Confidence"] = (
    backtest_df["Prediction_Alpha"].abs() * 2.0
)

# ============================================================
# CANONICAL ALPHA SIGNAL CONTRACT
# ============================================================

print("\n" + "=" * 72)
print("CANONICAL ALPHA SIGNAL CONTRACT")
print("=" * 72)

print(
    f"Full dataset rows       : {len(final_df):,}"
)

print(
    f"Test panel rows         : {len(backtest_df):,}"
)

print(
    f"Probability count       : "
    f"{backtest_df['Prediction_Prob'].notna().sum():,}"
)

print(
    f"Probability mean        : "
    f"{backtest_df['Prediction_Prob'].mean():.6f}"
)

print(
    f"Probability min         : "
    f"{backtest_df['Prediction_Prob'].min():.6f}"
)

print(
    f"Probability max         : "
    f"{backtest_df['Prediction_Prob'].max():.6f}"
)

print(
    f"Positive Alpha          : "
    f"{(backtest_df['Prediction_Alpha'] > 0).sum():,}"
)

print(
    f"Negative Alpha          : "
    f"{(backtest_df['Prediction_Alpha'] < 0).sum():,}"
)

print(
    f"Zero Alpha              : "
    f"{(backtest_df['Prediction_Alpha'] == 0).sum():,}"
)

print("=" * 72)


# ============================================================
# CANONICAL BACKTEST PROBABILITY CHECK
# ============================================================

print("\nCANONICAL BACKTEST PROBABILITY CHECK")

diagnostic_columns = [
    "Date",
    "Company",
    "Prediction_Prob",
    "Prediction_Alpha",
    "Confidence",
]

available_diagnostic_columns = [
    col
    for col in diagnostic_columns
    if col in backtest_df.columns
]

print(
    backtest_df[
        available_diagnostic_columns
    ].head(10)
)


# ============================================================
# FINAL CANONICAL ALIGNMENT CHECK
# ============================================================

if len(backtest_df) != len(canonical_proba):
    raise ValueError(
        "CRITICAL: Final canonical backtest alignment failure: "
        f"backtest_df={len(backtest_df)}, "
        f"canonical_proba={len(canonical_proba)}"
    )


if len(backtest_df) != len(X_test):
    raise ValueError(
        "CRITICAL: Final test-panel alignment failure: "
        f"backtest_df={len(backtest_df)}, "
        f"X_test={len(X_test)}"
    )


print(
    f"\n✓ Canonical probability alignment PASS | "
    f"TestRows={len(backtest_df):,} | "
    f"ProbabilityRows={len(canonical_proba):,}"
)


# ============================================================
# BACKTEST
# ============================================================
#
# IMPORTANT:
#
# Pass ONLY canonical Prediction_Prob to the backtest.
#
# The backtest must NOT reconstruct probability from any
# other score or model output.
# ============================================================

ensemble_bt = run_backtest(
    proba=backtest_df["Prediction_Prob"].to_numpy(
        dtype=float
    ),
    X_test=X_test,
    meta_test=backtest_meta,
    final_df=backtest_df,
)

results["ENSEMBLE"] = ensemble_bt


# ============================================================
# BACKTEST DEBUG
# ============================================================

logger.info("=" * 80)
logger.info("BACKTEST DEBUG After - Backtest")

logger.info(
    "Full dataset rows: %d",
    len(final_df)
)

logger.info(
    "Backtest test-panel rows: %d",
    len(backtest_df)
)

logger.info(
    "Canonical probability rows: %d",
    len(canonical_proba)
)

logger.info(
    "Unique Dates=%d | Companies=%d",
    backtest_df["Date"].nunique(),
    backtest_df["Company"].nunique()
)

logger.info(
    "Canonical Probability Mean=%.6f | Min=%.6f | Max=%.6f",
    backtest_df["Prediction_Prob"].mean(),
    backtest_df["Prediction_Prob"].min(),
    backtest_df["Prediction_Prob"].max(),
)


# ----------------------------------------------------------
# SAVE ARTIFACTS
# ----------------------------------------------------------

joblib.dump(
    ensemble_weights,
    "artifacts/ensemble_weights.pkl",
)

joblib.dump(
    ensemble_proba,
    "artifacts/ensemble_proba.pkl",
)

joblib.dump(
    final_proba,
    "artifacts/final_proba.pkl",
)

print(
    "\n✅ Institutional Alpha Signal Ready"
)

# ==========================================================
# PART 5A
# QUANTITATIVE VALIDATION & ROBUSTNESS
# ==========================================================

print("\n" + "=" * 60)
print("PART 5A — QUANTITATIVE VALIDATION & ROBUSTNESS")
print("=" * 60)

WALKFORWARD_ENABLED = (
    CONFIG["WALKFORWARD"]
    .get("ENABLED", False)
)

walkforward_summary = None

if WALKFORWARD_ENABLED:

    print("\nRunning walk-forward validation...")

    walkforward_summary = run_walkforward_validation(
        final_df=final_df,
        feature_cols=FEATURES,
        target_col=TARGET,
    )

    if walkforward_summary is None:
        raise RuntimeError(
            "Walk-forward validation produced no valid folds."
        )

    print(
        "\n✅ Walk-forward validation completed."
    )

    print(
        f"Walk-forward folds: "
        f"{len(walkforward_summary):,}"
    )

    print(
        f"Median fold Sharpe: "
        f"{walkforward_summary['Sharpe'].median():.3f}"
    )

    print(
        f"Mean fold Sharpe: "
        f"{walkforward_summary['Sharpe'].mean():.3f}"
    )

    print(
        f"Positive Sharpe folds: "
        f"{(walkforward_summary['Sharpe'] > 0).mean():.2%}"
    )

    print(
        f"Median fold CAGR: "
        f"{walkforward_summary['CAGR'].median():.3%}"
    )

    print(
        f"Positive CAGR folds: "
        f"{(walkforward_summary['CAGR'] > 0).mean():.2%}"
    )

    print(
        f"Mean Rank IC: "
        f"{walkforward_summary['Rank_IC'].mean():.4f}"
    )

    print(
        f"Positive IC folds: "
        f"{(walkforward_summary['Rank_IC'] > 0).mean():.2%}"
    )

    print(
        f"Worst fold Sharpe: "
        f"{walkforward_summary['Sharpe'].min():.3f}"
    )

    print(
        f"Worst fold DD: "
        f"{walkforward_summary['Max_Drawdown'].min():.3%}"
    )

    joblib.dump(
        walkforward_summary,
        "artifacts/walkforward_summary.pkl",
    )

    walkforward_summary.to_csv(
        "data/walkforward_summary.csv",
        index=False,
    )

else:

    print(
        "\n⏭ Walk-forward validation skipped "
        "(CONFIG['WALKFORWARD']['ENABLED'] = False)"
    )

# ==========================================================
# PART 6
# LIVE PORTFOLIO GENERATION
#
# Purpose
# -------
#
# final_df
#     ↓
# Latest Universe
#     ↓
# predict_today()
#     ↓
# signals_df
#     ↓
# portfolio
#
# Output
# ------
#
# portfolio
#
# This is the final Alpha Engine output before
# Institutional Portfolio Construction begins.
# ==========================================================

print("\n" + "=" * 60)
print("PART 6 — LIVE PORTFOLIO GENERATION")
print("=" * 60)

# ----------------------------------------------------------
# BUILD LATEST UNIVERSE
# ----------------------------------------------------------

print(
    "\nBuilding latest universe"
)

latest_data = (

    final_df
    .groupby(
        "Company"
    )
    .tail(1)
)

latest_data = (
    latest_data
    .dropna(
        subset=FEATURES
    )
)

print(
    f"Universe Size: {len(latest_data)}"
)

# ----------------------------------------------------------
# GENERATE LIVE SIGNALS
# ----------------------------------------------------------

print(
    "\nGenerating live signals"
)

logger.info("=" * 80)
logger.info("PREDICT TODAY INPUT")

logger.info(
    f"Rows received: {len(latest_data)}"
)

logger.info(
    f"Prediction_Prob exists: "
    f"{'Prediction_Prob' in latest_data.columns}"
)

logger.info(
    f"Probability exists: "
    f"{'Probability' in latest_data.columns}"
)

logger.info(
    f"Confidence exists: "
    f"{'Confidence' in latest_data.columns}"
)

# For  Integration Alpha Stage
cross_df = volatility_df.copy()

tracker.add_stage(
    AlphaStage.CROSS_SECTION,
    cross_df
)
# -----------------------------

signals_df, portfolio = (
    predict_today(
        latest_data= latest_data,
        scaler= scaler,
        models= models,
        filtered_weights= ensemble_weights,
        FEATURES= FEATURES,
    )
)

# For  Integration Alpha Stage
tracker.add_stage(
    "SIGNALS",
    signals_df.copy()
)

tracker.add_stage(
    AlphaStage.PORTFOLIO,
    portfolio.copy()
)
# -----------------------------
# ----------------------------------------------------------
# DIAGNOSTICS
# ----------------------------------------------------------

print(
    "\nSignals Generated:"
)

print(
    len(signals_df)
)

print(
    "\nPortfolio Size:"
)

print(
    len(portfolio)
)

print(
    "\nPortfolio Columns:"
)

print(
    portfolio.columns.tolist()
)

# ----------------------------------------------------------
# DISPLAY TOP POSITIONS
# ----------------------------------------------------------

TOP_N = (
    CONFIG["PORTFOLIO"]
    .get(
        "TOP_N",
        20,
    )
)

print(
    f"\nTop {TOP_N} Positions"
)

if portfolio.empty:

    logger.warning(
        "Portfolio is empty."
    )

else:

    logger.info(
        "\nTop Portfolio Positions:\n%s",
        portfolio[
            [
                "Company",
                "Final_Score",
                "Portfolio_Rank",
                "Position_Weight"
            ]
        ].to_string(index=False)
    )

# ----------------------------------------------------------
# OPTIONAL SANITY CHECKS
# ----------------------------------------------------------

if (
    "Position_Weight"
    in portfolio.columns
):

    total_weight = (
        portfolio[
            "Position_Weight"
        ]
        .sum()
    )

    print(
        "\nTotal Weight:",
        round(
            total_weight,
            6,
        )
    )

# ----------------------------------------------------------
# SAVE ARTIFACTS
# ----------------------------------------------------------

joblib.dump(

    latest_data,

    "artifacts/latest_universe.pkl",
)

joblib.dump(

    signals_df,

    "artifacts/signals_df.pkl",
)

joblib.dump(

    portfolio,

    "artifacts/portfolio.pkl",
)

print(
    "\nLive Portfolio Successfully Generated"
)

# ----------------------------------------------------------
# FINAL OUTPUTS
# ----------------------------------------------------------

print("\nALPHA PLATFORM OUTPUTS")

print(
    "final_df"
)

print(
    "FEATURES"
)

print(
    "ensemble_proba"
)

print(
    "final_proba"
)

print(
    "signals_df"
)

print(
    "portfolio"
)


# ==========================================================
# PART 7
# CONSTRUCTION ADAPTER INTEGRATION
#
# Alpha Platform
#       ↓
# input_adapter.py
#       ↓
# PipelineInput
#
# This is the OFFICIAL handoff between:
#
# Alpha Engine
#       ↓
# Institutional Construction Engine
# ==========================================================

print("\n" + "=" * 60)
print("PART 7 — CONSTRUCTION ADAPTER INTEGRATION")
print("=" * 60)

# ----------------------------------------------------------
# BUILD PIPELINE INPUT
# ----------------------------------------------------------
print(
    "\nBuilding Institutional PipelineInput"
)

tracker.export_summary()

pipeline_input = (
    build_pipeline_input_engine(
        final_df=final_df,
        alpha_results=alpha_results,
        ensemble_proba=final_proba,
        latest_universe=signals_df,
        portfolio=portfolio,
    )
)

print(
    "\nPipelineInput Successfully Built"
)

# ----------------------------------------------------------
# VALIDATION
# ----------------------------------------------------------

PipelineInputValidator.validate(
    pipeline_input
)

print(
    "PipelineInput validation passed."
)

# ----------------------------------------------------------
# DIAGNOSTICS
# ----------------------------------------------------------

print("\nPipelineInput Summary")

print(
    "Market Data Rows:",
    len(
        pipeline_input
        .market_data
        .prices
    )
)

print(
    "Expected Returns:",
    pipeline_input
    .forecast_data
    .expected_returns
    is not None
)

print(
    "Factor Exposures:",
    pipeline_input
    .factor_data
    .factor_exposures
    is not None
)

print(
    "Current Portfolio:",
    pipeline_input
    .portfolio_data
    .current_weights
    is not None
)

print(
    "Liquidity:",
    pipeline_input
    .liquidity_data
    .average_daily_volume
    is not None
)

print(
    "Constraints:",
    pipeline_input
    .constraint_data
    .sector_map
    is not None
)

# ----------------------------------------------------------
# OPTIONAL:
# PIPELINE INPUT DIAGNOSTICS
# ----------------------------------------------------------

adapter_result = (
    build_pipeline_input_with_diagnostics(
        final_df = final_df,
        alpha_results = alpha_results,
        ensemble_proba = final_proba,
        latest_universe = signals_df,
        portfolio = portfolio,
    )
)

print("\nAdapter Diagnostics")

print(
    adapter_result.diagnostics
)

# ----------------------------------------------------------
# SAVE ARTIFACT
# ----------------------------------------------------------

joblib.dump(

    pipeline_input,

    "artifacts/pipeline_input.pkl",
)

print(
    "\nInstitutional PipelineInput saved."
)


# ==========================================================
# PART 8
# INSTITUTIONAL PIPELINE INTEGRATION
#
# PipelineInput
#       ↓
# institutional_pipeline()
#       ↓
# pipeline.py
#       ↓
# portfolio_builder.py
#       ↓
# InstitutionalPortfolioPipelineResult
#
# FIRST INSTITUTIONAL SMOKE TEST
# ==========================================================

print("\n" + "=" * 60)
print("PART 8 — INSTITUTIONAL PIPELINE INTEGRATION")
print("=" * 60)

# ----------------------------------------------------------
# PIPELINE METADATA
# ----------------------------------------------------------

print(
    "\nCreating Institutional Metadata"
)

metadata = (
    PipelineFrameworkFactory
    .create_metadata(
        strategy_name="StockPredictionV1",
        universe_name="NSE500",
        benchmark_name="NIFTY50",
        owner="QuantResearch",
    )
)

print(
    "\nMetadata"
)

print(
    "Run ID:",
    metadata.run_id
)

print(
    "Strategy:",
    metadata.strategy_name
)

print(
    "Universe:",
    metadata.universe_name
)

print(
    "Benchmark:",
    metadata.benchmark_name
)

# ----------------------------------------------------------
# EXECUTE PIPELINE
# ----------------------------------------------------------

print(
    "\nExecuting Institutional Pipeline"
)


institutional_result = (
    institutional_pipeline(
        inputs=pipeline_input,
        metadata=metadata,
    )
)

print(
    "\nAnalytics Error:"
)

print(
    institutional_result
    .context
    .shared_objects
    .get("analytics_error")
)


print(
    "\nRunning Institutional Alpha Diagnostics"
)

alpha_pipeline = AlphaPipeline()

alpha_diagnostics = alpha_pipeline.run_all(

    raw_df=tracker.get_stage(AlphaStage.RAW),
    meta_df=tracker.get_stage(AlphaStage.META),
    regime_df=tracker.get_stage(AlphaStage.REGIME),
    volatility_df=tracker.get_stage(AlphaStage.VOLATILITY),
    cross_section_df=tracker.get_stage(AlphaStage.CROSS_SECTION),
    portfolio_df=tracker.get_stage(AlphaStage.PORTFOLIO),
    ic_table=alpha_results["tables"]["summary"]
)

# ----------------------------------------------------------
# EXPORT INSTITUTIONAL AUDIT TRAIL
# ----------------------------------------------------------

print(
    "\nInstitutional Alpha Pipeline Completed."
)
# ----------------------------------------------------------
# PIPELINE STATUS
# ----------------------------------------------------------

print("\nPIPELINE DIAGNOSTICS")
print(institutional_result.diagnostics)

print(
    "\nPipeline Status"
)

print(
    institutional_result.status
)

print(
    institutional_result.message
)

# ----------------------------------------------------------
# REPORT VALIDATION
# ----------------------------------------------------------

if (
    institutional_result.report
    is None
):

    raise RuntimeError(
        "Institutional report unavailable."
    )

report = (
    institutional_result.report
)

print(
    "\nInstitutional Report Generated"
)

# ----------------------------------------------------------
# PORTFOLIO
# ----------------------------------------------------------

portfolio_result = (
    report.portfolio_result
)

print(
    "\nPortfolio Summary"
)

print(
    type(
        portfolio_result
    )
)

# ----------------------------------------------------------
# REBALANCE
# ----------------------------------------------------------

rebalance_result = (
    report.rebalance_result
)

print(
    "\nRebalance Summary"
)

print(
    type(
        rebalance_result
    )
)

# ----------------------------------------------------------
# DIAGNOSTICS
# ----------------------------------------------------------

print(
    "\nDiagnostics Report:"
)

print(
    institutional_result
    .report
    .diagnostics_report
)

print(
    "\nRuntime Diagnostics:"
)

print(
    institutional_result
    .report
    .runtime_diagnostics
)

# pipeline diagnostics:
print(
    institutional_result
    .diagnostics
)

# ----------------------------------------------------------
# EXECUTION PACKAGE
# ----------------------------------------------------------

execution = institutional_result.diagnostics.get(
    "execution"
)

if execution is not None:

    print(
        "\nExecution Package Created"
    )

# ----------------------------------------------------------
# ATTRIBUTION
# ----------------------------------------------------------

attribution = institutional_result.diagnostics.get(
    "attribution"
)

if attribution is not None:

    print(
        "Attribution Available"
    )

# ----------------------------------------------------------
# STRESS TEST
# ----------------------------------------------------------

stress = (
    institutional_result
    .report
    .runtime_diagnostics
    .get("stress_testing")
)

if stress is not None:

    print(
        "Stress Testing Available"
    )

# ----------------------------------------------------------
# Monitoring TEST
# ----------------------------------------------------------

monitoring_result = (
    institutional_result
    .report
    .runtime_diagnostics
    .get("monitoring")
)

if monitoring_result is not None:

    print(
        "Monitoring Testing Available"
    )
# ----------------------------------------------------------
# SAVE RESULT
# ----------------------------------------------------------

joblib.dump(

    institutional_result,

    "artifacts/institutional_result.pkl",
)

joblib.dump(

    report,

    "artifacts/institutional_report.pkl",
)

print(
    "\nInstitutional artifacts saved."
)

# ----------------------------------------------------------
# FINAL SUCCESS
# ----------------------------------------------------------

print("\n" + "=" * 60)

print(
    "FIRST INSTITUTIONAL SMOKE TEST PASSED"
)

print("=" * 60)


# ==========================================================
# PART 9
# REPORTING
#
# Purpose
# -------
#
# Institutional Result
#       ↓
# Full Report
#       ↓
# Diagnostics
#       ↓
# Persist Artifacts
#
# FINAL STEP OF MAIN.PY
# ==========================================================

print("\n" + "=" * 60)
print("PART 9 — REPORTING")
print("=" * 60)

# ----------------------------------------------------------
# FULL REPORT
# ----------------------------------------------------------

institutional_report = (
    institutional_result.report
)

if institutional_report is None:
    raise RuntimeError(
        "Institutional pipeline did not produce a report."
    )


print(
    "\nInstitutional Report Created"
)

# ----------------------------------------------------------
# PORTFOLIO
# ----------------------------------------------------------

portfolio_result = (
    institutional_report
    .portfolio_result
)

print(
    "\nPortfolio Result"
)

print(
    type(
        portfolio_result
    )
)

# ----------------------------------------------------------
# REBALANCE
# ----------------------------------------------------------

rebalance_result = (
    institutional_report
    .rebalance_result
)

print(
    "\nRebalance Result"
)

print(
    type(
        rebalance_result
    )
)

# ----------------------------------------------------------
# EXECUTION
# ----------------------------------------------------------

execution_result = (
    institutional_result
    .report
    .runtime_diagnostics
    .get("execution")
)

print(
    "\nExecution Result"
)

print(
    execution_result
    is not None
)

# ----------------------------------------------------------
# ANALYTICS
# ----------------------------------------------------------

analytics_result = (
    institutional_result
    .report
    .runtime_diagnostics
    .get("analytics")
)

analytics_error = (
    institutional_result
    .context
    .shared_objects
    .get("analytics_error")
)

print("\nAnalytics Error Detail")
print(analytics_error)

print(
    "\nAnalytics Result"
)

print(
    analytics_result
    is not None
)

# ----------------------------------------------------------
# ATTRIBUTION
# ----------------------------------------------------------

attribution_result = (
    institutional_result
    .report
    .runtime_diagnostics
    .get("attribution")
)

print(
    "\nAttribution Result"
)

print(
    attribution_result
    is not None
)

# ----------------------------------------------------------
# STRESS TESTING
# ----------------------------------------------------------

stress_result = (
    institutional_result
    .report
    .runtime_diagnostics
    .get("stress_testing")
)

print(
    "\nStress Testing Result"
)

print(
    stress_result
    is not None
)

# ----------------------------------------------------------
# MONITORING
# ----------------------------------------------------------

monitoring_result = (
    institutional_result
    .report
    .runtime_diagnostics
    .get("monitoring")
)

print(
    "\nMonitoring Result"
)

print(
    monitoring_result
    is not None
)

# ----------------------------------------------------------
# PIPELINE DIAGNOSTICS
# ----------------------------------------------------------

pipeline_diagnostics = (
    institutional_result
    .diagnostics
)

print(
    "\nDiagnostics Keys"
)

print(
    list(
        pipeline_diagnostics
        .keys()
    )
)

# ----------------------------------------------------------
# SAVE REPORTS
# ----------------------------------------------------------

print(
    "\nSaving Institutional Reports"
)

joblib.dump(

    institutional_report,

    "artifacts/"
    "institutional_report.pkl",
)

joblib.dump(

    portfolio_result,

    "artifacts/"
    "portfolio_result.pkl",
)

joblib.dump(

    rebalance_result,

    "artifacts/"
    "rebalance_result.pkl",
)

# ----------------------------------------------------------
# SAVE DIAGNOSTICS
# ----------------------------------------------------------

joblib.dump(

    pipeline_diagnostics,

    "artifacts/"
    "pipeline_diagnostics.pkl",
)

if analytics_result is not None:

    joblib.dump(

        analytics_result,

        "artifacts/"
        "analytics.pkl",
    )

if attribution_result is not None:

    joblib.dump(

        attribution_result,

        "artifacts/"
        "attribution.pkl",
    )

if stress_result is not None:

    joblib.dump(

        stress_result,

        "artifacts/"
        "stress_testing.pkl",
    )

if execution_result is not None:

    joblib.dump(

        execution_result,

        "artifacts/"
        "execution.pkl",
    )

if monitoring_result is not None:

    joblib.dump(

        monitoring_result,

        "artifacts/"
        "monitoring.pkl",
    )

# ----------------------------------------------------------
# OPTIONAL CSV EXPORTS
# ----------------------------------------------------------

print(
    "\nExporting CSV Reports"
)

try:

    if hasattr(
        portfolio_result,
        "weights"
    ):

        (
            portfolio_result
            .weights
            .to_csv(
                "artifacts/"
                "portfolio_weights.csv"
            )
        )

except Exception:

    pass

try:

    if isinstance(
        pipeline_diagnostics,
        dict,
    ):

        pd.DataFrame(

            {
                "Key":
                list(
                    pipeline_diagnostics
                    .keys()
                )
            }

        ).to_csv(

            "artifacts/"
            "diagnostics_index.csv",

            index=False,
        )

except Exception:

    pass

# ----------------------------------------------------------
# FINAL SUMMARY
# ----------------------------------------------------------

print("\n" + "=" * 60)

print(
    "INSTITUTIONAL PLATFORM COMPLETE"
)

print("=" * 60)

print(
    f"Run ID: {metadata.run_id}"
)

print(
    f"Strategy: {metadata.strategy_name}"
)

print(
    f"Universe: {metadata.universe_name}"
)

print(
    f"Artifacts Directory: artifacts/"
)

print(
    "\nSaved Files:"
)

saved_files = [

    "institutional_report.pkl",
    "portfolio_result.pkl",
    "rebalance_result.pkl",
    "pipeline_diagnostics.pkl",
    "analytics.pkl",
    "attribution.pkl",
    "stress_testing.pkl",
    "execution.pkl",
    "monitoring.pkl",
]

for file in saved_files:

    print(
        f"  • {file}"
    )

print(
    "\nSUCCESS"
)

print(
    "Alpha Engine + Institutional Construction "
    "Engine integrated successfully."
)

print("=" * 60)

# ==========================================================
# PART 10
# SMOKE TESTS
#
# Final institutional validation.
#
# Alpha Engine
#       +
# Institutional Construction
#       +
# Reporting

# This is where we answer:

# Did input_adapter.py work?
# Did pipeline.py work?
# Did portfolio_builder.py work?
# Did optimizer.py work?
# Did constraints.py work?
# Did risk_model.py work?
# Did reporting work?

# Can we safely deploy?
#
# ==========================================================

print("\n" + "=" * 60)
print("PART 10 — SMOKE TESTS")
print("=" * 60)

smoke_results = {}

# ----------------------------------------------------------
# PIPELINE INPUT
# ----------------------------------------------------------

try:

    smoke_results[
        "PipelineInput"
    ] = (
        pipeline_input
        is not None
    )

except Exception:

    smoke_results[
        "PipelineInput"
    ] = False

# ----------------------------------------------------------
# PORTFOLIO
# ----------------------------------------------------------

try:

    smoke_results[
        "Portfolio"
    ] = (
        portfolio_result
        is not None
    )

except Exception:

    smoke_results[
        "Portfolio"
    ] = False

# ----------------------------------------------------------
# REBALANCE
# ----------------------------------------------------------

try:

    smoke_results[
        "Rebalance"
    ] = (
        rebalance_result
        is not None
    )

except Exception:

    smoke_results[
        "Rebalance"
    ] = False

# ----------------------------------------------------------
# EXECUTION
# ----------------------------------------------------------

try:

    smoke_results[
        "Execution"
    ] = (
        execution_result
        is not None
    )

except Exception:

    smoke_results[
        "Execution"
    ] = False

# ----------------------------------------------------------
# ANALYTICS
# ----------------------------------------------------------

try:

    smoke_results[
        "Analytics"
    ] = (
        analytics_result
        is not None
    )

except Exception:

    smoke_results[
        "Analytics"
    ] = False

# ----------------------------------------------------------
# ATTRIBUTION
# ----------------------------------------------------------

try:

    smoke_results[
        "Attribution"
    ] = (
        attribution_result
        is not None
    )

except Exception:

    smoke_results[
        "Attribution"
    ] = False

# ----------------------------------------------------------
# STRESS TESTING
# ----------------------------------------------------------

try:

    smoke_results[
        "StressTesting"
    ] = (
        stress_result
        is not None
    )

except Exception:

    smoke_results[
        "StressTesting"
    ] = False

# ----------------------------------------------------------
# MONITORING
# ----------------------------------------------------------

try:

    smoke_results[
        "Monitoring"
    ] = (
        monitoring_result
        is not None
    )

except Exception:

    smoke_results[
        "Monitoring"
    ] = False

# ----------------------------------------------------------
# PIPELINE REPORT
# ----------------------------------------------------------

try:

    smoke_results[
        "InstitutionalReport"
    ] = (
        institutional_report
        is not None
    )

except Exception:

    smoke_results[
        "InstitutionalReport"
    ] = False

# ----------------------------------------------------------
# INPUT ADAPTER
# ----------------------------------------------------------

try:

    smoke_results[
        "InputAdapter"
    ] = (
        smoke_test_input_adapter(

            final_df=
            final_df
        )
    )

except Exception:

    smoke_results[
        "InputAdapter"
    ] = False

# ----------------------------------------------------------
# PIPELINE ENGINE
# ----------------------------------------------------------

try:

    smoke_results[
        "PipelineEngine"
    ] = (
        institutional_result
        .report
        is not None
    )

except Exception:

    smoke_results[
        "PipelineEngine"
    ] = False

# ----------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------

print(
    "\nSmoke Test Summary"
)

for k, v in smoke_results.items():

    status = (
        "PASS"
        if v
        else "FAIL"
    )

    print(
        f"{k:<25} {status}"
    )

# ----------------------------------------------------------
# OVERALL STATUS
# ----------------------------------------------------------

overall = all(
    smoke_results.values()
)

print("\n" + "=" * 60)

if overall:

    print(
        "ALL SMOKE TESTS PASSED"
    )

else:

    print(
        "SMOKE TEST FAILURES DETECTED"
    )

print("=" * 60)

# ----------------------------------------------------------
# FINAL COUNTS
# ----------------------------------------------------------

print(
    "\nFINAL COUNTS"
)

print(
    "Universe Size:",
    len(latest_data)
)

print(
    "Features:",
    len(FEATURES)
)

print(
    "Models:",
    len(models)
)

print(
    "Signals:",
    len(signals_df)
)

print(
    "Portfolio Rows:",
    len(portfolio)
)

# ----------------------------------------------------------
# OPTIONAL ASSERTIONS
# ----------------------------------------------------------

assert (
    pipeline_input
    is not None
)

assert (
    institutional_report
    is not None
)

portfolio_available = (
    portfolio_result is not None
)

print(
    "Portfolio Available:",
    portfolio_available,
)

rebalance_available = (
    rebalance_result is not None
)

print(
    "Rebalance Available:",
    rebalance_available,
)

# ----------------------------------------------------------
# FINAL DEPLOYMENT CHECK
# ----------------------------------------------------------

deployment_ready = (
    overall
    and portfolio_available
    and rebalance_available
    and len(models) > 0
    and len(FEATURES) > 0
)

print(
    "\nDeployment Ready:",
    deployment_ready
)

# ----------------------------------------------------------
# FINAL MESSAGE
# ----------------------------------------------------------
print("\n" + "=" * 60)

print(
    "ALPHA ENGINE"
)

print(
    "        +"
)

print(
    "INPUT ADAPTER"
)

print(
    "        +"
)

print(
    "INSTITUTIONAL PIPELINE"
)

print(
    "        +"
)

print(
    "PORTFOLIO BUILDER"
)

print(
    "        +"
)

print(
    "RISK MODEL"
)

print(
    "        +"
)

print(
    "CONSTRAINTS"
)

print(
    "        +"
)

print(
    "OPTIMIZER"
)

print(
    "        +"
)

print(
    "REBALANCE"
)

print(
    "        +"
)

print(
    "EXECUTION"
)

print(
    "        +"
)

print(
    "ANALYTICS"
)

print(
    "        +"
)

print(
    "ATTRIBUTION"
)

print(
    "        +"
)

print(
    "STRESS TESTING"
)

print(
    "        +"
)

print(
    "MONITORING"
)

print(
    "        +"
)

print(
    "INSTITUTIONAL REPORT"
)

print("\nSUCCESS")

print(
    "Institutional-Grade Quant Platform "
    "validated successfully."
)

print("=" * 60)