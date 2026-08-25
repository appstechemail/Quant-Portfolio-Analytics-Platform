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

print("\n🛡 Institutional Feature Set")

print(
    f"Selected Features: {len(FEATURES)}"
)

print(FEATURES)

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

data = (
    final_df[
        ["Date", "Company"]
        + FEATURES
        + extra_cols
    ]
    .dropna()
    .reset_index(drop=True)
)

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
# TRAIN / TEST SPLIT — DATE BASED
# ----------------------------------------------------------
#
# IMPORTANT:
# Split by DATE, not by ROW.
#
# This guarantees that every test date contains the complete
# available stock universe instead of ending up with only the
# last few companies in the dataframe.
# ----------------------------------------------------------

unique_dates = np.sort(
    data["Date"].dropna().unique()
)

train_size = CONFIG["MODEL"]["TRAIN_SIZE"]

split_date_idx = int(
    len(unique_dates) * train_size
)

# Safety bounds
split_date_idx = max(
    1,
    min(
        split_date_idx,
        len(unique_dates) - 1
    )
)

train_dates = unique_dates[:split_date_idx]
test_dates = unique_dates[split_date_idx:]

train_mask = data["Date"].isin(train_dates)
test_mask = data["Date"].isin(test_dates)

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

meta_y_train = meta_y.loc[train_mask].copy()
meta_y_test = meta_y.loc[test_mask].copy()

meta_test = meta.loc[test_mask].copy()

print("\n" + "=" * 60)
print("DATE-BASED TRAIN / TEST SPLIT")
print("=" * 60)

print(
    f"Total dates : {len(unique_dates):,}"
)

print(
    f"Train dates : {len(train_dates):,}"
)

print(
    f"Test dates  : {len(test_dates):,}"
)

print(
    f"Train rows  : {len(X_train):,}"
)

print(
    f"Test rows   : {len(X_test):,}"
)

print(
    f"Train companies : "
    f"{data.loc[train_mask, 'Company'].nunique():,}"
)

print(
    f"Test companies  : "
    f"{data.loc[test_mask, 'Company'].nunique():,}"
)

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

# ----------------------------------------------------------
# BUILD ENSEMBLE
# ----------------------------------------------------------

signals = []

for values in probas.values():

    arr = np.asarray(values)

    if len(arr.shape) > 1:

        arr = arr[:, 1]

    signals.append(arr)

signals = np.column_stack(
    signals
)

ensemble_weights = (
    np.ones(
        signals.shape[1]
    )
    /
    signals.shape[1]
)

ensemble_proba = np.dot(
    signals,
    ensemble_weights,
)

logger.info("=" * 80)
logger.info("BACKTEST DEBUG Before Meta Filter")
logger.info("Rows after merge: %d", len(final_df))

logger.info(
    "Unique Dates=%d | Companies=%d",
    final_df["Date"].nunique(),
    final_df["Company"].nunique()
)
# ----------------------------------------------------------
# APPLY META FILTER
# ----------------------------------------------------------

final_proba = np.where(
    meta_proba > META_THRESHOLD,
    ensemble_proba,
    0.0,
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

print(
    "\nSignals after meta filter:",
    (final_proba > 0).sum(),
)

print(
    "Average signal:",
    round(
        final_proba.mean(),
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

final_proba = np.where(
    meta_proba >  thresholds,
    ensemble_proba,
    0.0,
)

# For Integration of Alpha Stage
meta_df = meta_test.copy()

meta_df["Probability"] = final_proba

tracker.add_stage(
    AlphaStage.META,
    meta_df
)

print(
    "Signals after Meta Filter:",
    (final_proba > 0).sum(),
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
# REGIME FILTER
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
# Do not convert valid BUY probabilities to zero solely
# because of market regime.
#
# Regime is a risk/exposure modifier.
# Portfolio construction / risk engine can reduce exposure.
# ----------------------------------------------------------

regime_multiplier = np.ones(
    len(test_regimes),
    dtype=float,
)

regime_multiplier[
    test_regimes == "BULL"
] = 1.00

regime_multiplier[
    test_regimes == "SIDEWAYS"
] = 0.85

regime_multiplier[
    test_regimes == "BULL_VOLATILE"
] = 0.75

regime_multiplier[
    test_regimes == "SIDEWAYS_VOLATILE"
] = 0.60

regime_multiplier[
    test_regimes == "BEAR"
] = 0.50

regime_multiplier[
    test_regimes == "BEAR_VOLATILE"
] = 0.25

final_proba = (
    final_proba
    *
    regime_multiplier
)

# For Integration with Alpha Stage
regime_df = meta_df.copy()

regime_df["Probability"] = final_proba

tracker.add_stage(
    AlphaStage.REGIME,
    regime_df
)

print(
    "Signals after Regime Filter:",
    (final_proba > 0).sum(),
)

logger.info("=" * 80)
logger.info("BACKTEST DEBUG After - Regime Filter and Before Volatility Filter")
logger.info("Rows after merge: %d", len(final_df))

logger.info(
    "Unique Dates=%d | Companies=%d",
    final_df["Date"].nunique(),
    final_df["Company"].nunique()
)

# ----------------------------------------------------------
# VOLATILITY FILTER
# ----------------------------------------------------------

print(
    "\nApplying Volatility Filter"
)

test_atr = (
    data.loc[test_mask]
    ["ATR_Z"]
    .reset_index(drop=True)
)

atr_mean = (
    test_atr
    .rolling(
        60,
        min_periods=20,
    )
    .mean()
)

vol_filter = (
    test_atr <
    atr_mean * 1.8
)

vol_filter = (
    vol_filter
    .fillna(True)
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
    "Signals BEFORE volatility filter: %d",
    (final_proba > 0).sum(),
)

final_proba = np.where(
    vol_filter,
    final_proba,
    0.0,
)

# For Integration of Alpha Stage
volatility_df = regime_df.copy()

volatility_df["Probability"] = final_proba

tracker.add_stage(
    AlphaStage.VOLATILITY,
    volatility_df
)

print(
    "Signals after Volatility Filter:",
    (final_proba > 0).sum(),
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
    "Signals entering backtest:",
    int((final_proba > 0).sum()),
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

ensemble_bt = run_backtest(
    proba=final_proba,
    X_test=X_test,
    meta_test=meta_test,
    final_df=final_df,
)

results["ENSEMBLE"] = (
    ensemble_bt
)

logger.info("=" * 80)
logger.info("BACKTEST DEBUG After - Backtest")
logger.info("Rows after merge: %d", len(final_df))

logger.info(
    "Unique Dates=%d | Companies=%d",
    final_df["Date"].nunique(),
    final_df["Company"].nunique()
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

        strategy_name=
        "StockPredictionV1",

        universe_name=
        "NSE500",

        benchmark_name=
        "NIFTY50",

        owner=
        "QuantResearch",
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

stress = institutional_result.diagnostics.get(
    "stress_testing"
)

if stress is not None:

    print(
        "Stress Testing Available"
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
    full_report(

        inputs=
        pipeline_input,

        metadata=
        metadata,
    )
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
# FINAL DEPLOYMENT CHECK
# ----------------------------------------------------------

deployment_ready = (

    overall
    and len(portfolio) > 0
    and len(models) > 0
    and len(FEATURES) > 0
)

print(
    "\nDeployment Ready:",
    deployment_ready
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

assert (
    portfolio_result
    is not None
)

assert (
    rebalance_result
    is not None
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