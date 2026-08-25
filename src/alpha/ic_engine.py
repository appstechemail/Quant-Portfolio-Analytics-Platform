# ============================================================
# INFORMATION COEFFICIENT (IC) ENGINE
# ============================================================
#
# FILE: ic_engine.py
#
# PURPOSE:
# --------
# This module evaluates the predictive power of alpha features
# using Information Coefficient (IC) analysis.
#
# The IC engine is one of the most important components in
# institutional quantitative trading systems.
#
# ------------------------------------------------------------
# WHAT IS INFORMATION COEFFICIENT (IC)?
# ------------------------------------------------------------
#
# IC measures how strongly a feature predicts future returns.
#
# Specifically:
#
#     IC = Correlation(
#              feature_t,
#              future_return_t+N
#          )
#
# Higher positive IC:
#
#     → Better predictive feature
#
# Negative IC:
#
#     → Feature predicts opposite direction
#
# Near-zero IC:
#
#     → Feature has no predictive power
#
# ------------------------------------------------------------
# WHY IC MATTERS
# ------------------------------------------------------------
#
# IC helps identify:
#
#     • Which features actually work
#     • Which features are stable
#     • Which features are degrading
#     • Which features deserve higher weights
#     • Which features should be removed
#
# Institutional quant funds heavily rely on IC analysis
# for alpha research and portfolio construction.
#
# ------------------------------------------------------------
# IC INTERPRETATION
# ------------------------------------------------------------
#
# IC Range          Interpretation
# ------------------------------------------------
# < 0.00            Bad / inverse signal
# 0.00 – 0.02       Weak
# 0.02 – 0.05       Useful
# 0.05 – 0.10       Strong
# > 0.10            Exceptional
#
# Even IC around 0.03 can be highly valuable in
# professional quantitative trading systems.
#
# ------------------------------------------------------------
# ICIR (IC INFORMATION RATIO)
# ------------------------------------------------------------
#
# ICIR measures stability of predictive power.
#
# Formula:
#
#     ICIR = Mean(IC) / Std(IC)
#
# Higher ICIR:
#
#     → More stable alpha
#     → More reliable feature
#
# ------------------------------------------------------------
# LEAKAGE SAFETY
# ------------------------------------------------------------
#
# This module is designed to avoid:
#
#     • Lookahead bias
#     • Future leakage
#     • Same-bar leakage
#
# Safeguards:
#
#     • Features must already be lagged
#     • Future returns computed separately
#     • Cross-sectional daily evaluation only
#
# ------------------------------------------------------------
# METHODOLOGY
# ------------------------------------------------------------
#
# For each date:
#
#     1. Rank stocks by feature value
#     2. Rank stocks by future returns
#     3. Compute Spearman rank correlation
#
# Then:
#
#     • Aggregate daily IC values
#     • Compute average IC
#     • Compute ICIR
#     • Rank features
#
# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------
#
# DataFrame containing:
#
#     Date
#     Company
#     Feature columns
#     Future return target
#
# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------
#
# 1. Daily IC DataFrame
# 2. Feature summary DataFrame
# 3. Ranked alpha features
#
# ============================================================

# Long-term IC → "Is this feature structurally good?"
# Rolling IC → "Is this feature working now?"
# Regime IC → "Is this feature good in the current market regime?"

# Global IC decides what is worth keeping.
# Rolling IC decides whether to increase or decrease confidence.
# Regime IC decides whether the current market favors or disfavors the feature.


# ================================================
# ARCHITECHTURE:
# ================================================

# Raw Market Data
#        │
#        ▼
# Feature Engineering
#     (~90 factors)
#        │
#        ▼
# Market Regime Detection
#        │
#        ▼
# Information Coefficient Pipeline
#  ├── Daily IC
#  ├── Summary IC
#  ├── Rolling IC
#  ├── Regime IC
#  └── Feature Correlation Filter
#        │
#        ▼
# Weight Generation
#  ├── Global Weights
#  ├── Rolling Multipliers
#  ├── Regime Multipliers
#        │
#        ▼
# Master IC Table
# (master_table_ic.csv)
#        │
#        ▼
# Adaptive Weight Builder
#        │
#        ▼
# Final Production Feature Set
#        │
#        ▼
# Stock Ranking Model
#        │
#        ▼
# Portfolio Construction
#        │
#        ▼
# Backtest




import numpy as np
import pandas as pd

import os
import logging

from scipy.stats import spearmanr
from config.config import CONFIG

from src.alpha.feature_clustering import diversify_features
from src.alpha.feature_decay import compute_feature_decay, build_decay_weights

from src.alpha.feature_category_budget import (
    build_feature_category_table,
    compute_category_statistics,
    compute_dynamic_category_budgets,
    allocate_feature_budgets,
)
from src.features.features import FEATURE_METADATA

from src.alpha.ic_stability import (
    compute_ic_stability,
    build_stability_weights,
)

import inspect


logger = logging.getLogger(__name__)
EPSILON = 1e-9


# IC FILTERS
IC_CONFIG_GLOBAL = CONFIG["IC_CONFIG"]["GLOBAL"]
IC_CONFIG_ROLLING = CONFIG["IC_CONFIG"]["ROLLING"]
IC_CONFIG_REGIME = CONFIG["IC_CONFIG"]["REGIME"]

IC_CONFIG_CLUSTER = CONFIG["IC_CONFIG"]["CLUSTERING"]

IC_CONFIG_DECAY = CONFIG["IC_CONFIG"]["DECAY"]

IC_CONFIG_ADAPTIVE = CONFIG["IC_CONFIG"]["ADAPTIVE_WEIGHTING"]

# IC GLOBAL
GLOBAL_MIN_ICIR = IC_CONFIG_GLOBAL.get("MIN_ICIR", 0.05)
GLOBAL_MIN_POSITIVE_PCT = IC_CONFIG_GLOBAL.get("MIN_POSITIVE_PCT", 0.50)
GLOBAL_MIN_OBSERVATIONS = IC_CONFIG_GLOBAL.get("MIN_OBSERVATIONS", 1500)
GLOBAL_CORR_THRESHOLD = IC_CONFIG_GLOBAL.get("CORR_THRESHOLD", 0.90)
GLOBAL_TOP_K = IC_CONFIG_GLOBAL.get("TOP_K", 10)

# IC ROLLING
ROLLING_MIN_IC = IC_CONFIG_ROLLING.get("MIN_IC", 0.02)
ROLLING_MIN_ICIR = IC_CONFIG_ROLLING.get("MIN_ICIR", 0.05)
ROLLING_WINDOW = IC_CONFIG_DECAY.get("ROLLING_WINDOW", 60)
ROLLING_MIN_MULTIPLIER = IC_CONFIG_DECAY.get("ROLLING_MIN_MULTIPLIER", 0.80)
ROLLING_MAX_MULTIPLIER = IC_CONFIG_REGIME.get("ROLLING_MAX_MULTIPLIER", 1.20)
ROLLING_TOP_K = IC_CONFIG_ROLLING.get("TOP_K", 10)


# IC REGIME
REGIME_MIN_ICIR = IC_CONFIG_REGIME.get("MIN_ICIR", 0.05)
REGIME_MIN_POSITIVE_PCT = IC_CONFIG_REGIME.get("MIN_POSITIVE_PCT", 0.50)
REGIME_MIN_OBSERVATIONS = IC_CONFIG_REGIME.get("MIN_OBSERVATIONS", 500)
REGIME_MIN_MULTIPLIER = IC_CONFIG_REGIME.get("REGIME_MIN_MULTIPLIER", 0.80)
REGIME_MAX_MULTIPLIER = IC_CONFIG_REGIME.get("REGIME_MAX_MULTIPLIER", 1.20)
REGIME_TOP_K = IC_CONFIG_REGIME.get("TOP_K", 10)

# IC CLUSTERING
CLUSTER_METHOD = IC_CONFIG_CLUSTER.get("METHOD", "hierarchical")
CLUSTER_DISTANCE = IC_CONFIG_CLUSTER.get("DISTANCE", "spearman")
CLUSTER_CORR_THRESHOLD = IC_CONFIG_CLUSTER.get("CORR_THRESHOLD", 0.80)
CLUSTER_MIN_CLUSTER_SIZE = IC_CONFIG_CLUSTER.get("MIN_CLUSTER_SIZE", 2)
CLUSTER_KEEP_TOP_PER_CLUSTER = IC_CONFIG_CLUSTER.get("KEEP_TOP_PER_CLUSTER", 1)

# IC DECAY
DECAY_ROLLING_WINDOW = IC_CONFIG_DECAY.get("ROLLING_WINDOW", 60)
DECAY_MIN_OBSERVATIONS = IC_CONFIG_DECAY.get("MIN_OBSERVATIONS", 30)
DECAY_MIN_RECENT_RATIO = IC_CONFIG_DECAY.get("MIN_RECENT_RATIO", 0.60)
DECAY_MIN_TREND = IC_CONFIG_DECAY.get("MIN_TREND", 0.60)
DECAY_TOP_K = IC_CONFIG_DECAY.get("TOP_K", 10)

# IC ADAPTIVE WEIGHTING
ADAPTIVE_GLOBAL_POWER = IC_CONFIG_ADAPTIVE.get("GLOBAL_POWER", 1.00)
ADAPTIVE_ROLLING_POWER = IC_CONFIG_ADAPTIVE.get("ROLLING_POWER", 0.50)
ADAPTIVE_STABILITY_POWER = IC_CONFIG_ADAPTIVE.get("STABILITY_POWER", 0.50)
ADAPTIVE_REGIME_POWER = IC_CONFIG_ADAPTIVE.get("REGIME_POWER", 0.40)
ADAPTIVE_DECAY_POWER  = IC_CONFIG_ADAPTIVE.get("DECAY_POWER", 0.40)
ADAPTIVE_DIVERSIFICATION_POWER = IC_CONFIG_ADAPTIVE.get("DIVERSIFICATION_POWER", 0.30)
ADAPTIVE_MIN_MULTIPLIER = IC_CONFIG_ADAPTIVE.get("MIN_MULTIPLIER", 0.50)
ADAPTIVE_MAX_MULTIPLIER = IC_CONFIG_ADAPTIVE.get("MAX_MULTIPLIER", 1.50)


MIN_ABS_IC = CONFIG["IC_CONFIG"].get("MIN_ABS_IC", 0.18)

# ============================================================
# GET FEATURE FAMILY
# ============================================================

FEATURE_FAMILY_SUFFIXES = (
    "_Z",
    "_Rank",
    "_Norm",
    "_Scaled",
    "_Std",
)



# ============================================================
# COMPUTE DAILY INFORMATION COEFFICIENT
# ============================================================

def compute_daily_ic(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target_col: str = "Future_Return",
) -> pd.DataFrame:
    """
    Compute daily cross-sectional Information Coefficient (IC)
    for every feature.

    Parameters
    ----------
    df : pd.DataFrame
        Complete feature dataset.

    features : list[str] | None, default=None
        Features to evaluate.
        If None, numeric feature columns are auto-detected.

    target_col : str, default="Future_Return"
        Forward return target.

    Returns
    -------
    pd.DataFrame

        Columns
        -------
        Date
        Feature
        Regime
        IC
        PValue
        N_Stocks
    """

    # --------------------------------------------------------
    # Constants
    # --------------------------------------------------------

    MIN_CROSS_SECTION = 5
    MIN_STD = 1e-9

    # --------------------------------------------------------
    # Input Validation
    # --------------------------------------------------------

    if df.empty:
        logger.warning("Input dataframe is empty.")
        return pd.DataFrame()

    required_columns = {
        "Date",
        target_col,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Auto-detect Features
    # --------------------------------------------------------

    if features is None:

        excluded_columns = {

            # Identifiers
            "Date",
            "Company",
            "Ticker",

            # OHLCV
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",

            # Targets
            "Target",
            "Neutral_Target",
            "CrossSection_Target",
            "Alpha_Target",
            "Alpha_Target_Z",
            "Future_Return",
            "Future_Return_Rank",
            "Future_Close",
            "Future_High",
            "Future_Low",
            "Risk_Adjusted_Return",

            # Labels
            "Meta_Target",
            "TB_Label",
            "TB_Exit",
            "TB_Days",

            # Regime
            "Market_Regime",

            # Misc
            "Threshold",
        }

        features = [
            column
            for column in df.columns

            if (
                column not in excluded_columns
                and
                pd.api.types.is_numeric_dtype(df[column])
            )

        ]

    if not features:
        raise ValueError(
            "No valid numeric feature columns available."
        )

    logger.info(
        "Evaluating %d features for Daily IC.",
        len(features),
    )

    # --------------------------------------------------------
    # Sort Data
    # --------------------------------------------------------

    sort_columns = ["Date"]

    if "Company" in df.columns:
        sort_columns.append("Company")

    df = (
        df
        .sort_values(sort_columns)
        .copy()
    )

    # --------------------------------------------------------
    # Daily IC Computation
    # --------------------------------------------------------

    records: list[dict] = []

    for current_date, daily_data in df.groupby("Date"):

        if len(daily_data) < MIN_CROSS_SECTION:
            continue

        regime = (
            daily_data["Market_Regime"].iloc[0]
            if "Market_Regime" in daily_data.columns
            else "UNKNOWN"
        )

        for feature in features:

            subset = (
                daily_data[
                    [feature, target_col]
                ]
                .dropna()
            )

            if len(subset) < MIN_CROSS_SECTION:
                continue

            if (
                subset[feature].std(ddof=0) < MIN_STD
                or
                subset[target_col].std(ddof=0) < MIN_STD
            ):
                continue

            try:

                ic_value, p_value = spearmanr(
                    subset[feature],
                    subset[target_col],
                )

            except Exception as exc:

                logger.debug(
                    "Daily IC failed for feature '%s' on %s: %s",
                    feature,
                    current_date,
                    exc,
                )

                continue

            if np.isnan(ic_value):
                continue

            records.append(

                {
                    "Date": current_date,
                    "Feature": feature,
                    "Regime": regime,
                    "IC": float(ic_value),
                    "PValue": float(p_value),
                    "N_Stocks": len(subset),
                }
            )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    daily_ic_df = pd.DataFrame(records)

    logger.info(
        "Daily IC computed for %d observations.",
        len(daily_ic_df),
    )

    return daily_ic_df


# ============================================================
# COMPUTE IC SUMMARY
# ============================================================

def compute_ic_summary(
    daily_ic_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute summary Information Coefficient (IC) statistics
    for every feature.

    Parameters
    ----------
    daily_ic_df : pd.DataFrame
        Output of ``compute_daily_ic()``.

    Returns
    -------
    pd.DataFrame

        Columns
        -------
        Feature
        Mean_IC
        Abs_IC
        IC_Std
        ICIR
        Positive_IC_Pct
        Observations
        Alpha_Score
    """

    # --------------------------------------------------------
    # Constants
    # --------------------------------------------------------

    MIN_OBSERVATIONS = 5

    # --------------------------------------------------------
    # Input Validation
    # --------------------------------------------------------

    if daily_ic_df.empty:

        logger.warning(
            "compute_ic_summary(): input dataframe is empty."
        )

        return pd.DataFrame()

    required_columns = {
        "Feature",
        "IC",
    }

    missing_columns = required_columns.difference(
        daily_ic_df.columns
    )

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Feature-Level Summary Statistics
    # --------------------------------------------------------

    summary_df = (
        daily_ic_df
        .groupby("Feature", as_index=False)
        .agg(
            Mean_IC=(
                "IC",
                "mean",
            ),

            Abs_IC=(
                "IC",
                lambda x: x.abs().mean(),
            ),

            IC_Std=(
                "IC",
                lambda x: x.std(ddof=1),
            ),

            Positive_IC_Pct=(
                "IC",
                lambda x: (x > 0).mean(),
            ),

            Observations=(
                "IC",
                "count",
            ),
        )
    )

    # --------------------------------------------------------
    # Minimum Observation Filter
    # --------------------------------------------------------

    summary_df = (
        summary_df[
            summary_df["Observations"] >= MIN_OBSERVATIONS
        ]
        .copy()
    )

    if summary_df.empty:
        logger.warning(
            "No features satisfied minimum IC observations."
        )

        return summary_df

    # --------------------------------------------------------
    # IC Information Ratio
    # --------------------------------------------------------

    summary_df["ICIR"] = (
        summary_df["Mean_IC"]
        /
        (
            summary_df["IC_Std"]
            + EPSILON
        )
    )

    # --------------------------------------------------------
    # Composite Alpha Score
    # --------------------------------------------------------

    summary_df["Alpha_Score"] = (
        0.30 * summary_df["Mean_IC"]
        +
        0.55 * summary_df["ICIR"]
        +
        0.15 * summary_df["Positive_IC_Pct"]
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    numeric_columns = [
        "Mean_IC",
        "Abs_IC",
        "IC_Std",
        "ICIR",
        "Positive_IC_Pct",
        "Alpha_Score",
    ]

    summary_df[numeric_columns] = (
        summary_df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    summary_df = (
        summary_df
        .sort_values(
            by=[
                "Alpha_Score",
                "ICIR",
                "Mean_IC",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    summary_df["Feature_Rank"] = np.arange(
        1,
        len(summary_df) + 1,
    )

    logger.info(
        "IC Summary computed for %d features.",
        len(summary_df),
    )

    return summary_df

# ============================================================
# COMPUTE ROLLING IC
# ============================================================

def compute_rolling_ic(
    daily_ic_df: pd.DataFrame,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Compute rolling Information Coefficient (IC) statistics
    for every feature.

    Parameters
    ----------
    daily_ic_df : pd.DataFrame
        Output of ``compute_daily_ic()``.

    window : int, default=ROLLING_WINDOW
        Rolling window length.

    Returns
    -------
    pd.DataFrame

        Original daily IC table augmented with

        - Rolling_IC
        - Rolling_IC_STD
        - Rolling_ICIR
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if daily_ic_df.empty:

        logger.warning(
            "compute_rolling_ic(): input dataframe is empty."
        )

        return pd.DataFrame()

    required_columns = {
        "Feature",
        "Date",
        "IC",
    }

    missing_columns = required_columns.difference(
        daily_ic_df.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if window < 2:
        raise ValueError(
            "Rolling window must be >= 2."
        )

    # --------------------------------------------------------
    # Prepare Data
    # --------------------------------------------------------

    rolling_df = (
        daily_ic_df
        .sort_values(
            ["Feature", "Date"]
        )
        .reset_index(drop=True)
        .copy()
    )

    min_periods = max(
        10,
        window // 3,
    )

    # --------------------------------------------------------
    # Rolling Mean
    # --------------------------------------------------------

    rolling_df["Rolling_IC"] = (
        rolling_df
        .groupby("Feature")["IC"]
        .transform(
            lambda x:
                x.rolling(
                    window=window,
                    min_periods=min_periods,
                ).mean()
        )
    )

    # --------------------------------------------------------
    # Rolling Standard Deviation
    # --------------------------------------------------------

    rolling_df["Rolling_IC_STD"] = (
        rolling_df
        .groupby("Feature")["IC"]
        .transform(
            lambda x:
                x.rolling(
                    window=window,
                    min_periods=min_periods,
                ).std(ddof=1)
        )
    )

    # --------------------------------------------------------
    # Rolling IC Information Ratio
    # --------------------------------------------------------

    rolling_df["Rolling_ICIR"] = (
        rolling_df["Rolling_IC"]
        /
        (
            rolling_df["Rolling_IC_STD"]
            +
            EPSILON
        )
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    rolling_columns = [
        "Rolling_IC",
        "Rolling_IC_STD",
        "Rolling_ICIR",
    ]

    rolling_df[rolling_columns] = (
        rolling_df[rolling_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    logger.info(
        "Rolling IC computed for %d features using window=%d.",
        rolling_df["Feature"].nunique(),
        window,
    )

    
    return rolling_df


# ============================================================
# FILTER ROLLING IC FEATURES
# ============================================================

def filter_rolling_ic_features(
    rolling_ic_df: pd.DataFrame,
    selected_features: list[str] | None = None,
    regime: str | None = None,
    min_rolling_ic: float = ROLLING_MIN_IC,
    min_rolling_icir: float = ROLLING_MIN_ICIR,
    top_k: int | None = ROLLING_TOP_K,
) -> pd.DataFrame:
    """
    Select features using the most recent rolling IC statistics.

    Parameters
    ----------
    rolling_ic_df : pd.DataFrame
        Output of ``compute_rolling_ic()``.

    selected_features : list[str], optional
        Restrict evaluation to globally-selected features.

    regime : str, optional
        Restrict evaluation to one market regime.

    min_rolling_ic : float
        Minimum acceptable Rolling IC.

    min_rolling_icir : float
        Minimum acceptable Rolling IC Information Ratio.

    top_k : int, optional
        Keep only the strongest rolling features.

    Returns
    -------
    pd.DataFrame
        Latest rolling statistics for selected features.
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if rolling_ic_df.empty:

        logger.warning(
            "Rolling IC dataframe is empty."
        )

        return pd.DataFrame()

    required_columns = {

        "Feature",
        "Date",
        "Rolling_IC",
        "Rolling_ICIR",

    }

    missing_columns = required_columns.difference(
        rolling_ic_df.columns
    )

    if missing_columns:

        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    filtered_df = rolling_ic_df.copy()

    # --------------------------------------------------------
    # Global Feature Filter
    # --------------------------------------------------------

    if selected_features:

        filtered_df = (
            filtered_df[
                filtered_df["Feature"].isin(
                    selected_features
                )
            ]
            .copy()
        )

    # --------------------------------------------------------
    # Regime Filter
    # --------------------------------------------------------

    if (
        regime is not None
        and
        "Regime" in filtered_df.columns
    ):

        filtered_df = (
            filtered_df[
                filtered_df["Regime"] == regime
            ]
            .copy()
        )

    if filtered_df.empty:
        logger.warning(
            "No rows remain after feature/regime filtering."
        )

        return pd.DataFrame()

    candidate_count = filtered_df["Feature"].nunique()

    # --------------------------------------------------------
    # Latest Observation Per Feature
    # --------------------------------------------------------

    latest_df = (
        filtered_df
        .sort_values("Date")
        .groupby(
            "Feature",
            as_index=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Remove Missing Rolling Metrics
    # --------------------------------------------------------

    latest_df = (
        latest_df
        .dropna(
            subset=[
                "Rolling_IC",
                "Rolling_ICIR",
            ]
        )
        .copy()
    )

    # --------------------------------------------------------
    # Apply Thresholds
    # --------------------------------------------------------

    latest_df = (
        latest_df[
            (latest_df["Rolling_IC"] >= min_rolling_ic)
            &
            (latest_df["Rolling_ICIR"] >= min_rolling_icir)
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Rank by Current Strength
    # --------------------------------------------------------

    latest_df = (
        latest_df
        .sort_values(
            by=[
                "Rolling_ICIR",
                "Rolling_IC",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Keep Top-K
    # --------------------------------------------------------

    if (
        top_k is not None
        and
        top_k > 0
    ):
        latest_df = latest_df.head(top_k).copy()

    # --------------------------------------------------------
    # Feature Rank
    # --------------------------------------------------------

    latest_df["Rolling_Rank"] = np.arange(
        1,
        len(latest_df) + 1,
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logger.info(
        "Rolling IC Filter | Candidates=%d | Selected=%d",
        candidate_count,
        len(latest_df),
    )

    
    return latest_df

# ============================================================
# BUILD ROLLING WEIGHTS
# ============================================================

def build_rolling_weights(
    rolling_ic_df: pd.DataFrame,
    selected_features_df: pd.DataFrame,
    min_multiplier: float = ROLLING_MIN_MULTIPLIER,
    max_multiplier: float = ROLLING_MAX_MULTIPLIER,
) -> pd.DataFrame:
    """
    Build rolling feature multipliers.

    Produces:

        Historical_IC
        Recent_IC
        Historical_ICIR
        Recent_ICIR
        Rolling_Rank
        Rolling_Multiplier

    Parameters
    ----------
    rolling_ic_df
        Output of compute_rolling_ic().

    selected_features_df
        Output of filter_ic_features().

    Returns
    -------
    pd.DataFrame
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if rolling_ic_df.empty:
        logger.warning("Rolling IC dataframe is empty.")
        return pd.DataFrame()

    if selected_features_df.empty:
        logger.warning("Selected feature dataframe is empty.")
        return pd.DataFrame()

    required_columns = {
        "Feature",
        "Date",
        "Rolling_IC",
        "Rolling_ICIR",
    }

    missing = required_columns.difference(rolling_ic_df.columns)

    if missing:
        raise ValueError(
            f"rolling_ic_df missing columns: {sorted(missing)}"
        )

    if "Feature" not in selected_features_df.columns:
        raise ValueError(
            "'Feature' column missing from selected_features_df."
        )

    # --------------------------------------------------------
    # Compute Historical vs Recent Statistics
    # --------------------------------------------------------

    records = []

    for feature, group in rolling_ic_df.groupby("Feature"):

        group = (
            group
            .sort_values("Date")
            .reset_index(drop=True)
        )

        rolling_ic = group["Rolling_IC"].dropna()
        rolling_icir = group["Rolling_ICIR"].dropna()

        if len(rolling_ic) == 0:
            continue

        historical_ic = rolling_ic.mean()
        historical_icir = rolling_icir.mean()

        recent_ic = rolling_ic.iloc[-1]
        recent_icir = rolling_icir.iloc[-1]

        records.append({
            "Feature": feature,
            "Historical_IC": historical_ic,
            "Recent_IC": recent_ic,
            "Historical_ICIR": historical_icir,
            "Recent_ICIR": recent_icir,
        })

    rolling_stats_df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Merge with globally selected features
    # --------------------------------------------------------

    rolling_weights_df = (
        selected_features_df[
            ["Feature"]
        ]
        .merge(
            rolling_stats_df,
            on="Feature",
            how="left",
        )
    )

    # --------------------------------------------------------
    # Fill missing values
    # --------------------------------------------------------

    numeric_defaults = {
        "Historical_IC": 0.0,
        "Recent_IC": 0.0,
        "Historical_ICIR": -1e9,
        "Recent_ICIR": -1e9,
    }

    for column, default in numeric_defaults.items():

        rolling_weights_df[column] = (
            rolling_weights_df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(default)
        )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    rolling_weights_df["Rolling_Rank"] = (
        rolling_weights_df["Recent_ICIR"]
        .rank(
            ascending=False,
            method="dense",
        )
    )

    feature_count = len(rolling_weights_df)

    # --------------------------------------------------------
    # Rolling Multiplier
    # --------------------------------------------------------

    if feature_count <= 1:

        rolling_weights_df["Rolling_Multiplier"] = 1.0

    else:

        rolling_weights_df["Rolling_Multiplier"] = (
            max_multiplier
            -
            (
                (rolling_weights_df["Rolling_Rank"] - 1)
                /
                (feature_count - 1)
            )
            *
            (max_multiplier - min_multiplier)
        )

    rolling_weights_df["Rolling_Multiplier"] = (
        rolling_weights_df["Rolling_Multiplier"]
        .clip(
            lower=min_multiplier,
            upper=max_multiplier,
        )
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    rolling_weights_df = (
        rolling_weights_df
        .sort_values(
            by=[
                "Recent_ICIR",
                "Recent_IC",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Rolling weights computed for %d features.",
        len(rolling_weights_df),
    )

    return rolling_weights_df[
        [
            "Feature",
            "Historical_IC",
            "Recent_IC",
            "Historical_ICIR",
            "Recent_ICIR",
            "Rolling_Rank",
            "Rolling_Multiplier",
        ]
    ]


def get_feature_family(
    feature: str,
    suffixes: tuple[str, ...] = FEATURE_FAMILY_SUFFIXES,
) -> str:
    """
    Return the base family name of a feature by removing
    known engineering suffixes.

    Examples
    --------
    RSI_Z            -> RSI
    RSI_Rank         -> RSI
    Momentum_Norm    -> Momentum
    EMA_Spread_Std   -> EMA_Spread
    PE               -> PE

    Parameters
    ----------
    feature : str
        Feature name.

    suffixes : tuple[str, ...], optional
        Recognized feature-engineering suffixes.

    Returns
    -------
    str
        Base feature family.
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not isinstance(feature, str):
        raise TypeError(
            "feature must be a string."
        )

    feature = feature.strip()

    if not feature:

        return feature

    # --------------------------------------------------------
    # Remove One Known Suffix
    # --------------------------------------------------------

    for suffix in suffixes:

        if feature.endswith(suffix):

            return feature.removesuffix(suffix)

    
    return feature


# ============================================================
# FILTER IC FEATURES
# ============================================================

def filter_ic_features(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    min_icir: float = GLOBAL_MIN_ICIR,
    min_positive_pct: float = GLOBAL_MIN_POSITIVE_PCT,
    min_observations: int = GLOBAL_MIN_OBSERVATIONS,
    corr_threshold: float = GLOBAL_CORR_THRESHOLD,
    top_k: int | None = GLOBAL_TOP_K,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Select high-quality alpha features using a multi-stage
    filtering pipeline.

    Filtering stages
    ----------------
    1. IC quality filter
    2. Feature family filter
    3. Correlation redundancy filter
    4. Top-K selection

    Parameters
    ----------
    df : pd.DataFrame
        Full feature dataframe.

    summary_df : pd.DataFrame
        Output of ``compute_ic_summary()``.

    min_icir : float
        Minimum IC Information Ratio.

    min_positive_pct : float
        Minimum positive IC percentage.

    min_observations : int
        Minimum IC observations.

    corr_threshold : float
        Maximum allowable Spearman correlation.

    top_k : int, optional
        Maximum number of features retained.

    Returns
    -------
    tuple

        (
            selected_features_df,
            removed_features,
        )
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if df.empty:

        logger.warning(
            "Input dataframe is empty."
        )

        return pd.DataFrame(), []

    if summary_df.empty:

        logger.warning(
            "IC summary dataframe is empty."
        )

        return pd.DataFrame(), []

    required_columns = {

        "Feature",
        "Abs_IC",
        "ICIR",
        "Positive_IC_Pct",
        "Observations",
        "Alpha_Score",

    }

    missing_columns = required_columns.difference(
        summary_df.columns
    )

    if missing_columns:

        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Stage 1
    # IC Quality Filter
    # --------------------------------------------------------

    quality_df = (

        summary_df[
            (summary_df["Abs_IC"] >= MIN_ABS_IC)
            &
            (summary_df["ICIR"] >= min_icir)
            &
            (
                summary_df["Positive_IC_Pct"]
                >= min_positive_pct
            )
            &
            (
                summary_df["Observations"]
                >= min_observations
            )
        ]
        .copy()
    )

    if quality_df.empty:

        logger.warning(
            "No features passed IC quality filtering."
        )

        return pd.DataFrame(), []

    quality_df = (
        quality_df
        .sort_values(
            "Alpha_Score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Stage 2
    # Family Filter
    # --------------------------------------------------------

    quality_df["Feature_Family"] = (
        quality_df["Feature"]
        .apply(get_feature_family)
    )

    family_df = (
        quality_df
        .drop_duplicates(
            subset="Feature_Family",
            keep="first",
        )
        .drop(columns="Feature_Family")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Candidate Features
    # --------------------------------------------------------

    candidate_features = [
        feature
        for feature in family_df["Feature"]
        if feature in df.columns
    ]

    if not candidate_features:

        logger.warning(
            "No candidate features exist in dataframe."
        )

        return pd.DataFrame(), []

    # --------------------------------------------------------
    # Stage 3
    # Correlation Filter
    # --------------------------------------------------------

    correlation_matrix = (
        df[candidate_features]
        .corr(method="spearman")
        .abs()
    )

    selected_features = []

    for feature in candidate_features:
        keep_feature = True
        for chosen_feature in selected_features:
            correlation = correlation_matrix.loc[
                feature,
                chosen_feature,
            ]
            if (
                pd.notna(correlation)
                and
                correlation >= corr_threshold
            ):

                keep_feature = False
                break

        if keep_feature:
            selected_features.append(feature)

    # --------------------------------------------------------
    # Stage 4
    # Top-K
    # --------------------------------------------------------

    if (
        top_k is not None
        and
        top_k > 0
    ):

        selected_features = selected_features[:top_k]

    # --------------------------------------------------------
    # Final Tables
    # --------------------------------------------------------

    selected_df = (
        family_df[
            family_df["Feature"].isin(
                selected_features
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    removed_features = sorted(
        set(candidate_features)
        -
        set(selected_features)
    )

    logger.info(
        "IC Feature Filter | "
        "Initial=%d | "
        "Quality=%d | "
        "Family=%d | "
        "Selected=%d | "
        "Removed=%d",

        len(summary_df),
        len(quality_df),
        len(family_df),
        len(selected_df),
        len(removed_features),
    )

    return (
        selected_df,
        removed_features,
    )



# ============================================================
# BUILD IC WEIGHTS
# ============================================================

def build_ic_weights(
    selected_features_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Build normalized global IC weights.

    Features are weighted primarily by Alpha Score.
    If Alpha Score is unavailable (or sums to zero),
    Absolute IC is used as a fallback.

    Parameters
    ----------
    selected_features_df : pd.DataFrame
        Output of ``filter_ic_features()``.

    Returns
    -------
    dict[str, float]
        Mapping

            Feature -> Global IC Weight
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if selected_features_df.empty:

        logger.warning(
            "Selected feature dataframe is empty."
        )

        return {}

    required_columns = {

        "Feature",
        "Alpha_Score",
        "Abs_IC",

    }

    missing_columns = required_columns.difference(
        selected_features_df.columns
    )

    if missing_columns:

        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Remove Duplicate Features
    # --------------------------------------------------------

    feature_df = (
        selected_features_df
        .drop_duplicates(
            subset="Feature",
            keep="first",
        )
        .copy()
    )

    # --------------------------------------------------------
    # Primary Weight
    # Alpha Score
    # --------------------------------------------------------

    weight_series = (
        feature_df
        .set_index("Feature")["Alpha_Score"]
        .clip(lower=0.0)
    )

    # --------------------------------------------------------
    # Fallback
    # Absolute IC
    # --------------------------------------------------------

    if weight_series.sum() <= EPSILON:

        logger.warning(
            "Alpha Score weights are zero. "
            "Falling back to Abs_IC."
        )

        weight_series = (
            feature_df
            .set_index("Feature")["Abs_IC"]
            .clip(lower=0.0)
        )

    # --------------------------------------------------------
    # Final Safety
    # --------------------------------------------------------

    total_weight = weight_series.sum()

    if total_weight <= EPSILON:
        logger.warning(
            "Unable to construct IC weights."
        )

        return {}

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    weight_series = weight_series / total_weight

    logger.info(
        "Built normalized IC weights for %d features.",
        len(weight_series),
    )

    
    return weight_series.to_dict()



# ============================================================
# COMPUTE REGIME IC
# ============================================================

def compute_regime_ic(
    daily_ic_df: pd.DataFrame,
    min_observations: int = REGIME_MIN_OBSERVATIONS,
) -> pd.DataFrame:
    """
    Compute regime-specific IC statistics.

    Statistics are calculated independently for every

        Regime × Feature

    combination.

    Parameters
    ----------
    daily_ic_df : pd.DataFrame
        Output of ``compute_daily_ic()``.

    min_observations : int
        Minimum IC observations required.

    Returns
    -------
    pd.DataFrame

        Columns
        -------
        Regime
        Feature
        Mean_IC
        Abs_IC
        IC_Std
        ICIR
        Positive_IC_Pct
        Observations
        Alpha_Score
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if daily_ic_df.empty:

        logger.warning(
            "Daily IC dataframe is empty."
        )

        return pd.DataFrame()

    required_columns = {
        "Regime",
        "Feature",
        "IC",
    }

    missing_columns = required_columns.difference(
        daily_ic_df.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Aggregate Statistics
    # --------------------------------------------------------

    regime_summary = (

        daily_ic_df

        .groupby(
            ["Regime", "Feature"],
            as_index=False,
        )

        .agg(
            Mean_IC=(
                "IC",
                "mean",
            ),

            Abs_IC=(
                "IC",
                lambda x: x.abs().mean(),
            ),

            IC_Std=(
                "IC",
                "std",
            ),

            Positive_IC_Pct=(
                "IC",
                lambda x: (x > 0).mean(),
            ),

            Observations=(
                "IC",
                "count",
            ),
        )
    )

    # --------------------------------------------------------
    # ICIR
    # --------------------------------------------------------

    regime_summary["ICIR"] = (
        regime_summary["Mean_IC"]
        /
        (
            regime_summary["IC_Std"]
            +
            EPSILON
        )
    )

    regime_summary["ICIR"] = (
        regime_summary["ICIR"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Alpha Score
    # --------------------------------------------------------

    regime_summary["Alpha_Score"] = (
        regime_summary["Abs_IC"]
        *
        regime_summary["ICIR"].clip(lower=0.0)
    )

    # --------------------------------------------------------
    # Observation Filter
    # --------------------------------------------------------

    regime_summary = (
        regime_summary[
            regime_summary["Observations"]
            >=
            min_observations
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Final Ordering
    # --------------------------------------------------------

    regime_summary = (
        regime_summary
        .sort_values(
            by=[
                "Regime",
                "Alpha_Score",
                "Abs_IC",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Computed regime IC statistics for %d "
        "Regime × Feature combinations.",

        len(regime_summary),
    )

    
    return regime_summary


# ============================================================
# FILTER REGIME IC FEATURES
# ============================================================

def filter_regime_ic_features(
    regime_ic_df: pd.DataFrame,
    selected_features: list[str] | None = None,
    regime: str | None = None,
    min_icir: float = REGIME_MIN_ICIR,
    min_positive_pct: float = REGIME_MIN_POSITIVE_PCT,
    min_observations: int = REGIME_MIN_OBSERVATIONS,
    top_k: int | None = REGIME_TOP_K,
) -> pd.DataFrame:
    """
    Select regime-specific alpha features.

    Filtering stages
    ----------------
    1. Regime selection
    2. Static feature selection
    3. Remove missing statistics
    4. Quality filters
    5. Ranking
    6. Top-K selection

    Parameters
    ----------
    regime_ic_df : pd.DataFrame
        Output of ``compute_regime_ic()``.

    selected_features : list[str], optional
        Features already selected by the global IC engine.

    regime : str, optional
        Current market regime.

    min_icir : float
        Minimum regime ICIR.

    min_positive_pct : float
        Minimum positive IC percentage.

    min_observations : int
        Minimum observations.

    top_k : int, optional
        Maximum number of regime features retained.

    Returns
    -------
    pd.DataFrame
        Filtered regime feature table.
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if regime_ic_df.empty:

        logger.warning(
            "Regime IC dataframe is empty."
        )

        return pd.DataFrame()

    required_columns = {
        "Regime",
        "Feature",
        "Alpha_Score",
        "ICIR",
        "Positive_IC_Pct",
        "Observations",
    }

    missing_columns = required_columns.difference(
        regime_ic_df.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    filtered_df = regime_ic_df.copy()

    # --------------------------------------------------------
    # Stage 1
    # Regime Filter
    # --------------------------------------------------------

    if regime is not None:

        filtered_df = (
            filtered_df[
                filtered_df["Regime"] == regime
            ]
            .copy()
        )

    if filtered_df.empty:
        logger.warning(
            "No features available for regime '%s'.",
            regime,
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Stage 2
    # Static Feature Filter
    # --------------------------------------------------------

    if selected_features is not None:
        filtered_df = (
            filtered_df[
                filtered_df["Feature"].isin(
                    selected_features
                )
            ]
            .copy()
        )

    if filtered_df.empty:
        logger.warning(
            "No regime features overlap with "
            "globally selected features."
        )

        return pd.DataFrame()

    candidate_count = len(filtered_df)

    # --------------------------------------------------------
    # Stage 3
    # Remove Missing Statistics
    # --------------------------------------------------------

    filtered_df = (
        filtered_df
        .dropna(
            subset=[
                "Alpha_Score",
                "ICIR",
            ]
        )
        .copy()
    )

    if filtered_df.empty:
        logger.warning(
            "All regime statistics are missing."
        )
        return pd.DataFrame()

    # --------------------------------------------------------
    # Stage 4
    # Quality Filters
    # --------------------------------------------------------

    filtered_df = (
        filtered_df[
            (filtered_df["ICIR"] >= min_icir)
            &
            (
                filtered_df["Positive_IC_Pct"]
                >= min_positive_pct
            )
            &
            (
                filtered_df["Observations"]
                >= min_observations
            )
        ]
        .copy()
    )

    if filtered_df.empty:
        logger.warning(
            "No regime features passed quality filters."
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Stage 5
    # Ranking
    # --------------------------------------------------------

    filtered_df = (
        filtered_df
        .sort_values(
            by=[
                "Alpha_Score",
                "ICIR",
                "Mean_IC",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Stage 6
    # Top-K
    # --------------------------------------------------------

    if (
        top_k is not None
        and
        top_k > 0
    ):

        filtered_df = (
            filtered_df
            .head(top_k)
            .reset_index(drop=True)
        )

    logger.info(
        "Regime Feature Filter | "
        "Regime=%s | "
        "Candidates=%d | "
        "Selected=%d",
        regime,
        candidate_count,
        len(filtered_df),
    )

    return filtered_df


# ============================================================
# BUILD REGIME WEIGHTS
# ============================================================

def build_regime_weights(
    regime_ic_df: pd.DataFrame,
    selected_features_df: pd.DataFrame,
    current_regime: str,
    min_multiplier: float = REGIME_MIN_MULTIPLIER,
    max_multiplier: float = REGIME_MAX_MULTIPLIER,
) -> pd.DataFrame:
    """
    Build regime-specific feature multipliers.

    Features that are not present in the current regime
    remain in the output but receive the lowest ranking.

    Parameters
    ----------
    regime_ic_df : pd.DataFrame
        Output of ``compute_regime_ic()``.

    selected_features_df : pd.DataFrame
        Output of ``filter_regime_ic_features()``.

    current_regime : str
        Active market regime.

    min_multiplier : float
        Minimum regime multiplier.

    max_multiplier : float
        Maximum regime multiplier.

    Returns
    -------
    pd.DataFrame

        Columns
        -------
        Feature
        Regime
        Mean_IC
        Abs_IC
        ICIR
        Alpha_Score
        Regime_Rank
        Regime_Weight
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if regime_ic_df.empty:
        logger.warning(
            "Regime IC dataframe is empty."
        )

        return pd.DataFrame()

    if selected_features_df.empty:
        logger.warning(
            "Selected feature dataframe is empty."
        )

        return pd.DataFrame()

    required_columns = {
        "Feature",
        "Regime",
        "Mean_IC",
        "Abs_IC",
        "ICIR",
        "Alpha_Score",
    }

    missing_columns = required_columns.difference(
        regime_ic_df.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Current Regime
    # --------------------------------------------------------

    regime_df = (
        regime_ic_df[
            regime_ic_df["Regime"] == current_regime
        ]

        .copy()

    )

    if regime_df.empty:

        logger.warning(
            "No IC statistics found for regime '%s'.",
            current_regime,
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Keep Globally Selected Features
    # --------------------------------------------------------

    regime_df = (
        selected_features_df[
            ["Feature"]
        ]
        .drop_duplicates()
        .merge(
            regime_df,
            on="Feature",
            how="left",
        )
    )

    # --------------------------------------------------------
    # Missing Regime Statistics
    # --------------------------------------------------------

    regime_df = regime_df.fillna({
        "Regime": current_regime,
        "Mean_IC": 0.0,
        "Abs_IC": 0.0,
        "ICIR": -np.inf,
        "Alpha_Score": -np.inf,
        "Positive_IC_Pct": 0.0,
        "Observations": 0,
    })

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    regime_df = (
        regime_df
        .sort_values(
            by=[
                "Alpha_Score",
                "ICIR",
                "Mean_IC",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )

        .reset_index(drop=True)

    )

    regime_df["Regime_Rank"] = (
        regime_df["Alpha_Score"]
        .rank(
            method="dense",
            ascending=False,
        )
    )

    # --------------------------------------------------------
    # Regime Multiplier
    # --------------------------------------------------------

    num_features = len(regime_df)

    if num_features == 1:

        regime_df["Regime_Weight"] = 1.0

    else:

        regime_df["Regime_Weight"] = (
            max_multiplier
            -
            (
                (regime_df["Regime_Rank"] - 1)
                /
                (num_features - 1)
            )
            *
            (
                max_multiplier
                -
                min_multiplier
            )
        )

    logger.info(
        "Built regime weights for %d features "
        "(Regime=%s).",
        len(regime_df),
        current_regime,
    )

    return regime_df[
        [
            "Feature",
            "Regime",
            "Mean_IC",
            "Abs_IC",
            "ICIR",
            "Alpha_Score",
            "Regime_Rank",
            "Regime_Weight",
        ]
    ]


# ============================================================
# BUILD MASTER IC TABLE
# PART 1
#
#   • Function definition
#   • Validation
#   • Production merge helper
#   • Global IC merge
#   • Rolling merge
#   • Stability merge
# ============================================================

def build_master_ic_table(
    summary_df: pd.DataFrame,
    ic_weights: dict,
    rolling_selected_df: pd.DataFrame | None = None,
    rolling_weights_df: pd.DataFrame | None = None,
    regime_selected_df: pd.DataFrame | None = None,
    regime_weights_df: pd.DataFrame | None = None,
    clustering_result: dict | None = None,
    decay_df: pd.DataFrame | None = None,
    decay_weights_df: pd.DataFrame | None = None,
    stability_weights_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the production Master IC Table.

    This table is the central integration layer of the Alpha Engine.
    Every phase contributes feature-level information which is merged
    into one deterministic dataframe.

    Parameters
    ----------
    summary_df
        Global IC summary.

    ic_weights
        Global normalized IC weights.

    rolling_selected_df
        Output of filter_rolling_ic_features().

    rolling_weights_df
        Output of build_rolling_weights().

    regime_selected_df
        Output of filter_regime_ic_features().

    regime_weights_df
        Output of build_regime_weights().

    clustering_result
        Output of feature clustering.

    decay_df
        Output of compute_feature_decay().

    decay_weights_df
        Output of build_decay_weights().

    stability_weights_df
        Output of build_stability_weights().

    Returns
    -------
    pd.DataFrame
    """

    logger.info(
        "Building Master IC table..."
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if summary_df is None or summary_df.empty:

        logger.warning(
            "Summary dataframe is empty."
        )

        return pd.DataFrame()

    if "Feature" not in summary_df.columns:

        raise ValueError(
            "summary_df must contain 'Feature'."
        )

    master = summary_df.copy()

    # ========================================================
    # Production Merge Helper
    # ========================================================

    def safe_merge(
        left: pd.DataFrame,
        right: pd.DataFrame | None,
        columns: list[str],
        *,
        rename: dict | None = None,
        on: str = "Feature",
    ) -> pd.DataFrame:
        """
        Safe production merge.

        Features
        --------
        • Ignores None dataframes

        • Ignores empty dataframes

        • Keeps only existing columns

        • Prevents duplicate _x/_y columns

        • Allows optional rename before merge
        """

        if right is None:
            return left

        if right.empty:
            return left

        if on not in right.columns:
            return left

        available = [
            c
            for c in columns
            if c in right.columns
        ]

        if not available:
            return left

        if on not in available:
            available.insert(0, on)

        merge_df = right.loc[:, available].copy()

        if rename is not None:
            merge_df.rename(
                columns=rename,
                inplace=True,
            )

        # ------------------------------------
        # Remove duplicate columns
        # ------------------------------------

        duplicate_columns = [
            c
            for c in merge_df.columns

            if (
                c != on
                and
                c in left.columns
            )

        ]

        if duplicate_columns:

            merge_df.drop(
                columns=duplicate_columns,
                inplace=True,
            )

        return left.merge(
            merge_df,
            on=on,
            how="left",
            validate="one_to_one",
        )

    # ========================================================
    # PHASE 1
    # GLOBAL IC
    # ========================================================

    logger.info(
        "Merging Global IC metrics..."
    )

    master["Global_Weight"] = (
        master["Feature"]
        .map(ic_weights)
        .fillna(0.0)
    )

    # ========================================================
    # PHASE 2
    # ROLLING IC
    # ========================================================

    logger.info(
        "Merging Rolling IC..."
    )

    master = safe_merge(
        master,
        rolling_selected_df,
        [
            "Feature",
            "Rolling_IC",
            "Rolling_ICIR",
        ],
    )

    master = safe_merge(
        master,
        rolling_weights_df,
        [
            "Feature",
            "Rolling_Rank",
            "Rolling_Multiplier",
        ],
    )

    if "Rolling_Multiplier" in master.columns:

        master.rename(
            columns={
                "Rolling_Multiplier":
                    "Rolling_Weight"
            },
            inplace=True,
        )

    # ========================================================
    # PHASE 3
    # STABILITY
    # ========================================================

    logger.info(
        "Merging Stability metrics..."
    )

    master = safe_merge(
        master,
        stability_weights_df,
        [
            "Feature",
            "IC_Volatility",
            "IC_FlipRate",
            "Rolling_IC_Volatility",
            "Rolling_ICIR_Volatility",
            "Stability_Score",
            "Stability_Multiplier",
            "Stability_Weight",
            "Stability_Rank",
        ],
    )

    # ========================================================
    # PHASE 4
    # FEATURE CLUSTERING
    # ========================================================

    logger.info(
        "Merging Feature Clustering..."
    )

    if clustering_result is not None:

        cluster_df = clustering_result.get("clusters")
        cluster_weight_df = clustering_result.get("cluster_weights")

        master = safe_merge(
            master,
            cluster_df,
            [
                "Feature",
                "Cluster_ID",
                "Cluster_Size",
                "Representative",
                "Dropped_By_Cluster",
            ],
        )

        if (
            cluster_weight_df is not None
            and
            not cluster_weight_df.empty
            and
            "Cluster_ID" in master.columns
        ):

            cluster_cols = [
                c
                for c in cluster_weight_df.columns

                if c != "Cluster_ID"
            ]

            duplicate = [
                c
                for c in cluster_cols

                if c in master.columns
            ]

            merge_df = cluster_weight_df.drop(
                columns=duplicate,
                errors="ignore",
            )

            master = master.merge(
                merge_df,
                on="Cluster_ID",
                how="left",
                validate="many_to_one",
            )

    # ========================================================
    # PHASE 5
    # REGIME IC
    # ========================================================

    logger.info(
        "Merging Regime IC..."
    )

    master = safe_merge(
        master,
        regime_selected_df,
        [
            "Feature",
            "Regime",
            "Mean_IC",
            "Abs_IC",
            "ICIR",
            "Alpha_Score",
            "Positive_IC_Pct",
            "Observations",
        ],
        rename={
            "Mean_IC": "Regime_Mean_IC",
            "Abs_IC": "Regime_Abs_IC",
            "ICIR": "Regime_ICIR",
            "Alpha_Score": "Regime_Alpha_Score",
            "Positive_IC_Pct": "Regime_Positive_IC_Pct",
            "Observations": "Regime_Observations",
        },
    )

    master = safe_merge(
        master,
        regime_weights_df,
        [
            "Feature",
            "Regime_Weight",
        ],
    )

    # ========================================================
    # PHASE 6
    # FEATURE DECAY
    # ========================================================

    logger.info(
        "Merging Feature Decay..."
    )

    master = safe_merge(
        master,
        decay_df,
        [
            "Feature",
            "Historical_IC",
            "Recent_IC",
            "IC_Delta",
            "Historical_ICIR",
            "Recent_ICIR",
            "ICIR_Delta",
            "Recent_Positive_Pct",
            "IC_Trend",
        ],
    )

    master = safe_merge(
        master,
        decay_weights_df,
        [
            "Feature",
            "Decay_Score",
            "Decay_Multiplier",
            "Decay_Weight",
        ],
    )

    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    numeric_defaults = {

        "Global_Weight": 0.0,
        "Rolling_Weight": 1.0,
        "Regime_Weight": 1.0,
        "Decay_Multiplier": 1.0,
        "Decay_Weight": 1.0,
        "Stability_Multiplier": 1.0,
        "Stability_Weight": 1.0,
        "Diversification_Multiplier": 1.0,
        "Cluster_Weight": 1.0,
    }

    for column, value in numeric_defaults.items():

        if column in master.columns:

            master[column] = (

                master[column]

                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )

                .fillna(value)

            )

    # ========================================================
    # FEATURE FLAG
    # ========================================================

    if "Representative" in master.columns:
        master["Use_Feature"] = (
            master["Representative"]
            .fillna(False)
            .astype(bool)
        )
    else:
        master["Use_Feature"] = True

    # ========================================================
    # CANONICAL COLUMN ORDER
    # ========================================================

    preferred_columns = [

        # ----------------------------------------------------
        # Feature Identity
        # ----------------------------------------------------
        "Feature",

        # ----------------------------------------------------
        # Global IC
        # ----------------------------------------------------
        "Mean_IC",
        "Abs_IC",
        "IC_Std",
        "Positive_IC_Pct",
        "Observations",
        "ICIR",
        "Alpha_Score",
        "Global_Weight",

        # ----------------------------------------------------
        # Rolling IC
        # ----------------------------------------------------
        "Rolling_IC",
        "Rolling_ICIR",
        "Rolling_Rank",
        "Rolling_Weight",

        # ----------------------------------------------------
        # Stability
        # ----------------------------------------------------
        "IC_Volatility",
        "IC_FlipRate",
        "Rolling_IC_Volatility",
        "Rolling_ICIR_Volatility",
        "Stability_Score",
        "Stability_Multiplier",
        "Stability_Weight",
        "Stability_Rank",

        # ----------------------------------------------------
        # Clustering
        # ----------------------------------------------------
        "Cluster_ID",
        "Cluster_Size",
        "Representative",
        "Dropped_By_Cluster",
        "Cluster_Weight",
        "Diversification_Multiplier",

        # ----------------------------------------------------
        # Regime
        # ----------------------------------------------------
        "Regime",
        "Regime_Mean_IC",
        "Regime_Abs_IC",
        "Regime_ICIR",
        "Regime_Alpha_Score",
        "Regime_Positive_IC_Pct",
        "Regime_Observations",
        "Regime_Weight",

        # ----------------------------------------------------
        # Decay
        # ----------------------------------------------------
        "Historical_IC",
        "Recent_IC",
        "IC_Delta",
        "Historical_ICIR",
        "Recent_ICIR",
        "ICIR_Delta",
        "Recent_Positive_Pct",
        "IC_Trend",
        "Decay_Score",
        "Decay_Multiplier",
        "Decay_Weight",

        # ----------------------------------------------------
        # Final Selection
        # ----------------------------------------------------
        "Feature_Rank",
        "Use_Feature",
    ]

    ordered_columns = [
        column
        for column in preferred_columns
        if column in master.columns
    ]

    remaining_columns = [
        column
        for column in master.columns
        if column not in ordered_columns
    ]

    master = master[
        ordered_columns + remaining_columns
    ]

    # ========================================================
    # FINAL SAFETY
    # ========================================================

    duplicate_suffix_columns = [
        column
        for column in master.columns
        if (
            column.endswith("_x")
            or
            column.endswith("_y")
        )
    ]

    if duplicate_suffix_columns:
        logger.warning(
            "Removing duplicate merge columns: %s",
            duplicate_suffix_columns,
        )
        master.drop(
            columns=duplicate_suffix_columns,
            inplace=True,
            errors="ignore",
        )

    # ========================================================
    # NUMERIC SANITY
    # ========================================================

    numeric_columns = master.select_dtypes(
        include=[np.number]
    ).columns

    master[numeric_columns] = (
        master[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # ========================================================
    # SORT
    # ========================================================

    if "Feature_Rank" in master.columns:

        master = master.sort_values(
            by=[
                "Feature_Rank",
                "Alpha_Score",
            ],
            ascending=[
                True,
                False,
            ],
            na_position="last",
        )

    else:

        master = master.sort_values(
            "Alpha_Score",
            ascending=False,
        )

    master.reset_index(
        drop=True,
        inplace=True,
    )

    # ========================================================
    # LOGGING
    # ========================================================

    logger.info(
        "Master table created with %d features.",
        len(master),
    )

    logger.info(
        "Master table columns:\n%s",
        "\n".join(master.columns),
    )

    # ========================================================
    # RETURN
    # ========================================================

    return master


# ============================================================
# BUILD ADAPTIVE IC WEIGHTS
# ============================================================

def build_adaptive_ic_weights(
    master_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build production Adaptive IC Weights.

    Final IC Score is computed as

        Global Weight
        × Rolling
        × Stability
        × Regime
        × Decay
        × Diversification

    with configurable exponentiation.

    Parameters
    ----------
    master_df

        Master IC table.

    Returns
    -------
    pd.DataFrame
    """

    if master_df.empty:
        return pd.DataFrame()

    df = master_df.copy()

    # ----------------------------------------
    # DEBUG
    # ----------------------------------------

    logger.info("Master IC columns:")
    logger.info(
        "Master table columns:\n%s",
        "\n".join(df.columns)
    )

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def _prepare_multiplier(
        column: str,
    ) -> pd.Series:
        """
        Return a valid multiplier Series.

        Missing multiplier columns default to 1.0.
        """

        if column in df.columns:
            values = df[column]
        else:
            values = pd.Series(
                1.0,
                index=df.index,
                dtype=float,
            )

        return (
            values
            .fillna(1.0)
            .clip(
                lower=ADAPTIVE_MIN_MULTIPLIER,
                upper=ADAPTIVE_MAX_MULTIPLIER,
            )
        )

    # --------------------------------------------------------
    # Components
    # --------------------------------------------------------

    global_weight = (

        df["Global_Weight"]

        .fillna(0.0)

        .clip(lower=0.0)

    )

    rolling = _prepare_multiplier(
        "Rolling_Weight"
    )

    stability = _prepare_multiplier(
        "Stability_Multiplier"
    )

    regime = _prepare_multiplier(
        "Regime_Weight"
    )

    decay = _prepare_multiplier(
        "Decay_Multiplier"
    )

    diversification = _prepare_multiplier(
        "Diversification_Multiplier"
    )

    # --------------------------------------------------------
    # Component Columns
    # --------------------------------------------------------

    df["Rolling_Component"] = rolling
    df["Stability_Component"] = stability
    df["Regime_Component"] = regime
    df["Decay_Component"] = decay
    df["Diversification_Component"] = diversification

    # --------------------------------------------------------
    # Component Contributions
    # --------------------------------------------------------

    df["Rolling_Contribution"] = (
        rolling ** ADAPTIVE_ROLLING_POWER
    )

    df["Stability_Contribution"] = (
        stability ** ADAPTIVE_STABILITY_POWER
    )

    df["Regime_Contribution"] = (
        regime ** ADAPTIVE_REGIME_POWER
    )

    df["Decay_Contribution"] = (
        decay ** ADAPTIVE_DECAY_POWER
    )

    df["Diversification_Contribution"] = (
        diversification
        ** ADAPTIVE_DIVERSIFICATION_POWER
    )

    # --------------------------------------------------------
    # Final IC Score
    # --------------------------------------------------------

    df["Final_IC_Score"] = (
        (global_weight ** ADAPTIVE_GLOBAL_POWER)
        *
        df["Rolling_Contribution"]
        *
        df["Stability_Contribution"]
        *
        df["Regime_Contribution"]
        *
        df["Decay_Contribution"]
        *
        df["Diversification_Contribution"]
    )

    df["Final_IC_Score"] = (
        df["Final_IC_Score"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    # --------------------------------------------------------
    # Adaptive Weight
    # --------------------------------------------------------

    total_score = df["Final_IC_Score"].sum()

    if total_score <= EPSILON:
        raise ValueError(
            "Adaptive weights sum to zero."
        )

    df["Adaptive_Weight"] = (
        df["Final_IC_Score"]
        / total_score
    )

    df["Adaptive_Score"] = (
        df["Adaptive_Weight"]
        * 100.0
    )

    # --------------------------------------------------------
    # Feature Ranking
    # --------------------------------------------------------

    rank_mask = (
        df["Adaptive_Weight"] > 0
    )

    ranking = (
        df.loc[rank_mask]
        .sort_values(
            [
                "Adaptive_Weight",
                "Global_Weight",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    ranking["Feature_Rank"] = np.arange(
        1,
        len(ranking) + 1,
    )

    df["Feature_Rank"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64",
    )

    df.loc[
        ranking.index,
        "Feature_Rank",
    ] = ranking["Feature_Rank"]

    # --------------------------------------------------------
    # Final Selection Flag
    # --------------------------------------------------------

    df["Use_Feature"] = (
        df["Feature_Rank"].notna()
        &
        (
            df["Feature_Rank"]
            <= GLOBAL_TOP_K
        )
    )

    return df


# ============================================================
# MAIN ALPHA PIPELINE
# ============================================================

def run_alpha_pipeline(
    df: pd.DataFrame,
    current_regime: str,
    features: list[str] | None = None,
    target_col: str = "Future_Return",
) -> dict:
    """
    Execute the complete Alpha Feature Selection pipeline.

    Pipeline Order
    --------------
    Phase 1
        • Global IC
        • Rolling IC
        • Regime IC

    Phase 2
        • Stability Engine

    Phase 3
        • Feature Clustering

    Phase 4
        • Feature Decay

    Phase 5
        • Master IC Table
        • Adaptive IC Weights

    Phase 6
        • Feature Categories
        • Dynamic Category Budgets
        • Final Feature Weights

    Parameters
    ----------
    df : pd.DataFrame
        Complete feature dataframe.

    current_regime : str
        Active market regime.

    features : list[str] | None
        Feature list.
        None = automatically detect.

    target_col : str
        Forward return column.

    Returns
    -------
    dict
        Complete Alpha Engine outputs.
    """

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    if df is None:
        raise ValueError("Input dataframe cannot be None.")

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found."
        )

    if current_regime is None:
        raise ValueError(
            "current_regime must be provided."
        )

    if not isinstance(current_regime, str):
        raise TypeError(
            "current_regime must be a string."
        )

    # ========================================================
    # PIPELINE INITIALIZATION
    # ========================================================

    logger.info("=" * 80)
    logger.info("STARTING ALPHA PIPELINE")
    logger.info("=" * 80)

    logger.info(
        "Rows              : %s",
        len(df),
    )

    logger.info(
        "Features Requested: %s",
        "Auto"
        if features is None
        else len(features),
    )

    logger.info(
        "Target Column     : %s",
        target_col,
    )

    logger.info(
        "Current Regime    : %s",
        current_regime,
    )

    # ========================================================
    # PIPELINE OUTPUT PLACEHOLDERS
    # ========================================================

    daily_ic_df = None
    summary_df = None

    selected_features_df = None
    removed_features = None
    ic_weights = None

    rolling_ic_df = None
    rolling_selected_df = None
    rolling_weights_df = None

    regime_ic_df = None
    regime_selected_df = None
    regime_weights_df = None

    stability_df = None
    stability_weights_df = None

    clustering_result = None
    cluster_df = None
    cluster_weights_df = None
    clustered_features_df = None

    decay_df = None
    decay_weights_df = None

    master_ic_df = None

    feature_table = None
    category_statistics = None
    category_budgets = None
    final_feature_weights = None

    # ========================================================
    # PHASE 1A — GLOBAL INFORMATION COEFFICIENT (IC)
    # ========================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 1A : GLOBAL IC ANALYSIS")
    logger.info("=" * 80)

    # --------------------------------------------------------
    # Step 1 : Daily Information Coefficient
    # --------------------------------------------------------

    logger.info("Computing Daily IC...")

    daily_ic_df = compute_daily_ic(
        df=df,
        features=features,
        target_col=target_col,
    )

    logger.info(
        "Daily IC observations : %d",
        len(daily_ic_df),
    )

    # --------------------------------------------------------
    # Step 2 : IC Summary
    # --------------------------------------------------------

    logger.info("Building IC summary...")

    summary_df = compute_ic_summary(
        daily_ic_df
    )

    logger.info(
        "Features evaluated : %d",
        len(summary_df),
    )

    # --------------------------------------------------------
    # Step 3 : Global Feature Selection
    # --------------------------------------------------------

    logger.info("Selecting production features...")

    (
        selected_features_df,
        removed_features,
    ) = filter_ic_features(
        df=df,
        summary_df=summary_df,
        min_icir=GLOBAL_MIN_ICIR,
        min_positive_pct=GLOBAL_MIN_POSITIVE_PCT,
        min_observations=GLOBAL_MIN_OBSERVATIONS,
        corr_threshold=GLOBAL_CORR_THRESHOLD,
        top_k=GLOBAL_TOP_K,
    )

    logger.info(
        "Selected features : %d",
        len(selected_features_df),
    )

    logger.info(
        "Removed features  : %d",
        len(removed_features),
    )

    # --------------------------------------------------------
    # Step 4 : Global IC Weights
    # --------------------------------------------------------

    logger.info("Computing global IC weights...")

    ic_weights = build_ic_weights(
        selected_features_df
    )

    logger.info(
        "Global IC weights created for %d features.",
        len(ic_weights),
    )

    # ========================================================
    # PHASE 1B — ROLLING IC ANALYSIS
    # ========================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 1B : ROLLING IC ANALYSIS")
    logger.info("=" * 80)

    # --------------------------------------------------------
    # Step 1 : Rolling IC Computation
    # --------------------------------------------------------

    logger.info("Computing Rolling IC...")

    rolling_ic_df = compute_rolling_ic(
        daily_ic_df=daily_ic_df,
    )

    logger.info(
        "Rolling IC observations : %d",
        len(rolling_ic_df),
    )

    # --------------------------------------------------------
    # Step 2 : Rolling Feature Selection
    # --------------------------------------------------------

    logger.info("Selecting rolling features...")

    rolling_selected_df = filter_rolling_ic_features(
        rolling_ic_df=rolling_ic_df,
        selected_features=selected_features_df["Feature"].tolist(),
        regime=current_regime,
        min_rolling_ic=ROLLING_MIN_IC,
        min_rolling_icir=ROLLING_MIN_ICIR,
        top_k=ROLLING_TOP_K,
    )

    logger.info(
        "Rolling-selected features : %d",
        len(rolling_selected_df),
    )

    # --------------------------------------------------------
    # Step 3 : Rolling Weights
    # --------------------------------------------------------

    logger.info("Computing rolling weights...")

    rolling_weights_df = build_rolling_weights(
        rolling_ic_df=rolling_ic_df,
        selected_features_df=rolling_selected_df,
    )

    logger.info(
        "Rolling weights generated."
    )

    # ========================================================
    # PHASE 1C — REGIME IC ANALYSIS
    # ========================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 1C : REGIME IC ANALYSIS")
    logger.info("=" * 80)

    # --------------------------------------------------------
    # Step 1 : Regime IC
    # --------------------------------------------------------

    logger.info("Computing regime IC...")

    regime_ic_df = compute_regime_ic(
        daily_ic_df=daily_ic_df,
    )

    logger.info(
        "Regime IC observations : %d",
        len(regime_ic_df),
    )

    # --------------------------------------------------------
    # Step 2 : Regime Feature Selection
    # --------------------------------------------------------

    logger.info("Selecting regime features...")

    regime_selected_df = filter_regime_ic_features(
        regime_ic_df=regime_ic_df,
        selected_features=selected_features_df["Feature"].tolist(),
        regime=current_regime,
        min_icir=REGIME_MIN_ICIR,
        min_positive_pct=REGIME_MIN_POSITIVE_PCT,
        min_observations=REGIME_MIN_OBSERVATIONS,
        top_k=REGIME_TOP_K,
    )

    logger.info(
        "Regime-selected features : %d",
        len(regime_selected_df),
    )

    # --------------------------------------------------------
    # Step 3 : Regime Weights
    # --------------------------------------------------------

    logger.info("Computing regime weights...")

    regime_weights_df = build_regime_weights(
        regime_ic_df=regime_ic_df,
        selected_features_df=regime_selected_df,
        current_regime=current_regime,
    )


    logger.info(
        "Regime weights generated."
    )

    logger.info("")
    logger.info(
        "Phase 1 completed successfully."
    )

    # ========================================================
    # PHASE 2 — IC STABILITY ENGINE
    # ========================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 2 : IC STABILITY ENGINE")
    logger.info("=" * 80)

    # --------------------------------------------------------
    # Step 1 : Stability Metrics
    # --------------------------------------------------------

    logger.info("Computing IC stability metrics...")

    stability_df = compute_ic_stability(
        daily_ic_df=daily_ic_df,
        rolling_ic_df=rolling_ic_df,
    )

    logger.info(
        "Stability metrics computed for %d features.",
        len(stability_df),
    )

    # --------------------------------------------------------
    # Step 2 : Stability Weights
    # --------------------------------------------------------

    logger.info("Building stability weights...")

    stability_weights_df = build_stability_weights(
        stability_df=stability_df,
        selected_features_df=selected_features_df,
    )

    logger.info(
        "Stability weights generated for %d features.",
        len(stability_weights_df),
    )

    # ========================================================
    # PHASE 3 — FEATURE CLUSTERING
    # ========================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 3 : FEATURE CLUSTERING")
    logger.info("=" * 80)

    # --------------------------------------------------------
    # Step 1 : Diversification Engine
    # --------------------------------------------------------

    logger.info("Running feature clustering...")

    clustering_result = diversify_features(
        df=df,
        selected_features_df=selected_features_df,
        feature_weights=ic_weights,
        cluster_method=CLUSTER_METHOD,
        correlation_method=CLUSTER_DISTANCE,
        corr_threshold=CLUSTER_CORR_THRESHOLD,
    )

    logger.info("Feature clustering completed.")

    # --------------------------------------------------------
    # Step 2 : Extract Outputs
    # --------------------------------------------------------

    cluster_df = clustering_result["clusters"]

    cluster_weights_df = clustering_result["cluster_weights"]

    clustered_features_df = clustering_result["selected"]

    logger.info(
        "Clusters discovered : %d",
        cluster_df["Cluster_ID"].nunique()
        if not cluster_df.empty
        else 0,
    )

    logger.info(
        "Diversified features retained : %d",
        len(clustered_features_df),
    )

    logger.info("")
    logger.info(
        "Phase 2 & Phase 3 completed successfully."
    )

    # ========================================================
    # PHASE 4 — FEATURE DECAY ENGINE
    # ========================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 4 : FEATURE DECAY ENGINE")
    logger.info("=" * 80)

    # --------------------------------------------------------
    # Step 1 : Compute Feature Decay
    # --------------------------------------------------------

    logger.info("Computing feature decay...")

    decay_df = compute_feature_decay(
        daily_ic_df=daily_ic_df,
        rolling_ic_df=rolling_ic_df,
    )

    logger.info("Decay columns:")
    logger.info(decay_df.columns.tolist())

    logger.info(
        "Decay metrics computed for %d features.",
        len(decay_df),
    )

    # --------------------------------------------------------
    # Step 2 : Build Decay Weights
    # --------------------------------------------------------

    logger.info("Building decay weights...")

    decay_weights_df = build_decay_weights(
        decay_df=decay_df,
        selected_features_df=clustered_features_df,
    )

    logger.info(
        "Decay weights generated for %d features.",
        len(decay_weights_df),
    )

    # ========================================================
    # PHASE 5 — MASTER IC ENGINE
    # ========================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 5 : MASTER IC ENGINE")
    logger.info("=" * 80)

    # --------------------------------------------------------
    # Step 1 : Build Master IC Table
    # --------------------------------------------------------

    logger.info("Building master feature table...")

    master_ic_df = build_master_ic_table(
        summary_df=summary_df,
        ic_weights=ic_weights,

        rolling_selected_df=rolling_selected_df,
        rolling_weights_df=rolling_weights_df,

        regime_selected_df=regime_selected_df,
        regime_weights_df=regime_weights_df,

        clustering_result=clustering_result,

        decay_df=decay_df,
        decay_weights_df=decay_weights_df,

        stability_weights_df=stability_weights_df,
    )


    logger.info(
        "Master table created with %d features.",
        len(master_ic_df),
    )

    # --------------------------------------------------------
    # Step 2 : Adaptive IC Weight Engine
    # --------------------------------------------------------

    logger.info("Building adaptive IC weights...")

    master_ic_df = build_adaptive_ic_weights(
        master_ic_df
    )

    logger.info(
        "Adaptive weighting completed."
    )

    logger.info("")
    logger.info(
        "Phase 4 & Phase 5 completed successfully."
    )

    # ========================================================
    # PHASE 6 — FEATURE CATEGORY ALLOCATION
    # ========================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 6 : FEATURE CATEGORY ALLOCATION")
    logger.info("=" * 80)

    # --------------------------------------------------------
    # Step 1 : Feature Category Mapping
    # --------------------------------------------------------

    logger.info("Building feature-category table...")

    feature_table = build_feature_category_table(
        master_ic_df,
        FEATURE_METADATA,
    )

    logger.info(
        "Feature-category table created with %d rows.",
        len(feature_table),
    )

    # --------------------------------------------------------
    # Step 2 : Category Statistics
    # --------------------------------------------------------

    logger.info("Computing category statistics...")

    category_statistics = compute_category_statistics(
        feature_table
    )

    logger.info(
        "Computed statistics for %d categories.",
        len(category_statistics),
    )

    # --------------------------------------------------------
    # Step 3 : Dynamic Category Budgets
    # --------------------------------------------------------

    logger.info("Allocating dynamic category budgets...")

    category_budgets = compute_dynamic_category_budgets(
        category_statistics
    )

    logger.info(
        "Generated budgets for %d categories.",
        len(category_budgets),
    )

    # --------------------------------------------------------
    # Step 4 : Final Feature Allocation
    # --------------------------------------------------------

    logger.info("Computing final feature weights...")

    final_feature_weights = allocate_feature_budgets(
        feature_table=feature_table,
        category_budget_df=category_budgets,
    )


    logger.info(
        "Final feature weights generated for %d features.",
        len(final_feature_weights),
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("ALPHA PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    # ========================================================
    # RETURN OBJECT
    # ========================================================

    return {

        # ----------------------------------------------------
        # Data Tables
        # ----------------------------------------------------

        "tables": {
            "daily": daily_ic_df,
            "summary": summary_df,
            "rolling": rolling_ic_df,
            "regime": regime_ic_df,
            "stability": stability_df,
            "stability_weights": stability_weights_df,
            "clusters": cluster_df,
            "cluster_weights": cluster_weights_df,
            "decay": decay_df,
            "decay_weights": decay_weights_df,
            "master": master_ic_df,
            "feature_table": feature_table,
            "category_statistics": category_statistics,
            "category_budgets": category_budgets,
        },

        # ----------------------------------------------------
        # Feature Sets
        # ----------------------------------------------------

        "features": {
            "selected": clustered_features_df,
            "rolling": rolling_selected_df,
            "regime": regime_selected_df,
            "removed": removed_features,
        },

        # ----------------------------------------------------
        # Weight Engines
        # ----------------------------------------------------

        "weights": {
            "global": ic_weights,
            "rolling": rolling_weights_df,
            "stability": stability_weights_df,
            "regime": regime_weights_df,
            "cluster": cluster_weights_df,
            "decay": decay_weights_df,
        },

        # ----------------------------------------------------
        # Final Production Weights
        # ----------------------------------------------------

        "final_feature_weights": final_feature_weights,

        # ----------------------------------------------------
        # Pipeline Metadata
        # ----------------------------------------------------

        "meta": {
            "current_regime": current_regime,
            "features_evaluated": len(summary_df),
            "features_selected": len(clustered_features_df),
            "clusters": (
                cluster_df["Cluster_ID"].nunique()
                if not cluster_df.empty
                else 0
            ),
            "categories": len(category_statistics),
        },
    }

# ============================================================
# SAVE ALPHA ENGINE RESULTS
# ============================================================

def save_ic_results(
    ic_results: dict,
    save_path: str = "data",
    save_debug_files: bool = True,
) -> None:
    """
    Save all Alpha Engine outputs to CSV.

    Parameters
    ----------
    ic_results : dict
        Output of run_alpha_pipeline().

    save_path : str, default="data"
        Output directory.

    save_debug_files : bool, default=True
        Whether to save removed-feature diagnostics.
    """

    os.makedirs(save_path, exist_ok=True)

    tables = ic_results.get("tables", {})
    features = ic_results.get("features", {})
    weights = ic_results.get("weights", {})

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def save_dataframe(
        dataframe: pd.DataFrame | None,
        filename: str,
    ) -> None:

        if dataframe is None:
            return

        if dataframe.empty:
            return

        dataframe.to_csv(
            os.path.join(save_path, filename),
            index=False,
        )

    # --------------------------------------------------------
    # Core Alpha Tables
    # --------------------------------------------------------

    table_files = {
        "daily": "daily_ic.csv",
        "summary": "ic_summary.csv",
        "rolling": "rolling_ic.csv",
        "regime": "regime_ic.csv",
        "stability": "ic_stability.csv",
        "clusters": "feature_clusters.csv",
        "cluster_weights": "cluster_weights.csv",
        "decay": "feature_decay.csv",
        "decay_weights": "decay_weights.csv",
        "feature_table": "feature_table.csv",
        "category_statistics": "category_statistics.csv",
        "category_budgets": "category_budgets.csv",
        "master": "master_ic_table.csv",
    }

    for key, filename in table_files.items():
        save_dataframe(
            tables.get(key),
            filename,
        )

    # --------------------------------------------------------
    # Weight Tables
    # --------------------------------------------------------

    weight_files = {
        "rolling": "rolling_weights.csv",
        "regime": "regime_weights.csv",
        "cluster": "cluster_weights.csv",
        "decay": "decay_weights.csv",
        "stability": "stability_weights.csv",
    }

    for key, filename in weight_files.items():
        save_dataframe(
            weights.get(key),
            filename,
        )

    # --------------------------------------------------------
    # Feature Selection Tables
    # --------------------------------------------------------

    feature_files = {
        "selected": "selected_features.csv",
        "rolling": "rolling_selected_features.csv",
        "regime": "regime_selected_features.csv",
    }

    for key, filename in feature_files.items():
        save_dataframe(
            features.get(key),
            filename,
        )

    # --------------------------------------------------------
    # Master Table Required
    # --------------------------------------------------------

    master_df = tables.get("master")

    if master_df is None or master_df.empty:

        logger.warning(
            "Master IC table is empty. Nothing further to save."
        )
        return

    # --------------------------------------------------------
    # Adaptive Feature Weights
    # --------------------------------------------------------

    adaptive_weights_df = (
        master_df[
            [
                "Feature",
                "Adaptive_Weight",
                "Feature_Rank",
            ]
        ]
        .rename(
            columns={
                "Adaptive_Weight": "Weight",
            }
        )
        .sort_values(
            "Weight",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    save_dataframe(
        adaptive_weights_df,
        "adaptive_ic_weights.csv",
    )

    # --------------------------------------------------------
    # Final Selected Features
    # --------------------------------------------------------

    selected_features_df = (
        master_df[
            master_df["Use_Feature"].fillna(False)
        ]
        .sort_values(
            "Feature_Rank",
        )
        .reset_index(drop=True)
    )

    save_dataframe(
        selected_features_df,
        "selected_ic_features.csv",
    )

    # --------------------------------------------------------
    # Diversified Features
    # --------------------------------------------------------

    diversified_columns = [
        "Feature",
        "Adaptive_Weight",
        "Feature_Rank",
        "Cluster_ID",
        "Cluster_Size",
        "Representative",
    ]

    diversified_columns = [
        column
        for column in diversified_columns
        if column in selected_features_df.columns
    ]

    diversified_df = (
        selected_features_df[
            diversified_columns
        ]
        .copy()
    )

    save_dataframe(
        diversified_df,
        "diversified_features.csv",
    )

    # --------------------------------------------------------
    # Removed Features
    # --------------------------------------------------------

    if save_debug_files:

        removed_features_df = (
            master_df[
                ~master_df["Use_Feature"].fillna(False)
            ]
            .copy()
        )

        save_dataframe(
            removed_features_df,
            "removed_ic_features.csv",
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    logger.info(
        "Alpha Engine results saved to '%s'.",
        save_path,
    )

# ======================
# Final execution order
# ======================

# Daily IC
#       ↓
# Summary
#       ↓
# Feature Filter
#       ↓
# IC Weights
#       ↓
# Rolling IC
#       ↓
# Rolling Weights
#       ↓
# Regime IC
#       ↓
# Regime Weights
#       ↓
# IC Stability
#       ↓
# Stability Weights
#       ↓
# Feature Clustering
#       ↓
# Feature Decay
#       ↓
# Decay Weights
#       ↓
# Master Table
#       ↓
# Adaptive Weights
#       ↓
# Feature Category Table
#       ↓
# Category Statistics
#       ↓
# Category Budgets
#       ↓
# Final Feature Weights