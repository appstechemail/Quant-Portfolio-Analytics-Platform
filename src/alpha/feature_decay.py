


"""
EXECUTION FLOW
========================

Feature A
↓
Daily IC history
↓
Historical IC
↓
Recent IC
↓
IC Delta
↓
Historical ICIR
↓
Recent ICIR
↓
ICIR Delta
↓
Positive %
↓
Rolling Trend


============================================================
FEATURE DECAY ENGINE
============================================================
feature_decay.py

1. Imports

2. Configuration

3. Validation

4. Helper Statistics
   • compute_trend()
   • recent_mean()
   • recent_std()
   • compute_recent_icir()
   • compute_recent_positive_pct()

5. Feature Decay Engine
   • compute_feature_decay()

6. Normalization Engine
   • normalize_decay_metrics()

7. Composite Score
   • compute_decay_score()

8. Weight Engine
   • build_decay_weights()

9. Diagnostics
   • print_decay_summary()

   

===============
PURPOSE:
===============

The Feature Decay Engine measures whether a predictive feature is improving or deteriorating over time.
Static IC measures answer:
    Was this feature historically useful?

Feature Decay answers:
    Is this feature still useful today?

This allows the Alpha Engine to reduce the influence of features whose predictive power is fading 
 while rewarding features whose predictive power is strengthening.

 
======================
 Pipeline Diagram
======================

 Daily IC
     │
     │
     ▼
Historical IC
Historical ICIR
     │
     │
     ▼
Recent IC
Recent ICIR
Recent Positive %
Rolling Trend
     │
     ▼
Feature Decay Table
     │
     ▼
Normalize
     │
     ▼
Decay Score
     │
     ▼
Decay Multiplier
     │
     ▼
Decay Weight
     │
     ▼
Master IC Table
============================================================
"""



from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.stats import linregress
from config.config import CONFIG

import logging
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
EPSILON = 1e-9
DEFAULT_RECENT_WINDOW = 60

MIN_TREND_OBSERVATIONS = 10

MIN_DECAY_OBSERVATIONS = 20


IC_CONFIG_DECAY = CONFIG["IC_CONFIG"]["DECAY"]
DECAY_MIN_MULTIPLIER = IC_CONFIG_DECAY.get("DECAY_MIN_MULTIPLIER", 0.80)
DECAY_MAX_MULTIPLIER = IC_CONFIG_DECAY.get("DECAY_MAX_MULTIPLIER", 1.20)


# ============================================================
# VALIDATION
# ============================================================

def validate_decay_inputs(
    daily_ic_df: pd.DataFrame,
    rolling_ic_df: pd.DataFrame,
) -> None:
    """
    Validate Feature Decay inputs.
    """

    if daily_ic_df is None or daily_ic_df.empty:
        raise ValueError(
            "daily_ic_df is empty."
        )

    if rolling_ic_df is None or rolling_ic_df.empty:
        raise ValueError(
            "rolling_ic_df is empty."
        )

    required_daily = {

        "Date",

        "Feature",

        "IC",

    }

    missing = required_daily - set(
        daily_ic_df.columns
    )

    if missing:

        raise ValueError(

            "daily_ic_df missing columns: "

            + ", ".join(sorted(missing))

        )

    required_rolling = {

        "Feature",

        "Rolling_IC",

    }

    missing = required_rolling - set(
        rolling_ic_df.columns
    )

    if missing:

        raise ValueError(

            "rolling_ic_df missing columns: "

            + ", ".join(sorted(missing))

        )


# ============================================================
# SAFE TREND REGRESSION
# ============================================================

def compute_trend(
    values: pd.Series,
) -> float:
    """
    Compute linear trend (slope).

    Positive slope

        -> improving feature

    Negative slope

        -> decaying feature
    """

    values = (
        pd.Series(values)
        .dropna()
    )

    if len(values) < MIN_TREND_OBSERVATIONS:
        return np.nan

    x = np.arange(len(values))

    slope, _, _, _, _ = linregress(
        x,
        values,
    )

    return float(slope)


# ============================================================
# RECENT WINDOW MEAN
# ============================================================

def recent_mean(
    series: pd.Series,
    window: int = DEFAULT_RECENT_WINDOW,
) -> float:
    """
    Mean over the most recent observations.
    """

    series = (
        pd.Series(series)
        .dropna()
    )

    if series.empty:
        return np.nan

    return float(
        series.tail(window).mean()
    )


# ============================================================
# RECENT WINDOW STANDARD DEVIATION
# ============================================================

def recent_std(
    series: pd.Series,
    window: int = DEFAULT_RECENT_WINDOW,
) -> float:
    """
    Standard deviation over recent observations.
    """

    series = (
        pd.Series(series)
        .dropna()
    )

    if series.empty:
        return np.nan

    return float(
        series.tail(window).std(ddof=0)
    )


# ============================================================
# RECENT ICIR
# ============================================================

def compute_recent_icir(
    ic_series: pd.Series,
    window: int = DEFAULT_RECENT_WINDOW,
) -> float:
    """
    Compute recent Information Coefficient Information Ratio.
    """

    recent = (

        pd.Series(ic_series)

        .dropna()

        .tail(window)

    )

    if len(recent) < MIN_TREND_OBSERVATIONS:
        return np.nan

    std = recent.std(ddof=0)

    if std < 1e-9:
        return np.nan

    return float(
        recent.mean() / std
    )


# ============================================================
# RECENT POSITIVE IC %
# ============================================================

def compute_recent_positive_pct(
    ic_series: pd.Series,
    window: int = DEFAULT_RECENT_WINDOW,
) -> float:
    """
    Percentage of positive IC values
    over the recent window.
    """

    recent = (

        pd.Series(ic_series)

        .dropna()

        .tail(window)

    )

    if recent.empty:
        return np.nan

    return float(
        (recent > 0).mean()
    )

# ============================================================
# FEATURE DECAY ENGINE
# ============================================================

def compute_feature_decay(
    daily_ic_df: pd.DataFrame,
    rolling_ic_df: pd.DataFrame,
    recent_window: int = DEFAULT_RECENT_WINDOW,
) -> pd.DataFrame:
    """
    Compute feature-level decay statistics.

    The decay engine compares recent IC behaviour against the
    long-term historical behaviour of each feature.

    Parameters
    ----------
    daily_ic_df : pd.DataFrame
        Daily IC table.

    rolling_ic_df : pd.DataFrame
        Rolling IC table.

    recent_window : int
        Number of most recent observations used for
        recent statistics.

    Returns
    -------
    pd.DataFrame

    Columns
    -------
    Feature
    Historical_IC
    Recent_IC
    IC_Delta
    Historical_ICIR
    Recent_ICIR
    ICIR_Delta
    Recent_Positive_Pct
    IC_Trend
    """

    validate_decay_inputs(
        daily_ic_df=daily_ic_df,
        rolling_ic_df=rolling_ic_df,
    )

    if daily_ic_df.empty:

        logger.warning(
            "Daily IC dataframe is empty."
        )

        return pd.DataFrame()

    rolling_groups = (
        rolling_ic_df.groupby("Feature")
        if not rolling_ic_df.empty
        else {}
    )

    decay_records = []

    # ========================================================
    # Feature Loop
    # ========================================================

    for feature_name, feature_df in (
        daily_ic_df.groupby("Feature")
    ):

        feature_df = (
            feature_df
            .sort_values("Date")
            .reset_index(drop=True)
        )

        ic_series = (
            feature_df["IC"]
            .dropna()
        )

        if len(ic_series) < MIN_DECAY_OBSERVATIONS:
            continue

        # ====================================================
        # Historical Statistics
        # ====================================================

        historical_ic = ic_series.mean()

        historical_ic_std = (
            ic_series.std(ddof=0)
        )

        historical_icir = (
            historical_ic
            /
            (historical_ic_std + EPSILON)
        )

        # ====================================================
        # Recent Statistics
        # ====================================================

        recent_ic = recent_mean(
            ic_series,
            recent_window,
        )

        recent_icir = compute_recent_icir(
            ic_series,
            recent_window,
        )

        recent_positive_pct = (
            compute_recent_positive_pct(
                ic_series,
                recent_window,
            )
        )

        # ====================================================
        # Decay Measures
        # ====================================================

        ic_delta = (
            recent_ic - historical_ic
            if (
                pd.notna(recent_ic)
                and
                pd.notna(historical_ic)
            )
            else np.nan
        )

        icir_delta = (
            recent_icir - historical_icir
            if (
                pd.notna(recent_icir)
                and
                pd.notna(historical_icir)
            )
            else np.nan
        )

        # ====================================================
        # Rolling Trend
        # ====================================================

        if (
            isinstance(
                rolling_groups,
                dict,
            )
            or
            feature_name not in rolling_groups.groups
        ):

            ic_trend = np.nan

        else:

            rolling_feature_df = (
                rolling_groups.get_group(
                    feature_name
                )
            )

            ic_trend = compute_trend(
                rolling_feature_df[
                    "Rolling_IC"
                ]
            )

        # ====================================================
        # Store
        # ====================================================

        decay_records.append({

            "Feature":
                feature_name,

            "Historical_IC":
                historical_ic,

            "Recent_IC":
                recent_ic,

            "IC_Delta":
                ic_delta,

            "Historical_ICIR":
                historical_icir,

            "Recent_ICIR":
                recent_icir,

            "ICIR_Delta":
                icir_delta,

            "Recent_Positive_Pct":
                recent_positive_pct,

            "IC_Trend":
                ic_trend,

        })

    # ========================================================
    # Output
    # ========================================================

    decay_df = pd.DataFrame(decay_records)

    if decay_df.empty:

        logger.warning(
            "No decay statistics computed."
        )

        return decay_df

    decay_df = (
        decay_df
        .sort_values("Feature")
        .reset_index(drop=True)
    )

    logger.info(
        "Computed decay statistics for %d features.",
        len(decay_df),
    )

    return decay_df
# ============================================================
# NORMALIZE DECAY METRICS
# ============================================================

def normalize_decay_metrics(
    decay_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize Feature Decay metrics to [0, 1].

    Higher normalized values always indicate
    stronger, more stable features.

    Parameters
    ----------
    decay_df

    Returns
    -------
    pd.DataFrame
    """

    if decay_df.empty:
        return decay_df.copy()

    df = decay_df.copy()

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def normalize(series: pd.Series) -> pd.Series:

        series = series.copy()

        median = series.median()

        if pd.isna(median):
            median = 0.0

        series = series.fillna(median)

        spread = series.max() - series.min()

        if spread < 1e-9:

            return pd.Series(
                1.0,
                index=series.index,
            )

        return (
            series - series.min()
        ) / spread

    # --------------------------------------------------------
    # Higher is Better
    # --------------------------------------------------------

    df["IC_Delta_Norm"] = normalize(
        df["IC_Delta"]
    )

    df["ICIR_Delta_Norm"] = normalize(
        df["ICIR_Delta"]
    )

    df["Positive_Norm"] = normalize(
        df["Recent_Positive_Pct"]
    )

    df["Trend_Norm"] = normalize(
        df["IC_Trend"]
    )

    return df


# ============================================================
# COMPUTE COMPOSITE DECAY SCORE
# ============================================================

def compute_decay_score(
    decay_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the production Feature Decay Score.

    Higher score
        -> feature remains robust

    Lower score
        -> feature is decaying

    Parameters
    ----------
    decay_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    if decay_df.empty:

        logger.warning(
            "Decay dataframe is empty."
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Normalize Metrics
    # --------------------------------------------------------

    df = normalize_decay_metrics(
        decay_df
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    required_columns = {

        "Feature",

        "Historical_IC",
        "Recent_IC",

        "Historical_ICIR",
        "Recent_ICIR",

        "IC_Delta_Norm",
        "ICIR_Delta_Norm",

        "Positive_Norm",
        "Trend_Norm",

    }

    missing = required_columns.difference(
        df.columns
    )

    if missing:

        raise ValueError(
            f"normalize_decay_metrics() missing columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Composite Decay Score
    # --------------------------------------------------------

    df["Decay_Score"] = (

        0.40 * df["IC_Delta_Norm"]

        +

        0.30 * df["ICIR_Delta_Norm"]

        +

        0.15 * df["Positive_Norm"]

        +

        0.15 * df["Trend_Norm"]

    )

    df["Decay_Score"] = (

        df["Decay_Score"]

        .clip(
            lower=0.0,
            upper=1.0,
        )

    )

    # --------------------------------------------------------
    # Final Ordering
    # --------------------------------------------------------

    df = (

        df

        .sort_values(
            by="Decay_Score",
            ascending=False,
        )

        .reset_index(drop=True)

    )

    logger.info(
        "Computed decay scores for %d features.",
        len(df),
    )

    return df

# ============================================================
# BUILD DECAY WEIGHTS
# ============================================================

def build_decay_weights(
    decay_df: pd.DataFrame,
    selected_features_df: pd.DataFrame,
    min_multiplier: float = DECAY_MIN_MULTIPLIER,
    max_multiplier: float = DECAY_MAX_MULTIPLIER,
) -> pd.DataFrame:
    """
    Convert feature decay scores into production-ready
    decay multipliers and normalized decay weights.

    Parameters
    ----------
    decay_df : pd.DataFrame
        Output from compute_feature_decay().

    selected_features_df : pd.DataFrame
        Output from filter_ic_features().

    min_multiplier : float
        Minimum allowed decay multiplier.

    max_multiplier : float
        Maximum allowed decay multiplier.

    Returns
    -------
    pd.DataFrame

    Columns
    -------
    Feature
    Historical_IC
    Recent_IC
    Historical_ICIR
    Recent_ICIR
    Decay_Score
    Decay_Multiplier
    Decay_Weight
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if decay_df.empty:
        logger.warning(
            "Decay dataframe is empty."
        )

        return pd.DataFrame()

    if selected_features_df.empty:
        logger.warning(
            "Selected feature dataframe is empty."
        )

        return pd.DataFrame()

    if "Feature" not in selected_features_df.columns:
        raise ValueError(
            "'Feature' column missing from selected_features_df."
        )

    # --------------------------------------------------------
    # Compute Decay Score
    # --------------------------------------------------------

    decay_weights_df = compute_decay_score(
        decay_df
    )

    if decay_weights_df.empty:

        logger.warning(
            "No decay scores were produced."
        )

        return pd.DataFrame()

    required_columns = {
        "Feature",
        "Decay_Score",
    }

    missing = required_columns.difference(
        decay_weights_df.columns
    )

    if missing:
        raise ValueError(
            f"compute_decay_score() missing columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Keep Only Production Features
    # --------------------------------------------------------

    decay_weights_df = (
        selected_features_df[
            ["Feature"]
        ]
        .merge(
            decay_weights_df,
            on="Feature",
            how="left",
        )
    )

    # --------------------------------------------------------
    # Fill Missing Values
    # --------------------------------------------------------

    metric_columns = [

        "Historical_IC",
        "Recent_IC",
        "Historical_ICIR",
        "Recent_ICIR",

    ]

    for column in metric_columns:

        if column in decay_weights_df.columns:
            decay_weights_df[column] = (
                decay_weights_df[column]
                .fillna(0.0)
            )

    decay_weights_df["Decay_Score"] = (
        decay_weights_df["Decay_Score"]
        .fillna(0.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    # --------------------------------------------------------
    # Linear Score → Multiplier
    # --------------------------------------------------------

    decay_weights_df["Decay_Multiplier"] = (
        min_multiplier
        +
        (
            max_multiplier
            - min_multiplier
        )
        *
        decay_weights_df["Decay_Score"]
    )

    decay_weights_df["Decay_Multiplier"] = (
        decay_weights_df["Decay_Multiplier"]
        .clip(
            lower=min_multiplier,
            upper=max_multiplier,
        )
    )

    # --------------------------------------------------------
    # Normalized Decay Weight
    # --------------------------------------------------------

    total_multiplier = (
        decay_weights_df["Decay_Multiplier"]
        .sum()
    )

    if total_multiplier <= EPSILON:
        logger.warning(
            "Decay multipliers sum to zero."
        )

        decay_weights_df["Decay_Weight"] = 0.0
    else:
        decay_weights_df["Decay_Weight"] = (
            decay_weights_df["Decay_Multiplier"]
            /
            total_multiplier
        )

    # --------------------------------------------------------
    # Final Ordering
    # --------------------------------------------------------

    decay_weights_df = (
        decay_weights_df
        .sort_values(
            by=[
                "Decay_Weight",
                "Decay_Score",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Decay weights computed for %d features.",
        len(decay_weights_df),
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_columns = [
        "Feature",
        "Historical_IC",
        "Recent_IC",
        "Historical_ICIR",
        "Recent_ICIR",
        "Decay_Score",
        "Decay_Multiplier",
        "Decay_Weight",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in decay_weights_df.columns
    ]

    return decay_weights_df[
        output_columns
    ]


# ============================================================
# PRINT FEATURE DECAY SUMMARY
# ============================================================

def print_decay_summary(
    decay_df: pd.DataFrame,
) -> None:
    """
    Pretty-print Feature Decay summary.
    """

    if decay_df.empty:

        print("\nNo Feature Decay results available.")

        return

    columns = [

        "Feature",

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

    ]

    available_columns = [
        c
        for c in columns
        if c in decay_df.columns
    ]

    print()

    print("=" * 60)

    print("FEATURE DECAY ENGINE")

    print("=" * 60)

    print()

    print(

        decay_df[available_columns]

        .sort_values(
            "Decay_Weight",
            ascending=False,
        )

    )

    print()

    print(

        "Total Weight :",

        round(

            decay_df["Decay_Weight"].sum(),

            6,

        ),

    )

    print("=" * 60)


# ====================================
# Position inside the Alpha Engine
# ====================================

# Daily IC
#       │
#       ▼
# IC Summary
#       │
#       ▼
# Feature Selection
#       │
#       ▼
# Rolling IC
#       │
#       ▼
# Feature Clustering
#       │
#       ▼
# Feature Decay
#       │
#       ▼
# Regime IC
#       │
#       ▼
# IC Stability
#       │
#       ▼
# Master IC Table
#       │
#       ▼
# Adaptive IC Weight
#       │
#       ▼
# Dynamic Category Budget
#       │
#       ▼
# Portfolio Construction


# ===========================
# Why this engine exists
# ===========================

# Without Feature Decay:

# A feature that worked well two years ago could still receive a large allocation even if its predictive power has faded.

# With Feature Decay:

# Features whose IC is improving receive higher multipliers.
# Features whose IC is deteriorating are gradually down-weighted rather than removed abruptly.
# The adaptive weighting system becomes more responsive to changing market conditions.

# This makes the Alpha Engine better suited for live markets, where factor effectiveness changes over time.