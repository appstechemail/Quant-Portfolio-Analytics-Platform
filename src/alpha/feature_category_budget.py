
# ===================================
# feature_category_budget.py
# ===================================

# Configuration
# Validation
# Feature Metadata
#   build_feature_category_table
# Category Analytics
#   compute_category_statistics
# Budget Engine
#   _normalize
#   compute_dynamic_category_budgets
# Feature Allocation
#   allocate_feature_budgets
# Validation
#   validate_feature_weights
# Reporting
#   print_category_budget_summary
#   print_feature_weight_summary
# Pipeline
#   build_dynamic_category_budget


# ===================================
# FLOW OF feature_category_budget
# ===================================

# │
# ├── Configuration
# │
# ├── Validation
# │
# ├── Metadata builders
# │
# ├── Feature table
# │
# ├── Category statistics
# │
# ├── Dynamic category budgets
# │
# ├── Feature allocation
# │
# ├── Validation
# │
# ├── Reporting
# │
# └── Complete pipeline



"""
============================================================
FEATURE CATEGORY BUDGET ENGINE
============================================================

Purpose
-------
The Feature Category Budget Engine converts feature-level alpha quality 
into portfolio allocation weights while preventing one family of features from dominating the model.

Pipeline
--------
1. Validate inputs
2. Build metadata lookups
3. Attach category/group information
4. Build feature category table

This module does NOT allocate budgets.
Budget allocation begins in Part 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPSILON = 1e-12


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CategoryBudgetConfig:
    """
    Configuration for the Dynamic Category Budget Engine.
    """

    min_budget: float = 0.05
    max_budget: float = 0.35

    # Controls concentration inside each category.
    concentration_power: float = 1.50


CONFIG = CategoryBudgetConfig()


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_inputs(
    feature_df: pd.DataFrame,
    feature_metadata: dict,
) -> pd.DataFrame:
    """
    Validate inputs before building category budgets.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature table produced by the IC pipeline.

    feature_metadata : dict
        FEATURE_METADATA dictionary.

    Returns
    -------
    pd.DataFrame
        Copy of validated dataframe.
    """

    if feature_df is None:
        raise ValueError("feature_df is None.")

    if feature_df.empty:
        raise ValueError("feature_df is empty.")

    if feature_metadata is None:
        raise ValueError("feature_metadata is None.")

    required_columns = [
        "Feature",
        "Decay_Multiplier",
        "Decay_Score",
        "Historical_IC",
        "Recent_IC",
        "Historical_ICIR",
        "Recent_ICIR",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in feature_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    return feature_df.copy()


# ============================================================
# METADATA LOOKUPS
# ============================================================

def build_category_lookup(
    feature_metadata: dict,
) -> dict:
    """
    Build Feature → Category lookup.
    """

    return {
        feature: metadata.get("category", "Unknown")
        for feature, metadata in feature_metadata.items()
    }


def build_group_lookup(
    feature_metadata: dict,
) -> dict:
    """
    Build Feature → Group lookup.
    """

    return {
        feature: metadata.get("group", "Unknown")
        for feature, metadata in feature_metadata.items()
    }


# ============================================================
# ATTACH METADATA
# ============================================================

def attach_feature_metadata(
    feature_df: pd.DataFrame,
    category_lookup: dict,
    group_lookup: dict,
) -> pd.DataFrame:
    """
    Attach Category and Group columns.

    Returns
    -------
    pd.DataFrame
    """

    df = feature_df.copy()

    df["Category"] = (
        df["Feature"]
        .map(category_lookup)
        .fillna("Unknown")
    )

    df["Group"] = (
        df["Feature"]
        .map(group_lookup)
        .fillna("Unknown")
    )

    return df


# ============================================================
# REORDER COLUMNS
# ============================================================

def reorder_feature_columns(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Arrange columns into production order.
    """

    ordered_columns = [

        "Feature",

        "Category",
        "Group",

        "Decay_Multiplier",
        "Decay_Score",

        "Historical_IC",
        "Recent_IC",

        "Historical_ICIR",
        "Recent_ICIR",

    ]

    return feature_df[ordered_columns]


# ============================================================
# BUILD FEATURE CATEGORY TABLE
# ============================================================

def build_feature_category_table(
    feature_df: pd.DataFrame,
    feature_metadata: dict,
) -> pd.DataFrame:
    """
    Build feature-category table.

    Parameters
    ----------
    feature_df
        Output of the Feature Decay Engine.

    feature_metadata
        FEATURE_METADATA dictionary.

    Returns
    -------
    pd.DataFrame

    Columns
    -------
    Feature
    Category
    Group
    Decay_Multiplier
    Decay_Score
    Historical_IC
    Recent_IC
    Historical_ICIR
    Recent_ICIR
    """

    features = validate_inputs(
        feature_df,
        feature_metadata,
    )

    category_lookup = build_category_lookup(
        feature_metadata,
    )

    group_lookup = build_group_lookup(
        feature_metadata,
    )

    features = attach_feature_metadata(
        features,
        category_lookup,
        group_lookup,
    )

    features = reorder_feature_columns(
        features,
    )

    return features

# =====================
# CATEGORY STATISTICS
# =====================

def compute_category_statistics(
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute category-level statistics.

    Parameters
    ----------
    feature_table : DataFrame

    Returns
    -------
    DataFrame
    """

    if feature_table.empty:
        return pd.DataFrame()

    stats = (
        feature_table
        .groupby("Category", dropna=False)
        .agg(
            Num_Features=("Feature", "count"),

            Mean_Decay_Multiplier=("Decay_Multiplier", "mean"),
            Median_Decay_Multiplier=("Decay_Multiplier", "median"),
            Std_Decay_Multiplier=("Decay_Multiplier", lambda x: x.std(ddof=0)),

            Mean_Decay_Score=("Decay_Score", "mean"),

            Mean_Historical_IC=("Historical_IC", "mean"),
            Mean_Recent_IC=("Recent_IC", "mean"),

            Mean_Historical_ICIR=("Historical_ICIR", "mean"),
            Mean_Recent_ICIR=("Recent_ICIR", "mean"),
        )
        .reset_index()
    )

    numeric_cols = [
        "Mean_Decay_Multiplier",
        "Median_Decay_Multiplier",
        "Std_Decay_Multiplier",
        "Mean_Decay_Score",
        "Mean_Historical_IC",
        "Mean_Recent_IC",
        "Mean_Historical_ICIR",
        "Mean_Recent_ICIR",
    ]

    stats[numeric_cols] = stats[numeric_cols].fillna(0.0)

    stats["IC_Improvement"] = (
        stats["Mean_Recent_IC"]
        - stats["Mean_Historical_IC"]
    )

    stats["ICIR_Improvement"] = (
        stats["Mean_Recent_ICIR"]
        - stats["Mean_Historical_ICIR"]
    )

    stats["Category_Stability"] = (
        1.0 /
        (stats["Std_Decay_Multiplier"] + 1e-6)
    )

    return (
        stats
        .sort_values(
            "Mean_Decay_Multiplier",
            ascending=False,
        )
        .reset_index(drop=True)
    )

# ============================================================
# NORMALIZE CATEGORY METRICS
# ============================================================

def _normalize_series(
    series: pd.Series,
    fill_strategy: str = "median",
) -> pd.Series:
    """
    Min-max normalize a numeric Series to [0, 1].

    Parameters
    ----------
    series
        Numeric pandas Series.

    fill_strategy
        "median", "mean", or "zero".

    Returns
    -------
    pd.Series
        Normalized series.
    """

    s = series.copy()

    if fill_strategy == "median":
        s = s.fillna(s.median())

    elif fill_strategy == "mean":
        s = s.fillna(s.mean())

    else:
        s = s.fillna(0.0)

    spread = s.max() - s.min()
 
    if spread < EPSILON:
        return pd.Series(
            1.0,
            index=s.index,
            dtype=float,
        )

    return (s - s.min()) / spread


# =========================
# DYNAMIC CATEGORY BUDGETS
# =========================

def compute_dynamic_category_budgets(
    category_stats_df: pd.DataFrame,
    config: CategoryBudgetConfig = CONFIG,
) -> pd.DataFrame:
    """
    Convert category quality into portfolio budget.
    """

    if category_stats_df.empty:
        return pd.DataFrame()

    budgets = category_stats_df.copy()

    budgets["Decay_Norm"] = _normalize_series(
        budgets["Mean_Decay_Multiplier"]
    )

    budgets["DecayScore_Norm"] = _normalize_series(
        budgets["Mean_Decay_Score"]
    )

    budgets["IC_Norm"] = _normalize_series(
        budgets["Mean_Recent_IC"]
    )

    budgets["ICIR_Norm"] = _normalize_series(
        budgets["Mean_Recent_ICIR"]
    )

    budgets["Stability_Norm"] = _normalize_series(
        budgets["Category_Stability"]
    )

    budgets["Raw_Score"] = (
        0.50 * budgets["Decay_Norm"]
        + 0.20 * budgets["DecayScore_Norm"]
        + 0.15 * budgets["ICIR_Norm"]
        + 0.10 * budgets["IC_Norm"]
        + 0.05 * budgets["Stability_Norm"]
    )

    budgets["Raw_Score"] = (
        budgets["Raw_Score"]
        .clip(lower=0.0)
    )

    total_score = budgets["Raw_Score"].sum()

    if total_score <= 1e-12:

        budgets["Category_Budget"] = (
            1.0 / len(budgets)
        )

    else:

        budgets["Category_Budget"] = (
            budgets["Raw_Score"] / total_score
        )


    budgets["Category_Budget"] = (
        budgets["Category_Budget"]
        .clip(
            lower=config.min_budget,
            upper=config.max_budget,
        )
    )

    budgets["Category_Budget"] /= (
        budgets["Category_Budget"].sum()
    )

    budgets = (
        budgets
        .sort_values(
            "Category_Budget",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    budgets["Budget_Rank"] = np.arange(
        1,
        len(budgets) + 1,
    )

    return budgets


# ============================================================
# FEATURE BUDGET ALLOCATION
# ============================================================

def allocate_feature_budgets(
    feature_table: pd.DataFrame,
    category_budget_df: pd.DataFrame,
    config: CategoryBudgetConfig = CONFIG,
) -> pd.DataFrame:
    """
    Allocate category budgets to individual features.

    Parameters
    ----------
    feature_table
        Output of build_feature_category_table()

    category_budgets
        Output of compute_dynamic_category_budgets()

    Returns
    -------
    DataFrame
    """

    if feature_table.empty or category_budget_df.empty:
        return pd.DataFrame()

    features = feature_table.copy()

    budgets = category_budget_df[
        [
            "Category",
            "Category_Budget",
        ]
    ]

    # -------------------------------------------------------
    # Attach category budgets
    # -------------------------------------------------------

    features = features.merge(
        budgets,
        on="Category",
        how="left",
    )

    features["Category_Budget"] = (
        features["Category_Budget"]
        .fillna(0.0)
    )

    # -------------------------------------------------------
    # Feature strength
    # -------------------------------------------------------

    feature_strength = (
        features["Decay_Multiplier"]
        ** config.concentration_power
    )

    category_strength = (
        feature_strength
        .groupby(features["Category"])
        .transform("sum")
    )

    features["Category_Weight"] = (
        feature_strength
        /
        (category_strength + 1e-12)
    )

    # -------------------------------------------------------
    # Allocate budget
    # -------------------------------------------------------

    features["Final_Weight"] = (
        features["Category_Budget"]
        *
        features["Category_Weight"]
    )

    total = features["Final_Weight"].sum()

    if total > 0:
        features["Final_Weight"] /= total

    # -------------------------------------------------------
    # Ranking
    # -------------------------------------------------------

    features = (
        features
        .sort_values(
            "Final_Weight",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    features["Feature_Rank"] = np.arange(
        1,
        len(features) + 1,
    )

    return features


# ============================================================
# VALIDATION
# ============================================================

def validate_feature_weights(
    feature_weights_df: pd.DataFrame,
) -> bool:
    """
    Validate final feature weights.
    """

    if feature_weights_df.empty:
        raise ValueError(
            "Feature weight table is empty."
        )

    if (
        feature_weights_df["Final_Weight"] < 0
    ).any():
        raise ValueError(
            "Negative feature weights detected."
        )

    total = feature_weights_df[
        "Final_Weight"
    ].sum()

    if not np.isclose(
        total,
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"Feature weights sum to {total:.8f}"
        )

    return True


# ============================================================
# PRINT CATEGORY SUMMARY
# ============================================================

def print_category_budget_summary(
    category_budget_df: pd.DataFrame,
):
    """
    Pretty-print category budgets.
    """

    if category_budget_df.empty:
        return

    print()
    print("=" * 70)
    print("DYNAMIC CATEGORY BUDGETS")
    print("=" * 70)

    print(
        category_budget_df[
            [
                "Category",
                "Num_Features",
                "Category_Budget",
            ]
        ]
        .sort_values(
            "Category_Budget",
            ascending=False,
        )
    )

    print()

    print(
        "Total Budget:",
        round(
            category_budget_df[
                "Category_Budget"
            ].sum(),
            6,
        ),
    )

    print("=" * 70)


# ============================================================
# PRINT FEATURE WEIGHTS
# ============================================================

def print_feature_weight_summary(
    feature_weight_df: pd.DataFrame,
    top_n: int = 25,
):
    """
    Pretty-print feature weights.
    """

    if feature_weight_df.empty:
        return

    print()
    print("=" * 70)
    print("FINAL FEATURE WEIGHTS")
    print("=" * 70)

    print(
        feature_weight_df[
            [
                "Feature",
                "Category",
                "Decay_Multiplier",
                "Final_Weight",
            ]
        ].head(top_n)
    )

    print("=" * 70)


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def build_dynamic_category_budget(
    feature_df: pd.DataFrame,
    feature_metadata: dict,
    config: CategoryBudgetConfig = CONFIG,
):
    """
    Complete Dynamic Category Budget pipeline.

    Parameters
    ----------
    feature_df
        Output of Feature Decay Engine.

    feature_metadata
        FEATURE_METADATA dictionary.

    Returns
    -------
    dict
    """

    if feature_df.empty:
        raise ValueError(
            "Feature dataframe is empty."
        )

    # -------------------------------------------------------
    # Feature table
    # -------------------------------------------------------

    feature_table = build_feature_category_table(
        feature_df,
        feature_metadata,
    )

    # -------------------------------------------------------
    # Category statistics
    # -------------------------------------------------------

    category_statistics = compute_category_statistics(
        feature_table,
    )

    # -------------------------------------------------------
    # Category budgets
    # -------------------------------------------------------

    category_budgets = compute_dynamic_category_budgets(
        category_statistics,
        config=config,
    )

    # -------------------------------------------------------
    # Feature allocation
    # -------------------------------------------------------

    feature_weights = allocate_feature_budgets(
        feature_table,
        category_budgets,
        config=config,
    )

    validate_feature_weights(
        feature_weights,
    )

    return {

        "feature_table": feature_table,

        "category_statistics": category_statistics,

        "category_budgets": category_budgets,

        "feature_weights": feature_weights,

    }