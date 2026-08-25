"""
============================================================
FEATURE CLUSTERING / DIVERSIFICATION ENGINE
============================================================

Purpose
-------
feature_clustering.py is responsible for removing redundant features from the alpha feature universe.

Many engineered features capture the same underlying market behavior. For example:

SMA20 and EMA20
RSI14 and RSI21
MACD Histogram and MACD Signal
Momentum20 and Rate of Change

Even if each has a high Information Coefficient (IC), using all of them increases multicollinearity and reduces diversification. The clustering engine groups highly correlated features and keeps only the strongest representative from each cluster.

The output is a diversified feature universe that preserves predictive power while reducing redundancy.

Pipeline
--------

                    FEATURE CLUSTERING ENGINE

                           Selected Features
                                   │
                                   ▼
                     Validate Inputs & Feature List
                                   │
                                   ▼
                    Compute Feature Correlation Matrix
                                   │
                                   ▼
                       Convert Correlation → Distance
                                   │
                                   ▼
                     Hierarchical Feature Clustering
                                   │
                                   ▼
                        Compute Cluster Statistics
                                   │
                                   ▼
                 Select Best Representative per Cluster
                                   │
                                   ▼
                    Enrich Cluster Metadata
                                   │
                                   ▼
                     Compute Cluster-Level Weights
                                   │
                                   ▼
                Compute Diversification Multipliers
                                   │
                                   ▼
                  Build Diversified Feature Universe
                                   │
                                   ▼
                         Return Clustering Results

This engine becomes the foundation for

    • Feature Decay
    • Dynamic Category Budgets
    • Portfolio Construction

============================================================

==========================
feature_clustering.py
==========================

├── Imports
│
├── Configuration
│   └── ClusteringConfig
│
├── Validation
│   └── validate_inputs()
│
├── Correlation Engine
│   ├── compute_feature_correlation()
│   └── compute_distance_matrix()
│
├── Clustering Engine
│   ├── hierarchical_clustering()
│   └── build_clusters()
│
├── Cluster Metadata
│   ├── compute_cluster_statistics()
│   ├── mark_cluster_representatives()
│   ├── enrich_cluster_metadata()
│   └── build_cluster_summary()
│
├── Representative Selection
│   └── select_cluster_representatives()
│
├── Cluster Weight Engine
│   ├── compute_cluster_weights()
│   └── compute_diversification_multiplier()
│
├── Final Diversified Feature Set
│   └── build_diversified_feature_set()
│
├── Diagnostics
│   └── print_cluster_statistics()
│
└── Public Pipeline
    └── diversify_features()


==========================    
Execution Flow
==========================

diversify_features()
        │
        ▼
validate_inputs()
        │
        ▼
build_clusters()
        │
        ├────────► compute_feature_correlation()
        │
        ├────────► compute_distance_matrix()
        │
        └────────► hierarchical_clustering()
        │
        ▼
select_cluster_representatives()
        │
        ▼
enrich_cluster_metadata()
        │
        ├────────► compute_cluster_statistics()
        │
        └────────► mark_cluster_representatives()
        │
        ▼
compute_cluster_weights()
        │
        ▼
compute_diversification_multiplier()
        │
        ▼
build_diversified_feature_set()
        │
        ▼
return results

"""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import (
    linkage,
    fcluster,
)

from scipy.spatial.distance import (
    squareform,
)

from config.config import CONFIG

VALID_CORRELATION_METHODS = set(
    CONFIG["IC_CONFIG"]["VALID_CORRELATION_METHODS"]
)

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ClusteringConfig:
    """
    Configuration for feature clustering.
    """

    method: str = "hierarchical"
    correlation_method: str = "spearman"
    correlation_threshold: float = 0.80
    minimum_cluster_size: int = 2
    representatives_per_cluster: int = 1

# ============================================================
# VALIDATION
# ============================================================

def validate_inputs(
    df: pd.DataFrame,
    selected_features_df: pd.DataFrame,
) -> list[str]:
    """
    Validate clustering inputs.

    Returns
    -------
    list[str]
        Features available for clustering.
    """

    if selected_features_df.empty:
        raise ValueError(
            "selected_features_df is empty."
        )

    if "Feature" not in selected_features_df.columns:
        raise ValueError(
            "'Feature' column missing."
        )

    features = (
        selected_features_df["Feature"]
        .drop_duplicates()
        .tolist()
    )

    missing = [
        f
        for f in features
        if f not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing feature columns:\n"
            + "\n".join(sorted(missing))
        )

    return features


# ============================================================
# FEATURE CORRELATION
# ============================================================

def compute_feature_correlation(
    df: pd.DataFrame,
    features: list[str],
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Compute feature correlation matrix.

    Parameters
    ----------
    df
        Feature dataframe.

    features
        Feature columns.

    method
        Correlation method.

    Returns
    -------
    pd.DataFrame
        Feature correlation matrix.
    """

    method = method.lower().strip()

    if method not in VALID_CORRELATION_METHODS:
        raise ValueError(
            f"Unsupported correlation method '{method}'. "
            f"Choose one of "
            f"{sorted(VALID_CORRELATION_METHODS)}."
        )

    corr = (
        df[features]
        .corr(
            method=method,
            min_periods=100,
        )
        .clip(-1.0, 1.0)
        .fillna(0.0)
        .copy()
    )

    for feature in corr.columns:
        corr.loc[feature, feature] = 1.0

    
    return corr


# ============================================================
# DISTANCE MATRIX
# ============================================================

def compute_distance_matrix(
    correlation_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert correlation into clustering distance.

    Distance

        d = 1 - |corr|
    """

    distance = (
        1.0
        - correlation_matrix.abs()
    ).copy()

    for feature in distance.index:
        distance.loc[feature, feature] = 0.0

    
    return distance


# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================

def hierarchical_clustering(
    distance_matrix: pd.DataFrame,
    correlation_threshold: float,
) -> pd.DataFrame:
    """
    Average-linkage hierarchical clustering.

    Returns
    -------
    Feature
    Cluster_ID
    """

    condensed = squareform(
        distance_matrix.values,
        checks=False,
    )

    linkage_matrix = linkage(
        condensed,
        method="average",
    )

    cluster_ids = fcluster(
        linkage_matrix,
        t=1.0 - correlation_threshold,
        criterion="distance",
    )

    return pd.DataFrame(
        {
            "Feature": distance_matrix.index,
            "Cluster_ID": cluster_ids,
        }
    )


# ============================================================
# BUILD CLUSTERS
# ============================================================

def build_clusters(
    df: pd.DataFrame,
    selected_features_df: pd.DataFrame,
    config: ClusteringConfig,
) -> dict:
    """
    Complete clustering engine.

    Returns
    -------
    dict
        correlation
        distance
        clusters
    """

    features = validate_inputs(
        df,
        selected_features_df,
    )

    # ----------------------------------------
    # Single-feature shortcut
    # ----------------------------------------

    if len(features) == 1:

        correlation = pd.DataFrame(
            [[1.0]],
            index=features,
            columns=features,
        )

        distance = pd.DataFrame(
            [[0.0]],
            index=features,
            columns=features,
        )

        clusters = pd.DataFrame(
            {
                "Feature": features,
                "Cluster_ID": [1],
            }
        )

        return {
            "correlation": correlation,
            "distance": distance,
            "clusters": clusters,
        }

    # ----------------------------------------
    # Correlation
    # ----------------------------------------

    correlation = compute_feature_correlation(
        df,
        features,
        method=config.correlation_method,
    )

    distance = compute_distance_matrix(
        correlation,
    )

    clusters = hierarchical_clustering(
        distance,
        config.correlation_threshold,
    )

    return {
        "correlation": correlation,
        "distance": distance,
        "clusters": clusters,
    }


# ============================================================
# CLUSTER STATISTICS
# ============================================================

def compute_cluster_statistics(
    clusters_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute cluster-level statistics.

    Adds
    ----
    Cluster_Size
    """

    clusters = clusters_df.copy()

    clusters["Cluster_Size"] = (
        clusters
        .groupby("Cluster_ID")["Feature"]
        .transform("size")
    )

    return clusters


# ============================================================
# REPRESENTATIVE FLAGS
# ============================================================

def mark_cluster_representatives(
    clusters_df: pd.DataFrame,
    representative_features: list[str],
) -> pd.DataFrame:
    """
    Mark representative feature
    inside every cluster.
    """

    clusters = clusters_df.copy()

    representatives = set(
        representative_features
    )

    clusters["Representative"] = (
        clusters["Feature"].isin(representatives)
    )

    clusters["Dropped_By_Cluster"] = (
        ~clusters["Representative"]
    )

    return clusters


# ============================================================
# ENRICH CLUSTER METADATA
# ============================================================

def enrich_cluster_metadata(
    clusters_df: pd.DataFrame,
    representative_features: list[str],
) -> pd.DataFrame:
    """
    Complete cluster metadata pipeline.
    """

    clusters = compute_cluster_statistics(
        clusters_df
    )

    clusters = mark_cluster_representatives(
        clusters,
        representative_features,
    )

    return clusters


# ============================================================
# REPRESENTATIVE SELECTION
# ============================================================

def select_cluster_representatives(
    clusters_df: pd.DataFrame,
    selected_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select one representative feature
    from every cluster.

    Priority

        Adaptive_Weight

    otherwise

        Alpha_Score

    otherwise

        ICIR
    """

    merged = (
        selected_features_df
        .merge(
            clusters_df,
            on="Feature",
            how="left",
        )
    )

    # --------------------------------------------------
    # Ranking metric
    # --------------------------------------------------

    ranking_columns = [

        "Adaptive_Weight",

        "Alpha_Score",

        "ICIR",

        "Mean_IC",

    ]

    ranking_metric = None

    for column in ranking_columns:

        if (
            column in merged.columns
            and
            merged[column].notna().any()
        ):

            ranking_metric = column
            break

    if ranking_metric is None:

        raise ValueError(
            "No ranking column found "
            "for representative selection."
        )

    representatives = (

        merged

        .sort_values(
            ranking_metric,
            ascending=False,
        )

        .groupby(
            "Cluster_ID",
            as_index=False,
        )

        .head(1)

        .reset_index(drop=True)

    )

    return representatives


# ============================================================
# CLUSTER SUMMARY
# ============================================================

def build_cluster_summary(
    clusters_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    One-row summary per cluster.

    Returns
    -------
    Cluster_ID

    Cluster_Size

    Representative_Feature

    Num_Dropped
    """

    summary = (

        clusters_df

        .groupby(
            "Cluster_ID",
            as_index=False,
        )

        .apply(

            lambda g: pd.Series({

                "Cluster_Size":
                    len(g),

                "Representative_Feature":
                    g.loc[
                        g["Representative"],
                        "Feature",
                    ].iloc[0]
                    if g["Representative"].any()
                    else None,

                "Num_Dropped":
                    int(
                        g["Dropped_By_Cluster"]
                        .sum()
                    ),

            })

        )

        .reset_index(drop=True)

    )

    return summary

# ============================================================
# CLUSTER WEIGHTS
# ============================================================

def compute_cluster_weights(
    clusters_df: pd.DataFrame,
    feature_weights: dict,
) -> pd.DataFrame:
    """
    Aggregate Global IC weights into cluster weights.

    Parameters
    ----------
    clusters_df

    feature_weights

    Returns
    -------
    Cluster_ID
    Cluster_Weight
    """

    clusters = clusters_df.copy()

    clusters["Global_Weight"] = (
        clusters["Feature"]
        .map(feature_weights)
        .fillna(0.0)
    )

    cluster_weights = (

        clusters

        .groupby(
            "Cluster_ID",
            as_index=False,
        )["Global_Weight"]

        .sum()

        .rename(
            columns={
                "Global_Weight":
                "Cluster_Weight"
            }
        )
    )

    total = cluster_weights["Cluster_Weight"].sum()

    if total > 0:
        cluster_weights["Cluster_Weight"] /= total
    else:
        cluster_weights["Cluster_Weight"] = 0.0

    
    return cluster_weights


# ============================================================
# DIVERSIFICATION MULTIPLIER
# ============================================================

def compute_diversification_multiplier(
    cluster_weights_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Penalize large clusters.

    Large clusters receive
    slightly smaller multipliers.
    """

    weights = cluster_weights_df.copy()

    weights["Diversification_Multiplier"] = (
        1.00
        -
        0.35
        *
        weights["Cluster_Weight"]
    )

    weights["Diversification_Multiplier"] = (
        weights["Diversification_Multiplier"]
        .clip(
            lower=0.70,
            upper=1.00,
        )
    )

    return weights


# ============================================================
# FINAL DIVERSIFIED FEATURE SET
# ============================================================

def build_diversified_feature_set(
    representatives_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Final production feature list.

    Returns
    -------
    One representative
    from each cluster.
    """

    diversified = (
        representatives_df
        .sort_values(
            "Cluster_ID"
        )
        .reset_index(drop=True)
    )

    diversified["Use_Feature"] = True


    return diversified


# ============================================================
# PRINT CLUSTER STATISTICS
# ============================================================

def print_cluster_statistics(
    clusters_df: pd.DataFrame,
) -> None:
    """
    Pretty-print clustering diagnostics.
    """

    if clusters_df.empty:
        return

    print()

    print("=" * 60)
    print("FEATURE CLUSTER STATISTICS")
    print("=" * 60)

    print(

        clusters_df[
            [
                "Feature",
                "Cluster_ID",
                "Cluster_Size",
                "Representative",
                "Dropped_By_Cluster",
            ]
        ]

        .sort_values(
            [
                "Cluster_ID",
                "Representative",
            ],
            ascending=[
                True,
                False,
            ],
        )

    )

    print()
    print(f"Clusters         : {clusters_df['Cluster_ID'].nunique()}" )
    print(f"Representatives  : {clusters_df['Representative'].sum()}" )
    print(f"Dropped Features : {clusters_df['Dropped_By_Cluster'].sum()}" )
    print("=" * 60)


# ============================================================
# COMPLETE FEATURE DIVERSIFICATION PIPELINE
# ============================================================

def diversify_features(
    df: pd.DataFrame,
    selected_features_df: pd.DataFrame,
    feature_weights: dict,
    cluster_method: str = "hierarchical",
    correlation_method: str = "spearman",
    corr_threshold: float = 0.80,
) -> dict:
    """
    Production Feature Clustering Pipeline.

    Returns
    -------
    dict
        correlation
        distance
        clusters
        cluster_weights
        selected
    """


    config = ClusteringConfig(
        method=cluster_method,
        correlation_method=correlation_method,
        correlation_threshold=corr_threshold,
    )

    # -------------------------------------------------------
    # Base clustering
    # -------------------------------------------------------

    base = build_clusters(
        df=df,
        selected_features_df=selected_features_df,
        config=config,
    )

    # -------------------------------------------------------
    # Representatives
    # -------------------------------------------------------

    representatives = select_cluster_representatives(
        clusters_df=base["clusters"],
        selected_features_df=selected_features_df,
    )

    # -------------------------------------------------------
    # Metadata
    # -------------------------------------------------------

    clusters = enrich_cluster_metadata(
        clusters_df=base["clusters"],
        representative_features=
            representatives["Feature"].tolist(),
    )

    # -------------------------------------------------------
    # Cluster weights
    # -------------------------------------------------------

    cluster_weights = compute_cluster_weights(
        clusters_df=clusters,
        feature_weights=feature_weights,
    )

    cluster_weights = compute_diversification_multiplier(
        cluster_weights
    )

    # -------------------------------------------------------
    # Final diversified features
    # -------------------------------------------------------

    diversified = build_diversified_feature_set(
        representatives
    )

    return {
        "correlation": base["correlation"],
        "distance": base["distance"],
        "clusters": clusters,
        "cluster_weights": cluster_weights,
        "selected": diversified,
    }
