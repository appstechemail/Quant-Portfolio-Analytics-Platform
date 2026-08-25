"""
============================================================
COMPOSITE SCORING ENGINE
============================================================
Purpose
-------
Everything we've built so far (IC, Rolling IC, Regime IC, Adaptive Weights, Feature Config, Normalization) exists for one purpose:
Produce one composite score per stock, per day.


Computes

Composite Score
=
Σ Adaptive Weight × Normalized Feature

Outputs

1. Individual Feature Contributions
2. Composite Score
============================================================
"""

from __future__ import annotations

import pandas as pd


# ============================================================
# VALIDATION
# ============================================================

def _validate_input(
    df,
    config,
    suffix="_Norm",
):
    """
    Validate required normalized features exist.
    """

    missing = []

    for feature in config.features:
        column = f"{feature}{suffix}"

        if column not in df.columns:
            missing.append(column)

    if missing:
        raise ValueError(
            "Missing normalized feature(s):\n"
            + "\n".join(missing)
        )


# ============================================================
# COMPOSITE SCORE
# ============================================================

def compute_composite_score(
    df,
    config,
    suffix="_Norm",
):

    out = df.copy()

    # ----------------------------------------
    # IC Feature Contributions
    # ----------------------------------------

    out["IC_Score"] = 0.0

    for feature in config.features:
        norm_col = f"{feature}{suffix}"
        contribution = (
            config.weights[feature]
            * out[norm_col]
        )
        
        out[f"{feature}_Score"] = contribution
        out["IC_Score"] += contribution

    # ----------------------------------------
    # Future score blocks
    # ----------------------------------------

    out["Technical_Score"] = 0.0
    out["Fundamental_Score"] = 0.0
    out["Quality_Score"] = 0.0
    out["Adaptive_Score"] = 0.0

    # ----------------------------------------
    # Composite Score
    # ----------------------------------------

    out["Composite_Score"] = (
        out["IC_Score"]
        + out["Technical_Score"]
        + out["Fundamental_Score"]
        + out["Quality_Score"]
        + out["Adaptive_Score"]
    )

    return out


# ============================================================
# FEATURE CONTRIBUTION SUMMARY
# ============================================================

def contribution_summary(
    scored_df,
    config,
):
    """
    Average contribution of every feature.
    Useful for
    Portfolio attribution
    Diagnostics
    Research
    Returns
    -------
    DataFrame
    """

    rows = []

    for feature in config.features:

        contribution = scored_df[
            f"{feature}_Score"
        ]

        rows.append({
            "Feature": feature,
            "Weight": config.weights[feature],
            "Average_Contribution": contribution.mean(),
            "Absolute_Contribution": contribution.abs().mean(),
            "Std_Contribution": contribution.std(),
        })

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        "Absolute_Contribution",
        ascending=False,
    )

    return summary.reset_index(drop=True)


# ============================================================
# PRINT MODEL
# ============================================================

def print_score_model(
    config,
):
    """
    Print scoring model.
    """

    print()
    print("=" * 60)
    print("COMPOSITE SCORING ENGINE")
    print("=" * 60)

    total = 0

    for feature in config.features:
        weight = config.weights[feature]
        total += weight

        print(
            f"{config.feature_rank[feature]:>2}. "
            f"{feature:<25}"
            f"{weight:>8.4f}"
        )

    print("-" * 60)
    print(
        f"Total Weight : {total:.6f}"
    )

    print("=" * 60)


# ======================================================
# EXAMPLE USAGE
# ======================================================

# config = load_feature_config(
#     "data/master_table_ic.csv"
# )

# normalized_df = normalize_features(
#     df,
#     config,
#     method="zscore",
#     group_cols=["Date"],
# )

# scored_df = compute_composite_score(
#     normalized_df,
#     config,
# )

# summary = contribution_summary(
#     scored_df,
#     config,
# )

# print(summary.head())

# ======================================================