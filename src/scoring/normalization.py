"""
============================================================
FEATURE NORMALIZATION
============================================================

Cross-sectional normalization for the Composite Scoring Engine.

Supported methods
-----------------
- zscore
- robust
- rank
- minmax
- none

Features are normalized independently inside each group
(default: each trading date).

Raw feature columns are NEVER modified.

Example
-------
normalize_features(
    df,
    config,
    method="zscore",
    group_cols=["Date"]
)

Future
------
Sector-neutral normalization:

group_cols=["Date", "Sector"]

============================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# INDIVIDUAL NORMALIZATION METHODS
# ============================================================

def _zscore(series: pd.Series) -> pd.Series:
    """
    Standard Z-score.
    """

    mean = series.mean()

    std = series.std(ddof=0)

    if (
        pd.isna(std)
        or
        std == 0
    ):
        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        series - mean
    ) / std


# ============================================================

def _robust(series: pd.Series) -> pd.Series:
    """
    Robust Z-score using Median and MAD.
    """

    median = series.median()

    mad = np.median(
        np.abs(series - median)
    )

    if (
        pd.isna(mad)
        or
        mad == 0
    ):
        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        series - median
    ) / (1.4826 * mad)


# ============================================================

def _rank(series: pd.Series) -> pd.Series:
    """
    Percentile rank.

    Returns values between 0 and 1.
    """

    return series.rank(
        pct=True,
        method="average"
    )


# ============================================================

def _minmax(series: pd.Series) -> pd.Series:
    """
    Min-Max normalization.
    """

    minimum = series.min()

    maximum = series.max()

    spread = maximum - minimum

    if spread == 0:

        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        series - minimum
    ) / spread


# ============================================================

def _none(series: pd.Series) -> pd.Series:
    """
    No normalization.
    """

    return series.copy()


# ============================================================
# METHOD REGISTRY
# ============================================================

NORMALIZERS = {

    "zscore": _zscore,

    "robust": _robust,

    "rank": _rank,

    "minmax": _minmax,

    "none": _none,

}


# ============================================================
# NORMALIZE ONE FEATURE
# ============================================================

def _normalize_feature(
    df: pd.DataFrame,
    feature: str,
    normalizer,
    group_cols,
):
    """
    Normalize a single feature.

    If group_cols is None:

        normalize over entire dataframe.

    Else:

        normalize inside each group.
    """

    if group_cols is None:

        return normalizer(
            df[feature]
        )

    return (
        df
        .groupby(
            group_cols,
            group_keys=False
        )[feature]
        .transform(normalizer)
    )


# ============================================================
# MAIN API
# ============================================================

def normalize_features(
    df: pd.DataFrame,
    config,
    method="zscore",
    group_cols=("Date",),
    suffix="_Norm",
):
    """
    Normalize production features.

    Parameters
    ----------
    df

    config

    method

    group_cols

        None
            Normalize entire dataframe.

        ("Date",)
            Cross-sectional normalization.

        ("Date","Sector")
            Sector-neutral normalization.

    suffix

    Returns
    -------
    pd.DataFrame
    """

    if method not in NORMALIZERS:

        raise ValueError(
            f"Unknown normalization method: {method}"
        )

    out = df.copy()

    normalizer = NORMALIZERS[method]

    if group_cols is not None:

        if isinstance(group_cols, str):

            group_cols = [group_cols]

        missing_groups = [

            col

            for col in group_cols

            if col not in out.columns

        ]

        if missing_groups:

            raise ValueError(

                "Missing grouping columns:\n"

                + "\n".join(missing_groups)

            )

    missing_features = []

    for feature in config.features:

        if feature not in out.columns:

            missing_features.append(feature)

            continue

        normalized = _normalize_feature(

            out,

            feature,

            normalizer,

            group_cols,

        )

        normalized = (

            normalized

            .replace(
                [np.inf, -np.inf],
                np.nan,
            )

            .fillna(0.0)

        )

        out[
            f"{feature}{suffix}"
        ] = normalized

    if missing_features:

        raise ValueError(

            "Missing production features:\n"

            + "\n".join(missing_features)

        )

    return out


# ============================================================
# HELPER
# ============================================================

def normalized_columns(
    config,
    suffix="_Norm",
):
    """
    Return normalized feature names.
    """

    return [

        f"{feature}{suffix}"

        for feature in config.features

    ]


# ============================================================
# INFORMATION
# ============================================================

def print_normalization_summary(
    config,
    method,
    group_cols,
):
    """
    Print normalization settings.
    """

    print("\n===================================")
    print("FEATURE NORMALIZATION")
    print("===================================")

    print(
        f"Method      : {method}"
    )

    print(
        f"Groups      : {group_cols}"
    )

    print(
        f"Features    : {len(config.features)}"
    )

    print("\nNormalized Features:")

    for feature in config.features:

        print(
            f"  • {feature}"
        )

    print("===================================")



# ======================================================
# EXAMPLE USAGE
# ======================================================

# from scoring.feature_config import (
#     load_feature_config,
# )

# from scoring.normalization import (
#     normalize_features,
#     print_normalization_summary,
# )

# config = load_feature_config(
#     "data/master_table_ic.csv"
# )

# print_normalization_summary(
#     config=config,
#     method="zscore",
#     group_cols=["Date"],
# )

# normalized_df = normalize_features(
#     df=master_df,
#     config=config,
#     method="zscore",
#     group_cols=["Date"],
# )

# ======================================================