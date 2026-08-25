"""
============================================================
IC STABILITY ENGINE
============================================================

Evaluates whether an alpha factor is consistently predictive.

Produces

- IC Volatility
- Positive IC %
- Sign Flip Rate
- Rolling IC Std
- Rolling ICIR Std
- Stability Score
- Stability Weight

============================================================
STRUCTURE:

ic_stability.py
      │
      ▼
Configuration
      │
      ▼
normalize()
      │
      ▼
compute_ic_stability()
      │
      ▼
normalize_stability_metrics()
      │
      ▼
compute_stability_score()
      │
      ▼
build_stability_weights()
      │
      ▼
print_stability_summary()
"""



import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

MIN_OBSERVATIONS = 10

EPSILON = 1e-9

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize(series: pd.Series) -> pd.Series:
    """
    Normalize a series into [0,1].

    If all values are identical,
    returns 1 for every observation.
    """

    series = series.copy()

    median = series.median()

    if pd.isna(median):
        median = 0.0

    series = series.fillna(median)

    spread = series.max() - series.min()

    if spread < EPSILON:

        return pd.Series(
            1.0,
            index=series.index,
        )

    return (
        series - series.min()
    ) / spread


# ============================================================
# COMPUTE IC STABILITY
# ============================================================

def compute_ic_stability(
    daily_ic_df: pd.DataFrame,
    rolling_ic_df: pd.DataFrame,
    min_observations: int = MIN_OBSERVATIONS,
) -> pd.DataFrame:
    """
    Compute raw IC stability metrics for every feature.

    This function computes only raw stability statistics.
    Derived quantities such as Stability_Score,
    Stability_Multiplier and Stability_Weight are computed
    by build_stability_weights().

    Parameters
    ----------
    daily_ic_df
        Output of compute_daily_ic().

    rolling_ic_df
        Output of compute_rolling_ic().

    min_observations
        Minimum number of daily IC observations required.

    Returns
    -------
    pd.DataFrame

        Columns
        -------
        Feature
        Num_Observations
        IC_Volatility
        Positive_IC_Pct
        IC_FlipRate
        Rolling_IC_Volatility
        Rolling_ICIR_Volatility
    """

    # --------------------------------------------------------
    # Empty Input
    # --------------------------------------------------------

    if daily_ic_df.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Validate Daily IC
    # --------------------------------------------------------

    required_daily = {
        "Feature",
        "Date",
        "IC",
    }

    missing = required_daily - set(daily_ic_df.columns)

    if missing:

        raise ValueError(
            "daily_ic_df missing columns:\n"
            + "\n".join(sorted(missing))
        )

    # --------------------------------------------------------
    # Validate Rolling IC
    # --------------------------------------------------------

    if not rolling_ic_df.empty:

        required_rolling = {
            "Feature",
            "Rolling_IC",
            "Rolling_ICIR",
        }

        missing = (
            required_rolling
            - set(rolling_ic_df.columns)
        )

        if missing:
            raise ValueError(
                "rolling_ic_df missing columns:\n"
                + "\n".join(sorted(missing))
            )

        rolling_lookup = {
            feature: grp
            for feature, grp in
            rolling_ic_df.groupby("Feature")
        }

    else:

        rolling_lookup = {}

    # --------------------------------------------------------
    # Compute Stability
    # --------------------------------------------------------

    records = []

    for feature, group in daily_ic_df.groupby("Feature"):

        group = (
            group
            .sort_values("Date")
            .reset_index(drop=True)
        )

        ic = (
            group["IC"]
            .dropna()
        )

        observations = len(ic)

        if observations < min_observations:
            continue

        # ----------------------------------------------------
        # Daily IC Metrics
        # ----------------------------------------------------

        ic_volatility = ic.std(ddof=0)

        positive_ic_pct = (
            (ic > 0)
            .mean()
        )

        signs = np.sign(ic)

        sign_flips = (
            signs
            .diff()
            .fillna(0)
            .ne(0)
            .sum()
        )

        ic_flip_rate = (
            sign_flips
            /
            max(observations - 1, 1)
        )

        # ----------------------------------------------------
        # Rolling Metrics
        # ----------------------------------------------------

        rolling_group = rolling_lookup.get(feature)

        if rolling_group is None:

            rolling_ic_volatility = np.nan
            rolling_icir_volatility = np.nan

        else:

            rolling_ic_volatility = (
                rolling_group["Rolling_IC"]
                .dropna()
                .std(ddof=0)
            )

            rolling_icir_volatility = (
                rolling_group["Rolling_ICIR"]
                .dropna()
                .std(ddof=0)
            )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        records.append({
            "Feature": feature,
            "Num_Observations": observations,
            "IC_Volatility": ic_volatility,
            "Positive_IC_Pct": positive_ic_pct,
            "IC_FlipRate": ic_flip_rate,
            "Rolling_IC_Volatility": rolling_ic_volatility,
            "Rolling_ICIR_Volatility": rolling_icir_volatility,
        })

    # --------------------------------------------------------
    # Final DataFrame
    # --------------------------------------------------------

    stability_df = pd.DataFrame(records)

    if stability_df.empty:
        return stability_df

    stability_df = (
        stability_df
        .sort_values("Feature")
        .reset_index(drop=True)
    )

    
    return stability_df

# ============================================================
# BUILD STABILITY WEIGHTS
# ============================================================

def build_stability_weights(
    stability_df: pd.DataFrame,
    selected_features_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build Stability Scores and Stability Weights.

    Parameters
    ----------
    stability_df
        Output of compute_ic_stability().

    selected_features_df
        Output of filter_ic_features().

        If supplied, stability weights are computed only
        for selected features.

    Returns
    -------
    pd.DataFrame

        Columns
        -------
        Feature
        IC_Volatility
        Positive_IC_Pct
        IC_FlipRate
        Rolling_IC_Volatility
        Rolling_ICIR_Volatility

        Positive_Norm
        FlipRate_Norm
        IC_Volatility_Norm
        Rolling_IC_Volatility_Norm
        Rolling_ICIR_Volatility_Norm

        Stability_Score
        Stability_Multiplier
        Stability_Weight
        Stability_Rank
    """

    if stability_df.empty:
        return pd.DataFrame()

    df = stability_df.copy()

    # --------------------------------------------------------
    # Restrict to selected features
    # --------------------------------------------------------

    if (
        selected_features_df is not None
        and
        not selected_features_df.empty
    ):

        selected = set(
            selected_features_df["Feature"]
        )

        df = df[
            df["Feature"].isin(selected)
        ].copy()

    if df.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Required Columns
    # --------------------------------------------------------

    required_columns = [
        "Feature",
        "IC_Volatility",
        "Positive_IC_Pct",
        "IC_FlipRate",
        "Rolling_IC_Volatility",
        "Rolling_ICIR_Volatility",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing stability columns:\n"
            + "\n".join(sorted(missing))
        )

    # --------------------------------------------------------
    # Fill Missing Values
    # --------------------------------------------------------

    numeric_columns = [
        c
        for c in required_columns
        if c != "Feature"
    ]

    for column in numeric_columns:

        median = df[column].median()

        if pd.isna(median):
            median = 0.0

        df[column] = (
            df[column]
            .fillna(median)
        )

    # --------------------------------------------------------
    # Convert Risk Metrics
    #
    # Lower risk -> Higher score
    # --------------------------------------------------------

    df["IC_Volatility_INV"] = (
        1.0 /
        (df["IC_Volatility"] + EPSILON)
    )

    df["Rolling_IC_Volatility_INV"] = (
        1.0 /
        (df["Rolling_IC_Volatility"] + EPSILON)
    )

    df["Rolling_ICIR_Volatility_INV"] = (
        1.0 /
        (df["Rolling_ICIR_Volatility"] + EPSILON)
    )

    # --------------------------------------------------------
    # Normalize Metrics
    # --------------------------------------------------------

    df["Positive_Norm"] = normalize(
        df["Positive_IC_Pct"]
    )

    df["FlipRate_Norm"] = normalize(
        1.0 - df["IC_FlipRate"]
    )

    df["IC_Volatility_Norm"] = normalize(
        df["IC_Volatility_INV"]
    )

    df["Rolling_IC_Volatility_Norm"] = normalize(
        df["Rolling_IC_Volatility_INV"]
    )

    df["Rolling_ICIR_Volatility_Norm"] = normalize(
        df["Rolling_ICIR_Volatility_INV"]
    )

    # --------------------------------------------------------
    # Composite Stability Score
    # --------------------------------------------------------

    WEIGHT_POSITIVE = 0.30
    WEIGHT_FLIP = 0.20
    WEIGHT_IC_VOL = 0.20
    WEIGHT_ROLLING_VOL = 0.15
    WEIGHT_ROLLING_ICIR_VOL = 0.15

    df["Stability_Score"] = (

        WEIGHT_POSITIVE
        * df["Positive_Norm"]

        + WEIGHT_FLIP
        * df["FlipRate_Norm"]

        + WEIGHT_IC_VOL
        * df["IC_Volatility_Norm"]

        + WEIGHT_ROLLING_VOL
        * df["Rolling_IC_Volatility_Norm"]

        + WEIGHT_ROLLING_ICIR_VOL
        * df["Rolling_ICIR_Volatility_Norm"]

    )

    # --------------------------------------------------------
    # Stability Multiplier
    #
    # Range
    #
    # 0.70 -> 1.30
    # --------------------------------------------------------

    MULTIPLIER_BASE = 0.70
    MULTIPLIER_SCALE = 0.60

    df["Stability_Multiplier"] = (

        MULTIPLIER_BASE

        +

        MULTIPLIER_SCALE
        * df["Stability_Score"]

    )

    # --------------------------------------------------------
    # Stability Weight
    # --------------------------------------------------------

    total_score = df["Stability_Score"].sum()

    if total_score <= EPSILON:

        df["Stability_Weight"] = (
            1.0 / len(df)
        )

    else:

        df["Stability_Weight"] = (

            df["Stability_Score"]

            /

            total_score

        )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            "Stability_Score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    df["Stability_Rank"] = np.arange(
        1,
        len(df) + 1,
    )



    return df


# ============================================================
# PRINT STABILITY SUMMARY
# ============================================================

def print_stability_summary(
    stability_df: pd.DataFrame,
    top_n: int = 20,
) -> None:
    """
    Print Stability Engine summary.

    Parameters
    ----------
    stability_df
        Output of build_stability_weights().

    top_n
        Number of top-ranked features to display.

    Returns
    -------
    None
    """

    if stability_df.empty:

        print("\nNo Stability Engine results available.")

        return

    print()

    print("=" * 90)
    print("IC STABILITY ENGINE")
    print("=" * 90)

    print(
        f"Total Features : {len(stability_df)}"
    )

    if "Stability_Score" in stability_df.columns:

        print(
            f"Mean Stability Score : "
            f"{stability_df['Stability_Score'].mean():.4f}"
        )

    if "Stability_Weight" in stability_df.columns:

        print(
            f"Weight Sum : "
            f"{stability_df['Stability_Weight'].sum():.6f}"
        )

    print("-" * 90)

    display_columns = [
        "Stability_Rank",
        "Feature",
        "Positive_IC_Pct",
        "IC_FlipRate",
        "IC_Volatility",
        "Rolling_IC_Volatility",
        "Rolling_ICIR_Volatility",
        "Stability_Score",
        "Stability_Multiplier",
        "Stability_Weight",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in stability_df.columns
    ]

    display_df = (
        stability_df
        .sort_values(
            "Stability_Score",
            ascending=False,
        )
        .head(top_n)
        .loc[:, available_columns]
    )

    print(
        display_df.to_string(
            index=False,
            justify="left",
        )
    )

    print("=" * 90)