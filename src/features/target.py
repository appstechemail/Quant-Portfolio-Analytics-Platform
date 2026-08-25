# ==========================================================
# TARGET GENERATION ENGINE (LATEST QUANT PRODUCTION VERSION)
# ==========================================================
#
# PURPOSE
# -------
# This module creates leakage-safe machine learning targets
# for cross-sectional stock ranking systems.
#
#
# CORE PHILOSOPHY
# ---------------
#
# OLD APPROACH
# ------------
#
# Traditional binary targets:
#
#     if return > threshold:
#         BUY
#     else:
#         SELL
#
# Problems:
#
# ✘ unstable labels
# ✘ regime sensitive
# ✘ poor class balance
# ✘ weak ranking power
# ✘ noisy predictions
#
#
# NEW APPROACH (RECOMMENDED)
# --------------------------
#
# CROSS-SECTIONAL PERCENTILE TARGETS
#
# Instead of asking:
#
#     "Will stock go up?"
#
# We ask:
#
#     "Will stock outperform other stocks?"
#
#
# EXAMPLE
# -------
#
# On a given date:
#
#     Top 20% performers  -> BUY (1)
#     Remaining 80%       -> NOT BUY (0)
#
#
# BENEFITS
# --------
#
# ✔ More stable ML labels
# ✔ Stronger alpha signals
# ✔ Better ensemble learning
# ✔ Better Sharpe ratio
# ✔ Better cross-sectional ranking
# ✔ Lower overfitting
# ✔ Better live trading robustness
# ✔ Naturally adapts to market regimes
#
#
# LEAKAGE SAFETY
# --------------
#
# ✔ Only future prices generate labels
# ✔ Features remain historical only
# ✔ Fully safe for walk-forward validation
# ✔ Fully safe for live trading
#
#
# OUTPUT COLUMNS
# --------------
#
# - Future_Return
# - Volatility
# - Risk_Adjusted_Return
# - Return_Rank
# - Target
#
#
# TARGET DEFINITION
# -----------------
#
# Target = 1
#     Stock belongs to top X%
#     future performers
#
# Target = 0
#     Otherwise
#
#
# RECOMMENDED SETTINGS
# --------------------
#
# Top 20%:
#     strongest alpha
#
# Top 30%:
#     more stable
#
# Top 40%:
#     weaker signal
#
#
# BEST PRACTICE
# -------------
#
# Use:
#
#     TOP_PERCENTILE = 0.80
#
# Meaning:
#
#     Top 20% winners become BUY class
#
#
# SAFE FOR
# --------
#
# ✔ Backtesting
# ✔ Walk-forward validation
# ✔ Live trading
# ✔ Ensemble learning
# ✔ Cross-sectional ranking
# ✔ Portfolio optimization
#
# ==========================================================

import numpy as np
import pandas as pd

from config.config import CONFIG


# ==========================================================
# CONFIG
# ==========================================================

tgt_config = CONFIG.get("TARGET", {})

PREDICT_DAYS_AHEAD = tgt_config.get(
    "PREDICT_DAYS_AHEAD",
    5
)

VOL_LOOKBACK = tgt_config.get(
    "VOL_LOOKBACK",
    20
)

MAX_DAILY_MOVE_CAP = tgt_config.get(
    "MAX_DAILY_MOVE_CAP",
    0.30
)

TOP_PERCENTILE = tgt_config.get(
    "TOP_PERCENTILE",
    0.80
)

# ==========================================================
# TRIPLE BARRIER SETTINGS
# ==========================================================

HOLDING_DAYS = tgt_config.get(
    "HOLDING_DAYS",
        5
)

# ==========================================================
# VOLATILITY-ADJUSTED BARRIERS
# ==========================================================

TP_VOL_MULTIPLIER = tgt_config.get(
    "TP_VOL_MULTIPLIER",
    2.0
)

SL_VOL_MULTIPLIER = tgt_config.get(
    "SL_VOL_MULTIPLIER",
    1.0
)

# ==========================================================
# META LABEL SETTINGS
# ==========================================================

META_RETURN_THRESHOLD = tgt_config.get(
    "META_RETURN_THRESHOLD",
    0.02
)

def triple_barrier_labeling(group):

    group = group.copy()

    closes = group["Close"].values

    # ======================================================
    # HISTORICAL VOLATILITY
    # STRICTLY BACKWARD LOOKING
    # ======================================================

    returns = pd.Series(closes).pct_change()

    volatility = (

        returns
        .rolling(
            VOL_LOOKBACK,
            min_periods=5
        )
        .std()
        .shift(1)
    )

    volatility = volatility.fillna(
        volatility.median()
    )

    targets = []

    n = len(group)

    for i in range(n):

        entry_price = closes[i]

        vol = volatility.iloc[i]

        # --------------------------------------------------
        # VOL-ADJUSTED BARRIERS
        # --------------------------------------------------

        tp_pct = vol * TP_VOL_MULTIPLIER

        sl_pct = vol * SL_VOL_MULTIPLIER

        # Safety floor
        tp_pct = max(tp_pct, 0.02)
        sl_pct = max(sl_pct, 0.01)

        tp_price = entry_price * (
            1 + tp_pct
        )

        sl_price = entry_price * (
            1 - sl_pct
        )

        end_idx = min(
            i + HOLDING_DAYS,
            n - 1
        )

        label = np.nan

        for j in range(i + 1, end_idx + 1):

            future_price = closes[j]

            # ----------------------------------------------
            # TAKE PROFIT HIT FIRST
            # ----------------------------------------------
            if future_price >= tp_price:
                label = 1
                break

            # ----------------------------------------------
            # STOP LOSS HIT FIRST
            # ----------------------------------------------
            elif future_price <= sl_price:

                label = 0
                break

        # ======================================
        # TIMEOUT LABEL
        # ======================================

        if np.isnan(label):

            final_price = closes[end_idx]

            label = int(
                final_price > entry_price
            )

        targets.append(label)

    group["Target"] = targets

    # ======================================================
    # META LABEL
    # ======================================================
    #
    # Meta target asks:
    #
    # "Did the primary signal actually make money?"
    #
    # Here:
    #
    # 1 = profitable TP hit
    # 0 = SL hit / failed trade
    #
    # ======================================================

    # group["Meta_Target"] = np.where(
    #     group["Target"] == 1,
    #     1,
    #     0
    # )

    return group
# ==========================================================
# ADD TARGET
# ==========================================================
def add_target(df):

    print("\n🎯 GENERATING CROSS-SECTIONAL TARGETS...")

    df = df.copy()
    print("\nINSIDE ADD_TARGET ENTRY")
    print(df.columns.tolist())
    print(df.shape)

    # ======================================================
    # REQUIRED COLUMNS
    # ======================================================
    required_cols = [
        "Date",
        "Company",
        "Close"
    ]

    missing_cols = [

        c for c in required_cols

        if c not in df.columns
    ]

    if missing_cols:

        raise ValueError(
            f"❌ Missing required columns: "
            f"{missing_cols}"
        )

    # ======================================================
    # SORT DATA
    # ======================================================
    df = (
        df.sort_values(
            ["Company", "Date"]
        )
        .reset_index(drop=True)
    )

    # ======================================================
    # CLEAN CLOSE
    # ======================================================
    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    df.loc[
        df["Close"] <= 0,
        "Close"
    ] = np.nan

    # ======================================================
    # FUTURE RETURN
    # ======================================================

    df["Future_Return"] = (

        df.groupby("Company")["Close"]

        .shift(-PREDICT_DAYS_AHEAD)

        /

        df["Close"]

        - 1
    )

    # ======================================================
    # NEUTRAL ZONE TARGET
    # ======================================================

    UPPER_THRESHOLD = 0.03
    LOWER_THRESHOLD = -0.03

    df["Neutral_Target"] = np.nan

    df.loc[
        df["Future_Return"] >= UPPER_THRESHOLD,
        "Neutral_Target"
    ] = 1

    df.loc[
        df["Future_Return"] <= LOWER_THRESHOLD,
        "Neutral_Target"
    ] = 0

    # ======================================================
    # CROSS-SECTIONAL RELATIVE RETURN TARGET
    # ======================================================

    df["Future_Return_Rank"] = (

        df.groupby("Date")["Future_Return"]

        .rank(
            pct=True,
            method="average"
        )

    )

    # ==========================================
    # ALPHA TARGET
    # ==========================================

    df["Alpha_Target"] = (
        df["Future_Return_Rank"]
    )

    df["Alpha_Target_Z"] = (
        df.groupby("Date")["Future_Return"]
        .transform(
            lambda x:
            (x - x.mean()) /
            (x.std() + 1e-9)
        )
    )

    # ======================================================
    # INSTITUTIONAL LONG/SHORT TARGET
    # ======================================================

    df["CrossSection_Target"] = np.nan

    # Top 20% winners

    df.loc[
        df["Future_Return_Rank"] >= 0.80,
        "CrossSection_Target"
    ] = 1

    # Bottom 20% losers

    df.loc[
        df["Future_Return_Rank"] <= 0.20,
        "CrossSection_Target"
    ] = 0

    # ======================================================
    # RETURN CLIP
    # ======================================================

    df["Future_Return"] = (

        df["Future_Return"]

        .clip(
            -MAX_DAILY_MOVE_CAP,
            MAX_DAILY_MOVE_CAP
        )
    )

    # ======================================================
    # TRIPLE BARRIER LABELING
    # ======================================================

    print("\nBEFORE GROUPBY")
    print(type(df))
    print(df.columns.tolist())

    dfs = []

    for company, group in df.groupby("Company"):
        result = triple_barrier_labeling(group)
        result["Company"] = company
        dfs.append(result)

    df = pd.concat(
        dfs,
        ignore_index=True
    )

    print("\nAFTER TRIPLE BARRIER")

    print("Columns:")
    print(df.columns.tolist())

    print("Index names:")
    print(df.index.names)

    print(
        "Company exists:",
        "Company" in df.columns
    )


    # ======================================================
    # HISTORICAL VOLATILITY
    # STRICTLY BACKWARD LOOKING
    # ======================================================
    daily_returns = (
        df.groupby("Company")["Close"]
        .pct_change()
    )

    df["Volatility"] = (
        daily_returns
        .groupby(df["Company"])
        .transform(
            lambda x:
            x.shift(1).rolling(
                VOL_LOOKBACK,
                min_periods=10
            ).std()
        )
    )



    # ======================================================
    # SAFE VOL FILL
    # ======================================================
    median_vol = df["Volatility"].median()

    if pd.isna(median_vol):

        median_vol = 0.02

    df["Volatility"] = (
        df["Volatility"]
        .fillna(median_vol)
    )

    # ======================================================
    # CLEAN INVALID VALUES
    # ======================================================
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ======================================================
    # REMOVE INVALID TARGETS
    # ======================================================

    # df = df.dropna(
    #     subset=["Target"]
    # )

    df = df.dropna(
        subset=["Future_Return"]
    )

    # ======================================================
    # NEUTRAL TARGET
    # ======================================================

    valid_neutral = (
        df["Neutral_Target"]
        .notna()
    )

    df.loc[
        valid_neutral,
        "Neutral_Target"
    ] = (
        df.loc[
            valid_neutral,
            "Neutral_Target"
        ].astype(int)
    )

    # ======================================================
    # CROSS SECTION TARGET
    # ======================================================

    valid_cs = (
        df["CrossSection_Target"]
        .notna()
    )

    df.loc[
        valid_cs,
        "CrossSection_Target"
    ] = (
        df.loc[
            valid_cs,
            "CrossSection_Target"
        ].astype(int)
    )

    # ======================================================
    # INTEGER TARGET
    # ======================================================

    df = df.dropna(
        subset=["Target"]
    )

    df["Target"] = (
        df["Target"]
        .astype(int)
    )

    # ======================================================
    # META LABEL TARGET
    # ======================================================
    #
    # Meta target:
    #
    # 1 = trade was profitable enough
    # 0 = weak/bad trade
    #
    # ======================================================

    df["Meta_Target"] = (

        df["Future_Return"]

        >

        META_RETURN_THRESHOLD

    ).astype(int)

    # ======================================================
    # TARGET DISTRIBUTION
    # ======================================================
    print("\n🎯 TARGET DISTRIBUTION")

    target_counts = (
        df["Target"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print(target_counts)

    # ======================================================
    # UNIQUE CLASSES
    # ======================================================
    unique_classes = sorted(
        df["Target"].unique()
    )

    print(
        f"\n📌 Unique classes: "
        f"{unique_classes}"
    )

    if len(unique_classes) < 2:

        raise ValueError(
            "❌ Only one target class generated"
        )

    # ======================================================
    # FINAL REPORT
    # ======================================================
    print("\n📊 TARGET SETTINGS")

    print(
        "Prediction Horizon:",
        PREDICT_DAYS_AHEAD
    )

    print(
        "Volatility Lookback:",
        VOL_LOOKBACK
    )

    print(
        "Top Percentile:",
        TOP_PERCENTILE
    )

    print(
        "Max Daily Move Cap:",
        MAX_DAILY_MOVE_CAP
    )

    print(
        f"\n✅ Final dataset shape: "
        f"{df.shape}"
    )

    # ======================================================
    # CLASS BALANCE
    # ======================================================
    buy_ratio = (
        df["Target"]
        .mean()
    )

    print(
        f"\n📈 BUY CLASS RATIO: "
        f"{buy_ratio:.2%}"
    )

    # ======================================================
    # FINAL CLEANUP
    # ======================================================
    keep_cols = [

        c for c in df.columns

        if df[c].notna().sum() > 0
    ]

    df = df[keep_cols]


    # ======================================================
    # CLEAN CROSS-SECTION TARGET
    # ======================================================

    df["CrossSection_Target"] = (
        df["CrossSection_Target"]
    )

    print("\n📊 CROSS SECTION TARGET")

    print(
        df["CrossSection_Target"]
        .value_counts(dropna=False)
    )

    return df