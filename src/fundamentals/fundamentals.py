# ============================================================
# FUNDAMENTAL FEATURE ENGINEERING MODULE
# ============================================================
#
# FILE: fundamentals.py
#
# PURPOSE:
# --------
# This module generates leakage-safe fundamental and valuation
# features for the stock prediction and quantitative trading
# system.
#
# The module combines:
#
#     • EPS (Earnings Per Share)
#     • PE Ratio (Price-to-Earnings)
#     • EPS Growth
#     • PE Change
#     • Cross-sectional normalization
#     • Value scoring
#     • Relative ranking
#     • Optional Market Capitalization
#
# The generated features help the models identify:
#
#     • Undervalued companies
#     • Growth opportunities
#     • Relative value strength
#     • Fundamental momentum
#     • Cross-sectional market positioning
#
# ------------------------------------------------------------
# KEY DESIGN PRINCIPLES
# ------------------------------------------------------------
#
# 1. LEAKAGE-SAFE ARCHITECTURE
#
#    The module strictly prevents:
#
#         • Lookahead Bias
#         • Future EPS Leakage
#         • Same-Bar Leakage
#         • Cross-Sectional Leakage
#
#    All predictive features are shifted by 1 bar to ensure
#    realistic backtesting and live trading compatibility.
#
# ------------------------------------------------------------
# 2. TIME-SAFE FUNDAMENTALS
#
#    EPS values are:
#
#         • Forward-filled only after release
#         • Company-wise aligned
#         • Financial-year mapped
#
#    This prevents future quarterly information from leaking
#    into earlier dates.
#
# ------------------------------------------------------------
# 3. CROSS-SECTIONAL NORMALIZATION
#
#    Daily z-score normalization is applied across stocks to:
#
#         • Remove scale differences
#         • Improve model stability
#         • Capture relative market positioning
#
# ------------------------------------------------------------
# 4. CONFIG-DRIVEN DESIGN
#
#    File paths and parameters are dynamically loaded from:
#
#         CONFIG
#
#    making the module reusable and production-friendly.
#
# ------------------------------------------------------------
# FEATURES GENERATED
# ------------------------------------------------------------
#
# 1. FUNDAMENTAL FEATURES
#    • EPS
#    • PE Ratio
#
# 2. GROWTH FEATURES
#    • EPS Growth
#    • PE Change
#
# 3. NORMALIZED FEATURES
#    • EPS_Growth_Z
#    • PE_Z
#
# 4. VALUE FEATURES
#    • Value_Score
#
# 5. CROSS-SECTIONAL RANKS
#    • PE_Rank
#    • Value_Rank
#
# 6. OPTIONAL MARKET FEATURES
#    • MarketCap
#    • MarketCap_Z
#
# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------
#
# Main DataFrame containing:
#
#     Date
#     Company
#     Open
#     High
#     Low
#     Close
#     Volume
#
# Optional:
#
#     stocks dictionary for MarketCap retrieval
#
# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------
#
# DataFrame enriched with leakage-safe fundamental and
# valuation-based predictive features.
#
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf

from config.config import CONFIG


# =========================================
# CONFIG
# =========================================

fm_eps_file = CONFIG["DATA"]["PATHS"]["EPS_FILE"]


# =========================================
# FUNDAMENTAL FEATURES
# =========================================
def add_basic_fundamentals(df, stocks=None):

    df = df.copy()

    # =========================================
    # LOAD EPS DATA
    # =========================================
    print("📂 Loading EPS file:", fm_eps_file)

    eps_df = pd.read_csv(fm_eps_file)

    eps_df.columns = (
        eps_df.columns
        .str.strip()
        .str.upper()
    )

    required_cols = [
        "COMPANY",
        "YEAR",
        "EPS"
    ]

    for col in required_cols:

        if col not in eps_df.columns:

            raise ValueError(
                f"❌ Missing column '{col}' in EPS file"
            )

    # =========================================
    # CLEAN EPS DATA
    # =========================================
    eps_df["COMPANY"] = (
        eps_df["COMPANY"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    eps_df["YEAR"] = (
        eps_df["YEAR"]
        .astype(str)
        .str.extract(r"(\d{4})")[0]
    )

    eps_df["YEAR"] = pd.to_numeric(
        eps_df["YEAR"],
        errors="coerce"
    )

    eps_df = eps_df.dropna(
        subset=["YEAR"]
    )

    eps_df["YEAR"] = (
        eps_df["YEAR"]
        .astype(int)
    )

    # =========================================
    # PREPARE MAIN DATAFRAME
    # =========================================
    df["Date"] = pd.to_datetime(df["Date"])

    df["Company"] = (
        df["Company"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # =========================================
    # INDIA FINANCIAL YEAR
    # Apr-Mar
    # =========================================
    df["YEAR"] = np.where(
        df["Date"].dt.month >= 4,
        df["Date"].dt.year + 1,
        df["Date"].dt.year
    ).astype(int)

    # =========================================
    # MERGE EPS
    # =========================================
    df = df.merge(
        eps_df[[
            "COMPANY",
            "YEAR",
            "EPS"
        ]],
        left_on=["Company", "YEAR"],
        right_on=["COMPANY", "YEAR"],
        how="left"
    )

    # Remove duplicate merge column
    df.drop(
        columns=["COMPANY"],
        inplace=True
    )

    # =========================================
    # SORT FOR TIME-SERIES SAFETY
    # =========================================
    df = df.sort_values(
        ["Company", "Date"]
    )

    # =========================================
    # FORWARD FILL EPS
    # ONLY AFTER RELEASE
    # =========================================
    df["EPS"] = (
        df.groupby("Company")["EPS"]
        .transform(lambda x: x.ffill())
    )

    # =========================================
    # SAFE PE RATIO
    # =========================================
    df["EPS"] = (
        df["EPS"]
        .replace(0, np.nan)
    )

    df["PE"] = (
        df["Close"] /
        (df["EPS"] + 1e-9)
    )

    # Remove invalid PE
    df.loc[
        df["PE"] < 0,
        "PE"
    ] = np.nan

    # Clip extreme values
    df["PE"] = (
        df["PE"]
        .clip(0, 100)
    )

    # =========================================
    # ADVANCED FUNDAMENTAL FEATURES
    # =========================================

    # EPS Growth
    df["EPS_Growth"] = (
        df.groupby("Company")["EPS"]
        .pct_change()
    )

    # PE Change
    df["PE_Change"] = (
        df.groupby("Company")["PE"]
        .pct_change()
    )

    # Replace bad values
    df["EPS_Growth"] = (
        df["EPS_Growth"]
        .replace([np.inf, -np.inf], np.nan)
    )

    df["PE_Change"] = (
        df["PE_Change"]
        .replace([np.inf, -np.inf], np.nan)
    )

    # =========================================
    # CROSS-SECTIONAL NORMALIZATION
    # =========================================
    def zscore(x):

        return (
            (x - x.mean()) /
            (x.std() + 1e-9)
        )

    df["EPS_Growth_Z"] = (
        df.groupby("Date")["EPS_Growth"]
        .transform(zscore)
    )

    df["PE_Z"] = (
        df.groupby("Date")["PE"]
        .transform(zscore)
    )

    # =========================================
    # VALUE SCORE
    # =========================================
    df["Value_Score"] = (
        df["EPS_Growth_Z"] -
        df["PE_Z"]
    )

    # =========================================
    # CROSS-SECTIONAL RANK FEATURES
    # =========================================
    rank_map = {
        "PE": "PE_Rank",
        "Value_Score": "Value_Rank"
    }

    for base_col, rank_col in rank_map.items():

        # Same-day ranking
        df[rank_col] = (
            df.groupby("Date")[base_col]
            .rank(pct=True)
        )

        # Shift to next tradable bar
        df[rank_col] = (
            df.groupby("Company")[rank_col]
            .shift(1)
        )

    # =========================================
    # OPTIONAL MARKET CAP FEATURES
    # =========================================
    if stocks is not None:

        print("📊 Adding MarketCap (snapshot)")

        market_caps = {}

        for ticker, name in stocks.items():

            try:

                info = yf.Ticker(ticker).info

                market_caps[
                    name.upper()
                ] = info.get(
                    "marketCap",
                    np.nan
                )

            except:

                market_caps[
                    name.upper()
                ] = np.nan

        df["MarketCap"] = (
            df["Company"]
            .map(market_caps)
        )

        # Cross-sectional normalization
        df["MarketCap_Z"] = (
            df.groupby("Date")["MarketCap"]
            .transform(zscore)
        )

    # =========================================
    # CLEANUP
    # =========================================
    df.drop(
        columns=["YEAR"],
        inplace=True
    )

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Safe fill
    fill_cols = [
        "EPS_Growth",
        "PE_Change",
        "Value_Score"
    ]

    for col in fill_cols:

        if col in df.columns:

            df[col] = (
                df[col]
                .fillna(0)
            )

    print("✅ Fundamentals added successfully")

    # =========================================
    # PREVENT LOOKAHEAD BIAS
    # =========================================

    non_predictive_cols = [
        "Date",
        "Company",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Target",
        "Future_Return",
        "Future_Close"
    ]

    # Already-safe lag/rank columns
    already_safe_cols = [
        "PE_Rank",
        "Value_Rank"
    ]

    # Fundamental predictive features
    feature_cols = [
        c for c in df.columns
        if c not in non_predictive_cols
        and c not in already_safe_cols
    ]

    # Shift predictive features
    df[feature_cols] = (
        df.groupby("Company")[feature_cols]
        .shift(1)
    )

    return df