# ============================================================
# TECHNICAL FEATURE ENGINEERING MODULE
# ============================================================

import numpy as np
import pandas as pd
import pandas_ta as ta

from config.config import CONFIG


# ============================================================
# CONFIG
# ============================================================

f_config = CONFIG["FEATURES"]

f_ma_windows = f_config["MA_WINDOWS"]
f_vol_window = f_config["VOL_WINDOW"]
f_mom_window = f_config["MOMENTUM_WINDOW"]
f_rsi_window = f_config["RSI_WINDOW"]


# ============================================================
# RSI FUNCTION
# ============================================================

def compute_rsi(series, window=14):

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        gain
        .rolling(window, min_periods=5)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(window, min_periods=5)
        .mean()
    )

    rs = avg_gain / (avg_loss + 1e-9)

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ============================================================
# MACD FUNCTION
# ============================================================

def compute_macd(series):

    ema12 = series.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = series.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    hist = macd - signal

    return pd.DataFrame({

        "MACD": macd,
        "MACD_Signal": signal,
        "MACD_Hist": hist

    })


# ============================================================
# MAIN FEATURE FUNCTION
# ============================================================

def add_technical_features(df):

    print("\n⚙️ GENERATING TECHNICAL FEATURES...")

    # ========================================================
    # SORT
    # ========================================================

    df = (
        df.sort_values(
            ["Company", "Date"]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # RETURNS
    # ========================================================

    df["Return"] = (
        df.groupby("Company")["Close"]
        .pct_change()
    )

    df["Log_Return"] = (
        df.groupby("Company")["Close"]
        .transform(
            lambda x:
            np.log(x / x.shift(1))
        )
    )

    # ========================================================
    # MOVING AVERAGES
    # ========================================================

    for window in f_ma_windows:

        # SMA
        df[f"MA{window}"] = (
            df.groupby("Company")["Close"]
            .transform(
                lambda x:
                x.rolling(
                    window,
                    min_periods=max(5, window // 2)
                ).mean()
            )
        )

        # EMA
        df[f"EMA{window}"] = (
            df.groupby("Company")["Close"]
            .transform(
                lambda x:
                x.ewm(
                    span=window,
                    adjust=False
                ).mean()
            )
        )

        # Price / MA
        df[f"Price_to_MA{window}"] = (
            df["Close"] /
            (df[f"MA{window}"] + 1e-9)
        )

    # ========================================================
    # ATR (Average True Range)
    # ========================================================

    print("⚙️ Calculating ATR")

    df["ATR_14"] = np.nan

    for company in df["Company"].unique():

        mask = df["Company"] == company

        atr = ta.atr(
            df.loc[mask, "High"],
            df.loc[mask, "Low"],
            df.loc[mask, "Close"],
            length=14
        )

        df.loc[mask, "ATR_14"] = atr.values

    # ========================================================
    # VOLATILITY
    # ========================================================

    df["Volatility"] = (
        df.groupby("Company")["Return"]
        .transform(
            lambda x:
            x.rolling(
                f_vol_window,
                min_periods=5
            ).std()
        )
    )

    # ========================================================
    # MOMENTUM
    # ========================================================

    df["Momentum"] = (
        df.groupby("Company")["Close"]
        .transform(
            lambda x:
            x / x.shift(f_mom_window) - 1
        )
    )

    # Risk-adjusted momentum using ATR
    df["Momentum_ATR"] = (
        df["Momentum"] /
        (df["ATR_14"] + 1e-9)
    )

    # Risk-adjusted momentum using rolling volatility
    df["Momentum_Vol"] = (
        df["Momentum"] /
        (df["Volatility"] + 1e-9)
    )

    df["Reversal"] = -df["Momentum"]

    # ========================================================
    # RSI
    # ========================================================

    df["RSI"] = (
        df.groupby("Company")["Close"]
        .transform(
            lambda x:
            compute_rsi(
                x,
                f_rsi_window
            )
        )
    )

    # ========================================================
    # MACD
    # ========================================================

    macd_df = (
        df.groupby("Company")["Close"]
        .apply(compute_macd)
        .reset_index(level=0, drop=True)
    )

    df["MACD"] = macd_df["MACD"]

    df["MACD_Signal"] = macd_df["MACD_Signal"]

    df["MACD_Hist"] = macd_df["MACD_Hist"]

    # ========================================================
    # BOLLINGER Z-SCORE
    # ========================================================

    for window in f_ma_windows:

        rolling_mean = (
            df.groupby("Company")["Close"]
            .transform(
                lambda x:
                x.rolling(
                    window,
                    min_periods=max(5, window // 2)
                ).mean()
            )
        )

        rolling_std = (
            df.groupby("Company")["Close"]
            .transform(
                lambda x:
                x.rolling(
                    window,
                    min_periods=max(5, window // 2)
                ).std()
            )
        )

        df[f"BB_Z{window}"] = (
            (
                df["Close"] -
                rolling_mean
            )
            /
            (rolling_std + 1e-9)
        )

    # ========================================================
    # LAG FEATURES
    # ========================================================

    close_lags = [1, 2, 3, 5]

    return_lags = [1, 2, 3, 5, 10]

    for lag in close_lags:

        df[f"Close_Lag{lag}"] = (
            df.groupby("Company")["Close"]
            .shift(lag)
        )

    for lag in return_lags:

        df[f"Return_Lag{lag}"] = (
            df.groupby("Company")["Return"]
            .shift(lag)
        )

    # ========================================================
    # TREND FEATURES
    # ========================================================

    short_ma = min(f_ma_windows)

    long_ma = max(f_ma_windows)

    df["Trend"] = (
        df[f"EMA{short_ma}"] >
        df[f"EMA{long_ma}"]
    ).astype(int)

    df["Trend_Strength"] = (
        (
            df[f"EMA{short_ma}"] -
            df[f"EMA{long_ma}"]
        )
        /
        (df[f"EMA{long_ma}"] + 1e-9)
    )

    # ========================================================
    # EMA SPREAD
    # ========================================================

    if "EMA10" in df.columns and "EMA50" in df.columns:

        df["EMA_Spread"] = (
            (
                df["EMA10"] -
                df["EMA50"]
            )
            /
            (df["EMA50"] + 1e-9)
        )

    # ========================================================
    # VOLUME FEATURES
    # ========================================================

    df["Volume_Change"] = (
        df.groupby("Company")["Volume"]
        .pct_change()
    )

    df["Volume_MA"] = (
        df.groupby("Company")["Volume"]
        .transform(
            lambda x:
            x.rolling(
                20,
                min_periods=5
            ).mean()
        )
    )

    # Volume Ratio
    df["Volume_Ratio"] = (
        df["Volume"]
        /
        (df["Volume_MA"] + 1e-9)
    )

    # ========================================================
    # ATR
    # ========================================================

    prev_close = (
        df.groupby("Company")["Close"]
        .shift(1)
    )

    tr1 = df["High"] - df["Low"]

    tr2 = (
        df["High"] - prev_close
    ).abs()

    tr3 = (
        df["Low"] - prev_close
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = (
        tr.groupby(df["Company"])
        .transform(
            lambda x:
            x.rolling(
                14,
                min_periods=5
            ).mean()
        )
    )

    # ========================================================
    # GAP FEATURE
    # ========================================================

    df["Gap"] = (
        df["Open"]
        /
        (prev_close + 1e-9)
    ) - 1

    # ========================================================
    # RANGE POSITION
    # ========================================================

    df["High_20"] = (
        df.groupby("Company")["High"]
        .transform(
            lambda x:
            x.rolling(
                20,
                min_periods=5
            ).max()
        )
    )

    df["Low_20"] = (
        df.groupby("Company")["Low"]
        .transform(
            lambda x:
            x.rolling(
                20,
                min_periods=5
            ).min()
        )
    )

    df["Range_Position"] = (
        (
            df["Close"] -
            df["Low_20"]
        )
        /
        (
            df["High_20"] -
            df["Low_20"] +
            1e-9
        )
    )

    # ========================================================
    # ROLLING RETURNS
    # ========================================================

    df["Return_5"] = (
        df.groupby("Company")["Return"]
        .transform(
            lambda x:
            x.rolling(
                5,
                min_periods=3
            ).sum()
        )
    )

    df["Return_10"] = (
        df.groupby("Company")["Return"]
        .transform(
            lambda x:
            x.rolling(
                10,
                min_periods=5
            ).sum()
        )
    )


    # =========================================
    # ROLLING RETURNS
    # =========================================

    df["Return_5"] = (
        df.groupby("Company")["Return"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(5, min_periods=3)
            .sum()
        )
    )

    df["Return_10"] = (
        df.groupby("Company")["Return"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(10, min_periods=5)
            .sum()
        )
    )

    # =========================================
    # REVERSAL FEATURES
    # =========================================

    df["Reversal_5"] = -df["Return_5"]
    df["Reversal_10"] = -df["Return_10"]

    # ========================================================
    # CROSS-SECTIONAL RANKS
    # ========================================================

    rank_features = {
        "Return": "Return_Rank",
        "Momentum": "Momentum_Rank",
        "Volatility": "Volatility_Rank"
    }

    for base_col, rank_col in rank_features.items():

        df[rank_col] = (
            df.groupby("Date")[base_col]
            .rank(pct=True)
        )

    # ========================================================
    # VOLATILITY REGIME
    # ========================================================

    median_vol = (
        df.groupby("Date")["Volatility"]
        .transform("median")
    )

    df["Vol_Regime"] = (
        df["Volatility"] > median_vol
    ).astype(int)

    # ========================================================
    # CROSS-SECTIONAL Z-SCORES
    # ========================================================

    zscore_features = [

        "Return",
        "Momentum",
        "Momentum_ATR",
        "Momentum_Vol",
        "Volatility",
        "RSI",
        "MACD",
        "MACD_Hist",
        "Trend_Strength",
        "Return_5",
        "Return_10",
        "Volume_Change",
        "Volume_Ratio",
        "Range_Position",
        "ATR",
        "Gap",
        "EMA_Spread"

    ]

    zscore_features = [

        c for c in zscore_features
        if c in df.columns

    ]

    for col in zscore_features:

        df[f"{col}_Z"] = (
            df.groupby("Date")[col]
            .transform(
                lambda x:
                (
                    x - x.mean()
                )
                /
                (x.std() + 1e-9)
            )
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ========================================================
    # PREVENT LOOKAHEAD BIAS
    # GLOBAL SHIFT
    # ========================================================

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

    already_safe_cols = [

        c for c in df.columns
        if "Lag" in c

    ]

    feature_cols = [

        c for c in df.columns

        if c not in non_predictive_cols

        and c not in already_safe_cols

    ]

    # SINGLE GLOBAL SHIFT
    df[feature_cols] = (
        df.groupby("Company")[feature_cols]
        .shift(1)
    )

    # ========================================================
    # WINSORIZATION
    # CROSS-SECTIONAL (LEAKAGE SAFE)
    # ========================================================

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    exclude_cols = [

        "Target",
        "Future_Return",
        "Future_Close"

    ]

    winsor_cols = [

        c for c in numeric_cols
        if c not in exclude_cols

    ]

    for col in winsor_cols:

        lower = (
            df.groupby("Date")[col]
            .transform(
                lambda x:
                x.quantile(0.01)
            )
        )

        upper = (
            df.groupby("Date")[col]
            .transform(
                lambda x:
                x.quantile(0.99)
            )
        )

        df[col] = df[col].clip(
            lower,
            upper
        )

    # ========================================================
    # FINAL CLEANUP
    # ========================================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print(
        f"\n✅ Technical features generated: "
        f"{len(feature_cols)}"
    )

    return df