
from config.config import CONFIG

cfg = CONFIG["REGIME"]

# AUTO WINDOWS (GENERIC)
short = cfg["WINDOWS"]["SHORT"]
medium = cfg["WINDOWS"]["MEDIUM"]
long = cfg["WINDOWS"]["LONG"]


z_win = cfg["WINDOWS"]["Z_SCORE"]

w_trend = cfg["WEIGHTS"]["TREND"]
w_mom = cfg["WEIGHTS"]["MOMENTUM"]
w_vol = cfg["WEIGHTS"]["VOLATILITY"]

std_mult = cfg["THRESHOLDS"]["REGIME_STD_MULTIPLIER"]
vol_z_high = cfg["THRESHOLDS"]["VOL_Z_HIGH"]


# ### ARCHITECTURE #########
# Put function inside detection.py
# Put it ABOVE detect_market_regime()
# Call it INSIDE detect_market_regime()
# Use it later in:
    # model selection
    # leverage scaling
    # ensemble weighting
    # stop-loss widening
    # portfolio exposure control

# This is the correct professional architecture.


# helper functions
# ==========================================================
# ==========================================================
# VOLATILITY REGIME DETECTION
# ==========================================================
def detect_volatility_regime(df):

    temp = df.copy()

    # Market-level volatility
    market_returns = (
        temp.groupby("Date")["Close"]
        .mean()
        .pct_change()
    )

    temp["Market_Return"] = market_returns

    market_series = (
        temp.groupby("Date")["Market_Return"]
        .mean()
    )

    market_vol = market_series.rolling(long).std()

    latest_vol = market_vol.iloc[-1]

    high_threshold = market_vol.quantile(0.8)
    low_threshold = market_vol.quantile(0.2)

    if latest_vol > high_threshold:
        return "HIGH_VOL"

    elif latest_vol < low_threshold:
        return "LOW_VOL"

    else:
        return "NORMAL_VOL"




# main regime engine
# ==========================================================
# 1. DETECT MARKET REGIME (FULLY GENERIC + VECTORISED)
# ==========================================================
def detect_market_regime(df):

    df = df.copy()
    df = df.sort_values(["Company", "Date"])

    # =========================
    # 1. GLOBAL VOL REGIME
    # =========================
    global_vol_regime = detect_volatility_regime(df)
    print(f"\n🌍 Global Volatility Regime: {global_vol_regime}")


    # =========================
    # 2. TREND FEATURES (SMA + EMA HYBRID)
    # =========================
    df["SMA_S"] = df.groupby("Company")["Close"].transform(lambda x: x.rolling(short).mean())
    df["SMA_L"] = df.groupby("Company")["Close"].transform(lambda x: x.rolling(long).mean())

    df["EMA_S"] = df.groupby("Company")["Close"].transform(lambda x: x.ewm(span=short, adjust=False).mean())
    df["EMA_M"] = df.groupby("Company")["Close"].transform(lambda x: x.ewm(span=medium, adjust=False).mean())

    # Trend strength
    df["Trend"] = (df["EMA_S"] - df["SMA_L"]) / (df["SMA_L"] + 1e-9)

    # =========================
    # 3. MOMENTUM + VOL
    # =========================
    df["Momentum"] = df.groupby("Company")["Close"].pct_change(short)

    df["Volatility"] = df.groupby("Company")["Close"].transform(
        lambda x: x.pct_change().rolling(short).std()
    )

    # =========================
    # 4. NORMALIZATION (ROLLING Z-SCORE)
    # =========================
    def rolling_z(x):
        return (x - x.rolling(z_win).mean()) / (x.rolling(z_win).std() + 1e-9)

    df["Trend_Z"] = df.groupby("Company")["Trend"].transform(rolling_z)
    df["Momentum_Z"] = df.groupby("Company")["Momentum"].transform(rolling_z)
    df["Vol_Z"] = df.groupby("Company")["Volatility"].transform(rolling_z)

    df["Trend_Z"] = df["Trend_Z"].clip(-5, 5)
    df["Momentum_Z"] = df["Momentum_Z"].clip(-5, 5)
    df["Vol_Z"] = df["Vol_Z"].clip(-5, 5)

    # =========================
    # 5. COMPOSITE SCORE (GENERIC)
    # =========================

    df["Regime_Score"] = (
        w_trend * df["Trend_Z"] +
        w_mom * df["Momentum_Z"] -
        w_vol * df["Vol_Z"]
    )


    # =========================
    # 6. ADAPTIVE THRESHOLDS (NO LEAKAGE)
    # =========================

    rolling_mean = df.groupby("Company")["Regime_Score"].transform(
        lambda x: x.rolling(z_win).mean()
    )
    rolling_std = df.groupby("Company")["Regime_Score"].transform(
        lambda x: x.rolling(z_win).std()
    )


    upper = rolling_mean + std_mult * rolling_std
    lower = rolling_mean - std_mult * rolling_std

    # =========================
    # 7. REGIME CLASSIFICATION (VECTORISED)
    # =========================

    df["Market_Regime"] = "SIDEWAYS"

    bull = df["Regime_Score"] > upper
    bear = df["Regime_Score"] < lower
    high_vol = df["Vol_Z"] > vol_z_high

    df.loc[bull & ~high_vol, "Market_Regime"] = "BULL"
    df.loc[bear & ~high_vol, "Market_Regime"] = "BEAR"
    df.loc[high_vol & bull, "Market_Regime"] = "BULL_VOLATILE"
    df.loc[high_vol & bear, "Market_Regime"] = "BEAR_VOLATILE"
    df.loc[high_vol & ~(bull | bear), "Market_Regime"] = "SIDEWAYS_VOLATILE"

    df["Market_Regime"] = df.groupby("Company")["Market_Regime"].transform(
        lambda x: x.ffill(limit=1)
    )
    # =========================
    # 8. REGIME STRENGTH
    # =========================
    df["Regime_Strength"] = df.groupby("Company")["Regime_Score"].transform(
        lambda x: (
            (x - x.rolling(z_win).min()) /
            (x.rolling(z_win).max() - x.rolling(z_win).min() + 1e-9)
        )
    ).fillna(0)



    print("\n📊 REGIME DISTRIBUTION:")
    print(df["Market_Regime"].value_counts(normalize=True))

    return df