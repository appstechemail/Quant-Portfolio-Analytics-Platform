# features.py

# ├── FEATURES_DEF
# ├── FEATURE_METADATA
# ├── FEATURE_CATEGORIES      (optional helper)
# ├── CATEGORY_LIST           (optional helper)
# ├── FEATURE_GROUPS          (optional helper)


# It becomes the one place that answers:

# What features exist?
# What category do they belong to?
# What family are they in?
# Are they technical, fundamental, macro, cross-sectional?
# How should they be budgeted?
# How should they be interpreted?


FEATURES_DEF = [
    "Return","Log_Return","MA10","MA20","MA50",
    "Volatility","Momentum","RSI","MACD","NSE_Return",
    "PE","EPS","Close_Lag1","Close_Lag2","Close_Lag3","Close_Lag5",
    "Trend","Volume_Change","Return_5","Return_10",
    "EPS_Growth","PE_Change","Value_Score",
    "PE_Rank","Value_Rank",
    "Momentum_Rank","Volatility_Rank"
    ]


FEATURE_CATEGORY_MAP = {

    # ======================================
    # PRICE
    # ======================================

    "Return": "Price",
    "Log_Return": "Price",
    "Return_5": "Price",
    "Return_10": "Price",

    # ======================================
    # TREND
    # ======================================

    "MA10": "Trend",
    "MA20": "Trend",
    "MA50": "Trend",
    "Trend": "Trend",

    # ======================================
    # MOMENTUM
    # ======================================

    "Momentum": "Momentum",
    "RSI": "Momentum",
    "MACD": "Momentum",

    # ======================================
    # VOLATILITY
    # ======================================

    "Volatility": "Volatility",

    # ======================================
    # VOLUME
    # ======================================

    "Volume_Change": "Volume",

    # ======================================
    # LAGS
    # ======================================

    "Close_Lag1": "Lag",
    "Close_Lag2": "Lag",
    "Close_Lag3": "Lag",
    "Close_Lag5": "Lag",

    # ======================================
    # VALUE
    # ======================================

    "PE": "Value",
    "EPS": "Value",
    "PE_Change": "Value",
    "Value_Score": "Value",

    # ======================================
    # GROWTH
    # ======================================

    "EPS_Growth": "Growth",

    # ======================================
    # RANKS
    # ======================================

    "PE_Rank": "CrossSection",
    "Value_Rank": "CrossSection",
    "Momentum_Rank": "CrossSection",
    "Volatility_Rank": "CrossSection",

    # ======================================
    # MARKET
    # ======================================

    "NSE_Return": "Market",

}


FEATURE_METADATA = {

    "Return": {
        "category": "Price",
        "group": "Returns",
    },

    "Log_Return": {
        "category": "Price",
        "group": "Returns",
    },

    "Momentum": {
        "category": "Momentum",
        "group": "Momentum",
    },

    "Momentum_ATR": {
        "category": "Momentum",
        "group": "Momentum",
    },

    "RSI": {
        "category": "Momentum",
        "group": "Oscillator",
    },

    "MACD": {
        "category": "Momentum",
        "group": "Oscillator",
    },

    "Volatility": {
        "category": "Volatility",
        "group": "Risk",
    },

    "PE": {
        "category": "Value",
        "group": "Fundamental",
    },

    "EPS": {
        "category": "Value",
        "group": "Fundamental",
    },

}