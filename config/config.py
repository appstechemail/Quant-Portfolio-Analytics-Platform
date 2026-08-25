from datetime import datetime

# ==========================================
# MASTER CONFIG
# ==========================================
CONFIG = {

    # =========================
    # 1. DATA CONFIG
    # =========================
    "DATA": {
        "START_DATE": "2014-01-01",
        "END_DATE": None,

        "STOCKS": {
            "TBOTEK.NS": "TBO TEK",
            "NSDL.BO": "NSDL",
            "MAPMYINDIA.NS": "C. E. INFO",
            "SUNPHARMA.NS": "SUN PHARMA",
            "PFC.NS": "POWER FINANCE",
            "DRREDDY.NS": "DR REDDY",
            "CIPLA.NS": "CIPLA",
            "TORNTPHARM.NS": "TORRENT PHARMA",
            "CRAMC.NS": "Canara Robeco Asset",
            "SUZLON.NS": "Suzlon Energy Ltd",
            "IREDA.NS": "Indian Renewable Energy",
            "ICICIAMC.NS": "ICICI Prudential Asset",
            "ORKLAINDIA.NS": "Orkla India Ltd",
            "YESBANK.NS": "Yes Bank Ltd",
            "BAJAJHFL.NS": "Bajaj Housing Finance",
            "HDBFS.NS": "HDB Financial Services",
            "POWERINDIA.NS": "Hitachi Energy",
            "INOXINDIA.NS": "Inox India Ltd",
            "CPPLUS.NS": "Aditya Infotech Ltd",
            "TCS.NS": "Tata Consultancy",
            "INFY.NS": "Infosys Ltd",
            "ADVENZYMES.NS": "Advanced Enzyme",
            "CAPLIPOINT.NS": "Caplin Point Lab",
            "CDSL.NS": "Central Depository",
            "COALINDIA.NS": "Coal India",
            "CUPID.NS": "Cupid Ltd",
            "HUDCO.NS": "HUDCO",
            "HAL.NS": "Hindustan Aeronautics",
            "ICICIBANK.NS": "ICICI Bank Ltd",
            "SBIN.NS": "STATE Bank of India",
            "ITC.NS": "ITC Ltd",
            "JISLJALEQS.NS": "Jain Irrigations",
            "MCX.NS": "Multi Commodity Exchange",
            "NTPCGREEN.NS": "NTPC Green",
            "PCJEWELLER.NS": "PC Jeweller",
            "SHAKTIPUMP.NS": "Shakti Pumps",
            "UNITDSPR.NS": "United Spirits",
            "UBL.NS": "United Breweries Ltd",
            "MUTHOOTFIN.NS": "Muthoot Finance",
            "EPACK.NS": "Epack Durable Ltd",
            "LGEINDIA.NS": "LG Electronics",
            "IDEAFORGE.NS": "Ideaforge Tech",
            "SHREEJISPG.NS": "Shreeji Shipping",
            "LOTUSDEV.NS": "Sri Lotus Developers",
            "UNIECOM.NS": "Unicommerce eSolutions"
        },

        "PATHS": {
            "EPS_FILE": "data/ALL_STOCKS_EPS.csv",
            "FINAL_DF": "data/final_df.csv",
            "ARTIFACTS": "artifacts/"
        }
    },

    # =========================
    # 2. FEATURE CONFIG
    # =========================
    "FEATURES": {
        "USE_TECHNICAL": True,
        "USE_FUNDAMENTAL": True,

        "MA_WINDOWS": [10, 20, 50, 200],
        "EMA_WINDOWS": [10, 20, 50],

        "RSI_WINDOW": 14,
        "VOL_WINDOW": 10,
        "MOMENTUM_WINDOW": 10,

        "LAG_PERIODS": [1, 2, 3, 5, 10]
    },

    # =========================
    # 3. TARGET CONFIG
    # =========================
    "TARGET": {
        "PREDICT_DAYS_AHEAD": 5,
        "USE_NEUTRAL_CLASS": False,

        # Adaptive threshold
        "USE_DYNAMIC_THRESHOLD": True,
        "BASE_THRESHOLD": 0.02,

        # volatility multiplier (used if dynamic)
        "VOL_MULTIPLIER": 1.25,
        "TOP_PERCENTILE": 0.80,
        "VOL_LOOKBACK": 20,

        # Triple-barrier labeling
        "HOLDING_DAYS": 5,
        "TAKE_PROFIT": 0.08,
        "STOP_LOSS": 0.04,
        # Meta Labels
        "META_RETURN_THRESHOLD":0.02,
        "META_THRESHOLD": 0.60,
        "BULL_META_THRESHOLD": 0.60,
        "SIDEWAYS_META_THRESHOLD": 0.70,
        "BULL_VOL_META_THRESHOLD": 0.75,
        "SIDEWAYS_VOL_META_THRESHOLD": 0.80,

    },

    # =========================
    # 4. MODEL CONFIG
    # =========================
    "MODEL": {
        "TRAIN_SIZE": 0.8,
        "THRESHOLD": 0.5,

        # Model control
        "USE_ENSEMBLE": True,
        "USE_REGIME_SELECTION": True,
        "USE_DYNAMIC_WEIGHTS": True,

        # Scaling
        "USE_SCALING": True,

        # Models list (fully dynamic)
        "MODEL_LIST": ["xgb", "rf", "lr", "mlp"]
    },

    # =========================
    # 5. REGIME CONFIG
    # =========================
    "REGIME": {
        "USE_REGIME": True,

        # SMA / EMA hybrid
        "SMA_FAST": 50,
        "SMA_SLOW": 200,
        "EMA_FAST": 10,
        "EMA_SLOW": 20,

        # Lookback
        "RETURN_WINDOW": 20,
        "VOL_WINDOW": 20,

        # Quantile thresholds (adaptive)
        "HIGH_QUANTILE": 0.7,
        "LOW_QUANTILE": 0.3,
        "WINDOWS":{
            "SHORT": 20,
            "MEDIUM": 50,
            "LONG": 200,
            "Z_SCORE": 100,
        },
        "WEIGHTS": {
            "TREND": 1.0,
            "MOMENTUM": 1.0,
            "VOLATILITY": 0.5
        },

        "METRICS": {
            "SHARPE": 0.5,
            "DRAWDOWN": 0.3,
            "VOLATILITY": 0.2
        },

        "THRESHOLDS": {
            "REGIME_STD_MULTIPLIER": 0.5,
            "VOL_Z_HIGH": 1.0
        },
        "MODEL_SELECTION": {
            "TOP_K_PCT": 0.4,
            "MIN_MODELS": 2
        },

        "NORMALIZE": True,
        "USE_SOFTMAX": True
    },

    # =========================
    # 6. BACKTEST CONFIG
    # =========================
    "BACKTEST": {
        "MIN_HOLD_DAYS": 10,
        "LIQUIDITY_THRESHOLD":0.0, # can be 10_000_000
        "BUY_PCT": 0.03,
        "SELL_PCT": 0.20,
        "TOP_PCT": 0.10,
        "MIN_CONFIDENCE": 0.02, # can be 0.20
        "ENTRY_CONFIDENCE": 0.20, # can be 0.15, 0.20
        "EXIT_CONFIDENCE": 0.10, # can be 0.08, 0.10
        "TURNOVER_BAND": 0.10, # may be 0.02, 0.05 or 0.07
        "SMOOTHING": 0.00, 
        "EXECUTION_LAG": 1,

        "MAX_PORTFOLIO_SIZE": 3,
        "MAX_POSITION_SIZE": 0.50,
        "MAX_GROSS_EXPOSURE": 1.00,

        "VOL_WINDOW": 20,
        "TARGET_VOL": 0.02,

        "TRANSACTION_COST": 0.0005,
        "SLIPPAGE": 0.0005,

        "MAX_POSITION": 1.0,
        "CLIP_RETURN": 0.2,

        # Portfolio control
        "LONG_SHORT": True,
        "NEUTRALIZE": True,
        "NEUTRALITY": 0.5,
        "RETURN_CLIP": 0.2,

        "USE_CROSS_SECTIONAL_RANK": True,
        "USE_LIQUIDITY_FILTER": False,
        "USE_VOL_TARGET": True,
        "USE_REGIME_EXPOSURE": True,
        "USE_DEADBAND": False,
    },

    # =========================
    # 7. PORTFOLIO CONFIG
    # =========================
    "PORTFOLIO": {
        "CAPITAL": 500000,

        "MAX_STOCKS": 10,
        "MIN_WEIGHT": 0.01,

        # Risk management
        "STOP_LOSS_MULTIPLIER": 2,
        "TARGET_MULTIPLIER": 3,
        "MIN_PORTFOLIO_SCORE": 0.05,

        "BUY_THRESHOLD": 0.60,
        "STRONG_BUY_THRESHOLD": 0.75,

        "SELL_THRESHOLD": 0.40,
        "STRONG_SELL_THRESHOLD": 0.25,

        "MIN_CONFIDENCE": 0.30,
        "MIN_PORTFOLIO_SIZE":5,
        "USE_DYNAMIC_CONFIDENCE": True,
    },

    # =========================
    # 8. ENSEMBLE CONFIG
    # =========================
    "ENSEMBLE": {
        "METHOD": "weighted",   # options: equal, weighted, softmax
        "SOFTMAX_TEMP": 3,

        # domination control
        "MAX_MODEL_WEIGHT": 0.7,
        "MIN_MODEL_WEIGHT": 0.05
    },

    # =========================
    # 9. LOGGING CONFIG
    # =========================
    "LOGGING": {
        "VERBOSE": True,
        "SAVE_RESULTS": False
    },

    # =========================
    # 10. EVALUATION CONFIG
    # =========================
    "EVALUATION": {

        # Metric weights (must sum ~1)
        "METRIC_WEIGHTS": {
            "Sharpe Ratio": 0.4,
            "Strategy Return": 0.3,
            "CAGR": 0.2,
            "Max Drawdown": 0.1
        },

        # Auto-normalization
        "NORMALIZE_METRICS": True,

        # Penalize drawdown
        "DRAWDOWN_PENALTY": True
    },

    # =========================
    # 11. IC CONFIG
    # =========================

    "IC_CONFIG": {
        "MIN_ABS_IC": 0.18,
        "GLOBAL": {
            "MIN_ICIR": 0.05,
            "MIN_POSITIVE_PCT": 0.50,
            "MIN_OBSERVATIONS": 1500,
            "CORR_THRESHOLD": 0.90,
            "TOP_K": 10
        },

        "ROLLING": {
            "MIN_IC": 0.02,
            "MIN_ICIR": 0.05,
            "TOP_K": 10,
            "ROLLING_MIN_MULTIPLIER": 0.80,
            "ROLLING_MAX_MULTIPLIER": 1.20,
        },

        "REGIME": {
            "MIN_ICIR": 0.05,
            "MIN_POSITIVE_PCT": 0.50,
            "MIN_OBSERVATIONS": 500,
            "REGIME_MIN_MULTIPLIER": 0.80,
            "REGIME_MAX_MULTIPLIER": 1.20,
            "TOP_K": 10
        },

        "CLUSTERING": {
            "METHOD": "hierarchical",
            "DISTANCE": "spearman",
            "CORR_THRESHOLD": 0.80,
            "MIN_CLUSTER_SIZE": 2,
            "KEEP_TOP_PER_CLUSTER": 1
        },

        "DECAY": {
            "ROLLING_WINDOW": 60,
            "MIN_OBSERVATIONS": 30,
            "MIN_RECENT_RATIO": 0.60,
            "MIN_TREND": -0.0005,
            "DECAY_MIN_MULTIPLIER": 0.80,
            "DECAY_MAX_MULTIPLIER": 1.20,
            "TOP_K": 10,

        },

        "ADAPTIVE_WEIGHTING": {
            "GLOBAL_POWER": 1.00,
            "ROLLING_POWER": 0.50,
            "STABILITY_POWER": 0.50,
            "REGIME_POWER": 0.40,
            "DECAY_POWER": 0.40,
            "DIVERSIFICATION_POWER": 0.30,
            "MIN_MULTIPLIER": 0.50,
            "MAX_MULTIPLIER": 1.50,
        },

        "DYNAMIC_CATEGORY_BUDGET": {
            "CATEGORY_CONCENTRATION_POWER": 1.30,
        },

        "VALID_CORRELATION_METHODS":  {
            "pearson",
            "spearman",
            "kendall",
        }

    }

}

# ==========================================
# DERIVED VARIABLES (BACKWARD COMPATIBILITY)
# ==========================================

DATA = CONFIG["DATA"]

START_DATE = DATA["START_DATE"]

END_DATE = DATA["END_DATE"]
if END_DATE is None:
    END_DATE = datetime.today().strftime("%Y-%m-%d")

stocks = DATA["STOCKS"]

