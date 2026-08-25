import numpy as np
from src.evaluation.metrics import compute_sharpe, compute_drawdown, compute_cagr

# ==========================================
# STRATEGY EVALUATION
# ==========================================
def evaluate_strategy(df):

    if df is None or df.empty:
        return {
            "Market Return": 0,
            "Strategy Return": 0,
            "Sharpe Ratio": 0,
            "Max Drawdown": 0,
            "CAGR": 0,
            "Volatility": 0
        }

    strategy = df["Strategy_Return"]
    market = df["Market"]

    cum_strategy = df["Cumulative_Strategy"]
    cum_market = df["Cumulative_Market"]

    return {
        "Market Return": cum_market.iloc[-1] - 1,
        "Strategy Return": cum_strategy.iloc[-1] - 1,
        "Sharpe Ratio": compute_sharpe(strategy),
        "Max Drawdown": compute_drawdown(cum_strategy),
        "CAGR": compute_cagr(cum_strategy),
        "Volatility": strategy.std() * np.sqrt(252)
    }
