# ==========================================
# IMPORTS
# ==========================================
import numpy as np
import pandas as pd
from config.config import CONFIG

eval_config = CONFIG["EVALUATION"]

ev_dd_penalty = eval_config["DRAWDOWN_PENALTY"]



# ==========================================
# METRIC FUNCTIONS
# ==========================================
def compute_sharpe(returns):
    if len(returns) == 0:
        return 0.0

    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0

    return (returns.mean() / std) * np.sqrt(252)


def compute_drawdown(cum_returns):
    if len(cum_returns) == 0:
        return 0.0

    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max

    return drawdown.min()


def compute_cagr(cum_returns):
    if len(cum_returns) < 2:
        return 0.0

    n_days = len(cum_returns)
    total_return = cum_returns.iloc[-1]

    if total_return <= 0:
        return 0.0

    return (total_return ** (252 / n_days)) - 1


# ==========================================
# ALPHA METRICS
# ==========================================
def compute_alpha_metrics(
    predictions,
    future_returns
):

    pred = np.asarray(
        predictions
    )

    ret = np.asarray(
        future_returns
    )

    valid = (
        np.isfinite(pred)
        &
        np.isfinite(ret)
    )

    pred = pred[valid]
    ret = ret[valid]

    if len(pred) < 5:

        return {

            "Spearman_IC": np.nan,

            "Pearson_IC": np.nan,

            "Rank_IC": np.nan

        }

    try:

        spearman_ic = spearmanr(
            pred,
            ret
        )[0]

    except Exception:

        spearman_ic = np.nan

    try:

        pearson_ic = pearsonr(
            pred,
            ret
        )[0]

    except Exception:

        pearson_ic = np.nan

    return {

        "Spearman_IC": spearman_ic,

        "Pearson_IC": pearson_ic,

        "Rank_IC": spearman_ic

    }

# ==========================================
# ICIR
# ==========================================
def compute_icir(
    ic_series
):

    ic_series = pd.Series(
        ic_series
    ).dropna()

    if len(ic_series) < 2:
        return np.nan

    mean_ic = (
        ic_series.mean()
    )

    std_ic = (
        ic_series.std()
    )

    return (
        mean_ic
        /
        (std_ic + 1e-9)
    )


# ==========================================
# NORMALIZATION
# ==========================================
def normalize_metrics(df, metrics):

    for col in metrics:
        if col in df.columns:
            df[col] = (df[col] - df[col].min()) / (
                df[col].max() - df[col].min() + 1e-9
            )

    return df

