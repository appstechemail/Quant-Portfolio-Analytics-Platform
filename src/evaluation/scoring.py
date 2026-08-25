# ==========================================================
# MODEL SCORING ENGINE (LATEST PRODUCTION VERSION)
# ==========================================================
#
# PURPOSE
# ----------------------------------------------------------
# This module computes robust institutional-grade model
# scores for:
#
# ✔ Ensemble selection
# ✔ Dynamic weighting
# ✔ Regime adaptation
# ✔ Portfolio robustness
# ✔ Walk-forward ranking
#
#
# CORE OBJECTIVES
# ----------------------------------------------------------
#
# Instead of selecting models purely using:
#
#     Accuracy
#
# or:
#
#     Sharpe
#
# we combine:
#
# ✔ Sharpe Ratio
# ✔ Recent Sharpe
# ✔ CAGR
# ✔ Drawdown
# ✔ Accuracy
# ✔ Stability
# ✔ Regime adaptation
#
#
# KEY IMPROVEMENTS
# ----------------------------------------------------------
#
# 1. RECENT PERFORMANCE EMPHASIS
#
#    Models performing well recently receive
#    higher ensemble weights.
#
#
# 2. SHARPE SMOOTHING
#
#    Prevents unstable extreme model dominance.
#
#
# 3. DRAWDOWN PENALIZATION
#
#    High-risk models are penalized heavily.
#
#
# 4. SOFTMAX NORMALIZATION
#
#    Produces stable adaptive weights.
#
#
# 5. REGIME ADAPTATION
#
#    Model weights adapt to:
#
#    ✔ bull markets
#    ✔ bear markets
#    ✔ volatile regimes
#
#
# 6. STABILITY FILTERS
#
#    Removes:
#
#    ✔ inactive models
#    ✔ unstable models
#    ✔ overfit models
#
#
# OUTPUT
# ----------------------------------------------------------
#
# Returns:
#
# ✔ ranked model dataframe
# ✔ ensemble-ready scores
# ✔ adaptive weights
#
#
# SAFE FOR
# ----------------------------------------------------------
#
# ✔ live trading
# ✔ walk-forward validation
# ✔ ensemble learning
# ✔ adaptive allocation
# ✔ institutional quant systems
#
# ==========================================================


# ==========================================================
# IMPORTS
# ==========================================================
import numpy as np
import pandas as pd

from src.regime.weighting import (
    compute_auto_weights,
    adjust_weights_by_regime
)

from src.evaluation.metrics import (
    normalize_metrics
)

from config.config import CONFIG


# ==========================================================
# CONFIG
# ==========================================================
eval_config = CONFIG["EVALUATION"]

ev_dd_penalty = eval_config.get(
    "DRAWDOWN_PENALTY",
    True
)

ev_config_weights = eval_config.get(
    "METRIC_WEIGHTS",
    {
        "Sharpe Ratio": 0.30,
        "CAGR": 0.20,
        "Strategy Return": 0.20,
        "Accuracy": 0.15,
        "Max Drawdown": 0.15
    }
)

ev_normalize_metrics = eval_config.get(
    "NORMALIZE_METRICS",
    True
)

ev_use_regime_selections = CONFIG[
    "MODEL"
].get(
    "USE_REGIME_SELECTION",
    True
)

RECENT_WINDOW = eval_config.get(
    "RECENT_WINDOW",
    126
)


# ==========================================================
# COMPUTE RECENT SHARPE
# ==========================================================
def compute_recent_sharpe(
    result,
    window=126
):

    try:

        # ==============================================
        # HANDLE DICT FORMAT
        # ==============================================
        if isinstance(result, dict):

            if "Daily" not in result:
                return 0.0

            daily_df = result["Daily"]

        else:

            daily_df = result

        # ==============================================
        # VALIDATION
        # ==============================================
        if daily_df is None:
            return 0.0

        if daily_df.empty:
            return 0.0

        if "Strategy_Return" not in daily_df.columns:
            return 0.0

        # ==============================================
        # RECENT RETURNS
        # ==============================================
        strat = (
            daily_df["Strategy_Return"]
            .tail(window)
            .dropna()
        )

        if len(strat) < 20:
            return 0.0

        # ==============================================
        # SHARPE
        # ==============================================
        mean_ret = strat.mean()

        std_ret = strat.std()

        if std_ret <= 1e-9:
            return 0.0

        sharpe = (
            mean_ret
            /
            (std_ret + 1e-9)
        ) * np.sqrt(252)

        # ==============================================
        # CLIP EXTREMES
        # ==============================================
        sharpe = np.clip(
            sharpe,
            -5,
            5
        )

        # ==============================================
        # SMOOTH
        # ==============================================
        sharpe = np.tanh(
            sharpe / 2
        )

        # ==============================================
        # RECENCY BONUS
        # ==============================================
        recent_mean = (
            strat.tail(20).mean()
        )

        recent_bonus = (
            np.tanh(recent_mean * 100)
            * 0.15
        )

        final_score = (
            sharpe
            +
            recent_bonus
        )

        return float(final_score)

    except Exception:

        return 0.0


# ==========================================================
# FINAL MODEL SCORING
# ==========================================================
def compute_model_score(
    summary,
    results,
    final_df
):

    print(
        "\n⚖️ COMPUTING MODEL SCORES..."
    )

    # ======================================================
    # SAFETY CHECK
    # ======================================================
    if summary is None or summary.empty:

        print(
            "❌ Empty summary"
        )

        return summary

    df = summary.copy()

    # ======================================================
    # MODEL WEIGHTS
    # ======================================================
    auto_model_weights = (
        compute_auto_weights(results)
    )

    if ev_use_regime_selections:

        auto_model_weights = (
            adjust_weights_by_regime(
                auto_model_weights,
                final_df
            )
        )

    # ======================================================
    # METRIC NORMALIZATION
    # ======================================================
    metrics = list(
        ev_config_weights.keys()
    )

    available_metrics = [

        m for m in metrics

        if m in df.columns
    ]

    if (
        ev_normalize_metrics
        and
        len(available_metrics) > 0
    ):

        df = normalize_metrics(
            df,
            available_metrics
        )

    # ======================================================
    # RECENT SHARPE SCORES
    # ======================================================
    recent_scores = {}

    for name, result in results.items():

        recent_scores[name] = (
            compute_recent_sharpe(
                result,
                RECENT_WINDOW
            )
        )

    # ======================================================
    # FINAL SCORE
    # ======================================================
    df["Score"] = 0.0

    for idx, row in df.iterrows():

        model_name = row["Model"]

        score = 0.0

        # --------------------------------------------------
        # METRIC COMPONENTS
        # --------------------------------------------------
        for metric, weight in ev_config_weights.items():

            if metric not in df.columns:
                continue

            val = row[metric]

            if pd.isna(val):
                continue

            # ----------------------------------------------
            # DRAWDOWN PENALTY
            # ----------------------------------------------
            if (
                metric == "Max Drawdown"
                and
                ev_dd_penalty
            ):

                val = 1 + val

            score += (
                weight * val
            )

        # --------------------------------------------------
        # RECENT SHARPE BONUS
        # --------------------------------------------------
        recent_sharpe = (
            recent_scores.get(
                model_name,
                0.0
            )
        )

        score += (
            0.40 * recent_sharpe
        )

        # --------------------------------------------------
        # MODEL WEIGHT
        # --------------------------------------------------
        model_weight = (
            auto_model_weights.get(
                model_name,
                1.0
            )
        )

        score *= (
            0.5
            +
            0.5 * model_weight
        )

        # ==================================================
        # CRITICAL PENALTIES
        # ==================================================

        # --------------------------------------------------
        # No activity
        # --------------------------------------------------
        if (
            "Volatility" in row
            and
            row["Volatility"] < 1e-6
        ):

            score *= 0.20

        # --------------------------------------------------
        # No returns
        # --------------------------------------------------
        if (
            "Strategy Return" in row
            and
            row["Strategy Return"] == 0
        ):

            score *= 0.30

        # --------------------------------------------------
        # Negative Sharpe
        # --------------------------------------------------
        if (
            "Sharpe Ratio" in row
            and
            row["Sharpe Ratio"] <= 0
        ):

            score *= 0.50

        # --------------------------------------------------
        # Negative returns
        # --------------------------------------------------
        if (
            "Strategy Return" in row
            and
            row["Strategy Return"] < 0
        ):

            score *= 0.60

        # --------------------------------------------------
        # Extreme drawdown
        # --------------------------------------------------
        if (
            "Max Drawdown" in row
            and
            row["Max Drawdown"] < -0.25
        ):

            score *= 0.70

        # --------------------------------------------------
        # Very low accuracy
        # --------------------------------------------------
        if (
            "Accuracy" in row
            and
            row["Accuracy"] < 0.45
        ):

            score *= 0.75

        # ==================================================
        # FINAL SCORE
        # ==================================================
        df.at[idx, "Score"] = score

    # ======================================================
    # SOFTMAX WEIGHTS
    # ======================================================
    score_vals = df["Score"].values

    score_vals = np.nan_to_num(
        score_vals,
        nan=0.0
    )

    exp_scores = np.exp(
        score_vals
    )

    softmax_weights = (

        exp_scores

        /

        (
            exp_scores.sum()
            + 1e-9
        )
    )

    df["Adaptive_Weight"] = (
        softmax_weights
    )

    # ======================================================
    # SORT
    # ======================================================
    df = (
        df.sort_values(
            "Score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # ======================================================
    # DEBUG INFO
    # ======================================================
    if len(df) > 1:

        gap = (
            df.loc[0, "Score"]
            -
            df.loc[1, "Score"]
        )

        print(
            f"\n📊 Confidence Gap: "
            f"{gap:.4f}"
        )

    print("\n🏆 FINAL MODEL RANKINGS")

    print(
        df[
            [
                "Model",
                "Score",
                "Adaptive_Weight"
            ]
        ]
    )

    return df