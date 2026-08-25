# ============================================================
# PORTFOLIO SELECTION ENGINE
# ============================================================
#
# FILE:
# -----
# portfolio_selection.py
#
# PURPOSE:
# --------
# Converts model predictions + alpha features into an
# institutional candidate set for portfolio construction.
#
# RESPONSIBILITIES
# ----------------
# 1. IC-weighted alpha scoring
# 2. Cross-sectional normalization
# 3. Risk-aware signal adjustment
# 4. Probability / confidence eligibility
# 5. Composite selection score
# 6. Cross-sectional ranking
# 7. Entry / exit hysteresis
# 8. Candidate selection
#
# IMPORTANT ARCHITECTURAL RULE
# ----------------------------
# This module DOES NOT perform institutional portfolio
# optimization or final position sizing.
#
# Final portfolio construction is handled by:
#
#     portfolio_builder.py
#
# which owns:
#
#     Forecast
#     Risk
#     Constraints
#     Optimization
#     Portfolio Assembly
#     Rebalance
#     Validation
#     Reporting
#
# Therefore:
#
#     portfolio_selection.py
#             |
#             v
#     selected/ranked candidates
#             |
#             v
#     portfolio_builder.py
#             |
#             v
#     optimized portfolio
#
# ============================================================

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config.config import CONFIG
from src.alpha.ic_loader import load_ic_weights


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

PORTFOLIO_CFG = CONFIG.get(
    "PORTFOLIO",
    {}
)

SELECTION_CFG = CONFIG.get(
    "PORTFOLIO_SELECTION",
    {}
)


# ------------------------------------------------------------
# Selection limits
# ------------------------------------------------------------

MIN_PORTFOLIO_SCORE = float(
    SELECTION_CFG.get(
        "MIN_PORTFOLIO_SCORE",
        PORTFOLIO_CFG.get(
            "MIN_PORTFOLIO_SCORE",
            0.05,
        ),
    )
)

MIN_CONFIDENCE = float(
    SELECTION_CFG.get(
        "MIN_CONFIDENCE",
        PORTFOLIO_CFG.get(
            "MIN_CONFIDENCE",
            0.30,
        ),
    )
)

MIN_PORTFOLIO_SIZE = int(
    SELECTION_CFG.get(
        "MIN_PORTFOLIO_SIZE",
        PORTFOLIO_CFG.get(
            "MIN_PORTFOLIO_SIZE",
            5,
        ),
    )
)

USE_DYNAMIC_CONFIDENCE = bool(
    SELECTION_CFG.get(
        "USE_DYNAMIC_CONFIDENCE",
        PORTFOLIO_CFG.get(
            "USE_DYNAMIC_CONFIDENCE",
            True,
        ),
    )
)


# ------------------------------------------------------------
# Expected return clipping
# ------------------------------------------------------------

EXPECTED_RETURN_MIN = float(
    SELECTION_CFG.get(
        "EXPECTED_RETURN_MIN",
        -0.25,
    )
)

EXPECTED_RETURN_MAX = float(
    SELECTION_CFG.get(
        "EXPECTED_RETURN_MAX",
        0.25,
    )
)


# ------------------------------------------------------------
# Composite score weights
# ------------------------------------------------------------

DEFAULT_SCORE_WEIGHTS = {
    "prediction": 0.35,
    "confidence": 0.15,
    "risk_reward": 0.10,
    "expected_return": 0.15,
    "alpha": 0.25,
}

SCORE_WEIGHTS = {
    key: float(
        SELECTION_CFG.get(
            f"SCORE_WEIGHT_{key.upper()}",
            value,
        )
    )
    for key, value in DEFAULT_SCORE_WEIGHTS.items()
}


# ------------------------------------------------------------
# Risk penalty weights
# ------------------------------------------------------------

RISK_VOL_WEIGHT = float(
    SELECTION_CFG.get(
        "RISK_VOL_WEIGHT",
        0.80,
    )
)

RISK_GAP_WEIGHT = float(
    SELECTION_CFG.get(
        "RISK_GAP_WEIGHT",
        0.20,
    )
)


# ------------------------------------------------------------
# Concentration penalty
# ------------------------------------------------------------

CONCENTRATION_PENALTY = float(
    SELECTION_CFG.get(
        "CONCENTRATION_PENALTY",
        0.03,
    )
)

CONCENTRATION_PERCENTILE = float(
    SELECTION_CFG.get(
        "CONCENTRATION_PERCENTILE",
        0.95,
    )
)


# ------------------------------------------------------------
# Risk penalty contribution
# ------------------------------------------------------------

RISK_SCORE_PENALTY = float(
    SELECTION_CFG.get(
        "RISK_SCORE_PENALTY",
        0.25,
    )
)


# ------------------------------------------------------------
# Default regime probability thresholds
# ------------------------------------------------------------

DEFAULT_REGIME_PROB_THRESHOLDS = {
    "BULL": 0.50,
    "BULL_VOLATILE": 0.55,
    "SIDEWAYS": 0.55,
    "SIDEWAYS_VOLATILE": 0.60,
    "BEAR": 0.60,
    "BEAR_VOLATILE": 0.65,
}


REGIME_PROB_THRESHOLDS = (
    SELECTION_CFG.get(
        "REGIME_PROB_THRESHOLDS",
        DEFAULT_REGIME_PROB_THRESHOLDS,
    )
)


DEFAULT_FALLBACK_PROBABILITY = float(
    SELECTION_CFG.get(
        "DEFAULT_PROBABILITY_THRESHOLD",
        0.60,
    )
)


# ------------------------------------------------------------
# Hysteresis
# ------------------------------------------------------------

DEFAULT_ENTRY_RANK = int(
    SELECTION_CFG.get(
        "ENTRY_RANK",
        3,
    )
)

DEFAULT_EXIT_RANK = int(
    SELECTION_CFG.get(
        "EXIT_RANK",
        5,
    )
)


# ============================================================
# SAFE NUMERIC HELPERS
# ============================================================

def _safe_numeric(
    series: pd.Series,
    default: float = 0.0,
) -> pd.Series:

    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(default)
    )


# ============================================================
# DATE COLUMN
# ============================================================

def _get_date_column(
    df: pd.DataFrame,
) -> str:
    """
    Return the preferred cross-sectional date column.

    Signal_Date is preferred when available because it
    represents the signal-generation date.
    """

    if "Signal_Date" in df.columns:
        return "Signal_Date"

    if "Date" in df.columns:
        return "Date"

    raise ValueError(
        "portfolio_selection requires either "
        "'Signal_Date' or 'Date'."
    )


# ============================================================
# SAFE Z-SCORE
# ============================================================

def safe_zscore(
    x: pd.Series,
) -> pd.Series:
    """
    Robust z-score.

    A constant or invalid cross-section receives zero
    instead of producing NaNs.
    """

    values = _safe_numeric(
        x,
        default=0.0,
    )

    std = values.std(
        ddof=0
    )

    if (
        pd.isna(std)
        or std < 1e-9
    ):
        return pd.Series(
            0.0,
            index=x.index,
        )

    return (
        values - values.mean()
    ) / (
        std + 1e-9
    )


# ============================================================
# CROSS-SECTIONAL Z-SCORE
# ============================================================

def cross_sectional_zscore(
    df: pd.DataFrame,
    column: str,
    date_column: str | None = None,
) -> pd.Series:
    """
    Calculate cross-sectional z-score by signal date.
    """

    if column not in df.columns:
        return pd.Series(
            0.0,
            index=df.index,
        )

    if date_column is None:
        date_column = _get_date_column(
            df
        )

    values = (
        df.groupby(
            date_column,
            sort=False,
        )[column]
        .transform(
            safe_zscore
        )
    )

    return values.fillna(
        0.0
    )


# ============================================================
# FACTOR WEIGHT NORMALIZATION
# ============================================================

def _normalize_factor_weights(
    factor_weights: dict[str, float],
) -> dict[str, float]:
    """
    Normalize factor weights safely.

    Absolute-weight normalization is used so signed IC weights
    remain signed while the total contribution remains stable.
    """

    cleaned = {}

    for factor, weight in (
        factor_weights or {}
    ).items():

        try:
            value = float(
                weight
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if not np.isfinite(
            value
        ):
            continue

        cleaned[factor] = value

    if not cleaned:
        return {}

    total = sum(
        abs(v)
        for v in cleaned.values()
    )

    if (
        not np.isfinite(total)
        or total <= 1e-12
    ):
        return {}

    return {
        factor: weight / total
        for factor, weight
        in cleaned.items()
    }


# ============================================================
# IC-WEIGHTED FACTOR BLENDING
# ============================================================

def compute_alpha_score(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the cross-sectional Alpha_Score.

    IC weights are loaded from the alpha IC engine.

    Only factors actually present in the dataframe are used.
    """

    if df is None:
        raise ValueError(
            "compute_alpha_score received None."
        )

    if df.empty:
        return df.copy()

    df = df.copy()

    date_column = _get_date_column(
        df
    )

    # --------------------------------------------------------
    # Load IC weights
    # --------------------------------------------------------

    try:
        factor_weights = (
            load_ic_weights()
        )
    except Exception as exc:

        logger.warning(
            "Unable to load IC weights: %s",
            exc,
        )

        factor_weights = {}

    # --------------------------------------------------------
    # Fallback weights
    # --------------------------------------------------------

    if not factor_weights:

        factor_weights = (
            SELECTION_CFG.get(
                "FALLBACK_FACTOR_WEIGHTS",
                {
                    "Prediction_Alpha": 0.50,
                    "Trend_Strength": 0.50,
                },
            )
        )

    factor_weights = (
        _normalize_factor_weights(
            factor_weights
        )
    )

    # --------------------------------------------------------
    # Detect usable factors
    # --------------------------------------------------------

    factor_cols = [
        factor
        for factor in factor_weights
        if factor in df.columns
    ]

    # --------------------------------------------------------
    # No usable factors
    # --------------------------------------------------------

    if not factor_cols:

        logger.warning(
            "No IC-weighted factors available "
            "in portfolio selection dataframe."
        )

        df["Alpha_Score"] = 0.0

        return df

    # --------------------------------------------------------
    # Cross-sectional normalization
    # --------------------------------------------------------

    for column in factor_cols:

        df[column] = _safe_numeric(
            df[column]
        )

        df[
            f"{column}_CSZ"
        ] = (
            df.groupby(
                date_column,
                sort=False,
            )[column]
            .transform(
                safe_zscore
            )
        )

    # --------------------------------------------------------
    # Active weights
    # --------------------------------------------------------

    active_weights = {
        factor: factor_weights[factor]
        for factor in factor_cols
    }

    active_weights = (
        _normalize_factor_weights(
            active_weights
        )
    )

    # --------------------------------------------------------
    # Alpha score
    # --------------------------------------------------------

    df["Alpha_Score"] = 0.0

    for factor, weight in (
        active_weights.items()
    ):

        normalized_column = (
            f"{factor}_CSZ"
        )

        if normalized_column not in df.columns:
            continue

        df["Alpha_Score"] += (
            weight
            *
            df[normalized_column]
        )

    df["Alpha_Score"] = (
        df["Alpha_Score"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return df


# ============================================================
# RISK PENALTY
# ============================================================

def compute_risk_penalty(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute a cross-sectional risk penalty.

    Current components:
        - Volatility
        - Absolute Gap
    """

    if df is None:
        raise ValueError(
            "compute_risk_penalty received None."
        )

    if df.empty:
        return df.copy()

    df = df.copy()

    date_column = _get_date_column(
        df
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    if "Volatility" in df.columns:

        volatility = _safe_numeric(
            df["Volatility"]
        )

        vol_penalty = (
            volatility
            .groupby(
                df[date_column],
                sort=False,
            )
            .transform(
                safe_zscore
            )
        )

    else:

        vol_penalty = pd.Series(
            0.0,
            index=df.index,
        )

    # --------------------------------------------------------
    # Gap
    # --------------------------------------------------------

    if "Gap" in df.columns:

        gap_abs = (
            _safe_numeric(
                df["Gap"]
            )
            .abs()
        )

        df["Gap_Abs"] = (
            gap_abs
        )

        gap_penalty = (
            gap_abs
            .groupby(
                df[date_column],
                sort=False,
            )
            .transform(
                safe_zscore
            )
        )

    else:

        gap_penalty = pd.Series(
            0.0,
            index=df.index,
        )

    # --------------------------------------------------------
    # Final risk penalty
    # --------------------------------------------------------

    total_risk_weight = (
        abs(RISK_VOL_WEIGHT)
        +
        abs(RISK_GAP_WEIGHT)
    )

    if total_risk_weight <= 1e-12:

        df["Risk_Penalty"] = 0.0

        return df

    vol_weight = (
        RISK_VOL_WEIGHT
        /
        total_risk_weight
    )

    gap_weight = (
        RISK_GAP_WEIGHT
        /
        total_risk_weight
    )

    df["Risk_Penalty"] = (
        vol_weight
        *
        vol_penalty
        +
        gap_weight
        *
        gap_penalty
    )

    df["Risk_Penalty"] = (
        df["Risk_Penalty"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return df


# ============================================================
# STANDARDIZE MODEL COLUMNS
# ============================================================

def _standardize_model_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize prediction-related columns.

    Prediction_Alpha is the primary selection probability.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Prediction Alpha
    # --------------------------------------------------------

    if (
        "Prediction_Alpha"
        not in df.columns
    ):

        if "Prediction_Prob" in df.columns:

            df["Prediction_Alpha"] = (
                _safe_numeric(
                    df["Prediction_Prob"],
                    default=0.50,
                )
            )

        elif "Probability" in df.columns:

            df["Prediction_Alpha"] = (
                _safe_numeric(
                    df["Probability"],
                    default=0.50,
                )
            )

        else:

            df["Prediction_Alpha"] = 0.50

    else:

        df["Prediction_Alpha"] = (
            _safe_numeric(
                df["Prediction_Alpha"],
                default=0.50,
            )
        )

    # --------------------------------------------------------
    # Prediction probability compatibility
    # --------------------------------------------------------

    if (
        "Prediction_Prob"
        not in df.columns
    ):

        df["Prediction_Prob"] = (
            df["Prediction_Alpha"]
        )

    else:

        df["Prediction_Prob"] = (
            _safe_numeric(
                df["Prediction_Prob"],
                default=0.50,
            )
        )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    if (
        "Confidence"
        not in df.columns
    ):

        df["Confidence"] = (
            np.abs(
                df["Prediction_Alpha"]
                - 0.50
            )
            * 2.0
        )

    df["Confidence"] = (
        _safe_numeric(
            df["Confidence"]
        )
        .clip(
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Expected Return
    # --------------------------------------------------------

    if (
        "Expected_Return"
        not in df.columns
    ):

        df["Expected_Return"] = 0.0

    df["Expected_Return"] = (
        _safe_numeric(
            df["Expected_Return"]
        )
        .clip(
            EXPECTED_RETURN_MIN,
            EXPECTED_RETURN_MAX,
        )
    )

    # --------------------------------------------------------
    # Risk Reward
    # --------------------------------------------------------

    if (
        "RR_Ratio"
        not in df.columns
    ):

        df["RR_Ratio"] = 0.0

    df["RR_Ratio"] = (
        _safe_numeric(
            df["RR_Ratio"]
        )
        .clip(
            lower=0.0
        )
    )

    return df


# ============================================================
# REGIME PROBABILITY THRESHOLD
# ============================================================

def _add_probability_threshold(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    if "Market_Regime" in df.columns:

        df["Min_Prob"] = (
            df["Market_Regime"]
            .map(
                REGIME_PROB_THRESHOLDS
            )
            .fillna(
                DEFAULT_FALLBACK_PROBABILITY
            )
        )

    else:

        df["Min_Prob"] = (
            DEFAULT_FALLBACK_PROBABILITY
        )

    return df


# ============================================================
# QUALITY FILTER
# ============================================================

def _apply_quality_filters(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply probability and confidence eligibility.

    The original universe is retained for fallback selection.
    """

    if df.empty:
        return df.copy()

    df = _standardize_model_columns(
        df
    )

    df = _add_probability_threshold(
        df
    )

    candidate_df = df.copy()

    # --------------------------------------------------------
    # Probability filter
    # --------------------------------------------------------

    probability_mask = (
        df["Prediction_Alpha"]
        >=
        df["Min_Prob"]
    )

    after_probability = int(
        probability_mask.sum()
    )

    # --------------------------------------------------------
    # Confidence threshold
    # --------------------------------------------------------

    probability_candidates = (
        df.loc[
            probability_mask
        ]
    )

    if probability_candidates.empty:

        confidence_threshold = (
            MIN_CONFIDENCE
        )

    elif USE_DYNAMIC_CONFIDENCE:

        confidence_threshold = max(
            MIN_CONFIDENCE,
            float(
                probability_candidates[
                    "Confidence"
                ].quantile(
                    0.50
                )
            ),
        )

    else:

        confidence_threshold = (
            MIN_CONFIDENCE
        )

    confidence_mask = (
        df["Confidence"]
        >=
        confidence_threshold
    )

    combined_mask = (
        probability_mask
        &
        confidence_mask
    )

    df = df.loc[
        combined_mask
    ].copy()

    logger.info(
        "Portfolio selection filters | "
        "Initial=%d | Probability=%d | "
        "Confidence=%d | Final=%d",
        len(candidate_df),
        after_probability,
        int(confidence_mask.sum()),
        len(df),
    )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if (
        len(df)
        <
        MIN_PORTFOLIO_SIZE
    ):

        fallback_count = min(
            MIN_PORTFOLIO_SIZE,
            len(candidate_df),
        )

        if fallback_count > 0:

            fallback_sort_columns = [
                "Confidence",
                "Prediction_Alpha",
                "Alpha_Score",
            ]

            fallback_sort_columns = [
                col
                for col
                in fallback_sort_columns
                if col in candidate_df.columns
            ]

            if fallback_sort_columns:

                df = (
                    candidate_df
                    .sort_values(
                        fallback_sort_columns,
                        ascending=False,
                    )
                    .head(
                        fallback_count
                    )
                    .copy()
                )

            else:

                df = (
                    candidate_df
                    .head(
                        fallback_count
                    )
                    .copy()
                )

            df["Selection_Fallback"] = 1

            logger.warning(
                "Portfolio selection fallback applied: "
                "%d candidates.",
                len(df),
            )

    else:

        df["Selection_Fallback"] = 0

    return df


# ============================================================
# FINAL SELECTION SCORE
# ============================================================

def compute_final_score(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the composite stock-selection score.

    IMPORTANT:
    ----------
    Final_Score is a ranking/selection signal.

    It is NOT the final portfolio weight.

    Portfolio weights are produced later by the
    institutional portfolio builder.
    """

    if df is None:
        raise ValueError(
            "compute_final_score received None."
        )

    if df.empty:
        return df.copy()

    df = df.copy()

    date_column = _get_date_column(
        df
    )

    df = _standardize_model_columns(
        df
    )

    # --------------------------------------------------------
    # Alpha score
    # --------------------------------------------------------

    if "Alpha_Score" not in df.columns:

        df["Alpha_Score"] = 0.0

    df["Alpha_Score"] = (
        _safe_numeric(
            df["Alpha_Score"]
        )
    )

    # --------------------------------------------------------
    # Risk penalty
    # --------------------------------------------------------

    if "Risk_Penalty" not in df.columns:

        df["Risk_Penalty"] = 0.0

    df["Risk_Penalty"] = (
        _safe_numeric(
            df["Risk_Penalty"]
        )
    )

    # ========================================================
    # COMPONENT 1 — PREDICTION
    # ========================================================

    prediction_component = (
        df.groupby(
            date_column,
            sort=False,
        )["Prediction_Alpha"]
        .transform(
            safe_zscore
        )
    )

    # ========================================================
    # COMPONENT 2 — CONFIDENCE
    # ========================================================

    confidence_component = (
        df["Confidence"]
        .clip(
            0.0,
            1.0,
        )
    )

    # ========================================================
    # COMPONENT 3 — RISK REWARD
    # ========================================================

    rr_component = (
        df["RR_Ratio"]
        .clip(
            0.0,
            5.0,
        )
        /
        5.0
    )

    # ========================================================
    # COMPONENT 4 — EXPECTED RETURN
    # ========================================================

    expected_return_component = (
        df.groupby(
            date_column,
            sort=False,
        )["Expected_Return"]
        .transform(
            safe_zscore
        )
    )

    # ========================================================
    # COMPONENT 5 — ALPHA
    # ========================================================

    alpha_component = (
        df.groupby(
            date_column,
            sort=False,
        )["Alpha_Score"]
        .transform(
            safe_zscore
        )
    )

    # ========================================================
    # COMPOSITE SCORE
    # ========================================================

    df["Final_Score"] = (

        SCORE_WEIGHTS["prediction"]
        *
        prediction_component

        +

        SCORE_WEIGHTS["confidence"]
        *
        confidence_component

        +

        SCORE_WEIGHTS["risk_reward"]
        *
        rr_component

        +

        SCORE_WEIGHTS["expected_return"]
        *
        expected_return_component

        +

        SCORE_WEIGHTS["alpha"]
        *
        alpha_component
    )

    # ========================================================
    # CONCENTRATION PENALTY
    # ========================================================

    probability_rank = (
        df.groupby(
            date_column,
            sort=False,
        )["Prediction_Alpha"]
        .rank(
            pct=True
        )
    )

    concentration_mask = (
        probability_rank
        >
        CONCENTRATION_PERCENTILE
    )

    df["Concentration_Penalty"] = (
        concentration_mask
        .astype(float)
        *
        CONCENTRATION_PENALTY
    )

    df["Final_Score"] -= (
        df["Concentration_Penalty"]
    )

    # ========================================================
    # RISK PENALTY
    # ========================================================

    risk_rank = (
        df.groupby(
            date_column,
            sort=False,
        )["Risk_Penalty"]
        .rank(
            pct=True,
            ascending=True,
        )
    )

    df["Risk_Score_Penalty"] = (
        RISK_SCORE_PENALTY
        *
        risk_rank
    )

    df["Final_Score"] -= (
        df["Risk_Score_Penalty"]
    )

    # ========================================================
    # FINAL CROSS-SECTIONAL NORMALIZATION
    # ========================================================

    df["Final_Score"] = (
        df.groupby(
            date_column,
            sort=False,
        )["Final_Score"]
        .transform(
            safe_zscore
        )
    )

    df["Final_Score"] = (
        df["Final_Score"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Compatibility alias
    # --------------------------------------------------------

    df["Selection_Score"] = (
        df["Final_Score"]
    )

    return df


# ============================================================
# RANK STOCKS
# ============================================================

def rank_stocks(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank securities cross-sectionally by Final_Score.
    """

    if df is None:
        raise ValueError(
            "rank_stocks received None."
        )

    if df.empty:
        return df.copy()

    df = df.copy()

    date_column = _get_date_column(
        df
    )

    if "Final_Score" not in df.columns:

        raise ValueError(
            "rank_stocks requires "
            "'Final_Score'."
        )

    df["Rank_Score"] = (
        _safe_numeric(
            df["Final_Score"]
        )
    )

    df["Portfolio_Rank"] = (
        df.groupby(
            date_column,
            sort=False,
        )["Rank_Score"]
        .rank(
            ascending=False,
            method="first",
        )
        .astype(int)
    )

    return df


# ============================================================
# HYSTERESIS SELECTION
# ============================================================

def select_top_stocks(
    df: pd.DataFrame,
    entry_rank: int = DEFAULT_ENTRY_RANK,
    exit_rank: int = DEFAULT_EXIT_RANK,
) -> pd.DataFrame:
    """
    Select stocks using rank-based hysteresis.

    New positions:
        rank <= entry_rank

    Existing positions:
        rank <= exit_rank

    This creates an entry/exit buffer and helps reduce
    unnecessary turnover.

    IMPORTANT:
    ----------
    This function does NOT calculate final portfolio weights.
    """

    if df is None:
        raise ValueError(
            "select_top_stocks received None."
        )

    if df.empty:
        return df.copy()

    df = df.copy()

    date_column = _get_date_column(
        df
    )

    entry_rank = max(
        1,
        int(entry_rank),
    )

    exit_rank = max(
        entry_rank,
        int(exit_rank),
    )

    # --------------------------------------------------------
    # Previous holdings
    # --------------------------------------------------------

    if "Prev_Selected" not in df.columns:

        df["Prev_Selected"] = 0

    df["Prev_Selected"] = (
        _safe_numeric(
            df["Prev_Selected"]
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    if "Portfolio_Rank" not in df.columns:

        df = rank_stocks(
            df
        )

    # --------------------------------------------------------
    # Initial selection
    # --------------------------------------------------------

    df["Selected"] = 0

    entry_mask = (
        df["Portfolio_Rank"]
        <=
        entry_rank
    )

    df.loc[
        entry_mask,
        "Selected",
    ] = 1

    # --------------------------------------------------------
    # Existing holdings
    # --------------------------------------------------------

    hold_mask = (
        (
            df["Prev_Selected"]
            == 1
        )
        &
        (
            df["Portfolio_Rank"]
            <=
            exit_rank
        )
    )

    df.loc[
        hold_mask,
        "Selected",
    ] = 1

    # --------------------------------------------------------
    # Maximum candidate count
    # --------------------------------------------------------

    for _, group in df.groupby(
        date_column,
        sort=False,
    ):

        selected_idx = (
            group.index[
                group["Selected"]
                == 1
            ]
        )

        if (
            len(selected_idx)
            >
            exit_rank
        ):

            keep_idx = (
                group.loc[
                    selected_idx
                ]
                .sort_values(
                    "Final_Score",
                    ascending=False,
                )
                .head(
                    exit_rank
                )
                .index
            )

            remove_idx = (
                selected_idx
                .difference(
                    keep_idx
                )
            )

            df.loc[
                remove_idx,
                "Selected",
            ] = 0

    # --------------------------------------------------------
    # Selection metadata
    # --------------------------------------------------------

    df["New_Entry"] = (
        (
            df["Selected"]
            == 1
        )
        &
        (
            df["Prev_Selected"]
            == 0
        )
    ).astype(int)

    df["Retained_Holding"] = (
        (
            df["Selected"]
            == 1
        )
        &
        (
            df["Prev_Selected"]
            == 1
        )
    ).astype(int)

    df["Exit_Signal"] = (
        (
            df["Prev_Selected"]
            == 1
        )
        &
        (
            df["Selected"]
            == 0
        )
    ).astype(int)

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    logger.info(
        "Selection complete | "
        "Candidates=%d | Selected=%d | "
        "New=%d | Retained=%d | Exits=%d",
        len(df),
        int(
            (
                df["Selected"]
                == 1
            ).sum()
        ),
        int(
            df["New_Entry"].sum()
        ),
        int(
            df["Retained_Holding"].sum()
        ),
        int(
            df["Exit_Signal"].sum()
        ),
    )

    return df


# ============================================================
# LEGACY POSITION WEIGHT COMPATIBILITY
# ============================================================

def compute_position_weights(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compatibility function.

    IMPORTANT:
    ----------
    This is retained so existing callers do not immediately
    break.

    It is NO LONGER the institutional portfolio weighting
    engine.

    The authoritative position weights must come from
    portfolio_builder.py.

    This function therefore creates only a temporary
    signal-weight representation.

    Do NOT use this output as the final production portfolio
    weight when the institutional builder is active.
    """

    if df is None:
        raise ValueError(
            "compute_position_weights received None."
        )

    if df.empty:
        return df.copy()

    df = df.copy()

    if "Selected" not in df.columns:

        df["Selected"] = 0

    selected_mask = (
        df["Selected"]
        == 1
    )

    df["Position_Weight"] = 0.0

    if not selected_mask.any():

        return df

    date_column = _get_date_column(
        df
    )

    # --------------------------------------------------------
    # Signal weights
    # --------------------------------------------------------

    signal = (
        df["Final_Score"]
        .clip(
            -3.0,
            3.0,
        )
    )

    signal = (
        np.exp(
            signal
        )
    )

    probability = (
        _safe_numeric(
            df["Prediction_Alpha"],
            default=0.50,
        )
        .clip(
            0.0,
            1.0,
        )
    )

    raw_weight = (
        signal
        *
        probability
    )

    raw_weight = (
        raw_weight
        .where(
            selected_mask,
            0.0,
        )
        .clip(
            lower=0.0
        )
    )

    # --------------------------------------------------------
    # Cross-sectional normalization
    # --------------------------------------------------------

    totals = (
        raw_weight
        .groupby(
            df[date_column],
            sort=False,
        )
        .transform(
            "sum"
        )
    )

    valid = (
        selected_mask
        &
        (
            totals
            > 1e-12
        )
    )

    df.loc[
        valid,
        "Position_Weight",
    ] = (
        raw_weight.loc[
            valid
        ]
        /
        totals.loc[
            valid
        ]
    )

    logger.warning(
        "compute_position_weights() is a "
        "legacy compatibility function. "
        "Use portfolio_builder.py for final "
        "institutional weights."
    )

    return df


# ============================================================
# SELECTION DIAGNOSTICS
# ============================================================

def selection_diagnostics(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Produce machine-readable diagnostics for the selection
    stage.
    """

    if df is None or df.empty:

        return {
            "rows": 0,
            "selected": 0,
            "new_entries": 0,
            "retained": 0,
            "exits": 0,
        }

    result = {
        "rows": int(
            len(df)
        ),
        "selected": int(
            (
                df.get(
                    "Selected",
                    pd.Series(
                        0,
                        index=df.index,
                    ),
                )
                == 1
            ).sum()
        ),
        "new_entries": int(
            df.get(
                "New_Entry",
                pd.Series(
                    0,
                    index=df.index,
                ),
            ).sum()
        ),
        "retained": int(
            df.get(
                "Retained_Holding",
                pd.Series(
                    0,
                    index=df.index,
                ),
            ).sum()
        ),
        "exits": int(
            df.get(
                "Exit_Signal",
                pd.Series(
                    0,
                    index=df.index,
                ),
            ).sum()
        ),
    }

    if "Final_Score" in df.columns:

        result.update(
            {
                "score_mean": float(
                    df[
                        "Final_Score"
                    ].mean()
                ),
                "score_std": float(
                    df[
                        "Final_Score"
                    ].std()
                ),
                "score_max": float(
                    df[
                        "Final_Score"
                    ].max()
                ),
                "score_min": float(
                    df[
                        "Final_Score"
                    ].min()
                ),
            }
        )

    if "Prediction_Alpha" in df.columns:

        result.update(
            {
                "prediction_mean": float(
                    df[
                        "Prediction_Alpha"
                    ].mean()
                ),
                "prediction_max": float(
                    df[
                        "Prediction_Alpha"
                    ].max()
                ),
            }
        )

    return result


# ============================================================
# COMPLETE SELECTION PIPELINE
# ============================================================

def build_portfolio(
    df: pd.DataFrame,
    top_n: int = DEFAULT_ENTRY_RANK,
) -> pd.DataFrame:
    """
    Complete candidate-selection pipeline.

    Despite the historical function name, this function now
    performs PORTFOLIO SELECTION rather than institutional
    portfolio construction.

    Pipeline:

        Raw candidate universe
                |
                v
        Standardize model fields
                |
                v
        IC Alpha Score
                |
                v
        Risk Penalty
                |
                v
        Probability / Confidence filters
                |
                v
        Composite Selection Score
                |
                v
        Cross-sectional ranking
                |
                v
        Hysteresis selection
                |
                v
        Selected candidates
                |
                v
        portfolio_builder.py

    Returns
    -------
    pd.DataFrame
        Candidate universe with selection metadata.
    """

    if df is None:
        logger.error(
            "Portfolio selection received None."
        )

        return pd.DataFrame()

    if df.empty:

        logger.warning(
            "Portfolio selection received "
            "an empty dataframe."
        )

        return pd.DataFrame()

    logger.info(
        "Portfolio selection input: %d rows",
        len(df),
    )

    # ========================================================
    # 1. STANDARDIZE INPUT
    # ========================================================

    df = (
        df.copy()
    )

    _get_date_column(
        df
    )

    df = (
        _standardize_model_columns(
            df
        )
    )

    # ========================================================
    # 2. ALPHA SCORE
    # ========================================================

    df = (
        compute_alpha_score(
            df
        )
    )

    # ========================================================
    # 3. RISK PENALTY
    # ========================================================

    df = (
        compute_risk_penalty(
            df
        )
    )

    # ========================================================
    # 4. QUALITY FILTERS
    # ========================================================

    df = (
        _apply_quality_filters(
            df
        )
    )

    if df.empty:

        logger.warning(
            "No candidates survived "
            "probability/confidence filters."
        )

        return df

    # ========================================================
    # 5. FINAL SELECTION SCORE
    # ========================================================

    df = (
        compute_final_score(
            df
        )
    )

    if df.empty:

        logger.warning(
            "No candidates available "
            "after score construction."
        )

        return df

    # ========================================================
    # 6. RANK
    # ========================================================

    df = (
        rank_stocks(
            df
        )
    )

    # ========================================================
    # 7. TOP-N / HYSTERESIS SELECTION
    # ========================================================

    top_n = max(
        1,
        int(top_n),
    )

    exit_rank = max(
        top_n + 2,
        DEFAULT_EXIT_RANK,
    )

    df = (
        select_top_stocks(
            df,
            entry_rank=top_n,
            exit_rank=exit_rank,
        )
    )

    # ========================================================
    # 8. COMPATIBILITY SIGNAL WEIGHT
    # ========================================================
    #
    # This does NOT replace the institutional optimizer.
    #
    # It is retained only because some downstream analytics
    # currently expect Position_Weight to exist.
    #
    # The institutional builder should eventually be the
    # sole producer of final weights.
    #
    # ========================================================

    df = (
        compute_position_weights(
            df
        )
    )

    # ========================================================
    # 9. DIAGNOSTICS
    # ========================================================

    diagnostics = (
        selection_diagnostics(
            df
        )
    )

    logger.info(
        "Portfolio selection diagnostics: %s",
        diagnostics,
    )

    # ========================================================
    # 10. LATEST DATE SUMMARY
    # ========================================================

    date_column = (
        _get_date_column(
            df
        )
    )

    if not df.empty:

        latest_date = (
            df[
                date_column
            ].max()
        )

        latest = (
            df[
                df[
                    date_column
                ]
                ==
                latest_date
            ]
            .copy()
        )

        latest = (
            latest
            .sort_values(
                "Portfolio_Rank"
            )
        )

        logger.info(
            "Latest selection date: %s",
            latest_date,
        )

        selected_latest = (
            latest[
                latest["Selected"]
                == 1
            ]
        )

        display_columns = [
            column
            for column in [
                "Company",
                "Signal",
                "Prediction_Prob",
                "Prediction_Alpha",
                "Confidence",
                "Alpha_Score",
                "Risk_Penalty",
                "Expected_Return",
                "RR_Ratio",
                "Final_Score",
                "Portfolio_Rank",
                "Selected",
                "New_Entry",
                "Retained_Holding",
                "Position_Weight",
            ]
            if column in selected_latest.columns
        ]

        if (
            not selected_latest.empty
            and display_columns
        ):

            logger.info(
                "\nLatest selected candidates:\n%s",
                selected_latest[
                    display_columns
                ].to_string(
                    index=False
                ),
            )

    return df


# ============================================================
# EXPLICIT SELECTION API
# ============================================================

def select_candidates(
    df: pd.DataFrame,
    top_n: int = DEFAULT_ENTRY_RANK,
) -> pd.DataFrame:
    """
    Explicit API for the new architecture.

    This name makes the separation between:

        selection

    and:

        portfolio construction

    clear.

    Equivalent to build_portfolio(), but preferred for new
    code.
    """

    return build_portfolio(
        df=df,
        top_n=top_n,
    )


# ============================================================
# SELECTED-ONLY DATAFRAME
# ============================================================

def get_selected_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return only currently selected securities.
    """

    if df is None or df.empty:

        return pd.DataFrame()

    if "Selected" not in df.columns:

        raise ValueError(
            "get_selected_candidates requires "
            "'Selected' column."
        )

    return (
        df.loc[
            df["Selected"]
            == 1
        ]
        .copy()
    )


# ============================================================
# LATEST SELECTED CANDIDATES
# ============================================================

def get_latest_selected_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return selected candidates for the latest signal date.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    date_column = (
        _get_date_column(
            df
        )
    )

    selected = (
        get_selected_candidates(
            df
        )
    )

    if selected.empty:
        return selected

    latest_date = (
        selected[
            date_column
        ].max()
    )

    return (
        selected[
            selected[
                date_column
            ]
            ==
            latest_date
        ]
        .sort_values(
            "Portfolio_Rank"
        )
        .copy()
    )


# ============================================================
# EXPORT CANDIDATE SIGNALS
# ============================================================

def prepare_builder_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the selected dataframe for the institutional
    portfolio builder.

    This deliberately does NOT calculate weights.

    The returned dataframe contains the information that
    downstream portfolio-construction code can use to build:

        ExpectedReturnInput
        SecurityUniverse
        FactorExposureInput
        LiquidityInput
        CurrentPortfolioInput
        etc.
    """

    if df is None or df.empty:

        return pd.DataFrame()

    selected = (
        get_selected_candidates(
            df
        )
    )

    if selected.empty:

        return selected

    # --------------------------------------------------------
    # Remove legacy weight as authoritative output
    # --------------------------------------------------------

    selected = (
        selected.copy()
    )

    selected["Builder_Eligible"] = 1

    # --------------------------------------------------------
    # Expected return source
    # --------------------------------------------------------
    #
    # The builder's ForecastIntegrationLayer consumes
    # PortfolioConstructionInput.forecast.expected_returns.
    #
    # We preserve the model's Expected_Return rather than
    # silently replacing it with Final_Score.
    #
    # --------------------------------------------------------

    if (
        "Expected_Return"
        in selected.columns
    ):

        selected[
            "Builder_Expected_Return"
        ] = (
            _safe_numeric(
                selected[
                    "Expected_Return"
                ]
            )
        )

    # --------------------------------------------------------
    # Signal strength
    # --------------------------------------------------------

    selected[
        "Builder_Signal_Strength"
    ] = (
        _safe_numeric(
            selected[
                "Final_Score"
            ]
        )
    )

    return selected


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "safe_zscore",
    "cross_sectional_zscore",
    "compute_alpha_score",
    "compute_risk_penalty",
    "compute_final_score",
    "rank_stocks",
    "select_top_stocks",
    "compute_position_weights",
    "selection_diagnostics",
    "build_portfolio",
    "select_candidates",
    "get_selected_candidates",
    "get_latest_selected_candidates",
    "prepare_builder_candidates",
]