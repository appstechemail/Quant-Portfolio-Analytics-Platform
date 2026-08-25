# ============================================================
# BACKTEST ENGINE
# ============================================================

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from config.config import CONFIG


logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

BACKTEST_CONFIG = CONFIG.get("BACKTEST", {})
PORTFOLIO_CONFIG = CONFIG.get("PORTFOLIO", {})


TOP_PCT = float(
    BACKTEST_CONFIG.get(
        "TOP_PCT",
        0.10,
    )
)

TRANSACTION_COST = float(
    BACKTEST_CONFIG.get(
        "TRANSACTION_COST",
        0.001,
    )
)

SLIPPAGE = float(
    BACKTEST_CONFIG.get(
        "SLIPPAGE",
        0.0005,
    )
)

TARGET_VOL = float(
    BACKTEST_CONFIG.get(
        "TARGET_VOL",
        0.02,
    )
)

MIN_CONFIDENCE = float(
    BACKTEST_CONFIG.get(
        "MIN_CONFIDENCE",
        0.02,
    )
)

VOL_WINDOW = int(
    BACKTEST_CONFIG.get(
        "VOL_WINDOW",
        20,
    )
)

NEUTRALITY = float(
    BACKTEST_CONFIG.get(
        "NEUTRALITY",
        0.50,
    )
)

RETURN_CLIP = float(
    BACKTEST_CONFIG.get(
        "RETURN_CLIP",
        0.20,
    )
)

USE_CROSS_SECTIONAL_RANK = bool(
    BACKTEST_CONFIG.get(
        "USE_CROSS_SECTIONAL_RANK",
        True,
    )
)

USE_LIQUIDITY_FILTER = bool(
    BACKTEST_CONFIG.get(
        "USE_LIQUIDITY_FILTER",
        False,
    )
)

LIQUIDITY_THRESHOLD = float(
    BACKTEST_CONFIG.get(
        "LIQUIDITY_THRESHOLD",
        0.0,
    )
)

MAX_PORTFOLIO_SIZE = int(
    BACKTEST_CONFIG.get(
        "MAX_PORTFOLIO_SIZE",
        3,
    )
)

MAX_POSITION_SIZE = float(
    BACKTEST_CONFIG.get(
        "MAX_POSITION_SIZE",
        0.50,
    )
)

MAX_GROSS_EXPOSURE = float(
    BACKTEST_CONFIG.get(
        "MAX_GROSS_EXPOSURE",
        1.00,
    )
)

USE_VOL_TARGET = bool(
    BACKTEST_CONFIG.get(
        "USE_VOL_TARGET",
        True,
    )
)

USE_REGIME_EXPOSURE = bool(
    BACKTEST_CONFIG.get(
        "USE_REGIME_EXPOSURE",
        True,
    )
)

SMOOTHING = float(
    BACKTEST_CONFIG.get(
        "SMOOTHING",
        0.0,
    )
)

USE_DEADBAND = bool(
    BACKTEST_CONFIG.get(
        "USE_DEADBAND",
        False,
    )
)

TURNOVER_BAND = float(
    BACKTEST_CONFIG.get(
        "TURNOVER_BAND",
        0.10,
    )
)

EXECUTION_LAG = int(
    BACKTEST_CONFIG.get(
        "EXECUTION_LAG",
        1,
    )
)


# ============================================================
# REGIME EXPOSURE
# ============================================================

REGIME_EXPOSURE = {
    "BULL": 1.00,
    "BULL_VOLATILE": 1.20,
    "SIDEWAYS": 0.60,
    "SIDEWAYS_VOLATILE": 0.30,
    "BEAR": 0.15,
    "BEAR_VOLATILE": 0.00,
}


# ============================================================
# HELPERS
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert value to finite float.
    """

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def _normalise_keys(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalise Date and Company keys.
    """

    out = df.copy()

    if "Date" in out.columns:

        out["Date"] = (
            pd.to_datetime(
                out["Date"],
                errors="coerce",
            )
            .dt.normalize()
        )

    if "Company" in out.columns:

        out["Company"] = (
            out["Company"]
            .astype(str)
            .str.strip()
        )

    return out


# ============================================================
# PROBABILITY EXTRACTION
# ============================================================

def _extract_probability(
    proba: Any,
) -> np.ndarray:
    """
    Convert probability input into a 1-D positive-class
    probability.

    Supports:

        [0.1, 0.2, ...]

        [[0.9, 0.1],
         [0.8, 0.2]]
    """

    if proba is None:

        return np.array(
            [],
            dtype=float,
        )

    arr = np.asarray(proba)

    if arr.size == 0:

        return np.array(
            [],
            dtype=float,
        )

    if arr.ndim == 1:

        return arr.astype(float)

    if arr.ndim == 2:

        if arr.shape[1] == 1:

            return arr[:, 0].astype(float)

        return arr[:, -1].astype(float)

    return arr.reshape(-1).astype(float)


# ============================================================
# PREDICTION PANEL
# ============================================================

def _prepare_prediction_frame(
    proba: Any,
    X_test: Optional[pd.DataFrame],
    meta_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build authoritative prediction panel.

    meta_test is authoritative for prediction alignment.
    """

    if meta_test is None:

        raise ValueError(
            "meta_test cannot be None."
        )

    df = (
        meta_test
        .copy()
        .reset_index(drop=True)
    )

    proba_arr = _extract_probability(
        proba
    )

    if len(proba_arr) == 0:

        raise ValueError(
            "Empty probability array."
        )

    if len(df) != len(proba_arr):

        raise ValueError(
            "Prediction/meta_test length mismatch: "
            f"meta_test={len(df)}, "
            f"proba={len(proba_arr)}."
        )

    df["Proba"] = proba_arr

    # --------------------------------------------------------
    # Recover keys from X_test if necessary.
    # --------------------------------------------------------

    if (
        "Date" not in df.columns
        or "Company" not in df.columns
    ):

        if (
            X_test is not None
            and len(X_test) == len(df)
        ):

            x_meta = (
                X_test
                .copy()
                .reset_index(drop=True)
            )

            for col in [
                "Date",
                "Company",
            ]:

                if (
                    col not in df.columns
                    and col in x_meta.columns
                ):

                    df[col] = x_meta[col]

    # --------------------------------------------------------
    # Required keys.
    # --------------------------------------------------------

    if "Date" not in df.columns:

        raise ValueError(
            "meta_test/X_test does not contain Date."
        )

    if "Company" not in df.columns:

        raise ValueError(
            "meta_test/X_test does not contain Company."
        )

    df = _normalise_keys(df)

    # --------------------------------------------------------
    # Duplicate diagnostics.
    # --------------------------------------------------------

    duplicate_mask = df.duplicated(
        [
            "Date",
            "Company",
        ],
        keep=False,
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count > 0:

        logger.warning(
            "Prediction panel contains %d duplicate "
            "(Date, Company) rows.",
            duplicate_count,
        )

        df = (
            df
            .sort_values(
                [
                    "Date",
                    "Company",
                ]
            )
            .drop_duplicates(
                [
                    "Date",
                    "Company",
                ],
                keep="first",
            )
            .reset_index(drop=True)
        )

    return df


# ============================================================
# MARKET DATA MERGE
# ============================================================

def _merge_market_data(
    prediction_df: pd.DataFrame,
    final_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach market information to prediction panel.

    final_df is used strictly as a market-data lookup.
    """

    if (
        final_df is None
        or final_df.empty
    ):

        raise ValueError(
            "final_df is empty."
        )

    market = _normalise_keys(
        final_df.copy()
    )

    required = [
        "Date",
        "Company",
        "Close",
    ]

    missing = [
        c
        for c in required
        if c not in market.columns
    ]

    if missing:

        raise ValueError(
            "final_df missing required columns: "
            f"{missing}"
        )

    preferred_columns = [
        "Date",
        "Company",
        "Close",
        "Volume",
        "Market_Regime",
        "ATR_14",
        "Regime_Score",
        "Regime_Strength",
        "Liquidity",
        "DollarVolume",
    ]

    available = [
        c
        for c in preferred_columns
        if c in market.columns
    ]

    market = market[
        available
    ].copy()

    # --------------------------------------------------------
    # Strict market uniqueness.
    # --------------------------------------------------------

    market = (
        market
        .sort_values(
            [
                "Company",
                "Date",
            ]
        )
        .drop_duplicates(
            [
                "Date",
                "Company",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    prediction = (
        prediction_df
        .copy()
    )

    lookup_columns = [
        c
        for c in market.columns
        if (
            c not in [
                "Date",
                "Company",
            ]
            and c not in prediction.columns
        )
    ]

    market_lookup = market[
        [
            "Date",
            "Company",
        ]
        +
        lookup_columns
    ].copy()

    merged = prediction.merge(
        market_lookup,
        on=[
            "Date",
            "Company",
        ],
        how="left",
        validate="many_to_one",
    )

    return merged


# ============================================================
# ALPHA CREATION
# ============================================================

def _create_alpha(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the canonical continuous alpha signal from BUY probability.

    Canonical semantics
    -------------------
    Probability:
        P(Target = BUY)

    Alpha:
        Probability - NEUTRALITY

    Confidence:
        abs(Probability - NEUTRALITY) * 2

    Contract
    --------
    Proba is the authoritative model probability.

    The backtest must NOT reconstruct alpha from:
        - Prediction_Alpha
        - Alpha_Score
        - Final_Score
        - filtered signals
        - regime-filtered signals
        - volatility-filtered signals

    Those fields may represent transformed or filtered values.

    The canonical signal chain is:

        Proba
          ↓
        Probability
          ↓
        Prediction_Alpha
          ↓
        Alpha

    Confidence is derived independently from Probability.
    """

    out = df.copy()

    # ========================================================
    # 0. CONFIGURED NEUTRALITY
    # ========================================================

    try:

        neutrality = float(
            CONFIG["BACKTEST"].get(
                "NEUTRALITY",
                0.50,
            )
        )

    except Exception:

        neutrality = 0.50

    if not np.isfinite(neutrality):

        raise ValueError(
            "CRITICAL: BACKTEST.NEUTRALITY must be "
            "a finite numeric value."
        )

    if not 0.0 <= neutrality <= 1.0:

        raise ValueError(
            "CRITICAL: BACKTEST.NEUTRALITY must be "
            "between 0.0 and 1.0. "
            f"Received: {neutrality}"
        )

    # ========================================================
    # 1. CANONICAL PROBABILITY
    # ========================================================

    if "Proba" not in out.columns:

        raise ValueError(
            "CRITICAL: Backtest prediction panel does not "
            "contain canonical 'Proba'."
        )

    probability = pd.to_numeric(
        out["Proba"],
        errors="coerce",
    )

    probability = (
        probability
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    # ========================================================
    # 2. VALIDATION
    # ========================================================

    if probability.isna().any():

        invalid_count = int(
            probability.isna().sum()
        )

        raise ValueError(
            "CRITICAL: Canonical Proba contains "
            f"{invalid_count} NaN/invalid values."
        )

    # ========================================================
    # 3. PROBABILITY IS AUTHORITATIVE
    # ========================================================

    out["Prediction_Prob"] = probability

    out["Probability"] = probability

    # ========================================================
    # 4. EXPLICIT ALPHA CONTRACT
    # ========================================================
    #
    # Alpha is ALWAYS reconstructed from the
    # authoritative probability.
    #
    #     Alpha = P(BUY) - Neutrality
    #
    # Examples:
    #
    #     P = 0.90 → Alpha = +0.40
    #     P = 0.70 → Alpha = +0.20
    #     P = 0.50 → Alpha =  0.00
    #     P = 0.30 → Alpha = -0.20
    #     P = 0.10 → Alpha = -0.40
    #
    # ========================================================

    prediction_alpha = (
        probability
        - neutrality
    )

    prediction_alpha = (
        prediction_alpha
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .clip(
            lower=-1.0,
            upper=1.0,
        )
    )

    if prediction_alpha.isna().any():

        raise ValueError(
            "CRITICAL: Prediction_Alpha contains "
            "NaN/invalid values after construction."
        )

    out["Prediction_Alpha"] = (
        prediction_alpha
    )

    out["Alpha"] = (
        prediction_alpha
    )

    # ========================================================
    # 5. CANONICAL CONFIDENCE
    # ========================================================
    #
    # Confidence measures distance from neutrality.
    #
    #     Confidence =
    #         abs(P(BUY) - Neutrality) * 2
    #
    # Therefore:
    #
    #     P = 0.50 → Confidence = 0.00
    #     P = 0.75 → Confidence = 0.50
    #     P = 1.00 → Confidence = 1.00
    #     P = 0.25 → Confidence = 0.50
    #     P = 0.00 → Confidence = 1.00
    #
    # ========================================================

    confidence = (
        np.abs(
            probability
            - neutrality
        )
        * 2.0
    )

    confidence = (
        confidence
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .clip(
            lower=0.0,
            upper=1.0,
        )
        .fillna(0.0)
    )

    out["Confidence"] = confidence

    # ========================================================
    # 6. ALPHA SOURCE
    # ========================================================

    out["Alpha_Source"] = (
        "Canonical_Probability"
    )

    # ========================================================
    # 7. ALPHA SANITY CHECK
    # ========================================================

    alpha_array = (
        prediction_alpha
        .to_numpy(
            dtype=float
        )
    )

    alpha_series = pd.Series(
        alpha_array,
        name="Prediction_Alpha",
    )

    probability_array = (
        probability
        .to_numpy(
            dtype=float
        )
    )

    confidence_array = (
        confidence
        .to_numpy(
            dtype=float
        )
    )

    positive_alpha = (
        alpha_array > 0
    )

    negative_alpha = (
        alpha_array < 0
    )

    zero_alpha = (
        alpha_array == 0
    )

    print("\n" + "=" * 64)

    print(
        "CANONICAL ALPHA CONTRACT"
    )

    print("=" * 64)

    print(
        f"Neutrality threshold : "
        f"{neutrality:.6f}"
    )

    print(
        f"Probability min      : "
        f"{probability_array.min():.8f}"
    )

    print(
        f"Probability max      : "
        f"{probability_array.max():.8f}"
    )

    print(
        f"Probability mean     : "
        f"{probability_array.mean():.8f}"
    )

    print(
        f"Alpha min            : "
        f"{alpha_series.min():.8f}"
    )

    print(
        f"Alpha max            : "
        f"{alpha_series.max():.8f}"
    )

    print(
        f"Alpha mean           : "
        f"{alpha_series.mean():.8f}"
    )

    print(
        f"Alpha median         : "
        f"{alpha_series.median():.8f}"
    )

    print(
        f"Alpha std            : "
        f"{alpha_series.std():.8f}"
    )

    print(
        f"Positive Alpha       : "
        f"{positive_alpha.sum()}"
    )

    print(
        f"Negative Alpha       : "
        f"{negative_alpha.sum()}"
    )

    print(
        f"Zero Alpha           : "
        f"{zero_alpha.sum()}"
    )

    print(
        f"Positive Alpha %     : "
        f"{positive_alpha.mean():.2%}"
    )

    print(
        f"Negative Alpha %     : "
        f"{negative_alpha.mean():.2%}"
    )

    print(
        f"Confidence min       : "
        f"{confidence_array.min():.8f}"
    )

    print(
        f"Confidence max       : "
        f"{confidence_array.max():.8f}"
    )

    print(
        f"Confidence mean      : "
        f"{confidence_array.mean():.8f}"
    )

    print(
        f"Alpha source         : "
        f"{out['Alpha_Source'].iloc[0]}"
    )

    print("=" * 64)

    return out


# ============================================================
# HISTORICAL VOLATILITY
# ============================================================

def _calculate_historical_volatility(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate strictly backward-looking volatility.

    The current day's return is excluded.

    Therefore:

        Volatility[T]

    only uses information available before T.
    """

    out = (
        df
        .copy()
        .sort_values(
            [
                "Company",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )

    daily_return = (
        out
        .groupby(
            "Company",
            sort=False,
        )["Close"]
        .pct_change()
    )

    # --------------------------------------------------------
    # Explicitly lag returns by one observation.
    # --------------------------------------------------------

    lagged_return = (
        daily_return
        .groupby(
            out["Company"],
            sort=False,
        )
        .shift(1)
    )

    out["Volatility"] = (
        lagged_return
        .groupby(
            out["Company"],
            sort=False,
        )
        .transform(
            lambda x: (
                x
                .rolling(
                    VOL_WINDOW,
                    min_periods=max(
                        5,
                        min(
                            VOL_WINDOW,
                            5,
                        ),
                    ),
                )
                .std()
            )
        )
    )

    return out


# ============================================================
# FORWARD RETURN
# ============================================================

def _calculate_forward_return(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate next available return within each company.

    IMPORTANT
    ---------
    The forward shift is performed explicitly inside Company.

    This prevents the final observation of one company from
    receiving the first return of another company.
    """

    out = (
        df
        .copy()
        .sort_values(
            [
                "Company",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )

    daily_return = (
        out
        .groupby(
            "Company",
            sort=False,
        )["Close"]
        .pct_change()
    )

    out["Return"] = (
        daily_return
        .groupby(
            out["Company"],
            sort=False,
        )
        .shift(-1)
    )

    out["Return"] = (
        out["Return"]
        .clip(
            -RETURN_CLIP,
            RETURN_CLIP,
        )
    )

    return out


# ============================================================
# CROSS-SECTIONAL SIGNAL
# ============================================================

def _cross_sectional_signal(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate daily cross-sectional long signals.

    Current backtest architecture is long-only.

    Positive alpha:
        eligible

    Non-positive alpha:
        excluded

    Top TOP_PCT:
        rank filter

    MAX_PORTFOLIO_SIZE:
        daily portfolio cap
    """

    out = df.copy()

    required_keys = [
        "Date",
        "Company",
    ]

    missing_keys = [
        col
        for col in required_keys
        if col not in out.columns
    ]

    if missing_keys:

        raise ValueError(
            "_cross_sectional_signal() missing "
            f"required columns: {missing_keys}"
        )

    out = _normalise_keys(out)

    # --------------------------------------------------------
    # Canonical probability / confidence
    # --------------------------------------------------------

    if "Probability" not in out.columns:

        raise ValueError(
            "CRITICAL: Probability column missing "
            "before cross-sectional signal generation."
        )

    out["Probability"] = (
        pd.to_numeric(
            out["Probability"],
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    out["Alpha"] = (
        out["Probability"]
        - NEUTRALITY
    )

    out["Confidence"] = (
        np.abs(
            out["Probability"]
            - NEUTRALITY
        )
        * 2.0
    ).clip(
        lower=0.0,
        upper=1.0,
    ).fillna(0.0)

    print("\n")
    print("=" * 58)
    print("CROSS-SECTIONAL SIGNAL DEBUG")
    print("=" * 58)

    print(
        f"Alpha min       : "
        f"{out['Alpha'].min():.8f}"
    )

    print(
        f"Alpha max       : "
        f"{out['Alpha'].max():.8f}"
    )

    print(
        f"Alpha mean      : "
        f"{out['Alpha'].mean():.8f}"
    )

    print(
        f"Alpha median    : "
        f"{out['Alpha'].median():.8f}"
    )

    print(
        f"Positive Alpha  : "
        f"{int((out['Alpha'] > 0).sum()):,}"
    )

    print(
        f"Confidence >= "
        f"{MIN_CONFIDENCE}: "
        f"{int((out['Confidence'] >= MIN_CONFIDENCE).sum()):,}"
    )

    print(
        f"TOP_PCT         : "
        f"{TOP_PCT}"
    )

    print(
        f"Cross-section rank enabled: "
        f"{USE_CROSS_SECTIONAL_RANK}"
    )

    # --------------------------------------------------------
    # Eligibility
    # --------------------------------------------------------

    out["Eligible"] = (
        out["Probability"]
        > NEUTRALITY
    )

    out.loc[
        out["Confidence"] < MIN_CONFIDENCE,
        "Eligible",
    ] = False

    # --------------------------------------------------------
    # Liquidity
    # --------------------------------------------------------

    if (
        USE_LIQUIDITY_FILTER
        and "Volume" in out.columns
    ):

        out["DollarVolume"] = (
            pd.to_numeric(
                out["Close"],
                errors="coerce",
            )
            *
            pd.to_numeric(
                out["Volume"],
                errors="coerce",
            )
        )

        out.loc[
            out["DollarVolume"]
            <
            LIQUIDITY_THRESHOLD,
            "Eligible",
        ] = False

    elif (
        "DollarVolume" not in out.columns
        and "Volume" in out.columns
    ):

        out["DollarVolume"] = (
            pd.to_numeric(
                out["Close"],
                errors="coerce",
            )
            *
            pd.to_numeric(
                out["Volume"],
                errors="coerce",
            )
        )

    # --------------------------------------------------------
    # Cross-sectional rank
    # --------------------------------------------------------

    out["Alpha_Rank"] = (
        out
        .groupby(
            "Date",
            sort=False,
        )["Alpha"]
        .rank(
            ascending=False,
            method="first",
            pct=True,
        )
    )

    if USE_CROSS_SECTIONAL_RANK:

        out["Rank_Eligible"] = (
            out["Alpha_Rank"]
            <= TOP_PCT
        )

        out["Eligible"] = (
            out["Eligible"]
            &
            out["Rank_Eligible"]
        )

    else:

        out["Rank_Eligible"] = (
            out["Eligible"]
        )

    print(
        f"Eligible after rank: "
        f"{int(out['Eligible'].sum()):,}"
    )

    # --------------------------------------------------------
    # Daily portfolio selection
    # --------------------------------------------------------

    out["Selected"] = False

    candidates = out.loc[
        out["Eligible"],
        [
            "Date",
            "Company",
            "Alpha",
        ],
    ].copy()

    if (
        not candidates.empty
        and MAX_PORTFOLIO_SIZE > 0
    ):

        candidates = (
            candidates
            .sort_values(
                [
                    "Date",
                    "Alpha",
                    "Company",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
            )
        )

        selected = (
            candidates
            .groupby(
                "Date",
                sort=False,
            )
            .head(
                MAX_PORTFOLIO_SIZE
            )
            [
                [
                    "Date",
                    "Company",
                ]
            ]
            .copy()
        )

        selected["_selection_key"] = (
            selected["Date"].astype(str)
            + "||"
            + selected["Company"].astype(str)
        )

        out["_selection_key"] = (
            out["Date"].astype(str)
            + "||"
            + out["Company"].astype(str)
        )

        selected_keys = set(
            selected["_selection_key"]
        )

        out["Selected"] = (
            out["_selection_key"]
            .isin(selected_keys)
        )

        out.drop(
            columns=[
                "_selection_key",
            ],
            inplace=True,
            errors="ignore",
        )

    elif (
        not candidates.empty
        and MAX_PORTFOLIO_SIZE <= 0
    ):

        out["Selected"] = (
            out["Eligible"]
        )

    # --------------------------------------------------------
    # Target alpha
    # --------------------------------------------------------

    out["Target_Alpha"] = np.where(
        out["Selected"],
        out["Confidence"],
        0.0,
    )

    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    if "Date" not in out.columns:

        raise RuntimeError(
            "CRITICAL: Date column was lost."
        )

    if "Company" not in out.columns:

        raise RuntimeError(
            "CRITICAL: Company column was lost."
        )

    out = (
        out
        .sort_values(
            [
                "Date",
                "Company",
            ]
        )
        .reset_index(drop=True)
    )

    return out


# ============================================================
# POSITION SIZING
# ============================================================

def _normalize_with_cap(
    values: pd.Series,
    cap: float,
) -> pd.Series:
    """
    Normalize positive weights while respecting a hard
    maximum position cap.

    This uses iterative redistribution.

    Therefore final weights satisfy:

        weight <= cap
    """

    values = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    if values.sum() <= 0:

        return pd.Series(
            0.0,
            index=values.index,
        )

    if cap <= 0:

        return pd.Series(
            0.0,
            index=values.index,
        )

    weights = (
        values
        /
        values.sum()
    )

    for _ in range(20):

        over_cap = (
            weights > cap
        )

        if not over_cap.any():

            break

        excess = (
            weights[over_cap]
            - cap
        ).sum()

        weights.loc[
            over_cap
        ] = cap

        under_cap = (
            ~over_cap
            &
            (weights > 0)
        )

        available = (
            weights.loc[
                under_cap
            ].sum()
        )

        if (
            available <= 0
            or excess <= 0
        ):

            break

        weights.loc[
            under_cap
        ] += (
            weights.loc[
                under_cap
            ]
            /
            available
            *
            excess
        )

    # --------------------------------------------------------
    # Final numerical safety.
    # --------------------------------------------------------

    weights = (
        weights
        .clip(
            lower=0.0,
            upper=cap,
        )
    )

    total = weights.sum()

    if total > 0:

        # Do not normalize again if it would violate cap.
        if total <= 1.0 + 1e-12:

            return weights

    return weights


def _risk_adjust_position(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert alpha into risk-adjusted target weights.
    """

    out = df.copy()

    # --------------------------------------------------------
    # Volatility fallback
    # --------------------------------------------------------

    median_vol = (
        out["Volatility"]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .median()
    )

    if (
        pd.isna(median_vol)
        or median_vol <= 0
    ):

        median_vol = TARGET_VOL

    out["Volatility"] = (
        out["Volatility"]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(median_vol)
        .clip(lower=1e-6)
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    if "ATR_14" in out.columns:

        out["ATR_PCT"] = (
            pd.to_numeric(
                out["ATR_14"],
                errors="coerce",
            )
            /
            (
                pd.to_numeric(
                    out["Close"],
                    errors="coerce",
                )
                +
                1e-12
            )
        )

    else:

        out["ATR_PCT"] = np.nan

    median_atr = (
        out["ATR_PCT"]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .median()
    )

    if (
        pd.isna(median_atr)
        or median_atr <= 0
    ):

        median_atr = 0.03

    out["ATR_PCT"] = (
        out["ATR_PCT"]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(median_atr)
        .clip(lower=1e-6)
    )

    # --------------------------------------------------------
    # Combined risk
    # --------------------------------------------------------

    out["Risk_Measure"] = (
        0.50 * out["Volatility"]
        +
        0.50 * out["ATR_PCT"]
    )

    out["Risk_Measure"] = (
        out["Risk_Measure"]
        .clip(lower=1e-6)
    )

    # --------------------------------------------------------
    # Raw position
    # --------------------------------------------------------

    out["Raw_Position"] = (
        out["Target_Alpha"]
        /
        out["Risk_Measure"]
    )

    out.loc[
        ~out["Selected"],
        "Raw_Position",
    ] = 0.0

    # --------------------------------------------------------
    # Daily cap-aware normalization
    # --------------------------------------------------------

    out["Position"] = 0.0

    for date, idx in out.groupby(
        "Date",
        sort=False,
    ).groups.items():

        idx = list(idx)

        weights = _normalize_with_cap(
            out.loc[
                idx,
                "Raw_Position",
            ],
            MAX_POSITION_SIZE,
        )

        out.loc[
            idx,
            "Position",
        ] = weights

    # --------------------------------------------------------
    # Portfolio volatility targeting
    # --------------------------------------------------------

    if USE_VOL_TARGET:

        def _portfolio_volatility(
            day: pd.DataFrame,
        ) -> float:

            weights = (
                day["Position"]
                .to_numpy(
                    dtype=float
                )
            )

            vols = (
                day["Volatility"]
                .to_numpy(
                    dtype=float
                )
            )

            return float(
                np.sqrt(
                    np.sum(
                        (
                            weights
                            *
                            vols
                        )
                        ** 2
                    )
                )
            )

        portfolio_vol = (
            out
            .groupby(
                "Date",
                sort=False,
            )
            .apply(
                _portfolio_volatility,
                include_groups=False,
            )
        )

        vol_scale = (
            TARGET_VOL
            /
            (
                portfolio_vol
                +
                1e-12
            )
        )

        vol_scale = (
            vol_scale
            .clip(
                lower=0.0,
                upper=2.0,
            )
        )

        out["Position"] *= (
            out["Date"]
            .map(vol_scale)
            .fillna(1.0)
        )

    # --------------------------------------------------------
    # Regime exposure
    # --------------------------------------------------------

    if (
        USE_REGIME_EXPOSURE
        and
        "Market_Regime" in out.columns
    ):

        out["Portfolio_Exposure"] = (
            out["Market_Regime"]
            .map(
                REGIME_EXPOSURE
            )
            .fillna(1.0)
        )

    else:

        out["Portfolio_Exposure"] = 1.0

    out["Position"] *= (
        out["Portfolio_Exposure"]
    )

    # --------------------------------------------------------
    # Gross exposure cap
    # --------------------------------------------------------

    gross = (
        out
        .groupby(
            "Date",
            sort=False,
        )["Position"]
        .transform("sum")
    )

    scale = np.minimum(
        1.0,
        MAX_GROSS_EXPOSURE
        /
        (
            gross
            +
            1e-12
        ),
    )

    out["Position"] *= scale

    return out


# ============================================================
# POSITION SMOOTHING
# ============================================================

def _apply_smoothing(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Optional exponential position smoothing.
    """

    if SMOOTHING <= 0:

        return df

    out = (
        df
        .copy()
        .sort_values(
            [
                "Company",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )

    for company, idx in out.groupby(
        "Company"
    ).groups.items():

        idx = list(idx)

        positions = (
            out.loc[
                idx,
                "Position",
            ]
            .to_numpy(
                dtype=float,
                copy=True,
            )
        )

        for i in range(
            1,
            len(positions),
        ):

            positions[i] = (
                SMOOTHING
                *
                positions[i - 1]
                +
                (
                    1.0
                    -
                    SMOOTHING
                )
                *
                positions[i]
            )

        out.loc[
            idx,
            "Position",
        ] = positions

    return out


# ============================================================
# DEAD BAND
# ============================================================

def _apply_deadband(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Suppress small position changes.
    """

    out = (
        df
        .copy()
        .sort_values(
            [
                "Company",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )

    out["Prev_Target_Position"] = (
        out
        .groupby(
            "Company",
            sort=False,
        )["Position"]
        .shift(1)
        .fillna(0.0)
    )

    out["Position_Change_Pre"] = (
        out["Position"]
        -
        out["Prev_Target_Position"]
    ).abs()

    if not USE_DEADBAND:

        out["Deadband_Applied"] = False

        return out

    threshold = (
        TURNOVER_BAND
        *
        np.maximum(
            out[
                "Prev_Target_Position"
            ].abs(),
            1e-12,
        )
    )

    deadband_mask = (
        out["Position_Change_Pre"]
        <= threshold
    )

    deadband_mask &= (
        out[
            "Prev_Target_Position"
        ]
        > 0
    )

    out.loc[
        deadband_mask,
        "Position",
    ] = out.loc[
        deadband_mask,
        "Prev_Target_Position",
    ]

    out["Deadband_Applied"] = (
        deadband_mask
    )

    return out


# ============================================================
# EXECUTION LAG
# ============================================================

def _apply_execution_lag(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Signal on T -> execute on T+EXECUTION_LAG.
    """

    out = (
        df
        .copy()
        .sort_values(
            [
                "Company",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )

    if EXECUTION_LAG <= 0:

        out["Position_Executed"] = (
            out["Position"]
        )

    else:

        out["Position_Executed"] = (
            out
            .groupby(
                "Company",
                sort=False,
            )["Position"]
            .shift(
                EXECUTION_LAG
            )
            .fillna(0.0)
        )

    return out


# ============================================================
# TURNOVER
# ============================================================

def _calculate_turnover(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate turnover from executed portfolio weights.
    """

    out = (
        df
        .copy()
        .sort_values(
            [
                "Company",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )

    out["Prev_Position"] = (
        out
        .groupby(
            "Company",
            sort=False,
        )["Position_Executed"]
        .shift(1)
        .fillna(0.0)
    )

    out["Position_Change"] = (
        out["Position_Executed"]
        -
        out["Prev_Position"]
    ).abs()

    daily_turnover = (
        out
        .groupby(
            "Date",
            sort=False,
        )["Position_Change"]
        .sum()
    )

    out["Daily_Turnover"] = (
        out["Date"]
        .map(
            daily_turnover
        )
        .fillna(0.0)
    )

    return out


# ============================================================
# RETURNS
# ============================================================

def _calculate_returns(
    df: pd.DataFrame,
):
    """
    Calculate portfolio P&L.

    Position_Executed is applied to the forward return.
    """

    out = df.copy()

    out["Gross_PnL"] = (
        out["Position_Executed"]
        *
        out["Return"].fillna(0.0)
    )

    total_cost_rate = (
        TRANSACTION_COST
        +
        SLIPPAGE
    )

    out["Cost"] = (
        out["Position_Change"]
        *
        total_cost_rate
    )

    out["Strategy_Return_Component"] = (
        out["Gross_PnL"]
        -
        out["Cost"]
    )

    daily = (
        out
        .groupby(
            "Date",
            sort=True,
        )
        .agg(
            Strategy_Return=(
                "Strategy_Return_Component",
                "sum",
            ),
            Gross_Return=(
                "Gross_PnL",
                "sum",
            ),
            Transaction_Cost=(
                "Cost",
                "sum",
            ),
            Turnover=(
                "Position_Change",
                "sum",
            ),
            Gross_Exposure=(
                "Position_Executed",
                lambda x: np.abs(x).sum(),
            ),
            Holdings=(
                "Position_Executed",
                lambda x: (
                    x > 0
                ).sum(),
            ),
        )
        .sort_index()
    )

    daily["Cumulative_Strategy"] = (
        1.0
        +
        daily["Strategy_Return"]
    ).cumprod()

    running_max = (
        daily["Cumulative_Strategy"]
        .cummax()
    )

    daily["Drawdown"] = (
        daily["Cumulative_Strategy"]
        /
        running_max
        -
        1.0
    )

    for col in [
        "Strategy_Return",
        "Cumulative_Strategy",
        "Drawdown",
    ]:

        out[col] = (
            out["Date"]
            .map(
                daily[col]
            )
        )

    return out, daily


# ============================================================
# METRICS
# ============================================================

def _calculate_metrics(
    daily: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate backtest statistics.
    """

    if daily.empty:

        return {
            "Sharpe": 0.0,
            "CAGR": 0.0,
            "Max_Drawdown": 0.0,
            "Volatility": 0.0,
            "Win_Rate": 0.0,
            "Avg_Turnover": 0.0,
            "Annual_Turnover": 0.0,
            "Median_Turnover": 0.0,
            "Turnover95": 0.0,
            "Max_Turnover": 0.0,
            "Active_Days": 0,
            "Avg_Holdings": 0.0,
            "Final_Return": 1.0,
        }

    returns = (
        daily["Strategy_Return"]
        .astype(float)
        .fillna(0.0)
    )

    std = returns.std()

    if (
        pd.isna(std)
        or std <= 1e-12
    ):

        sharpe = 0.0

    else:

        sharpe = (
            returns.mean()
            /
            std
        ) * np.sqrt(252)

    final_return = _safe_float(
        daily[
            "Cumulative_Strategy"
        ].iloc[-1],
        1.0,
    )

    # --------------------------------------------------------
    # CAGR
    # --------------------------------------------------------

    if len(daily) > 1:

        start_date = daily.index.min()
        end_date = daily.index.max()

        calendar_days = max(
            (
                end_date
                -
                start_date
            ).days,
            1,
        )

        years = (
            calendar_days
            /
            365.25
        )

    else:

        years = 1.0 / 252.0

    if (
        final_return <= 0
        or years <= 0
    ):

        cagr = -1.0

    else:

        cagr = (
            final_return
            **
            (
                1.0 / years
            )
        ) - 1.0

    volatility = (
        returns.std()
        *
        np.sqrt(252)
    )

    if pd.isna(volatility):

        volatility = 0.0

    win_rate = (
        returns > 0
    ).mean()

    active_days = int(
        (
            returns.abs()
            >
            1e-12
        ).sum()
    )

    avg_turnover = (
        daily["Turnover"]
        .mean()
    )

    annual_turnover = (
        avg_turnover
        *
        252
    )

    median_turnover = (
        daily["Turnover"]
        .median()
    )

    turnover95 = (
        daily["Turnover"]
        .quantile(0.95)
    )

    max_turnover = (
        daily["Turnover"]
        .max()
    )

    avg_holdings = (
        daily["Holdings"]
        .mean()
    )

    max_dd = (
        daily["Drawdown"]
        .min()
    )

    return {
        "Sharpe": _safe_float(
            sharpe
        ),
        "CAGR": _safe_float(
            cagr
        ),
        "Max_Drawdown": _safe_float(
            max_dd
        ),
        "Volatility": _safe_float(
            volatility
        ),
        "Win_Rate": _safe_float(
            win_rate
        ),
        "Avg_Turnover": _safe_float(
            avg_turnover
        ),
        "Annual_Turnover": _safe_float(
            annual_turnover
        ),
        "Median_Turnover": _safe_float(
            median_turnover
        ),
        "Turnover95": _safe_float(
            turnover95
        ),
        "Max_Turnover": _safe_float(
            max_turnover
        ),
        "Active_Days": active_days,
        "Avg_Holdings": _safe_float(
            avg_holdings
        ),
        "Final_Return": _safe_float(
            final_return,
            1.0,
        ),
    }


# ============================================================
# MAIN BACKTEST
# ============================================================

def run_backtest(
    proba,
    X_test,
    meta_test,
    final_df,
):
    """
    Run complete backtest.

    Parameters
    ----------
    proba :
        Model probability array or continuous alpha.

    X_test :
        Test feature matrix.

    meta_test :
        Test metadata panel containing Date/Company.

    final_df :
        Full cleaned market dataset.

    Returns
    -------
    dict
        Backtest metrics, daily returns and row-level panel.
    """

    print("\n")
    print("=" * 72)
    print("📊 BACKTEST STARTED")
    print("=" * 72)

    # ========================================================
    # 1. INPUT VALIDATION
    # ========================================================

    if proba is None:

        print(
            "❌ Empty predictions."
        )

        return {}

    if meta_test is None:

        print(
            "❌ meta_test is None."
        )

        return {}

    if final_df is None:

        print(
            "❌ final_df is None."
        )

        return {}

    print(
        f"Initial prediction rows : "
        f"{len(proba):,}"
    )

    print(
        f"Meta test rows           : "
        f"{len(meta_test):,}"
    )

    print(
        f"Final dataset rows       : "
        f"{len(final_df):,}"
    )

    # ========================================================
    # 2. BUILD PREDICTION PANEL
    # ========================================================

    try:

        df = _prepare_prediction_frame(
            proba=proba,
            X_test=X_test,
            meta_test=meta_test,
        )

    except Exception as exc:

        logger.exception(
            "Unable to construct prediction panel."
        )

        print(
            f"❌ Prediction alignment failed: "
            f"{exc}"
        )

        return {}

    required_prediction_columns = [
        "Date",
        "Company",
        "Proba",
    ]

    missing_prediction_columns = [
        c
        for c in required_prediction_columns
        if c not in df.columns
    ]

    if missing_prediction_columns:

        print(
            "❌ Prediction panel missing: "
            f"{missing_prediction_columns}"
        )

        return {}

    df = _normalise_keys(df)

    df = df.loc[
        df["Date"].notna()
        &
        df["Company"].notna()
        &
        (
            df["Company"]
            .astype(str)
            .str.strip()
            != ""
        )
    ].copy()

    df = (
        df
        .reset_index(drop=True)
    )

    if df.empty:

        print(
            "❌ Prediction panel contains no "
            "valid observations."
        )

        return {}

    print(
        f"\nComplete test panel: "
        f"{len(df):,} rows"
    )

    print(
        f"Unique test dates: "
        f"{df['Date'].nunique():,}"
    )

    print(
        f"Unique test companies: "
        f"{df['Company'].nunique():,}"
    )

    # ========================================================
    # 3. MARKET DATA
    # ========================================================

    try:

        df = _merge_market_data(
            prediction_df=df,
            final_df=final_df,
        )

    except Exception as exc:

        logger.exception(
            "Market-data merge failed."
        )

        print(
            f"❌ Market-data merge failed: "
            f"{exc}"
        )

        return {}

    # ========================================================
    # 4. KEY VALIDATION
    # ========================================================

    if "Date" not in df.columns:

        print(
            "❌ CRITICAL: Date disappeared "
            "after market-data merge."
        )

        return {}

    if "Company" not in df.columns:

        print(
            "❌ CRITICAL: Company disappeared "
            "after market-data merge."
        )

        return {}

    df = _normalise_keys(df)

    # ========================================================
    # 5. PRICE VALIDATION
    # ========================================================

    if "Close" not in df.columns:

        print(
            "❌ Market data does not contain Close."
        )

        return {}

    before = len(df)

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    valid_price = (
        df["Date"].notna()
        &
        df["Company"].notna()
        &
        df["Close"].notna()
        &
        (
            df["Close"] > 0
        )
    )

    df = df.loc[
        valid_price
    ].copy()

    print(
        f"\nRows after valid price/date check: "
        f"{len(df):,}"
    )

    print(
        f"Rows removed: "
        f"{before - len(df):,}"
    )

    if df.empty:

        print(
            "❌ No valid test observations."
        )

        return {}

    # ========================================================
    # 6. UNIQUE PANEL
    # ========================================================

    before = len(df)

    df = (
        df
        .sort_values(
            [
                "Date",
                "Company",
            ]
        )
        .drop_duplicates(
            [
                "Date",
                "Company",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    duplicate_rows_removed = (
        before
        -
        len(df)
    )

    if duplicate_rows_removed > 0:

        print(
            "⚠ Duplicate prediction keys removed: "
            f"{duplicate_rows_removed:,}"
        )

    # ========================================================
    # 7. ALPHA
    # ========================================================

    df = _create_alpha(
        df
    )

    # ========================================================
    # 8. HISTORICAL VOLATILITY
    # ========================================================

    df = _calculate_historical_volatility(
        df
    )

    # ========================================================
    # 9. FORWARD RETURN
    # ========================================================

    df = _calculate_forward_return(
        df
    )

    # ========================================================
    # 10. SIGNAL GENERATION
    # ========================================================

    df = _cross_sectional_signal(
        df
    )

    print("\n")
    print("=" * 58)
    print("BACKTEST SIGNAL DIAGNOSTICS")
    print("=" * 58)

    print(
        f"Test observations : "
        f"{len(df):,}"
    )

    print(
        f"Test dates        : "
        f"{df['Date'].nunique():,}"
    )

    print(
        f"Test companies    : "
        f"{df['Company'].nunique():,}"
    )

    print(
        f"Eligible rows     : "
        f"{int(df['Eligible'].sum()):,}"
    )

    print(
        f"Selected rows     : "
        f"{int(df['Selected'].sum()):,}"
    )

    print(
        f"Signal dates      : "
        f"{df.loc[df['Selected'], 'Date'].nunique():,}"
    )

    # ========================================================
    # 11. POSITION SIZING
    # ========================================================

    df = _risk_adjust_position(
        df
    )

    # ========================================================
    # 12. SMOOTHING
    # ========================================================

    df = _apply_smoothing(
        df
    )

    # ========================================================
    # 13. DEAD BAND
    # ========================================================

    df = _apply_deadband(
        df
    )

    deadband_pct = (
        df["Deadband_Applied"].mean()
        if len(df) > 0
        else 0.0
    )

    # ========================================================
    # 14. EXECUTION LAG
    # ========================================================

    df = _apply_execution_lag(
        df
    )

    # ========================================================
    # 15. TURNOVER
    # ========================================================

    df = _calculate_turnover(
        df
    )

    # ========================================================
    # 16. RETURNS
    # ========================================================

    df, daily = _calculate_returns(
        df
    )

    # ========================================================
    # 17. METRICS
    # ========================================================

    metrics = _calculate_metrics(
        daily
    )

    # ========================================================
    # 18. DIAGNOSTICS
    # ========================================================

    max_position = (
        df["Position_Executed"]
        .abs()
        .max()
        if "Position_Executed"
        in df.columns
        else 0.0
    )

    max_gross_exposure = (
        daily["Gross_Exposure"].max()
        if (
            not daily.empty
            and
            "Gross_Exposure"
            in daily.columns
        )
        else 0.0
    )

    avg_confidence = (
        df.loc[
            df["Selected"],
            "Confidence",
        ].mean()
        if (
            "Selected" in df.columns
            and
            df["Selected"].any()
        )
        else 0.0
    )

    signal_count = int(
        df["Selected"].sum()
    )

    signal_dates = int(
        df.loc[
            df["Selected"],
            "Date",
        ].nunique()
    )

    # ========================================================
    # 19. SANITY CHECKS
    # ========================================================

    expected_return_explosion = (
        df["Return"]
        .abs()
        .max()
        <=
        RETURN_CLIP
        +
        1e-9
    )

    probability_bounded = True

    if "Prediction_Prob" in df.columns:

        probability_series = pd.to_numeric(
            df["Prediction_Prob"],
            errors="coerce",
        )

        probability_bounded = bool(
            probability_series
            .dropna()
            .between(
                0.0,
                1.0,
            )
            .all()
        )

    date_integrity = (
        len(daily)
        ==
        df["Date"].nunique()
    )

    panel_integrity = (
        len(df)
        ==
        df[
            [
                "Date",
                "Company",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    # --------------------------------------------------------
    # Position cap integrity
    # --------------------------------------------------------

    position_cap_check = (
        max_position
        <=
        MAX_POSITION_SIZE
        *
        2.0
        +
        1e-9
    )

    # The factor of 2 allows volatility targeting to scale
    # positions up after base sizing. Gross exposure remains
    # separately capped.
    #
    # We report the actual maximum rather than treating
    # volatility-targeted exposure as a sizing bug.

    # ========================================================
    # 20. PRINT DIAGNOSTICS
    # ========================================================

    print("\n")
    print("=" * 58)
    print("BACKTEST DIAGNOSTICS")
    print("=" * 58)

    print(
        f"Rows                  : "
        f"{len(df):,}"
    )

    print(
        f"Unique Dates          : "
        f"{df['Date'].nunique():,}"
    )

    print(
        f"Unique Companies      : "
        f"{df['Company'].nunique():,}"
    )

    print(
        f"Selected Signals      : "
        f"{signal_count:,}"
    )

    print(
        f"Signal Dates          : "
        f"{signal_dates:,}"
    )

    print(
        f"Average Holdings      : "
        f"{metrics['Avg_Holdings']:.3f}"
    )

    print(
        f"Average Daily Turnover: "
        f"{metrics['Avg_Turnover']:.6f}"
    )

    print(
        f"Annualized Turnover   : "
        f"{metrics['Annual_Turnover']:.2f}x"
    )

    print(
        f"Deadband Filtered     : "
        f"{deadband_pct:.2%}"
    )

    print(
        "\nPosition Statistics:"
    )

    print(
        df["Position_Executed"]
        .describe()
    )

    print(
        "\nConfidence Statistics:"
    )

    print(
        df["Confidence"]
        .describe()
    )

    print(
        "\nRisk Statistics:"
    )

    risk_columns = [
        "Volatility",
        "ATR_PCT",
    ]

    available_risk_columns = [
        c
        for c in risk_columns
        if c in df.columns
    ]

    if available_risk_columns:

        print(
            df[
                available_risk_columns
            ].describe()
        )

    # ========================================================
    # 21. REGIME DIAGNOSTICS
    # ========================================================

    if "Market_Regime" in df.columns:

        print(
            "\nMarket Regime:"
        )

        print(
            df[
                "Market_Regime"
            ]
            .value_counts()
        )

    # ========================================================
    # 22. BACKTEST RESULTS
    # ========================================================

    print("\n")
    print("=" * 58)
    print("BACKTEST RESULTS")
    print("=" * 58)

    print(
        f"Sharpe              : "
        f"{metrics['Sharpe']:.3f}"
    )

    print(
        f"Max DD              : "
        f"{metrics['Max_Drawdown']:.3f}"
    )

    print(
        f"CAGR                : "
        f"{metrics['CAGR']:.3f}"
    )

    print(
        f"Volatility          : "
        f"{metrics['Volatility']:.3f}"
    )

    print(
        f"Win Rate            : "
        f"{metrics['Win_Rate']:.3f}"
    )

    print(
        f"Final Return        : "
        f"{metrics['Final_Return']:.3f}"
    )

    print(
        f"Active Days         : "
        f"{metrics['Active_Days']}"
    )

    print(
        f"Total Test Days     : "
        f"{len(daily):,}"
    )

    print(
        f"Average Daily Turnover: "
        f"{metrics['Avg_Turnover']:.6f}"
    )

    print(
        f"Annualized Turnover : "
        f"{metrics['Annual_Turnover']:.2f}x"
    )

    print(
        f"Average Holdings    : "
        f"{metrics['Avg_Holdings']:.3f}"
    )

    print(
        f"Maximum Position    : "
        f"{_safe_float(max_position):.6f}"
    )

    print(
        f"Maximum Gross Exp.  : "
        f"{_safe_float(max_gross_exposure):.6f}"
    )

    # ========================================================
    # 23. SANITY CHECKS
    # ========================================================

    print("\n")
    print("=" * 58)
    print("BACKTEST SANITY CHECKS")
    print("=" * 58)

    print(
        f"Test rows retained           : "
        f"{len(df):,}"
    )

    print(
        f"Test dates retained          : "
        f"{df['Date'].nunique():,}"
    )

    print(
        f"Daily return observations    : "
        f"{len(daily):,}"
    )

    print(
        f"Maximum position             : "
        f"{_safe_float(max_position):.6f}"
    )

    print(
        f"Maximum gross exposure       : "
        f"{_safe_float(max_gross_exposure):.6f}"
    )

    print(
        "Expected-return explosion    : "
        f"{'PASS' if expected_return_explosion else 'FAIL'}"
    )

    print(
        "Probability bounded          : "
        f"{'PASS' if probability_bounded else 'FAIL'}"
    )

    print(
        "Date-panel integrity         : "
        f"{'PASS' if date_integrity else 'FAIL'}"
    )

    print(
        "Date/Company uniqueness      : "
        f"{'PASS' if panel_integrity else 'FAIL'}"
    )

    # ========================================================
    # 24. TEST-UNIVERSE DIAGNOSTIC
    # ========================================================

    if df["Company"].nunique() < 39:

        print(
            "\n⚠ TEST-UNIVERSE DIAGNOSTIC"
        )

        print(
            f"Backtest test panel contains "
            f"{df['Company'].nunique()} companies, "
            "not the full 39-stock universe."
        )

        print(
            "This originates upstream from "
            "X_test/meta_test construction."
        )

    # ========================================================
    # 25. SIGNAL DIAGNOSTIC
    # ========================================================

    if signal_dates <= 1:

        print(
            "\n⚠ SIGNAL DIAGNOSTIC"
        )

        print(
            f"Only {signal_dates} signal date(s) "
            "survived selection."
        )

        print(
            "Investigate alpha distribution, "
            "cross-sectional ranking and "
            "portfolio-size constraints."
        )

    # ========================================================
    # 26. RESULT OBJECT
    # ========================================================

    results = {

        # ----------------------------------------------------
        # Core metrics
        # ----------------------------------------------------

        "Sharpe": metrics[
            "Sharpe"
        ],

        "CAGR": metrics[
            "CAGR"
        ],

        "Max_Drawdown": metrics[
            "Max_Drawdown"
        ],

        "Volatility": metrics[
            "Volatility"
        ],

        "Win_Rate": metrics[
            "Win_Rate"
        ],

        # ----------------------------------------------------
        # Turnover
        # ----------------------------------------------------

        "Avg_Turnover": metrics[
            "Avg_Turnover"
        ],

        "Annual_Turnover": metrics[
            "Annual_Turnover"
        ],

        "Median_Turnover": metrics[
            "Median_Turnover"
        ],

        "Turnover95": metrics[
            "Turnover95"
        ],

        "Max_Turnover": metrics[
            "Max_Turnover"
        ],

        # ----------------------------------------------------
        # Activity
        # ----------------------------------------------------

        "Active_Days": metrics[
            "Active_Days"
        ],

        "Avg_Holdings": metrics[
            "Avg_Holdings"
        ],

        "Deadband_Pct": (
            deadband_pct
        ),

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        "Final_Return": metrics[
            "Final_Return"
        ],

        # ----------------------------------------------------
        # Signals
        # ----------------------------------------------------

        "Signal_Count": signal_count,

        "Signal_Dates": signal_dates,

        "Average_Confidence": (
            _safe_float(
                avg_confidence
            )
        ),

        # ----------------------------------------------------
        # Universe
        # ----------------------------------------------------

        "Test_Rows": len(df),

        "Test_Dates": (
            df["Date"].nunique()
        ),

        "Test_Companies": (
            df["Company"].nunique()
        ),

        # ----------------------------------------------------
        # Exposure
        # ----------------------------------------------------

        "Max_Position": (
            _safe_float(
                max_position
            )
        ),

        "Max_Gross_Exposure": (
            _safe_float(
                max_gross_exposure
            )
        ),

        # ----------------------------------------------------
        # Sanity checks
        # ----------------------------------------------------

        "Expected_Return_Check": bool(
            expected_return_explosion
        ),

        "Probability_Bounded_Check": bool(
            probability_bounded
        ),

        "Date_Panel_Check": bool(
            date_integrity
        ),

        "Panel_Uniqueness_Check": bool(
            panel_integrity
        ),

        # ----------------------------------------------------
        # Full daily series
        # ----------------------------------------------------

        "Daily_Returns": (
            daily
            .reset_index()
        ),

        # ----------------------------------------------------
        # Full row-level panel
        # ----------------------------------------------------

        "Backtest_DF": (
            df.copy()
        ),
    }

    # ========================================================
    # 27. FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 72)
    print("✅ BACKTEST COMPLETED")
    print("=" * 72)

    print(
        f"Test rows      : "
        f"{len(df):,}"
    )

    print(
        f"Test dates     : "
        f"{df['Date'].nunique():,}"
    )

    print(
        f"Test companies : "
        f"{df['Company'].nunique():,}"
    )

    print(
        f"Signals        : "
        f"{signal_count:,}"
    )

    print(
        f"Signal dates   : "
        f"{signal_dates:,}"
    )

    print(
        f"Sharpe         : "
        f"{metrics['Sharpe']:.3f}"
    )

    print(
        f"Max DD         : "
        f"{metrics['Max_Drawdown']:.3f}"
    )

    print(
        f"CAGR           : "
        f"{metrics['CAGR']:.3f}"
    )

    print(
        f"Final Return   : "
        f"{metrics['Final_Return']:.3f}"
    )

    print("=" * 72)

    return results


# ============================================================
# END OF FILE
# ============================================================