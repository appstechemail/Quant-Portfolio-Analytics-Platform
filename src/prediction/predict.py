import logging
import numpy as np
import pandas as pd

from config.config import CONFIG
from src.portfolio.portfolio_selection import build_portfolio


logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

PORTFOLIO_CONFIG = CONFIG.get("PORTFOLIO", {})
MODEL_CONFIG = CONFIG.get("MODEL", {})
TARGET_CONFIG = CONFIG.get("TARGET", {})


BUY_THRESHOLD = float(
    PORTFOLIO_CONFIG.get("BUY_THRESHOLD", 0.60)
)

STRONG_BUY_THRESHOLD = float(
    PORTFOLIO_CONFIG.get("STRONG_BUY_THRESHOLD", 0.75)
)

SELL_THRESHOLD = float(
    PORTFOLIO_CONFIG.get("SELL_THRESHOLD", 0.40)
)

STRONG_SELL_THRESHOLD = float(
    PORTFOLIO_CONFIG.get("STRONG_SELL_THRESHOLD", 0.25)
)

MIN_CONFIDENCE = float(
    PORTFOLIO_CONFIG.get(
        "MIN_CONFIDENCE",
        MODEL_CONFIG.get("MIN_CONFIDENCE", 0.02)
    )
)

MODEL_THRESHOLD = float(
    MODEL_CONFIG.get("THRESHOLD", BUY_THRESHOLD)
)

PREDICT_DAYS_AHEAD = int(
    TARGET_CONFIG.get("PREDICT_DAYS_AHEAD", 5)
)

MAX_TARGET_RETURN = float(
    TARGET_CONFIG.get(
        "MAX_THRESHOLD",
        TARGET_CONFIG.get("MAX_DAILY_MOVE_CAP", 0.30)
    )
)

MIN_TARGET_RETURN = float(
    TARGET_CONFIG.get(
        "MIN_THRESHOLD",
        0.005
    )
)

MIN_PORTFOLIO_SIZE = int(
        PORTFOLIO_CONFIG.get(
            "MIN_PORTFOLIO_SIZE",
            5,
        )
    )


# =============================================================================
# HELPERS
# =============================================================================

def _get_model_probability(
    model,
    X,
    X_scaled,
):
    """
    Generate positive-class probability from a trained classifier.

    Models such as LR/MLP/SVM use scaled features.
    Tree models generally use the original feature matrix.
    """

    model_name = type(model).__name__.lower()

    scaled_models = (
        "logistic",
        "mlp",
        "svc",
        "svm",
        "calibrated"
    )

    if any(x in model_name for x in scaled_models):
        X_input = X_scaled
    else:
        X_input = X

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"{type(model).__name__} does not provide predict_proba()."
        )

    proba = model.predict_proba(X_input)

    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(
            f"Invalid probability shape from {type(model).__name__}: "
            f"{proba.shape}"
        )

    return np.asarray(proba[:, 1], dtype=float)


def _normalise_model_weights(
    filtered_weights,
    model_names
):
    """
    Convert model weights into a normalized numpy array.
    """

    n_models = len(model_names)

    if n_models == 0:
        return np.array([], dtype=float)

    if filtered_weights is None:

        return np.ones(
            n_models,
            dtype=float
        ) / n_models

    try:

        if isinstance(filtered_weights, dict):

            weights = np.array(
                [
                    float(
                        filtered_weights.get(
                            name,
                            filtered_weights.get(
                                name.lower(),
                                0.0
                            )
                        )
                    )
                    for name in model_names
                ],
                dtype=float
            )

        else:

            weights = np.asarray(
                filtered_weights,
                dtype=float
            )

        if (
            len(weights) != n_models
            or
            not np.all(np.isfinite(weights))
            or
            weights.sum() <= 0
        ):

            logger.warning(
                "Invalid model weights. Falling back to equal weights."
            )

            return np.ones(
                n_models,
                dtype=float
            ) / n_models

        weights = np.clip(
            weights,
            0.0,
            None
        )

        if weights.sum() <= 0:

            return np.ones(
                n_models,
                dtype=float
            ) / n_models

        return weights / weights.sum()

    except Exception as exc:

        logger.warning(
            "Could not process model weights: %s. "
            "Falling back to equal weights.",
            exc
        )

        return np.ones(
            n_models,
            dtype=float
        ) / n_models


def _map_signal(probability):
    """
    Convert probability into directional signal.

    IMPORTANT:
    Probability is always interpreted as:

        P(Target = BUY)

    Therefore low probabilities are SELL signals.
    """

    if probability >= STRONG_BUY_THRESHOLD:
        return "STRONG BUY"

    if probability >= BUY_THRESHOLD:
        return "BUY"

    if probability <= STRONG_SELL_THRESHOLD:
        return "STRONG SELL"

    if probability <= SELL_THRESHOLD:
        return "SELL"

    return "HOLD"


def _calculate_volatility(result):
    """
    Obtain volatility without contaminating the model probability.

    Prefer an existing Volatility feature when available.
    Otherwise calculate cross-row volatility as a fallback.
    """

    if "Volatility" in result.columns:

        vol = pd.to_numeric(
            result["Volatility"],
            errors="coerce"
        )

        vol = vol.replace(
            [np.inf, -np.inf],
            np.nan
        )

        if vol.notna().any():
            return vol.fillna(
                vol.median()
                if np.isfinite(vol.median())
                else 0.02
            ).clip(
                lower=0.001,
                upper=0.50
            )

    # -------------------------------------------------------------------------
    # Fallback
    #
    # IMPORTANT:
    # Do not calculate pct_change() across different companies.
    # The old implementation did:
    #
    #     result["Close"].pct_change()
    #
    # which is invalid when latest_data contains multiple stocks.
    # -------------------------------------------------------------------------

    if (
        "Company" in result.columns
        and
        "Close" in result.columns
    ):

        temp = result.copy()

        temp["_Close"] = pd.to_numeric(
            temp["Close"],
            errors="coerce"
        )

        temp["_Company_Return"] = (
            temp.groupby("Company")["_Close"]
            .pct_change()
        )

        vol = (
            temp.groupby("Company")["_Company_Return"]
            .transform(
                lambda x: x.rolling(
                    20,
                    min_periods=5
                ).std()
            )
        )

        vol = vol.replace(
            [np.inf, -np.inf],
            np.nan
        )

        fallback = (
            vol.median()
            if vol.notna().any()
            else 0.02
        )

        if not np.isfinite(fallback):
            fallback = 0.02

        return vol.fillna(
            fallback
        ).clip(
            lower=0.001,
            upper=0.50
        )

    return pd.Series(
        0.02,
        index=result.index,
        dtype=float
    )


def _calculate_expected_return(
    result,
    volatility
):
    """
    Calculate a conservative expected return estimate.

    IMPORTANT:
    This is NOT the same thing as the take-profit distance.

    The old implementation used:

        Expected_Return = Take_Profit distance

    which allowed extremely large values to enter portfolio scoring.

    Here expected return is based on:

        probability edge
        × volatility
        × prediction horizon

    and is explicitly bounded.
    """

    probability = result["Probability"].astype(float)

    direction = np.sign(
        probability - 0.50
    )

    confidence = np.abs(
        probability - 0.50
    ) * 2.0

    # Expected return estimate.
    #
    # sqrt(horizon) scales volatility rather than multiplying
    # daily volatility linearly by the number of days.
    horizon_scale = np.sqrt(
        max(PREDICT_DAYS_AHEAD, 1)
    )

    expected_return = (
        direction
        * volatility
        * confidence
        * horizon_scale
    )

    expected_return = expected_return.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0.0)

    # -------------------------------------------------------------------------
    # Hard safety bound.
    #
    # This prevents values such as:
    #
    #     Expected_Return = 75
    #
    # from entering portfolio construction.
    # -------------------------------------------------------------------------

    expected_return = expected_return.clip(
        lower=-MAX_TARGET_RETURN,
        upper=MAX_TARGET_RETURN
    )

    return expected_return


# =============================================================================
# MAIN LIVE PREDICTION ENGINE
# =============================================================================

def predict_today(
    latest_data,
    scaler,
    models,
    filtered_weights=None,
    FEATURES=None
):
    """
    Generate live stock predictions and portfolio candidates.

    Pipeline:

        latest_data
            ↓
        model probabilities
            ↓
        weighted ensemble
            ↓
        BUY / HOLD / SELL classification
            ↓
        confidence
            ↓
        risk / expected return
            ↓
        LONG-only portfolio candidate filter
            ↓
        portfolio construction

    Key invariant:

        Probability = P(Target = BUY)

    Low probability stocks are SELL candidates and are NEVER
    passed into the long-only portfolio construction engine.
    """

    print("\n📡 GENERATING LIVE SIGNALS...")

    # =========================================================================
    # 1. SAFETY CHECKS
    # =========================================================================

    if latest_data is None or latest_data.empty:

        logger.warning(
            "predict_today received empty latest_data."
        )

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    if FEATURES is None or len(FEATURES) == 0:

        logger.error(
            "No FEATURES supplied to predict_today."
        )

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    if scaler is None:

        logger.error(
            "Scaler is None."
        )

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    if models is None or len(models) == 0:

        logger.error(
            "No trained models supplied."
        )

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    # =========================================================================
    # 2. FEATURE VALIDATION
    # =========================================================================

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in latest_data.columns
    ]

    if missing_features:

        logger.error(
            "Missing prediction features: %s",
            missing_features
        )

        print(
            "❌ Missing features:",
            missing_features
        )

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    # =========================================================================
    # 3. PREPARE MODEL INPUTS
    # =========================================================================

    try:

        X = latest_data[
            FEATURES
        ].copy()

        # Convert numeric features explicitly.
        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        if X.isna().any().any():

            nan_count = int(
                X.isna().sum().sum()
            )

            logger.warning(
                "Prediction matrix contains %d NaN values.",
                nan_count
            )

            # Use median fallback from the latest universe.
            X = X.fillna(
                X.median()
            )

            # Any all-NaN feature receives zero.
            X = X.fillna(0.0)

        X_scaled = scaler.transform(
            X
        )

    except Exception as exc:

        logger.exception(
            "Feature preparation failed."
        )

        print(
            f"❌ Feature error: {exc}"
        )

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    # =========================================================================
    # 4. MODEL PREDICTIONS
    # =========================================================================

    model_probabilities = []
    model_used = []

    print("\n🤖 GENERATING MODEL PROBABILITIES...")

    for name, model in models.items():

        try:

            probability = _get_model_probability(
                model=model,
                X=X,
                X_scaled=X_scaled
            )

            probability = np.asarray(
                probability,
                dtype=float
            )

            probability = np.nan_to_num(
                probability,
                nan=0.5,
                posinf=1.0,
                neginf=0.0
            )

            probability = np.clip(
                probability,
                0.0,
                1.0
            )

            if len(probability) != len(latest_data):

                raise ValueError(
                    f"Probability length mismatch: "
                    f"{len(probability)} != "
                    f"{len(latest_data)}"
                )

            model_probabilities.append(
                probability
            )

            model_used.append(
                str(name).lower()
            )

            print(
                f"✅ {name.upper()} probability generated"
            )

        except Exception as exc:

            logger.warning(
                "Skipping model %s: %s",
                name,
                exc
            )

            print(
                f"❌ Skipping {name}: {exc}"
            )

    if len(model_probabilities) == 0:

        logger.error(
            "No model produced valid probabilities."
        )

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    # =========================================================================
    # 5. MODEL WEIGHTS
    # =========================================================================

    weights = _normalise_model_weights(
        filtered_weights=filtered_weights,
        model_names=model_used
    )

    print(
        "\nMODEL ENSEMBLE"
    )

    print(
        "Models:",
        model_used
    )

    print(
        "Weights:",
        weights
    )

    # =========================================================================
    # 6. WEIGHTED PROBABILITY ENSEMBLE
    # =========================================================================
    #
    # IMPORTANT CHANGE:
    #
    # OLD:
    #
    #     probability
    #       -> confidence
    #       -> signed confidence
    #       -> ensemble
    #       -> probability
    #
    # NEW:
    #
    #     model probability
    #       -> weighted probability
    #       -> final probability
    #
    # This preserves the semantic meaning:
    #
    #     Probability = P(BUY)
    #
    # =========================================================================

    probability_matrix = np.column_stack(
        model_probabilities
    )

    final_proba = np.average(
        probability_matrix,
        axis=1,
        weights=weights
    )

    final_proba = np.nan_to_num(
        final_proba,
        nan=0.5,
        posinf=1.0,
        neginf=0.0
    )

    final_proba = np.clip(
        final_proba,
        0.0,
        1.0
    )

    # =========================================================================
    # 7. RESULT FRAME
    # =========================================================================

    result = latest_data.copy()

    result["Probability"] = final_proba

    result["Confidence"] = (
        np.abs(
            result["Probability"] - 0.50
        ) * 2.0
    ).clip(
        0.0,
        1.0
    )

    # =========================================================================
    # 8. SIGNAL CLASSIFICATION
    # =========================================================================

    result["Signal"] = (
        result["Probability"]
        .apply(_map_signal)
    )

    print(
        "\n===== SIGNAL DISTRIBUTION ====="
    )

    print(
        result["Signal"].value_counts()
    )

    print(
        "\n===== PROBABILITY DISTRIBUTION ====="
    )

    print(
        result["Probability"].describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
    )

    # =========================================================================
    # 9. POSITION / SIDE
    # =========================================================================

    # Long-only interpretation.
    #
    # HOLD / SELL / STRONG SELL are NOT portfolio positions.

    result["Position"] = np.where(
        result["Probability"] >= BUY_THRESHOLD,
        1.0,
        0.0
    )

    result["Side"] = np.where(
        result["Position"] > 0,
        "LONG",
        "FLAT"
    )

    # =========================================================================
    # 10. DATES
    # =========================================================================

    if "Date" in result.columns:

        result["Signal_Date"] = pd.to_datetime(
            result["Date"],
            errors="coerce"
        )

        result["Action_Date"] = (
            result["Signal_Date"]
        )

        holding_days = (
            2
            +
            (
                1.0
                -
                result["Confidence"]
            )
            * 8
        ).round().astype(int)

        result["Exit_Date"] = (
            result["Action_Date"]
            +
            pd.to_timedelta(
                holding_days,
                unit="D"
            )
        )

    else:

        result["Signal_Date"] = pd.NaT
        result["Action_Date"] = pd.NaT
        result["Exit_Date"] = pd.NaT

    # =========================================================================
    # 11. REGIME
    # =========================================================================

    if "Market_Regime" not in result.columns:

        result["Market_Regime"] = "UNKNOWN"

    if "Regime_Strength" not in result.columns:

        result["Regime_Strength"] = 0.0

    # =========================================================================
    # 12. VOLATILITY
    # =========================================================================

    volatility = _calculate_volatility(
        result
    )

    result["Prediction_Volatility"] = (
        volatility
    )

    # =========================================================================
    # 13. EXPECTED RETURN
    # =========================================================================

    expected_return = _calculate_expected_return(
        result=result,
        volatility=volatility
    )

    result["Expected_Return"] = (
        expected_return
    )

    result["Expected_return(%)"] = (
        expected_return * 100.0
    )

    # =========================================================================
    # 14. TARGET PRICE
    # =========================================================================

    result["Target"] = (
        result["Close"]
        *
        (
            1.0
            +
            result["Expected_Return"]
        )
    )

    # =========================================================================
    # 15. STOP LOSS
    # =========================================================================

    confidence = result["Confidence"]

    sl_buffer = (
        volatility
        *
        (
            1.2
            +
            (1.0 - confidence)
        )
    )

    sl_buffer = sl_buffer.clip(
        lower=0.005,
        upper=MAX_TARGET_RETURN
    )

    result["Stop_Loss"] = np.where(

        result["Probability"] >= 0.50,

        result["Close"]
        *
        (
            1.0
            -
            sl_buffer
        ),

        result["Close"]
        *
        (
            1.0
            +
            sl_buffer
        )
    )

    # =========================================================================
    # 16. TAKE PROFIT
    # =========================================================================

    tp_buffer = (
        volatility
        *
        (
            2.0
            +
            confidence * 2.0
        )
    )

    tp_buffer = tp_buffer.clip(
        lower=0.005,
        upper=MAX_TARGET_RETURN
    )

    result["Take_Profit"] = np.where(

        result["Probability"] >= 0.50,

        result["Close"]
        *
        (
            1.0
            +
            tp_buffer
        ),

        result["Close"]
        *
        (
            1.0
            -
            tp_buffer
        )
    )

    # =========================================================================
    # 17. RISK / REWARD
    # =========================================================================

    result["Risk"] = (
        np.abs(
            result["Close"]
            -
            result["Stop_Loss"]
        )
    )

    result["Reward"] = np.where(

        result["Probability"] >= 0.50,

        np.abs(
            result["Take_Profit"]
            -
            result["Close"]
        ),

        0.0
    )

    result["RR_Ratio"] = (
        result["Reward"]
        /
        (
            result["Risk"]
            +
            1e-9
        )
    )

    result["RR_Ratio"] = (
        result["RR_Ratio"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
        .clip(
            lower=0.0,
            upper=100.0
        )
    )

    # =========================================================================
    # 18. ATR DIAGNOSTICS
    # =========================================================================

    print(
        "\nATR available:",
        "ATR_14" in result.columns
    )

    # =========================================================================
    # 19. FINAL SIGNAL OUTPUT
    # =========================================================================

    final_cols = [
        "Company",
        "Signal",
        "Signal_Date",
        "Action_Date",
        "Close",
        "Volume",
        "Probability",
        "Confidence",
        "Position",
        "Target",
        "Exit_Date",
        "Stop_Loss",
        "Take_Profit",
        "Expected_Return",
        "Expected_return(%)",
        "Risk",
        "Reward",
        "RR_Ratio",
        "Side",
        "ATR_14",
        "Market_Regime",
        "Regime_Strength",
        "Prediction_Volatility"
    ]

    display_cols = [
        column
        for column in final_cols
        if column in result.columns
    ]

    signal_output = result[
        display_cols
    ].copy()

    # =========================================================================
    # 20. SIGNAL DIAGNOSTICS
    # =========================================================================

    print("\n===== SIGNAL DISTRIBUTION =====")

    print(
        result["Signal"]
        .value_counts()
    )

    print("\nSignal Counts:")

    print(
        f"STRONG BUY       : "
        f"{int((result['Probability'] >= STRONG_BUY_THRESHOLD).sum()):,}"
    )

    print(
        f"BUY              : "
        f"{int(((result['Probability'] >= BUY_THRESHOLD) & (result['Probability'] < STRONG_BUY_THRESHOLD)).sum()):,}"
    )

    print(
        f"HOLD             : "
        f"{int((
            (result["Probability"] > SELL_THRESHOLD)
            &
            (result["Probability"] < BUY_THRESHOLD)
        ).sum()):,}"
    )

    print(
        f"SELL             : "
        f"{int((
            (result["Probability"] > STRONG_SELL_THRESHOLD)
            &
            (result["Probability"] <= SELL_THRESHOLD)
        ).sum()):,}"
    )

    print(
        f"STRONG SELL      : "
        f"{int((
            result["Probability"] <= STRONG_SELL_THRESHOLD
        ).sum()):,}"
    )
    # =======================================================================
    # 21. PORTFOLIO INPUT
    # =======================================================================

    portfolio_input = result.copy()

    if portfolio_input.empty:

        return (
            signal_output,
            pd.DataFrame()
        )

    portfolio_input["Prediction_Prob"] = (
        portfolio_input["Probability"]
    )

    if "Signal_Date" in portfolio_input.columns:

        portfolio_input["Date"] = (
            portfolio_input["Signal_Date"]
        )

    # =========================================================================
    # 22. LONG-ONLY BUY FILTER
    # =========================================================================
    #
    # THIS IS THE MOST IMPORTANT FIX.
    #
    # Confidence alone cannot be used to identify BUY candidates.
    #
    # Example:
    #
    # Probability = 0.20
    # Confidence  = 0.60
    #
    # That is a strong SELL, NOT a BUY.
    #
    # Therefore:
    #
    #     Probability >= BUY_THRESHOLD
    #
    # must happen BEFORE portfolio construction.
    # =========================================================================

    print(
        "\n============================================================"
    )

    print(
        "PORTFOLIO CANDIDATE FILTER"
    )

    print(
        "============================================================"
    )

    print(
        f"Initial Universe       : {len(portfolio_input)}"
    )

    buy_candidates = portfolio_input[
        portfolio_input["Probability"]
        >=
        BUY_THRESHOLD
    ].copy()

    print(
        f"After BUY probability : {len(buy_candidates)}"
    )

    if not buy_candidates.empty:

        print(
            buy_candidates[
                [
                    "Company",
                    "Probability",
                    "Confidence",
                    "Expected_Return",
                    "Market_Regime"
                ]
            ]
            .sort_values(
                "Probability",
                ascending=False
            )
        )

    # =========================================================================
    # 23. CONFIDENCE FILTER
    # =========================================================================

    buy_candidates = buy_candidates[
        buy_candidates["Confidence"]
        >=
        MIN_CONFIDENCE
    ].copy()

    print(
        f"After Confidence      : {len(buy_candidates)}"
    )

    # =========================================================================
    # 24. MODEL THRESHOLD DIAGNOSTIC
    # =========================================================================

    signal_count = (
        result["Probability"]
        >=
        MODEL_THRESHOLD
    ).sum()

    logger.info(
        "Signals above threshold %.2f : %d",
        MODEL_THRESHOLD,
        signal_count
    )

    # =========================================================================
    # 25. PORTFOLIO CONSTRUCTION
    # =========================================================================

    if buy_candidates.empty:

        print(
            "\n⚠ No BUY candidates survived portfolio filters."
        )

        print(
            "No portfolio will be fabricated from SELL/HOLD stocks."
        )

        return (
            signal_output,
            pd.DataFrame()
        )

    print(
        "\nPASSING TRUE BUY CANDIDATES "
        "TO PORTFOLIO ENGINE"
    )

    print(
        f"BUY Candidates : {len(buy_candidates)}"
    )

    print(
        buy_candidates[
            [
                "Company",
                "Probability",
                "Confidence",
                "Expected_Return",
                "Market_Regime"
            ]
        ]
        .sort_values(
            "Probability",
            ascending=False
        )
    )

    # =========================================================================
    # 26. BUILD PORTFOLIO
    # =========================================================================

    try:

        portfolio = build_portfolio(
            buy_candidates,
            top_n=3
        )

    except Exception as exc:

        logger.exception(
            "Portfolio construction failed."
        )

        print(
            f"❌ Portfolio builder failed: {exc}"
        )

        return (
            signal_output,
            pd.DataFrame()
        )

    # =========================================================================
    # 27. FINAL PORTFOLIO SAFETY FILTERS
    # =========================================================================
    #
    # Institutional rule:
    #
    #   Portfolio Builder
    #       ↓
    #   Probability Safety Filter
    #       ↓
    #   Selected Filter
    #       ↓
    #   Minimum Portfolio Size Gate
    #       ↓
    #   Weight Normalization
    #       ↓
    #   Final Validation
    #
    # IMPORTANT:
    # The minimum portfolio-size gate MUST be evaluated AFTER all
    # final eligibility/safety filters. Otherwise a portfolio can
    # appear sufficiently large before invalid rows are removed.
    # ============================================================================

    if portfolio is None:
        portfolio = pd.DataFrame()

    portfolio = portfolio.copy()


    # ---------------------------------------------------------------------------
    # 27A. Probability safety invariant
    # ---------------------------------------------------------------------------
    #
    # Long-only portfolio invariant:
    #
    #     Probability >= BUY_THRESHOLD
    #
    # No stock below the BUY threshold may survive into the final
    # live portfolio, regardless of any downstream fallback behaviour.
    # ---------------------------------------------------------------------------

    if not portfolio.empty:

        if "Probability" not in portfolio.columns:

            logger.error(
                "CRITICAL: Final portfolio missing 'Probability' column."
            )

            portfolio = portfolio.iloc[0:0].copy()

        else:

            probability = pd.to_numeric(
                portfolio["Probability"],
                errors="coerce",
            )

            invalid_probability_mask = (
                probability.isna()
                |
                (probability < BUY_THRESHOLD)
            )

            invalid_probability_count = int(
                invalid_probability_mask.sum()
            )

            if invalid_probability_count > 0:

                logger.error(
                    "CRITICAL: Removing %d portfolio rows "
                    "below BUY_THRESHOLD %.2f or with invalid probability.",
                    invalid_probability_count,
                    BUY_THRESHOLD,
                )

                portfolio = portfolio.loc[
                    ~invalid_probability_mask
                ].copy()


    # ---------------------------------------------------------------------------
    # 27B. Final Selected filter
    # ---------------------------------------------------------------------------
    #
    # If the portfolio builder provides the explicit Selected flag,
    # only Selected == 1 rows are allowed into the final portfolio.
    # ---------------------------------------------------------------------------

    if not portfolio.empty:

        if "Selected" in portfolio.columns:

            selected_mask = (
                pd.to_numeric(
                    portfolio["Selected"],
                    errors="coerce",
                )
                .fillna(0)
                == 1
            )

            removed_unselected = int(
                (~selected_mask).sum()
            )

            if removed_unselected > 0:

                logger.warning(
                    "Removed %d portfolio rows with Selected != 1.",
                    removed_unselected,
                )

            portfolio = portfolio.loc[
                selected_mask
            ].copy()


    # =======================================================================
    # 28. INSTITUTIONAL LIVE PORTFOLIO SIZE GATE
    # =======================================================================
    #
    # This gate is intentionally AFTER all final safety filters.
    #
    # Current configuration:
    #
    #     MIN_PORTFOLIO_SIZE = 5
    #
    # Therefore:
    #
    #     1 valid stock  -> REJECT
    #     2 valid stocks -> REJECT
    #     3 valid stocks -> REJECT
    #     4 valid stocks -> REJECT
    #     5+ valid stocks -> ACCEPT
    #
    # We do NOT manufacture additional positions merely to satisfy
    # the minimum size.
    # =======================================================================

    if (
        not portfolio.empty
        and len(portfolio) < MIN_PORTFOLIO_SIZE
    ):

        logger.warning(
            "Live portfolio rejected: "
            "only %d valid candidates survived final safety filters; "
            "minimum required=%d.",
            len(portfolio),
            MIN_PORTFOLIO_SIZE,
        )

        portfolio = portfolio.iloc[0:0].copy()


    # =======================================================================
    # 29. PORTFOLIO WEIGHT NORMALIZATION
    # =======================================================================
    #
    # Normalize only AFTER the portfolio has passed all eligibility filters.
    #
    # Never assign an artificial 1/N weight merely because the surviving
    # portfolio is too small. A portfolio that fails the size gate is empty.
    # =======================================================================

    if (
        not portfolio.empty
        and "Position_Weight" in portfolio.columns
    ):

        portfolio["Position_Weight"] = pd.to_numeric(
            portfolio["Position_Weight"],
            errors="coerce",
        ).fillna(0.0)

        portfolio["Position_Weight"] = (
            portfolio["Position_Weight"]
            .clip(lower=0.0)
        )

        weight_sum = float(
            portfolio["Position_Weight"].sum()
        )

        if weight_sum > 0.0:

            portfolio["Position_Weight"] = (
                portfolio["Position_Weight"]
                /
                weight_sum
            )

        else:

            logger.error(
                "CRITICAL: Final portfolio has no positive "
                "Position_Weight."
            )

            portfolio = portfolio.iloc[0:0].copy()


    # =======================================================================
    # 30. FINAL PORTFOLIO SORT
    # =======================================================================

    if (
        not portfolio.empty
        and "Position_Weight" in portfolio.columns
    ):

        portfolio = (
            portfolio
            .sort_values(
                "Position_Weight",
                ascending=False,
            )
            .copy()
        )


    # =======================================================================
    # 31. FINAL SCORE DIAGNOSTICS
    # =======================================================================
    #
    # Final_Score is a ranking/selection signal.
    # It is NOT the authoritative portfolio weight.
    # =======================================================================

    if (
        not portfolio.empty
        and "Final_Score" in portfolio.columns
    ):

        print(
            "\nFINAL SCORE DISTRIBUTION"
        )

        print(
            portfolio["Final_Score"].describe(
                percentiles=[
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                ]
            )
        )


    # =======================================================================
    # 32. FINAL PORTFOLIO VALIDATION
    # =======================================================================

    if not portfolio.empty:

        validation_errors = []


        # -------------------------------------------------------------------
        # 32A. Probability invariant
        # -------------------------------------------------------------------

        if "Probability" not in portfolio.columns:

            validation_errors.append(
                "Missing Probability column"
            )

        else:

            probability = pd.to_numeric(
                portfolio["Probability"],
                errors="coerce",
            )

            invalid_probability = (
                probability.isna()
                |
                (probability < BUY_THRESHOLD)
            )

            invalid_count = int(
                invalid_probability.sum()
            )

            if invalid_count > 0:

                validation_errors.append(
                    f"{invalid_count} stocks below BUY_THRESHOLD"
                )


        # -------------------------------------------------------------------
        # 32B. Selected invariant
        # -------------------------------------------------------------------

        if "Selected" in portfolio.columns:

            invalid_selected = (
                pd.to_numeric(
                    portfolio["Selected"],
                    errors="coerce",
                )
                .fillna(0)
                != 1
            )

            invalid_selected_count = int(
                invalid_selected.sum()
            )

            if invalid_selected_count > 0:

                validation_errors.append(
                    f"{invalid_selected_count} stocks have Selected != 1"
                )


        # -------------------------------------------------------------------
        # 32C. Weight invariant
        # ---------------------------------------------------------------

        if "Position_Weight" not in portfolio.columns:

            validation_errors.append(
                "Missing Position_Weight column"
            )

        else:

            weights = pd.to_numeric(
                portfolio["Position_Weight"],
                errors="coerce",
            )

            invalid_weight = (
                weights.isna()
                |
                (weights < 0.0)
            )

            invalid_weight_count = int(
                invalid_weight.sum()
            )

            if invalid_weight_count > 0:

                validation_errors.append(
                    f"{invalid_weight_count} invalid portfolio weights"
                )

            weight_sum = float(
                weights.sum()
            )

            if not np.isclose(
                weight_sum,
                1.0,
                atol=1e-6,
            ):

                validation_errors.append(
                    f"Portfolio weight sum={weight_sum:.8f}, expected 1.0"
                )


        # -------------------------------------------------------------------
        # 32D. Portfolio-size invariant
        # --------------------------------------------------------------------

        if len(portfolio) < MIN_PORTFOLIO_SIZE:

            validation_errors.append(
                "Portfolio size below minimum required threshold"
            )


        # -------------------------------------------------------------------
        # 32E. Final response
        # -------------------------------------------------------------------

        if validation_errors:

            logger.error(
                "CRITICAL: Final portfolio validation failed: %s",
                validation_errors,
            )

            portfolio = portfolio.iloc[0:0].copy()

        else:

            print(
                "\n🏆 FINAL PORTFOLIO"
            )

            portfolio_display_cols = [
                column
                for column in [
                    "Company",
                    "Signal",
                    "Probability",
                    "Confidence",
                    "Expected_Return",
                    "Final_Score",
                    "Portfolio_Rank",
                    "Position_Weight",
                    "Market_Regime",
                ]
                if column in portfolio.columns
            ]

            print(
                portfolio[
                    portfolio_display_cols
                ]
            )

            if "Position_Weight" in portfolio.columns:

                print(
                    "\nTotal Weight:",
                    portfolio["Position_Weight"].sum(),
                )

    else:

        print(
            "\n⚠ FINAL PORTFOLIO IS EMPTY"
        )


    # =======================================================================
    # 33. FINAL SUMMARY
    # =======================================================================

    print(
        "\n============================================================"
    )

    print(
        "FINAL PORTFOLIO SUMMARY"
    )

    print(
        "============================================================"
    )

    print(
        f"Portfolio Size        : {len(portfolio)}"
    )

    print(
        f"Minimum Required      : {MIN_PORTFOLIO_SIZE}"
    )

    print(
        f"BUY Threshold         : {BUY_THRESHOLD:.2f}"
    )

    print(
        f"Portfolio Accepted    : "
        f"{'YES' if not portfolio.empty else 'NO'}"
    )

    if portfolio.empty:

        print(
            "Portfolio Status      : REJECTED / NO DEPLOYABLE PORTFOLIO"
        )

    else:

        print(
            "Portfolio Status      : ACCEPTED"
        )

        if "Position_Weight" in portfolio.columns:

            print(
                f"Total Weight          : "
                f"{portfolio['Position_Weight'].sum():.6f}"
            )

    # =======================================================================
    # 34. RETURN FINAL RESULTS
    # =======================================================================

    return (
        signal_output,
        portfolio,
    )