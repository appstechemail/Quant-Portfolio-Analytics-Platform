# ==========================================================
# MODEL UTILS
# ==========================================================
#
# FILE: model/utils.py
#
# PURPOSE:
# --------
# Institutional-grade utilities for:
#
#     • probability extraction
#     • ensemble generation
#     • model blending
#     • confidence filtering
#     • performance tracking
#
# FEATURES:
# ---------
#
# 1. Safe probability extraction
# 2. Decision-function fallback
# 3. Confidence-aware ensemble
# 4. Dynamic model weighting
# 5. Dispersion filtering
# 6. Leakage-safe performance tracking
#
# ==========================================================

import numpy as np


# ==========================================================
# SAFE PROBABILITY EXTRACTION
# ==========================================================
def _safe_predict_proba(model, X):

    """
    Safely extract probabilities from any model.

    Priority:
        1. predict_proba
        2. decision_function
        3. predict normalization
    """

    try:

        # ======================================
        # PREDICT_PROBA
        # ======================================
        if hasattr(model, "predict_proba"):

            proba = model.predict_proba(X)

            # Binary classifier
            if len(proba.shape) == 2:
                
                if proba.shape[1] == 2:
                    return np.asarray(
                        proba[:, 1]
                    ).flatten()

                return np.max(proba, axis=1)

            return proba

        # ======================================
        # DECISION FUNCTION
        # ======================================
        if hasattr(model, "decision_function"):

            scores = model.decision_function(X)

            # Sigmoid transform
            probs = 1 / (1 + np.exp(-scores))

            return probs

        # ======================================
        # PREDICT FALLBACK
        # ======================================
        preds = model.predict(X)

        preds = np.array(preds).astype(float)

        return np.clip(preds, 0, 1)

    except Exception as e:

        print(f"⚠️ Probability extraction failed: {e}")

        return None


# ==========================================================
# MODEL PROBABILITIES
# ==========================================================
def get_model_probabilities(
    models,
    X_test,
    X_test_scaled=None
):

    """
    Extract probabilities for all models.
    """

    probas = {}

    for name, model in models.items():

        name = name.lower()

        try:

            # ======================================
            # SCALED MODELS
            # ======================================
            if name in ["lr", "svm", "mlp"]:

                X_input = (
                    X_test_scaled
                    if X_test_scaled is not None
                    else X_test
                )

            else:

                X_input = X_test

            # ======================================
            # EXTRACT
            # ======================================
            p = _safe_predict_proba(
                model,
                X_input
            )

            # ======================================
            # FALLBACK
            # ======================================
            if p is None:

                print(
                    f"⚠️ Using fallback probability for {name}"
                )

                p = np.full(
                    len(X_test),
                    0.5
                )

            # ======================================
            # CLEANUP
            # ======================================
            p = np.nan_to_num(
                p,
                nan=0.5,
                posinf=0.5,
                neginf=0.5
            )

            p = np.clip(
                p,
                0.01,
                0.99
            )

            probas[name] = p

        except Exception as e:

            print(f"❌ Error in {name}: {e}")

            probas[name] = np.full(
                len(X_test),
                0.5
            )
        
    # ======================================
    # PROBABILITY CALIBRATION
    # ======================================

    for model_name in probas:

        p = probas[model_name]

        p = np.asarray(p).astype(float)

        p = np.nan_to_num(
            p,
            nan=0.5,
            posinf=0.5,
            neginf=0.5
        )

        # shrink extreme probabilities
        p = 0.5 + 0.8 * (p - 0.5)

        p = np.clip(
            p,
            0.05,
            0.95
        )

        probas[model_name] = p

    return probas


# ==========================================================
# RECENT MODEL PERFORMANCE
# ==========================================================
def get_recent_model_performance(
    results,
    window=120
):

    """
    Computes rolling recent Sharpe ratio.
    """

    scores = {}

    for name, df in results.items():

        if df is None or df.empty:
            continue

        if "Strategy_Return" not in df.columns:
            continue

        strat = (
            df["Strategy_Return"]
            .dropna()
            .tail(window)
        )

        if len(strat) < 20:
            continue

        std = strat.std()

        if std <= 1e-9:
            continue

        sharpe = (
            strat.mean()
            /
            (std + 1e-9)
        ) * np.sqrt(252)

        scores[name.lower()] = sharpe

    return scores


# ==========================================================
# CONFIDENCE-AWARE ENSEMBLE
# ==========================================================
def combine_probabilities(
    probas,
    weights=None
):

    """
    Institutional ensemble engine.

    Features:
        • dynamic weighting
        • confidence filtering
        • disagreement filtering
        • robust normalization
    """

    if not probas:
        return np.array([])

    # ==========================================
    # PREPARE MATRIX
    # ==========================================
    model_names = list(probas.keys())

    prob_array = np.column_stack([
        probas[m]
        for m in model_names
    ])

    # ==========================================
    # BASE WEIGHTS
    # ==========================================
    if weights is None or len(weights) == 0:

        weights_arr = (
            np.ones(len(model_names))
            /
            len(model_names)
        )

    else:

        weights_arr = np.array([
            weights.get(m, 0)
            for m in model_names
        ])

        weights_arr = (
            weights_arr
            /
            (weights_arr.sum() + 1e-9)
        )

    # ==========================================
    # MODEL CONFIDENCE
    # ==========================================
    confidence = np.abs(
        prob_array - 0.5
    )

    # average row confidence
    row_conf = confidence.mean(axis=1)

    # ==========================================
    # DYNAMIC CONFIDENCE WEIGHTING
    # ==========================================
    dynamic_weights = confidence * weights_arr

    dynamic_weights = (
        dynamic_weights
        /
        (
            dynamic_weights.sum(axis=1, keepdims=True)
            + 1e-9
        )
    )

    # ==========================================
    # WEIGHTED ENSEMBLE
    # ==========================================
    combined = np.sum(
        prob_array * dynamic_weights,
        axis=1
    )

    # ==========================================
    # DISPERSION FILTER
    # ==========================================
    dispersion = np.std(
        prob_array,
        axis=1
    )

    # High disagreement → neutralize
    combined = np.where(
        dispersion > 0.25,
        0.5,
        combined
    )

    # ==========================================
    # LOW CONFIDENCE FILTER
    # ==========================================
    combined = np.where(
        row_conf < 0.05,
        0.5,
        combined
    )

    # ==========================================
    # FINAL CLIP
    # ==========================================
    combined = np.clip(
        combined,
        0.01,
        0.99
    )

    return combined

# ==========================================================
# ENSEMBLE CONFIDENCE
# ==========================================================
def compute_ensemble_confidence(
    probas
):

    if not probas:
        return np.array([])

    matrix = np.column_stack(
        list(probas.values())
    )

    dispersion = matrix.std(axis=1)

    confidence = np.exp(-5 * dispersion)

    confidence = np.clip(
        confidence,
        0,
        1
    )

    return confidence