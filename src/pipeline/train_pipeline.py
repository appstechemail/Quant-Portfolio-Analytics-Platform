# ==========================================================
# TRAIN PIPELINE
# ==========================================================
#
# PURPOSE
# ----------------------------------------------------------
# Centralized prediction + probability pipeline
# for all trained ML models.
#
# FEATURES
# ----------------------------------------------------------
# ✔ Calibrated probabilities
# ✔ Leakage-safe
# ✔ Ensemble-ready
# ✔ Scaled/unscaled model handling
# ✔ Robust prediction handling
#
# ==========================================================

# ==========================================================
# IMPORTS
# ==========================================================
import numpy as np

from models.model_utils import (
    get_model_probabilities
)


# ==========================================================
# GET PREDICTIONS
# ==========================================================
def get_predictions(
    models,
    X_test,
    X_test_scaled,
    ensemble_proba,
    threshold
):

    predictions = {}

    # ======================================================
    # MODEL PREDICTIONS
    # ======================================================
    for name, model in models.items():

        try:

            # ------------------------------------------------
            # SCALED MODELS
            # ------------------------------------------------
            if name.lower() in [
                "lr",
                "svm",
                "mlp"
            ]:

                X_input = X_test_scaled

            # ------------------------------------------------
            # TREE MODELS
            # ------------------------------------------------
            else:

                X_input = X_test

            preds = model.predict(
                X_input
            )

            predictions[name.lower()] = preds

        except Exception as e:

            print(
                f"❌ Prediction error "
                f"for {name}: {e}"
            )

            predictions[name.lower()] = np.zeros(
                len(X_test)
            )

    # ======================================================
    # ENSEMBLE PREDICTION
    # ======================================================
    predictions["ensemble"] = (
        ensemble_proba > threshold
    ).astype(int)

    return predictions


# ==========================================================
# GET ALL MODEL PROBABILITIES
# ==========================================================
def get_all_probabilities(
    models,
    X_test,
    X_test_scaled=None
):

    return get_model_probabilities(
        models=models,
        X_test=X_test,
        X_test_scaled=X_test_scaled
    )