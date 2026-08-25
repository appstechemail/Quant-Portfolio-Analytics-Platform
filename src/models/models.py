# ==========================================================
# GENERIC MODEL TRAINING ENGINE
# Production-Grade Quant ML Training Pipeline
# ==========================================================
#
# WHAT THIS MODULE DOES
# ----------------------------------------------------------
# This module is responsible for:
#
# 1. Dynamically configuring ML models
# 2. Training multiple models safely
# 3. Applying probability calibration
# 4. Handling scaled vs non-scaled models
# 5. Supporting regime-aware ensembles
# 6. Preventing unstable model behavior
# 7. Improving probability quality
# 8. Making models more robust for trading
#
# ==========================================================
# KEY IMPROVEMENTS
# ==========================================================
#
# ✅ Probability Calibration
#    Improves:
#    - ranking quality
#    - ensemble quality
#    - confidence filtering
#    - live prediction stability
#
# ✅ Reduced Overfitting
#    Through:
#    - smaller tree depth
#    - regularization
#    - subsampling
#    - early stopping
#
# ✅ Dynamic Complexity Scaling
#    Model complexity adapts automatically
#    based on dataset size.
#
# ✅ Safer Ensemble Inputs
#    Probabilities become smoother and
#    more comparable across models.
#
# ✅ Supports:
#    - Logistic Regression
#    - XGBoost
#    - LightGBM
#    - CatBoost
#    - Random Forest
#    - SVM
#    - MLP
#
# ==========================================================
# IMPORTANT NOTES
# ==========================================================
#
# 1. Scaled Models
#    These require standardized features:
#
#    - LR
#    - SVM
#    - MLP
#
# 2. Tree Models
#    These use raw features:
#
#    - XGB
#    - LGB
#    - CAT
#    - RF
#
# 3. Calibration
#    Applied mainly to:
#
#    - RF
#    - SVM
#    - XGB
#
#    because these models usually produce
#    overconfident probabilities.
#
# ==========================================================
# TRADING SYSTEM DESIGN GOAL
# ==========================================================
#
# The goal is NOT:
#    maximum train accuracy
#
# The goal IS:
#    stable probability ranking
#    robust cross-sectional alpha
#    low drawdown
#    realistic live trading behavior
#
# ==========================================================


# ==========================================================
# IMPORTS
# ==========================================================
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ==========================================================
# GLOBAL SETTINGS
# ==========================================================
RANDOM_STATE = 42


# ==========================================================
# DYNAMIC PARAMETER GENERATOR
# ==========================================================
def get_dynamic_params(X_train):

    n_samples = len(X_train)
    n_features = X_train.shape[1]

    # ======================================================
    # AUTO MODEL COMPLEXITY
    # ======================================================
    if n_samples < 5000:

        tree_estimators = 300
        tree_depth = 4
        learning_rate = 0.04

    elif n_samples < 50000:

        tree_estimators = 500
        tree_depth = 4
        learning_rate = 0.03

    else:

        tree_estimators = 700
        tree_depth = 5
        learning_rate = 0.02

    # ======================================================
    # MLP AUTO SIZE
    # ======================================================
    hidden1 = 64
    hidden2 = 32

    return {

        # ==================================================
        # LOGISTIC REGRESSION
        # ==================================================
        "lr": dict(
            max_iter=3000,
            C=0.1,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        # ==================================================
        # XGBOOST
        # ==================================================
        "xgb": dict(
            n_estimators=tree_estimators,
            max_depth=tree_depth,
            learning_rate=learning_rate,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=8,
            gamma=1.5,
            reg_alpha=0.5,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        ),

        # ==================================================
        # LIGHTGBM
        # ==================================================
        "lgb": dict(
            n_estimators=tree_estimators,
            learning_rate=learning_rate,
            num_leaves=min(31, 2 ** tree_depth),
            max_depth=tree_depth,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_samples=50,
            objective="binary",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            deterministic=True,
            force_col_wise=True,
            n_jobs=-1
        ),

        # ==================================================
        # CATBOOST
        # ==================================================
        "cat": dict(
            iterations=tree_estimators,
            depth=tree_depth,
            learning_rate=learning_rate,
            l2_leaf_reg=5,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=RANDOM_STATE,
            verbose=0
        ),

        # ==================================================
        # RANDOM FOREST
        # ==================================================
        "rf": dict(
            n_estimators=500,
            max_depth=4,
            min_samples_split=50,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        # ==================================================
        # SVM
        # ==================================================
        "svm": dict(
            probability=True,
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        # ==================================================
        # MLP
        # ==================================================
        "mlp": dict(
            hidden_layer_sizes=(hidden1, hidden2),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate="adaptive",
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
            random_state=RANDOM_STATE
        )
    }


# ==========================================================
# MODEL FACTORY
# ==========================================================
def create_model(name, params):

    # ======================================================
    # LOGISTIC REGRESSION
    # ======================================================
    if name == "lr":

        return LogisticRegression(**params)

    # ======================================================
    # XGBOOST + CALIBRATION
    # ======================================================
    elif name == "xgb":

        base_model = XGBClassifier(**params)

        return CalibratedClassifierCV(
            estimator=base_model,
            # For your dataset size (~30k rows), you should switch to: isotonic from sigmoid
            method="isotonic",
            cv=3
        )

    # ======================================================
    # LIGHTGBM
    # ======================================================
    elif name == "lgb":
        base_model = LGBMClassifier(**params)
        return CalibratedClassifierCV( estimator=base_model, method="isotonic", cv=3 )

    # ======================================================
    # CATBOOST
    # ======================================================
    elif name == "cat":

        base_model = CatBoostClassifier(**params)
        return CalibratedClassifierCV( estimator=base_model, method="isotonic", cv=3 )

    # ======================================================
    # RANDOM FOREST + CALIBRATION
    # ======================================================
    elif name == "rf":

        base_model = RandomForestClassifier(**params)

        return CalibratedClassifierCV(
            estimator=base_model,
            # For your dataset size (~30k rows), you should switch to: isotonic from sigmoid
            method="isotonic",
            cv=3
        )

    # ======================================================
    # SVM + CALIBRATION
    # ======================================================
    elif name == "svm":

        base_model = SVC(**params)

        return CalibratedClassifierCV(
            estimator=base_model,
            method="sigmoid",
            cv=3
        )

    # ======================================================
    # MLP
    # ======================================================
    elif name == "mlp":

        return MLPClassifier(**params)

    return None


# ==========================================================
# TRAIN MODELS
# ==========================================================
def train_models(
    X_train,
    y_train,
    X_train_scaled,
    selected_models=None
):

    print("\n🤖 TRAINING MODELS...")

    # ======================================================
    # TARGET SAFETY
    # ======================================================
    unique_classes = sorted(np.unique(y_train))

    print(f"📊 Target Classes: {unique_classes}")

    if len(unique_classes) < 2:

        raise ValueError(
            "Training failed → only one target class"
        )

    # ======================================================
    # AUTO PARAMETERS
    # ======================================================
    dynamic_params = get_dynamic_params(X_train)

    # ======================================================
    # MODEL DEFINITIONS
    # True  = scaled input required
    # False = raw input
    # ======================================================
    model_info = {

        "lr": True,
        "xgb": False,
        "lgb": False,
        "cat": False,
        "rf": False,
        "svm": True,
        "mlp": True
    }

    # ======================================================
    # DEFAULT MODELS
    # ======================================================
    if selected_models is None:

        selected_models = [
            "xgb",
            "rf",
            "lr",
            "mlp"
        ]

    selected_models = [

        m.lower()

        for m in selected_models

        if m.lower() in model_info
    ]

    print(
        f"\n📊 Selected Models: "
        f"{selected_models}"
    )

    models = {}

    # ======================================================
    # TRAIN LOOP
    # ======================================================
    for name in selected_models:

        try:

            print(
                f"\n⚙️ Training "
                f"{name.upper()}"
            )

            params = dynamic_params[name]

            model = create_model(name, params)

            if model is None:
                continue

            # ==================================================
            # SCALED VS RAW
            # ==================================================
            if model_info[name]:

                model.fit(
                    X_train_scaled,
                    y_train
                )

            else:

                model.fit(
                    X_train,
                    y_train
                )

            models[name] = model

            print(
                f"✅ {name.upper()} "
                f"trained successfully"
            )

        except Exception as e:

            print(
                f"❌ {name.upper()} failed: "
                f"{e}"
            )

    # ======================================================
    # FINAL SAFETY
    # ======================================================
    if len(models) == 0:

        print(
            "\n❌ No models "
            "successfully trained"
        )

    else:

        print(
            "\n✅ Final trained models:"
        )

        print(list(models.keys()))

    return models