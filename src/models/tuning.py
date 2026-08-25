# ==========================================
# HYPERPARAMETER TUNING
# ==========================================
import numpy as np
from sklearn.model_selection import ParameterGrid


# ==========================================
# FIX PARAM GRID (ENSURE LIST FORMAT)
# ==========================================
def ensure_param_grid_format(param_grid):
    fixed_grid = {}

    for k, v in param_grid.items():
        if isinstance(v, (list, tuple)):
            fixed_grid[k] = v
        else:
            fixed_grid[k] = [v]

    return fixed_grid


# ==========================================
# MAIN TUNING FUNCTION
# ==========================================
def tune_model(
    ModelClass,
    param_grid,
    X_train,
    y_train,
    X_test,
    y_test,
    meta_test,
    final_df,
    run_backtest,
    evaluate_strategy,
    scaled=False,
    X_train_scaled=None,
    X_test_scaled=None,
    verbose=True
):

    best_model = None
    best_score = -np.inf
    best_params = None

    # Fix param grid
    param_grid = ensure_param_grid_format(param_grid)

    # Loop through combinations
    for params in ParameterGrid(param_grid):

        try:
            model = ModelClass(**params)

            # =========================
            # TRAIN MODEL
            # =========================
            if scaled:
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)

            # =========================
            # GET PROBABILITIES
            # =========================
            if hasattr(model, "predict_proba"):
                if scaled:
                    proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    proba = model.predict_proba(X_test)[:, 1]
            else:
                # fallback (rare case)
                if scaled:
                    preds = model.predict(X_test_scaled)
                else:
                    preds = model.predict(X_test)

                proba = np.array(preds)

            # =========================
            # BACKTEST
            # =========================
            result = run_backtest(proba, X_test, meta_test, final_df)

            # =========================
            # EVALUATE
            # =========================
            metrics = evaluate_strategy(result)

            score = metrics.get("Sharpe Ratio", -np.inf)

            # Safety check
            if np.isnan(score):
                score = -np.inf

            # =========================
            # TRACK BEST MODEL
            # =========================
            if score > best_score:
                best_score = score
                best_model = model
                best_params = params

                if verbose:
                    print(f"🔝 New Best Score: {best_score:.4f}")
                    print(f"⚙️ Params: {best_params}")

        except Exception as e:
            if verbose:
                print(f"❌ Skipping params {params} | Error: {e}")
            continue

    return best_model, best_params, best_score
