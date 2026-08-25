import numpy as np
import pandas as pd
from config.config import CONFIG

cfg = CONFIG["REGIME"]

w_vol = cfg["WEIGHTS"]["VOLATILITY"]

w_sharpe = cfg["METRICS"]["SHARPE"]
w_dd = cfg["METRICS"]["DRAWDOWN"]

r_normalize = cfg["NORMALIZE"]
r_use_siftmax = cfg["USE_SOFTMAX"]


# ==========================================================
# 2. GENERIC WEIGHT ADAPTATION (NO HARD WEIGHTS)
# ==========================================================
def compute_auto_weights(results):

    import numpy as np
    import pandas as pd

    print("\n⚖️ Computing adaptive weights...")

    if not results:
        return {}

    metrics_dict = {}

    # =========================
    # 1. EXTRACT METRICS
    # =========================
    for name, df in results.items():

        if name.upper() == "ENSEMBLE":
            continue

        if df is None or df.empty:
            continue

        strat = df["Strategy_Return"].dropna()

        if len(strat) < 15:   # 🔥 more stability
            continue

        mean_ret = strat.mean()
        std_ret = strat.std()

        if std_ret == 0:
            continue

        sharpe = mean_ret / (std_ret + 1e-9)

        drawdown = df["Drawdown"].min() if "Drawdown" in df else 0
        volatility = std_ret

        metrics_dict[name] = {
            "sharpe": sharpe,
            "drawdown": drawdown,
            "volatility": volatility
        }

    # =========================
    # 2. FALLBACK
    # =========================
    if len(metrics_dict) == 0:
        keys = list(results.keys())
        return {k: 1 / len(keys) for k in keys}

    df_metrics = pd.DataFrame(metrics_dict).T.fillna(0)

    # =========================
    # 3. ROBUST NORMALIZATION
    # =========================
    # 🔥 Use rank + clip to avoid outliers
    df_rank = df_metrics.rank(pct=True)

    df_rank = df_rank.clip(0.05, 0.95)

    # =========================
    # 4. AUTO METRIC IMPORTANCE
    # =========================
    # 🔥 entropy-based weighting (better than variance)
    metric_matrix = df_rank.values

    entropy = -np.nansum(
        metric_matrix * np.log(metric_matrix + 1e-9), axis=0
    )

    metric_weights = 1 - entropy
    metric_weights = metric_weights / (metric_weights.sum() + 1e-9)

    w_sharpe, w_dd, w_vol = metric_weights

    # =========================
    # 5. SCORE CALCULATION
    # =========================
    scores = {}

    for name in df_rank.index:

        sharpe_rank = df_rank.loc[name, "sharpe"]
        dd_rank = df_rank.loc[name, "drawdown"]
        vol_rank = df_rank.loc[name, "volatility"]

        raw_sharpe = df_metrics.loc[name, "sharpe"]

        score = (
            w_sharpe * sharpe_rank +
            w_dd * (1 - dd_rank) +
            w_vol * (1 - vol_rank) +
            0.1 * np.tanh(raw_sharpe)   # 🔥 mild nonlinear boost
        )

        scores[name] = max(score, 1e-6)

    print("📊 Model Scores:", scores)

    # =========================
    # 6. CONVERT TO WEIGHTS
    # =========================
    model_names = list(scores.keys())
    vals = np.array([scores[k] for k in model_names])

    weights_arr = vals / (vals.sum() + 1e-9)

    # =========================
    # 7. SOFT CLIPPING (IMPORTANT)
    # =========================
    weights_arr = np.clip(weights_arr, 0.05, 0.6)
    weights_arr = weights_arr / weights_arr.sum()

    weights = dict(zip(model_names, weights_arr))

    # =========================
    # 8. KEEP TOP MODELS ONLY
    # =========================
    max_models = min(5, len(weights))

    weights = dict(
        sorted(weights.items(), key=lambda x: x[1], reverse=True)[:max_models]
    )

    # =========================
    # 9. FINAL NORMALIZATION
    # =========================
    total = sum(weights.values()) + 1e-9
    weights = {k: v / total for k, v in weights.items()}

    print("⚖️ Final Weights:", weights)

    return weights





# ==========================================================
# 3. REGIME-BASED WEIGHT ADJUSTMENT (GENERIC)
# ==========================================================
def adjust_weights_by_regime(weights, final_df):

    import numpy as np

    if not weights:
        return weights

    regime = final_df["Market_Regime"].iloc[-1]
    strength = final_df["Regime_Strength"].iloc[-1]

    print(f"\n🌍 Regime: {regime} | Strength: {strength:.2f}")

    w = weights.copy()

    # =========================
    # 1. CONVERT TO ARRAY
    # =========================
    keys = list(w.keys())
    vals = np.array(list(w.values()))

    # =========================
    # 2. MEASURE CONCENTRATION
    # =========================
    # Higher variance = one model dominating
    concentration = np.var(vals)

    # =========================
    # 3. ADAPTIVE EXPONENT (🔥 CORE LOGIC)
    # =========================
    # Base exponent depends on regime strength
    # Weak regime → flatten
    # Strong regime → concentrate
    base_exp = 1 + (strength - 0.5)   # range approx [0.5 → 1.5]

    # Adjust based on concentration
    # If already concentrated → reduce aggression
    adj_exp = base_exp * (1 - concentration)

    # =========================
    # 4. REGIME TYPE ADJUSTMENT
    # =========================
    if "VOLATILE" in regime:
        # Reduce dominance in volatile markets
        adj_exp *= 0.9

    elif "TREND" in regime:
        # Allow stronger models to dominate slightly
        adj_exp *= 1.1

    # Clamp exponent (safety)
    adj_exp = np.clip(adj_exp, 0.7, 1.3)

    # =========================
    # 5. APPLY TRANSFORMATION
    # =========================
    vals = np.power(vals, adj_exp)

    # =========================
    # 6. NORMALIZE
    # =========================
    vals = vals / (vals.sum() + 1e-9)

    # =========================
    # 7. PREVENT EXTREMES
    # =========================
    vals = np.clip(vals, 0.05, 0.7)
    vals = vals / vals.sum()

    adjusted_weights = dict(zip(keys, vals))

    # =========================
    # DEBUG
    # =========================
    print(f"📊 Concentration: {concentration:.4f}")
    print(f"⚙️ Adjustment Exponent: {adj_exp:.3f}")
    print("⚖️ Adjusted Weights:", adjusted_weights)

    return adjusted_weights
