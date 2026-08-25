# regime logic
import numpy as np


# ==========================================================
# REGIME → MODEL CANDIDATES
# ==========================================================
def get_regime_candidates(current_regime, all_models):
    """
    Returns preferred models for a regime (soft filter, not strict)
    """

    # Detect regime type
    regime_map = {

        "BULL": {"xgb", "lgb", "rf"},
        "BULL_VOLATILE": {"cat", "rf"},
        "BEAR": {"cat", "mlp"},
        "BEAR_VOLATILE": {"cat"},
        "SIDEWAYS": {"mlp", "svm"},
        "SIDEWAYS_VOLATILE": {"cat", "mlp"}
    }



    preferred = regime_map.get(current_regime, set())


    # 🔥 Soft filter → intersection
    candidates = [m for m in all_models if m in preferred]

    # 🔥 Fallback → if too few candidates
    if len(candidates) < 2:
        return all_models  # fallback to all models

    return candidates


# ==========================================================
# SMART MODEL SELECTION (REGIME + PERFORMANCE)
# ==========================================================
def select_models_smart(results, probas, current_regime, top_k=3, window=40):

    import numpy as np
    import pandas as pd

    # =========================
    # NORMALIZE KEYS
    # =========================
    probas = {k.lower(): v for k, v in probas.items()}
    results = {k.lower(): v for k, v in results.items()}

    # =========================
    # 1. RECENT PERFORMANCE
    # =========================

    recent_scores = {}

    for name, result in results.items():

        # ==========================================
        # SKIP INVALID
        # ==========================================
        if name not in probas:
            continue

        if result is None:
            continue

        # ==========================================
        # HANDLE BOTH DICT + DATAFRAME
        # ==========================================
        if isinstance(result, dict):

            if "Daily" not in result:
                continue

            daily_df = result["Daily"]

        else:

            daily_df = result

        if daily_df is None or daily_df.empty:
            continue

        # ==========================================
        # REQUIRE STRATEGY COLUMN
        # ==========================================
        if "Strategy_Return" not in daily_df.columns:
            continue

        # ==========================================
        # LAST 6 MONTHS
        # ==========================================
        strat = (
            daily_df["Strategy_Return"]
            .tail(window)
            .dropna()
        )

        # ==========================================
        # MINIMUM OBSERVATIONS
        # ==========================================
        min_obs = max(
            20,
            window // 4
        )

        if len(strat) < min_obs:
            continue

        # ==========================================
        # STABILITY FILTER
        # ==========================================
        std_ret = strat.std()

        if (
            std_ret is None
            or
            std_ret <= 1e-9
        ):
            continue

        # ==========================================
        # ANNUALIZED SHARPE
        # ==========================================
        sharpe = (

            strat.mean()

            /

            (std_ret + 1e-9)

        ) * np.sqrt(252)

        # ==========================================
        # CLIP EXTREMES
        # ==========================================
        sharpe = np.clip(
            sharpe,
            -5,
            5
        )

        # ==========================================
        # SMOOTH SHARPE
        # ==========================================
        sharpe = np.tanh(
            sharpe / 2
        )

        # ==========================================
        # RECENCY BONUS
        # ==========================================
        recent_mean = strat.tail(20).mean()

        recent_bonus = (
            np.tanh(recent_mean * 100)
            * 0.15
        )

        # ==========================================
        # FINAL SCORE
        # ==========================================
        final_score = (
            sharpe
            +
            recent_bonus
        )

        recent_scores[name] = final_score
    
    # =========================
    # FALLBACK IF EMPTY
    # =========================
    if len(recent_scores) == 0:
        models = list(probas.keys())[:top_k]
        weights = np.ones(len(models)) / len(models)
        return models, weights, {}

    # =========================
    # 2. GLOBAL RANKING
    # =========================
    ranked_all = sorted(
        recent_scores,
        key=recent_scores.get,
        reverse=True
    )

    # =========================
    # 3. REGIME CANDIDATES
    # =========================
    regime_candidates = get_regime_candidates(
        current_regime,
        list(probas.keys())
    )

    if current_regime == "BEAR_VOLATILE":
        top_k = 2

    regime_candidates = [m for m in regime_candidates if m in recent_scores]

    # =========================
    # 4. MERGE (ADAPTIVE BLEND)
    # =========================
    top_global_n = max(2, int(len(ranked_all) * 0.4))

    candidates = []
    for m in regime_candidates + ranked_all[:top_global_n]:
        if m not in candidates:
            candidates.append(m)

    # =========================
    # 5. RANK NORMALIZATION
    # =========================
    score_series = pd.Series(recent_scores)

    # convert to percentile rank (robust)
    score_rank = score_series.rank(pct=True)

    # =========================
    # 6. FINAL RANKING
    # =========================
    ranked = sorted(
        candidates,
        key=lambda x: score_rank.get(x, 0),
        reverse=True
    )

    # =========================
    # 7. DYNAMIC FILTER (NO HARD THRESHOLD)
    # =========================
    if len(ranked) > 1:

        scores_array = np.array([score_rank[m] for m in ranked])

        # adaptive cutoff based on distribution
        cutoff = np.mean(scores_array) - 0.5 * np.std(scores_array)

        ranked = [m for m in ranked if score_rank[m] >= cutoff]

    # =========================
    # 8. SELECT TOP MODELS
    # =========================
    top_models = ranked[:top_k]

    if len(top_models) < top_k:
        fallback = ranked_all[:top_k]
        top_models = fallback

    # =========================
    # 9. WEIGHT GENERATION (SMOOTH + GENERIC)
    # =========================
    scores = np.array([
        score_rank.get(m, 0)
        for m in top_models
    ])

    # 🔥 Non-linear smoothing (important)
    scores = np.power(scores, 1.5)

    if scores.sum() == 0:
        weights = np.ones(len(top_models)) / len(top_models)
    else:
        weights = scores / scores.sum()

    # =========================
    # 10. ADAPTIVE CAPPING
    # =========================
    if len(weights) > 1:

        cap_min = 0.1 / len(weights)
        cap_max = 0.70

        weights = np.clip(weights, cap_min, cap_max)
        weights = weights / weights.sum()

    # =========================
    # 11. DOMINANCE CHECK (ADAPTIVE)
    # =========================
    if len(top_models) > 1:

        # score_vals = np.array([recent_scores[m] for m in top_models])
        # dominance_ratio = score_vals.max() / (np.mean(score_vals) + 1e-9)

        weights = np.sqrt(weights)
        weights = weights / weights.sum()

    # =========================
    # DEBUG
    # =========================
    print("\n SMART MODEL SELECTION")
    print("Regime:", current_regime)
    print("Regime Candidates:", regime_candidates)
    print("Merged Candidates:", candidates)
    print("Top Models:", top_models)
    print("Scores:", recent_scores)
    print("Weights:", dict(zip(top_models, weights)))

    return top_models, weights, recent_scores

