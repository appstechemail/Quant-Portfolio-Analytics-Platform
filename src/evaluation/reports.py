import pandas as pd
import numpy as np
from src.evaluation.strategy import evaluate_strategy
from src.evaluation.scoring import compute_model_score

# ==========================================
# BUILD SUMMARY
# ==========================================
def build_summary(results, accuracies=None):

    rows = []

    for model_name, df in results.items():

        if df is None or df.empty:
            continue

        metrics = evaluate_strategy(df)

        row = {
            "Model": model_name,
            **metrics
        }

        if accuracies and model_name in accuracies:
            row["% Accuracy"] = accuracies[model_name] * 100

        rows.append(row)

    summary = pd.DataFrame(rows)

    return summary



# ==========================================
# FINAL REPORT
# ==========================================
def generate_report(results, accuracies, final_df):

    print("\n📊 Generating evaluation report...")

    # =========================
    # 1. BUILD SUMMARY
    # =========================
    summary = build_summary(results, accuracies)

    if summary is None or summary.empty:
        print("❌ Empty summary → cannot generate report")
        return pd.DataFrame()

    # =========================
    # 2. ADD MODEL SCORE (GENERIC)
    # =========================
    summary = compute_model_score(summary, results, final_df)

    # Safety check
    if "Score" not in summary.columns:
        print("⚠️ Score column missing → fallback to Sharpe Ratio")
        summary["Score"] = summary.get("Sharpe Ratio", 0)

    # =========================
    # 3. CLEAN DATA (IMPORTANT)
    # =========================
    summary = summary.replace([np.inf, -np.inf], np.nan)
    summary = summary.fillna(0)

    # =========================
    # 4. SORT BY SCORE
    # =========================
    summary = summary.sort_values("Score", ascending=False)

    # =========================
    # 5. ROUNDING (DISPLAY ONLY)
    # =========================
    summary = summary.round(6)

    print("\n📊 FINAL MODEL PERFORMANCE SUMMARY -- INSIDE GENERATE")
    print(summary.to_string(index=False))

    # =========================
    # 6. BEST MODEL LOGIC
    # =========================
    best_model = None
    best_base_model = None

    if len(summary) > 0:
        best_model = summary.iloc[0]["Model"]

        base_models = summary[
            summary["Model"].str.upper() != "ENSEMBLE"
        ]

        if not base_models.empty:
            best_base_model = base_models.iloc[0]["Model"]

    # =========================
    # 7. CONFIDENCE GAP (🔥 NEW)
    # =========================
    if len(summary) > 1:
        top_score = summary.iloc[0]["Score"]
        second_score = summary.iloc[1]["Score"]

        gap = top_score - second_score

        print(f"\n📊 Confidence Gap: {gap:.4f}")

        if gap < 0.02:
            print("⚠️ Models are very close → no strong winner")
        elif gap > 0.15:
            print("🔥 Strong dominant model detected")

    # =========================
    # 8. PRINT FINAL RESULTS
    # =========================
    if best_model:
        print(f"\n🏆 Best Overall Model: {best_model}")

    if best_base_model:
        print(f"🏆 Best Base Model (Ex-ENSEMBLE): {best_base_model}")

    # =========================
    # 9. RETURN
    # =========================
    return summary
