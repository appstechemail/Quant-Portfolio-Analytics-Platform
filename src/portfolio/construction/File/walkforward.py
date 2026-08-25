# ==========================================================
# WALK-FORWARD VALIDATION ENGINE (DATE-BASED VERSION)
# ==========================================================
#
# PURPOSE
# -------
# This module performs institutional-grade walk-forward
# validation for cross-sectional stock prediction systems.
#
# WHY THIS IS IMPORTANT
# ---------------------
# Traditional train/test split is NOT sufficient for:
#
# - Multi-stock ML systems
# - Cross-sectional alpha models
# - Regime-adaptive strategies
# - Financial time-series forecasting
#
# Walk-forward validation simulates REAL trading by:
#
# 1. Training on past data
# 2. Testing on future unseen data
# 3. Rolling forward through time
#
# This prevents:
#
# ❌ Lookahead bias
# ❌ Temporal leakage
# ❌ Overfitting to one market regime
# ❌ Unrealistic backtest performance
#
# IMPORTANT DESIGN
# ----------------
# This implementation is:
#
# ✅ DATE-BASED
# ✅ Cross-sectional safe
# ✅ Multi-stock compatible
# ✅ Expanding-window capable
#
# Folds are created using UNIQUE DATES,
# NOT raw dataframe rows.
#
# This is CRITICAL for panel data.
#
# ==========================================================

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss
    )

from config.config import CONFIG

from src.models.models import train_models
from src.backtest.backtest import run_backtest

from src.models.model_utils import (
            get_model_probabilities,
            combine_probabilities,
            compute_ensemble_confidence
        )

from src.evaluation.metrics import (
    compute_alpha_metrics,
    compute_icir
)

from src.evaluation.probability_diagnostics import (
    compute_probability_diagnostics
)

# ==========================================================
# MAIN WALK-FORWARD FUNCTION
# ==========================================================
def run_walkforward_validation(
    final_df,
    feature_cols,
    target_col="Alpha_Target"
):

    print("\n🚀 STARTING WALK-FORWARD VALIDATION")

    # ======================================================
    # CONFIG
    # ======================================================
    wf_cfg = CONFIG.get("WALKFORWARD", {})

    train_window = wf_cfg.get("TRAIN_WINDOW", 504)
    test_window = wf_cfg.get("TEST_WINDOW", 63)
    step_size = wf_cfg.get("STEP_SIZE", 63)

    rolling_window = wf_cfg.get(
        "EXPANDING_WINDOW",
        True
    )

    threshold = CONFIG["MODEL"]["THRESHOLD"]

    # ======================================================
    # PREPARE DATA
    # ======================================================
    data = final_df.copy()

    required_cols = (
        [
            "Date",
            "Company",
            target_col,
            "Market_Regime",
            "Future_Return"
        ]
        + feature_cols
    )

    data = data[required_cols].copy()

    data["Date"] = pd.to_datetime(data["Date"])

    # Remove bad rows
    data = data.dropna(
        subset=feature_cols + [target_col]
    )

    # Sort correctly
    data = data.sort_values(
        ["Date", "Company"]
    ).reset_index(drop=True)

    # ======================================================
    # UNIQUE DATES (CRITICAL FIX)
    # ======================================================
    unique_dates = sorted(
        data["Date"].unique()
    )

    n_dates = len(unique_dates)

    print(f"\n📅 Total unique trading dates: {n_dates}")

    if n_dates < (train_window + test_window):
        print("❌ Not enough dates for walk-forward")
        return None

    # ======================================================
    # STORAGE
    # ======================================================
    fold_results = []

    fold_num = 1

    # ======================================================
    # WALK-FORWARD LOOP
    # ======================================================
    start_idx = train_window

    while start_idx + test_window <= n_dates:

        print(f"\n📊 WALK-FORWARD FOLD {fold_num}")

        # ==================================================
        # DATE WINDOWS
        # ==================================================

        if rolling_window:

            # ==========================================
            # ROLLING TRAIN WINDOW
            # Use only last 2 years
            # ==========================================

            ROLLING_TRAIN_YEARS = 2

            approx_days = 252 * ROLLING_TRAIN_YEARS

            train_start = max(
                0,
                start_idx - approx_days
            )

            train_dates = unique_dates[
                train_start:start_idx
            ]

        else:

            # ==========================================
            # FIXED WINDOW
            # ==========================================

            train_dates = unique_dates[
                start_idx - train_window:start_idx
            ]

        # ==========================================
        # TEST WINDOW
        # ==========================================

        test_dates = unique_dates[
            start_idx:start_idx + test_window
        ]
        # ==================================================
        # SPLIT USING DATES
        # ==================================================
        train_data = data[
            data["Date"].isin(train_dates)
        ].copy()

        test_data = data[
            data["Date"].isin(test_dates)
        ].copy()

        # ==================================================
        # SAFETY CHECKS
        # ==================================================
        if len(train_data) == 0 or len(test_data) == 0:

            print("⚠️ Empty fold → skipping")

            start_idx += step_size
            continue

        # ==================================================
        # FEATURES / TARGET
        # ==================================================
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]

        X_test = test_data[feature_cols]
        y_test = test_data[target_col]


        meta_test = test_data[
            ["Date","Company"]
        ].copy()


        # ==================================================
        # SCALING
        # ==================================================
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ==================================================
        # TRAIN MODELS
        # ==================================================
        selected_models = CONFIG["MODEL"].get(
            "MODEL_LIST",
            None
        )

        models = train_models(
            X_train=X_train,
            y_train=y_train,
            X_train_scaled=X_train_scaled,
            selected_models=selected_models
        )

        if not models:

            print("⚠️ No models trained")

            start_idx += step_size
            continue

        # ==================================================
        # PROBABILITIES
        # ==================================================
        probas = get_model_probabilities(
            models,
            X_test,
            X_test_scaled
        )

        if not probas:

            print("⚠️ No probabilities generated")

            start_idx += step_size
            continue

        # ==================================================
        # ENSEMBLE
        # ==================================================
        signals = []

        for model_name, proba in probas.items():

            try:

                if proba is None:
                    continue

                p = np.array(proba).flatten()

                if len(p) != len(X_test):
                    continue

                signals.append(p)

            except Exception as e:

                print(f"⚠️ {model_name} failed: {e}")

        if len(signals) == 0:

            print("⚠️ No ensemble signals")

            start_idx += step_size
            continue

        # ==================================================
        # ENSEMBLE
        # ==================================================

        ensemble_proba = combine_probabilities(
            probas
        )



        ensemble_confidence = (
            compute_ensemble_confidence(
                probas
            )
        )


        prob_diag = compute_probability_diagnostics(
            probabilities=ensemble_proba,
            confidence=ensemble_confidence,
            dates=test_data["Date"]
        )

        meta_test["Confidence"] = (
            ensemble_confidence
        )

        # ==================================================
        # PROBABILITY-BASED CONVICTION SCALING
        # ==================================================

        meta_test["Conviction_Multiplier"] = np.clip(
            0.5 + 1.2 * (ensemble_proba - 0.50),
            0.50,
            1.30
        )

        # meta_test["Confidence"] = ensemble_confidence

        meta_test["Adjusted_Confidence"] = (
            meta_test["Confidence"]
            *
            meta_test["Conviction_Multiplier"]
        )

        signal_for_ic = (
            ensemble_proba
            * meta_test["Confidence"]
        )

        print(test_data.columns.tolist())

        alpha_metrics = compute_alpha_metrics(
            predictions=signal_for_ic,
            future_returns=test_data["Future_Return"]
        )


        spearman_ic = (
            alpha_metrics["Spearman_IC"]
        )

        pearson_ic = (
            alpha_metrics["Pearson_IC"]
        )

        rank_ic = (
            alpha_metrics["Rank_IC"]
        )

        print("\nENSEMBLE DIAGNOSTICS")

        print(
            pd.Series(ensemble_proba)
            .describe()
        )

        print(
            "Average Ensemble Confidence:",
            round(
                ensemble_confidence.mean(),
                3
            )
        )

        print(
            pd.Series(ensemble_confidence)
            .describe()
        )
        # ==================================================
        # BACKTEST
        # ==================================================
        bt = run_backtest(
            ensemble_proba,
            X_test,
            meta_test,
            final_df
        )


        if (
            bt is None
            or not isinstance(bt, dict)
            or len(bt) == 0
        ):

            print("⚠️ Empty backtest")

            start_idx += step_size
            continue

        # ==================================================
        # BACKTEST DATAFRAME
        # ==================================================
        bt_df = bt["Backtest_DF"]


        # ==================================================
        # ACCURACY
        # ==================================================
        preds = (
            ensemble_proba > threshold
        ).astype(int)

        acc = accuracy_score(
            y_test,
            preds
        )

        # ==================================================
        # METRICS
        # ==================================================

        sharpe = bt["Sharpe"]

        cagr = bt["CAGR"]

        max_dd = bt["Max_Drawdown"]

        final_return = bt["Final_Return"]

        # ==================================================
        # REGIME DISTRIBUTION FOR THIS FOLD
        # ==================================================

        print(test_data.columns.tolist())

        if "Market_Regime" in test_data.columns:

            regime_pct = (
                test_data["Market_Regime"]
                .value_counts(normalize=True)
            )

            pct_bear = regime_pct.get(
                "BEAR",
                0
            )

            pct_bear_volatile = regime_pct.get(
                "BEAR_VOLATILE",
                0
            )

            pct_sideways = regime_pct.get(
                "SIDEWAYS",
                0
            )

            pct_sideways_volatile = regime_pct.get(
                "SIDEWAYS_VOLATILE",
                0
            )

            pct_bull = regime_pct.get(
                "BULL",
                0
            )

            pct_bull_volatile = regime_pct.get(
                "BULL_VOLATILE",
                0
            )

            overall_avg_position = (
                bt_df["Position"]
                .mean()
            )

            # ========================================
            # REGIME PNL DIAGNOSTICS
            # ========================================

            avgret_by_regime = (
                bt_df.groupby("Market_Regime")["Strategy_Return"]
                .mean()
            )

            sumpnl_by_regime = (
                bt_df.groupby("Market_Regime")["Gross_PnL"]
                .sum()
            )

            # =========================================


            avgret_bear = avgret_by_regime.get("BEAR", 0)
            avgret_bear_vol = avgret_by_regime.get("BEAR_VOLATILE", 0)

            avgret_sideways = avgret_by_regime.get("SIDEWAYS", 0)
            avgret_sideways_vol = avgret_by_regime.get("SIDEWAYS_VOLATILE", 0)

            avgret_bull = avgret_by_regime.get("BULL", 0)
            avgret_bull_vol = avgret_by_regime.get("BULL_VOLATILE", 0)


            sumpnl_bear = sumpnl_by_regime.get("BEAR", 0)
            sumpnl_bear_vol = sumpnl_by_regime.get("BEAR_VOLATILE", 0)

            sumpnl_sideways = sumpnl_by_regime.get("SIDEWAYS", 0)
            sumpnl_sideways_vol = sumpnl_by_regime.get("SIDEWAYS_VOLATILE", 0)

            sumpnl_bull = sumpnl_by_regime.get("BULL", 0)
            sumpnl_bull_vol = sumpnl_by_regime.get("BULL_VOLATILE", 0)


            regime_pnl_efficiency = (
                sumpnl_bull_vol
                + sumpnl_bull
                - abs(sumpnl_bear)
                - abs(sumpnl_bear_vol)
            )

            bullvol_pnl_share = (
                sumpnl_bull_vol /
                (
                    abs(sumpnl_bear)
                    + abs(sumpnl_bear_vol)
                    + abs(sumpnl_sideways)
                    + abs(sumpnl_sideways_vol)
                    + abs(sumpnl_bull)
                    + abs(sumpnl_bull_vol)
                    + 1e-9
                )
            )


            # --------------------------------------------------
            # REGIME-NORMALIZED METRICS
            # --------------------------------------------------

            bullvol_pnl_per_day = (
                sumpnl_bull_vol / pct_bull_volatile
                if pct_bull_volatile > 0
                else 0
            )

            bull_pnl_per_day = (
                sumpnl_bull / pct_bull
                if pct_bull > 0
                else 0
            )

            bear_pnl_per_day = (
                sumpnl_bear / pct_bear
                if pct_bear > 0
                else 0
            )

            sideways_pnl_per_day = (
                sumpnl_sideways / pct_sideways
                if pct_sideways > 0
                else 0
            )



            avgpos_by_regime = (
                bt_df.groupby("Market_Regime")["Position"]
                .mean()
            )

            position_weighted_return = (
                bt_df["Strategy_Return"].sum()
                /
                (bt_df["Position"].abs().sum() + 1e-9)
            )

            avgpos_bear = avgpos_by_regime.get(
                "BEAR",
                0
            )

            avgpos_bear_volatile = avgpos_by_regime.get(
                "BEAR_VOLATILE",
                0
            )

            avgpos_sideways = avgpos_by_regime.get(
                "SIDEWAYS",
                0
            )

            avgpos_sideways_volatile = avgpos_by_regime.get(
                "SIDEWAYS_VOLATILE",
                0
            )

            avgpos_bull = avgpos_by_regime.get(
                "BULL",
                0
            )

            avgpos_bull_volatile = avgpos_by_regime.get(
                "BULL_VOLATILE",
                0
            )

            bull_vol_return_efficiency = avgret_bull_vol


            bull_exposure_alpha = (
                avgpos_bull
                - overall_avg_position
            )

            bull_vol_exposure_alpha = (
                avgpos_bull_volatile
                - overall_avg_position
            )

            bear_exposure_alpha = (
                avgpos_bear
                - overall_avg_position
            )

            bear_vol_exposure_alpha = (
                avgpos_bear_volatile
                - overall_avg_position
            )

            sideways_exposure_alpha = (
                avgpos_sideways
                - overall_avg_position
            )

            sideways_vol_exposure_alpha = (
                avgpos_sideways_volatile
                - overall_avg_position
            )

            fold_quality_score = (
                bull_vol_exposure_alpha
                - bear_exposure_alpha
            )

            # Safe Ratio Function ====================
            
            def safe_ratio(pos, pct):
                return pos / pct if pct > 0 else np.nan
            
            # ========================================
            
            
            exposure_efficiency_BULL = safe_ratio(
                avgpos_bull,
                pct_bull
            )

            exposure_efficiency_BEAR = safe_ratio(
                avgpos_bear,
                pct_bear
            )

            exposure_efficiency_SIDEWAYS = safe_ratio(
                avgpos_sideways,
                pct_sideways
            )

            exposure_efficiency_BULL_VOL = safe_ratio(
                avgpos_bull_volatile,
                pct_bull_volatile
            )

            exposure_efficiency_BEAR_VOL = safe_ratio(
                avgpos_bear_volatile,
                pct_bear_volatile
            )

            exposure_efficiency_SIDEWAYS_VOL = safe_ratio(
                avgpos_sideways_volatile,
                pct_sideways_volatile
            )
            
        else:

            pct_bear = 0
            pct_bear_volatile = 0
            pct_sideways = 0
            pct_sideways_volatile = 0
            pct_bull = 0
            pct_bull_volatile = 0

            avgpos_bear = 0
            avgpos_bear_volatile = 0
            avgpos_sideways = 0
            avgpos_sideways_volatile = 0
            avgpos_bull = 0
            avgpos_bull_volatile = 0

            exposure_efficiency_BULL = 0
            exposure_efficiency_BEAR = 0
            exposure_efficiency_SIDEWAYS = 0
            exposure_efficiency_BULL_VOL = 0
            exposure_efficiency_BEAR_VOL = 0
            exposure_efficiency_SIDEWAYS_VOL = 0

            avgret_bear = 0
            avgret_bear_vol = 0

            avgret_sideways = 0
            avgret_sideways_vol = 0

            avgret_bull = 0
            avgret_bull_vol = 0

            sumpnl_bear = 0
            sumpnl_bear_vol = 0

            sumpnl_sideways = 0
            sumpnl_sideways_vol = 0

            sumpnl_bull = 0
            sumpnl_bull_vol = 0

            bullvol_pnl_per_day = 0
            bull_pnl_per_day = 0
            bear_pnl_per_day = 0
            sideways_pnl_per_day = 0

            regime_pnl_efficiency = 0
            bull_vol_return_efficiency = 0
            bullvol_pnl_share = 0
            fold_quality_score = 0

            overall_avg_position = 0

            bull_exposure_alpha = 0
            bull_vol_exposure_alpha = 0

            bear_exposure_alpha = 0
            bear_vol_exposure_alpha = 0

            sideways_exposure_alpha = 0
            sideways_vol_exposure_alpha = 0
            position_weighted_return = 0


        # --------------------------------------------------
        # Adjusted Confidence
        #
        # This is intentionally calculated here because
        # Adjusted_Confidence is created in walkforward.py
        # after the probability diagnostics are computed.
        # --------------------------------------------------

        adjusted_confidence_mean = float(
            meta_test["Adjusted_Confidence"].mean()
        )

        adjusted_confidence_std = float(
            meta_test["Adjusted_Confidence"].std()
        )


        # ==================================================
        # PROBABILITY CALIBRATION
        # ==================================================

        if (
            "Proba" in bt_df.columns
            and len(np.unique(y_test)) > 1
        ):

            proba = np.clip(
                bt_df["Proba"].values,
                1e-8,
                1 - 1e-8
            )

            truth = y_test.values.astype(int)

            # ------------------------------------------
            # Brier Score
            # Lower is better
            # ------------------------------------------

            brier = brier_score_loss(
                truth,
                proba
            )

            # ------------------------------------------
            # Log Loss
            # Lower is better
            # ------------------------------------------

            logloss = log_loss(
                truth,
                proba
            )

            # ------------------------------------------
            # Reliability Diagram Statistics
            # ------------------------------------------

            n_bins = 10

            bins = np.linspace(
                0,
                1,
                n_bins + 1
            )

            bin_ids = np.digitize(
                proba,
                bins
            ) - 1

            ece = 0.0
            mce = 0.0

            observed_accuracy = []
            predicted_confidence = []

            total = len(proba)

            for b in range(n_bins):

                mask = bin_ids == b

                if mask.sum() == 0:
                    continue

                acc_bin = truth[mask].mean()

                conf_bin = proba[mask].mean()

                gap = abs(
                    acc_bin -
                    conf_bin
                )

                ece += (
                    gap *
                    mask.sum() /
                    total
                )

                mce = max(
                    mce,
                    gap
                )

                observed_accuracy.append(
                    acc_bin
                )

                predicted_confidence.append(
                    conf_bin
                )

            observed_accuracy = np.asarray(
                observed_accuracy
            )

            predicted_confidence = np.asarray(
                predicted_confidence
            )

            reliability = np.mean(
                np.abs(
                    observed_accuracy -
                    predicted_confidence
                )
            )

            avg_confidence = proba.mean()

            avg_accuracy = (
                (proba >= 0.5)
                ==
                truth
            ).mean()

            calibration_gap = (
                avg_confidence -
                avg_accuracy
            )

            overconfidence = max(
                calibration_gap,
                0
            )

            underconfidence = max(
                -calibration_gap,
                0
            )

        else:

            brier = np.nan
            logloss = np.nan

            ece = np.nan
            mce = np.nan

            reliability = np.nan

            avg_confidence = np.nan
            avg_accuracy = np.nan

            calibration_gap = np.nan

            overconfidence = np.nan
            underconfidence = np.nan
        # ==================================================
        # FINAL SCORE DIAGNOSTICS
        # ==================================================

        if "Final_Score" in bt_df.columns:
            avg_final_score = bt_df["Final_Score"].mean()
            min_final_score = bt_df["Final_Score"].min()
            max_final_score = bt_df["Final_Score"].max()
        else:
            avg_final_score = np.nan
            min_final_score = np.nan
            max_final_score = np.nan

        # ==================================================
        # STORE FOLD RESULTS
        # ==================================================

        fold_results.append({

            # --------------------------------------------------
            # Fold Information
            # --------------------------------------------------

            "Fold": fold_num,

            "Train_Start": pd.to_datetime(train_dates[0]),
            "Train_End": pd.to_datetime(train_dates[-1]),

            "Test_Start": pd.to_datetime(test_dates[0]),
            "Test_End": pd.to_datetime(test_dates[-1]),

            "Train_Days": len(train_dates),
            "Test_Days": len(test_dates),

            "Train_Samples": len(train_data),
            "Test_Samples": len(test_data),

            # --------------------------------------------------
            # Portfolio Metrics
            # --------------------------------------------------

            "Sharpe": sharpe,
            "CAGR": cagr,
            "Max_Drawdown": max_dd,
            "Volatility": bt["Volatility"],
            "Accuracy": acc,

            "Final_Return": final_return,

            "Win_Rate": bt["Win_Rate"],
            "Active_Days": bt["Active_Days"],

            "Avg_Turnover": bt["Avg_Turnover"],
            "Median_Turnover": bt["Median_Turnover"],
            "Turnover95": bt["Turnover95"],
            "Max_Turnover": bt["Max_Turnover"],

            "Avg_Holdings": bt["Avg_Holdings"],
            "Deadband_Pct": bt["Deadband_Pct"],

            # --------------------------------------------------
            # Final Score Diagnostics
            # --------------------------------------------------

            "Avg_Final_Score": avg_final_score,
            "Min_Final_Score": min_final_score,
            "Max_Final_Score": max_final_score,

            # --------------------------------------------------
            # Overall Exposure
            # --------------------------------------------------

            "Overall_Avg_Position": overall_avg_position,

            # --------------------------------------------------
            # Market Regime Distribution
            # --------------------------------------------------

            "Pct_BEAR": pct_bear,
            "Pct_BEAR_VOLATILE": pct_bear_volatile,

            "Pct_SIDEWAYS": pct_sideways,
            "Pct_SIDEWAYS_VOLATILE": pct_sideways_volatile,

            "Pct_BULL": pct_bull,
            "Pct_BULL_VOLATILE": pct_bull_volatile,

            # --------------------------------------------------
            # Average Position by Regime
            # --------------------------------------------------

            "AvgPos_BEAR": avgpos_bear,
            "AvgPos_BEAR_VOLATILE": avgpos_bear_volatile,

            "AvgPos_SIDEWAYS": avgpos_sideways,
            "AvgPos_SIDEWAYS_VOLATILE": avgpos_sideways_volatile,

            "AvgPos_BULL": avgpos_bull,
            "AvgPos_BULL_VOLATILE": avgpos_bull_volatile,

            # --------------------------------------------------
            # Regime Exposure Diagnostics
            # --------------------------------------------------

            "Bull_Exposure_Alpha": bull_exposure_alpha,
            "BullVol_Exposure_Alpha": bull_vol_exposure_alpha,

            "Bear_Exposure_Alpha": bear_exposure_alpha,
            "BearVol_Exposure_Alpha": bear_vol_exposure_alpha,

            "Sideways_Exposure_Alpha": sideways_exposure_alpha,
            "SidewaysVol_Exposure_Alpha": sideways_vol_exposure_alpha,

            "Exposure_Efficiency_BULL": exposure_efficiency_BULL,
            "Exposure_Efficiency_BEAR": exposure_efficiency_BEAR,

            "Exposure_Efficiency_SIDEWAYS":
                exposure_efficiency_SIDEWAYS,

            "Exposure_Efficiency_BULL_VOL":
                exposure_efficiency_BULL_VOL,

            "Exposure_Efficiency_BEAR_VOL":
                exposure_efficiency_BEAR_VOL,

            "Exposure_Efficiency_SIDEWAYS_VOL":
                exposure_efficiency_SIDEWAYS_VOL,

            # --------------------------------------------------
            # Regime Quality Metrics
            # --------------------------------------------------

            "Fold_Quality_Score": fold_quality_score,

            "Regime_PnL_Efficiency":
                regime_pnl_efficiency,

            "BullVol_Pnl_Share":
                bullvol_pnl_share,

            "Bull_Vol_Return_Efficiency":
                bull_vol_return_efficiency,

            # --------------------------------------------------
            # Regime Return Metrics
            # --------------------------------------------------

            "AvgRet_BEAR":
                avgret_bear,

            "AvgRet_BEAR_VOLATILE":
                avgret_bear_vol,

            "AvgRet_SIDEWAYS":
                avgret_sideways,

            "AvgRet_SIDEWAYS_VOLATILE":
                avgret_sideways_vol,

            "AvgRet_BULL":
                avgret_bull,

            "AvgRet_BULL_VOLATILE":
                avgret_bull_vol,

            # --------------------------------------------------
            # Regime PnL Metrics
            # --------------------------------------------------

            "SumPnL_BEAR":
                sumpnl_bear,

            "SumPnL_BEAR_VOLATILE":
                sumpnl_bear_vol,

            "SumPnL_SIDEWAYS":
                sumpnl_sideways,

            "SumPnL_SIDEWAYS_VOLATILE":
                sumpnl_sideways_vol,

            "SumPnL_BULL":
                sumpnl_bull,

            "SumPnL_BULL_VOLATILE":
                sumpnl_bull_vol,

            # --------------------------------------------------
            # Regime PnL Normalization
            # --------------------------------------------------

            "BullVol_PnL_Per_Day":
                bullvol_pnl_per_day,

            "Bull_PnL_Per_Day":
                bull_pnl_per_day,

            "Bear_PnL_Per_Day":
                bear_pnl_per_day,

            "Sideways_PnL_Per_Day":
                sideways_pnl_per_day,

            # --------------------------------------------------
            # Exposure Efficiency
            # --------------------------------------------------

            "Return_Per_Unit_Exposure":
                position_weighted_return,

            # --------------------------------------------------
            # IC Metrics
            # --------------------------------------------------

            "Spearman_IC":
                spearman_ic,

            "Pearson_IC":
                pearson_ic,

            "Rank_IC":
                rank_ic,

            # --------------------------------------------------
            # Probability Diagnostics
            # --------------------------------------------------

            "Avg_Probability":
                prob_diag["Avg_Probability"],

            "Std_Probability":
                prob_diag["Std_Probability"],

            "Median_Probability":
                prob_diag["Median_Probability"],

            "Min_Probability":
                prob_diag["Min_Probability"],

            "Max_Probability":
                prob_diag["Max_Probability"],

            "Probability_IQR":
                prob_diag["Probability_IQR"],

            "Probability_Spread_90_10":
                prob_diag["Probability_Spread_90_10"],

            "Probability_Spread_95_05":
                prob_diag["Probability_Spread_95_05"],

            "Probability_Entropy":
                prob_diag["Probability_Entropy"],

            "Probability_Skewness":
                prob_diag["Probability_Skewness"],

            "Probability_Kurtosis":
                prob_diag["Probability_Kurtosis"],

            "CrossSectional_Dispersion":
                prob_diag["CrossSectional_Dispersion"],

            # --------------------------------------------------
            # Confidence Diagnostics
            # --------------------------------------------------

            "Confidence_Mean":
                prob_diag["Confidence_Mean"],

            "Confidence_Std":
                prob_diag["Confidence_Std"],

            "Confidence_CV":
                prob_diag["Confidence_CV"],

            # --------------------------------------------------
            # Adjusted Confidence
            # --------------------------------------------------

            "Mean_Adjusted_Confidence":
                adjusted_confidence_mean,

            "Std_Adjusted_Confidence":
                adjusted_confidence_std,

            # --------------------------------------------------
            # Calibration Diagnostics
            # --------------------------------------------------

            "Brier_Score":
                brier,

            "Log_Loss":
                logloss,

            "Expected_Calibration_Error":
                ece,

            "Reliability_Error":
                reliability,

            "Maximum_Calibration_Error":
                mce,

            "Calibration_Gap":
                calibration_gap,

            "Overconfidence":
                overconfidence,

            "Underconfidence":
                underconfidence,

            # --------------------------------------------------
            # Accuracy / Confidence Calibration
            # --------------------------------------------------

            "Average_Confidence":
                avg_confidence,

            "Average_Accuracy":
                avg_accuracy,
        })


        # ==================================================
        # LOGGING
        # ==================================================
        print(
            f"✅ Fold {fold_num} | "
            f"TrainDays={len(train_dates)} | "
            f"TestDays={len(test_dates)} | "
            f"Sharpe={sharpe:.3f} | "
            f"CAGR={cagr:.3f} | "
            f"DD={max_dd:.3f} | "
            f"Acc={acc:.3f}"
        )

        # ==================================================
        # NEXT FOLD
        # ==================================================
        fold_num += 1
        start_idx += step_size

    # ======================================================
    # FINAL SUMMARY
    # ======================================================

    if len(fold_results) == 0:

        print("❌ No valid folds")

        return None


    # ======================================================
    # BUILD WALK-FORWARD SUMMARY
    # ======================================================

    wf_summary = pd.DataFrame(fold_results)


    # ======================================================
    # MODEL IC SUMMARY
    # ======================================================

    mean_spearman_ic = (
        wf_summary["Spearman_IC"]
        .mean()
    )

    mean_pearson_ic = (
        wf_summary["Pearson_IC"]
        .mean()
    )

    mean_rank_ic = (
        wf_summary["Rank_IC"]
        .mean()
    )

    icir = compute_icir(
        wf_summary["Rank_IC"]
    )

    positive_ic_pct = (
        (
            wf_summary["Rank_IC"] > 0
        ).mean()
        * 100.0
    )


    print("\n" + "=" * 70)
    print("MODEL IC METRICS")
    print("=" * 70)

    print(
        f"Mean Spearman IC : {mean_spearman_ic:.4f}"
    )

    print(
        f"Mean Pearson IC  : {mean_pearson_ic:.4f}"
    )

    print(
        f"Mean Rank IC     : {mean_rank_ic:.4f}"
    )

    print(
        f"ICIR             : {icir:.4f}"
    )

    print(
        f"Positive IC %    : {positive_ic_pct:.2f}%"
    )


    # ======================================================
    # SELECT IMPORTANT FOLDS
    # ======================================================

    summary_cols = [

        "Fold",
        "Sharpe",
        "Final_Return",

        # Probability
        "Avg_Proba",
        "Median_Proba",
        "P90_Proba",
        "P95_Proba",
        "High_Confidence_Pct",
        "Probability_Spread",

        # Calibration
        "Brier_Score",
        "Log_Loss",
        "Expected_Calibration_Error",
        "Reliability_Error",
        "Maximum_Calibration_Error",
        "Calibration_Gap",

        # Regime
        "Regime_PnL_Efficiency",
        "BullVol_Pnl_Share",
        "Bull_Vol_Return_Efficiency",

        "Pct_BEAR",
        "Pct_BEAR_VOLATILE",
        "Pct_SIDEWAYS",
        "Pct_SIDEWAYS_VOLATILE",
        "Pct_BULL",
        "Pct_BULL_VOLATILE",

        # Score
        "Avg_Final_Score",
        "Min_Final_Score",
        "Max_Final_Score",

        # Exposure
        "Bull_Exposure_Alpha",
        "BullVol_Exposure_Alpha",
        "Bear_Exposure_Alpha",
        "BearVol_Exposure_Alpha",
        "Sideways_Exposure_Alpha",
        "SidewaysVol_Exposure_Alpha",

        # Portfolio
        "Avg_Holdings",
        "Avg_Turnover",
        "Deadband_Pct",

        # Return efficiency
        "Return_Per_Unit_Exposure",
    ]


    # Keep only columns actually available

    available_summary_cols = [
        col
        for col in summary_cols
        if col in wf_summary.columns
    ]


    # ======================================================
    # BEST / WORST FOLDS
    # ======================================================

    best_folds = (
        wf_summary
        .sort_values(
            "Sharpe",
            ascending=False
        )
        .head(3)["Fold"]
        .tolist()
    )

    worst_folds = (
        wf_summary
        .sort_values(
            "Sharpe",
            ascending=True
        )
        .head(3)["Fold"]
        .tolist()
    )

    interesting_folds = (
        best_folds
        +
        worst_folds
    )


    print("\n" + "=" * 70)
    print("BEST / WORST WALK-FORWARD FOLDS")
    print("=" * 70)

    print(
        wf_summary.loc[
            wf_summary["Fold"].isin(
                interesting_folds
            ),
            available_summary_cols
        ]
        .sort_values("Fold")
    )


    # ======================================================
    # CORRELATION TO SHARPE
    # ======================================================

    print("\n" + "=" * 70)
    print("CORRELATION TO SHARPE")
    print("=" * 70)


    correlation_cols = [

        # Performance / portfolio
        "Sharpe",
        "Avg_Turnover",
        "Avg_Holdings",
        "Deadband_Pct",

        # Final score
        "Avg_Final_Score",
        "Min_Final_Score",
        "Max_Final_Score",

        # Probability calibration
        "Brier_Score",
        "Log_Loss",
        "Expected_Calibration_Error",
        "Maximum_Calibration_Error",
        "Reliability_Error",
        "Calibration_Gap",
        "Overconfidence",
        "Underconfidence",

        # Confidence
        "Ensemble_Confidence_Mean",
        "Ensemble_Confidence_Std",
        "Confidence_CV",
        "Mean_Adjusted_Confidence",
        "Std_Adjusted_Confidence",

        # Probability
        "Avg_Proba",
        "Std_Proba",
        "Median_Proba",
        "Min_Proba",
        "Max_Proba",
        "P05_Proba",
        "P10_Proba",
        "P25_Proba",
        "P75_Proba",
        "P90_Proba",
        "P95_Proba",
        "Probability_Range",
        "Probability_Spread",
        "High_Confidence_Pct",
        "Low_Confidence_Pct",
        "Neutral_Probability_Pct",
        "Probability_Entropy",

        # Market regime distribution
        "Pct_BEAR",
        "Pct_BEAR_VOLATILE",
        "Pct_SIDEWAYS",
        "Pct_SIDEWAYS_VOLATILE",
        "Pct_BULL",
        "Pct_BULL_VOLATILE",

        # Regime exposure
        "AvgPos_BEAR",
        "AvgPos_BEAR_VOLATILE",
        "AvgPos_SIDEWAYS",
        "AvgPos_SIDEWAYS_VOLATILE",
        "AvgPos_BULL",
        "AvgPos_BULL_VOLATILE",

        "Overall_Avg_Position",

        "Bull_Exposure_Alpha",
        "BullVol_Exposure_Alpha",

        "Bear_Exposure_Alpha",
        "BearVol_Exposure_Alpha",

        "Sideways_Exposure_Alpha",
        "SidewaysVol_Exposure_Alpha",

        "Fold_Quality_Score",
        "Regime_PnL_Efficiency",
        "BullVol_Pnl_Share",
        "Bull_Vol_Return_Efficiency",

        # Exposure efficiency
        "Exposure_Efficiency_BULL",
        "Exposure_Efficiency_BEAR",
        "Exposure_Efficiency_SIDEWAYS",
        "Exposure_Efficiency_BULL_VOL",
        "Exposure_Efficiency_BEAR_VOL",
        "Exposure_Efficiency_SIDEWAYS_VOL",

        # Regime returns
        "AvgRet_BEAR",
        "AvgRet_BEAR_VOLATILE",
        "AvgRet_SIDEWAYS",
        "AvgRet_SIDEWAYS_VOLATILE",
        "AvgRet_BULL",
        "AvgRet_BULL_VOLATILE",

        # Regime PnL
        "SumPnL_BEAR",
        "SumPnL_BEAR_VOLATILE",
        "SumPnL_SIDEWAYS",
        "SumPnL_SIDEWAYS_VOLATILE",
        "SumPnL_BULL",
        "SumPnL_BULL_VOLATILE",

        # Regime normalized PnL
        "BullVol_PnL_Per_Day",
        "Bull_PnL_Per_Day",
        "Bear_PnL_Per_Day",
        "Sideways_PnL_Per_Day",

        # Return efficiency
        "Return_Per_Unit_Exposure",
    ]


    # Keep only columns that exist
    available_corr_cols = [
        col
        for col in correlation_cols
        if col in wf_summary.columns
    ]


    # Need at least Sharpe + one diagnostic
    if (
        "Sharpe" in available_corr_cols
        and len(available_corr_cols) > 1
    ):

        sharpe_corr = (
            wf_summary[
                available_corr_cols
            ]
            .corr()["Sharpe"]
            .sort_values(
                ascending=False
            )
        )

    else:

        sharpe_corr = pd.Series(
            dtype=float
        )


    print("\nSHARPE CORRELATIONS")

    print(
        sharpe_corr
    )


    # ======================================================
    # TOP REGIME / EXPOSURE DRIVERS
    # ======================================================

    print(
        "\nTOP REGIME EXPOSURE DRIVERS"
    )


    if not sharpe_corr.empty:

        exposure_mask = (
            sharpe_corr.index
            .str.contains(
                "Exposure|AvgPos|Efficiency",
                regex=True
            )
        )

        exposure_corr = (
            sharpe_corr[
                exposure_mask
            ]
        )

        print(
            exposure_corr
        )

    else:

        print(
            "⚠️ No correlation diagnostics available."
        )


    # ======================================================
    # SAVE SHARPE CORRELATIONS
    # ======================================================

    if not sharpe_corr.empty:

        sharpe_corr_to_save = (
            sharpe_corr
            .drop(
                labels=["Sharpe"],
                errors="ignore"
            )
        )

        sharpe_corr_to_save.to_csv(
            "data/sharpe_correlations.csv",
            header=["Correlation"]
        )

        print(
            "\n✅ Saved Sharpe correlations:"
        )

        print(
            "data/sharpe_correlations.csv"
        )


    # ======================================================
    # SAVE FOLD DIAGNOSTICS
    # ======================================================

    wf_summary.to_csv(
        "data/fold_diagnostics.csv",
        index=False
    )

    print(
        "\n✅ Saved fold diagnostics:"
    )

    print(
        "data/fold_diagnostics.csv"
    )


    # ======================================================
    # WALK-FORWARD SUMMARY
    # ======================================================

    print("\n" + "=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)

    print(
        wf_summary
    )


    # ======================================================
    # CORE PERFORMANCE SUMMARY
    # ======================================================

    core_summary_cols = [

        "Fold",
        "Sharpe",
        "CAGR",
        "Max_Drawdown",
        "Volatility",
        "Final_Return",
        "Avg_Turnover",
        "Avg_Holdings",
        "Deadband_Pct",
    ]


    available_core_cols = [
        col
        for col in core_summary_cols
        if col in wf_summary.columns
    ]


    print(
        wf_summary[
            available_core_cols
        ]
    )


    # ======================================================
    # WORST 5 FOLDS
    # ======================================================

    print("\n" + "=" * 70)
    print("🚨 WORST 5 FOLDS")
    print("=" * 70)


    worst_cols = [

        "Fold",
        "Sharpe",
        "CAGR",
        "Max_Drawdown",

        "Avg_Final_Score",

        "Pct_BEAR",
        "Pct_BEAR_VOLATILE",
        "Pct_SIDEWAYS",
        "Pct_SIDEWAYS_VOLATILE",
        "Pct_BULL",
        "Pct_BULL_VOLATILE",

        "Avg_Turnover",
        "Deadband_Pct",
        "Avg_Holdings",
        "Win_Rate",

        "Brier_Score",
        "Expected_Calibration_Error",

        "Probability_Spread",
    ]


    available_worst_cols = [
        col
        for col in worst_cols
        if col in wf_summary.columns
    ]


    worst = (
        wf_summary
        .sort_values(
            "Sharpe",
            ascending=True
        )
        .head(5)
    )


    print(
        worst[
            available_worst_cols
        ]
    )


    # ======================================================
    # BEST 5 FOLDS
    # ======================================================

    print("\n" + "=" * 70)
    print("🏆 BEST 5 FOLDS")
    print("=" * 70)


    best_cols = [

        "Fold",

        "Test_Start",
        "Test_End",

        "Sharpe",
        "CAGR",

        "Avg_Final_Score",

        "Pct_BEAR",
        "Pct_BEAR_VOLATILE",
        "Pct_SIDEWAYS",
        "Pct_SIDEWAYS_VOLATILE",
        "Pct_BULL",
        "Pct_BULL_VOLATILE",

        "Avg_Turnover",
        "Deadband_Pct",
        "Avg_Holdings",

        "Max_Drawdown",
        "Volatility",
        "Final_Return",

        "Brier_Score",
        "Expected_Calibration_Error",
        "Probability_Spread",
    ]


    available_best_cols = [
        col
        for col in best_cols
        if col in wf_summary.columns
    ]


    print(
        wf_summary[
            available_best_cols
        ]
        .sort_values(
            "Sharpe",
            ascending=False
        )
        .head(5)
    )


    # ======================================================
    # AGGREGATED PERFORMANCE
    # ======================================================

    print("\n" + "=" * 70)
    print("📈 AGGREGATED PERFORMANCE")
    print("=" * 70)


    print(
        f"Average Turnover       : "
        f"{wf_summary['Avg_Turnover'].mean():.3f}"
    )


    print(
        f"Average Holdings       : "
        f"{wf_summary['Avg_Holdings'].mean():.2f}"
    )


    print(
        f"Average Deadband       : "
        f"{wf_summary['Deadband_Pct'].mean():.3f}"
    )


    print(
        f"Average Final Score    : "
        f"{wf_summary['Avg_Final_Score'].mean():.3f}"
    )


    print(
        f"Average Sharpe         : "
        f"{wf_summary['Sharpe'].mean():.3f}"
    )


    print(
        f"Average CAGR           : "
        f"{wf_summary['CAGR'].mean():.3f}"
    )


    print(
        f"Average Accuracy       : "
        f"{wf_summary['Accuracy'].mean():.3f}"
    )


    print(
        f"Worst Drawdown         : "
        f"{wf_summary['Max_Drawdown'].min():.3f}"
    )


    print(
        f"Average Volatility     : "
        f"{wf_summary['Volatility'].mean():.3f}"
    )


    print(
        f"Average Rank IC        : "
        f"{wf_summary['Rank_IC'].mean():.4f}"
    )


    print(
        f"ICIR                   : "
        f"{icir:.4f}"
    )


    print(
        f"Positive IC %          : "
        f"{positive_ic_pct:.2f}%"
    )


    print(
        f"Total Folds            : "
        f"{len(wf_summary)}"
    )


    # ======================================================
    # FINAL RETURN
    # ======================================================

    return wf_summary