# alpha/alpha_crowding_engine.py

"""
==============================================================================
ALPHA CROWDING ENGINE
==============================================================================

Purpose
-------
Institutional crowding detection.

Measures:

1. Signal Concentration
2. Portfolio HHI
3. Feature Crowding
4. Factor Crowding
5. Regime Crowding
6. Turnover Crowding
7. Capacity Stress
8. Alpha Uniqueness
9. Final Crowding Score

Outputs
-------

alpha/outputs/

    crowding_report.csv
    crowding_timeseries.csv
    alpha_uniqueness.csv

Institutional Usage
-------------------

    Portfolio Construction
    Risk Engine
    Capacity Engine
    Execution Engine

==============================================================================

Author:
Institutional Quant Platform
==============================================================================
"""

import os
import numpy as np
import pandas as pd

OUTPUT_DIR = "data/alpha/outputs"


from typing import Dict


class AlphaCrowdingEngine:

    def __init__(
        self,
        final_df: pd.DataFrame | None = None,
        portfolio: pd.DataFrame | None = None,
        adaptive_weights: pd.DataFrame | None = None,
        retention_scores: pd.DataFrame | None = None,
        capacity_scores: pd.DataFrame | None = None,
    ):

        self.final_df = (
            final_df.copy()
            if isinstance(final_df, pd.DataFrame)
            else pd.DataFrame()
        )

        self.portfolio = (
            portfolio.copy()
            if isinstance(portfolio, pd.DataFrame)
            else pd.DataFrame()
        )

        self.adaptive_weights = (
            adaptive_weights.copy()
            if isinstance(adaptive_weights, pd.DataFrame)
            else pd.DataFrame()
        )

        self.retention_scores = (
            retention_scores.copy()
            if isinstance(retention_scores, pd.DataFrame)
            else pd.DataFrame()
        )

        self.capacity_scores = (
            capacity_scores.copy()
            if isinstance(capacity_scores, pd.DataFrame)
            else pd.DataFrame()
        )

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    #################################################################
    # SIGNAL CONCENTRATION
    #################################################################

    def signal_concentration(self):

        if len(self.portfolio) == 0:
            return 1.0

        probs = self.portfolio["Probability"]

        probs = probs / probs.sum()

        top5 = probs.nlargest(min(5, len(probs))).sum()

        return top5

    #################################################################
    # PORTFOLIO HHI
    #################################################################

    def portfolio_hhi(self):

        if len(self.portfolio) == 0:
            return 1.0

        w = self.portfolio["Position"]

        w = w / w.sum()

        return np.sum(w ** 2)

    #################################################################
    # TURNOVER CROWDING
    #################################################################

    def turnover_score(self):

        if "Daily_Turnover" not in self.final_df.columns:
            return 0

        annual_turnover = (
            self.final_df["Daily_Turnover"]
            .mean()
            * 252
        )

        return annual_turnover

    #################################################################
    # FEATURE CROWDING
    #################################################################

    def feature_crowding(self):

        if "Feature" not in self.adaptive_weights.columns:
            return 0

        top = (
            self.adaptive_weights
            .sort_values(
                "Adaptive_Weight",
                ascending=False
            )
            .head(10)
        )

        trend_count = (
                top["Feature"]
                .str.contains(
                    "MA|EMA|Trend|Momentum",
                    case=False
                )
                .sum()
        )

        return trend_count / max(len(top), 1)

    #################################################################
    # FACTOR CROWDING
    #################################################################

    def factor_crowding(self):

        if (
            self.adaptive_weights.empty
            or
            "Adaptive_Weight" not in self.adaptive_weights.columns
            or
            "Feature" not in self.adaptive_weights.columns
        ):
            return 0

        factors = {
            "Value": ["EPS", "PE", "Value"],
            "Momentum": ["Momentum"],
            "Trend": ["Trend", "EMA", "MA"],
            "Volatility": ["ATR", "Volatility"],
            "Quality": ["Quality"],
            "Reversal": ["Reversal"]
        }

        weights = {}

        total = self.adaptive_weights[
            "Adaptive_Weight"
        ].sum()

        for f, terms in factors.items():

            mask = (
                self.adaptive_weights["Feature"]
                .str.contains(
                    "|".join(terms),
                    case=False,
                    na=False
                )
            )

            weights[f] = (

                self.adaptive_weights.loc[
                    mask,
                    "Adaptive_Weight"
                ].sum()

                / max(total, 1e-9)

            )

        max_factor = max(weights.values())

        return max_factor

    #################################################################
    # REGIME CROWDING
    #################################################################

    def regime_crowding(self):

        if "Market_Regime" not in self.portfolio.columns:

            return 0

        counts = (
            self.portfolio
            .Market_Regime
            .value_counts(normalize=True)
        )

        return counts.max()

    #################################################################
    # CAPACITY STRESS
    #################################################################

    def capacity_stress(self):

        if "Capacity_Score" not in \
                self.capacity_scores.columns:

            return 1

        return self.capacity_scores[
            "Capacity_Score"
        ].mean()

    #################################################################
    # ALPHA UNIQUENESS
    #################################################################

    def alpha_uniqueness(self):

        if "Retention_Score" not in \
                self.retention_scores.columns:

            retention = 1

        else:

            retention = self.retention_scores[
                "Retention_Score"
            ].mean()

        capacity = self.capacity_stress()

        crowding = self.final_crowding()

        uniqueness = (
                retention
                * capacity
                * (1 - crowding)
        )

        return uniqueness

    #################################################################
    # FINAL CROWDING SCORE
    #################################################################

    def final_crowding(self):

        signal = min(
            self.signal_concentration(),
            1
        )

        turnover = min(
            self.turnover_score() / 30,
            1
        )

        feature = min(
            self.feature_crowding(),
            1
        )

        factor = min(
            self.factor_crowding(),
            1
        )

        regime = min(
            self.regime_crowding(),
            1
        )

        capacity = 1 - min(
            self.capacity_stress(),
            1
        )

        score = (
                signal * .30
                + turnover * .20
                + feature * .20
                + factor * .15
                + regime * .10
                + capacity * .05
        )

        return round(score, 4)

    #################################################################
    # REPORT
    #################################################################

    def build_report(self):

        report = {

            "Signal_Concentration":
                self.signal_concentration(),

            "Portfolio_HHI":
                self.portfolio_hhi(),

            "Annual_Turnover":
                self.turnover_score(),

            "Feature_Crowding":
                self.feature_crowding(),

            "Factor_Crowding":
                self.factor_crowding(),

            "Regime_Crowding":
                self.regime_crowding(),

            "Capacity":
                self.capacity_stress(),

            "Alpha_Uniqueness":
                self.alpha_uniqueness(),

            "Crowding_Score":
                self.final_crowding()
        }

        report = pd.DataFrame(
            [report]
        )

        report.to_csv(
            f"{OUTPUT_DIR}/crowding_report.csv",
            index=False
        )

        return report

    #################################################################
    # TIMESERIES
    #################################################################

    def build_timeseries(self):

        if "Date" not in self.final_df.columns:
            return

        ts = (
            self.final_df
            .groupby("Date")
            .agg(
                Signals=("Signal", "sum"),
                Turnover=(
                    "Daily_Turnover",
                    "mean"
                )
            )
            .reset_index()
        )

        ts.to_csv(
            f"{OUTPUT_DIR}/crowding_timeseries.csv",
            index=False
        )

    #################################################################
    # EXPORT UNIQUENESS
    #################################################################

    def export_uniqueness(self):

        if len(self.adaptive_weights) == 0:
            return

        df = self.adaptive_weights.copy()

        if "Retention_Score" in \
                self.retention_scores.columns:

            df = df.merge(
                self.retention_scores[
                    ["Feature",
                     "Retention_Score"]
                ],
                on="Feature",
                how="left"
            )

        if "Capacity_Score" in \
                self.capacity_scores.columns:

            df = df.merge(
                self.capacity_scores[
                    ["Feature",
                     "Capacity_Score"]
                ],
                on="Feature",
                how="left"
            )

        df["Uniqueness"] = (
                df["Retention_Score"]
                * df["Capacity_Score"]
                * (1 - self.final_crowding())
        )

        df.to_csv(
            f"{OUTPUT_DIR}/alpha_uniqueness.csv",
            index=False
        )

    #################################################################
    # RUN
    #################################################################

    def run(self):

        print("\n" + "=" * 70)
        print("ALPHA CROWDING ENGINE")
        print("=" * 70)

        report = self.build_report()

        self.build_timeseries()

        self.export_uniqueness()

        print(report.T)

        return report


#########################################################################
# PUBLIC ENTRY POINT
#########################################################################

def run_alpha_crowding(
        final_df,
        portfolio,
        adaptive_weights,
        retention_scores,
        capacity_scores
):

    engine = AlphaCrowdingEngine(
        final_df,
        portfolio,
        adaptive_weights,
        retention_scores,
        capacity_scores
    )

    return engine.run()