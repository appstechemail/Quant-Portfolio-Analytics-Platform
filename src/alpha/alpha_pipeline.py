"""
===============================================================================
File: alpha_pipeline.py
Author: Parmod Chaudhary
Created: 2026

Institutional-Grade Quant Platform
----------------------------------

Master Alpha Coordinator

Pipeline:

RAW
 ↓
META
 ↓
REGIME
 ↓
VOLATILITY
 ↓
CROSS_SECTION
 ↓
PORTFOLIO

            ↓

AlphaStageTracker

            ↓

Retention Engine
Capacity Engine
Crowding Engine
Lifecycle Engine

            ↓

Final Alpha Health Score

Outputs:

data/alpha/outputs/

    alpha_retention.csv
    alpha_capacity.csv
    alpha_crowding.csv
    alpha_lifecycle.csv

    alpha_health_score.json
    alpha_pipeline_summary.json

    RAW.parquet
    META.parquet
    REGIME.parquet
    VOLATILITY.parquet
    CROSS_SECTION.parquet
    PORTFOLIO.parquet

===============================================================================
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import json

from src.alpha.alpha_stage_tracker import (
    AlphaStageTracker,
    AlphaStage
)

from src.alpha.alpha_retention_engine import (
    AlphaRetentionEngine
)

from src.alpha.alpha_capacity_engine import (
    AlphaCapacityEngine
)

from src.alpha.alpha_crowding_engine import (
    AlphaCrowdingEngine
)

from src.alpha.alpha_lifecycle_engine import (
    AlphaLifecycleEngine
)


OUTPUT_DIR = Path("data/alpha/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class AlphaPipeline:

    def __init__(self):

        # -----------------------------------------------------
        # TRACKING
        # -----------------------------------------------------

        self.tracker = AlphaStageTracker()

        # -----------------------------------------------------
        # ENGINES
        # -----------------------------------------------------

        self.retention = AlphaRetentionEngine()
        self.capacity = AlphaCapacityEngine()
        self.crowding = AlphaCrowdingEngine()
        self.lifecycle = AlphaLifecycleEngine()

        # -----------------------------------------------------
        # PIPELINE ARTIFACTS
        # -----------------------------------------------------

        self.adaptive_weights = pd.DataFrame()
        self.retention_scores = pd.DataFrame()
        self.capacity_scores = pd.DataFrame()
        self.crowding_scores = pd.DataFrame()

        # -----------------------------------------------------
        # OPTIONAL FUTURE OBJECTS
        # -----------------------------------------------------

        self.metadata = {}
        self.pipeline_input = None
        self.diagnostics = {}

    # ==========================================================
    # TRACK STAGES
    # ==========================================================

    def register_stages(
        self,
        raw_df,
        meta_df,
        regime_df,
        volatility_df,
        cross_section_df,
        portfolio_df
    ):

        self.tracker.add_stage(
            AlphaStage.RAW,
            raw_df
        )

        self.tracker.add_stage(
            AlphaStage.META,
            meta_df
        )

        self.tracker.add_stage(
            AlphaStage.REGIME,
            regime_df
        )

        self.tracker.add_stage(
            AlphaStage.VOLATILITY,
            volatility_df
        )

        self.tracker.add_stage(
            AlphaStage.CROSS_SECTION,
            cross_section_df
        )

        self.tracker.add_stage(
            AlphaStage.PORTFOLIO,
            portfolio_df
        )

    # ==========================================================
    # RETENTION
    # ==========================================================

    def run_retention(self):

        for stage, df in self.tracker.stages.items():

            self.retention.evaluate_stage(
                stage,
                df
            )

        report = self.retention.generate_report()

        self.retention.export(
            OUTPUT_DIR
        )

        return report

    # ==========================================================
    # CAPACITY
    # ==========================================================

    def run_capacity(self, portfolio_df):

        if (
            portfolio_df is None
            or portfolio_df.empty
        ):
            return {
                "report": pd.DataFrame(),
                "summary": {}
            }

        capacity_input = portfolio_df.copy()

        # ----------------------------------
        # Dollar Volume
        # ----------------------------------

        if "DollarVolume" not in capacity_input.columns:

            capacity_input["DollarVolume"] = (
                capacity_input["Close"]
                * capacity_input["Volume"]
            )

        # ----------------------------------
        # Prediction Alpha
        # ----------------------------------

        if "Prediction_Alpha" not in capacity_input.columns:

            capacity_input["Prediction_Alpha"] = (
                capacity_input.get(
                    "Expected_Return",
                    capacity_input["Probability"]
                )
            )

        report = self.capacity.build_capacity_report(
            capacity_input
        )

        summary = self.capacity.portfolio_capacity(
            report
        )

        print("\n" + "=" * 80)
        print("ALPHA CAPACITY REPORT")
        print("=" * 80)

        print(
            report[
                [
                    "Company",
                    "Max_Capital",
                    "Market_Impact",
                    "Alpha_Remaining",
                    "Alpha_Retention",
                    "Deployable"
                ]
            ].head()
        )

        print("\nPORTFOLIO CAPACITY")

        print(summary)

        return {
            "report": report,
            "summary": summary
        }

    # ==========================================================
    # CROWDING
    # ==========================================================


    def run_crowding(
        self,
        portfolio_df
    ):

        self.crowding.portfolio = (
            portfolio_df.copy()
            if (
                portfolio_df is not None
                and
                not portfolio_df.empty
            )
            else pd.DataFrame()
        )

        return self.crowding.run()

    # ==========================================================
    # LIFECYCLE
    # ==========================================================

    def run_lifecycle(self, ic_table):

        if (
            self.adaptive_weights.empty
        ):
            return pd.DataFrame()

        self.lifecycle.fit(
            adaptive_df=self.adaptive_weights,
            retention_df=self.retention_scores,
            capacity_df=self.capacity_scores,
            crowding_df=self.crowding_scores
        )

        return self.lifecycle.run(
            ic_table=ic_table
        )

    # ==========================================================
    # HEALTH SCORE
    # ==========================================================

    def calculate_health_score(
        self,
        retention,
        capacity,
        crowding,
        lifecycle
    ):

        score = {

            "retention_score":

                float(
                    retention["sharpe"].iloc[-1]
                )

                if (
                    isinstance(
                        retention,
                        pd.DataFrame
                    )
                    and len(retention)
                    and "sharpe" in retention.columns
                )

                else 0.0,

            "capacity_score":

                float(

                    capacity.get(
                        "summary",
                        {}
                    ).get(
                        "Average_Retention",
                        0.0
                    )

                )

                if isinstance(
                    capacity,
                    dict
                )

                else 0.0,

            "crowding_score":

                float(

                    1
                    -
                    crowding.get(
                        "Crowding_Score",
                        1.0
                    )

                )

                if isinstance(
                    crowding,
                    (dict, pd.Series)
                )

                else 0.0,

            "lifecycle_score":

                float(

                    lifecycle[
                        "Lifecycle_Score"
                    ].mean()

                )

                if (
                    isinstance(
                        lifecycle,
                        pd.DataFrame
                    )
                    and not lifecycle.empty
                    and "Lifecycle_Score"
                    in lifecycle.columns
                )

                else 0.0,
        }

        score["overall"] = round(

            np.mean(
                list(
                    score.values()
                )
            ),

            4
        )

        return score

    # ==========================================================
    # EXPORT
    # ==========================================================

    def export_summary(
        self,
        summary
    ):

        with open(
            OUTPUT_DIR /
            "alpha_health_score.json",
            "w"
        ) as f:

            json.dump(
                summary,
                f,
                indent=4
            )

        with open(
            OUTPUT_DIR /
            "alpha_pipeline_summary.json",
            "w"
        ) as f:

            json.dump(
                summary,
                f,
                indent=4
            )

    # ==========================================================
    # MASTER
    # ==========================================================

    def run_all(
            self,
            raw_df=None,
            meta_df=None,
            regime_df=None,
            volatility_df=None,
            cross_section_df=None,
            portfolio_df=None,
            ic_table=None
    ):

        print("\n" + "=" * 80)
        print("INSTITUTIONAL ALPHA PIPELINE")
        print("=" * 80)

        # -------------------------------------------------
        # INPUT VALIDATION
        # -------------------------------------------------

        required_inputs = {
            "raw_df": raw_df,
            "meta_df": meta_df,
            "regime_df": regime_df,
            "volatility_df": volatility_df,
            "cross_section_df": cross_section_df,
            "portfolio_df": portfolio_df,
            "ic_table": ic_table
        }

        missing = [
            name
            for name, value in required_inputs.items()
            if value is None
        ]

        if missing:

            raise ValueError(
                f"AlphaPipeline missing inputs: {missing}"
            )

        # -------------------------------------------------
        # STAGE REGISTRATION
        # -------------------------------------------------

        self.register_stages(
            raw_df,
            meta_df,
            regime_df,
            volatility_df,
            cross_section_df,
            portfolio_df
        )

        # -------------------------------------------------
        # EXPORT STAGE TRACKER
        # -------------------------------------------------

        self.tracker.export_summary()

        # -------------------------------------------------
        # ENGINES
        # -------------------------------------------------

        retention = self.run_retention()
        capacity = self.run_capacity(portfolio_df)
        crowding = self.run_crowding(portfolio_df)
        lifecycle = self.run_lifecycle(ic_table)

        # -------------------------------------------------
        # ALPHA HEALTH
        # -------------------------------------------------

        health = self.calculate_health_score(
            retention,
            capacity,
            crowding,
            lifecycle
        )

        # -------------------------------------------------
        # EXPORT SUMMARY
        # -------------------------------------------------

        self.export_summary(health)

        print("\nFINAL ALPHA HEALTH SCORE")
        print(health)

        return {
            "retention": retention,
            "capacity": capacity,
            "crowding": crowding,
            "lifecycle": lifecycle,
            "health": health
        }

# =============================================================================
# EXAMPLE
# =============================================================================
#
# pipeline = AlphaPipeline()
#
# results = pipeline.run_all(
#
#     raw_df,
#     meta_df,
#     regime_df,
#     volatility_df,
#     cross_section_df,
#     portfolio_df,
#     metrics_ic_table
#
# )
#
# =============================================================================