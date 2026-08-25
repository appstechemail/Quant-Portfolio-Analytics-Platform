"""
==============================================================================
INSTITUTIONAL QUANT PLATFORM
==============================================================================

Module:
    alpha_lifecycle_engine.py

Purpose
-------
Tracks complete alpha lifecycle.

Institutional Lifecycle States
------------------------------

1. NEW
2. GROWING
3. MATURE
4. WEAKENING
5. DEAD

Inputs
------

1. Adaptive Weights
2. Retention Scores
3. Capacity Scores
4. Crowding Scores
5. IC Trend
6. Decay Scores

Outputs
-------

data/alpha/outputs/

    alpha_lifecycle.csv
    alpha_state_transition.csv
    lifecycle_summary.json

==============================================================================

Author:
    Institutional Quant Platform
==============================================================================
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


OUTPUT_DIR = "data/alpha/outputs"


class AlphaLifecycleEngine:

    def __init__(self):

        self.lifecycle_df = None
        self.ic_table = pd.DataFrame()  

    def determine_state(
        self,
        row
    ):

        adaptive = row["Adaptive_Weight"]
        decay = row["Decay_Score"]
        crowding = row["Crowding_Score"]
        retention = row["Retention_Score"]

        if adaptive < 5:

            return "DEAD"

        if (
            adaptive > 25
            and retention > 0.8
            and decay > 0.8
            and crowding < 0.50
        ):

            return "MATURE"

        if (
            adaptive > 15
            and decay > 0.60
        ):

            return "GROWING"

        if (
            decay < 0.50
            or retention < 0.50
        ):

            return "WEAKENING"

        return "NEW"

    def calculate_lifecycle_score(
        self,
        df
    ):

        score = (

            0.35 * df["Adaptive_Weight"]

            + 0.25 * (
                df["Retention_Score"] * 100
            )

            + 0.20 * (
                df["Capacity_Score"] * 100
            )

            + 0.20 * (
                (
                    1
                    - df["Crowding_Score"]
                )
                * 100
            )
        )

        return score

    def fit(
        self,
        adaptive_df,
        retention_df,
        capacity_df,
        crowding_df
    ):

        df = adaptive_df.copy()

        df = df.merge(

            retention_df[
                [
                    "Feature",
                    "Retention_Score"
                ]
            ],

            on="Feature",
            how="left"
        )

        df = df.merge(

            capacity_df[
                [
                    "Feature",
                    "Capacity_Score"
                ]
            ],

            on="Feature",
            how="left"
        )

        df = df.merge(

            crowding_df[
                [
                    "Feature",
                    "Crowding_Score"
                ]
            ],

            on="Feature",
            how="left"
        )

        df.fillna(0, inplace=True)

        df["Lifecycle_Score"] = (

            self.calculate_lifecycle_score(
                df
            )
        )

        df["Lifecycle_State"] = (

            df.apply(
                self.determine_state,
                axis=1
            )
        )

        self.lifecycle_df = (

            df.sort_values(
                "Lifecycle_Score",
                ascending=False
            )
        )

        return self.lifecycle_df

    def lifecycle_summary(
        self
    ):

        if self.lifecycle_df is None:

            return {}

        return {

            "total_alphas":

                len(
                    self.lifecycle_df
                ),

            "new":

                int(
                    (
                        self.lifecycle_df[
                            "Lifecycle_State"
                        ]
                        == "NEW"
                    ).sum()
                ),

            "growing":

                int(
                    (
                        self.lifecycle_df[
                            "Lifecycle_State"
                        ]
                        == "GROWING"
                    ).sum()
                ),

            "mature":

                int(
                    (
                        self.lifecycle_df[
                            "Lifecycle_State"
                        ]
                        == "MATURE"
                    ).sum()
                ),

            "weakening":

                int(
                    (
                        self.lifecycle_df[
                            "Lifecycle_State"
                        ]
                        == "WEAKENING"
                    ).sum()
                ),

            "dead":

                int(
                    (
                        self.lifecycle_df[
                            "Lifecycle_State"
                        ]
                        == "DEAD"
                    ).sum()
                ),
        }

    def export(
        self,
        output_dir=OUTPUT_DIR
    ):

        if self.lifecycle_df is None:
            return

        output = Path(output_dir)

        output.mkdir(
            parents=True,
            exist_ok=True
        )

        self.lifecycle_df.to_csv(
            output / "alpha_lifecycle.csv",
            index=False
        )

        transition = (
            self.lifecycle_df[
                [
                    "Feature",
                    "Lifecycle_State",
                    "Lifecycle_Score"
                ]
            ]
        )

        transition.to_csv(
            output / "alpha_state_transition.csv",
            index=False
        )

        with open(
            output / "lifecycle_summary.json",
            "w"
        ) as f:

            json.dump(
                self.lifecycle_summary(),
                f,
                indent=4
            )

        print(
            "\n✓ Alpha Lifecycle exported"
        )

    def report(self):

        if self.lifecycle_df is None:

            print(
                "\nLifecycle dataframe not initialized."
            )

            return

        print("\n")

        print("=" * 80)
        print("ALPHA LIFECYCLE REPORT")
        print("=" * 80)

        print(

            self.lifecycle_df[
                [
                    "Feature",
                    "Lifecycle_State",
                    "Lifecycle_Score"
                ]
            ].head(20)

        )

        print("\n")

        print(
            self.lifecycle_summary()
        )

    #################################################################
    # RUN
    #################################################################

    def run(
        self,
        adaptive_df=None,
        retention_df=None,
        capacity_df=None,
        crowding_df=None,
        ic_table=None
    ):

        print("\n" + "=" * 80)
        print("ALPHA LIFECYCLE ENGINE")
        print("=" * 80)

        self.ic_table = (
            ic_table.copy()
            if isinstance(ic_table, pd.DataFrame)
            else pd.DataFrame()
        )

        adaptive_df = (
            adaptive_df
            if adaptive_df is not None
            else pd.DataFrame(
                columns=[
                    "Feature",
                    "Adaptive_Weight",
                    "Decay_Score"
                ]
            )
        )

        retention_df = (
            retention_df
            if retention_df is not None
            else pd.DataFrame(
                columns=[
                    "Feature",
                    "Retention_Score"
                ]
            )
        )

        capacity_df = (
            capacity_df
            if capacity_df is not None
            else pd.DataFrame(
                columns=[
                    "Feature",
                    "Capacity_Score"
                ]
            )
        )

        crowding_df = (
            crowding_df
            if crowding_df is not None
            else pd.DataFrame(
                columns=[
                    "Feature",
                    "Crowding_Score"
                ]
            )
        )

        # Build lifecycle dataframe
        report = self.fit(
            adaptive_df,
            retention_df,
            capacity_df,
            crowding_df
        )

        # Export artifacts
        self.export()

        # Display report
        self.report()

        return report