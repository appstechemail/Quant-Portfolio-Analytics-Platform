"""
===============================================================================
File: alpha_stage_tracker.py
Path: src/alpha/alpha_stage_tracker.py

Institutional-Grade Quant Platform
----------------------------------

Tracks alpha through every stage of the investment pipeline.

Pipeline:

RAW
META
REGIME
VOLATILITY
CROSS_SECTION
PORTFOLIO

Used By:
--------
1. AlphaRetentionEngine
2. AlphaCapacityEngine
3. AlphaCrowdingEngine
4. AlphaLifecycleEngine
5. AlphaPipeline

Author : OpenAI
===============================================================================
"""

from __future__ import annotations

from enum import Enum
from typing import Dict

import pandas as pd


# =============================================================================
# ALPHA STAGES
# =============================================================================

class AlphaStage(Enum):

    RAW = "RAW"
    META = "META"
    REGIME = "REGIME"
    VOLATILITY = "VOLATILITY"
    CROSS_SECTION = "CROSS_SECTION"
    PORTFOLIO = "PORTFOLIO"


# =============================================================================
# STAGE TRACKER
# =============================================================================

class AlphaStageTracker:
    """
    Institutional Alpha Stage Tracker.

    Example
    -------
    tracker = AlphaStageTracker()

    tracker.add_stage(
        AlphaStage.RAW,
        raw_df
    )

    tracker.add_stage(
        AlphaStage.META,
        meta_df
    )

    tracker.get_stage(
        AlphaStage.META
    )
    """

    def __init__(self):

        self._stages: Dict[str, pd.DataFrame] = {}

    # --------------------------------------------------------------------- #
    # PUBLIC READ-ONLY STAGE ACCESS
    # --------------------------------------------------------------------- #

    @property
    def stages(self) -> Dict[str, pd.DataFrame]:
        """
        Read-only public access to tracked stages.

        Used by:
        --------
        - AlphaPipeline
        - AlphaRetentionEngine
        - AlphaCapacityEngine
        - AlphaCrowdingEngine
        - AlphaLifecycleEngine
        """

        return self._stages

    # --------------------------------------------------------------------- #
    # ADD STAGE
    # --------------------------------------------------------------------- #

    def add_stage(
        self,
        stage: AlphaStage | str,
        df: pd.DataFrame
    ) -> None:

        name = (
            stage.value
            if isinstance(stage, AlphaStage)
            else str(stage)
        )

        self._stages[name] = df.copy()

    # --------------------------------------------------------------------- #
    # GET STAGE
    # --------------------------------------------------------------------- #

    def get_stage(
        self,
        stage: AlphaStage | str
    ) -> pd.DataFrame:

        name = (
            stage.value
            if isinstance(stage, AlphaStage)
            else str(stage)
        )

        if name not in self._stages:
            raise ValueError(
                f"Stage {name} not found."
            )

        return self._stages[name]

    # --------------------------------------------------------------------- #
    # CHECK STAGE
    # --------------------------------------------------------------------- #

    def has_stage(
        self,
        stage: AlphaStage | str
    ) -> bool:

        name = stage.value if isinstance(stage, AlphaStage) else str(stage)

        return name in self._stages

    # --------------------------------------------------------------------- #
    # LIST STAGES
    # --------------------------------------------------------------------- #

    def list_stages(self):

        return list(self._stages.keys())

    # --------------------------------------------------------------------- #
    # SUMMARY
    # --------------------------------------------------------------------- #

    def summary(self):

        rows = []

        for stage, df in self._stages.items():

            rows.append(
                {
                    "Stage": stage,
                    "Rows": len(df),
                    "Columns": len(df.columns),
                    "Dates":
                        df["Date"].nunique()
                        if "Date" in df.columns
                        else 0,
                    "Companies":
                        df["Company"].nunique()
                        if "Company" in df.columns
                        else 0,
                }
            )

        return pd.DataFrame(rows)

    # --------------------------------------------------------------------- #
    # EXPORT
    # --------------------------------------------------------------------- #

    def export_summary(self):

        summary = self.summary()

        print("\n")
        print("=" * 80)
        print("ALPHA STAGE TRACKER")
        print("=" * 80)

        print(summary)

        print("=" * 80)

        return summary


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":

    tracker = AlphaStageTracker()

    df = pd.DataFrame(
        {
            "Date": ["2025-01-01"] * 3,
            "Company": ["A", "B", "C"]
        }
    )

    tracker.add_stage(
        AlphaStage.RAW,
        df
    )

    tracker.export_summary()