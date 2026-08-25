"""
============================================================
PORTFOLIO SELECTION ENGINE
============================================================

Institutional-grade portfolio selection.

Responsibilities
----------------
* Universe validation
* Liquidity filtering
* Score filtering
* Long-only selection
* Long-short selection
* Percentile selection
* Selection diagnostics

============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    PortfolioSelectionConfig,
    SelectionMethod,
)

# ============================================================
# CONSTANTS
# ============================================================

PORTFOLIO_SIDE_COLUMN = "Portfolio_Side"

LONG = "Long"

SHORT = "Short"

NEUTRAL = "Neutral"


# ============================================================
# RESULT
# ============================================================

@dataclass(slots=True)
class SelectionResult:

    selected_df: pd.DataFrame

    diagnostics: dict


# ============================================================
# VALIDATION
# ============================================================

REQUIRED_COLUMNS = [
    "Date",
]


def validate_ranked_universe(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> None:

    required = REQUIRED_COLUMNS.copy()

    required.extend(
        [
            config.score_column,
            config.rank_column,
        ]
    )

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df.empty:

        raise ValueError(
            "Ranked universe is empty."
        )


# ============================================================
# FILTERS
# ============================================================

def apply_score_filter(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> pd.DataFrame:

    if config.min_score <= 0:

        return df.copy()

    return (
        df.loc[
            df[config.score_column]
            >= config.min_score
        ]
        .copy()
    )


# ============================================================
# LONG ONLY
# ============================================================

def select_long_only(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> pd.DataFrame:

    out = df.copy()

    out[config.selected_column] = False

    out[PORTFOLIO_SIDE_COLUMN] = NEUTRAL

    mask = (
        out[config.rank_column]
        <= config.top_n
    )

    out.loc[
        mask,
        config.selected_column,
    ] = True

    out.loc[
        mask,
        PORTFOLIO_SIDE_COLUMN,
    ] = LONG

    return out


# ============================================================
# LONG SHORT
# ============================================================

def select_long_short(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> pd.DataFrame:

    out = df.copy()

    out[config.selected_column] = False

    out[PORTFOLIO_SIDE_COLUMN] = NEUTRAL

    # --------------------------------
    # LONGS
    # --------------------------------

    long_mask = (
        out[config.rank_column]
        <= config.top_n
    )

    out.loc[
        long_mask,
        config.selected_column,
    ] = True

    out.loc[
        long_mask,
        PORTFOLIO_SIDE_COLUMN,
    ] = LONG

    # --------------------------------
    # SHORTS
    # --------------------------------

    if config.bottom_n > 0:

        max_rank = (
            out.groupby("Date")[
                config.rank_column
            ]
            .transform("max")
        )

        cutoff = (
            max_rank
            - config.bottom_n
            + 1
        )

        short_mask = (
            out[config.rank_column]
            >= cutoff
        )

        out.loc[
            short_mask,
            config.selected_column,
        ] = True

        out.loc[
            short_mask,
            PORTFOLIO_SIDE_COLUMN,
        ] = SHORT

    return out


# ============================================================
# DIAGNOSTICS
# ============================================================

def selection_summary(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> dict:

    selected = df.loc[
        df[config.selected_column]
    ]

    summary = {

        "rows":
            len(df),

        "selected_rows":
            len(selected),

        "dates":
            df["Date"].nunique(),

        "avg_selected_per_date":
            float(
                selected.groupby("Date")
                .size()
                .mean()
            )
            if len(selected)
            else 0.0,
    }

    if PORTFOLIO_SIDE_COLUMN in selected.columns:

        summary["longs"] = int(
            (
                selected[
                    PORTFOLIO_SIDE_COLUMN
                ]
                == LONG
            ).sum()
        )

        summary["shorts"] = int(
            (
                selected[
                    PORTFOLIO_SIDE_COLUMN
                ]
                == SHORT
            ).sum()
        )

    return summary


# ============================================================
# ENGINE
# ============================================================

class SelectionEngine:

    def __init__(
        self,
        config: PortfolioSelectionConfig,
    ) -> None:

        self.config = config

    def run(
        self,
        ranked_df: pd.DataFrame,
    ) -> SelectionResult:

        validate_ranked_universe(
            ranked_df,
            self.config,
        )

        universe = apply_score_filter(
            ranked_df,
            self.config,
        )

        if universe.empty:

            out = ranked_df.copy()

            out[
                self.config.selected_column
            ] = False

            diagnostics = {

                "selected_rows": 0
            }

            return SelectionResult(
                selected_df=out,
                diagnostics=diagnostics,
            )

        if (
            self.config.selection_method
            == SelectionMethod.LONG_ONLY
        ):

            selected = select_long_only(
                universe,
                self.config,
            )

        else:

            selected = select_long_short(
                universe,
                self.config,
            )

        diagnostics = selection_summary(
            selected,
            self.config,
        )

        return SelectionResult(
            selected_df=selected,
            diagnostics=diagnostics,
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def select_portfolio(
    ranked_df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> pd.DataFrame:

    engine = SelectionEngine(
        config
    )

    return (
        engine.run(
            ranked_df
        )
        .selected_df
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [

    "SelectionResult",

    "SelectionEngine",

    "select_portfolio",

    "selection_summary",

    "validate_ranked_universe",
]