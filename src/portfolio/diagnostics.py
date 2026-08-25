# Part 1: Framework & Validation

"""
============================================================
PORTFOLIO DIAGNOSTICS
============================================================

Institutional Portfolio Monitoring Layer

Responsibilities
----------------

Selection Stability
Holding Overlap
Portfolio Turnover
Weight Drift
Sector Monitoring
Industry Monitoring
Capacity Analysis
Liquidity Analysis
Portfolio Reporting

This module NEVER

• changes weights
• changes selections
• computes scores

============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    PortfolioSelectionConfig,
)

EPS = 1e-12


# ============================================================
# REQUIRED COLUMNS
# ============================================================

BASE_REQUIRED_COLUMNS = [

    "Date",
]

from dataclasses import dataclass
from typing import Any


# ============================================================
# RESULT OBJECTS
# ============================================================

@dataclass(slots=True)
class TurnoverResult:
    """
    Portfolio turnover metrics.
    """

    average_turnover: float

    median_turnover: float

    maximum_turnover: float

    turnover_series: pd.Series


@dataclass(slots=True)
class OverlapResult:
    """
    Portfolio overlap metrics.
    """

    average_overlap: float

    minimum_overlap: float

    overlap_series: pd.Series


@dataclass(slots=True)
class CapacityResult:
    """
    Capacity diagnostics.
    """

    median_adv: float

    minimum_adv: float

    average_adv: float


# ============================================================
# VALIDATION
# ============================================================

def validate_portfolio(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> None:
    """
    Validate portfolio dataframe.
    """

    if df.empty:

        raise ValueError(
            "Portfolio dataframe is empty."
        )

    required = list(
        BASE_REQUIRED_COLUMNS
    )

    required.extend([

        config.selected_column,

        config.weight_column,

        config.score_column,
    ])

    missing = [

        col

        for col in required

        if col not in df.columns
    ]

    if missing:

        raise ValueError(

            "Missing portfolio columns:\n"

            + "\n".join(missing)
        )

    if not np.issubdtype(
        df["Date"].dtype,
        np.datetime64,
    ):

        try:

            pd.to_datetime(
                df["Date"]
            )

        except Exception:

            raise ValueError(
                "Date column must be datetime."
            )


# ============================================================
# HELPERS
# ============================================================

def selected_portfolio(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> pd.DataFrame:
    """
    Return active positions only.
    """

    validate_portfolio(
        df,
        config,
    )

    return (

        df.loc[
            df[
                config.selected_column
            ]
        ]
        .copy()
    )


def get_identifier_column(
    df: pd.DataFrame,
) -> str:
    """
    Detect security identifier.
    """

    candidates = [

        "Ticker",

        "Symbol",

        "Security",

        "Stock",
    ]

    for col in candidates:

        if col in df.columns:

            return col

    raise ValueError(

        "No security identifier column found.\n"

        "Expected one of:\n"

        + "\n".join(candidates)
    )


# ============================================================
# DATE UTILITIES
# ============================================================

def rebalance_dates(
    df: pd.DataFrame,
) -> list[pd.Timestamp]:
    """
    Sorted rebalance dates.
    """

    return sorted(
        pd.to_datetime(
            df["Date"]
        ).unique()
    )


def portfolio_by_date(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> dict:
    """
    Date -> portfolio dataframe.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    return {

        date: group.copy()

        for date, group
        in portfolio.groupby(
            "Date"
        )
    }


# ============================================================
# PART 2: PORTFOLIO SUMMARY & POSITION MONITORING
# ============================================================
"""
Portfolio Summary
Position Counts
Weight Statistics
Score Statistics
Top Holdings
Concentration Metrics
"""

# ============================================================
# POSITION COUNTS
# ============================================================

def positions_per_date(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> pd.Series:
    """
    Number of active positions
    on each rebalance date.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    return (
        portfolio
        .groupby("Date")
        .size()
        .rename(
            "PositionCount"
        )
    )


# ============================================================
# WEIGHT DISTRIBUTION
# ============================================================

def weight_distribution(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> dict:
    """
    Portfolio weight statistics.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    weights = (
        portfolio[
            config.weight_column
        ]
        .fillna(0.0)
    )

    if len(weights) == 0:

        return {}

    return {

        "Minimum":
            float(
                weights.min()
            ),

        "P25":
            float(
                weights.quantile(
                    0.25
                )
            ),

        "Median":
            float(
                weights.median()
            ),

        "P75":
            float(
                weights.quantile(
                    0.75
                )
            ),

        "Maximum":
            float(
                weights.max()
            ),

        "Mean":
            float(
                weights.mean()
            ),

        "Std":
            float(
                weights.std()
            ),
    }


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

def score_distribution(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> dict:
    """
    Distribution of selected scores.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    scores = (
        portfolio[
            config.score_column
        ]
        .fillna(0.0)
    )

    if len(scores) == 0:

        return {}

    return {

        "Minimum":
            float(
                scores.min()
            ),

        "P25":
            float(
                scores.quantile(
                    0.25
                )
            ),

        "Median":
            float(
                scores.median()
            ),

        "P75":
            float(
                scores.quantile(
                    0.75
                )
            ),

        "Maximum":
            float(
                scores.max()
            ),

        "Mean":
            float(
                scores.mean()
            ),

        "Std":
            float(
                scores.std()
            ),
    }


# ============================================================
# TOP POSITIONS
# ============================================================

def top_positions(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Largest positions.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    cols = [

        "Date",

        get_identifier_column(
            portfolio
        ),

        config.score_column,

        config.weight_column,
    ]

    cols = [

        c

        for c in cols

        if c in portfolio.columns
    ]

    return (
        portfolio
        .sort_values(
            config.weight_column,
            ascending=False,
        )
        [cols]
        .head(top_n)
        .copy()
    )


# ============================================================
# CONCENTRATION METRICS
# ============================================================

def concentration_metrics(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> dict:
    """
    Portfolio concentration.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    weights = (
        portfolio[
            config.weight_column
        ]
        .fillna(0.0)
        .values
    )

    if len(weights) == 0:

        return {}

    hhi = float(
        np.sum(
            weights ** 2
        )
    )

    effective_n = (

        float(
            1.0 / hhi
        )

        if hhi > EPS

        else 0.0
    )

    sorted_weights = np.sort(
        weights
    )

    top5 = float(
        sorted_weights[-5:].sum()
    )

    top10 = float(
        sorted_weights[-10:].sum()
    )

    return {

        "HHI":
            hhi,

        "EffectivePositions":
            effective_n,

        "LargestPosition":
            float(
                weights.max()
            ),

        "Top5Concentration":
            top5,

        "Top10Concentration":
            top10,
    }


# ============================================================
# PORTFOLIO SUMMARY
# ============================================================

def portfolio_summary(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> dict:
    """
    High-level portfolio summary.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    if portfolio.empty:

        return {}

    position_counts = (
        positions_per_date(
            portfolio,
            config,
        )
    )

    weights = (
        portfolio[
            config.weight_column
        ]
        .fillna(0.0)
    )

    return {

        "Dates":
            int(
                portfolio[
                    "Date"
                ].nunique()
            ),

        "TotalRows":
            int(
                len(portfolio)
            ),

        "AveragePositions":
            float(
                position_counts.mean()
            ),

        "MinimumPositions":
            int(
                position_counts.min()
            ),

        "MaximumPositions":
            int(
                position_counts.max()
            ),

        "AverageWeight":
            float(
                weights.mean()
            ),

        "MaximumWeight":
            float(
                weights.max()
            ),

        "MinimumWeight":
            float(
                weights.min()
            ),

        "AverageScore":
            float(
                portfolio[
                    config.score_column
                ].mean()
            ),
    }


# ============================================================
# SUMMARY TABLE
# ============================================================

def portfolio_summary_table(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> pd.DataFrame:
    """
    Dashboard-friendly summary.
    """

    summary = portfolio_summary(
        df,
        config,
    )

    return pd.DataFrame(

        [
            {
                "Metric": k,
                "Value": v,
            }
            for k, v
            in summary.items()
        ]
    )


# ============================================================
# PART 3
# HOLDING OVERLAP & PORTFOLIO STABILITY
# ============================================================
"""
Holding Overlap
Retention Rate
Selection Stability Score
Position Churn
Holding Persistence
Stability Report
"""

# ============================================================
# HOLDINGS SNAPSHOT
# ============================================================

def holdings_by_date(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> dict[pd.Timestamp, set]:
    """
    Date -> holdings set
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    identifier = (
        get_identifier_column(
            portfolio
        )
    )

    return {

        date: set(
            group[
                identifier
            ]
        )

        for date, group
        in portfolio.groupby(
            "Date"
        )
    }


# ============================================================
# HOLDING OVERLAP
# ============================================================

def holding_overlap(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> OverlapResult:
    """
    Jaccard overlap.

    1.0 = identical portfolio

    0.0 = complete replacement
    """

    holdings = holdings_by_date(
        df,
        config,
    )

    dates = sorted(
        holdings.keys()
    )

    overlaps = []

    overlap_dates = []

    for i in range(
        1,
        len(dates),
    ):

        prev_set = holdings[
            dates[i - 1]
        ]

        curr_set = holdings[
            dates[i]
        ]

        union = (
            prev_set
            | curr_set
        )

        if len(union) == 0:

            overlap = 1.0

        else:

            overlap = (
                len(
                    prev_set
                    & curr_set
                )
                / len(union)
            )

        overlaps.append(
            overlap
        )

        overlap_dates.append(
            dates[i]
        )

    overlap_series = pd.Series(
        overlaps,
        index=overlap_dates,
        name="HoldingOverlap",
    )

    return OverlapResult(

        average_overlap=
        float(
            overlap_series.mean()
        ),

        minimum_overlap=
        float(
            overlap_series.min()
        ),

        overlap_series=
        overlap_series,
    )


# ============================================================
# HOLDING RETENTION
# ============================================================

def holding_retention_rate(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> pd.Series:
    """
    Fraction of previous holdings
    retained in next rebalance.

    Better institutional metric
    than pure Jaccard overlap.
    """

    holdings = holdings_by_date(
        df,
        config,
    )

    dates = sorted(
        holdings.keys()
    )

    retention = []

    retention_dates = []

    for i in range(
        1,
        len(dates),
    ):

        prev_set = holdings[
            dates[i - 1]
        ]

        curr_set = holdings[
            dates[i]
        ]

        if len(prev_set) == 0:

            value = 1.0

        else:

            value = (
                len(
                    prev_set
                    & curr_set
                )
                / len(prev_set)
            )

        retention.append(
            value
        )

        retention_dates.append(
            dates[i]
        )

    return pd.Series(
        retention,
        index=retention_dates,
        name="RetentionRate",
    )


# ============================================================
# SELECTION STABILITY
# ============================================================

def selection_stability_score(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> float:
    """
    Portfolio stability score.

    Institutional score:

    100 = identical portfolio

    0 = complete replacement
    """

    overlap = holding_overlap(
        df,
        config,
    )

    return float(
        overlap.average_overlap
        * 100.0
    )


# ============================================================
# POSITION CHURN
# ============================================================

def position_churn(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> pd.Series:
    """
    Number of names replaced
    every rebalance.
    """

    holdings = holdings_by_date(
        df,
        config,
    )

    dates = sorted(
        holdings.keys()
    )

    churn_values = []

    churn_dates = []

    for i in range(
        1,
        len(dates),
    ):

        prev_set = holdings[
            dates[i - 1]
        ]

        curr_set = holdings[
            dates[i]
        ]

        exited = (
            prev_set
            - curr_set
        )

        entered = (
            curr_set
            - prev_set
        )

        churn = max(
            len(exited),
            len(entered),
        )

        churn_values.append(
            churn
        )

        churn_dates.append(
            dates[i]
        )

    return pd.Series(
        churn_values,
        index=churn_dates,
        name="PositionChurn",
    )


# ============================================================
# HOLDING PERSISTENCE
# ============================================================

def holding_persistence(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> pd.DataFrame:
    """
    Number of rebalances
    each security survives.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    identifier = (
        get_identifier_column(
            portfolio
        )
    )

    persistence = (

        portfolio

        .groupby(
            identifier
        )["Date"]

        .nunique()

        .sort_values(
            ascending=False
        )

        .rename(
            "RebalancesSurvived"
        )
    )

    return persistence.to_frame()


# ============================================================
# STABILITY REPORT
# ============================================================

def stability_report(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> dict:
    """
    Institutional stability report.
    """

    overlap = holding_overlap(
        df,
        config,
    )

    retention = (
        holding_retention_rate(
            df,
            config,
        )
    )

    churn = position_churn(
        df,
        config,
    )

    return {

        "AverageOverlap":
            overlap.average_overlap,

        "MinimumOverlap":
            overlap.minimum_overlap,

        "AverageRetention":
            float(
                retention.mean()
            ),

        "MinimumRetention":
            float(
                retention.min()
            ),

        "AverageChurn":
            float(
                churn.mean()
            ),

        "MaximumChurn":
            float(
                churn.max()
            ),

        "StabilityScore":
            selection_stability_score(
                df,
                config,
            ),
    }



# ============================================================
# PART 4: PORTFOLIO TURNOVER & WEIGHT DRIFT
# ============================================================

"""
Holding Overlap
Retention Rate
Selection Stability
Position Churn

Portfolio Turnover
Weight Drift
Trade Size Analytics
Rebalance Analytics
"""

# ============================================================
# PORTFOLIO TURNOVER
# ============================================================

def portfolio_turnover(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> TurnoverResult:
    """
    Institutional turnover metric.

    Turnover =
    0.5 * sum(abs(w_t - w_t-1))

    Returns turnover for every rebalance.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    identifier = (
        get_identifier_column(
            portfolio
        )
    )

    dates = sorted(
        portfolio["Date"]
        .unique()
    )

    turnover_values = []

    turnover_dates = []

    for i in range(
        1,
        len(dates),
    ):

        prev_df = portfolio.loc[
            portfolio["Date"]
            == dates[i - 1]
        ]

        curr_df = portfolio.loc[
            portfolio["Date"]
            == dates[i]
        ]

        prev_w = (
            prev_df
            .set_index(identifier)
            [config.weight_column]
        )

        curr_w = (
            curr_df
            .set_index(identifier)
            [config.weight_column]
        )

        universe = (
            prev_w.index
            .union(
                curr_w.index
            )
        )

        prev_w = (
            prev_w
            .reindex(universe)
            .fillna(0.0)
        )

        curr_w = (
            curr_w
            .reindex(universe)
            .fillna(0.0)
        )

        turnover = (
            0.5
            * np.abs(
                curr_w
                - prev_w
            ).sum()
        )

        turnover_values.append(
            float(turnover)
        )

        turnover_dates.append(
            dates[i]
        )

    turnover_series = pd.Series(
        turnover_values,
        index=turnover_dates,
        name="Turnover",
    )

    return TurnoverResult(

        average_turnover=
        float(
            turnover_series.mean()
        ),

        median_turnover=
        float(
            turnover_series.median()
        ),

        maximum_turnover=
        float(
            turnover_series.max()
        ),

        turnover_series=
        turnover_series,
    )


# ============================================================
# WEIGHT DRIFT
# ============================================================

def weight_drift(
    current_weights: pd.Series,
    target_weights: pd.Series,
) -> float:
    """
    L1 distance between
    current and target weights.
    """

    universe = (
        current_weights.index
        .union(
            target_weights.index
        )
    )

    current = (
        current_weights
        .reindex(universe)
        .fillna(0.0)
    )

    target = (
        target_weights
        .reindex(universe)
        .fillna(0.0)
    )

    return float(

        np.abs(
            current
            - target
        ).sum()
    )


# ============================================================
# DRIFT BY REBALANCE DATE
# ============================================================

def drift_series(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> pd.Series:
    """
    Drift between consecutive
    portfolio weights.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    identifier = (
        get_identifier_column(
            portfolio
        )
    )

    dates = sorted(
        portfolio["Date"]
        .unique()
    )

    drift_values = []

    drift_dates = []

    for i in range(
        1,
        len(dates),
    ):

        prev_df = portfolio.loc[
            portfolio["Date"]
            == dates[i - 1]
        ]

        curr_df = portfolio.loc[
            portfolio["Date"]
            == dates[i]
        ]

        prev_w = (
            prev_df
            .set_index(identifier)
            [config.weight_column]
        )

        curr_w = (
            curr_df
            .set_index(identifier)
            [config.weight_column]
        )

        drift = weight_drift(
            prev_w,
            curr_w,
        )

        drift_values.append(
            drift
        )

        drift_dates.append(
            dates[i]
        )

    return pd.Series(
        drift_values,
        index=drift_dates,
        name="WeightDrift",
    )


# ============================================================
# TRADE SIZE ANALYTICS
# ============================================================

def trade_size_distribution(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> dict:
    """
    Distribution of absolute
    weight changes.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    identifier = (
        get_identifier_column(
            portfolio
        )
    )

    dates = sorted(
        portfolio["Date"]
        .unique()
    )

    trades = []

    for i in range(
        1,
        len(dates),
    ):

        prev_df = portfolio.loc[
            portfolio["Date"]
            == dates[i - 1]
        ]

        curr_df = portfolio.loc[
            portfolio["Date"]
            == dates[i]
        ]

        prev_w = (
            prev_df
            .set_index(identifier)
            [config.weight_column]
        )

        curr_w = (
            curr_df
            .set_index(identifier)
            [config.weight_column]
        )

        universe = (
            prev_w.index
            .union(
                curr_w.index
            )
        )

        prev_w = (
            prev_w
            .reindex(universe)
            .fillna(0.0)
        )

        curr_w = (
            curr_w
            .reindex(universe)
            .fillna(0.0)
        )

        trades.extend(
            np.abs(
                curr_w
                - prev_w
            ).tolist()
        )

    if len(trades) == 0:

        return {}

    trades = np.asarray(
        trades
    )

    return {

        "AverageTrade":
            float(
                trades.mean()
            ),

        "MedianTrade":
            float(
                np.median(
                    trades
                )
            ),

        "MaximumTrade":
            float(
                trades.max()
            ),

        "P95Trade":
            float(
                np.quantile(
                    trades,
                    0.95,
                )
            ),
    }


# ============================================================
# REBALANCE ANALYTICS
# ============================================================

def rebalance_analytics(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> dict:
    """
    Full rebalance diagnostics.
    """

    turnover = portfolio_turnover(
        df,
        config,
    )

    drift = drift_series(
        df,
        config,
    )

    return {

        "AverageTurnover":
            turnover.average_turnover,

        "MedianTurnover":
            turnover.median_turnover,

        "MaximumTurnover":
            turnover.maximum_turnover,

        "AverageDrift":
            float(
                drift.mean()
            ),

        "MaximumDrift":
            float(
                drift.max()
            ),

        "TradeDistribution":
            trade_size_distribution(
                df,
                config,
            ),
    }


# ============================================================
# PART 5: CAPACITY & LIQUIDITY DIAGNOSTICS
# ============================================================
"""
Portfolio Summary
Position Monitoring

Holding Overlap
Selection Stability

Turnover
Weight Drift

Liquidity Diagnostics
Capacity Estimation
ADV Analysis
Market-Cap Analysis
Dollar Volume Analysis
"""

# ============================================================
# DEFAULT COLUMN NAMES
# ============================================================

DEFAULT_ADV_COLUMN = "ADV"

DEFAULT_DOLLAR_VOLUME_COLUMN = "DollarVolume"

DEFAULT_MARKET_CAP_COLUMN = "MarketCap"


# ============================================================
# CAPACITY VALIDATION
# ============================================================

def validate_liquidity_columns(
    df: pd.DataFrame,
    adv_column: str,
) -> bool:
    """
    Check whether liquidity data exists.
    """

    return adv_column in df.columns


# ============================================================
# ADV EXPOSURE
# ============================================================

def adv_exposure(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    adv_column: str = DEFAULT_ADV_COLUMN,
) -> pd.DataFrame:
    """
    Exposure by ADV.

    Useful for identifying
    liquidity concentration.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    if adv_column not in portfolio.columns:

        return pd.DataFrame()

    result = portfolio[[
        "Date",
        adv_column,
        config.weight_column,
    ]].copy()

    result["Weight_x_ADV"] = (
        result[
            config.weight_column
        ]
        * result[
            adv_column
        ]
    )

    return result


# ============================================================
# LIQUIDITY SUMMARY
# ============================================================

def liquidity_summary(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    adv_column: str = DEFAULT_ADV_COLUMN,
) -> CapacityResult:
    """
    Aggregate liquidity metrics.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    if adv_column not in portfolio.columns:

        return CapacityResult(
            median_adv=0.0,
            minimum_adv=0.0,
            average_adv=0.0,
        )

    adv = (
        portfolio[
            adv_column
        ]
        .fillna(0.0)
    )

    return CapacityResult(

        median_adv=float(
            adv.median()
        ),

        minimum_adv=float(
            adv.min()
        ),

        average_adv=float(
            adv.mean()
        ),
    )


# ============================================================
# LOW LIQUIDITY POSITIONS
# ============================================================

def low_liquidity_positions(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    adv_threshold: float,
    adv_column: str = DEFAULT_ADV_COLUMN,
) -> pd.DataFrame:
    """
    Securities below liquidity threshold.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    if adv_column not in portfolio.columns:

        return pd.DataFrame()

    return (

        portfolio.loc[
            portfolio[
                adv_column
            ]
            < adv_threshold
        ]
        .copy()
    )


# ============================================================
# MARKET CAP SUMMARY
# ============================================================

def market_cap_summary(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    market_cap_column: str = (
        DEFAULT_MARKET_CAP_COLUMN
    ),
) -> dict:
    """
    Portfolio market-cap profile.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    if (
        market_cap_column
        not in portfolio.columns
    ):

        return {}

    mcap = (
        portfolio[
            market_cap_column
        ]
        .fillna(0.0)
    )

    return {

        "AverageMarketCap":
            float(
                mcap.mean()
            ),

        "MedianMarketCap":
            float(
                mcap.median()
            ),

        "MinimumMarketCap":
            float(
                mcap.min()
            ),

        "MaximumMarketCap":
            float(
                mcap.max()
            ),
    }


# ============================================================
# DOLLAR VOLUME SUMMARY
# ============================================================

def dollar_volume_summary(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    volume_column: str = (
        DEFAULT_DOLLAR_VOLUME_COLUMN
    ),
) -> dict:
    """
    Portfolio trading liquidity.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    if (
        volume_column
        not in portfolio.columns
    ):

        return {}

    dv = (
        portfolio[
            volume_column
        ]
        .fillna(0.0)
    )

    return {

        "AverageDollarVolume":
            float(
                dv.mean()
            ),

        "MedianDollarVolume":
            float(
                dv.median()
            ),

        "MinimumDollarVolume":
            float(
                dv.min()
            ),

        "MaximumDollarVolume":
            float(
                dv.max()
            ),
    }


# ============================================================
# CAPACITY ESTIMATE
# ============================================================

def estimate_capacity(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    max_adv_participation: float = 0.05,
    adv_column: str = DEFAULT_ADV_COLUMN,
) -> float:
    """
    Estimate deployable capital.

    Simplified institutional rule:

    Capacity =
        min(ADV * participation)
        across holdings
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    if adv_column not in portfolio.columns:

        return 0.0

    adv = (
        portfolio[
            adv_column
        ]
        .fillna(0.0)
    )

    capacity = (
        adv.min()
        * max_adv_participation
    )

    return float(
        capacity
    )


# ============================================================
# LIQUIDITY CONCENTRATION
# ============================================================

def liquidity_concentration(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    adv_column: str = DEFAULT_ADV_COLUMN,
) -> dict:
    """
    Measures whether weights
    are concentrated in
    illiquid securities.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    if adv_column not in portfolio.columns:

        return {}

    ranked = (
        portfolio
        .sort_values(
            adv_column
        )
    )

    bottom20 = ranked.head(
        max(
            1,
            int(
                len(ranked)
                * 0.20
            )
        )
    )

    return {

        "Bottom20PctLiquidityWeight":
            float(
                bottom20[
                    config.weight_column
                ].sum()
            )
    }


# ============================================================
# MASTER CAPACITY REPORT
# ============================================================

def capacity_report(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> dict:
    """
    Institutional liquidity report.
    """

    return {

        "Liquidity":
            liquidity_summary(
                df,
                config,
            ),

        "MarketCap":
            market_cap_summary(
                df,
                config,
            ),

        "DollarVolume":
            dollar_volume_summary(
                df,
                config,
            ),

        "EstimatedCapacity":
            estimate_capacity(
                df,
                config,
            ),

        "LiquidityConcentration":
            liquidity_concentration(
                df,
                config,
            ),
    }

# ============================================================
# PART 6: SECTOR / INDUSTRY EXPOSURE ANALYTICS
# ============================================================

"""
Portfolio Summary
Position Monitoring

Holding Overlap
Selection Stability

Turnover
Weight Drift

Liquidity Diagnostics
Capacity Diagnostics

Sector Exposure
Industry Exposure
Diversification Metrics
Concentration Monitoring
Exposure Drift
"""

DEFAULT_SECTOR_COLUMN = "Sector"

DEFAULT_INDUSTRY_COLUMN = "Industry"


# ============================================================
# SECTOR EXPOSURE
# ============================================================

def sector_exposure(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    sector_column: str = DEFAULT_SECTOR_COLUMN,
) -> pd.DataFrame:
    """
    Aggregate portfolio weight
    by sector.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    if sector_column not in portfolio.columns:

        return pd.DataFrame()

    exposure = (

        portfolio

        .groupby(
            sector_column
        )[config.weight_column]

        .sum()

        .sort_values(
            ascending=False
        )

        .rename(
            "Weight"
        )
    )

    return exposure.to_frame()


# ============================================================
# INDUSTRY EXPOSURE
# ============================================================

def industry_exposure(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    industry_column: str = DEFAULT_INDUSTRY_COLUMN,
) -> pd.DataFrame:
    """
    Aggregate portfolio weight
    by industry.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    if industry_column not in portfolio.columns:

        return pd.DataFrame()

    exposure = (

        portfolio

        .groupby(
            industry_column
        )[config.weight_column]

        .sum()

        .sort_values(
            ascending=False
        )

        .rename(
            "Weight"
        )
    )

    return exposure.to_frame()


# ============================================================
# SECTOR CONCENTRATION
# ============================================================

def sector_concentration(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    sector_column: str = DEFAULT_SECTOR_COLUMN,
) -> dict:
    """
    Sector HHI and top-sector metrics.
    """

    exposure = sector_exposure(
        df,
        config,
        sector_column,
    )

    if exposure.empty:

        return {}

    weights = exposure[
        "Weight"
    ].values

    hhi = float(
        np.sum(
            weights ** 2
        )
    )

    effective_sectors = (

        float(
            1.0 / hhi
        )

        if hhi > EPS

        else 0.0
    )

    return {

        "SectorHHI":
            hhi,

        "EffectiveSectors":
            effective_sectors,

        "LargestSectorWeight":
            float(
                weights.max()
            ),
    }


# ============================================================
# INDUSTRY CONCENTRATION
# ============================================================

def industry_concentration(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    industry_column: str = DEFAULT_INDUSTRY_COLUMN,
) -> dict:
    """
    Industry concentration metrics.
    """

    exposure = industry_exposure(
        df,
        config,
        industry_column,
    )

    if exposure.empty:

        return {}

    weights = exposure[
        "Weight"
    ].values

    hhi = float(
        np.sum(
            weights ** 2
        )
    )

    effective_industries = (

        float(
            1.0 / hhi
        )

        if hhi > EPS

        else 0.0
    )

    return {

        "IndustryHHI":
            hhi,

        "EffectiveIndustries":
            effective_industries,

        "LargestIndustryWeight":
            float(
                weights.max()
            ),
    }


# ============================================================
# TOP SECTOR RISK
# ============================================================

def top_sector_risk(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    sector_column: str = DEFAULT_SECTOR_COLUMN,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Largest sector exposures.
    """

    exposure = sector_exposure(
        df,
        config,
        sector_column,
    )

    if exposure.empty:

        return exposure

    return exposure.head(
        top_n
    )


# ============================================================
# TOP INDUSTRY RISK
# ============================================================

def top_industry_risk(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    industry_column: str = DEFAULT_INDUSTRY_COLUMN,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Largest industry exposures.
    """

    exposure = industry_exposure(
        df,
        config,
        industry_column,
    )

    if exposure.empty:

        return exposure

    return exposure.head(
        top_n
    )


# ============================================================
# EXPOSURE DRIFT
# ============================================================

def exposure_drift(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    category_column: str,
) -> pd.DataFrame:
    """
    Exposure evolution over time.

    Useful for monitoring
    style drift and sector drift.
    """

    portfolio = selected_portfolio(
        df,
        config,
    )

    if category_column not in portfolio.columns:

        return pd.DataFrame()

    return (

        portfolio

        .pivot_table(

            index="Date",

            columns=category_column,

            values=config.weight_column,

            aggfunc="sum",

            fill_value=0.0,
        )

        .sort_index()
    )


# ============================================================
# DIVERSIFICATION SCORE
# ============================================================

def diversification_score(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    sector_column: str = DEFAULT_SECTOR_COLUMN,
) -> float:
    """
    Diversification score.

    0 = highly concentrated

    100 = highly diversified
    """

    concentration = sector_concentration(
        df,
        config,
        sector_column,
    )

    if not concentration:

        return 0.0

    effective = concentration[
        "EffectiveSectors"
    ]

    return float(
        min(
            100.0,
            effective * 10.0,
        )
    )


# ============================================================
# REGULATORY CONCENTRATION CHECK
# ============================================================

def concentration_limit_check(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
    sector_limit: float = 0.25,
    sector_column: str = DEFAULT_SECTOR_COLUMN,
) -> pd.DataFrame:
    """
    Identify sectors exceeding
    policy limits.
    """

    exposure = sector_exposure(
        df,
        config,
        sector_column,
    )

    if exposure.empty:

        return pd.DataFrame()

    return exposure.loc[
        exposure["Weight"]
        > sector_limit
    ].copy()


# ============================================================
# MASTER EXPOSURE REPORT
# ============================================================

def exposure_report(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> dict:
    """
    Institutional exposure report.
    """

    return {

        "SectorExposure":
            sector_exposure(
                df,
                config,
            ),

        "IndustryExposure":
            industry_exposure(
                df,
                config,
            ),

        "SectorConcentration":
            sector_concentration(
                df,
                config,
            ),

        "IndustryConcentration":
            industry_concentration(
                df,
                config,
            ),

        "DiversificationScore":
            diversification_score(
                df,
                config,
            ),

        "ConcentrationBreaches":
            concentration_limit_check(
                df,
                config,
            ),
    }


# ============================================================
# PART 7: INSTITUTIONAL MASTER REPORTING LAYER
# ============================================================
"""
Framework & Validation
Portfolio Summary
Position Monitoring

Holding Overlap
Selection Stability
Position Churn

Portfolio Turnover
Weight Drift
Trade Analytics

Liquidity Diagnostics
Capacity Estimation

Sector / Industry Exposure
Diversification Metrics
Concentration Monitoring

Risk Committee Reporting
Portfolio Manager Reporting
Master Dashboard
Portfolio Report

"""

# ============================================================
# RISK COMMITTEE SUMMARY
# ============================================================

def risk_committee_summary(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> dict:
    """
    Risk Committee report.

    Focus:
        Concentration
        Turnover
        Liquidity
        Stability
    """

    turnover = portfolio_turnover(
        df,
        config,
    )

    stability = stability_report(
        df,
        config,
    )

    concentration = concentration_metrics(
        df,
        config,
    )

    capacity = capacity_report(
        df,
        config,
    )

    return {

        "AverageTurnover":
            turnover.average_turnover,

        "MaximumTurnover":
            turnover.maximum_turnover,

        "StabilityScore":
            stability[
                "StabilityScore"
            ],

        "EffectivePositions":
            concentration[
                "EffectivePositions"
            ],

        "LargestPosition":
            concentration[
                "LargestPosition"
            ],

        "EstimatedCapacity":
            capacity[
                "EstimatedCapacity"
            ],
    }


# ============================================================
# PORTFOLIO MANAGER SUMMARY
# ============================================================

def portfolio_manager_summary(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> dict:
    """
    PM-focused diagnostics.
    """

    summary = portfolio_summary(
        df,
        config,
    )

    concentration = concentration_metrics(
        df,
        config,
    )

    exposure = exposure_report(
        df,
        config,
    )

    return {

        "PortfolioSummary":
            summary,

        "Concentration":
            concentration,

        "DiversificationScore":
            exposure[
                "DiversificationScore"
            ],

        "TopSectors":
            top_sector_risk(
                df,
                config,
            ),
    }


# ============================================================
# DIAGNOSTIC DASHBOARD
# ============================================================

def diagnostics_dashboard(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> dict:
    """
    Dashboard-ready structure.

    Suitable for
    Streamlit / Dash / BI tools.
    """

    return {

        "Summary":
            portfolio_summary(
                df,
                config,
            ),

        "Concentration":
            concentration_metrics(
                df,
                config,
            ),

        "Stability":
            stability_report(
                df,
                config,
            ),

        "Turnover":
            rebalance_analytics(
                df,
                config,
            ),

        "Capacity":
            capacity_report(
                df,
                config,
            ),

        "Exposure":
            exposure_report(
                df,
                config,
            ),
    }


# ============================================================
# EXPORTABLE SUMMARY TABLE
# ============================================================

def summary_table(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> pd.DataFrame:
    """
    Flattened report table.

    Useful for:

        Excel
        PDF
        reporting
    """

    summary = portfolio_summary(
        df,
        config,
    )

    concentration = concentration_metrics(
        df,
        config,
    )

    stability = stability_report(
        df,
        config,
    )

    turnover = portfolio_turnover(
        df,
        config,
    )

    records = []

    def add_block(
        source: dict,
        category: str,
    ) -> None:

        for key, value in source.items():

            if isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            ):

                records.append({

                    "Category":
                        category,

                    "Metric":
                        key,

                    "Value":
                        float(value),
                })

    add_block(
        summary,
        "Summary",
    )

    add_block(
        concentration,
        "Concentration",
    )

    add_block(
        stability,
        "Stability",
    )

    records.append({

        "Category":
            "Turnover",

        "Metric":
            "AverageTurnover",

        "Value":
            float(
                turnover.average_turnover
            ),
    })

    records.append({

        "Category":
            "Turnover",

        "Metric":
            "MaximumTurnover",

        "Value":
            float(
                turnover.maximum_turnover
            ),
    })

    return pd.DataFrame(
        records
    )


# ============================================================
# DIAGNOSTICS REPORT DTO
# ============================================================

@dataclass(slots=True)
class PortfolioDiagnosticsReport:

    summary: dict[str, Any]
    weight_distribution: dict[str, Any]
    score_distribution: dict[str, Any]
    concentration: dict[str, Any]

    stability: dict[str, Any]

    turnover: dict[str, Any]

    capacity: dict[str, Any]

    exposure: dict[str, Any]

    risk_committee: dict[str, Any]

    portfolio_manager: dict[str, Any]

# ============================================================
# MASTER PORTFOLIO REPORT
# ============================================================

def portfolio_report(
    df: pd.DataFrame,
    config: PortfolioSelectionConfig,
) -> PortfolioDiagnosticsReport:
    """
    Institutional-grade portfolio report.

    Single entry point.
    """

    return {

        # ---------------------
        # Core
        # ---------------------

        "Summary":
            portfolio_summary(
                df,
                config,
            ),

        "WeightDistribution":
            weight_distribution(
                df,
                config,
            ),

        "ScoreDistribution":
            score_distribution(
                df,
                config,
            ),

        "Concentration":
            concentration_metrics(
                df,
                config,
            ),

        # ---------------------
        # Stability
        # ---------------------

        "Stability":
            stability_report(
                df,
                config,
            ),

        # ---------------------
        # Turnover
        # ---------------------

        "Turnover":
            rebalance_analytics(
                df,
                config,
            ),

        # ---------------------
        # Liquidity
        # ---------------------

        "Capacity":
            capacity_report(
                df,
                config,
            ),

        # ---------------------
        # Exposure
        # ---------------------

        "Exposure":
            exposure_report(
                df,
                config,
            ),

        # ---------------------
        # Governance
        # ---------------------

        "RiskCommittee":
            risk_committee_summary(
                df,
                config,
            ),

        "PortfolioManager":
            portfolio_manager_summary(
                df,
                config,
            ),
    }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [

    # Master Reports
    "portfolio_report",
    "diagnostics_dashboard",
    "summary_table",

    # PM / Risk Reports
    "portfolio_manager_summary",
    "risk_committee_summary",

    # Core Diagnostics
    "portfolio_summary",
    "concentration_metrics",
    "stability_report",
    "capacity_report",
    "exposure_report",
]