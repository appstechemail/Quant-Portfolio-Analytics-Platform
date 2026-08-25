"""
==========================================================
PORTFOLIO CONSTRUCTION
Exposure Analytics Engine
==========================================================

Purpose
-------
Provides immutable exposure analytics for Portfolio objects.

This module is the single source of truth for every
portfolio exposure calculation.

Consumers

    Constraints
    Optimizer
    Rebalancer
    Diagnostics
    Backtester
    Risk Engine

Design

Portfolio
      │
      ▼
PortfolioExposure
      │
      ├── Gross Exposure
      ├── Net Exposure
      ├── Long Exposure
      ├── Short Exposure
      ├── Sector Exposure
      ├── Industry Exposure
      ├── Concentration
      └── Cash Allocation

==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from .portfolio_builder import Portfolio


# ==========================================================
# HELPERS
# ==========================================================

_WEIGHT_COLUMN = "Position_Weight"

_SECTOR_COLUMN = "Sector"

_INDUSTRY_COLUMN = "Industry"


# ==========================================================
# EXPOSURE EXCEPTION
# ==========================================================

class ExposureError(Exception):
    """
    Raised when exposure analytics cannot be computed.
    """

    pass


# ==========================================================
# VALIDATION
# ==========================================================

def _validate_portfolio(
    portfolio: Portfolio,
) -> None:
    """
    Validate portfolio object.
    """

    if portfolio is None:

        raise ExposureError(
            "Portfolio is None."
        )

    holdings = portfolio.holdings

    if holdings.empty:

        raise ExposureError(
            "Portfolio has no holdings."
        )

    if _WEIGHT_COLUMN not in holdings.columns:

        raise ExposureError(
            "Missing Position_Weight column."
        )

    if not pd.api.types.is_numeric_dtype(
        holdings[_WEIGHT_COLUMN]
    ):

        raise ExposureError(
            "Position_Weight must be numeric."
        )


# ==========================================================
# EXPOSURE SNAPSHOT
# ==========================================================

@dataclass(frozen=True, slots=True)
class ExposureSnapshot:
    """
    Immutable exposure snapshot.

    Returned by PortfolioExposure.summary().
    """

    gross_exposure: float

    net_exposure: float

    long_exposure: float

    short_exposure: float

    largest_position: float

    cash_weight: float

    effective_holdings: float


# ==========================================================
# PORTFOLIO EXPOSURE ENGINE
# ==========================================================

class PortfolioExposure:
    """
    Immutable exposure analytics.

    Parameters
    ----------
    portfolio

        Portfolio domain object.
    """

    def __init__(
        self,
        portfolio: Portfolio,
    ) -> None:

        _validate_portfolio(
            portfolio
        )

        self._portfolio = portfolio

    # ------------------------------------------------------
    # Convenience
    # ------------------------------------------------------

    @property
    def portfolio(self) -> Portfolio:

        return self._portfolio

    # ------------------------------------------------------

    @cached_property
    def holdings(
        self,
    ) -> pd.DataFrame:
        """
        Cached holdings.

        Safe because Portfolio is immutable.
        """

        return self._portfolio.holdings.copy()

    # ------------------------------------------------------

    @cached_property
    def weights(
        self,
    ) -> pd.Series:
        """
        Position weights.
        """

        return self.holdings[
            _WEIGHT_COLUMN
        ].astype(float)

    # ------------------------------------------------------

    @cached_property
    def absolute_weights(
        self,
    ) -> pd.Series:

        return self.weights.abs()

    # ------------------------------------------------------

    @cached_property
    def position_count(
        self,
    ) -> int:

        return len(
            self.holdings
        )

    # ------------------------------------------------------

    def __len__(
        self,
    ) -> int:

        return self.position_count

    # ------------------------------------------------------

    def __repr__(
        self,
    ) -> str:

        return (

            f"PortfolioExposure("
            f"positions={self.position_count})"

        )


    # ==========================================================
    # CORE EXPOSURE METRICS
    # ==========================================================

    @cached_property
    def gross_exposure(
        self,
    ) -> float:
        """
        Gross exposure.

        Gross = Σ |weights|
        """

        return float(
            self.absolute_weights.sum()
        )

    # ------------------------------------------------------

    @cached_property
    def net_exposure(
        self,
    ) -> float:
        """
        Net exposure.

        Net = Σ weights
        """

        return float(
            self.weights.sum()
        )

    # ------------------------------------------------------

    @cached_property
    def long_exposure(
        self,
    ) -> float:
        """
        Total long exposure.
        """

        return float(

            self.weights[
                self.weights > 0
            ].sum()

        )

    # ------------------------------------------------------

    @cached_property
    def short_exposure(
        self,
    ) -> float:
        """
        Absolute short exposure.

        Always returned as positive.
        """

        return float(

            np.abs(

                self.weights[
                    self.weights < 0
                ].sum()

            )

        )

    # ------------------------------------------------------

    @cached_property
    def weight_sum(
        self,
    ) -> float:
        """
        Sum of raw weights.
        """

        return float(
            self.weights.sum()
        )

    # ------------------------------------------------------

    @cached_property
    def normalized_weights(
        self,
    ) -> pd.Series:
        """
        Normalize weights to gross exposure.

        Σ |w| = 1

        Used by concentration statistics.
        """

        gross = self.gross_exposure

        if gross <= 0:

            return pd.Series(
                np.zeros(
                    len(self.weights)
                ),
                index=self.weights.index,
                dtype=float,
            )

        return self.weights / gross

    # ------------------------------------------------------

    @cached_property
    def absolute_normalized_weights(
        self,
    ) -> pd.Series:
        """
        Absolute normalized weights.

        Σ |w| = 1
        """

        return self.normalized_weights.abs()

    # ------------------------------------------------------

    @cached_property
    def largest_position(
        self,
    ) -> float:
        """
        Largest absolute position.
        """

        if self.position_count == 0:

            return 0.0

        return float(

            self.absolute_weights.max()

        )

    # ------------------------------------------------------

    @cached_property
    def smallest_position(
        self,
    ) -> float:
        """
        Smallest absolute position.
        """

        if self.position_count == 0:

            return 0.0

        return float(

            self.absolute_weights.min()

        )

    # ------------------------------------------------------

    @cached_property
    def average_position(
        self,
    ) -> float:
        """
        Mean absolute position size.
        """

        if self.position_count == 0:

            return 0.0

        return float(

            self.absolute_weights.mean()

        )

    # ------------------------------------------------------

    @cached_property
    def median_position(
        self,
    ) -> float:
        """
        Median absolute position.
        """

        if self.position_count == 0:

            return 0.0

        return float(

            self.absolute_weights.median()

        )

    # ------------------------------------------------------

    @cached_property
    def cash_weight(
        self,
    ) -> float:
        """
        Remaining portfolio cash.

        Assumes target gross exposure = 1.
        """

        return float(

            max(
                0.0,
                1.0 - self.gross_exposure,
            )

        )

    # ------------------------------------------------------

    @cached_property
    def is_long_only(
        self,
    ) -> bool:
        """
        True if portfolio has no short positions.
        """

        return bool(

            (self.weights >= 0).all()

        )

    # ------------------------------------------------------

    @cached_property
    def is_market_neutral(
        self,
    ) -> bool:
        """
        True if net exposure is approximately zero.
        """

        return bool(

            abs(self.net_exposure)
            < 1e-8

        )


    # ==========================================================
    # SECTOR / INDUSTRY EXPOSURES
    # ==========================================================

    @cached_property
    def sector_weights(
        self,
    ) -> pd.Series:
        """
        Gross sector exposure.

        Returns
        -------
        Series

            index : Sector

            values : absolute portfolio weight
        """

        if _SECTOR_COLUMN not in self.holdings.columns:

            return pd.Series(
                dtype=float
            )

        return (

            self.holdings
            .assign(
                __abs_weight__=self.absolute_weights
            )
            .groupby(_SECTOR_COLUMN)["__abs_weight__"]
            .sum()
            .sort_values(
                ascending=False
            )

        )

    # ------------------------------------------------------

    @cached_property
    def industry_weights(
        self,
    ) -> pd.Series:
        """
        Gross industry exposure.
        """

        if _INDUSTRY_COLUMN not in self.holdings.columns:

            return pd.Series(
                dtype=float
            )

        return (

            self.holdings
            .assign(
                __abs_weight__=self.absolute_weights
            )
            .groupby(_INDUSTRY_COLUMN)["__abs_weight__"]
            .sum()
            .sort_values(
                ascending=False
            )

        )

    # ------------------------------------------------------

    @cached_property
    def sector_count(
        self,
    ) -> int:

        return len(
            self.sector_weights
        )

    # ------------------------------------------------------

    @cached_property
    def industry_count(
        self,
    ) -> int:

        return len(
            self.industry_weights
        )

    # ------------------------------------------------------

    @cached_property
    def largest_sector(
        self,
    ):
        """
        Largest sector exposure.
        """

        if self.sector_weights.empty:

            return None

        return self.sector_weights.index[0]

    # ------------------------------------------------------

    @cached_property
    def largest_sector_weight(
        self,
    ) -> float:

        if self.sector_weights.empty:

            return 0.0

        return float(

            self.sector_weights.iloc[0]

        )

    # ------------------------------------------------------

    @cached_property
    def largest_industry(
        self,
    ):

        if self.industry_weights.empty:

            return None

        return self.industry_weights.index[0]

    # ------------------------------------------------------

    @cached_property
    def largest_industry_weight(
        self,
    ) -> float:

        if self.industry_weights.empty:

            return 0.0

        return float(

            self.industry_weights.iloc[0]

        )

    # ------------------------------------------------------

    def sector_weight(
        self,
        sector: str,
    ) -> float:
        """
        Exposure of a single sector.
        """

        if sector not in self.sector_weights.index:

            return 0.0

        return float(

            self.sector_weights.loc[sector]

        )

    # ------------------------------------------------------

    def industry_weight(
        self,
        industry: str,
    ) -> float:
        """
        Exposure of a single industry.
        """

        if industry not in self.industry_weights.index:

            return 0.0

        return float(

            self.industry_weights.loc[industry]

        )

    # ------------------------------------------------------

    def top_sectors(
        self,
        n: int = 10,
    ) -> pd.Series:
        """
        Largest sectors.
        """

        return self.sector_weights.head(n)

    # ------------------------------------------------------

    def top_industries(
        self,
        n: int = 10,
    ) -> pd.Series:
        """
        Largest industries.
        """

        return self.industry_weights.head(n)

    # ------------------------------------------------------

    @cached_property
    def sector_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        DataFrame representation.
        """

        if self.sector_weights.empty:

            return pd.DataFrame(
                columns=[
                    "Sector",
                    "Weight",
                ]
            )

        return (

            self.sector_weights
            .rename("Weight")
            .rename_axis("Sector")
            .reset_index()

        )

    # ------------------------------------------------------

    @cached_property
    def industry_dataframe(
        self,
    ) -> pd.DataFrame:

        if self.industry_weights.empty:

            return pd.DataFrame(
                columns=[
                    "Industry",
                    "Weight",
                ]
            )

        return (

            self.industry_weights
            .rename("Weight")
            .rename_axis("Industry")
            .reset_index()

        )
    
    # ==========================================================
    # CONCENTRATION METRICS
    # ==========================================================

    @cached_property
    def hhi(
        self,
    ) -> float:
        """
        Herfindahl–Hirschman Index (HHI).

        HHI = Σ(w²)

        Uses absolute normalized weights.

        Range
        -----
        Equal-weight portfolio:
            ≈ 1/N

        Fully concentrated:
            1.0
        """

        w = self.absolute_normalized_weights

        if len(w) == 0:
            return 0.0

        return float(
            np.square(w).sum()
        )

    # ------------------------------------------------------

    @cached_property
    def effective_holdings(
        self,
    ) -> float:
        """
        Effective Number of Holdings (ENH).

        ENH = 1 / HHI
        """

        if self.hhi <= 0:
            return 0.0

        return float(
            1.0 / self.hhi
        )

    # ------------------------------------------------------

    def concentration_ratio(
        self,
        top_n: int = 5,
    ) -> float:
        """
        Concentration ratio.

        Sum of largest N absolute positions.
        """

        if self.position_count == 0:
            return 0.0

        return float(

            self.absolute_weights
            .sort_values(
                ascending=False
            )
            .head(top_n)
            .sum()

        )

    # ------------------------------------------------------

    @cached_property
    def top5_concentration(
        self,
    ) -> float:

        return self.concentration_ratio(
            5
        )

    # ------------------------------------------------------

    @cached_property
    def top10_concentration(
        self,
    ) -> float:

        return self.concentration_ratio(
            10
        )

    # ------------------------------------------------------

    @cached_property
    def equal_weight(
        self,
    ) -> float:
        """
        Equal-weight allocation.
        """

        if self.position_count == 0:
            return 0.0

        return float(
            1.0 / self.position_count
        )

    # ==========================================================
    # SUMMARY
    # ==========================================================

    def summary(
        self,
    ) -> ExposureSnapshot:
        """
        Immutable exposure snapshot.
        """

        return ExposureSnapshot(

            gross_exposure=self.gross_exposure,

            net_exposure=self.net_exposure,

            long_exposure=self.long_exposure,

            short_exposure=self.short_exposure,

            largest_position=self.largest_position,

            cash_weight=self.cash_weight,

            effective_holdings=self.effective_holdings,

        )

    # ==========================================================
    # REPORTS
    # ==========================================================

    def exposure_report(
        self,
    ) -> dict:
        """
        Complete exposure report.

        Returns
        -------
        dict
        """

        return {

            "Positions": self.position_count,
            "Gross Exposure": self.gross_exposure,
            "Net Exposure": self.net_exposure,
            "Long Exposure": self.long_exposure,
            "Short Exposure": self.short_exposure,
            "Cash Weight": self.cash_weight,
            "Largest Position": self.largest_position,
            "Average Position": self.average_position,
            "Median Position": self.median_position,
            "HHI": self.hhi,
            "Effective Holdings": self.effective_holdings,
            "Top 5 Concentration": self.top5_concentration,
            "Top 10 Concentration": self.top10_concentration,
            "Sector Count": self.sector_count,
            "Industry Count": self.industry_count,
            "Largest Sector": self.largest_sector,
            "Largest Sector Weight": self.largest_sector_weight,
            "Largest Industry": self.largest_industry,
            "Largest Industry Weight": self.largest_industry_weight,
            "Market Neutral": self.is_market_neutral,
            "Long Only": self.is_long_only,
        }

    # ==========================================================
    # EXPORT
    # ==========================================================#

    def to_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Holdings DataFrame.

        Defensive copy.
        """

        return self.holdings.copy()

# ==========================================================
# END
# ==========================================================