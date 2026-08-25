"""
===========================================================
Institutional Portfolio Analytics Engine
===========================================================

Purpose
-------
Provides portfolio construction analytics used by:

    • Portfolio Managers
    • Risk Teams
    • Execution Teams
    • Attribution Systems
    • Reporting Systems
    • Backtests

Analytics Domains
-----------------

Part 1  Framework & Core Objects
Part 2  Portfolio Statistics
Part 3  Exposure Analytics
Part 4  Concentration Analytics
Part 5  Risk Analytics
Part 6  Factor Analytics
Part 7  Performance Analytics
Part 8  Rebalance Analytics
Part 9  Execution Analytics Integration
Part 10 Capacity & Liquidity Analytics
Part 11 Institutional Reporting
Part 12 Analytics Engine
Part 13 Factory APIs

===========================================================
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import logging
import numpy as np
import pandas as pd

from dataclasses import asdict
import json

logger = logging.getLogger(__name__)


# ============================================================
# NUMERICAL CONSTANTS
# ============================================================

EPSILON: float = 1e-12

TRADING_DAYS: int = 252

MONTHS_PER_YEAR: int = 12


# ============================================================
# ANALYTICS FREQUENCY
# ============================================================

class AnalyticsFrequency(str, Enum):
    """
    Analytics sampling frequency.
    """

    DAILY = "daily"

    WEEKLY = "weekly"

    MONTHLY = "monthly"

    QUARTERLY = "quarterly"

    YEARLY = "yearly"


# ============================================================
# ANALYTICS METADATA
# ============================================================

@dataclass(slots=True)
class AnalyticsMetadata:
    """
    Metadata attached to every report.
    """

    generated_at: datetime

    portfolio_name: str | None = None

    benchmark_name: str | None = None

    strategy_name: str | None = None

    universe_name: str | None = None

    frequency: AnalyticsFrequency = (
        AnalyticsFrequency.DAILY
    )

    version: str = "1.0"


# ============================================================
# BASE ANALYTICS RESULT
# ============================================================

@dataclass(slots=True)
class AnalyticsResult:
    """
    Parent object used by all analytics.

    Every analytics section inherits
    from this structure.
    """

    metadata: AnalyticsMetadata

    analytics_name: str

    success: bool = True

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# ANALYTICS EXCEPTION
# ============================================================

class AnalyticsError(Exception):
    """
    Base analytics exception.
    """

    pass


# ============================================================
# VALIDATION MIXIN
# ============================================================

class ValidationMixin:
    """
    Shared validation helpers.
    """

    # --------------------------------------------------------

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        name: str,
    ) -> None:

        if df is None:

            raise AnalyticsError(
                f"{name} cannot be None."
            )

        if not isinstance(
            df,
            pd.DataFrame,
        ):

            raise AnalyticsError(
                f"{name} must be DataFrame."
            )

        if df.empty:

            raise AnalyticsError(
                f"{name} is empty."
            )

    # --------------------------------------------------------

    @staticmethod
    def validate_columns(
        df: pd.DataFrame,
        required_columns: list[str],
        dataframe_name: str,
    ) -> None:

        missing = (
            set(required_columns)
            -
            set(df.columns)
        )

        if missing:

            raise AnalyticsError(
                f"{dataframe_name} missing columns: "
                f"{sorted(missing)}"
            )

    # --------------------------------------------------------

    @staticmethod
    def validate_series(
        series: pd.Series,
        name: str,
    ) -> None:

        if series is None:

            raise AnalyticsError(
                f"{name} cannot be None."
            )

        if len(series) == 0:

            raise AnalyticsError(
                f"{name} is empty."
            )

    # --------------------------------------------------------

    @staticmethod
    def validate_weights(
        weights: pd.Series,
    ) -> None:

        if np.isnan(weights).any():

            raise AnalyticsError(
                "NaN weights detected."
            )

        if np.isinf(weights).any():

            raise AnalyticsError(
                "Infinite weights detected."
            )


# ============================================================
# ANALYTICS BASE CLASS
# ============================================================

class BaseAnalytics(
    ValidationMixin,
    ABC,
):
    """
    Parent analytics object.

    Shared by:

        Portfolio analytics
        Risk analytics
        Factor analytics
        Performance analytics
        Attribution analytics
    """

    def __init__(
        self,
        metadata: AnalyticsMetadata,
    ) -> None:

        self.metadata = metadata

    # --------------------------------------------------------

    @property
    def generated_at(
        self,
    ) -> datetime:

        return (
            self.metadata.generated_at
        )


# ============================================================
# ANALYTICS SNAPSHOT
# ============================================================

@dataclass(slots=True)
class AnalyticsSnapshot:
    """
    Point-in-time portfolio snapshot.

    Used throughout analytics engine.
    """

    date: pd.Timestamp

    portfolio_value: float

    gross_exposure: float

    net_exposure: float

    cash_weight: float

    holdings_count: int


# ============================================================
# ANALYTICS CONFIG
# ============================================================

@dataclass(slots=True)
class AnalyticsConfig:
    """
    Global analytics settings.
    """

    trading_days: int = TRADING_DAYS

    annualization_factor: int = (
        TRADING_DAYS
    )

    frequency: AnalyticsFrequency = (
        AnalyticsFrequency.DAILY
    )

    confidence_level: float = 0.95

    benchmark_required: bool = False


# ============================================================
# ANALYTICS UTILITIES
# ============================================================

class AnalyticsUtils:
    """
    Shared analytics helpers.
    """

    # --------------------------------------------------------

    @staticmethod
    def annualize_return(
        return_series: pd.Series,
        periods_per_year: int = (
            TRADING_DAYS
        ),
    ) -> float:

        if len(return_series) == 0:

            return 0.0

        compounded = (
            (1.0 + return_series)
            .prod()
        )

        years = (
            len(return_series)
            /
            periods_per_year
        )

        if years <= 0:

            return 0.0

        return float(
            compounded
            ** (1 / years)
            - 1
        )

    # --------------------------------------------------------

    @staticmethod
    def annualize_volatility(
        return_series: pd.Series,
        periods_per_year: int = (
            TRADING_DAYS
        ),
    ) -> float:

        if len(return_series) < 2:

            return 0.0

        return float(
            return_series.std()
            *
            np.sqrt(
                periods_per_year
            )
        )

    # --------------------------------------------------------

    @staticmethod
    def safe_divide(
        numerator: float,
        denominator: float,
    ) -> float:

        if abs(
            denominator
        ) < EPSILON:

            return 0.0

        return float(
            numerator
            /
            denominator
        )

    # --------------------------------------------------------

    @staticmethod
    def normalize_weights(
        weights: pd.Series,
    ) -> pd.Series:

        total = weights.sum()

        if abs(total) < EPSILON:

            return weights * 0.0

        return weights / total


# ============================================================
# PART 2
# PORTFOLIO STATISTICS
# ============================================================

# ============================================================
# PORTFOLIO STATISTICS RESULT
# ============================================================

@dataclass(slots=True)
class PortfolioStatisticsResult(
    AnalyticsResult,
):
    """
    Institutional portfolio statistics.

    Core return/risk metrics used by:

        • PM dashboards
        • Risk reporting
        • Attribution
        • Optimizer diagnostics
    """

    observations: int = 0

    total_return: float = 0.0

    annualized_return: float = 0.0

    annualized_volatility: float = 0.0

    sharpe_ratio: float = 0.0

    sortino_ratio: float = 0.0

    calmar_ratio: float = 0.0

    max_drawdown: float = 0.0

    average_return: float = 0.0

    median_return: float = 0.0

    skewness: float = 0.0

    kurtosis: float = 0.0

    hit_ratio: float = 0.0

    positive_periods: int = 0

    negative_periods: int = 0

    flat_periods: int = 0


# ============================================================
# DRAWDOWN RESULT
# ============================================================

@dataclass(slots=True)
class DrawdownResult:
    """
    Drawdown analytics.
    """

    max_drawdown: float

    current_drawdown: float

    drawdown_series: pd.Series

    underwater_series: pd.Series

    peak_series: pd.Series


# ============================================================
# PORTFOLIO STATISTICS ANALYZER
# ============================================================

class PortfolioStatisticsAnalyzer(
    BaseAnalytics,
):
    """
    Core portfolio statistics engine.

    Computes:

        Returns
        Volatility
        Sharpe
        Sortino
        Calmar
        Drawdowns
        Distribution statistics
    """

    def __init__(
        self,
        metadata: AnalyticsMetadata,
        config: AnalyticsConfig | None = None,
    ) -> None:

        super().__init__(metadata)

        self.config = (
            config
            if config is not None
            else AnalyticsConfig()
        )

    # --------------------------------------------------------
    # Return Validation
    # --------------------------------------------------------

    def validate_returns(
        self,
        returns: pd.Series,
    ) -> pd.Series:

        self.validate_series(
            returns,
            "returns",
        )

        returns = (
            returns
            .astype(float)
            .dropna()
        )

        if len(returns) == 0:

            raise AnalyticsError(
                "No valid returns available."
            )

        return returns

    # --------------------------------------------------------
    # Total Return
    # --------------------------------------------------------

    @staticmethod
    def total_return(
        returns: pd.Series,
    ) -> float:

        return float(
            (1.0 + returns)
            .prod()
            - 1.0
        )

    # --------------------------------------------------------
    # Average Return
    # --------------------------------------------------------

    @staticmethod
    def average_return(
        returns: pd.Series,
    ) -> float:

        return float(
            returns.mean()
        )

    # --------------------------------------------------------
    # Median Return
    # --------------------------------------------------------

    @staticmethod
    def median_return(
        returns: pd.Series,
    ) -> float:

        return float(
            returns.median()
        )

    # --------------------------------------------------------
    # Sharpe Ratio
    # --------------------------------------------------------

    def sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
    ) -> float:

        ann_return = (
            AnalyticsUtils.annualize_return(
                returns,
                self.config.annualization_factor,
            )
        )

        ann_vol = (
            AnalyticsUtils.annualize_volatility(
                returns,
                self.config.annualization_factor,
            )
        )

        return AnalyticsUtils.safe_divide(
            ann_return - risk_free_rate,
            ann_vol,
        )

    # --------------------------------------------------------
    # Sortino Ratio
    # --------------------------------------------------------

    def sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
    ) -> float:

        downside = returns[
            returns < 0
        ]

        if len(downside) == 0:

            return np.inf

        downside_vol = float(
            downside.std()
            *
            np.sqrt(
                self.config.annualization_factor
            )
        )

        ann_return = (
            AnalyticsUtils.annualize_return(
                returns,
                self.config.annualization_factor,
            )
        )

        return AnalyticsUtils.safe_divide(
            ann_return - risk_free_rate,
            downside_vol,
        )

    # --------------------------------------------------------
    # Drawdowns
    # --------------------------------------------------------

    @staticmethod
    def compute_drawdowns(
        returns: pd.Series,
    ) -> DrawdownResult:

        cumulative = (
            1.0 + returns
        ).cumprod()

        peaks = (
            cumulative
            .cummax()
        )

        drawdown = (
            cumulative
            /
            peaks
            - 1.0
        )

        max_dd = float(
            drawdown.min()
        )

        current_dd = float(
            drawdown.iloc[-1]
        )

        return DrawdownResult(
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            drawdown_series=drawdown,
            underwater_series=drawdown,
            peak_series=peaks,
        )

    # --------------------------------------------------------
    # Calmar Ratio
    # --------------------------------------------------------

    def calmar_ratio(
        self,
        returns: pd.Series,
    ) -> float:

        ann_return = (
            AnalyticsUtils.annualize_return(
                returns,
                self.config.annualization_factor,
            )
        )

        dd = self.compute_drawdowns(
            returns
        )

        return AnalyticsUtils.safe_divide(
            ann_return,
            abs(
                dd.max_drawdown
            ),
        )

    # --------------------------------------------------------
    # Hit Ratio
    # --------------------------------------------------------

    @staticmethod
    def hit_ratio(
        returns: pd.Series,
    ) -> float:

        positive = (
            returns > 0
        ).sum()

        return float(
            positive
            /
            len(returns)
        )

    # --------------------------------------------------------
    # Distribution Statistics
    # --------------------------------------------------------

    @staticmethod
    def skewness(
        returns: pd.Series,
    ) -> float:

        return float(
            returns.skew()
        )

    # --------------------------------------------------------

    @staticmethod
    def kurtosis(
        returns: pd.Series,
    ) -> float:

        return float(
            returns.kurtosis()
        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    def analyze(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
    ) -> PortfolioStatisticsResult:

        returns = (
            self.validate_returns(
                returns
            )
        )

        dd = self.compute_drawdowns(
            returns
        )

        positives = int(
            (returns > 0).sum()
        )

        negatives = int(
            (returns < 0).sum()
        )

        flats = int(
            (returns == 0).sum()
        )

        return PortfolioStatisticsResult(
            metadata=self.metadata,

            analytics_name=
            "Portfolio Statistics",

            observations=
            len(returns),

            total_return=
            self.total_return(
                returns
            ),

            annualized_return=
            AnalyticsUtils
            .annualize_return(
                returns,
                self.config
                .annualization_factor,
            ),

            annualized_volatility=
            AnalyticsUtils
            .annualize_volatility(
                returns,
                self.config
                .annualization_factor,
            ),

            sharpe_ratio=
            self.sharpe_ratio(
                returns,
                risk_free_rate,
            ),

            sortino_ratio=
            self.sortino_ratio(
                returns,
                risk_free_rate,
            ),

            calmar_ratio=
            self.calmar_ratio(
                returns,
            ),

            max_drawdown=
            dd.max_drawdown,

            average_return=
            self.average_return(
                returns
            ),

            median_return=
            self.median_return(
                returns
            ),

            skewness=
            self.skewness(
                returns
            ),

            kurtosis=
            self.kurtosis(
                returns
            ),

            hit_ratio=
            self.hit_ratio(
                returns
            ),

            positive_periods=
            positives,

            negative_periods=
            negatives,

            flat_periods=
            flats,
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def portfolio_statistics(
    returns: pd.Series,
    *,
    metadata: AnalyticsMetadata,
    risk_free_rate: float = 0.0,
    config: AnalyticsConfig | None = None,
) -> PortfolioStatisticsResult:
    """
    One-line portfolio statistics API.
    """

    analyzer = (
        PortfolioStatisticsAnalyzer(
            metadata=metadata,
            config=config,
        )
    )

    return analyzer.analyze(
        returns,
        risk_free_rate=risk_free_rate,
    )

# ============================================================
# PART 3
# EXPOSURE ANALYTICS
# ============================================================

# ============================================================
# EXPOSURE RESULT
# ============================================================

@dataclass(slots=True)
class ExposureAnalyticsResult(
    AnalyticsResult,
):
    """
    Institutional exposure analytics.

    Portfolio-level exposures.
    """

    gross_exposure: float = 0.0

    net_exposure: float = 0.0

    long_exposure: float = 0.0

    short_exposure: float = 0.0

    cash_exposure: float = 0.0

    leverage: float = 0.0

    sector_exposure: pd.Series | None = None

    industry_exposure: pd.Series | None = None

    country_exposure: pd.Series | None = None

    currency_exposure: pd.Series | None = None

    market_cap_exposure: pd.Series | None = None

    beta_exposure: float = 0.0

    style_exposure: pd.Series | None = None


# ============================================================
# EXPOSURE ANALYZER
# ============================================================

class ExposureAnalyzer(
    BaseAnalytics,
):
    """
    Institutional exposure engine.

    Computes:

        Gross exposure
        Net exposure
        Long exposure
        Short exposure
        Cash exposure
        Sector exposure
        Industry exposure
        Country exposure
        Currency exposure
        Market-cap exposure
        Beta exposure
        Style exposure
    """

    REQUIRED_COLUMNS = [
        "Position_Weight",
    ]

    # --------------------------------------------------------

    def validate_portfolio(
        self,
        portfolio: pd.DataFrame,
    ) -> pd.DataFrame:

        self.validate_dataframe(
            portfolio,
            "portfolio",
        )

        self.validate_columns(
            portfolio,
            self.REQUIRED_COLUMNS,
            "portfolio",
        )

        return portfolio

    # --------------------------------------------------------
    # Gross Exposure
    # --------------------------------------------------------

    @staticmethod
    def gross_exposure(
        weights: pd.Series,
    ) -> float:

        return float(
            weights.abs().sum()
        )

    # --------------------------------------------------------
    # Net Exposure
    # --------------------------------------------------------

    @staticmethod
    def net_exposure(
        weights: pd.Series,
    ) -> float:

        return float(
            weights.sum()
        )

    # --------------------------------------------------------
    # Long Exposure
    # --------------------------------------------------------

    @staticmethod
    def long_exposure(
        weights: pd.Series,
    ) -> float:

        return float(
            weights[
                weights > 0
            ].sum()
        )

    # --------------------------------------------------------
    # Short Exposure
    # --------------------------------------------------------

    @staticmethod
    def short_exposure(
        weights: pd.Series,
    ) -> float:

        return float(
            abs(
                weights[
                    weights < 0
                ].sum()
            )
        )

    # --------------------------------------------------------
    # Cash Exposure
    # --------------------------------------------------------

    @staticmethod
    def cash_exposure(
        weights: pd.Series,
    ) -> float:

        cash = (
            1.0
            -
            weights.sum()
        )

        return float(
            max(
                cash,
                0.0,
            )
        )

    # --------------------------------------------------------
    # Sector Exposure
    # --------------------------------------------------------

    @staticmethod
    def sector_exposure(
        portfolio: pd.DataFrame,
    ) -> pd.Series | None:

        if "Sector" not in portfolio.columns:

            return None

        return (
            portfolio
            .groupby("Sector")
            ["Position_Weight"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

    # --------------------------------------------------------
    # Industry Exposure
    # --------------------------------------------------------

    @staticmethod
    def industry_exposure(
        portfolio: pd.DataFrame,
    ) -> pd.Series | None:

        if "Industry" not in portfolio.columns:

            return None

        return (
            portfolio
            .groupby("Industry")
            ["Position_Weight"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

    # --------------------------------------------------------
    # Country Exposure
    # --------------------------------------------------------

    @staticmethod
    def country_exposure(
        portfolio: pd.DataFrame,
    ) -> pd.Series | None:

        if "Country" not in portfolio.columns:

            return None

        return (
            portfolio
            .groupby("Country")
            ["Position_Weight"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

    # --------------------------------------------------------
    # Currency Exposure
    # --------------------------------------------------------

    @staticmethod
    def currency_exposure(
        portfolio: pd.DataFrame,
    ) -> pd.Series | None:

        if "Currency" not in portfolio.columns:

            return None

        return (
            portfolio
            .groupby("Currency")
            ["Position_Weight"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

    # --------------------------------------------------------
    # Market Cap Exposure
    # --------------------------------------------------------

    @staticmethod
    def market_cap_exposure(
        portfolio: pd.DataFrame,
    ) -> pd.Series | None:

        if "MarketCapBucket" not in portfolio.columns:

            return None

        return (
            portfolio
            .groupby(
                "MarketCapBucket"
            )
            ["Position_Weight"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

    # --------------------------------------------------------
    # Beta Exposure
    # --------------------------------------------------------

    @staticmethod
    def beta_exposure(
        portfolio: pd.DataFrame,
    ) -> float:

        if "Beta" not in portfolio.columns:

            return 0.0

        return float(

            (
                portfolio["Position_Weight"]
                *
                portfolio["Beta"]
            ).sum()

        )

    # --------------------------------------------------------
    # Style Exposure
    # --------------------------------------------------------

    @staticmethod
    def style_exposure(
        portfolio: pd.DataFrame,
    ) -> pd.Series | None:
        """
        Weighted style-factor exposures.

        Example columns:

            Value
            Growth
            Momentum
            Quality
            Size
            LowVol
        """

        style_columns = [

            c

            for c in portfolio.columns

            if c in {
                "Value",
                "Growth",
                "Momentum",
                "Quality",
                "Size",
                "LowVol",
            }
        ]

        if len(style_columns) == 0:

            return None

        exposures = {}

        for col in style_columns:

            exposures[col] = float(

                (
                    portfolio[
                        "Position_Weight"
                    ]
                    *
                    portfolio[col]
                ).sum()

            )

        return pd.Series(
            exposures
        ).sort_values(
            ascending=False
        )

    # --------------------------------------------------------
    # Leverage
    # --------------------------------------------------------

    @staticmethod
    def leverage(
        gross_exposure: float,
    ) -> float:

        return float(
            gross_exposure
        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    def analyze(
        self,
        portfolio: pd.DataFrame,
    ) -> ExposureAnalyticsResult:

        portfolio = (
            self.validate_portfolio(
                portfolio
            )
        )

        weights = (
            portfolio[
                "Position_Weight"
            ]
            .astype(float)
        )

        gross = (
            self.gross_exposure(
                weights
            )
        )

        net = (
            self.net_exposure(
                weights
            )
        )

        long_exp = (
            self.long_exposure(
                weights
            )
        )

        short_exp = (
            self.short_exposure(
                weights
            )
        )

        cash_exp = (
            self.cash_exposure(
                weights
            )
        )

        return ExposureAnalyticsResult(

            metadata=
            self.metadata,

            analytics_name=
            "Exposure Analytics",

            gross_exposure=
            gross,

            net_exposure=
            net,

            long_exposure=
            long_exp,

            short_exposure=
            short_exp,

            cash_exposure=
            cash_exp,

            leverage=
            self.leverage(
                gross
            ),

            sector_exposure=
            self.sector_exposure(
                portfolio
            ),

            industry_exposure=
            self.industry_exposure(
                portfolio
            ),

            country_exposure=
            self.country_exposure(
                portfolio
            ),

            currency_exposure=
            self.currency_exposure(
                portfolio
            ),

            market_cap_exposure=
            self.market_cap_exposure(
                portfolio
            ),

            beta_exposure=
            self.beta_exposure(
                portfolio
            ),

            style_exposure=
            self.style_exposure(
                portfolio
            ),
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def exposure_analytics(
    portfolio: pd.DataFrame,
    *,
    metadata: AnalyticsMetadata,
) -> ExposureAnalyticsResult:
    """
    One-line exposure analytics API.
    """

    analyzer = ExposureAnalyzer(
        metadata=metadata,
    )

    return analyzer.analyze(
        portfolio
    )


# ============================================================
# PART 4
# CONCENTRATION ANALYTICS
# ============================================================

# ============================================================
# CONCENTRATION RESULT
# ============================================================

@dataclass(slots=True)
class ConcentrationAnalyticsResult(
    AnalyticsResult,
):
    """
    Institutional concentration analytics.

    Measures diversification quality
    and concentration risk.
    """

    top_1_weight: float = 0.0

    top_5_weight: float = 0.0

    top_10_weight: float = 0.0

    top_20_weight: float = 0.0

    hhi: float = 0.0

    effective_number_of_holdings: float = 0.0

    diversification_ratio: float = 0.0

    max_sector_weight: float = 0.0

    max_industry_weight: float = 0.0

    sector_hhi: float = 0.0

    industry_hhi: float = 0.0

    concentration_score: float = 0.0


# ============================================================
# CONCENTRATION ANALYZER
# ============================================================

class ConcentrationAnalyzer(
    BaseAnalytics,
):
    """
    Institutional concentration engine.

    Computes:

        Top-N concentration

        HHI

        Effective holdings

        Sector concentration

        Industry concentration

        Diversification metrics
    """

    # --------------------------------------------------------

    def validate_portfolio(
        self,
        portfolio: pd.DataFrame,
    ) -> pd.DataFrame:

        self.validate_dataframe(
            portfolio,
            "portfolio",
        )

        self.validate_columns(
            portfolio,
            ["Position_Weight"],
            "portfolio",
        )

        return portfolio

    # --------------------------------------------------------
    # Top-N concentration
    # --------------------------------------------------------

    @staticmethod
    def top_n_weight(
        weights: pd.Series,
        n: int,
    ) -> float:

        if len(weights) == 0:

            return 0.0

        return float(
            weights
            .abs()
            .nlargest(n)
            .sum()
        )

    # --------------------------------------------------------
    # HHI
    # --------------------------------------------------------

    @staticmethod
    def hhi(
        weights: pd.Series,
    ) -> float:
        """
        Herfindahl-Hirschman Index.

        Range:

            1/N → diversified

            1.0 → fully concentrated
        """

        w = (
            weights.abs()
        )

        total = w.sum()

        if total <= EPSILON:

            return 0.0

        w = w / total

        return float(
            np.square(w)
            .sum()
        )

    # --------------------------------------------------------
    # Effective Holdings
    # --------------------------------------------------------

    @staticmethod
    def effective_holdings(
        weights: pd.Series,
    ) -> float:
        """
        Effective number of holdings.

            1 / HHI
        """

        hhi = (
            ConcentrationAnalyzer
            .hhi(weights)
        )

        if hhi <= EPSILON:

            return 0.0

        return float(
            1.0 / hhi
        )

    # --------------------------------------------------------
    # Sector HHI
    # --------------------------------------------------------

    @staticmethod
    def sector_hhi(
        portfolio: pd.DataFrame,
    ) -> float:

        if "Sector" not in portfolio.columns:

            return 0.0

        exposures = (
            portfolio
            .groupby("Sector")
            ["Position_Weight"]
            .sum()
        )

        return (
            ConcentrationAnalyzer
            .hhi(exposures)
        )

    # --------------------------------------------------------
    # Industry HHI
    # --------------------------------------------------------

    @staticmethod
    def industry_hhi(
        portfolio: pd.DataFrame,
    ) -> float:

        if "Industry" not in portfolio.columns:

            return 0.0

        exposures = (
            portfolio
            .groupby("Industry")
            ["Position_Weight"]
            .sum()
        )

        return (
            ConcentrationAnalyzer
            .hhi(exposures)
        )

    # --------------------------------------------------------
    # Max Sector Exposure
    # --------------------------------------------------------

    @staticmethod
    def max_sector_weight(
        portfolio: pd.DataFrame,
    ) -> float:

        if "Sector" not in portfolio.columns:

            return 0.0

        sector = (
            portfolio
            .groupby("Sector")
            ["Position_Weight"]
            .sum()
            .abs()
        )

        return float(
            sector.max()
        )

    # --------------------------------------------------------
    # Max Industry Exposure
    # --------------------------------------------------------

    @staticmethod
    def max_industry_weight(
        portfolio: pd.DataFrame,
    ) -> float:

        if "Industry" not in portfolio.columns:

            return 0.0

        industry = (
            portfolio
            .groupby("Industry")
            ["Position_Weight"]
            .sum()
            .abs()
        )

        return float(
            industry.max()
        )

    # --------------------------------------------------------
    # Diversification Ratio
    # --------------------------------------------------------

    @staticmethod
    def diversification_ratio(
        weights: pd.Series,
    ) -> float:
        """
        Simple diversification proxy.

            Effective Holdings
            ------------------
            Actual Holdings
        """

        n_actual = len(
            weights[
                weights.abs() > EPSILON
            ]
        )

        if n_actual == 0:

            return 0.0

        n_effective = (
            ConcentrationAnalyzer
            .effective_holdings(
                weights
            )
        )

        return float(
            n_effective
            /
            n_actual
        )

    # --------------------------------------------------------
    # Concentration Score
    # --------------------------------------------------------

    @staticmethod
    def concentration_score(
        weights: pd.Series,
    ) -> float:
        """
        0 = highly diversified
        100 = highly concentrated
        """

        hhi = (
            ConcentrationAnalyzer
            .hhi(weights)
        )

        return float(
            min(
                100.0,
                hhi * 100.0,
            )
        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    def analyze(
        self,
        portfolio: pd.DataFrame,
    ) -> ConcentrationAnalyticsResult:

        portfolio = (
            self.validate_portfolio(
                portfolio
            )
        )

        weights = (
            portfolio[
                "Position_Weight"
            ]
            .astype(float)
        )

        return ConcentrationAnalyticsResult(

            metadata=
            self.metadata,

            analytics_name=
            "Concentration Analytics",

            top_1_weight=
            self.top_n_weight(
                weights,
                1,
            ),

            top_5_weight=
            self.top_n_weight(
                weights,
                5,
            ),

            top_10_weight=
            self.top_n_weight(
                weights,
                10,
            ),

            top_20_weight=
            self.top_n_weight(
                weights,
                20,
            ),

            hhi=
            self.hhi(
                weights,
            ),

            effective_number_of_holdings=
            self.effective_holdings(
                weights,
            ),

            diversification_ratio=
            self.diversification_ratio(
                weights,
            ),

            max_sector_weight=
            self.max_sector_weight(
                portfolio,
            ),

            max_industry_weight=
            self.max_industry_weight(
                portfolio,
            ),

            sector_hhi=
            self.sector_hhi(
                portfolio,
            ),

            industry_hhi=
            self.industry_hhi(
                portfolio,
            ),

            concentration_score=
            self.concentration_score(
                weights,
            ),
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def concentration_analytics(
    portfolio: pd.DataFrame,
    *,
    metadata: AnalyticsMetadata,
) -> ConcentrationAnalyticsResult:
    """
    One-line concentration analytics.
    """

    analyzer = (
        ConcentrationAnalyzer(
            metadata=metadata,
        )
    )

    return analyzer.analyze(
        portfolio
    )


# ============================================================
# PART 5
# RISK ANALYTICS
# ============================================================

# ============================================================
# RISK ANALYTICS RESULT
# ============================================================

@dataclass(slots=True)
class RiskAnalyticsResult(
    AnalyticsResult,
):
    """
    Institutional portfolio risk report.
    """

    portfolio_volatility: float = 0.0

    annualized_volatility: float = 0.0

    tracking_error: float = 0.0

    portfolio_beta: float = 0.0

    information_ratio: float = 0.0

    value_at_risk: float = 0.0

    expected_shortfall: float = 0.0

    downside_deviation: float = 0.0

    factor_risk: float = 0.0

    specific_risk: float = 0.0

    diversification_ratio: float = 0.0

    risk_concentration: float = 0.0

    marginal_risk_contribution: pd.Series | None = None

    component_risk_contribution: pd.Series | None = None

    factor_exposures: pd.Series | None = None

    factor_contributions: pd.Series | None = None


# ============================================================
# RISK ANALYZER
# ============================================================

class RiskAnalyticsAnalyzer(
    BaseAnalytics,
):
    """
    Institutional risk analytics layer.

    Consumes:

        returns
        benchmark returns
        covariance matrix
        factor model

    Produces:

        volatility
        VaR
        CVaR
        beta
        TE
        risk contributions
        factor risk
    """

    def __init__(
        self,
        metadata: AnalyticsMetadata,
        config: AnalyticsConfig | None = None,
    ) -> None:

        super().__init__(metadata)

        self.config = (
            config
            if config is not None
            else AnalyticsConfig()
        )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    def portfolio_volatility(
        self,
        returns: pd.Series,
    ) -> float:

        returns = returns.dropna()

        if len(returns) < 2:

            return 0.0

        return float(
            returns.std()
        )

    # --------------------------------------------------------

    def annualized_volatility(
        self,
        returns: pd.Series,
    ) -> float:

        return float(
            returns.std()
            *
            np.sqrt(
                self.config.annualization_factor
            )
        )

    # --------------------------------------------------------
    # Tracking Error
    # --------------------------------------------------------

    def tracking_error(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:

        aligned = pd.concat(
            [
                portfolio_returns,
                benchmark_returns,
            ],
            axis=1,
        ).dropna()

        if len(aligned) < 2:

            return 0.0

        active = (
            aligned.iloc[:, 0]
            -
            aligned.iloc[:, 1]
        )

        return float(
            active.std()
            *
            np.sqrt(
                self.config.annualization_factor
            )
        )

    # --------------------------------------------------------
    # Information Ratio
    # --------------------------------------------------------

    def information_ratio(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:

        aligned = pd.concat(
            [
                portfolio_returns,
                benchmark_returns,
            ],
            axis=1,
        ).dropna()

        if len(aligned) < 2:

            return 0.0

        active = (
            aligned.iloc[:, 0]
            -
            aligned.iloc[:, 1]
        )

        active_return = float(
            active.mean()
            *
            self.config.annualization_factor
        )

        te = self.tracking_error(
            portfolio_returns,
            benchmark_returns,
        )

        return AnalyticsUtils.safe_divide(
            active_return,
            te,
        )

    # --------------------------------------------------------
    # Beta
    # --------------------------------------------------------

    def portfolio_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:

        aligned = pd.concat(
            [
                portfolio_returns,
                benchmark_returns,
            ],
            axis=1,
        ).dropna()

        if len(aligned) < 2:

            return 0.0

        p = aligned.iloc[:, 0]
        b = aligned.iloc[:, 1]

        variance = b.var()

        if abs(variance) < EPSILON:

            return 0.0

        beta = (
            np.cov(
                p,
                b,
            )[0, 1]
            /
            variance
        )

        return float(beta)

    # --------------------------------------------------------
    # Downside Deviation
    # --------------------------------------------------------

    def downside_deviation(
        self,
        returns: pd.Series,
    ) -> float:

        downside = (
            returns[
                returns < 0
            ]
        )

        if len(downside) == 0:

            return 0.0

        return float(
            downside.std()
            *
            np.sqrt(
                self.config.annualization_factor
            )
        )

    # --------------------------------------------------------
    # Historical VaR
    # --------------------------------------------------------

    def value_at_risk(
        self,
        returns: pd.Series,
        confidence: float | None = None,
    ) -> float:

        confidence = (
            confidence
            if confidence is not None
            else self.config.confidence_level
        )

        alpha = (
            1.0 - confidence
        )

        return float(
            np.percentile(
                returns,
                alpha * 100,
            )
        )

    # --------------------------------------------------------
    # Historical CVaR
    # --------------------------------------------------------

    def expected_shortfall(
        self,
        returns: pd.Series,
        confidence: float | None = None,
    ) -> float:

        confidence = (
            confidence
            if confidence is not None
            else self.config.confidence_level
        )

        var = self.value_at_risk(
            returns,
            confidence,
        )

        tail = (
            returns[
                returns <= var
            ]
        )

        if len(tail) == 0:

            return var

        return float(
            tail.mean()
        )

    # --------------------------------------------------------
    # Risk Contribution
    # --------------------------------------------------------

    def component_risk_contribution(
        self,
        weights: pd.Series,
        covariance_matrix: pd.DataFrame,
    ) -> pd.Series:

        w = weights.values

        cov = covariance_matrix.values

        portfolio_vol = np.sqrt(
            w.T @ cov @ w
        )

        if portfolio_vol <= EPSILON:

            return pd.Series(
                0.0,
                index=weights.index,
            )

        mrc = (
            cov @ w
        ) / portfolio_vol

        crc = (
            w * mrc
        )

        return pd.Series(
            crc,
            index=weights.index,
        )

    # --------------------------------------------------------

    def marginal_risk_contribution(
        self,
        weights: pd.Series,
        covariance_matrix: pd.DataFrame,
    ) -> pd.Series:

        w = weights.values

        cov = covariance_matrix.values

        portfolio_vol = np.sqrt(
            w.T @ cov @ w
        )

        if portfolio_vol <= EPSILON:

            return pd.Series(
                0.0,
                index=weights.index,
            )

        mrc = (
            cov @ w
        ) / portfolio_vol

        return pd.Series(
            mrc,
            index=weights.index,
        )

    # --------------------------------------------------------
    # Factor Risk
    # --------------------------------------------------------

    def factor_risk(
        self,
        factor_contributions: pd.Series | None,
    ) -> float:

        if factor_contributions is None:

            return 0.0

        return float(
            np.sqrt(
                np.square(
                    factor_contributions
                ).sum()
            )
        )

    # --------------------------------------------------------
    # Diversification Ratio
    # --------------------------------------------------------

    def diversification_ratio(
        self,
        weights: pd.Series,
        covariance_matrix: pd.DataFrame,
    ) -> float:

        vols = np.sqrt(
            np.diag(
                covariance_matrix.values
            )
        )

        weighted_vol = (
            np.abs(
                weights.values
            )
            * vols
        ).sum()

        portfolio_vol = np.sqrt(
            weights.values.T
            @ covariance_matrix.values
            @ weights.values
        )

        return AnalyticsUtils.safe_divide(
            weighted_vol,
            portfolio_vol,
        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    def analyze(
        self,
        *,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
        weights: pd.Series | None = None,
        covariance_matrix: pd.DataFrame | None = None,
        factor_exposures: pd.Series | None = None,
        factor_contributions: pd.Series | None = None,
    ) -> RiskAnalyticsResult:

        volatility = (
            self.portfolio_volatility(
                portfolio_returns
            )
        )

        annual_vol = (
            self.annualized_volatility(
                portfolio_returns
            )
        )

        beta = 0.0
        te = 0.0
        ir = 0.0

        if benchmark_returns is not None:

            beta = (
                self.portfolio_beta(
                    portfolio_returns,
                    benchmark_returns,
                )
            )

            te = (
                self.tracking_error(
                    portfolio_returns,
                    benchmark_returns,
                )
            )

            ir = (
                self.information_ratio(
                    portfolio_returns,
                    benchmark_returns,
                )
            )

        mrc = None
        crc = None
        dr = 0.0

        if (
            weights is not None
            and covariance_matrix is not None
        ):

            mrc = (
                self.marginal_risk_contribution(
                    weights,
                    covariance_matrix,
                )
            )

            crc = (
                self.component_risk_contribution(
                    weights,
                    covariance_matrix,
                )
            )

            dr = (
                self.diversification_ratio(
                    weights,
                    covariance_matrix,
                )
            )

        return RiskAnalyticsResult(

            metadata=self.metadata,

            analytics_name=
            "Risk Analytics",

            portfolio_volatility=
            volatility,

            annualized_volatility=
            annual_vol,

            tracking_error=
            te,

            portfolio_beta=
            beta,

            information_ratio=
            ir,

            value_at_risk=
            self.value_at_risk(
                portfolio_returns
            ),

            expected_shortfall=
            self.expected_shortfall(
                portfolio_returns
            ),

            downside_deviation=
            self.downside_deviation(
                portfolio_returns
            ),

            factor_risk=
            self.factor_risk(
                factor_contributions
            ),

            specific_risk=0.0,

            diversification_ratio=
            dr,

            risk_concentration=
            float(
                crc.max()
            )
            if crc is not None
            else 0.0,

            marginal_risk_contribution=
            mrc,

            component_risk_contribution=
            crc,

            factor_exposures=
            factor_exposures,

            factor_contributions=
            factor_contributions,
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def risk_analytics(
    *,
    metadata: AnalyticsMetadata,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    weights: pd.Series | None = None,
    covariance_matrix: pd.DataFrame | None = None,
    factor_exposures: pd.Series | None = None,
    factor_contributions: pd.Series | None = None,
) -> RiskAnalyticsResult:

    analyzer = RiskAnalyticsAnalyzer(
        metadata=metadata,
    )

    return analyzer.analyze(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        weights=weights,
        covariance_matrix=covariance_matrix,
        factor_exposures=factor_exposures,
        factor_contributions=factor_contributions,
    )


# ============================================================
# PART 6
# FACTOR ANALYTICS
# ============================================================

# ============================================================
# FACTOR ANALYTICS RESULT
# ============================================================

@dataclass(slots=True)
class FactorAnalyticsResult(
    AnalyticsResult,
):
    """
    Institutional factor report.
    """

    factor_exposures: pd.Series | None = None

    factor_returns: pd.Series | None = None

    factor_return_contributions: pd.Series | None = None

    factor_risk_contributions: pd.Series | None = None

    factor_information_ratios: pd.Series | None = None

    factor_t_statistics: pd.Series | None = None

    factor_crowding_scores: pd.Series | None = None

    factor_diversification_score: float = 0.0

    total_factor_return: float = 0.0

    total_factor_risk: float = 0.0

    dominant_factor: str | None = None


# ============================================================
# FACTOR ANALYZER
# ============================================================

class FactorAnalyticsAnalyzer(
    BaseAnalytics,
):
    """
    Institutional factor analytics engine.

    Produces:

        Factor exposures
        Factor returns
        Factor attribution
        Factor risk attribution
        Factor crowding
        Factor diversification
    """

    # --------------------------------------------------------
    # Exposure Normalization
    # --------------------------------------------------------

    @staticmethod
    def normalize_exposures(
        exposures: pd.Series,
    ) -> pd.Series:

        if exposures is None:

            return pd.Series(dtype=float)

        exposures = exposures.fillna(0.0)

        total = exposures.abs().sum()

        if total <= EPSILON:

            return exposures * 0.0

        return exposures / total

    # --------------------------------------------------------
    # Factor Return Contribution
    # --------------------------------------------------------

    @staticmethod
    def factor_return_contribution(
        exposures: pd.Series,
        factor_returns: pd.Series,
    ) -> pd.Series:

        common = (
            exposures.index
            .intersection(
                factor_returns.index
            )
        )

        if len(common) == 0:

            return pd.Series(dtype=float)

        return (
            exposures[common]
            *
            factor_returns[common]
        )

    # --------------------------------------------------------
    # Factor Risk Contribution
    # --------------------------------------------------------

    @staticmethod
    def factor_risk_contribution(
        exposures: pd.Series,
        factor_covariance: pd.DataFrame,
    ) -> pd.Series:

        common = (
            exposures.index
            .intersection(
                factor_covariance.index
            )
        )

        if len(common) == 0:

            return pd.Series(dtype=float)

        x = (
            exposures[common]
            .values
        )

        cov = (
            factor_covariance
            .loc[
                common,
                common,
            ]
            .values
        )

        portfolio_factor_vol = np.sqrt(
            x.T @ cov @ x
        )

        if portfolio_factor_vol <= EPSILON:

            return pd.Series(
                0.0,
                index=common,
            )

        mrc = (
            cov @ x
        ) / portfolio_factor_vol

        crc = (
            x * mrc
        )

        return pd.Series(
            crc,
            index=common,
        )

    # --------------------------------------------------------
    # Information Ratio Per Factor
    # --------------------------------------------------------

    @staticmethod
    def factor_information_ratios(
        factor_return_history: pd.DataFrame,
    ) -> pd.Series:

        if factor_return_history.empty:

            return pd.Series(dtype=float)

        annual_return = (
            factor_return_history.mean()
            * TRADING_DAYS
        )

        annual_vol = (
            factor_return_history.std()
            * np.sqrt(
                TRADING_DAYS
            )
        )

        ir = (
            annual_return
            /
            annual_vol.replace(
                0,
                np.nan,
            )
        )

        return ir.fillna(0.0)

    # --------------------------------------------------------
    # T Statistics
    # --------------------------------------------------------

    @staticmethod
    def factor_t_statistics(
        factor_return_history: pd.DataFrame,
    ) -> pd.Series:

        if factor_return_history.empty:

            return pd.Series(dtype=float)

        mean = (
            factor_return_history.mean()
        )

        std = (
            factor_return_history.std()
        )

        n = len(
            factor_return_history
        )

        tstats = (
            mean
            /
            (
                std
                /
                np.sqrt(n)
            ).replace(
                0,
                np.nan,
            )
        )

        return tstats.fillna(0.0)

    # --------------------------------------------------------
    # Factor Crowding
    # --------------------------------------------------------

    @staticmethod
    def factor_crowding(
        exposures: pd.Series,
    ) -> pd.Series:
        """
        Simple crowding proxy.

        Higher absolute exposure
        implies greater crowding.
        """

        if exposures is None:

            return pd.Series(dtype=float)

        crowding = (
            exposures.abs()
            /
            max(
                exposures.abs().max(),
                EPSILON,
            )
        )

        return crowding

    # --------------------------------------------------------
    # Diversification Score
    # --------------------------------------------------------

    @staticmethod
    def factor_diversification_score(
        exposures: pd.Series,
    ) -> float:

        if exposures is None:

            return 0.0

        x = exposures.abs()

        total = x.sum()

        if total <= EPSILON:

            return 0.0

        x = x / total

        hhi = np.square(
            x
        ).sum()

        return float(
            1.0 - hhi
        )

    # --------------------------------------------------------
    # Dominant Factor
    # --------------------------------------------------------

    @staticmethod
    def dominant_factor(
        exposures: pd.Series,
    ) -> str | None:

        if (
            exposures is None
            or len(exposures) == 0
        ):
            return None

        return str(
            exposures.abs()
            .idxmax()
        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    def analyze(
        self,
        *,
        factor_exposures: pd.Series,
        factor_returns: pd.Series | None = None,
        factor_covariance: pd.DataFrame | None = None,
        factor_return_history: pd.DataFrame | None = None,
    ) -> FactorAnalyticsResult:

        factor_exposures = (
            self.normalize_exposures(
                factor_exposures
            )
        )

        return_contrib = None
        risk_contrib = None
        ir = None
        tstats = None

        if factor_returns is not None:

            return_contrib = (
                self.factor_return_contribution(
                    factor_exposures,
                    factor_returns,
                )
            )

        if (
            factor_covariance is not None
        ):

            risk_contrib = (
                self.factor_risk_contribution(
                    factor_exposures,
                    factor_covariance,
                )
            )

        if (
            factor_return_history
            is not None
        ):

            ir = (
                self.factor_information_ratios(
                    factor_return_history
                )
            )

            tstats = (
                self.factor_t_statistics(
                    factor_return_history
                )
            )

        crowding = (
            self.factor_crowding(
                factor_exposures
            )
        )

        return FactorAnalyticsResult(

            metadata=
            self.metadata,

            analytics_name=
            "Factor Analytics",

            factor_exposures=
            factor_exposures,

            factor_returns=
            factor_returns,

            factor_return_contributions=
            return_contrib,

            factor_risk_contributions=
            risk_contrib,

            factor_information_ratios=
            ir,

            factor_t_statistics=
            tstats,

            factor_crowding_scores=
            crowding,

            factor_diversification_score=
            self.factor_diversification_score(
                factor_exposures
            ),

            total_factor_return=
            float(
                return_contrib.sum()
            )
            if return_contrib is not None
            else 0.0,

            total_factor_risk=
            float(
                np.sqrt(
                    np.square(
                        risk_contrib
                    ).sum()
                )
            )
            if risk_contrib is not None
            else 0.0,

            dominant_factor=
            self.dominant_factor(
                factor_exposures
            ),
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def factor_analytics(
    *,
    metadata: AnalyticsMetadata,
    factor_exposures: pd.Series,
    factor_returns: pd.Series | None = None,
    factor_covariance: pd.DataFrame | None = None,
    factor_return_history: pd.DataFrame | None = None,
) -> FactorAnalyticsResult:

    analyzer = (
        FactorAnalyticsAnalyzer(
            metadata=metadata,
        )
    )

    return analyzer.analyze(
        factor_exposures=factor_exposures,
        factor_returns=factor_returns,
        factor_covariance=factor_covariance,
        factor_return_history=factor_return_history,
    )

# ============================================================
# PART 7
# PERFORMANCE ANALYTICS
# ============================================================

# ============================================================
# PERFORMANCE RESULT
# ============================================================

@dataclass(slots=True)
class PerformanceAnalyticsResult(
    AnalyticsResult,
):
    """
    Institutional performance report.
    """

    total_return: float = 0.0

    annualized_return: float = 0.0

    benchmark_return: float = 0.0

    annualized_benchmark_return: float = 0.0

    active_return: float = 0.0

    annualized_active_return: float = 0.0

    alpha: float = 0.0

    beta: float = 0.0

    information_ratio: float = 0.0

    upside_capture_ratio: float = 0.0

    downside_capture_ratio: float = 0.0

    hit_ratio_vs_benchmark: float = 0.0

    rolling_sharpe: pd.Series | None = None

    rolling_volatility: pd.Series | None = None

    rolling_drawdown: pd.Series | None = None

    active_return_series: pd.Series | None = None


# ============================================================
# PERFORMANCE ANALYZER
# ============================================================

class PerformanceAnalyticsAnalyzer(
    BaseAnalytics,
):
    """
    Institutional performance analytics.

    Produces:

        Alpha
        Beta
        Active Return
        Information Ratio
        Capture Ratios
        Rolling Metrics
    """

    DEFAULT_ROLLING_WINDOW = 63

    # --------------------------------------------------------
    # Total Return
    # --------------------------------------------------------

    @staticmethod
    def total_return(
        returns: pd.Series,
    ) -> float:

        return float(
            (1.0 + returns).prod()
            - 1.0
        )

    # --------------------------------------------------------
    # Annualized Return
    # --------------------------------------------------------

    @staticmethod
    def annualized_return(
        returns: pd.Series,
    ) -> float:

        return AnalyticsUtils.annualize_return(
            returns,
            TRADING_DAYS,
        )

    # --------------------------------------------------------
    # Active Return
    # --------------------------------------------------------

    @staticmethod
    def active_return_series(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> pd.Series:

        aligned = pd.concat(
            [
                portfolio_returns,
                benchmark_returns,
            ],
            axis=1,
        ).dropna()

        if len(aligned) == 0:

            return pd.Series(dtype=float)

        return (
            aligned.iloc[:, 0]
            -
            aligned.iloc[:, 1]
        )

    # --------------------------------------------------------
    # Alpha Beta
    # --------------------------------------------------------

    @staticmethod
    def alpha_beta(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> tuple[float, float]:

        aligned = pd.concat(
            [
                portfolio_returns,
                benchmark_returns,
            ],
            axis=1,
        ).dropna()

        if len(aligned) < 2:

            return (
                0.0,
                0.0,
            )

        y = aligned.iloc[:, 0]
        x = aligned.iloc[:, 1]

        variance = x.var()

        if variance <= EPSILON:

            return (
                0.0,
                0.0,
            )

        beta = (
            np.cov(
                y,
                x,
            )[0, 1]
            /
            variance
        )

        alpha = (
            y.mean()
            -
            beta
            * x.mean()
        )

        alpha *= TRADING_DAYS

        return (
            float(alpha),
            float(beta),
        )

    # --------------------------------------------------------
    # Information Ratio
    # --------------------------------------------------------

    @staticmethod
    def information_ratio(
        active_returns: pd.Series,
    ) -> float:

        if len(active_returns) < 2:

            return 0.0

        active_mean = (
            active_returns.mean()
            * TRADING_DAYS
        )

        active_vol = (
            active_returns.std()
            *
            np.sqrt(
                TRADING_DAYS
            )
        )

        return AnalyticsUtils.safe_divide(
            active_mean,
            active_vol,
        )

    # --------------------------------------------------------
    # Upside Capture
    # --------------------------------------------------------

    @staticmethod
    def upside_capture_ratio(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:

        aligned = pd.concat(
            [
                portfolio_returns,
                benchmark_returns,
            ],
            axis=1,
        ).dropna()

        if len(aligned) == 0:

            return 0.0

        up = aligned[
            aligned.iloc[:, 1] > 0
        ]

        if len(up) == 0:

            return 0.0

        p = (
            (1 + up.iloc[:, 0]).prod()
            - 1
        )

        b = (
            (1 + up.iloc[:, 1]).prod()
            - 1
        )

        return AnalyticsUtils.safe_divide(
            p,
            b,
        )

    # --------------------------------------------------------
    # Downside Capture
    # --------------------------------------------------------

    @staticmethod
    def downside_capture_ratio(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:

        aligned = pd.concat(
            [
                portfolio_returns,
                benchmark_returns,
            ],
            axis=1,
        ).dropna()

        if len(aligned) == 0:

            return 0.0

        down = aligned[
            aligned.iloc[:, 1] < 0
        ]

        if len(down) == 0:

            return 0.0

        p = (
            (1 + down.iloc[:, 0]).prod()
            - 1
        )

        b = (
            (1 + down.iloc[:, 1]).prod()
            - 1
        )

        return AnalyticsUtils.safe_divide(
            p,
            b,
        )

    # --------------------------------------------------------
    # Hit Ratio vs Benchmark
    # --------------------------------------------------------

    @staticmethod
    def hit_ratio_vs_benchmark(
        active_returns: pd.Series,
    ) -> float:

        if len(active_returns) == 0:

            return 0.0

        return float(
            (
                active_returns > 0
            ).mean()
        )

    # --------------------------------------------------------
    # Rolling Sharpe
    # --------------------------------------------------------

    @staticmethod
    def rolling_sharpe(
        returns: pd.Series,
        window: int = 63,
    ) -> pd.Series:

        mean = (
            returns
            .rolling(window)
            .mean()
            * TRADING_DAYS
        )

        vol = (
            returns
            .rolling(window)
            .std()
            *
            np.sqrt(
                TRADING_DAYS
            )
        )

        return (
            mean
            /
            vol.replace(
                0,
                np.nan,
            )
        )

    # --------------------------------------------------------
    # Rolling Volatility
    # --------------------------------------------------------

    @staticmethod
    def rolling_volatility(
        returns: pd.Series,
        window: int = 63,
    ) -> pd.Series:

        return (
            returns
            .rolling(window)
            .std()
            *
            np.sqrt(
                TRADING_DAYS
            )
        )

    # --------------------------------------------------------
    # Rolling Drawdown
    # --------------------------------------------------------

    @staticmethod
    def rolling_drawdown(
        returns: pd.Series,
    ) -> pd.Series:

        cumulative = (
            (1 + returns)
            .cumprod()
        )

        peak = (
            cumulative.cummax()
        )

        return (
            cumulative
            /
            peak
            - 1.0
        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    def analyze(
        self,
        *,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        rolling_window: int = 63,
    ) -> PerformanceAnalyticsResult:

        active = (
            self.active_return_series(
                portfolio_returns,
                benchmark_returns,
            )
        )

        alpha, beta = (
            self.alpha_beta(
                portfolio_returns,
                benchmark_returns,
            )
        )

        return PerformanceAnalyticsResult(

            metadata=self.metadata,

            analytics_name=
            "Performance Analytics",

            total_return=
            self.total_return(
                portfolio_returns
            ),

            annualized_return=
            self.annualized_return(
                portfolio_returns
            ),

            benchmark_return=
            self.total_return(
                benchmark_returns
            ),

            annualized_benchmark_return=
            self.annualized_return(
                benchmark_returns
            ),

            active_return=
            self.total_return(
                active
            ),

            annualized_active_return=
            self.annualized_return(
                active
            ),

            alpha=
            alpha,

            beta=
            beta,

            information_ratio=
            self.information_ratio(
                active
            ),

            upside_capture_ratio=
            self.upside_capture_ratio(
                portfolio_returns,
                benchmark_returns,
            ),

            downside_capture_ratio=
            self.downside_capture_ratio(
                portfolio_returns,
                benchmark_returns,
            ),

            hit_ratio_vs_benchmark=
            self.hit_ratio_vs_benchmark(
                active
            ),

            rolling_sharpe=
            self.rolling_sharpe(
                portfolio_returns,
                rolling_window,
            ),

            rolling_volatility=
            self.rolling_volatility(
                portfolio_returns,
                rolling_window,
            ),

            rolling_drawdown=
            self.rolling_drawdown(
                portfolio_returns,
            ),

            active_return_series=
            active,
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def performance_analytics(
    *,
    metadata: AnalyticsMetadata,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    rolling_window: int = 63,
) -> PerformanceAnalyticsResult:

    analyzer = (
        PerformanceAnalyticsAnalyzer(
            metadata=metadata,
        )
    )

    return analyzer.analyze(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        rolling_window=rolling_window,
    )


# ============================================================
# PART 8
# REBALANCE ANALYTICS
# ============================================================

# ============================================================
# REBALANCE ANALYTICS RESULT
# ============================================================

@dataclass(slots=True)
class RebalanceAnalyticsResult(
    AnalyticsResult,
):
    """
    Institutional rebalance report.
    """

    rebalance_count: int = 0

    average_turnover: float = 0.0

    median_turnover: float = 0.0

    max_turnover: float = 0.0

    min_turnover: float = 0.0

    annualized_turnover: float = 0.0

    average_trade_count: float = 0.0

    total_trade_count: int = 0

    average_buys: float = 0.0

    average_sells: float = 0.0

    buy_sell_ratio: float = 0.0

    average_drift_before_rebalance: float = 0.0

    average_drift_after_rebalance: float = 0.0

    drift_reduction: float = 0.0

    average_cost_per_rebalance: float = 0.0

    total_rebalance_cost: float = 0.0

    turnover_series: pd.Series | None = None

    trade_count_series: pd.Series | None = None

    cost_series: pd.Series | None = None

    drift_before_series: pd.Series | None = None

    drift_after_series: pd.Series | None = None


# ============================================================
# REBALANCE ANALYZER
# ============================================================

class RebalanceAnalyticsAnalyzer(
    BaseAnalytics,
):
    """
    Institutional rebalance analytics.

    Supports:

        Turnover analytics
        Trade analytics
        Drift analytics
        Cost analytics
        Rebalance efficiency
    """

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    @staticmethod
    def _safe_mean(
        values: pd.Series | list[float],
    ) -> float:

        if len(values) == 0:

            return 0.0

        return float(
            np.mean(values)
        )

    # --------------------------------------------------------
    # Turnover
    # --------------------------------------------------------

    @staticmethod
    def annualized_turnover(
        turnover_series: pd.Series,
        rebalance_frequency_per_year: int,
    ) -> float:

        if len(turnover_series) == 0:

            return 0.0

        return float(
            turnover_series.mean()
            *
            rebalance_frequency_per_year
        )

    # --------------------------------------------------------
    # Drift Reduction
    # --------------------------------------------------------

    @staticmethod
    def drift_reduction(
        drift_before: pd.Series,
        drift_after: pd.Series,
    ) -> float:

        if (
            len(drift_before) == 0
            or
            len(drift_after) == 0
        ):
            return 0.0

        before = float(
            drift_before.mean()
        )

        after = float(
            drift_after.mean()
        )

        return float(
            before - after
        )

    # --------------------------------------------------------
    # Buy/Sell Ratio
    # --------------------------------------------------------

    @staticmethod
    def buy_sell_ratio(
        avg_buys: float,
        avg_sells: float,
    ) -> float:

        return AnalyticsUtils.safe_divide(
            avg_buys,
            avg_sells,
        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    def analyze(
        self,
        *,
        turnover_series: pd.Series,
        trade_count_series: pd.Series,
        buy_count_series: pd.Series | None = None,
        sell_count_series: pd.Series | None = None,
        cost_series: pd.Series | None = None,
        drift_before_series: pd.Series | None = None,
        drift_after_series: pd.Series | None = None,
        rebalance_frequency_per_year: int = 12,
    ) -> RebalanceAnalyticsResult:

        turnover_series = (
            turnover_series.fillna(0.0)
        )

        trade_count_series = (
            trade_count_series.fillna(0.0)
        )

        if buy_count_series is None:

            buy_count_series = pd.Series(
                dtype=float
            )

        if sell_count_series is None:

            sell_count_series = pd.Series(
                dtype=float
            )

        if cost_series is None:

            cost_series = pd.Series(
                dtype=float
            )

        if drift_before_series is None:

            drift_before_series = pd.Series(
                dtype=float
            )

        if drift_after_series is None:

            drift_after_series = pd.Series(
                dtype=float
            )

        avg_buys = (
            self._safe_mean(
                buy_count_series
            )
        )

        avg_sells = (
            self._safe_mean(
                sell_count_series
            )
        )

        return RebalanceAnalyticsResult(

            metadata=
            self.metadata,

            analytics_name=
            "Rebalance Analytics",

            rebalance_count=
            len(turnover_series),

            average_turnover=
            self._safe_mean(
                turnover_series
            ),

            median_turnover=
            float(
                turnover_series.median()
            )
            if len(turnover_series)
            else 0.0,

            max_turnover=
            float(
                turnover_series.max()
            )
            if len(turnover_series)
            else 0.0,

            min_turnover=
            float(
                turnover_series.min()
            )
            if len(turnover_series)
            else 0.0,

            annualized_turnover=
            self.annualized_turnover(
                turnover_series,
                rebalance_frequency_per_year,
            ),

            average_trade_count=
            self._safe_mean(
                trade_count_series
            ),

            total_trade_count=
            int(
                trade_count_series.sum()
            ),

            average_buys=
            avg_buys,

            average_sells=
            avg_sells,

            buy_sell_ratio=
            self.buy_sell_ratio(
                avg_buys,
                avg_sells,
            ),

            average_drift_before_rebalance=
            self._safe_mean(
                drift_before_series
            ),

            average_drift_after_rebalance=
            self._safe_mean(
                drift_after_series
            ),

            drift_reduction=
            self.drift_reduction(
                drift_before_series,
                drift_after_series,
            ),

            average_cost_per_rebalance=
            self._safe_mean(
                cost_series
            ),

            total_rebalance_cost=
            float(
                cost_series.sum()
            ),

            turnover_series=
            turnover_series,

            trade_count_series=
            trade_count_series,

            cost_series=
            cost_series,

            drift_before_series=
            drift_before_series,

            drift_after_series=
            drift_after_series,
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def rebalance_analytics(
    *,
    metadata: AnalyticsMetadata,
    turnover_series: pd.Series,
    trade_count_series: pd.Series,
    buy_count_series: pd.Series | None = None,
    sell_count_series: pd.Series | None = None,
    cost_series: pd.Series | None = None,
    drift_before_series: pd.Series | None = None,
    drift_after_series: pd.Series | None = None,
    rebalance_frequency_per_year: int = 12,
) -> RebalanceAnalyticsResult:

    analyzer = (
        RebalanceAnalyticsAnalyzer(
            metadata=metadata,
        )
    )

    return analyzer.analyze(
        turnover_series=turnover_series,
        trade_count_series=trade_count_series,
        buy_count_series=buy_count_series,
        sell_count_series=sell_count_series,
        cost_series=cost_series,
        drift_before_series=drift_before_series,
        drift_after_series=drift_after_series,
        rebalance_frequency_per_year=
        rebalance_frequency_per_year,
    )
# ============================================================
# PART 9
# EXECUTION ANALYTICS
# ============================================================

# ============================================================
# EXECUTION ANALYTICS RESULT
# ============================================================

@dataclass(slots=True)
class ExecutionAnalyticsResult(
    AnalyticsResult,
):
    """
    Institutional execution report.
    """

    total_orders: int = 0

    filled_orders: int = 0

    partially_filled_orders: int = 0

    cancelled_orders: int = 0

    rejected_orders: int = 0

    fill_rate: float = 0.0

    completion_rate: float = 0.0

    average_fill_ratio: float = 0.0

    average_slippage_bps: float = 0.0

    median_slippage_bps: float = 0.0

    max_slippage_bps: float = 0.0

    min_slippage_bps: float = 0.0

    average_market_impact_bps: float = 0.0

    total_transaction_cost: float = 0.0

    average_transaction_cost: float = 0.0

    total_notional_traded: float = 0.0

    average_participation_rate: float = 0.0

    execution_shortfall: float = 0.0

    broker_statistics: pd.DataFrame | None = None

    slippage_series: pd.Series | None = None

    impact_series: pd.Series | None = None

    cost_series: pd.Series | None = None


# ============================================================
# EXECUTION ANALYZER
# ============================================================

class ExecutionAnalyticsAnalyzer(
    BaseAnalytics,
):
    """
    Institutional execution analytics.

    Produces:

        Fill quality
        Slippage analytics
        Market impact analytics
        Cost analytics
        Participation analytics
        Broker diagnostics
    """

    # --------------------------------------------------------
    # Fill Rate
    # --------------------------------------------------------

    @staticmethod
    def fill_rate(
        filled_orders: int,
        total_orders: int,
    ) -> float:

        return AnalyticsUtils.safe_divide(
            filled_orders,
            total_orders,
        )

    # --------------------------------------------------------
    # Completion Rate
    # --------------------------------------------------------

    @staticmethod
    def completion_rate(
        completed_orders: int,
        total_orders: int,
    ) -> float:

        return AnalyticsUtils.safe_divide(
            completed_orders,
            total_orders,
        )

    # --------------------------------------------------------
    # Average Fill Ratio
    # --------------------------------------------------------

    @staticmethod
    def average_fill_ratio(
        fill_ratios: pd.Series,
    ) -> float:

        if len(fill_ratios) == 0:

            return 0.0

        return float(
            fill_ratios.mean()
        )

    # --------------------------------------------------------
    # Execution Shortfall
    # --------------------------------------------------------

    @staticmethod
    def execution_shortfall(
        decision_price: pd.Series,
        execution_price: pd.Series,
        quantity: pd.Series,
    ) -> float:

        if len(decision_price) == 0:

            return 0.0

        shortfall = (
            (
                execution_price
                -
                decision_price
            )
            *
            quantity.abs()
        )

        return float(
            shortfall.sum()
        )

    # --------------------------------------------------------
    # Broker Statistics
    # --------------------------------------------------------

    @staticmethod
    def broker_statistics(
        execution_df: pd.DataFrame,
    ) -> pd.DataFrame | None:

        if (
            "Broker" not in execution_df.columns
        ):
            return None

        grouped = (
            execution_df
            .groupby("Broker")
            .agg(
                Orders=("Broker", "count"),
                AvgSlippage=("SlippageBps", "mean"),
                AvgImpact=("ImpactBps", "mean"),
                TotalCost=("TransactionCost", "sum"),
                FillRatio=("FillRatio", "mean"),
            )
        )

        return grouped.sort_values(
            "Orders",
            ascending=False,
        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    def analyze(
        self,
        execution_df: pd.DataFrame,
    ) -> ExecutionAnalyticsResult:

        self.validate_dataframe(
            execution_df,
            "execution_df",
        )

        total_orders = len(execution_df)

        # --------------------------------------------------------
        # Detect execution data model
        # --------------------------------------------------------

        has_actual_execution_status = (
            "OrderStatus" in execution_df.columns
        )

        has_estimated_execution_data = (
            "Slippage" in execution_df.columns
            or "MarketImpact" in execution_df.columns
            or "TransactionCost" in execution_df.columns
        )

        # --------------------------------------------------------
        # Actual fill statistics
        # --------------------------------------------------------

        if has_actual_execution_status:

            filled_orders = int(
                (
                    execution_df["OrderStatus"]
                    == "FILLED"
                ).sum()
            )

            partial_orders = int(
                (
                    execution_df["OrderStatus"]
                    == "PARTIAL"
                ).sum()
            )

            cancelled_orders = int(
                (
                    execution_df["OrderStatus"]
                    == "CANCELLED"
                ).sum()
            )

            rejected_orders = int(
                (
                    execution_df["OrderStatus"]
                    == "REJECTED"
                ).sum()
            )

        else:

            # Current execution engine is a deterministic
            # execution estimate rather than a real fill engine.
            filled_orders = 0
            partial_orders = 0
            cancelled_orders = 0
            rejected_orders = 0

        # --------------------------------------------------------
        # Slippage
        # --------------------------------------------------------

        if "SlippageBps" in execution_df.columns:

            slippage = (
                execution_df["SlippageBps"]
                .astype(float)
                .dropna()
            )

        elif "Slippage" in execution_df.columns:

            slippage = (
                execution_df["Slippage"]
                .astype(float)
                .dropna()
            )

        else:

            slippage = pd.Series(
                dtype=float
            )

        # --------------------------------------------------------
        # Market impact
        # --------------------------------------------------------

        if "ImpactBps" in execution_df.columns:

            impact = (
                execution_df["ImpactBps"]
                .astype(float)
                .dropna()
            )

        elif "MarketImpact" in execution_df.columns:

            impact = (
                execution_df["MarketImpact"]
                .astype(float)
                .dropna()
            )

        else:

            impact = pd.Series(
                dtype=float
            )

        # --------------------------------------------------------
        # Transaction costs
        # --------------------------------------------------------

        costs = (
            execution_df.get(
                "TransactionCost",
                pd.Series(dtype=float),
            )
            .astype(float)
            .dropna()
        )

        # --------------------------------------------------------
        # Fill ratio
        # --------------------------------------------------------

        if "FillRatio" in execution_df.columns:

            fill_ratio = (
                execution_df["FillRatio"]
                .astype(float)
                .dropna()
            )

        else:

            fill_ratio = pd.Series(
                dtype=float
            )

        # --------------------------------------------------------
        # Participation
        # --------------------------------------------------------

        participation = (
            execution_df.get(
                "ParticipationRate",
                pd.Series(dtype=float),
            )
            .astype(float)
            .dropna()
        )

        # --------------------------------------------------------
        # Prices / shortfall
        # --------------------------------------------------------

        decision_price = (
            execution_df.get(
                "DecisionPrice",
                pd.Series(dtype=float),
            )
            .astype(float)
            .dropna()
        )

        execution_price = (
            execution_df.get(
                "ExecutionPrice",
                pd.Series(dtype=float),
            )
            .astype(float)
            .dropna()
        )

        quantity = (
            execution_df.get(
                "Quantity",
                pd.Series(dtype=float),
            )
            .astype(float)
            .dropna()
        )

        # --------------------------------------------------------
        # Notional
        # --------------------------------------------------------

        total_notional = float(
            execution_df.get(
                "Notional",
                pd.Series(dtype=float),
            )
            .astype(float)
            .sum()
        )

        return ExecutionAnalyticsResult(
            metadata=self.metadata,

            analytics_name=
            "Execution Analytics",

            total_orders=total_orders,

            filled_orders=filled_orders,

            partially_filled_orders=partial_orders,

            cancelled_orders=cancelled_orders,

            rejected_orders=rejected_orders,

            fill_rate=self.fill_rate(
                filled_orders,
                total_orders,
            ),

            completion_rate=self.completion_rate(
                filled_orders + partial_orders,
                total_orders,
            ),

            average_fill_ratio=self.average_fill_ratio(
                fill_ratio,
            ),

            average_slippage_bps=float(
                slippage.mean()
            ) if len(slippage) else 0.0,

            median_slippage_bps=float(
                slippage.median()
            ) if len(slippage) else 0.0,

            max_slippage_bps=float(
                slippage.max()
            ) if len(slippage) else 0.0,

            min_slippage_bps=float(
                slippage.min()
            ) if len(slippage) else 0.0,

            average_market_impact_bps=float(
                impact.mean()
            ) if len(impact) else 0.0,

            total_transaction_cost=float(
                costs.sum()
            ) if len(costs) else 0.0,

            average_transaction_cost=float(
                costs.mean()
            ) if len(costs) else 0.0,

            total_notional_traded=total_notional,

            average_participation_rate=float(
                participation.mean()
            ) if len(participation) else 0.0,

            execution_shortfall=self.execution_shortfall(
                decision_price,
                execution_price,
                quantity,
            ),

            broker_statistics=self.broker_statistics(
                execution_df,
            ),

            slippage_series=slippage,

            impact_series=impact,

            cost_series=costs,
        )

# ============================================================
# CONVENIENCE API
# ============================================================

def execution_analytics(
    *,
    metadata: AnalyticsMetadata,
    execution_df: pd.DataFrame,
) -> ExecutionAnalyticsResult:

    analyzer = (
        ExecutionAnalyticsAnalyzer(
            metadata=metadata,
        )
    )

    return analyzer.analyze(
        execution_df,
    )

# ============================================================
# PART 10
# CAPACITY & LIQUIDITY ANALYTICS
# ============================================================

# ============================================================
# CAPACITY RESULT
# ============================================================

@dataclass(slots=True)
class CapacityAnalyticsResult(
    AnalyticsResult,
):
    """
    Institutional capacity report.
    """

    total_portfolio_notional: float = 0.0

    average_adv_utilization: float = 0.0

    maximum_adv_utilization: float = 0.0

    median_adv_utilization: float = 0.0

    average_days_to_liquidate: float = 0.0

    maximum_days_to_liquidate: float = 0.0

    portfolio_capacity: float = 0.0

    liquidity_score: float = 0.0

    liquidity_concentration: float = 0.0

    illiquid_weight: float = 0.0

    liquid_weight: float = 0.0

    liquidity_bucket_exposure: pd.Series | None = None

    capacity_bottlenecks: pd.DataFrame | None = None

    adv_utilization_series: pd.Series | None = None

    days_to_liquidate_series: pd.Series | None = None


# ============================================================
# CAPACITY ANALYZER
# ============================================================

class CapacityAnalyticsAnalyzer(
    BaseAnalytics,
):
    """
    Institutional capacity analytics.

    Measures:

        ADV utilization
        Liquidity quality
        Portfolio scalability
        Days-to-liquidate
        Capacity bottlenecks
    """

    DEFAULT_PARTICIPATION_RATE = 0.10

    # --------------------------------------------------------
    # ADV Utilization
    # --------------------------------------------------------

    @staticmethod
    def adv_utilization(
        position_notional: pd.Series,
        average_daily_dollar_volume: pd.Series,
    ) -> pd.Series:

        adv = (
            average_daily_dollar_volume
            .replace(
                0.0,
                np.nan,
            )
        )

        return (
            position_notional
            /
            adv
        ).fillna(0.0)

    # --------------------------------------------------------
    # Days To Liquidate
    # --------------------------------------------------------

    @staticmethod
    def days_to_liquidate(
        position_notional: pd.Series,
        average_daily_dollar_volume: pd.Series,
        participation_rate: float = 0.10,
    ) -> pd.Series:

        adv_capacity = (
            average_daily_dollar_volume
            * participation_rate
        )

        adv_capacity = adv_capacity.replace(
            0.0,
            np.nan,
        )

        return (
            position_notional
            /
            adv_capacity
        ).fillna(0.0)

    # --------------------------------------------------------
    # Portfolio Capacity
    # --------------------------------------------------------

    @staticmethod
    def portfolio_capacity(
        average_daily_dollar_volume: pd.Series,
        participation_rate: float = 0.10,
    ) -> float:

        if len(
            average_daily_dollar_volume
        ) == 0:

            return 0.0

        return float(

            (
                average_daily_dollar_volume
                *
                participation_rate
            ).sum()

        )

    # --------------------------------------------------------
    # Liquidity Score
    # --------------------------------------------------------

    @staticmethod
    def liquidity_score(
        adv_utilization: pd.Series,
    ) -> float:
        """
        Higher is better.

        100 = highly liquid
        0   = extremely illiquid
        """

        if len(
            adv_utilization
        ) == 0:

            return 0.0

        utilization = float(
            adv_utilization.mean()
        )

        score = (
            100.0
            *
            np.exp(
                -5.0 * utilization
            )
        )

        return float(
            np.clip(
                score,
                0.0,
                100.0,
            )
        )

    # --------------------------------------------------------
    # Liquidity Concentration
    # --------------------------------------------------------

    @staticmethod
    def liquidity_concentration(
        position_notional: pd.Series,
    ) -> float:

        total = (
            position_notional.sum()
        )

        if total <= EPSILON:

            return 0.0

        w = (
            position_notional
            /
            total
        )

        return float(
            np.square(
                w
            ).sum()
        )

    # --------------------------------------------------------
    # Liquidity Buckets
    # --------------------------------------------------------

    @staticmethod
    def liquidity_bucket_exposure(
        adv_utilization: pd.Series,
        weights: pd.Series,
    ) -> pd.Series:

        bucket = pd.cut(

            adv_utilization,

            bins=[
                -np.inf,
                0.01,
                0.05,
                0.10,
                0.25,
                np.inf,
            ],

            labels=[
                "Highly Liquid",
                "Liquid",
                "Moderate",
                "Illiquid",
                "Highly Illiquid",
            ],
        )

        tmp = pd.DataFrame({

            "bucket": bucket,

            "weight": weights,

        })

        return (
            tmp
            .groupby("bucket")
            ["weight"]
            .sum()
        )

    # --------------------------------------------------------
    # Bottlenecks
    # --------------------------------------------------------

    @staticmethod
    def capacity_bottlenecks(
        portfolio: pd.DataFrame,
        adv_utilization: pd.Series,
        days_to_liquidate: pd.Series,
        top_n: int = 10,
    ) -> pd.DataFrame:

        tmp = portfolio.copy()

        tmp[
            "ADV_Utilization"
        ] = adv_utilization

        tmp[
            "DaysToLiquidate"
        ] = days_to_liquidate

        sort_cols = [
            "ADV_Utilization",
            "DaysToLiquidate",
        ]

        return (
            tmp
            .sort_values(
                sort_cols,
                ascending=False,
            )
            .head(top_n)
        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    def analyze(
        self,
        portfolio: pd.DataFrame,
        *,
        participation_rate: float = 0.10,
    ) -> CapacityAnalyticsResult:

        self.validate_dataframe(
            portfolio,
            "portfolio",
        )

        required = [

            "Position_Weight",
            "Market_Value",
            "ADV",

        ]

        self.validate_columns(
            portfolio,
            required,
            "portfolio",
        )

        position_notional = (
            portfolio[
                "Market_Value"
            ]
            .abs()
        )

        adv = (
            portfolio["ADV"]
        )

        weights = (
            portfolio[
                "Position_Weight"
            ]
            .abs()
        )

        adv_util = (
            self.adv_utilization(
                position_notional,
                adv,
            )
        )

        dtl = (
            self.days_to_liquidate(
                position_notional,
                adv,
                participation_rate,
            )
        )

        liquid_weight = float(
            weights[
                adv_util < 0.05
            ].sum()
        )

        illiquid_weight = float(
            weights[
                adv_util > 0.25
            ].sum()
        )

        return CapacityAnalyticsResult(

            metadata=
            self.metadata,

            analytics_name=
            "Capacity Analytics",

            total_portfolio_notional=
            float(
                position_notional.sum()
            ),

            average_adv_utilization=
            float(
                adv_util.mean()
            ),

            maximum_adv_utilization=
            float(
                adv_util.max()
            ),

            median_adv_utilization=
            float(
                adv_util.median()
            ),

            average_days_to_liquidate=
            float(
                dtl.mean()
            ),

            maximum_days_to_liquidate=
            float(
                dtl.max()
            ),

            portfolio_capacity=
            self.portfolio_capacity(
                adv,
                participation_rate,
            ),

            liquidity_score=
            self.liquidity_score(
                adv_util,
            ),

            liquidity_concentration=
            self.liquidity_concentration(
                position_notional,
            ),

            illiquid_weight=
            illiquid_weight,

            liquid_weight=
            liquid_weight,

            liquidity_bucket_exposure=
            self.liquidity_bucket_exposure(
                adv_util,
                weights,
            ),

            capacity_bottlenecks=
            self.capacity_bottlenecks(
                portfolio,
                adv_util,
                dtl,
            ),

            adv_utilization_series=
            adv_util,

            days_to_liquidate_series=
            dtl,
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def capacity_analytics(
    *,
    metadata: AnalyticsMetadata,
    portfolio: pd.DataFrame,
    participation_rate: float = 0.10,
) -> CapacityAnalyticsResult:

    analyzer = (
        CapacityAnalyticsAnalyzer(
            metadata=metadata,
        )
    )

    return analyzer.analyze(
        portfolio,
        participation_rate=
        participation_rate,
    )

# ============================================================
# PART 11
# INSTITUTIONAL MASTER REPORTING LAYER
# ============================================================

# ============================================================
# MASTER REPORT
# ============================================================

@dataclass(slots=True)
class InstitutionalAnalyticsReport(
    AnalyticsResult,
):
    """
    Master institutional analytics report.

    Aggregates all analytics modules.
    """

    portfolio_statistics: PortfolioStatisticsResult | None = None

    exposure_analytics: ExposureAnalyticsResult | None = None

    concentration_analytics: ConcentrationAnalyticsResult | None = None

    risk_analytics: RiskAnalyticsResult | None = None

    factor_analytics: FactorAnalyticsResult | None = None

    performance_analytics: PerformanceAnalyticsResult | None = None

    rebalance_analytics: RebalanceAnalyticsResult | None = None

    execution_analytics: ExecutionAnalyticsResult | None = None

    capacity_analytics: CapacityAnalyticsResult | None = None

    report_timestamp: datetime = field(
        default_factory=datetime.utcnow
    )

    report_version: str = "1.0"

    summary_metrics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# MASTER REPORT BUILDER
# ============================================================

class InstitutionalAnalyticsReportBuilder(
    BaseAnalytics,
):
    """
    Institutional reporting engine.

    Creates unified analytics reports.
    """

    # --------------------------------------------------------

    def build_summary(
        self,
        report: InstitutionalAnalyticsReport,
    ) -> dict[str, Any]:

        summary = {}

        # --------------------------------------
        # Performance
        # --------------------------------------

        if (
            report.performance_analytics
            is not None
        ):

            p = report.performance_analytics

            summary["AnnualizedReturn"] = (
                p.annualized_return
            )

            summary["Alpha"] = (
                p.alpha
            )

            summary["Beta"] = (
                p.beta
            )

            summary["InformationRatio"] = (
                p.information_ratio
            )

        # --------------------------------------
        # Risk
        # --------------------------------------

        if (
            report.risk_analytics
            is not None
        ):

            r = report.risk_analytics

            summary["Volatility"] = (
                r.annualized_volatility
            )

            summary["VaR"] = (
                r.value_at_risk
            )

            summary["CVaR"] = (
                r.expected_shortfall
            )

        # --------------------------------------
        # Concentration
        # --------------------------------------

        if (
            report.concentration_analytics
            is not None
        ):

            c = (
                report
                .concentration_analytics
            )

            summary["HHI"] = (
                c.hhi
            )

            summary["EffectiveHoldings"] = (
                c.effective_number_of_holdings
            )

        # --------------------------------------
        # Exposure
        # --------------------------------------

        if (
            report.exposure_analytics
            is not None
        ):

            e = (
                report
                .exposure_analytics
            )

            summary["GrossExposure"] = (
                e.gross_exposure
            )

            summary["NetExposure"] = (
                e.net_exposure
            )

            summary["Leverage"] = (
                e.leverage
            )

        # --------------------------------------
        # Capacity
        # --------------------------------------

        if (
            report.capacity_analytics
            is not None
        ):

            cap = (
                report
                .capacity_analytics
            )

            summary["LiquidityScore"] = (
                cap.liquidity_score
            )

            summary["PortfolioCapacity"] = (
                cap.portfolio_capacity
            )

        return summary

    # --------------------------------------------------------

    def build(
        self,
        *,
        portfolio_statistics: PortfolioStatisticsResult | None = None,
        exposure_analytics: ExposureAnalyticsResult | None = None,
        concentration_analytics: ConcentrationAnalyticsResult | None = None,
        risk_analytics: RiskAnalyticsResult | None = None,
        factor_analytics: FactorAnalyticsResult | None = None,
        performance_analytics: PerformanceAnalyticsResult | None = None,
        rebalance_analytics: RebalanceAnalyticsResult | None = None,
        execution_analytics: ExecutionAnalyticsResult | None = None,
        capacity_analytics: CapacityAnalyticsResult | None = None,
    ) -> InstitutionalAnalyticsReport:

        report = InstitutionalAnalyticsReport(

            metadata=self.metadata,

            analytics_name=
            "Institutional Analytics Report",

            portfolio_statistics=
            portfolio_statistics,

            exposure_analytics=
            exposure_analytics,

            concentration_analytics=
            concentration_analytics,

            risk_analytics=
            risk_analytics,

            factor_analytics=
            factor_analytics,

            performance_analytics=
            performance_analytics,

            rebalance_analytics=
            rebalance_analytics,

            execution_analytics=
            execution_analytics,

            capacity_analytics=
            capacity_analytics,
        )

        report.summary_metrics = (
            self.build_summary(
                report
            )
        )

        return report


# ============================================================
# REPORT EXPORTER
# ============================================================

class InstitutionalReportExporter:
    """
    Export institutional reports.

    Supports:

        dict
        dataframe
        json
    """

    # --------------------------------------------------------

    @staticmethod
    def to_dict(
        report: InstitutionalAnalyticsReport,
    ) -> dict[str, Any]:

        return asdict(
            report
        )

    # --------------------------------------------------------

    @staticmethod
    def summary_dataframe(
        report: InstitutionalAnalyticsReport,
    ) -> pd.DataFrame:

        return pd.DataFrame(
            [
                report.summary_metrics
            ]
        )

    # --------------------------------------------------------

    @staticmethod
    def to_json(
        report: InstitutionalAnalyticsReport,
    ) -> str:

        return json.dumps(

            InstitutionalReportExporter
            .to_dict(report),

            default=str,

            indent=2,
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def build_institutional_report(
    *,
    metadata: AnalyticsMetadata,
    portfolio_statistics: PortfolioStatisticsResult | None = None,
    exposure_analytics: ExposureAnalyticsResult | None = None,
    concentration_analytics: ConcentrationAnalyticsResult | None = None,
    risk_analytics: RiskAnalyticsResult | None = None,
    factor_analytics: FactorAnalyticsResult | None = None,
    performance_analytics: PerformanceAnalyticsResult | None = None,
    rebalance_analytics: RebalanceAnalyticsResult | None = None,
    execution_analytics: ExecutionAnalyticsResult | None = None,
    capacity_analytics: CapacityAnalyticsResult | None = None,
) -> InstitutionalAnalyticsReport:

    builder = (
        InstitutionalAnalyticsReportBuilder(
            metadata=metadata,
        )
    )

    return builder.build(
        portfolio_statistics=
        portfolio_statistics,

        exposure_analytics=
        exposure_analytics,

        concentration_analytics=
        concentration_analytics,

        risk_analytics=
        risk_analytics,

        factor_analytics=
        factor_analytics,

        performance_analytics=
        performance_analytics,

        rebalance_analytics=
        rebalance_analytics,

        execution_analytics=
        execution_analytics,

        capacity_analytics=
        capacity_analytics,
    )

# ============================================================
# PART 12
# ANALYTICS ENGINE
# ============================================================

class AnalyticsEngine:
    """
    Institutional analytics orchestration engine.

    Responsibilities
    ----------------
    1. Run all analytics modules
    2. Manage dependency ordering
    3. Build institutional report
    4. Serve as single analytics entry point

    Used by:

        construction/pipeline.py
        backtest framework
        PM dashboards
        Risk dashboards
        Reporting systems
    """

    # --------------------------------------------------------

    def __init__(
        self,
        metadata: AnalyticsMetadata,
        config: AnalyticsConfig | None = None,
    ) -> None:

        self.metadata = metadata

        self.config = (
            config
            if config is not None
            else AnalyticsConfig()
        )

    # --------------------------------------------------------
    # Portfolio Statistics
    # --------------------------------------------------------

    def run_portfolio_statistics(
        self,
        portfolio: pd.DataFrame,
        returns: pd.Series | None = None,
    ) -> PortfolioStatisticsResult | None:

        try:

            analyzer = PortfolioStatisticsAnalyzer(
                metadata=self.metadata,
            )

            return analyzer.analyze(
                returns=returns,
            )

        except Exception as exc:

            logger.exception(
                "PortfolioStatistics failed: %s",
                exc,
            )

            return None

    # --------------------------------------------------------
    # Exposure
    # --------------------------------------------------------

    def run_exposure(
        self,
        portfolio: pd.DataFrame,
    ) -> ExposureAnalyticsResult | None:

        try:

            return exposure_analytics(
                portfolio,
                metadata=self.metadata,
            )

        except Exception as exc:

            logger.exception(
                "ExposureAnalytics failed: %s",
                exc,
            )

            return None

    # --------------------------------------------------------
    # Concentration
    # --------------------------------------------------------

    def run_concentration(
        self,
        portfolio: pd.DataFrame,
    ) -> ConcentrationAnalyticsResult | None:

        try:

            return concentration_analytics(
                portfolio,
                metadata=self.metadata,
            )

        except Exception as exc:

            logger.exception(
                "ConcentrationAnalytics failed: %s",
                exc,
            )

            return None

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    def run_risk(
        self,
        *,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
        weights: pd.Series | None = None,
        covariance_matrix: pd.DataFrame | None = None,
        factor_exposures: pd.Series | None = None,
        factor_contributions: pd.Series | None = None,
    ) -> RiskAnalyticsResult | None:

        try:

            return risk_analytics(
                metadata=self.metadata,
                portfolio_returns=portfolio_returns,
                benchmark_returns=benchmark_returns,
                weights=weights,
                covariance_matrix=covariance_matrix,
                factor_exposures=factor_exposures,
                factor_contributions=factor_contributions,
            )

        except Exception as exc:

            logger.exception(
                "RiskAnalytics failed: %s",
                exc,
            )

            return None

    # --------------------------------------------------------
    # Factor
    # --------------------------------------------------------

    def run_factor(
        self,
        *,
        factor_exposures: pd.Series,
        factor_returns: pd.Series | None = None,
        factor_covariance: pd.DataFrame | None = None,
        factor_return_history: pd.DataFrame | None = None,
    ) -> FactorAnalyticsResult | None:

        try:

            return factor_analytics(
                metadata=self.metadata,
                factor_exposures=factor_exposures,
                factor_returns=factor_returns,
                factor_covariance=factor_covariance,
                factor_return_history=factor_return_history,
            )

        except Exception as exc:

            logger.exception(
                "FactorAnalytics failed: %s",
                exc,
            )

            return None

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    def run_performance(
        self,
        *,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> PerformanceAnalyticsResult | None:

        try:

            return performance_analytics(
                metadata=self.metadata,
                portfolio_returns=portfolio_returns,
                benchmark_returns=benchmark_returns,
            )

        except Exception as exc:

            logger.exception(
                "PerformanceAnalytics failed: %s",
                exc,
            )

            return None

    # --------------------------------------------------------
    # Rebalance
    # --------------------------------------------------------

    def run_rebalance(
        self,
        *,
        turnover_series: pd.Series,
        trade_count_series: pd.Series,
        buy_count_series: pd.Series | None = None,
        sell_count_series: pd.Series | None = None,
        cost_series: pd.Series | None = None,
        drift_before_series: pd.Series | None = None,
        drift_after_series: pd.Series | None = None,
        rebalance_frequency_per_year: int = 12,
    ) -> RebalanceAnalyticsResult | None:

        try:

            return rebalance_analytics(
                metadata=self.metadata,
                turnover_series=turnover_series,
                trade_count_series=trade_count_series,
                buy_count_series=buy_count_series,
                sell_count_series=sell_count_series,
                cost_series=cost_series,
                drift_before_series=drift_before_series,
                drift_after_series=drift_after_series,
                rebalance_frequency_per_year=
                rebalance_frequency_per_year,
            )

        except Exception as exc:

            logger.exception(
                "RebalanceAnalytics failed: %s",
                exc,
            )

            return None

    # --------------------------------------------------------
    # Execution
    # --------------------------------------------------------

    def run_execution(
        self,
        execution_df: pd.DataFrame,
    ) -> ExecutionAnalyticsResult | None:

        try:

            return execution_analytics(
                metadata=self.metadata,
                execution_df=execution_df,
            )

        except Exception as exc:

            logger.exception(
                "ExecutionAnalytics failed: %s",
                exc,
            )

            return None

    # --------------------------------------------------------
    # Capacity
    # --------------------------------------------------------

    def run_capacity(
        self,
        portfolio: pd.DataFrame,
        participation_rate: float = 0.10,
    ) -> CapacityAnalyticsResult | None:

        try:

            return capacity_analytics(
                metadata=self.metadata,
                portfolio=portfolio,
                participation_rate=
                participation_rate,
            )

        except Exception as exc:

            logger.exception(
                "CapacityAnalytics failed: %s",
                exc,
            )

            return None

    # --------------------------------------------------------
    # MASTER RUNNER
    # --------------------------------------------------------

    def run_all(
        self,
        *,
        portfolio: pd.DataFrame,
        portfolio_returns: pd.Series | None = None,
        benchmark_returns: pd.Series | None = None,
        execution_df: pd.DataFrame | None = None,
        turnover_series: pd.Series | None = None,
        trade_count_series: pd.Series | None = None,
        covariance_matrix: pd.DataFrame | None = None,
        factor_exposures: pd.Series | None = None,
        factor_returns: pd.Series | None = None,
        factor_covariance: pd.DataFrame | None = None,
        factor_return_history: pd.DataFrame | None = None,
    ) -> InstitutionalAnalyticsReport:

        # --------------------------------------
        # Portfolio Statistics
        # --------------------------------------

        stats = (
            self.run_portfolio_statistics(
                portfolio,
                portfolio_returns,
            )
        )

        # --------------------------------------
        # Exposure
        # --------------------------------------

        exposure = (
            self.run_exposure(
                portfolio,
            )
        )

        # --------------------------------------
        # Concentration
        # --------------------------------------

        concentration = (
            self.run_concentration(
                portfolio,
            )
        )

        # --------------------------------------
        # Factor
        # --------------------------------------

        factor = None

        if factor_exposures is not None:

            factor = (
                self.run_factor(
                    factor_exposures=
                    factor_exposures,
                    factor_returns=
                    factor_returns,
                    factor_covariance=
                    factor_covariance,
                    factor_return_history=
                    factor_return_history,
                )
            )

        # --------------------------------------
        # Risk
        # --------------------------------------

        risk = None

        if portfolio_returns is not None:

            weights = (
                portfolio[
                    "Position_Weight"
                ]
                if "Position_Weight"
                in portfolio.columns
                else None
            )

            risk = (
                self.run_risk(
                    portfolio_returns=
                    portfolio_returns,
                    benchmark_returns=
                    benchmark_returns,
                    weights=
                    weights,
                    covariance_matrix=
                    covariance_matrix,
                    factor_exposures=
                    factor_exposures,
                )
            )

        # --------------------------------------
        # Performance
        # --------------------------------------

        performance = None

        if (
            portfolio_returns is not None
            and
            benchmark_returns is not None
        ):

            performance = (
                self.run_performance(
                    portfolio_returns=
                    portfolio_returns,
                    benchmark_returns=
                    benchmark_returns,
                )
            )

        # --------------------------------------
        # Rebalance
        # --------------------------------------

        rebalance = None

        if (
            turnover_series is not None
            and
            trade_count_series is not None
        ):

            rebalance = (
                self.run_rebalance(
                    turnover_series=
                    turnover_series,
                    trade_count_series=
                    trade_count_series,
                )
            )

        # --------------------------------------
        # Execution
        # --------------------------------------

        execution = None

        if execution_df is not None:

            execution = (
                self.run_execution(
                    execution_df,
                )
            )

        # --------------------------------------
        # Capacity
        # --------------------------------------

        capacity = (
            self.run_capacity(
                portfolio,
            )
        )

        # --------------------------------------
        # Master Report
        # --------------------------------------

        return build_institutional_report(

            metadata=
            self.metadata,

            portfolio_statistics=
            stats,

            exposure_analytics=
            exposure,

            concentration_analytics=
            concentration,

            risk_analytics=
            risk,

            factor_analytics=
            factor,

            performance_analytics=
            performance,

            rebalance_analytics=
            rebalance,

            execution_analytics=
            execution,

            capacity_analytics=
            capacity,
        )
    

# ============================================================
# PART 13
# FACTORY & CONVENIENCE APIS
# This is the public API layer that hides internal analytics complexity and exposes
# institutional-grade entry points for PM dashboards, risk systems, backtests,
# reporting engines
# ============================================================

# ============================================================
# ENGINE FACTORY
# ============================================================

def create_analytics_engine(
    *,
    metadata: AnalyticsMetadata,
    config: AnalyticsConfig | None = None,
) -> AnalyticsEngine:
    """
    Create institutional analytics engine.

    Example
    -------
    engine = create_analytics_engine(
        metadata=metadata,
    )
    """

    return AnalyticsEngine(
        metadata=metadata,
        config=config,
    )


# ============================================================
# FULL ANALYTICS RUNNER
# ============================================================

def run_full_analytics(
    *,
    metadata: AnalyticsMetadata,
    portfolio: pd.DataFrame,
    portfolio_returns: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
    execution_df: pd.DataFrame | None = None,
    turnover_series: pd.Series | None = None,
    trade_count_series: pd.Series | None = None,
    covariance_matrix: pd.DataFrame | None = None,
    factor_exposures: pd.Series | None = None,
    factor_returns: pd.Series | None = None,
    factor_covariance: pd.DataFrame | None = None,
    factor_return_history: pd.DataFrame | None = None,
    config: AnalyticsConfig | None = None,
) -> InstitutionalAnalyticsReport:
    """
    Single institutional entry point.

    Used by:

        pipeline.py
        backtests
        PM dashboards
        reporting layer
    """

    engine = AnalyticsEngine(
        metadata=metadata,
        config=config,
    )

    return engine.run_all(
        portfolio=portfolio,
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        execution_df=execution_df,
        turnover_series=turnover_series,
        trade_count_series=trade_count_series,
        covariance_matrix=covariance_matrix,
        factor_exposures=factor_exposures,
        factor_returns=factor_returns,
        factor_covariance=factor_covariance,
        factor_return_history=factor_return_history,
    )


# ============================================================
# PM DASHBOARD VIEW
# ============================================================

def build_pm_dashboard(
    report: InstitutionalAnalyticsReport,
) -> pd.DataFrame:
    """
    Portfolio Manager dashboard.

    Focus:
        Returns
        Alpha
        Beta
        Exposures
        Concentration
    """

    metrics = {}

    if report.performance_analytics:

        metrics["AnnualizedReturn"] = (
            report.performance_analytics
            .annualized_return
        )

        metrics["Alpha"] = (
            report.performance_analytics
            .alpha
        )

        metrics["Beta"] = (
            report.performance_analytics
            .beta
        )

    if report.exposure_analytics:

        metrics["GrossExposure"] = (
            report.exposure_analytics
            .gross_exposure
        )

        metrics["NetExposure"] = (
            report.exposure_analytics
            .net_exposure
        )

    if report.concentration_analytics:

        metrics["HHI"] = (
            report.concentration_analytics
            .hhi
        )

    return pd.DataFrame(
        [metrics]
    )


# ============================================================
# RISK DASHBOARD VIEW
# ============================================================

def build_risk_dashboard(
    report: InstitutionalAnalyticsReport,
) -> pd.DataFrame:
    """
    Risk committee dashboard.
    """

    metrics = {}

    if report.risk_analytics:

        metrics["Volatility"] = (
            report.risk_analytics
            .annualized_volatility
        )

        metrics["VaR"] = (
            report.risk_analytics
            .value_at_risk
        )

        metrics["CVaR"] = (
            report.risk_analytics
            .expected_shortfall
        )

    if report.capacity_analytics:

        metrics["LiquidityScore"] = (
            report.capacity_analytics
            .liquidity_score
        )

    return pd.DataFrame(
        [metrics]
    )


# ============================================================
# EXECUTION DASHBOARD VIEW
# ============================================================

def build_execution_dashboard(
    report: InstitutionalAnalyticsReport,
) -> pd.DataFrame:
    """
    Execution committee dashboard.
    """

    if (
        report.execution_analytics
        is None
    ):
        return pd.DataFrame()

    e = report.execution_analytics

    return pd.DataFrame(
        [{
            "FillRate":
            e.fill_rate,

            "CompletionRate":
            e.completion_rate,

            "AvgSlippage":
            e.average_slippage_bps,

            "AvgImpact":
            e.average_market_impact_bps,

            "ExecutionShortfall":
            e.execution_shortfall,

            "TotalCost":
            e.total_transaction_cost,
        }]
    )


# ============================================================
# CAPACITY DASHBOARD VIEW
# ============================================================

def build_capacity_dashboard(
    report: InstitutionalAnalyticsReport,
) -> pd.DataFrame:
    """
    Capacity dashboard.
    """

    if (
        report.capacity_analytics
        is None
    ):
        return pd.DataFrame()

    c = report.capacity_analytics

    return pd.DataFrame(
        [{
            "PortfolioCapacity":
            c.portfolio_capacity,

            "LiquidityScore":
            c.liquidity_score,

            "AverageADVUtilization":
            c.average_adv_utilization,

            "AverageDaysToLiquidate":
            c.average_days_to_liquidate,

            "IlliquidWeight":
            c.illiquid_weight,
        }]
    )


# ============================================================
# EXPORT HELPERS
# ============================================================

def export_report_dict(
    report: InstitutionalAnalyticsReport,
) -> dict[str, Any]:

    return (
        InstitutionalReportExporter
        .to_dict(report)
    )


def export_report_json(
    report: InstitutionalAnalyticsReport,
) -> str:

    return (
        InstitutionalReportExporter
        .to_json(report)
    )


def export_report_dataframe(
    report: InstitutionalAnalyticsReport,
) -> pd.DataFrame:

    return (
        InstitutionalReportExporter
        .summary_dataframe(report)
    )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [

    # Engine
    "AnalyticsEngine",

    # Report
    "InstitutionalAnalyticsReport",

    # Factories
    "create_analytics_engine",

    # Runner
    "run_full_analytics",

    # Dashboards
    "build_pm_dashboard",
    "build_risk_dashboard",
    "build_execution_dashboard",
    "build_capacity_dashboard",

    # Export
    "export_report_dict",
    "export_report_json",
    "export_report_dataframe",
]