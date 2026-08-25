# ============================================================
# ATTRIBUTION.PY
# PART 1
# FRAMEWORK & CORE OBJECTS
# ============================================================

from __future__ import annotations

# ============================================================
# STANDARD LIBRARIES
# ============================================================

import logging

from abc import (
    ABC,
    abstractmethod,
)

from enum import Enum

from dataclasses import (
    dataclass,
    field,
)

from datetime import (
    datetime,
)

from typing import (
    Any,
    Optional,
    Iterable,
)

# ============================================================
# THIRD PARTY
# ============================================================

import numpy as np
import pandas as pd

from dataclasses import asdict
import json
from datetime import datetime

# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(__name__)

# ============================================================
# GLOBALS
# ============================================================

EPSILON: float = 1e-12

# ============================================================
# ENUMS
# ============================================================


class AttributionType(
    str,
    Enum,
):
    """
    Institutional attribution types.
    """

    RETURN = "return"

    BRINSON = "brinson"

    FACTOR = "factor"

    RISK = "risk"

    TRADE = "trade"

    REBALANCE = "rebalance"

    EXECUTION = "execution"

    MULTI_PERIOD = "multi_period"


# ------------------------------------------------------------


class AttributionFrequency(
    str,
    Enum,
):
    """
    Attribution horizon.
    """

    DAILY = "daily"

    WEEKLY = "weekly"

    MONTHLY = "monthly"

    QUARTERLY = "quarterly"

    YEARLY = "yearly"


# ------------------------------------------------------------


class LinkingMethod(
    str,
    Enum,
):
    """
    Multi-period linking methodology.
    """

    ARITHMETIC = "arithmetic"

    GEOMETRIC = "geometric"

    CARINO = "carino"

    MENCHERO = "menchero"


# ============================================================
# METADATA
# ============================================================


@dataclass(slots=True)
class AttributionMetadata:
    """
    Metadata attached to every attribution report.
    """

    created_at: datetime = field(
        default_factory=datetime.utcnow
    )

    version: str = "1.0"

    source: str = "Institutional Attribution Engine"

    portfolio_name: str = ""

    benchmark_name: str = ""

    frequency: AttributionFrequency = (
        AttributionFrequency.MONTHLY
    )

    notes: str = ""


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(slots=True)
class AttributionConfig:
    """
    Institutional attribution configuration.
    """

    frequency: AttributionFrequency = (
        AttributionFrequency.MONTHLY
    )

    linking_method: LinkingMethod = (
        LinkingMethod.GEOMETRIC
    )

    annualization_factor: int = 252

    confidence_level: float = 0.95

    use_geometric_linking: bool = True

    normalize_contributions: bool = True

    store_intermediate_results: bool = True

    diagnostics_enabled: bool = True


# ============================================================
# UTILITIES
# ============================================================


class AttributionUtils:
    """
    Shared attribution helpers.
    """

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
            numerator / denominator
        )

    # --------------------------------------------------------

    @staticmethod
    def ensure_series(
        values: pd.Series | np.ndarray | list,
        name: str = "",
    ) -> pd.Series:

        if isinstance(
            values,
            pd.Series,
        ):
            return values

        return pd.Series(
            values,
            name=name,
        )

    # --------------------------------------------------------

    @staticmethod
    def validate_no_nan(
        values: pd.Series,
        field_name: str,
    ) -> None:

        if values.isna().any():

            raise ValueError(
                f"{field_name} contains NaN values."
            )

    # --------------------------------------------------------

    @staticmethod
    def validate_same_length(
        *series: pd.Series,
    ) -> None:

        lengths = [
            len(s)
            for s in series
        ]

        if len(
            set(lengths)
        ) > 1:

            raise ValueError(
                f"Series lengths mismatch: {lengths}"
            )

    # --------------------------------------------------------

    @staticmethod
    def normalize_contributions(
        contributions: pd.Series,
    ) -> pd.Series:

        total = float(
            contributions.sum()
        )

        if abs(
            total
        ) < EPSILON:

            return contributions * 0.0

        return (
            contributions
            / total
        )

    # --------------------------------------------------------

    @staticmethod
    def annualize_return(
        return_series: pd.Series,
        factor: int,
    ) -> float:

        if len(
            return_series
        ) == 0:

            return 0.0

        cumulative = float(
            (
                1.0
                +
                return_series
            ).prod()
        )

        years = (
            len(return_series)
            / factor
        )

        if years <= 0:

            return 0.0

        return float(
            cumulative ** (1.0 / years)
            - 1.0
        )


# ============================================================
# BASE ATTRIBUTION OBJECT
# ============================================================


@dataclass(slots=True)
class AttributionBase:
    """
    Root attribution object.
    """

    metadata: AttributionMetadata

    attribution_type: AttributionType

    attribution_name: str

    created_at: datetime = field(
        default_factory=datetime.utcnow
    )


# ============================================================
# ABSTRACT BASE ANALYZER
# ============================================================


class BaseAttribution(
    ABC,
):
    """
    Institutional attribution interface.
    """

    def __init__(
        self,
        metadata: AttributionMetadata,
        config: AttributionConfig | None = None,
    ) -> None:

        self.metadata = metadata

        self.config = (
            config
            if config is not None
            else AttributionConfig()
        )

    # --------------------------------------------------------

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        name: str,
    ) -> None:

        if not isinstance(
            df,
            pd.DataFrame,
        ):

            raise TypeError(
                f"{name} must be DataFrame."
            )

        if df.empty:

            raise ValueError(
                f"{name} is empty."
            )

    # --------------------------------------------------------

    @staticmethod
    def validate_columns(
        df: pd.DataFrame,
        required: Iterable[str],
        name: str,
    ) -> None:

        missing = [

            col
            for col in required
            if col not in df.columns

        ]

        if missing:

            raise ValueError(
                f"{name} missing columns: {missing}"
            )

    # --------------------------------------------------------

    @abstractmethod
    def analyze(
        self,
        *args,
        **kwargs,
    ):
        """
        Run attribution analysis.
        """

        raise NotImplementedError


# ============================================================
# FRAMEWORK COMPLETE
# ============================================================

# ============================================================
# PART 2
# ATTRIBUTION RESULT OBJECTS
# ============================================================

# ============================================================
# GENERIC ATTRIBUTION RESULT
# ============================================================

@dataclass(slots=True)
class AttributionResult(
    AttributionBase,
):
    """
    Generic attribution result.

    Parent class for all attribution outputs.
    """

    total_portfolio_return: float = 0.0

    total_benchmark_return: float = 0.0

    active_return: float = 0.0

    contribution_table: pd.DataFrame | None = None

    contribution_series: pd.Series | None = None

    residual: float = 0.0

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# RETURN ATTRIBUTION
# ============================================================

@dataclass(slots=True)
class ReturnAttributionResult(
    AttributionResult,
):
    """
    Asset-level return attribution.
    """

    asset_contributions: pd.Series | None = None

    sector_contributions: pd.Series | None = None

    cash_contribution: float = 0.0

    residual_contribution: float = 0.0


# ============================================================
# BRINSON ATTRIBUTION
# ============================================================

@dataclass(slots=True)
class BrinsonAttributionResult(
    AttributionResult,
):
    """
    Brinson-Fachler attribution.
    """

    allocation_effect: float = 0.0

    selection_effect: float = 0.0

    interaction_effect: float = 0.0

    sector_effects: pd.DataFrame | None = None

    allocation_by_sector: pd.Series | None = None

    selection_by_sector: pd.Series | None = None

    interaction_by_sector: pd.Series | None = None


# ============================================================
# FACTOR ATTRIBUTION
# ============================================================

@dataclass(slots=True)
class FactorAttributionResult(
    AttributionResult,
):
    """
    Factor return attribution.
    """

    factor_exposures: pd.Series | None = None

    factor_returns: pd.Series | None = None

    factor_contributions: pd.Series | None = None

    factor_covariance: pd.DataFrame | None = None

    residual_return: float = 0.0

    explained_return: float = 0.0


# ============================================================
# RISK ATTRIBUTION
# ============================================================

@dataclass(slots=True)
class RiskAttributionResult(
    AttributionResult,
):
    """
    Risk decomposition.
    """

    total_risk: float = 0.0

    marginal_contribution_to_risk: pd.Series | None = None

    component_contribution_to_risk: pd.Series | None = None

    percent_contribution_to_risk: pd.Series | None = None

    factor_risk_contribution: pd.Series | None = None

    specific_risk_contribution: float = 0.0


# ============================================================
# TRADE ATTRIBUTION
# ============================================================

@dataclass(slots=True)
class TradeAttributionResult(
    AttributionResult,
):
    """
    Trade-level attribution.
    """

    total_trade_pnl: float = 0.0

    average_trade_pnl: float = 0.0

    hit_rate: float = 0.0

    win_loss_ratio: float = 0.0

    trade_alpha: float = 0.0

    trade_contributions: pd.Series | None = None

    trade_table: pd.DataFrame | None = None


# ============================================================
# REBALANCE ATTRIBUTION
# ============================================================

@dataclass(slots=True)
class RebalanceAttributionResult(
    AttributionResult,
):
    """
    Rebalance attribution.
    """

    turnover_effect: float = 0.0

    drift_reduction_effect: float = 0.0

    rebalance_alpha: float = 0.0

    rebalance_cost: float = 0.0

    rebalance_contributions: pd.Series | None = None


# ============================================================
# EXECUTION ATTRIBUTION
# ============================================================

@dataclass(slots=True)
class ExecutionAttributionResult(
    AttributionResult,
):
    """
    Execution attribution.
    """

    slippage_cost: float = 0.0

    market_impact_cost: float = 0.0

    opportunity_cost: float = 0.0

    execution_shortfall: float = 0.0

    total_execution_cost: float = 0.0

    execution_contributions: pd.Series | None = None


# ============================================================
# MULTI-PERIOD ATTRIBUTION
# ============================================================

@dataclass(slots=True)
class MultiPeriodAttributionResult(
    AttributionResult,
):
    """
    Multi-period linked attribution.
    """

    linking_method: LinkingMethod = (
        LinkingMethod.GEOMETRIC
    )

    period_contributions: pd.DataFrame | None = None

    linked_contributions: pd.Series | None = None

    cumulative_active_return: float = 0.0

    cumulative_portfolio_return: float = 0.0

    cumulative_benchmark_return: float = 0.0


# ============================================================
# MASTER ATTRIBUTION REPORT
# ============================================================

@dataclass(slots=True)
class InstitutionalAttributionReport(
    AttributionBase,
):
    """
    Master attribution report.

    Aggregates all attribution modules.
    """

    return_attribution: (
        ReturnAttributionResult | None
    ) = None

    brinson_attribution: (
        BrinsonAttributionResult | None
    ) = None

    factor_attribution: (
        FactorAttributionResult | None
    ) = None

    risk_attribution: (
        RiskAttributionResult | None
    ) = None

    trade_attribution: (
        TradeAttributionResult | None
    ) = None

    rebalance_attribution: (
        RebalanceAttributionResult | None
    ) = None

    execution_attribution: (
        ExecutionAttributionResult | None
    ) = None

    multi_period_attribution: (
        MultiPeriodAttributionResult | None
    ) = None

    report_timestamp: datetime = field(
        default_factory=datetime.utcnow
    )

    summary_metrics: dict[str, Any] = field(
        default_factory=dict
    )

# ============================================================
# PART 3
# RETURN ATTRIBUTION
# ============================================================

# ============================================================
# RETURN ATTRIBUTION ANALYZER
# ============================================================

class ReturnAttributionAnalyzer(
    BaseAttribution,
):
    """
    Institutional return attribution.

    Decomposes:

        Portfolio Return

            =
              Asset Contributions
            + Cash Contribution
            + Residual

    Supports:

        Asset Attribution
        Sector Attribution
        Active Return Attribution
    """

    # --------------------------------------------------------
    # Asset Contributions
    # --------------------------------------------------------

    @staticmethod
    def asset_contributions(
        weights: pd.Series,
        returns: pd.Series,
    ) -> pd.Series:

        AttributionUtils.validate_same_length(
            weights,
            returns,
        )

        return (
            weights
            * returns
        )

    # --------------------------------------------------------
    # Sector Contributions
    # --------------------------------------------------------

    @staticmethod
    def sector_contributions(
        contributions: pd.Series,
        sectors: pd.Series,
    ) -> pd.Series:

        tmp = pd.DataFrame({

            "sector":
            sectors,

            "contribution":
            contributions,

        })

        return (
            tmp
            .groupby(
                "sector"
            )
            ["contribution"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

    # --------------------------------------------------------
    # Cash Contribution
    # --------------------------------------------------------

    @staticmethod
    def cash_contribution(
        cash_weight: float,
        cash_return: float = 0.0,
    ) -> float:

        return float(
            cash_weight
            * cash_return
        )

    # --------------------------------------------------------
    # Active Return
    # --------------------------------------------------------

    @staticmethod
    def active_return(
        portfolio_return: float,
        benchmark_return: float,
    ) -> float:

        return float(
            portfolio_return
            -
            benchmark_return
        )

    # --------------------------------------------------------
    # Residual
    # --------------------------------------------------------

    @staticmethod
    def residual(
        portfolio_return: float,
        contributions: pd.Series,
        cash_contribution: float,
    ) -> float:

        explained = (

            float(
                contributions.sum()
            )

            + cash_contribution

        )

        return float(
            portfolio_return
            -
            explained
        )

    # --------------------------------------------------------
    # Attribution Table
    # --------------------------------------------------------

    @staticmethod
    def build_contribution_table(
        asset_names: pd.Index,
        weights: pd.Series,
        returns: pd.Series,
        contributions: pd.Series,
    ) -> pd.DataFrame:

        return pd.DataFrame({

            "Weight":
            weights,

            "Return":
            returns,

            "Contribution":
            contributions,

        }).set_index(
            asset_names
        )

    # --------------------------------------------------------
    # Main Analysis
    # --------------------------------------------------------

    def analyze(
        self,
        *,
        weights: pd.Series,
        returns: pd.Series,
        sectors: pd.Series | None = None,
        benchmark_return: float = 0.0,
        cash_weight: float = 0.0,
        cash_return: float = 0.0,
    ) -> ReturnAttributionResult:

        weights = (
            AttributionUtils
            .ensure_series(
                weights,
                "weights",
            )
        )

        returns = (
            AttributionUtils
            .ensure_series(
                returns,
                "returns",
            )
        )

        AttributionUtils.validate_same_length(
            weights,
            returns,
        )

        AttributionUtils.validate_no_nan(
            weights,
            "weights",
        )

        AttributionUtils.validate_no_nan(
            returns,
            "returns",
        )

        # ----------------------------------
        # Asset Contributions
        # ----------------------------------

        contributions = (
            self.asset_contributions(
                weights,
                returns,
            )
        )

        # ----------------------------------
        # Portfolio Return
        # ----------------------------------

        portfolio_return = float(
            contributions.sum()
        )

        # ----------------------------------
        # Cash
        # ----------------------------------

        cash_contrib = (
            self.cash_contribution(
                cash_weight,
                cash_return,
            )
        )

        portfolio_return += (
            cash_contrib
        )

        # ----------------------------------
        # Active Return
        # ----------------------------------

        active = (
            self.active_return(
                portfolio_return,
                benchmark_return,
            )
        )

        # ----------------------------------
        # Sector Attribution
        # ----------------------------------

        sector_contrib = None

        if sectors is not None:

            sector_contrib = (
                self.sector_contributions(
                    contributions,
                    sectors,
                )
            )

        # ----------------------------------
        # Residual
        # ----------------------------------

        residual_return = (
            self.residual(
                portfolio_return,
                contributions,
                cash_contrib,
            )
        )

        # ----------------------------------
        # Contribution Table
        # ----------------------------------

        contribution_table = (
            self.build_contribution_table(
                asset_names=
                weights.index,

                weights=
                weights,

                returns=
                returns,

                contributions=
                contributions,
            )
        )

        # ----------------------------------
        # Result
        # ----------------------------------

        return ReturnAttributionResult(

            metadata=
            self.metadata,

            attribution_type=
            AttributionType.RETURN,

            attribution_name=
            "Return Attribution",

            total_portfolio_return=
            portfolio_return,

            total_benchmark_return=
            benchmark_return,

            active_return=
            active,

            contribution_table=
            contribution_table,

            contribution_series=
            contributions,

            residual=
            residual_return,

            asset_contributions=
            contributions,

            sector_contributions=
            sector_contrib,

            cash_contribution=
            cash_contrib,

            residual_contribution=
            residual_return,

            diagnostics={

                "num_assets":
                len(weights),

                "cash_weight":
                cash_weight,

                "explained_return":
                float(
                    contributions.sum()
                ),

            },
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def return_attribution(
    *,
    metadata: AttributionMetadata,
    weights: pd.Series,
    returns: pd.Series,
    sectors: pd.Series | None = None,
    benchmark_return: float = 0.0,
    cash_weight: float = 0.0,
    cash_return: float = 0.0,
    config: AttributionConfig | None = None,
) -> ReturnAttributionResult:

    analyzer = ReturnAttributionAnalyzer(
        metadata=metadata,
        config=config,
    )

    return analyzer.analyze(
        weights=weights,
        returns=returns,
        sectors=sectors,
        benchmark_return=benchmark_return,
        cash_weight=cash_weight,
        cash_return=cash_return,
    )


# ============================================================
# PART 4
# BRINSON ATTRIBUTION
# ============================================================

# ============================================================
# BRINSON ANALYZER
# ============================================================

class BrinsonAttributionAnalyzer(
    BaseAttribution,
):
    """
    Institutional Brinson-Fachler attribution.

    Active Return

    =
      Allocation Effect
    + Selection Effect
    + Interaction Effect
    """

    # --------------------------------------------------------
    # Build Sector Table
    # --------------------------------------------------------

    @staticmethod
    def build_sector_table(
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> pd.DataFrame:

        df = pd.DataFrame({

            "Pw":
            portfolio_weights,

            "Bw":
            benchmark_weights,

            "Pr":
            portfolio_returns,

            "Br":
            benchmark_returns,

        })

        return df.fillna(0.0)

    # --------------------------------------------------------
    # Allocation Effect
    # --------------------------------------------------------

    @staticmethod
    def allocation_effect(
        sector_table: pd.DataFrame,
        benchmark_total_return: float,
    ) -> pd.Series:

        return (

            (
                sector_table["Pw"]
                -
                sector_table["Bw"]
            )

            *

            (
                sector_table["Br"]
                -
                benchmark_total_return
            )

        )

    # --------------------------------------------------------
    # Selection Effect
    # --------------------------------------------------------

    @staticmethod
    def selection_effect(
        sector_table: pd.DataFrame,
    ) -> pd.Series:

        return (

            sector_table["Bw"]

            *

            (
                sector_table["Pr"]
                -
                sector_table["Br"]
            )

        )

    # --------------------------------------------------------
    # Interaction Effect
    # --------------------------------------------------------

    @staticmethod
    def interaction_effect(
        sector_table: pd.DataFrame,
    ) -> pd.Series:

        return (

            (
                sector_table["Pw"]
                -
                sector_table["Bw"]
            )

            *

            (
                sector_table["Pr"]
                -
                sector_table["Br"]
            )

        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    def analyze(
        self,
        *,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> BrinsonAttributionResult:

        AttributionUtils.validate_same_length(

            portfolio_weights,
            benchmark_weights,
            portfolio_returns,
            benchmark_returns,

        )

        sector_table = (

            self.build_sector_table(

                portfolio_weights=
                portfolio_weights,

                benchmark_weights=
                benchmark_weights,

                portfolio_returns=
                portfolio_returns,

                benchmark_returns=
                benchmark_returns,

            )

        )

        # ----------------------------------
        # Portfolio / Benchmark Returns
        # ----------------------------------

        portfolio_total_return = float(

            (
                sector_table["Pw"]
                *
                sector_table["Pr"]
            ).sum()

        )

        benchmark_total_return = float(

            (
                sector_table["Bw"]
                *
                sector_table["Br"]
            ).sum()

        )

        active_return = (

            portfolio_total_return
            -
            benchmark_total_return

        )

        # ----------------------------------
        # Effects
        # ----------------------------------

        allocation = (

            self.allocation_effect(
                sector_table,
                benchmark_total_return,
            )

        )

        selection = (

            self.selection_effect(
                sector_table,
            )

        )

        interaction = (

            self.interaction_effect(
                sector_table,
            )

        )

        allocation_total = float(
            allocation.sum()
        )

        selection_total = float(
            selection.sum()
        )

        interaction_total = float(
            interaction.sum()
        )

        explained = (

            allocation_total
            +
            selection_total
            +
            interaction_total

        )

        residual = (
            active_return
            -
            explained
        )

        # ----------------------------------
        # Sector Table
        # ----------------------------------

        sector_effects = pd.DataFrame({

            "PortfolioWeight":
            sector_table["Pw"],

            "BenchmarkWeight":
            sector_table["Bw"],

            "PortfolioReturn":
            sector_table["Pr"],

            "BenchmarkReturn":
            sector_table["Br"],

            "AllocationEffect":
            allocation,

            "SelectionEffect":
            selection,

            "InteractionEffect":
            interaction,

        })

        # ----------------------------------
        # Result
        # ----------------------------------

        return BrinsonAttributionResult(

            metadata=
            self.metadata,

            attribution_type=
            AttributionType.BRINSON,

            attribution_name=
            "Brinson Attribution",

            total_portfolio_return=
            portfolio_total_return,

            total_benchmark_return=
            benchmark_total_return,

            active_return=
            active_return,

            residual=
            residual,

            allocation_effect=
            allocation_total,

            selection_effect=
            selection_total,

            interaction_effect=
            interaction_total,

            sector_effects=
            sector_effects,

            allocation_by_sector=
            allocation,

            selection_by_sector=
            selection,

            interaction_by_sector=
            interaction,

            diagnostics={

                "explained":
                explained,

                "active_return":
                active_return,

                "residual":
                residual,

            },
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def brinson_attribution(
    *,
    metadata: AttributionMetadata,
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    config: AttributionConfig | None = None,
) -> BrinsonAttributionResult:

    analyzer = (
        BrinsonAttributionAnalyzer(
            metadata=metadata,
            config=config,
        )
    )

    return analyzer.analyze(

        portfolio_weights=
        portfolio_weights,

        benchmark_weights=
        benchmark_weights,

        portfolio_returns=
        portfolio_returns,

        benchmark_returns=
        benchmark_returns,

    )

# ============================================================
# PART 5
# FACTOR ATTRIBUTION
# ============================================================

# ============================================================
# FACTOR ATTRIBUTION ANALYZER
# ============================================================

class FactorAttributionAnalyzer(
    BaseAttribution,
):
    """
    Institutional factor attribution.

    Portfolio Return

    =
        Sum(Factor Exposure × Factor Return)
        + Residual

    Used by:

        PM Reports
        Risk Reports
        CIO Reports
        Investment Committee
    """

    # --------------------------------------------------------
    # Factor Contribution
    # --------------------------------------------------------

    @staticmethod
    def factor_contributions(
        factor_exposures: pd.Series,
        factor_returns: pd.Series,
    ) -> pd.Series:

        AttributionUtils.validate_same_length(
            factor_exposures,
            factor_returns,
        )

        return (
            factor_exposures
            * factor_returns
        )

    # --------------------------------------------------------
    # Explained Return
    # --------------------------------------------------------

    @staticmethod
    def explained_return(
        factor_contributions: pd.Series,
    ) -> float:

        return float(
            factor_contributions.sum()
        )

    # --------------------------------------------------------
    # Residual
    # --------------------------------------------------------

    @staticmethod
    def residual_return(
        portfolio_return: float,
        explained_return: float,
    ) -> float:

        return float(
            portfolio_return
            -
            explained_return
        )

    # --------------------------------------------------------
    # Factor Table
    # --------------------------------------------------------

    @staticmethod
    def build_factor_table(
        factor_exposures: pd.Series,
        factor_returns: pd.Series,
        factor_contributions: pd.Series,
    ) -> pd.DataFrame:

        return pd.DataFrame({

            "Exposure":
            factor_exposures,

            "FactorReturn":
            factor_returns,

            "Contribution":
            factor_contributions,

        })

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    def analyze(
        self,
        *,
        portfolio_return: float,
        factor_exposures: pd.Series,
        factor_returns: pd.Series,
        factor_covariance: pd.DataFrame | None = None,
    ) -> FactorAttributionResult:

        factor_exposures = (
            AttributionUtils.ensure_series(
                factor_exposures,
                "factor_exposures",
            )
        )

        factor_returns = (
            AttributionUtils.ensure_series(
                factor_returns,
                "factor_returns",
            )
        )

        AttributionUtils.validate_same_length(
            factor_exposures,
            factor_returns,
        )

        # ----------------------------------
        # Contributions
        # ----------------------------------

        contributions = (
            self.factor_contributions(
                factor_exposures,
                factor_returns,
            )
        )

        explained = (
            self.explained_return(
                contributions,
            )
        )

        residual = (
            self.residual_return(
                portfolio_return,
                explained,
            )
        )

        # ----------------------------------
        # Table
        # ----------------------------------

        factor_table = (
            self.build_factor_table(
                factor_exposures=
                factor_exposures,

                factor_returns=
                factor_returns,

                factor_contributions=
                contributions,
            )
        )

        # ----------------------------------
        # Active Return
        # ----------------------------------

        active_return = float(
            portfolio_return
        )

        # ----------------------------------
        # Result
        # ----------------------------------

        return FactorAttributionResult(

            metadata=
            self.metadata,

            attribution_type=
            AttributionType.FACTOR,

            attribution_name=
            "Factor Attribution",

            total_portfolio_return=
            portfolio_return,

            total_benchmark_return=
            0.0,

            active_return=
            active_return,

            contribution_table=
            factor_table,

            contribution_series=
            contributions,

            residual=
            residual,

            factor_exposures=
            factor_exposures,

            factor_returns=
            factor_returns,

            factor_contributions=
            contributions,

            factor_covariance=
            factor_covariance,

            explained_return=
            explained,

            residual_return=
            residual,

            diagnostics={

                "num_factors":
                len(
                    factor_exposures
                ),

                "explained_return":
                explained,

                "residual_return":
                residual,

                "explanatory_power":
                AttributionUtils.safe_divide(
                    explained,
                    portfolio_return
                    if abs(
                        portfolio_return
                    ) > EPSILON
                    else 1.0,
                ),
            },
        )


# ============================================================
# FACTOR GROUPING UTILITIES
# ============================================================

class FactorGrouping:
    """
    Institutional factor grouping.
    """

    VALUE = {
        "Value",
        "BookToPrice",
        "EarningsYield",
        "CFYield",
    }

    MOMENTUM = {
        "Momentum",
        "PriceMomentum",
        "ResidualMomentum",
    }

    QUALITY = {
        "Quality",
        "ROE",
        "ROA",
        "Profitability",
    }

    SIZE = {
        "Size",
        "MarketCap",
    }

    LOW_VOL = {
        "LowVol",
        "Volatility",
    }

    # --------------------------------------------------------

    @classmethod
    def group_contributions(
        cls,
        contributions: pd.Series,
    ) -> pd.Series:

        grouped = {}

        for group_name, factors in {

            "Value":
            cls.VALUE,

            "Momentum":
            cls.MOMENTUM,

            "Quality":
            cls.QUALITY,

            "Size":
            cls.SIZE,

            "LowVol":
            cls.LOW_VOL,

        }.items():

            grouped[group_name] = float(

                contributions[
                    contributions.index
                    .isin(factors)
                ].sum()

            )

        return pd.Series(
            grouped
        )


# ============================================================
# FACTOR SUMMARY REPORT
# ============================================================

@dataclass(slots=True)
class FactorSummary:
    """
    High-level factor summary.
    """

    value_contribution: float = 0.0

    momentum_contribution: float = 0.0

    quality_contribution: float = 0.0

    size_contribution: float = 0.0

    low_vol_contribution: float = 0.0

    residual_return: float = 0.0


# ------------------------------------------------------------

def summarize_factor_attribution(
    result: FactorAttributionResult,
) -> FactorSummary:

    grouped = (
        FactorGrouping
        .group_contributions(
            result.factor_contributions
        )
    )

    return FactorSummary(

        value_contribution=
        float(
            grouped.get(
                "Value",
                0.0,
            )
        ),

        momentum_contribution=
        float(
            grouped.get(
                "Momentum",
                0.0,
            )
        ),

        quality_contribution=
        float(
            grouped.get(
                "Quality",
                0.0,
            )
        ),

        size_contribution=
        float(
            grouped.get(
                "Size",
                0.0,
            )
        ),

        low_vol_contribution=
        float(
            grouped.get(
                "LowVol",
                0.0,
            )
        ),

        residual_return=
        result.residual_return,
    )


# ============================================================
# CONVENIENCE API
# ============================================================

def factor_attribution(
    *,
    metadata: AttributionMetadata,
    portfolio_return: float,
    factor_exposures: pd.Series,
    factor_returns: pd.Series,
    factor_covariance: pd.DataFrame | None = None,
    config: AttributionConfig | None = None,
) -> FactorAttributionResult:

    analyzer = (
        FactorAttributionAnalyzer(
            metadata=metadata,
            config=config,
        )
    )

    return analyzer.analyze(

        portfolio_return=
        portfolio_return,

        factor_exposures=
        factor_exposures,

        factor_returns=
        factor_returns,

        factor_covariance=
        factor_covariance,
    )

# ============================================================
# PART 6
# RISK ATTRIBUTION
# ============================================================

# ============================================================
# RISK ATTRIBUTION ANALYZER
# ============================================================

class RiskAttributionAnalyzer(
    BaseAttribution,
):
    """
    Institutional risk attribution.

    Produces:

        MCTR
        CCTR
        PCTR

    Risk decomposition:

        Portfolio Risk
            =
            Factor Risk
            +
            Specific Risk
    """

    # --------------------------------------------------------
    # Portfolio Volatility
    # --------------------------------------------------------

    @staticmethod
    def portfolio_volatility(
        weights: np.ndarray,
        covariance: np.ndarray,
    ) -> float:

        variance = float(

            weights.T
            @ covariance
            @ weights

        )

        return float(
            np.sqrt(
                max(
                    variance,
                    0.0,
                )
            )
        )

    # --------------------------------------------------------
    # Marginal Contribution To Risk
    # --------------------------------------------------------

    @staticmethod
    def marginal_contribution_to_risk(
        weights: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:

        sigma = (
            RiskAttributionAnalyzer
            .portfolio_volatility(
                weights,
                covariance,
            )
        )

        if sigma < EPSILON:

            return np.zeros(
                len(weights)
            )

        return (
            covariance
            @ weights
        ) / sigma

    # --------------------------------------------------------
    # Component Contribution To Risk
    # --------------------------------------------------------

    @staticmethod
    def component_contribution_to_risk(
        weights: np.ndarray,
        marginal_risk: np.ndarray,
    ) -> np.ndarray:

        return (
            weights
            * marginal_risk
        )

    # --------------------------------------------------------
    # Percent Contribution To Risk
    # --------------------------------------------------------

    @staticmethod
    def percent_contribution_to_risk(
        component_risk: np.ndarray,
    ) -> np.ndarray:

        total = float(
            component_risk.sum()
        )

        if abs(
            total
        ) < EPSILON:

            return np.zeros(
                len(component_risk)
            )

        return (
            component_risk
            / total
        )

    # --------------------------------------------------------
    # Factor Risk
    # --------------------------------------------------------

    @staticmethod
    def factor_risk_contribution(
        factor_exposures: pd.Series,
        factor_covariance: pd.DataFrame,
    ) -> pd.Series:

        factors = (
            factor_exposures.index
        )

        common = [

            f
            for f in factors
            if f in factor_covariance.index
        ]

        if len(common) == 0:

            return pd.Series(
                dtype=float
            )

        b = (
            factor_exposures
            .loc[common]
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

        total_factor_var = float(
            b.T
            @ cov
            @ b
        )

        if total_factor_var < EPSILON:

            return pd.Series(
                0.0,
                index=common,
            )

        marginal = (
            cov @ b
        )

        contribution = (
            b * marginal
        )

        return pd.Series(
            contribution,
            index=common,
        )

    # --------------------------------------------------------
    # Specific Risk
    # --------------------------------------------------------

    @staticmethod
    def specific_risk(
        total_risk: float,
        factor_risk: float,
    ) -> float:

        return float(
            max(
                total_risk
                -
                factor_risk,
                0.0,
            )
        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    def analyze(
        self,
        *,
        weights: pd.Series,
        covariance_matrix: pd.DataFrame,
        factor_exposures: pd.Series | None = None,
        factor_covariance: pd.DataFrame | None = None,
    ) -> RiskAttributionResult:

        weights = (
            AttributionUtils.ensure_series(
                weights,
                "weights",
            )
        )

        if not isinstance(
            covariance_matrix,
            pd.DataFrame,
        ):
            raise TypeError(
                "covariance_matrix must be DataFrame."
            )

        common_assets = [

            a
            for a in weights.index
            if a in covariance_matrix.index
        ]

        weights = (
            weights
            .loc[common_assets]
        )

        covariance_matrix = (
            covariance_matrix
            .loc[
                common_assets,
                common_assets,
            ]
        )

        w = weights.values

        cov = (
            covariance_matrix
            .values
        )

        # ----------------------------------
        # Total Risk
        # ----------------------------------

        total_risk = (
            self.portfolio_volatility(
                w,
                cov,
            )
        )

        # ----------------------------------
        # MCTR
        # ----------------------------------

        mctr = (
            self.marginal_contribution_to_risk(
                w,
                cov,
            )
        )

        # ----------------------------------
        # CCTR
        # ----------------------------------

        cctr = (
            self.component_contribution_to_risk(
                w,
                mctr,
            )
        )

        # ----------------------------------
        # PCTR
        # ----------------------------------

        pctr = (
            self.percent_contribution_to_risk(
                cctr,
            )
        )

        # ----------------------------------
        # Factor Risk
        # ----------------------------------

        factor_contrib = None

        factor_risk_total = 0.0

        if (

            factor_exposures is not None
            and
            factor_covariance is not None

        ):

            factor_contrib = (
                self.factor_risk_contribution(
                    factor_exposures,
                    factor_covariance,
                )
            )

            factor_risk_total = float(
                factor_contrib.sum()
            )

        # ----------------------------------
        # Specific Risk
        # ----------------------------------

        specific_risk = (
            self.specific_risk(
                total_risk,
                factor_risk_total,
            )
        )

        # ----------------------------------
        # Table
        # ----------------------------------

        risk_table = pd.DataFrame({

            "Weight":
            weights,

            "MCTR":
            mctr,

            "CCTR":
            cctr,

            "PCTR":
            pctr,

        })

        # ----------------------------------
        # Result
        # ----------------------------------

        return RiskAttributionResult(

            metadata=
            self.metadata,

            attribution_type=
            AttributionType.RISK,

            attribution_name=
            "Risk Attribution",

            total_risk=
            total_risk,

            contribution_table=
            risk_table,

            marginal_contribution_to_risk=
            pd.Series(
                mctr,
                index=weights.index,
            ),

            component_contribution_to_risk=
            pd.Series(
                cctr,
                index=weights.index,
            ),

            percent_contribution_to_risk=
            pd.Series(
                pctr,
                index=weights.index,
            ),

            factor_risk_contribution=
            factor_contrib,

            specific_risk_contribution=
            specific_risk,

            diagnostics={

                "num_assets":
                len(weights),

                "portfolio_vol":
                total_risk,

                "factor_risk":
                factor_risk_total,

                "specific_risk":
                specific_risk,

            },
        )


# ============================================================
# RISK SUMMARY
# ============================================================

@dataclass(slots=True)
class RiskAttributionSummary:

    total_risk: float

    factor_risk: float

    specific_risk: float

    largest_risk_contributor: str

    largest_risk_contribution: float


# ------------------------------------------------------------

def summarize_risk_attribution(
    result: RiskAttributionResult,
) -> RiskAttributionSummary:

    largest_name = ""

    largest_value = 0.0

    if (
        result.percent_contribution_to_risk
        is not None
        and
        len(
            result.percent_contribution_to_risk
        ) > 0
    ):

        largest_name = (
            result.percent_contribution_to_risk
            .idxmax()
        )

        largest_value = float(

            result.percent_contribution_to_risk
            .max()

        )

    factor_risk = 0.0

    if (
        result.factor_risk_contribution
        is not None
    ):

        factor_risk = float(
            result.factor_risk_contribution
            .sum()
        )

    return RiskAttributionSummary(

        total_risk=
        result.total_risk,

        factor_risk=
        factor_risk,

        specific_risk=
        result.specific_risk_contribution,

        largest_risk_contributor=
        largest_name,

        largest_risk_contribution=
        largest_value,
    )


# ============================================================
# CONVENIENCE API
# ============================================================

def risk_attribution(
    *,
    metadata: AttributionMetadata,
    weights: pd.Series,
    covariance_matrix: pd.DataFrame,
    factor_exposures: pd.Series | None = None,
    factor_covariance: pd.DataFrame | None = None,
    config: AttributionConfig | None = None,
) -> RiskAttributionResult:

    analyzer = (
        RiskAttributionAnalyzer(
            metadata=metadata,
            config=config,
        )
    )

    return analyzer.analyze(

        weights=
        weights,

        covariance_matrix=
        covariance_matrix,

        factor_exposures=
        factor_exposures,

        factor_covariance=
        factor_covariance,
    )


# ============================================================
# PART 7
# TRADE ATTRIBUTION
# ============================================================

# ============================================================
# TRADE ATTRIBUTION ANALYZER
# ============================================================

class TradeAttributionAnalyzer(
    BaseAttribution,
):
    """
    Institutional trade attribution.

    Trade-level attribution engine.
    """

    REQUIRED_COLUMNS = [

        "symbol",
        "entry_price",
        "exit_price",
        "quantity",

    ]

    OPTIONAL_COLUMNS = [

        "signal",
        "benchmark_return",
        "trade_return",

    ]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def validate_trade_table(
        self,
        trades: pd.DataFrame,
    ) -> None:

        self.validate_dataframe(
            trades,
            "trades",
        )

        self.validate_columns(
            trades,
            self.REQUIRED_COLUMNS,
            "trades",
        )

    # --------------------------------------------------------
    # Trade Return
    # --------------------------------------------------------

    @staticmethod
    def trade_return(
        trades: pd.DataFrame,
    ) -> pd.Series:

        return (

            trades["exit_price"]

            /
            trades["entry_price"]

            - 1.0

        )

    # --------------------------------------------------------
    # Trade PnL
    # --------------------------------------------------------

    @staticmethod
    def trade_pnl(
        trades: pd.DataFrame,
    ) -> pd.Series:

        return (

            (
                trades["exit_price"]

                -
                trades["entry_price"]

            )

            *

            trades["quantity"]

        )

    # --------------------------------------------------------
    # Trade Alpha
    # --------------------------------------------------------

    @staticmethod
    def trade_alpha(
        trade_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> pd.Series:

        AttributionUtils.validate_same_length(

            trade_returns,
            benchmark_returns,

        )

        return (
            trade_returns
            -
            benchmark_returns
        )

    # --------------------------------------------------------
    # Hit Rate
    # --------------------------------------------------------

    @staticmethod
    def hit_rate(
        pnl: pd.Series,
    ) -> float:

        if len(
            pnl
        ) == 0:

            return 0.0

        return float(

            (
                pnl > 0
            ).mean()

        )

    # --------------------------------------------------------
    # Win Loss Ratio
    # --------------------------------------------------------

    @staticmethod
    def win_loss_ratio(
        pnl: pd.Series,
    ) -> float:

        winners = pnl[
            pnl > 0
        ]

        losers = pnl[
            pnl < 0
        ]

        if (

            len(losers)
            == 0

        ):

            return np.inf

        avg_win = float(
            winners.mean()
        ) if len(winners) else 0.0

        avg_loss = float(
            abs(
                losers.mean()
            )
        )

        if avg_loss < EPSILON:

            return np.inf

        return (
            avg_win
            /
            avg_loss
        )

    # --------------------------------------------------------
    # Signal Effectiveness
    # --------------------------------------------------------

    @staticmethod
    def signal_effectiveness(
        signals: pd.Series,
        pnl: pd.Series,
    ) -> float:

        if len(
            signals
        ) == 0:

            return 0.0

        try:

            corr = np.corrcoef(

                signals,
                pnl,

            )[0, 1]

            if np.isnan(
                corr
            ):
                return 0.0

            return float(
                corr
            )

        except Exception:

            return 0.0

    # --------------------------------------------------------
    # Trade Contribution
    # --------------------------------------------------------

    @staticmethod
    def trade_contributions(
        pnl: pd.Series,
    ) -> pd.Series:

        total = float(
            pnl.sum()
        )

        if abs(
            total
        ) < EPSILON:

            return pnl * 0.0

        return (
            pnl
            / total
        )

    # --------------------------------------------------------
    # Main Analysis
    # --------------------------------------------------------

    def analyze(
        self,
        *,
        trades: pd.DataFrame,
    ) -> TradeAttributionResult:

        self.validate_trade_table(
            trades,
        )

        trades = trades.copy()

        # ----------------------------------
        # Returns
        # ----------------------------------

        trades["trade_return"] = (

            self.trade_return(
                trades,
            )

        )

        # ----------------------------------
        # PnL
        # ----------------------------------

        trades["trade_pnl"] = (

            self.trade_pnl(
                trades,
            )

        )

        pnl = trades[
            "trade_pnl"
        ]

        total_trade_pnl = float(
            pnl.sum()
        )

        avg_trade_pnl = float(
            pnl.mean()
        )

        # ----------------------------------
        # Alpha
        # ----------------------------------

        alpha_series = None

        trade_alpha_total = 0.0

        if (
            "benchmark_return"
            in trades.columns
        ):

            alpha_series = (

                self.trade_alpha(

                    trades[
                        "trade_return"
                    ],

                    trades[
                        "benchmark_return"
                    ],

                )

            )

            trade_alpha_total = float(
                alpha_series.sum()
            )

        # ----------------------------------
        # Hit Rate
        # ----------------------------------

        hit_rate = (
            self.hit_rate(
                pnl,
            )
        )

        # ----------------------------------
        # Win Loss Ratio
        # ----------------------------------

        wl_ratio = (
            self.win_loss_ratio(
                pnl,
            )
        )

        # ----------------------------------
        # Signal Effectiveness
        # ----------------------------------

        signal_effect = 0.0

        if (
            "signal"
            in trades.columns
        ):

            signal_effect = (

                self.signal_effectiveness(

                    trades["signal"],

                    pnl,

                )

            )

        # ----------------------------------
        # Contributions
        # ----------------------------------

        contributions = (

            self.trade_contributions(
                pnl,
            )

        )

        trades["contribution"] = (
            contributions
        )

        # ----------------------------------
        # Portfolio Return Proxy
        # ----------------------------------

        portfolio_return = float(
            trades[
                "trade_return"
            ].sum()
        )

        # ----------------------------------
        # Result
        # ----------------------------------

        return TradeAttributionResult(

            metadata=
            self.metadata,

            attribution_type=
            AttributionType.TRADE,

            attribution_name=
            "Trade Attribution",

            total_portfolio_return=
            portfolio_return,

            total_trade_pnl=
            total_trade_pnl,

            average_trade_pnl=
            avg_trade_pnl,

            hit_rate=
            hit_rate,

            win_loss_ratio=
            wl_ratio,

            trade_alpha=
            trade_alpha_total,

            trade_contributions=
            contributions,

            trade_table=
            trades,

            contribution_table=
            trades,

            contribution_series=
            contributions,

            diagnostics={

                "num_trades":
                len(trades),

                "signal_effectiveness":
                signal_effect,

                "trade_alpha":
                trade_alpha_total,

            },
        )


# ============================================================
# TRADE SUMMARY
# ============================================================

@dataclass(slots=True)
class TradeAttributionSummary:

    total_trade_pnl: float

    average_trade_pnl: float

    hit_rate: float

    win_loss_ratio: float

    trade_alpha: float

    signal_effectiveness: float


# ------------------------------------------------------------

def summarize_trade_attribution(
    result: TradeAttributionResult,
) -> TradeAttributionSummary:

    signal_effect = float(

        result.diagnostics.get(
            "signal_effectiveness",
            0.0,
        )

    )

    return TradeAttributionSummary(

        total_trade_pnl=
        result.total_trade_pnl,

        average_trade_pnl=
        result.average_trade_pnl,

        hit_rate=
        result.hit_rate,

        win_loss_ratio=
        result.win_loss_ratio,

        trade_alpha=
        result.trade_alpha,

        signal_effectiveness=
        signal_effect,
    )


# ============================================================
# CONVENIENCE API
# ============================================================

def trade_attribution(
    *,
    metadata: AttributionMetadata,
    trades: pd.DataFrame,
    config: AttributionConfig | None = None,
) -> TradeAttributionResult:

    analyzer = (
        TradeAttributionAnalyzer(
            metadata=metadata,
            config=config,
        )
    )

    return analyzer.analyze(
        trades=trades,
    )

# ============================================================
# PART 8
# REBALANCE ATTRIBUTION
# ============================================================

# ============================================================
# REBALANCE ATTRIBUTION ANALYZER
# ============================================================

class RebalanceAttributionAnalyzer(
    BaseAttribution,
):
    """
    Institutional rebalance attribution.

    Quantifies the impact of portfolio rebalancing
    on performance and risk.
    """

    REQUIRED_COLUMNS = [

        "symbol",

        "old_weight",

        "new_weight",

    ]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def validate_rebalance_table(
        self,
        rebalance_df: pd.DataFrame,
    ) -> None:

        self.validate_dataframe(
            rebalance_df,
            "rebalance_df",
        )

        self.validate_columns(
            rebalance_df,
            self.REQUIRED_COLUMNS,
            "rebalance_df",
        )

    # --------------------------------------------------------
    # Turnover
    # --------------------------------------------------------

    @staticmethod
    def turnover(
        old_weights: pd.Series,
        new_weights: pd.Series,
    ) -> float:

        AttributionUtils.validate_same_length(

            old_weights,
            new_weights,

        )

        return float(

            0.5
            *
            np.abs(
                new_weights
                -
                old_weights
            ).sum()

        )

    # --------------------------------------------------------
    # Drift Reduction
    # --------------------------------------------------------

    @staticmethod
    def drift_reduction_effect(
        target_weights: pd.Series,
        old_weights: pd.Series,
        new_weights: pd.Series,
    ) -> float:

        before = float(

            np.abs(

                old_weights
                -
                target_weights

            ).sum()

        )

        after = float(

            np.abs(

                new_weights
                -
                target_weights

            ).sum()

        )

        return float(
            before - after
        )

    # --------------------------------------------------------
    # Rebalance Alpha
    # --------------------------------------------------------

    @staticmethod
    def rebalance_alpha(
        expected_returns: pd.Series,
        old_weights: pd.Series,
        new_weights: pd.Series,
    ) -> float:

        AttributionUtils.validate_same_length(

            expected_returns,
            old_weights,
            new_weights,

        )

        delta_weights = (
            new_weights
            -
            old_weights
        )

        return float(
            (
                delta_weights
                * expected_returns
            ).sum()
        )

    # --------------------------------------------------------
    # Rebalance Cost
    # --------------------------------------------------------

    @staticmethod
    def rebalance_cost(
        turnover: float,
        transaction_cost_bps: float,
    ) -> float:

        return float(

            turnover

            *

            (
                transaction_cost_bps
                / 10000.0
            )

        )

    # --------------------------------------------------------
    # Contribution
    # --------------------------------------------------------

    @staticmethod
    def rebalance_contributions(
        rebalance_df: pd.DataFrame,
        expected_returns: pd.Series,
    ) -> pd.Series:

        delta = (

            rebalance_df[
                "new_weight"
            ]

            -

            rebalance_df[
                "old_weight"
            ]

        )

        return (
            delta
            * expected_returns
        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    def analyze(
        self,
        *,
        rebalance_df: pd.DataFrame,
        expected_returns: pd.Series,
        target_weights: pd.Series | None = None,
        transaction_cost_bps: float = 5.0,
    ) -> RebalanceAttributionResult:

        self.validate_rebalance_table(
            rebalance_df,
        )

        rebalance_df = (
            rebalance_df.copy()
        )

        old_weights = (
            rebalance_df[
                "old_weight"
            ]
        )

        new_weights = (
            rebalance_df[
                "new_weight"
            ]
        )

        # ----------------------------------
        # Turnover
        # ----------------------------------

        turnover_effect = (
            self.turnover(
                old_weights,
                new_weights,
            )
        )

        # ----------------------------------
        # Drift Reduction
        # ----------------------------------

        drift_effect = 0.0

        if (
            target_weights
            is not None
        ):

            common = [

                x
                for x in rebalance_df[
                    "symbol"
                ]
                if x in target_weights.index

            ]

            if len(common) > 0:

                tw = (
                    target_weights
                    .loc[common]
                    .values
                )

                ow = (
                    rebalance_df
                    .set_index(
                        "symbol"
                    )
                    .loc[
                        common,
                        "old_weight"
                    ]
                    .values
                )

                nw = (
                    rebalance_df
                    .set_index(
                        "symbol"
                    )
                    .loc[
                        common,
                        "new_weight"
                    ]
                    .values
                )

                drift_effect = (
                    self
                    .drift_reduction_effect(
                        pd.Series(tw),
                        pd.Series(ow),
                        pd.Series(nw),
                    )
                )

        # ----------------------------------
        # Alpha
        # ----------------------------------

        alpha = (
            self.rebalance_alpha(
                expected_returns=
                expected_returns,

                old_weights=
                old_weights,

                new_weights=
                new_weights,
            )
        )

        # ----------------------------------
        # Costs
        # ----------------------------------

        cost = (
            self.rebalance_cost(
                turnover_effect,
                transaction_cost_bps,
            )
        )

        # ----------------------------------
        # Contributions
        # ----------------------------------

        contributions = (
            self.rebalance_contributions(
                rebalance_df,
                expected_returns,
            )
        )

        rebalance_df[
            "rebalance_contribution"
        ] = contributions

        # ----------------------------------
        # Result
        # ----------------------------------

        return RebalanceAttributionResult(

            metadata=
            self.metadata,

            attribution_type=
            AttributionType.REBALANCE,

            attribution_name=
            "Rebalance Attribution",

            turnover_effect=
            turnover_effect,

            drift_reduction_effect=
            drift_effect,

            rebalance_alpha=
            alpha,

            rebalance_cost=
            cost,

            rebalance_contributions=
            contributions,

            contribution_table=
            rebalance_df,

            contribution_series=
            contributions,

            diagnostics={

                "num_positions":
                len(
                    rebalance_df
                ),

                "turnover":
                turnover_effect,

                "cost":
                cost,

                "alpha":
                alpha,

            },
        )


# ============================================================
# REBALANCE SUMMARY
# ============================================================

@dataclass(slots=True)
class RebalanceAttributionSummary:

    turnover_effect: float

    drift_reduction_effect: float

    rebalance_alpha: float

    rebalance_cost: float

    net_rebalance_value: float


# ------------------------------------------------------------

def summarize_rebalance_attribution(
    result: RebalanceAttributionResult,
) -> RebalanceAttributionSummary:

    net_value = (

        result.rebalance_alpha

        -

        result.rebalance_cost

    )

    return RebalanceAttributionSummary(

        turnover_effect=
        result.turnover_effect,

        drift_reduction_effect=
        result.drift_reduction_effect,

        rebalance_alpha=
        result.rebalance_alpha,

        rebalance_cost=
        result.rebalance_cost,

        net_rebalance_value=
        net_value,
    )


# ============================================================
# CONVENIENCE API
# ============================================================

def rebalance_attribution(
    *,
    metadata: AttributionMetadata,
    rebalance_df: pd.DataFrame,
    expected_returns: pd.Series,
    target_weights: pd.Series | None = None,
    transaction_cost_bps: float = 5.0,
    config: AttributionConfig | None = None,
) -> RebalanceAttributionResult:

    analyzer = (
        RebalanceAttributionAnalyzer(
            metadata=metadata,
            config=config,
        )
    )

    return analyzer.analyze(

        rebalance_df=
        rebalance_df,

        expected_returns=
        expected_returns,

        target_weights=
        target_weights,

        transaction_cost_bps=
        transaction_cost_bps,
    )

# ============================================================
# PART 9
# EXECUTION ATTRIBUTION
# ============================================================

# ============================================================
# EXECUTION ATTRIBUTION ANALYZER
# ============================================================

class ExecutionAttributionAnalyzer(
    BaseAttribution,
):
    """
    Institutional execution attribution.

    Decomposes execution costs into:

        Slippage
        Market Impact
        Opportunity Cost

    and computes:

        Execution Shortfall
        Total Execution Cost
    """

    REQUIRED_COLUMNS = [

        "symbol",
        "quantity",

        "decision_price",
        "execution_price",

    ]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def validate_execution_table(
        self,
        execution_df: pd.DataFrame,
    ) -> None:

        self.validate_dataframe(
            execution_df,
            "execution_df",
        )

        self.validate_columns(
            execution_df,
            self.REQUIRED_COLUMNS,
            "execution_df",
        )

    # --------------------------------------------------------
    # Notional
    # --------------------------------------------------------

    @staticmethod
    def notional(
        quantity: pd.Series,
        price: pd.Series,
    ) -> pd.Series:

        return (
            quantity.abs()
            * price
        )

    # --------------------------------------------------------
    # Slippage Cost
    # --------------------------------------------------------

    @staticmethod
    def slippage_cost(
        execution_df: pd.DataFrame,
    ) -> pd.Series:

        return (

            (
                execution_df[
                    "execution_price"
                ]

                -

                execution_df[
                    "decision_price"
                ]

            )

            *

            execution_df[
                "quantity"
            ]

        )

    # --------------------------------------------------------
    # Market Impact
    # --------------------------------------------------------

    @staticmethod
    def market_impact_cost(
        execution_df: pd.DataFrame,
    ) -> pd.Series:

        if (
            "market_impact_cost"
            in execution_df.columns
        ):

            return execution_df[
                "market_impact_cost"
            ]

        return pd.Series(
            0.0,
            index=execution_df.index,
        )

    # --------------------------------------------------------
    # Opportunity Cost
    # --------------------------------------------------------

    @staticmethod
    def opportunity_cost(
        execution_df: pd.DataFrame,
    ) -> pd.Series:

        if (
            "opportunity_cost"
            in execution_df.columns
        ):

            return execution_df[
                "opportunity_cost"
            ]

        return pd.Series(
            0.0,
            index=execution_df.index,
        )

    # --------------------------------------------------------
    # Execution Shortfall
    # --------------------------------------------------------

    @staticmethod
    def execution_shortfall(
        slippage: pd.Series,
        impact: pd.Series,
        opportunity: pd.Series,
    ) -> pd.Series:

        return (

            slippage

            + impact

            + opportunity

        )

    # --------------------------------------------------------
    # Contributions
    # --------------------------------------------------------

    @staticmethod
    def execution_contributions(
        shortfall: pd.Series,
    ) -> pd.Series:

        total = float(
            shortfall.sum()
        )

        if abs(
            total
        ) < EPSILON:

            return shortfall * 0.0

        return (
            shortfall
            / total
        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    def analyze(
        self,
        *,
        execution_df: pd.DataFrame,
    ) -> ExecutionAttributionResult:

        self.validate_execution_table(
            execution_df,
        )

        execution_df = (
            execution_df.copy()
        )

        # ----------------------------------
        # Notional
        # ----------------------------------

        execution_df[
            "notional"
        ] = self.notional(

            execution_df[
                "quantity"
            ],

            execution_df[
                "decision_price"
            ],

        )

        # ----------------------------------
        # Slippage
        # ----------------------------------

        slippage = (
            self.slippage_cost(
                execution_df,
            )
        )

        execution_df[
            "slippage_cost"
        ] = slippage

        # ----------------------------------
        # Impact
        # ----------------------------------

        impact = (
            self.market_impact_cost(
                execution_df,
            )
        )

        execution_df[
            "impact_cost"
        ] = impact

        # ----------------------------------
        # Opportunity
        # ----------------------------------

        opportunity = (
            self.opportunity_cost(
                execution_df,
            )
        )

        execution_df[
            "opportunity_cost"
        ] = opportunity

        # ----------------------------------
        # Shortfall
        # ----------------------------------

        shortfall = (
            self.execution_shortfall(

                slippage,

                impact,

                opportunity,

            )
        )

        execution_df[
            "execution_shortfall"
        ] = shortfall

        # ----------------------------------
        # Contributions
        # ----------------------------------

        contributions = (
            self.execution_contributions(
                shortfall,
            )
        )

        execution_df[
            "execution_contribution"
        ] = contributions

        # ----------------------------------
        # Totals
        # ----------------------------------

        total_slippage = float(
            slippage.sum()
        )

        total_impact = float(
            impact.sum()
        )

        total_opportunity = float(
            opportunity.sum()
        )

        total_shortfall = float(
            shortfall.sum()
        )

        # ----------------------------------
        # Result
        # ----------------------------------

        return ExecutionAttributionResult(

            metadata=
            self.metadata,

            attribution_type=
            AttributionType.EXECUTION,

            attribution_name=
            "Execution Attribution",

            slippage_cost=
            total_slippage,

            market_impact_cost=
            total_impact,

            opportunity_cost=
            total_opportunity,

            execution_shortfall=
            total_shortfall,

            total_execution_cost=
            total_shortfall,

            execution_contributions=
            contributions,

            contribution_table=
            execution_df,

            contribution_series=
            contributions,

            diagnostics={

                "num_orders":
                len(
                    execution_df
                ),

                "total_notional":
                float(
                    execution_df[
                        "notional"
                    ].sum()
                ),

                "avg_shortfall":
                float(
                    shortfall.mean()
                ),

            },
        )


# ============================================================
# EXECUTION SUMMARY
# ============================================================

@dataclass(slots=True)
class ExecutionAttributionSummary:

    total_slippage: float

    total_market_impact: float

    total_opportunity_cost: float

    total_execution_shortfall: float

    avg_shortfall_per_order: float


# ------------------------------------------------------------

def summarize_execution_attribution(
    result: ExecutionAttributionResult,
) -> ExecutionAttributionSummary:

    avg_shortfall = 0.0

    n_orders = int(

        result.diagnostics.get(
            "num_orders",
            0,
        )

    )

    if n_orders > 0:

        avg_shortfall = (

            result.execution_shortfall
            /
            n_orders

        )

    return ExecutionAttributionSummary(

        total_slippage=
        result.slippage_cost,

        total_market_impact=
        result.market_impact_cost,

        total_opportunity_cost=
        result.opportunity_cost,

        total_execution_shortfall=
        result.execution_shortfall,

        avg_shortfall_per_order=
        avg_shortfall,
    )


# ============================================================
# CONVENIENCE API
# ============================================================

def execution_attribution(
    *,
    metadata: AttributionMetadata,
    execution_df: pd.DataFrame,
    config: AttributionConfig | None = None,
) -> ExecutionAttributionResult:

    analyzer = (
        ExecutionAttributionAnalyzer(
            metadata=metadata,
            config=config,
        )
    )

    return analyzer.analyze(
        execution_df=
        execution_df,
    )

# ============================================================
# PART 10
# MULTI-PERIOD ATTRIBUTION
# ============================================================

# ============================================================
# MULTI-PERIOD ATTRIBUTION ANALYZER
# ============================================================

class MultiPeriodAttributionAnalyzer(
    BaseAttribution,
):
    """
    Institutional multi-period attribution.

    Supports:

        Arithmetic Linking
        Geometric Linking

    Converts multiple single-period attribution
    outputs into cumulative attribution.
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def validate_period_results(
        self,
        period_results: list[AttributionResult],
    ) -> None:

        if len(period_results) == 0:

            raise ValueError(
                "period_results cannot be empty."
            )

    # --------------------------------------------------------
    # Arithmetic Linking
    # --------------------------------------------------------

    @staticmethod
    def arithmetic_linking(
        contribution_matrix: pd.DataFrame,
    ) -> pd.Series:

        return contribution_matrix.sum(
            axis=0
        )

    # --------------------------------------------------------
    # Geometric Linking
    # --------------------------------------------------------

    @staticmethod
    def geometric_linking(
        contribution_matrix: pd.DataFrame,
    ) -> pd.Series:

        linked = {}

        for col in contribution_matrix.columns:

            vals = (
                contribution_matrix[col]
                .fillna(0.0)
            )

            linked[col] = (

                np.prod(
                    1.0 + vals
                )

                - 1.0

            )

        return pd.Series(
            linked
        )

    # --------------------------------------------------------
    # Portfolio Linking
    # --------------------------------------------------------

    @staticmethod
    def cumulative_return(
        returns: pd.Series,
        method: LinkingMethod,
    ) -> float:

        if (
            method
            ==
            LinkingMethod.ARITHMETIC
        ):

            return float(
                returns.sum()
            )

        return float(

            np.prod(
                1.0 + returns
            )

            - 1.0

        )

    # --------------------------------------------------------
    # Build Period Matrix
    # --------------------------------------------------------

    @staticmethod
    def build_contribution_matrix(
        period_results: list[
            AttributionResult
        ],
    ) -> pd.DataFrame:

        rows = []

        for result in period_results:

            if (
                result.contribution_series
                is None
            ):
                continue

            rows.append(
                result.contribution_series
            )

        if len(rows) == 0:

            return pd.DataFrame()

        return pd.DataFrame(
            rows
        ).fillna(
            0.0
        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    def analyze(
        self,
        *,
        period_results: list[
            AttributionResult
        ],
        linking_method: LinkingMethod = (
            LinkingMethod.GEOMETRIC
        ),
    ) -> MultiPeriodAttributionResult:

        self.validate_period_results(
            period_results,
        )

        # ----------------------------------
        # Contribution Matrix
        # ----------------------------------

        contribution_matrix = (

            self.build_contribution_matrix(
                period_results,
            )

        )

        # ----------------------------------
        # Linked Contributions
        # ----------------------------------

        if (
            linking_method
            ==
            LinkingMethod.ARITHMETIC
        ):

            linked_contributions = (

                self.arithmetic_linking(
                    contribution_matrix,
                )

            )

        else:

            linked_contributions = (

                self.geometric_linking(
                    contribution_matrix,
                )

            )

        # ----------------------------------
        # Returns
        # ----------------------------------

        portfolio_returns = pd.Series([

            r.total_portfolio_return

            for r in period_results

        ])

        benchmark_returns = pd.Series([

            r.total_benchmark_return

            for r in period_results

        ])

        active_returns = pd.Series([

            r.active_return

            for r in period_results

        ])

        cumulative_portfolio = (

            self.cumulative_return(

                portfolio_returns,
                linking_method,

            )

        )

        cumulative_benchmark = (

            self.cumulative_return(

                benchmark_returns,
                linking_method,

            )

        )

        cumulative_active = (

            self.cumulative_return(

                active_returns,
                linking_method,

            )

        )

        # ----------------------------------
        # Result
        # ----------------------------------

        return MultiPeriodAttributionResult(

            metadata=
            self.metadata,

            attribution_type=
            AttributionType.MULTI_PERIOD,

            attribution_name=
            "Multi Period Attribution",

            linking_method=
            linking_method,

            period_contributions=
            contribution_matrix,

            linked_contributions=
            linked_contributions,

            cumulative_portfolio_return=
            cumulative_portfolio,

            cumulative_benchmark_return=
            cumulative_benchmark,

            cumulative_active_return=
            cumulative_active,

            total_portfolio_return=
            cumulative_portfolio,

            total_benchmark_return=
            cumulative_benchmark,

            active_return=
            cumulative_active,

            contribution_series=
            linked_contributions,

            contribution_table=
            contribution_matrix,

            diagnostics={

                "periods":
                len(
                    period_results
                ),

                "linking_method":
                linking_method.value,

            },
        )


# ============================================================
# MULTI-PERIOD SUMMARY
# ============================================================

@dataclass(slots=True)
class MultiPeriodAttributionSummary:

    cumulative_portfolio_return: float

    cumulative_benchmark_return: float

    cumulative_active_return: float

    num_periods: int

    linking_method: str


# ------------------------------------------------------------

def summarize_multi_period_attribution(
    result: MultiPeriodAttributionResult,
) -> MultiPeriodAttributionSummary:

    return MultiPeriodAttributionSummary(

        cumulative_portfolio_return=
        result.cumulative_portfolio_return,

        cumulative_benchmark_return=
        result.cumulative_benchmark_return,

        cumulative_active_return=
        result.cumulative_active_return,

        num_periods=
        len(
            result.period_contributions
        )
        if result.period_contributions
        is not None
        else 0,

        linking_method=
        result.linking_method.value,
    )


# ============================================================
# CONVENIENCE API
# ============================================================

def multi_period_attribution(
    *,
    metadata: AttributionMetadata,
    period_results: list[
        AttributionResult
    ],
    linking_method: LinkingMethod = (
        LinkingMethod.GEOMETRIC
    ),
    config: AttributionConfig | None = None,
) -> MultiPeriodAttributionResult:

    analyzer = (
        MultiPeriodAttributionAnalyzer(
            metadata=metadata,
            config=config,
        )
    )

    return analyzer.analyze(

        period_results=
        period_results,

        linking_method=
        linking_method,
    )

# ============================================================
# PART 11
# INSTITUTIONAL MASTER ATTRIBUTION REPORT
# ============================================================

# ============================================================
# MASTER REPORT
# ============================================================

@dataclass(slots=True)
class InstitutionalAttributionReport:
    """
    Institutional attribution report.

    Single report object aggregating all
    attribution engines.
    """

    report_date: datetime

    portfolio_name: str

    benchmark_name: str

    metadata: AttributionMetadata

    # --------------------------------------
    # Attribution Modules
    # --------------------------------------

    return_attribution: ReturnAttributionResult | None = None

    brinson_attribution: BrinsonAttributionResult | None = None

    factor_attribution: FactorAttributionResult | None = None

    risk_attribution: RiskAttributionResult | None = None

    trade_attribution: TradeAttributionResult | None = None

    rebalance_attribution: RebalanceAttributionResult | None = None

    execution_attribution: ExecutionAttributionResult | None = None

    multi_period_attribution: MultiPeriodAttributionResult | None = None

    # --------------------------------------
    # Summary
    # --------------------------------------

    summary_metrics: dict[str, Any] = field(
        default_factory=dict
    )

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# MASTER REPORT BUILDER
# ============================================================

class InstitutionalAttributionReportBuilder:
    """
    Institutional report builder.
    """

    # --------------------------------------------------------

    def __init__(
        self,
        portfolio_name: str,
        benchmark_name: str,
        metadata: AttributionMetadata,
    ) -> None:

        self.portfolio_name = (
            portfolio_name
        )

        self.benchmark_name = (
            benchmark_name
        )

        self.metadata = metadata

    # --------------------------------------------------------
    # Summary Extraction
    # --------------------------------------------------------

    @staticmethod
    def build_summary(
        report: InstitutionalAttributionReport,
    ) -> dict[str, Any]:

        summary = {}

        # ----------------------------------
        # Return Attribution
        # ----------------------------------

        if (
            report.return_attribution
            is not None
        ):

            summary.update({

                "portfolio_return":

                report
                .return_attribution
                .total_portfolio_return,

                "benchmark_return":

                report
                .return_attribution
                .total_benchmark_return,

                "active_return":

                report
                .return_attribution
                .active_return,

            })

        # ----------------------------------
        # Brinson
        # ----------------------------------

        if (
            report.brinson_attribution
            is not None
        ):

            summary.update({

                "allocation_effect":

                report
                .brinson_attribution
                .allocation_effect,

                "selection_effect":

                report
                .brinson_attribution
                .selection_effect,

                "interaction_effect":

                report
                .brinson_attribution
                .interaction_effect,

            })

        # ----------------------------------
        # Factor
        # ----------------------------------

        if (
            report.factor_attribution
            is not None
        ):

            summary.update({

                "factor_explained_return":

                report
                .factor_attribution
                .explained_return,

                "factor_residual":

                report
                .factor_attribution
                .residual_return,

            })

        # ----------------------------------
        # Risk
        # ----------------------------------

        if (
            report.risk_attribution
            is not None
        ):

            summary.update({

                "portfolio_risk":

                report
                .risk_attribution
                .total_risk,

            })

        # ----------------------------------
        # Trade
        # ----------------------------------

        if (
            report.trade_attribution
            is not None
        ):

            summary.update({

                "trade_pnl":

                report
                .trade_attribution
                .total_trade_pnl,

                "hit_rate":

                report
                .trade_attribution
                .hit_rate,

            })

        # ----------------------------------
        # Rebalance
        # ----------------------------------

        if (
            report.rebalance_attribution
            is not None
        ):

            summary.update({

                "rebalance_alpha":

                report
                .rebalance_attribution
                .rebalance_alpha,

                "rebalance_cost":

                report
                .rebalance_attribution
                .rebalance_cost,

            })

        # ----------------------------------
        # Execution
        # ----------------------------------

        if (
            report.execution_attribution
            is not None
        ):

            summary.update({

                "execution_cost":

                report
                .execution_attribution
                .total_execution_cost,

            })

        return summary

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    @staticmethod
    def build_diagnostics(
        report: InstitutionalAttributionReport,
    ) -> dict[str, Any]:

        diagnostics = {}

        for attr_name in [

            "return_attribution",
            "brinson_attribution",
            "factor_attribution",
            "risk_attribution",
            "trade_attribution",
            "rebalance_attribution",
            "execution_attribution",
            "multi_period_attribution",

        ]:

            obj = getattr(
                report,
                attr_name,
                None,
            )

            if (
                obj is not None
                and
                hasattr(
                    obj,
                    "diagnostics",
                )
            ):

                diagnostics[
                    attr_name
                ] = obj.diagnostics

        return diagnostics

    # --------------------------------------------------------
    # Build Report
    # --------------------------------------------------------

    def build(
        self,
        *,
        return_attribution:
        ReturnAttributionResult | None = None,

        brinson_attribution:
        BrinsonAttributionResult | None = None,

        factor_attribution:
        FactorAttributionResult | None = None,

        risk_attribution:
        RiskAttributionResult | None = None,

        trade_attribution:
        TradeAttributionResult | None = None,

        rebalance_attribution:
        RebalanceAttributionResult | None = None,

        execution_attribution:
        ExecutionAttributionResult | None = None,

        multi_period_attribution:
        MultiPeriodAttributionResult | None = None,
    ) -> InstitutionalAttributionReport:

        report = InstitutionalAttributionReport(

            report_date=
            datetime.utcnow(),

            portfolio_name=
            self.portfolio_name,

            benchmark_name=
            self.benchmark_name,

            metadata=
            self.metadata,

            return_attribution=
            return_attribution,

            brinson_attribution=
            brinson_attribution,

            factor_attribution=
            factor_attribution,

            risk_attribution=
            risk_attribution,

            trade_attribution=
            trade_attribution,

            rebalance_attribution=
            rebalance_attribution,

            execution_attribution=
            execution_attribution,

            multi_period_attribution=
            multi_period_attribution,
        )

        report.summary_metrics = (
            self.build_summary(
                report
            )
        )

        report.diagnostics = (
            self.build_diagnostics(
                report
            )
        )

        return report


# ============================================================
# EXPORTERS
# ============================================================

class AttributionReportExporter:
    """
    Institutional report exporter.
    """

    # --------------------------------------------------------

    @staticmethod
    def to_dict(
        report: InstitutionalAttributionReport,
    ) -> dict[str, Any]:

        return asdict(
            report
        )

    # --------------------------------------------------------

    @staticmethod
    def summary_dataframe(
        report: InstitutionalAttributionReport,
    ) -> pd.DataFrame:

        return pd.DataFrame(
            [
                report.summary_metrics
            ]
        )

    # --------------------------------------------------------

    @staticmethod
    def to_json(
        report: InstitutionalAttributionReport,
    ) -> str:

        return json.dumps(

            AttributionReportExporter
            .to_dict(report),

            default=str,

            indent=2,
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def build_institutional_attribution_report(
    *,
    portfolio_name: str,
    benchmark_name: str,
    metadata: AttributionMetadata,

    return_attribution:
    ReturnAttributionResult | None = None,

    brinson_attribution:
    BrinsonAttributionResult | None = None,

    factor_attribution:
    FactorAttributionResult | None = None,

    risk_attribution:
    RiskAttributionResult | None = None,

    trade_attribution:
    TradeAttributionResult | None = None,

    rebalance_attribution:
    RebalanceAttributionResult | None = None,

    execution_attribution:
    ExecutionAttributionResult | None = None,

    multi_period_attribution:
    MultiPeriodAttributionResult | None = None,

) -> InstitutionalAttributionReport:

    builder = (
        InstitutionalAttributionReportBuilder(

            portfolio_name=
            portfolio_name,

            benchmark_name=
            benchmark_name,

            metadata=
            metadata,

        )
    )

    return builder.build(

        return_attribution=
        return_attribution,

        brinson_attribution=
        brinson_attribution,

        factor_attribution=
        factor_attribution,

        risk_attribution=
        risk_attribution,

        trade_attribution=
        trade_attribution,

        rebalance_attribution=
        rebalance_attribution,

        execution_attribution=
        execution_attribution,

        multi_period_attribution=
        multi_period_attribution,
    )

# ============================================================
# PART 12
# ATTRIBUTION ENGINE
# ============================================================

# ============================================================
# ENGINE CONFIG
# ============================================================

@dataclass(slots=True)
class AttributionEngineConfig:
    """
    Controls which attribution modules run.
    """

    enable_return_attribution: bool = True

    enable_brinson_attribution: bool = True

    enable_factor_attribution: bool = True

    enable_risk_attribution: bool = True

    enable_trade_attribution: bool = True

    enable_rebalance_attribution: bool = True

    enable_execution_attribution: bool = True

    enable_multi_period_attribution: bool = True


# ============================================================
# ATTRIBUTION ENGINE
# ============================================================

class AttributionEngine:
    """
    Institutional Attribution Engine.

    Central orchestration layer.

    Runs all enabled attribution modules.

    Produces a master report.
    """

    # --------------------------------------------------------

    def __init__(
        self,
        metadata: AttributionMetadata,
        portfolio_name: str,
        benchmark_name: str,
        config: AttributionEngineConfig | None = None,
    ) -> None:

        self.metadata = metadata

        self.portfolio_name = portfolio_name

        self.benchmark_name = benchmark_name

        self.config = (
            config
            if config is not None
            else AttributionEngineConfig()
        )

    # ========================================================
    # RETURN ATTRIBUTION
    # ========================================================

    def run_return_attribution(
        self,
        *,
        weights: pd.Series,
        returns: pd.Series,
        sectors: pd.Series | None = None,
        benchmark_return: float = 0.0,
        cash_weight: float = 0.0,
        cash_return: float = 0.0,
    ) -> ReturnAttributionResult:

        return return_attribution(

            metadata=self.metadata,

            weights=weights,

            returns=returns,

            sectors=sectors,

            benchmark_return=benchmark_return,

            cash_weight=cash_weight,

            cash_return=cash_return,
        )

    # ========================================================
    # BRINSON
    # ========================================================

    def run_brinson_attribution(
        self,
        *,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> BrinsonAttributionResult:

        return brinson_attribution(

            metadata=self.metadata,

            portfolio_weights=portfolio_weights,

            benchmark_weights=benchmark_weights,

            portfolio_returns=portfolio_returns,

            benchmark_returns=benchmark_returns,
        )

    # ========================================================
    # FACTOR
    # ========================================================

    def run_factor_attribution(
        self,
        *,
        portfolio_return: float,
        factor_exposures: pd.Series,
        factor_returns: pd.Series,
        factor_covariance: pd.DataFrame | None = None,
    ) -> FactorAttributionResult:

        return factor_attribution(

            metadata=self.metadata,

            portfolio_return=portfolio_return,

            factor_exposures=factor_exposures,

            factor_returns=factor_returns,

            factor_covariance=factor_covariance,
        )

    # ========================================================
    # RISK
    # ========================================================

    def run_risk_attribution(
        self,
        *,
        weights: pd.Series,
        covariance_matrix: pd.DataFrame,
        factor_exposures: pd.Series | None = None,
        factor_covariance: pd.DataFrame | None = None,
    ) -> RiskAttributionResult:

        return risk_attribution(

            metadata=self.metadata,

            weights=weights,

            covariance_matrix=covariance_matrix,

            factor_exposures=factor_exposures,

            factor_covariance=factor_covariance,
        )

    # ========================================================
    # TRADE
    # ========================================================

    def run_trade_attribution(
        self,
        *,
        trades: pd.DataFrame,
    ) -> TradeAttributionResult:

        return trade_attribution(

            metadata=self.metadata,

            trades=trades,
        )

    # ========================================================
    # REBALANCE
    # ========================================================

    def run_rebalance_attribution(
        self,
        *,
        rebalance_df: pd.DataFrame,
        expected_returns: pd.Series,
        target_weights: pd.Series | None = None,
        transaction_cost_bps: float = 5.0,
    ) -> RebalanceAttributionResult:

        return rebalance_attribution(

            metadata=self.metadata,

            rebalance_df=rebalance_df,

            expected_returns=expected_returns,

            target_weights=target_weights,

            transaction_cost_bps=transaction_cost_bps,
        )

    # ========================================================
    # EXECUTION
    # ========================================================

    def run_execution_attribution(
        self,
        *,
        execution_df: pd.DataFrame,
    ) -> ExecutionAttributionResult:

        return execution_attribution(

            metadata=self.metadata,

            execution_df=execution_df,
        )

    # ========================================================
    # MULTI PERIOD
    # ========================================================

    def run_multi_period_attribution(
        self,
        *,
        period_results: list[AttributionResult],
        linking_method: LinkingMethod = (
            LinkingMethod.GEOMETRIC
        ),
    ) -> MultiPeriodAttributionResult:

        return multi_period_attribution(

            metadata=self.metadata,

            period_results=period_results,

            linking_method=linking_method,
        )

    # ========================================================
    # MASTER REPORT
    # ========================================================

    def build_report(
        self,
        *,
        return_attribution_result:
        ReturnAttributionResult | None = None,

        brinson_attribution_result:
        BrinsonAttributionResult | None = None,

        factor_attribution_result:
        FactorAttributionResult | None = None,

        risk_attribution_result:
        RiskAttributionResult | None = None,

        trade_attribution_result:
        TradeAttributionResult | None = None,

        rebalance_attribution_result:
        RebalanceAttributionResult | None = None,

        execution_attribution_result:
        ExecutionAttributionResult | None = None,

        multi_period_attribution_result:
        MultiPeriodAttributionResult | None = None,
    ) -> InstitutionalAttributionReport:

        return build_institutional_attribution_report(

            portfolio_name=
            self.portfolio_name,

            benchmark_name=
            self.benchmark_name,

            metadata=
            self.metadata,

            return_attribution=
            return_attribution_result,

            brinson_attribution=
            brinson_attribution_result,

            factor_attribution=
            factor_attribution_result,

            risk_attribution=
            risk_attribution_result,

            trade_attribution=
            trade_attribution_result,

            rebalance_attribution=
            rebalance_attribution_result,

            execution_attribution=
            execution_attribution_result,

            multi_period_attribution=
            multi_period_attribution_result,
        )

    # ========================================================
    # FULL ORCHESTRATION
    # ========================================================

    def run(
        self,
        *,
        return_attribution_result:
        ReturnAttributionResult | None = None,

        brinson_attribution_result:
        BrinsonAttributionResult | None = None,

        factor_attribution_result:
        FactorAttributionResult | None = None,

        risk_attribution_result:
        RiskAttributionResult | None = None,

        trade_attribution_result:
        TradeAttributionResult | None = None,

        rebalance_attribution_result:
        RebalanceAttributionResult | None = None,

        execution_attribution_result:
        ExecutionAttributionResult | None = None,

        multi_period_attribution_result:
        MultiPeriodAttributionResult | None = None,
    ) -> InstitutionalAttributionReport:

        return self.build_report(

            return_attribution_result=
            return_attribution_result,

            brinson_attribution_result=
            brinson_attribution_result,

            factor_attribution_result=
            factor_attribution_result,

            risk_attribution_result=
            risk_attribution_result,

            trade_attribution_result=
            trade_attribution_result,

            rebalance_attribution_result=
            rebalance_attribution_result,

            execution_attribution_result=
            execution_attribution_result,

            multi_period_attribution_result=
            multi_period_attribution_result,
        )
    
# ============================================================
# PART 13
# FACTORY & CONVENIENCE APIS
# ============================================================

# ============================================================
# DEFAULT CONFIG FACTORIES
# ============================================================

def default_attribution_config() -> AttributionConfig:
    """
    Default attribution configuration.
    """

    return AttributionConfig()


def default_engine_config() -> AttributionEngineConfig:
    """
    Default engine configuration.
    """

    return AttributionEngineConfig()


# ============================================================
# ENGINE FACTORY
# ============================================================

def create_attribution_engine(
    *,
    metadata: AttributionMetadata,
    portfolio_name: str,
    benchmark_name: str,
    config: AttributionEngineConfig | None = None,
) -> AttributionEngine:
    """
    Create institutional attribution engine.
    """

    return AttributionEngine(
        metadata=metadata,
        portfolio_name=portfolio_name,
        benchmark_name=benchmark_name,
        config=config,
    )


# ============================================================
# REPORT FACTORY
# ============================================================

def build_attribution_report(
    *,
    portfolio_name: str,
    benchmark_name: str,
    metadata: AttributionMetadata,

    return_attribution:
    ReturnAttributionResult | None = None,

    brinson_attribution:
    BrinsonAttributionResult | None = None,

    factor_attribution:
    FactorAttributionResult | None = None,

    risk_attribution:
    RiskAttributionResult | None = None,

    trade_attribution:
    TradeAttributionResult | None = None,

    rebalance_attribution:
    RebalanceAttributionResult | None = None,

    execution_attribution:
    ExecutionAttributionResult | None = None,

    multi_period_attribution:
    MultiPeriodAttributionResult | None = None,
) -> InstitutionalAttributionReport:
    """
    Convenience wrapper around the
    institutional report builder.
    """

    return build_institutional_attribution_report(

        portfolio_name=
        portfolio_name,

        benchmark_name=
        benchmark_name,

        metadata=
        metadata,

        return_attribution=
        return_attribution,

        brinson_attribution=
        brinson_attribution,

        factor_attribution=
        factor_attribution,

        risk_attribution=
        risk_attribution,

        trade_attribution=
        trade_attribution,

        rebalance_attribution=
        rebalance_attribution,

        execution_attribution=
        execution_attribution,

        multi_period_attribution=
        multi_period_attribution,
    )


# ============================================================
# FULL ATTRIBUTION RUNNER
# ============================================================

def run_full_attribution(
    *,
    metadata: AttributionMetadata,
    portfolio_name: str,
    benchmark_name: str,

    return_attribution:
    ReturnAttributionResult | None = None,

    brinson_attribution:
    BrinsonAttributionResult | None = None,

    factor_attribution:
    FactorAttributionResult | None = None,

    risk_attribution:
    RiskAttributionResult | None = None,

    trade_attribution:
    TradeAttributionResult | None = None,

    rebalance_attribution:
    RebalanceAttributionResult | None = None,

    execution_attribution:
    ExecutionAttributionResult | None = None,

    multi_period_attribution:
    MultiPeriodAttributionResult | None = None,

    engine_config:
    AttributionEngineConfig | None = None,
) -> InstitutionalAttributionReport:
    """
    One-line institutional attribution API.
    """

    engine = create_attribution_engine(

        metadata=
        metadata,

        portfolio_name=
        portfolio_name,

        benchmark_name=
        benchmark_name,

        config=
        engine_config,
    )

    return engine.run(

        return_attribution_result=
        return_attribution,

        brinson_attribution_result=
        brinson_attribution,

        factor_attribution_result=
        factor_attribution,

        risk_attribution_result=
        risk_attribution,

        trade_attribution_result=
        trade_attribution,

        rebalance_attribution_result=
        rebalance_attribution,

        execution_attribution_result=
        execution_attribution,

        multi_period_attribution_result=
        multi_period_attribution,
    )


# ============================================================
# EXPORT HELPERS
# ============================================================

def attribution_report_to_dict(
    report: InstitutionalAttributionReport,
) -> dict[str, Any]:

    return AttributionReportExporter.to_dict(
        report
    )


def attribution_report_to_json(
    report: InstitutionalAttributionReport,
) -> str:

    return AttributionReportExporter.to_json(
        report
    )


def attribution_report_to_dataframe(
    report: InstitutionalAttributionReport,
) -> pd.DataFrame:

    return (
        AttributionReportExporter
        .summary_dataframe(
            report
        )
    )


# ============================================================
# SUMMARY HELPERS
# ============================================================

def attribution_summary(
    report: InstitutionalAttributionReport,
) -> dict[str, Any]:
    """
    Lightweight summary.
    """

    return dict(
        report.summary_metrics
    )


# ============================================================
# REGISTRY
# ============================================================

ATTRIBUTION_ANALYZER_REGISTRY = {

    AttributionType.RETURN:
    ReturnAttributionAnalyzer,

    AttributionType.BRINSON:
    BrinsonAttributionAnalyzer,

    AttributionType.FACTOR:
    FactorAttributionAnalyzer,

    AttributionType.RISK:
    RiskAttributionAnalyzer,

    AttributionType.TRADE:
    TradeAttributionAnalyzer,

    AttributionType.REBALANCE:
    RebalanceAttributionAnalyzer,

    AttributionType.EXECUTION:
    ExecutionAttributionAnalyzer,

    AttributionType.MULTI_PERIOD:
    MultiPeriodAttributionAnalyzer,
}


# ============================================================
# ANALYZER FACTORY
# ============================================================

def create_analyzer(
    attribution_type: AttributionType,
    *,
    metadata: AttributionMetadata,
    config: AttributionConfig | None = None,
) -> BaseAttribution:
    """
    Generic analyzer factory.
    """

    cls = (
        ATTRIBUTION_ANALYZER_REGISTRY[
            attribution_type
        ]
    )

    return cls(
        metadata=metadata,
        config=config,
    )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [

    # engine
    "AttributionEngine",
    "AttributionEngineConfig",

    # report
    "InstitutionalAttributionReport",
    "InstitutionalAttributionReportBuilder",

    # factories
    "create_attribution_engine",
    "build_attribution_report",
    "run_full_attribution",

    # exports
    "attribution_report_to_dict",
    "attribution_report_to_json",
    "attribution_report_to_dataframe",

    # helpers
    "attribution_summary",
    "create_analyzer",

]