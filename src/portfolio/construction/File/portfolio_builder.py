# ============================================================
# PORTFOLIO BUILDER
# PART 1 — FRAMEWORK & CORE OBJECTS
# ============================================================

from __future__ import annotations

# ============================================================
# STANDARD LIBRARIES
# ============================================================

from abc import ABC
from abc import abstractmethod

from dataclasses import dataclass
from dataclasses import field

from datetime import datetime
from datetime import timezone

from enum import Enum
from enum import auto   

from typing import Any, TYPE_CHECKING
from typing import Protocol

import uuid

from src.portfolio.diagnostics import (
    PortfolioDiagnosticsReport,
)

if TYPE_CHECKING:
    from .pipeline import InstitutionalDiagnosticsPackage
# ============================================================
# THIRD PARTY
# ============================================================

import numpy as np
import pandas as pd

# ============================================================
# GLOBAL CONSTANTS
# ============================================================

TRADING_DAYS = 252

# ============================================================
# ENUMS
# ============================================================


class PortfolioBuildStage(Enum):
    """
    Pipeline stage.
    """

    FORECAST = auto()

    RISK = auto()

    CONSTRAINTS = auto()

    OPTIMIZATION = auto()

    PORTFOLIO_ASSEMBLY = auto()

    REBALANCE = auto()

    VALIDATION = auto()

    REPORTING = auto()


class PortfolioBuildStatus(Enum):
    """
    Build status.
    """

    NOT_STARTED = auto()

    RUNNING = auto()

    COMPLETED = auto()

    FAILED = auto()


class PortfolioType(Enum):
    """
    Portfolio classification.
    """

    LONG_ONLY = auto()

    LONG_SHORT = auto()

    MARKET_NEUTRAL = auto()

    FACTOR = auto()

    RISK_PARITY = auto()

    MULTI_FACTOR = auto()

    CUSTOM = auto()


class RebalanceType(Enum):
    """
    Rebalance mode.
    """

    CALENDAR = auto()

    THRESHOLD = auto()

    VOLATILITY = auto()

    DRIFT = auto()

    CUSTOM = auto()


class ValidationStatus(Enum):
    """
    Portfolio validation.
    """

    PASSED = auto()

    WARNING = auto()

    FAILED = auto()


# ============================================================
# METADATA
# ============================================================


@dataclass(slots=True)
class PortfolioBuilderMetadata:
    """
    Builder metadata.
    """

    portfolio_id: str

    portfolio_name: str

    portfolio_type: PortfolioType

    created_by: str

    created_at: datetime = field(
        default_factory=lambda:
        datetime.now(timezone.utc)
    )

    run_id: str = field(
        default_factory=lambda:
        str(uuid.uuid4())
    )


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(slots=True)
class PortfolioBuilderConfig:
    """
    Master portfolio build config.
    """

    max_position_weight: float = 0.10

    min_position_weight: float = 0.0

    max_turnover: float = 1.00

    cash_buffer: float = 0.01

    target_volatility: float = 0.15

    long_only: bool = True

    allow_shorting: bool = False

    enforce_constraints: bool = True

    generate_rebalance_orders: bool = True

    validate_portfolio: bool = True


# ============================================================
# INPUT OBJECTS
# ============================================================


@dataclass(slots=True)
class PortfolioBuildInput:
    """
    Master builder input.
    """

    expected_returns: pd.Series

    covariance_matrix: pd.DataFrame

    current_weights: pd.Series | None = None

    benchmark_weights: pd.Series | None = None

    factor_exposures: pd.DataFrame | None = None

    security_metadata: pd.DataFrame | None = None

    prices: pd.Series | None = None

    available_cash: float = 0.0

    timestamp: datetime = field(
        default_factory=lambda:
        datetime.now(timezone.utc)
    )


# ============================================================
# PORTFOLIO OBJECTS
# ============================================================


@dataclass(slots=True)
class PortfolioPosition:
    """
    Single portfolio position.
    """

    asset: str

    weight: float

    quantity: float | None = None

    price: float | None = None

    market_value: float | None = None


@dataclass(slots=True)
class PortfolioWeights:
    """
    Target weights.
    """

    weights: pd.Series

    cash_weight: float = 0.0

    gross_exposure: float = 0.0

    net_exposure: float = 0.0


@dataclass(slots=True)
class TargetPortfolio:
    """
    Fully constructed portfolio.
    """

    weights: PortfolioWeights

    positions: list[PortfolioPosition]

    expected_return: float

    expected_volatility: float

    expected_sharpe: float


# ============================================================
# REBALANCE OBJECTS
# ============================================================


@dataclass(slots=True)
class RebalanceOrder:
    """
    Generated rebalance order.
    """

    asset: str

    current_weight: float

    target_weight: float

    trade_weight: float

    estimated_notional: float = 0.0


@dataclass(slots=True)
class RebalancePlan:
    """
    Rebalance package.
    """

    rebalance_type: RebalanceType

    orders: list[RebalanceOrder]

    turnover: float


# ============================================================
# VALIDATION
# ============================================================


@dataclass(slots=True)
class ValidationCheck:
    """
    Single validation result.
    """

    name: str

    status: ValidationStatus

    message: str


@dataclass(slots=True)
class PortfolioValidationReport:
    """
    Portfolio validation output.
    """

    status: ValidationStatus

    checks: list[ValidationCheck]


# ============================================================
# DIAGNOSTICS
# ============================================================


@dataclass(slots=True)
class PortfolioDiagnostics:
    """
    Builder diagnostics.
    """

    build_time_seconds: float

    optimization_success: bool

    constraint_violations: int

    warnings: list[str] = field(
        default_factory=list
    )


# ============================================================
# REPORTING OBJECTS
# ============================================================


@dataclass(slots=True)
class InstitutionalPortfolioReport:
    """
    Final portfolio builder report.
    """

    metadata: PortfolioBuilderMetadata

    portfolio: TargetPortfolio

    rebalance_plan: RebalancePlan | None

    validation_report: PortfolioValidationReport | None

    diagnostics: PortfolioDiagnostics | None

    summary_metrics: dict[str, Any]


# ============================================================
# ABSTRACT INTERFACES
# ============================================================


class ForecastProvider(ABC):
    """
    Forecast interface.
    """

    @abstractmethod
    def get_expected_returns(
        self,
        inputs:
        PortfolioBuildInput,
    ) -> pd.Series:
        pass


class RiskModelProvider(ABC):
    """
    Risk model interface.
    """

    @abstractmethod
    def get_covariance_matrix(
        self,
        inputs:
        PortfolioBuildInput,
    ) -> pd.DataFrame:
        pass


class ConstraintProvider(ABC):
    """
    Constraint interface.
    """

    @abstractmethod
    def apply_constraints(
        self,
        weights:
        pd.Series,
    ) -> pd.Series:
        pass


class OptimizerProvider(ABC):
    """
    Optimizer interface.
    """

    @abstractmethod
    def optimize(
        self,
        expected_returns:
        pd.Series,
        covariance_matrix:
        pd.DataFrame,
    ) -> pd.Series:
        pass


class RebalanceProvider(ABC):
    """
    Rebalance interface.
    """

    @abstractmethod
    def generate_rebalance(
        self,
        current_weights:
        pd.Series,
        target_weights:
        pd.Series,
    ) -> RebalancePlan:
        pass


# ============================================================
# HELPER UTILITIES
# ============================================================


class PortfolioBuilderUtils:
    """
    Shared helper functions.
    """

    @staticmethod
    def normalize_weights(
        weights: pd.Series,
    ) -> pd.Series:

        total = float(
            weights.sum()
        )

        if total == 0:
            return weights

        return weights / total

    # --------------------------------------------------------

    @staticmethod
    def portfolio_turnover(
        current_weights:
        pd.Series,
        target_weights:
        pd.Series,
    ) -> float:

        aligned = pd.concat(
            [
                current_weights,
                target_weights,
            ],
            axis=1,
        ).fillna(0.0)

        return float(
            np.abs(
                aligned.iloc[:, 0]
                -
                aligned.iloc[:, 1]
            ).sum()
            / 2.0
        )
    

# ============================================================
# PART 2 — PORTFOLIO INPUTS
# ============================================================

# ============================================================
# UNIVERSE INPUTS
# ============================================================


@dataclass(slots=True)
class SecurityUniverse:
    """
    Tradable universe definition.
    """

    assets: list[str]

    asset_metadata: pd.DataFrame | None = None

    benchmark_membership: pd.Series | None = None

    sector_map: pd.Series | None = None

    country_map: pd.Series | None = None

    currency_map: pd.Series | None = None


# ============================================================
# SIGNAL INPUTS
# ============================================================


@dataclass(slots=True)
class SignalInput:
    """
    Raw alpha signals.
    """

    signals: pd.Series

    signal_name: str = "alpha_signal"

    signal_timestamp: datetime = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        )
    )


@dataclass(slots=True)
class MultiSignalInput:
    """
    Multiple alpha signals.
    """

    signals: pd.DataFrame

    signal_weights: pd.Series | None = None


# ============================================================
# FORECAST INPUTS
# ============================================================


@dataclass(slots=True)
class ExpectedReturnInput:
    """
    Expected return estimates.
    """

    expected_returns: pd.Series

    confidence_scores: pd.Series | None = None

    forecast_horizon_days: int = 21


# ============================================================
# RISK INPUTS
# ============================================================


@dataclass(slots=True)
class CovarianceInput:
    """
    Covariance estimate.
    """

    covariance_matrix: pd.DataFrame

    annualized: bool = True


@dataclass(slots=True)
class VolatilityInput:
    """
    Security volatilities.
    """

    volatility: pd.Series

    annualized: bool = True


# ============================================================
# FACTOR INPUTS
# ============================================================


@dataclass(slots=True)
class FactorExposureInput:
    """
    Security factor exposures.
    """

    factor_exposures: pd.DataFrame

    factor_names: list[str] | None = None


@dataclass(slots=True)
class FactorReturnInput:
    """
    Factor return forecasts.
    """

    factor_returns: pd.Series


# ============================================================
# BENCHMARK INPUTS
# ============================================================


@dataclass(slots=True)
class BenchmarkInput:
    """
    Benchmark information.
    """

    benchmark_name: str

    benchmark_weights: pd.Series

    benchmark_returns: pd.Series | None = None


# ============================================================
# LIQUIDITY INPUTS
# ============================================================


@dataclass(slots=True)
class LiquidityInput:
    """
    Liquidity metrics.
    """

    average_daily_volume: pd.Series

    average_daily_dollar_volume: pd.Series | None = None

    bid_ask_spread: pd.Series | None = None

    participation_limit: float = 0.10


# ============================================================
# CONSTRAINT INPUTS
# ============================================================


@dataclass(slots=True)
class PositionLimitInput:
    """
    Position constraints.
    """

    max_weight: float

    min_weight: float = 0.0


@dataclass(slots=True)
class SectorConstraintInput:
    """
    Sector constraints.
    """

    sector_min: pd.Series | None = None

    sector_max: pd.Series | None = None


@dataclass(slots=True)
class CountryConstraintInput:
    """
    Country constraints.
    """

    country_min: pd.Series | None = None

    country_max: pd.Series | None = None


@dataclass(slots=True)
class TurnoverConstraintInput:
    """
    Turnover limits.
    """

    max_turnover: float = 1.0


# ============================================================
# CURRENT PORTFOLIO INPUTS
# ============================================================


@dataclass(slots=True)
class CurrentPortfolioInput:
    """
    Existing portfolio state.
    """

    current_weights: pd.Series

    current_positions: pd.Series | None = None

    cash_weight: float = 0.0


# ============================================================
# PORTFOLIO BUILD PACKAGE
# ============================================================


@dataclass(slots=True)
class PortfolioConstructionInput:
    """
    Institutional portfolio build package.

    This becomes the primary object
    passed through the builder engine.
    """

    universe: SecurityUniverse

    forecast: ExpectedReturnInput

    covariance: CovarianceInput

    benchmark: BenchmarkInput | None = None

    factor_exposures: FactorExposureInput | None = None

    factor_returns: FactorReturnInput | None = None

    liquidity: LiquidityInput | None = None

    current_portfolio: CurrentPortfolioInput | None = None

    position_limits: PositionLimitInput | None = None

    sector_constraints: SectorConstraintInput | None = None

    country_constraints: CountryConstraintInput | None = None

    turnover_constraints: TurnoverConstraintInput | None = None

    timestamp: datetime = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        )
    )


# ============================================================
# INPUT VALIDATOR
# ============================================================


class PortfolioInputValidator:
    """
    Institutional validation layer.
    """

    # --------------------------------------------------------

    @staticmethod
    def validate_expected_returns(
        expected_returns:
        pd.Series,
    ) -> None:

        if expected_returns.empty:

            raise ValueError(
                "expected_returns empty"
            )

    # --------------------------------------------------------

    @staticmethod
    def validate_covariance(
        covariance_matrix:
        pd.DataFrame,
    ) -> None:

        if covariance_matrix.empty:

            raise ValueError(
                "covariance_matrix empty"
            )

        if (
            covariance_matrix.shape[0]
            !=
            covariance_matrix.shape[1]
        ):

            raise ValueError(
                "covariance matrix must be square"
            )

    # --------------------------------------------------------

    @classmethod
    def validate(
        cls,
        inputs:
        PortfolioConstructionInput,
    ) -> None:

        cls.validate_expected_returns(
            inputs.forecast
            .expected_returns
        )

        cls.validate_covariance(
            inputs.covariance
            .covariance_matrix
        )


# ============================================================
# PART 3 — FORECAST INTEGRATION LAYER
# ============================================================

# ============================================================
# FORECAST RESULTS
# ============================================================


@dataclass(slots=True)
class ForecastResult:
    """
    Standardized forecast output.

    Everything downstream
    consumes this object.
    """

    expected_returns: pd.Series

    confidence_scores: pd.Series

    forecast_horizon_days: int

    signal_strength: pd.Series

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# FORECAST DIAGNOSTICS
# ============================================================


@dataclass(slots=True)
class ForecastDiagnostics:
    """
    Forecast quality metrics.
    """

    mean_forecast: float

    forecast_std: float

    max_forecast: float

    min_forecast: float

    positive_fraction: float

    negative_fraction: float


# ============================================================
# BASE FORECAST INTEGRATION
# ============================================================


class BaseForecastIntegration(
    ABC,
):
    """
    Base forecast interface.
    """

    def __init__(
        self,
        metadata:
        PortfolioBuilderMetadata,
        config:
        PortfolioBuilderConfig | None = None,
    ) -> None:

        self.metadata = metadata

        self.config = config

    # --------------------------------------------------------

    @abstractmethod
    def build_forecast(
        self,
        inputs:
        PortfolioConstructionInput,
    ) -> ForecastResult:
        pass


# ============================================================
# FORECAST NORMALIZER
# ============================================================


class ForecastNormalizer:
    """
    Institutional forecast scaling.
    """

    # --------------------------------------------------------

    @staticmethod
    def zscore(
        forecasts:
        pd.Series,
    ) -> pd.Series:

        std = float(
            forecasts.std()
        )

        if std <= 0:

            return (
                forecasts * 0.0
            )

        return (

            forecasts
            -
            forecasts.mean()

        ) / std

    # --------------------------------------------------------

    @staticmethod
    def rank_normalize(
        forecasts:
        pd.Series,
    ) -> pd.Series:

        ranks = (
            forecasts.rank(
                pct=True
            )
        )

        return (
            ranks - 0.5
        )

    # --------------------------------------------------------

    @staticmethod
    def clip(
        forecasts:
        pd.Series,
        lower:
        float = -3.0,
        upper:
        float = 3.0,
    ) -> pd.Series:

        return forecasts.clip(
            lower=lower,
            upper=upper,
        )


# ============================================================
# CONFIDENCE ENGINE
# ============================================================


class ForecastConfidenceEngine:
    """
    Converts forecasts into
    confidence scores.
    """

    # --------------------------------------------------------

    @staticmethod
    def from_signal_strength(
        forecasts:
        pd.Series,
    ) -> pd.Series:

        z = np.abs(

            ForecastNormalizer
            .zscore(
                forecasts
            )

        )

        confidence = (

            z
            /
            (
                z.max()
                + 1e-12
            )

        )

        return confidence.clip(
            0.0,
            1.0,
        )

    # --------------------------------------------------------

    @staticmethod
    def uniform(
        forecasts:
        pd.Series,
    ) -> pd.Series:

        return pd.Series(

            1.0,

            index=
            forecasts.index,
        )


# ============================================================
# FORECAST DIAGNOSTIC ENGINE
# ============================================================


class ForecastDiagnosticEngine:
    """
    Forecast analytics.
    """

    # --------------------------------------------------------

    @staticmethod
    def compute(
        forecasts:
        pd.Series,
    ) -> ForecastDiagnostics:

        return ForecastDiagnostics(

            mean_forecast=
            float(
                forecasts.mean()
            ),

            forecast_std=
            float(
                forecasts.std()
            ),

            max_forecast=
            float(
                forecasts.max()
            ),

            min_forecast=
            float(
                forecasts.min()
            ),

            positive_fraction=
            float(
                (
                    forecasts > 0
                ).mean()
            ),

            negative_fraction=
            float(
                (
                    forecasts < 0
                ).mean()
            ),
        )


# ============================================================
# STANDARD FORECAST INTEGRATION
# ============================================================


class ForecastIntegrationLayer(
    BaseForecastIntegration,
):
    """
    Institutional forecast adapter.

    Converts raw alpha forecasts
    into standardized portfolio
    construction forecasts.
    """

    # --------------------------------------------------------

    def build_forecast(
        self,
        inputs:
        PortfolioConstructionInput,
    ) -> ForecastResult:

        raw_forecasts = (

            inputs.forecast
            .expected_returns
            .copy()

        )

        normalized = (

            ForecastNormalizer
            .zscore(
                raw_forecasts
            )
        )

        normalized = (

            ForecastNormalizer
            .clip(
                normalized
            )
        )

        confidence = (

            ForecastConfidenceEngine
            .from_signal_strength(
                normalized
            )
        )

        diagnostics = (

            ForecastDiagnosticEngine
            .compute(
                normalized
            )
        )

        return ForecastResult(

            expected_returns=
            normalized,

            confidence_scores=
            confidence,

            forecast_horizon_days=
            inputs.forecast
            .forecast_horizon_days,

            signal_strength=
            normalized.abs(),

            diagnostics={
                "mean":
                diagnostics.mean_forecast,

                "std":
                diagnostics.forecast_std,

                "max":
                diagnostics.max_forecast,

                "min":
                diagnostics.min_forecast,

                "positive_fraction":
                diagnostics
                .positive_fraction,

                "negative_fraction":
                diagnostics
                .negative_fraction,
            },
        )


# ============================================================
# MULTI-SIGNAL FORECAST ENGINE
# ============================================================


class MultiSignalForecastIntegration(
    BaseForecastIntegration,
):
    """
    Combines multiple signals
    into one forecast.
    """

    # --------------------------------------------------------

    def combine_signals(
        self,
        signals:
        pd.DataFrame,
        weights:
        pd.Series | None,
    ) -> pd.Series:

        if weights is None:

            weights = pd.Series(

                1.0
                /
                signals.shape[1],

                index=
                signals.columns,
            )

        weights = (
            weights
            /
            weights.sum()
        )

        return signals.dot(
            weights
        )

    # --------------------------------------------------------

    def build_forecast(
        self,
        inputs:
        PortfolioConstructionInput,
    ) -> ForecastResult:

        raise NotImplementedError(
            "Requires MultiSignalInput"
        )


# ============================================================
# FORECAST FACTORY
# ============================================================


class ForecastIntegrationFactory:
    """
    Forecast builders.
    """

    # --------------------------------------------------------

    @staticmethod
    def standard(
        metadata:
        PortfolioBuilderMetadata,
        config:
        PortfolioBuilderConfig
        | None = None,
    ) -> ForecastIntegrationLayer:

        return (
            ForecastIntegrationLayer(

                metadata=
                metadata,

                config=
                config,
            )
        )

    # --------------------------------------------------------

    @staticmethod
    def multi_signal(
        metadata:
        PortfolioBuilderMetadata,
        config:
        PortfolioBuilderConfig
        | None = None,
    ) -> (
        MultiSignalForecastIntegration
    ):

        return (
            MultiSignalForecastIntegration(

                metadata=
                metadata,

                config=
                config,
            )
        )
    

# ============================================================
# PART 4 — RISK MODEL INTEGRATION
# ============================================================

# ============================================================
# RISK RESULTS
# ============================================================


@dataclass(slots=True)
class RiskModelResult:
    """
    Standardized risk output.

    Everything downstream
    consumes this object.
    """

    covariance_matrix: pd.DataFrame

    volatility: pd.Series

    correlation_matrix: pd.DataFrame

    risk_budget: pd.Series | None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# RISK DIAGNOSTICS
# ============================================================


@dataclass(slots=True)
class RiskDiagnostics:
    """
    Risk model diagnostics.
    """

    average_volatility: float

    max_volatility: float

    min_volatility: float

    average_correlation: float

    covariance_condition_number: float


# ============================================================
# BASE RISK INTEGRATION
# ============================================================


class BaseRiskModelIntegration(
    ABC,
):
    """
    Base risk interface.
    """

    def __init__(
        self,
        metadata:
        PortfolioBuilderMetadata,
        config:
        PortfolioBuilderConfig | None = None,
    ) -> None:

        self.metadata = metadata

        self.config = config

    # --------------------------------------------------------

    @abstractmethod
    def build_risk_model(
        self,
        inputs:
        PortfolioConstructionInput,
    ) -> RiskModelResult:
        pass


# ============================================================
# MATRIX UTILITIES
# ============================================================


class RiskMatrixUtils:
    """
    Matrix utilities.
    """

    # --------------------------------------------------------

    @staticmethod
    def covariance_to_volatility(
        covariance_matrix:
        pd.DataFrame,
    ) -> pd.Series:

        vols = np.sqrt(
            np.diag(
                covariance_matrix
            )
        )

        return pd.Series(

            vols,

            index=
            covariance_matrix.index,
        )

    # --------------------------------------------------------

    @staticmethod
    def covariance_to_correlation(
        covariance_matrix:
        pd.DataFrame,
    ) -> pd.DataFrame:

        vol = np.sqrt(
            np.diag(
                covariance_matrix
            )
        )

        vol = np.where(
            vol <= 0,
            1e-12,
            vol,
        )

        corr = (

            covariance_matrix.values

            /

            np.outer(
                vol,
                vol,
            )

        )

        corr = np.clip(
            corr,
            -1.0,
            1.0,
        )

        return pd.DataFrame(

            corr,

            index=
            covariance_matrix.index,

            columns=
            covariance_matrix.columns,
        )

    # --------------------------------------------------------

    @staticmethod
    def nearest_psd(
        matrix:
        pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Force PSD covariance.
        """

        eigvals, eigvecs = np.linalg.eigh(
            matrix.values
        )

        eigvals = np.maximum(
            eigvals,
            1e-8,
        )

        repaired = (

            eigvecs

            @

            np.diag(
                eigvals
            )

            @

            eigvecs.T

        )

        return pd.DataFrame(

            repaired,

            index=
            matrix.index,

            columns=
            matrix.columns,
        )


# ============================================================
# VOLATILITY ENGINE
# ============================================================


class VolatilityEngine:
    """
    Volatility analytics.
    """

    # --------------------------------------------------------

    @staticmethod
    def annualized_volatility(
        covariance_matrix:
        pd.DataFrame,
    ) -> pd.Series:

        vols = (
            RiskMatrixUtils
            .covariance_to_volatility(
                covariance_matrix
            )
        )

        return vols * np.sqrt(
            TRADING_DAYS
        )


# ============================================================
# RISK BUDGET ENGINE
# ============================================================


class RiskBudgetEngine:
    """
    Institutional risk budgets.
    """

    # --------------------------------------------------------

    @staticmethod
    def equal_risk_budget(
        assets:
        pd.Index,
    ) -> pd.Series:

        n_assets = len(
            assets
        )

        return pd.Series(

            1.0
            /
            n_assets,

            index=
            assets,
        )


# ============================================================
# RISK DIAGNOSTIC ENGINE
# ============================================================


class RiskDiagnosticEngine:
    """
    Risk diagnostics.
    """

    # --------------------------------------------------------

    @staticmethod
    def compute(
        covariance_matrix:
        pd.DataFrame,
        volatility:
        pd.Series,
        correlation_matrix:
        pd.DataFrame,
    ) -> RiskDiagnostics:

        corr_values = (
            correlation_matrix
            .values
        )

        avg_corr = float(
            np.nanmean(
                corr_values
            )
        )

        try:

            condition_number = float(
                np.linalg.cond(
                    covariance_matrix
                    .values
                )
            )

        except Exception:

            condition_number = np.nan

        return RiskDiagnostics(

            average_volatility=
            float(
                volatility.mean()
            ),

            max_volatility=
            float(
                volatility.max()
            ),

            min_volatility=
            float(
                volatility.min()
            ),

            average_correlation=
            avg_corr,

            covariance_condition_number=
            condition_number,
        )


# ============================================================
# STANDARD RISK INTEGRATION
# ============================================================


class RiskModelIntegrationLayer(
    BaseRiskModelIntegration,
):
    """
    Institutional risk adapter.
    """

    # --------------------------------------------------------

    def build_risk_model(
        self,
        inputs:
        PortfolioConstructionInput,
    ) -> RiskModelResult:

        covariance_matrix = (

            inputs.covariance
            .covariance_matrix
            .copy()

        )

        covariance_matrix = (
            RiskMatrixUtils
            .nearest_psd(
                covariance_matrix
            )
        )

        volatility = (

            VolatilityEngine
            .annualized_volatility(
                covariance_matrix
            )
        )

        correlation_matrix = (

            RiskMatrixUtils
            .covariance_to_correlation(
                covariance_matrix
            )
        )

        risk_budget = (

            RiskBudgetEngine
            .equal_risk_budget(
                covariance_matrix.index
            )
        )

        diagnostics = (

            RiskDiagnosticEngine
            .compute(

                covariance_matrix=
                covariance_matrix,

                volatility=
                volatility,

                correlation_matrix=
                correlation_matrix,
            )
        )

        return RiskModelResult(

            covariance_matrix=
            covariance_matrix,

            volatility=
            volatility,

            correlation_matrix=
            correlation_matrix,

            risk_budget=
            risk_budget,

            diagnostics={

                "average_volatility":
                diagnostics
                .average_volatility,

                "max_volatility":
                diagnostics
                .max_volatility,

                "min_volatility":
                diagnostics
                .min_volatility,

                "average_correlation":
                diagnostics
                .average_correlation,

                "condition_number":
                diagnostics
                .covariance_condition_number,
            },
        )


# ============================================================
# FACTOR RISK INTEGRATION
# ============================================================


class FactorRiskIntegrationLayer(
    RiskModelIntegrationLayer,
):
    """
    Extension for factor risk.

    Used later by optimizer
    and attribution modules.
    """

    # --------------------------------------------------------

    def factor_covariance(
        self,
        factor_exposures:
        pd.DataFrame,
    ) -> pd.DataFrame:

        return (
            factor_exposures
            .cov()
        )


# ============================================================
# FACTORY
# ============================================================


class RiskModelIntegrationFactory:
    """
    Risk model factory.
    """

    # --------------------------------------------------------

    @staticmethod
    def standard(
        metadata:
        PortfolioBuilderMetadata,
        config:
        PortfolioBuilderConfig
        | None = None,
    ) -> RiskModelIntegrationLayer:

        return (
            RiskModelIntegrationLayer(

                metadata=
                metadata,

                config=
                config,
            )
        )

    # --------------------------------------------------------

    @staticmethod
    def factor(
        metadata:
        PortfolioBuilderMetadata,
        config:
        PortfolioBuilderConfig
        | None = None,
    ) -> FactorRiskIntegrationLayer:

        return (
            FactorRiskIntegrationLayer(

                metadata=
                metadata,

                config=
                config,
            )
        )
    
# ============================================================
# PART 5 — CONSTRAINT INTEGRATION
# ============================================================

# ============================================================
# CONSTRAINT RESULTS
# ============================================================


@dataclass(slots=True)
class ConstraintViolation:
    """
    Single violation.
    """

    constraint_name: str

    asset: str | None

    current_value: float

    limit_value: float

    message: str


@dataclass(slots=True)
class ConstraintValidationResult:
    """
    Constraint validation output.
    """

    passed: bool

    violations: list[ConstraintViolation]

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class ConstraintSet:
    """
    Standardized constraint package
    consumed by optimizer.
    """

    lower_bounds: pd.Series

    upper_bounds: pd.Series

    sector_min: pd.Series | None

    sector_max: pd.Series | None

    country_min: pd.Series | None

    country_max: pd.Series | None

    max_turnover: float | None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# BASE CONSTRAINT INTEGRATION
# ============================================================


class BaseConstraintIntegration(
    ABC,
):
    """
    Base interface.
    """

    def __init__(
        self,
        metadata:
        PortfolioBuilderMetadata,
        config:
        PortfolioBuilderConfig | None = None,
    ) -> None:

        self.metadata = metadata

        self.config = config

    # --------------------------------------------------------

    @abstractmethod
    def build_constraints(
        self,
        inputs:
        PortfolioConstructionInput,
    ) -> ConstraintSet:
        pass


# ============================================================
# POSITION CONSTRAINT ENGINE
# ============================================================


class PositionConstraintEngine:
    """
    Position-level constraints.
    """

    # --------------------------------------------------------

    @staticmethod
    def lower_bounds(
        assets:
        pd.Index,
        min_weight:
        float,
    ) -> pd.Series:

        return pd.Series(

            min_weight,

            index=
            assets,
        )

    # --------------------------------------------------------

    @staticmethod
    def upper_bounds(
        assets:
        pd.Index,
        max_weight:
        float,
    ) -> pd.Series:

        return pd.Series(

            max_weight,

            index=
            assets,
        )


# ============================================================
# TURNOVER ENGINE
# ============================================================


class TurnoverConstraintEngine:
    """
    Turnover limits.
    """

    # --------------------------------------------------------

    @staticmethod
    def max_turnover(
        inputs:
        PortfolioConstructionInput,
    ) -> float | None:

        if (
            inputs.turnover_constraints
            is None
        ):

            return None

        return float(

            inputs
            .turnover_constraints
            .max_turnover

        )


# ============================================================
# SECTOR ENGINE
# ============================================================


class SectorConstraintEngine:
    """
    Sector constraints.
    """

    # --------------------------------------------------------

    @staticmethod
    def min_constraints(
        inputs:
        PortfolioConstructionInput,
    ) -> pd.Series | None:

        if (
            inputs.sector_constraints
            is None
        ):

            return None

        return (
            inputs
            .sector_constraints
            .sector_min
        )

    # --------------------------------------------------------

    @staticmethod
    def max_constraints(
        inputs:
        PortfolioConstructionInput,
    ) -> pd.Series | None:

        if (
            inputs.sector_constraints
            is None
        ):

            return None

        return (
            inputs
            .sector_constraints
            .sector_max
        )


# ============================================================
# COUNTRY ENGINE
# ============================================================


class CountryConstraintEngine:
    """
    Country constraints.
    """

    # --------------------------------------------------------

    @staticmethod
    def min_constraints(
        inputs:
        PortfolioConstructionInput,
    ) -> pd.Series | None:

        if (
            inputs.country_constraints
            is None
        ):

            return None

        return (
            inputs
            .country_constraints
            .country_min
        )

    # --------------------------------------------------------

    @staticmethod
    def max_constraints(
        inputs:
        PortfolioConstructionInput,
    ) -> pd.Series | None:

        if (
            inputs.country_constraints
            is None
        ):

            return None

        return (
            inputs
            .country_constraints
            .country_max
        )


# ============================================================
# CONSTRAINT DIAGNOSTICS
# ============================================================


class ConstraintDiagnosticEngine:
    """
    Constraint diagnostics.
    """

    # --------------------------------------------------------

    @staticmethod
    def summarize(
        constraint_set:
        ConstraintSet,
    ) -> dict[str, Any]:

        return {

            "n_assets":
            int(
                len(
                    constraint_set
                    .upper_bounds
                )
            ),

            "avg_upper_bound":
            float(
                constraint_set
                .upper_bounds
                .mean()
            ),

            "avg_lower_bound":
            float(
                constraint_set
                .lower_bounds
                .mean()
            ),

            "has_sector_constraints":
            (
                constraint_set
                .sector_max
                is not None
            ),

            "has_country_constraints":
            (
                constraint_set
                .country_max
                is not None
            ),

            "max_turnover":
            constraint_set
            .max_turnover,
        }


# ============================================================
# CONSTRAINT VALIDATOR
# ============================================================


class ConstraintValidator:
    """
    Institutional validation.
    """

    # --------------------------------------------------------

    @staticmethod
    def validate_weights(
        weights:
        pd.Series,
        constraint_set:
        ConstraintSet,
    ) -> (
        ConstraintValidationResult
    ):

        violations = []

        # ------------------------------
        # lower bounds
        # ------------------------------

        lower_breaks = (

            weights

            <

            constraint_set
            .lower_bounds

        )

        for asset in (
            lower_breaks[
                lower_breaks
            ].index
        ):

            violations.append(

                ConstraintViolation(

                    constraint_name=
                    "LOWER_BOUND",

                    asset=
                    str(asset),

                    current_value=
                    float(
                        weights.loc[
                            asset
                        ]
                    ),

                    limit_value=
                    float(
                        constraint_set
                        .lower_bounds
                        .loc[
                            asset
                        ]
                    ),

                    message=
                    "Below minimum weight",
                )
            )

        # ------------------------------
        # upper bounds
        # ------------------------------

        upper_breaks = (

            weights

            >

            constraint_set
            .upper_bounds

        )

        for asset in (
            upper_breaks[
                upper_breaks
            ].index
        ):

            violations.append(

                ConstraintViolation(

                    constraint_name=
                    "UPPER_BOUND",

                    asset=
                    str(asset),

                    current_value=
                    float(
                        weights.loc[
                            asset
                        ]
                    ),

                    limit_value=
                    float(
                        constraint_set
                        .upper_bounds
                        .loc[
                            asset
                        ]
                    ),

                    message=
                    "Above maximum weight",
                )
            )

        return ConstraintValidationResult(

            passed=
            len(
                violations
            )
            == 0,

            violations=
            violations,

            diagnostics={

                "n_violations":
                len(
                    violations
                )
            },
        )


# ============================================================
# STANDARD CONSTRAINT INTEGRATION
# ============================================================


class ConstraintIntegrationLayer(
    BaseConstraintIntegration,
):
    """
    Institutional constraint adapter.
    """

    # --------------------------------------------------------

    def build_constraints(
        self,
        inputs:
        PortfolioConstructionInput,
    ) -> ConstraintSet:

        assets = (

            inputs.forecast
            .expected_returns
            .index

        )

        # ------------------------------
        # position limits
        # ------------------------------

        if (
            inputs.position_limits
            is not None
        ):

            min_weight = (

                inputs
                .position_limits
                .min_weight

            )

            max_weight = (

                inputs
                .position_limits
                .max_weight

            )

        else:

            min_weight = (

                self.config
                .min_position_weight

                if self.config
                else 0.0
            )

            max_weight = (

                self.config
                .max_position_weight

                if self.config
                else 1.0
            )

        lower_bounds = (

            PositionConstraintEngine
            .lower_bounds(

                assets=
                assets,

                min_weight=
                min_weight,
            )
        )

        upper_bounds = (

            PositionConstraintEngine
            .upper_bounds(

                assets=
                assets,

                max_weight=
                max_weight,
            )
        )

        constraint_set = ConstraintSet(

            lower_bounds=
            lower_bounds,

            upper_bounds=
            upper_bounds,

            sector_min=
            SectorConstraintEngine
            .min_constraints(
                inputs
            ),

            sector_max=
            SectorConstraintEngine
            .max_constraints(
                inputs
            ),

            country_min=
            CountryConstraintEngine
            .min_constraints(
                inputs
            ),

            country_max=
            CountryConstraintEngine
            .max_constraints(
                inputs
            ),

            max_turnover=
            TurnoverConstraintEngine
            .max_turnover(
                inputs
            ),

            diagnostics={},
        )

        constraint_set.diagnostics = (

            ConstraintDiagnosticEngine
            .summarize(
                constraint_set
            )
        )

        return constraint_set


# ============================================================
# FACTORY
# ============================================================


class ConstraintIntegrationFactory:
    """
    Constraint factory.
    """

    # --------------------------------------------------------

    @staticmethod
    def standard(
        metadata:
        PortfolioBuilderMetadata,
        config:
        PortfolioBuilderConfig
        | None = None,
    ) -> ConstraintIntegrationLayer:

        return (
            ConstraintIntegrationLayer(

                metadata=
                metadata,

                config=
                config,
            )
        )
    
# ============================================================
# PART 6 — OPTIMIZATION INTEGRATION
# ============================================================

# ============================================================
# OPTIMIZATION RESULTS
# ============================================================


@dataclass(slots=True)
class OptimizationResult:
    """
    Standardized optimizer output.
    """

    weights: pd.Series

    objective_value: float

    expected_return: float

    expected_volatility: float

    expected_sharpe: float

    optimizer_name: str

    success: bool

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# OPTIMIZATION DIAGNOSTICS
# ============================================================


@dataclass(slots=True)
class OptimizationDiagnostics:
    """
    Optimizer diagnostics.
    """

    n_assets: int

    gross_exposure: float

    net_exposure: float

    largest_position: float

    smallest_position: float

    effective_n: float


# ============================================================
# BASE OPTIMIZER
# ============================================================


class BaseOptimizationIntegration(
    ABC,
):
    """
    Base optimizer interface.
    """

    def __init__(
        self,
        metadata:
        PortfolioBuilderMetadata,
        config:
        PortfolioBuilderConfig | None = None,
    ) -> None:

        self.metadata = metadata

        self.config = config

    # --------------------------------------------------------

    @abstractmethod
    def optimize(
        self,
        forecast:
        ForecastResult,

        risk:
        RiskModelResult,

        constraints:
        ConstraintSet,
    ) -> OptimizationResult:
        pass


# ============================================================
# WEIGHT UTILITIES
# ============================================================


class OptimizationWeightUtils:
    """
    Shared weight utilities.
    """

    # --------------------------------------------------------

    @staticmethod
    def normalize(
        weights:
        pd.Series,
    ) -> pd.Series:

        total = float(
            weights.sum()
        )

        if (
            abs(total)
            < 1e-12
        ):

            return weights

        return (
            weights / total
        )

    # --------------------------------------------------------

    @staticmethod
    def apply_bounds(
        weights:
        pd.Series,

        lower:
        pd.Series,

        upper:
        pd.Series,
    ) -> pd.Series:

        clipped = weights.clip(
            lower=lower,
            upper=upper,
        )

        return (
            OptimizationWeightUtils
            .normalize(
                clipped
            )
        )

    # --------------------------------------------------------

    @staticmethod
    def effective_n(
        weights:
        pd.Series,
    ) -> float:

        denom = float(
            np.square(
                weights
            ).sum()
        )

        if denom <= 0:

            return 0.0

        return (
            1.0 / denom
        )


# ============================================================
# PORTFOLIO METRICS ENGINE
# ============================================================


class PortfolioMetricEngine:
    """
    Portfolio statistics.
    """

    # --------------------------------------------------------

    @staticmethod
    def expected_return(
        weights:
        pd.Series,

        expected_returns:
        pd.Series,
    ) -> float:

        aligned = pd.concat(
            [
                weights,
                expected_returns,
            ],
            axis=1,
        ).fillna(0.0)

        return float(

            aligned.iloc[:, 0]
            .dot(
                aligned.iloc[:, 1]
            )

        )

    # --------------------------------------------------------

    @staticmethod
    def expected_volatility(
        weights:
        pd.Series,

        covariance:
        pd.DataFrame,
    ) -> float:

        w = (
            weights
            .reindex(
                covariance.index
            )
            .fillna(0.0)
            .values
        )

        vol = np.sqrt(

            np.dot(
                w.T,
                np.dot(
                    covariance.values,
                    w,
                ),
            )

        )

        return float(
            vol
        )

    # --------------------------------------------------------

    @staticmethod
    def sharpe(
        expected_return:
        float,

        expected_volatility:
        float,
    ) -> float:

        if (
            expected_volatility
            <= 0
        ):

            return 0.0

        return float(

            expected_return
            /
            expected_volatility

        )


# ============================================================
# SIMPLE INSTITUTIONAL OPTIMIZER
# ============================================================


class MeanVarianceOptimizer:
    """
    Lightweight institutional optimizer.

    No dependency on scipy.

    Forecast weighted by
    inverse volatility.
    """

    # --------------------------------------------------------

    def optimize(
        self,
        forecast:
        ForecastResult,

        risk:
        RiskModelResult,

        constraints:
        ConstraintSet,
    ) -> pd.Series:

        returns = (
            forecast
            .expected_returns
        )

        vol = (
            risk
            .volatility
        )

        scores = (

            returns

            /

            vol.replace(
                0,
                np.nan,
            )

        )

        scores = (
            scores.fillna(0.0)
        )

        scores = (
            scores.clip(
                lower=0.0
            )
        )

        if (
            scores.sum()
            <= 0
        ):

            scores[:] = (
                1.0
            )

        weights = (

            scores
            /
            scores.sum()

        )

        weights = (

            OptimizationWeightUtils
            .apply_bounds(

                weights=
                weights,

                lower=
                constraints
                .lower_bounds,

                upper=
                constraints
                .upper_bounds,
            )
        )

        return weights


# ============================================================
# OPTIMIZATION DIAGNOSTIC ENGINE
# ============================================================


class OptimizationDiagnosticEngine:
    """
    Optimization diagnostics.
    """

    # --------------------------------------------------------

    @staticmethod
    def compute(
        weights:
        pd.Series,
    ) -> OptimizationDiagnostics:

        return OptimizationDiagnostics(

            n_assets=
            int(
                len(weights)
            ),

            gross_exposure=
            float(
                np.abs(
                    weights
                ).sum()
            ),

            net_exposure=
            float(
                weights.sum()
            ),

            largest_position=
            float(
                weights.max()
            ),

            smallest_position=
            float(
                weights.min()
            ),

            effective_n=
            OptimizationWeightUtils
            .effective_n(
                weights
            ),
        )


# ============================================================
# STANDARD OPTIMIZATION LAYER
# ============================================================


class OptimizationIntegrationLayer(
    BaseOptimizationIntegration,
):
    """
    Institutional optimization adapter.
    """

    # --------------------------------------------------------

    def __init__(
        self,
        metadata:
        PortfolioBuilderMetadata,

        config:
        PortfolioBuilderConfig
        | None = None,
    ) -> None:

        super().__init__(
            metadata=
            metadata,

            config=
            config,
        )

        self.optimizer = (
            MeanVarianceOptimizer()
        )

    # --------------------------------------------------------

    def optimize(
        self,
        forecast:
        ForecastResult,

        risk:
        RiskModelResult,

        constraints:
        ConstraintSet,
    ) -> OptimizationResult:

        weights = (
            self.optimizer
            .optimize(

                forecast=
                forecast,

                risk=
                risk,

                constraints=
                constraints,
            )
        )

        expected_return = (

            PortfolioMetricEngine
            .expected_return(

                weights=
                weights,

                expected_returns=
                forecast
                .expected_returns,
            )
        )

        expected_volatility = (

            PortfolioMetricEngine
            .expected_volatility(

                weights=
                weights,

                covariance=
                risk
                .covariance_matrix,
            )
        )

        expected_sharpe = (

            PortfolioMetricEngine
            .sharpe(

                expected_return=
                expected_return,

                expected_volatility=
                expected_volatility,
            )
        )

        diagnostics = (

            OptimizationDiagnosticEngine
            .compute(
                weights
            )
        )

        return OptimizationResult(

            weights=
            weights,

            objective_value=
            expected_sharpe,

            expected_return=
            expected_return,

            expected_volatility=
            expected_volatility,

            expected_sharpe=
            expected_sharpe,

            optimizer_name=
            "MeanVarianceOptimizer",

            success=
            True,

            diagnostics={

                "effective_n":
                diagnostics
                .effective_n,

                "gross_exposure":
                diagnostics
                .gross_exposure,

                "net_exposure":
                diagnostics
                .net_exposure,

                "largest_position":
                diagnostics
                .largest_position,

                "smallest_position":
                diagnostics
                .smallest_position,
            },
        )


# ============================================================
# FACTORY
# ============================================================


class OptimizationIntegrationFactory:
    """
    Optimization factory.
    """

    # --------------------------------------------------------

    @staticmethod
    def standard(
        metadata:
        PortfolioBuilderMetadata,

        config:
        PortfolioBuilderConfig
        | None = None,
    ) -> (
        OptimizationIntegrationLayer
    ):

        return (
            OptimizationIntegrationLayer(

                metadata=
                metadata,

                config=
                config,
            )
        )
    
# ============================================================
# PART 7 — PORTFOLIO ASSEMBLY
# ============================================================

# ============================================================
# PORTFOLIO ASSEMBLY RESULTS
# ============================================================


@dataclass(slots=True)
class PortfolioAssemblyResult:
    """
    Portfolio assembly output.
    """

    portfolio: TargetPortfolio

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# PORTFOLIO EXPOSURE METRICS
# ============================================================


@dataclass(slots=True)
class PortfolioExposureMetrics:
    """
    Exposure metrics.
    """

    gross_exposure: float

    net_exposure: float

    long_exposure: float

    short_exposure: float

    cash_weight: float


# ============================================================
# POSITION BUILDER
# ============================================================


class PositionBuilder:
    """
    Converts weights into positions.
    """

    # --------------------------------------------------------

    @staticmethod
    def build_positions(
        weights:
        pd.Series,

        prices:
        pd.Series | None = None,

        portfolio_value:
        float = 1.0,
    ) -> list[PortfolioPosition]:

        positions = []

        for asset, weight in weights.items():

            price = None
            quantity = None
            market_value = None

            if (
                prices is not None
                and asset in prices.index
            ):

                price = float(
                    prices.loc[asset]
                )

                market_value = (
                    portfolio_value
                    * float(weight)
                )

                if (
                    price > 0
                ):

                    quantity = (
                        market_value
                        / price
                    )

            positions.append(

                PortfolioPosition(

                    asset=
                    str(asset),

                    weight=
                    float(weight),

                    quantity=
                    quantity,

                    price=
                    price,

                    market_value=
                    market_value,
                )
            )

        return positions


# ============================================================
# EXPOSURE ENGINE
# ============================================================


class PortfolioExposureEngine:
    """
    Exposure calculations.
    """

    # --------------------------------------------------------

    @staticmethod
    def compute(
        weights:
        pd.Series,
    ) -> (
        PortfolioExposureMetrics
    ):

        long_weights = (
            weights[
                weights > 0
            ]
        )

        short_weights = (
            weights[
                weights < 0
            ]
        )

        long_exposure = float(
            long_weights.sum()
        )

        short_exposure = float(
            np.abs(
                short_weights.sum()
            )
        )

        gross_exposure = float(
            np.abs(
                weights
            ).sum()
        )

        net_exposure = float(
            weights.sum()
        )

        cash_weight = float(
            1.0
            -
            net_exposure
        )

        return (
            PortfolioExposureMetrics(

                gross_exposure=
                gross_exposure,

                net_exposure=
                net_exposure,

                long_exposure=
                long_exposure,

                short_exposure=
                short_exposure,

                cash_weight=
                cash_weight,
            )
        )


# ============================================================
# PORTFOLIO STATISTICS ENGINE
# ============================================================


class PortfolioAssemblyStatistics:
    """
    Portfolio metrics.
    """

    # --------------------------------------------------------

    @staticmethod
    def expected_return(
        optimization:
        OptimizationResult,
    ) -> float:

        return float(
            optimization
            .expected_return
        )

    # --------------------------------------------------------

    @staticmethod
    def expected_volatility(
        optimization:
        OptimizationResult,
    ) -> float:

        return float(
            optimization
            .expected_volatility
        )

    # --------------------------------------------------------

    @staticmethod
    def expected_sharpe(
        optimization:
        OptimizationResult,
    ) -> float:

        return float(
            optimization
            .expected_sharpe
        )


# ============================================================
# PORTFOLIO DIAGNOSTICS
# ============================================================


class PortfolioAssemblyDiagnostics:
    """
    Diagnostics.
    """

    # --------------------------------------------------------

    @staticmethod
    def compute(
        weights:
        pd.Series,
    ) -> dict[str, Any]:

        non_zero = int(
            (
                weights.abs()
                > 1e-8
            ).sum()
        )

        effective_n = 0.0

        denom = float(
            np.square(
                weights
            ).sum()
        )

        if denom > 0:

            effective_n = (
                1.0 / denom
            )

        return {

            "n_assets":
            int(
                len(weights)
            ),

            "active_positions":
            non_zero,

            "largest_weight":
            float(
                weights.max()
            ),

            "smallest_weight":
            float(
                weights.min()
            ),

            "effective_n":
            float(
                effective_n
            ),
        }


# ============================================================
# TARGET PORTFOLIO BUILDER
# ============================================================


class TargetPortfolioBuilder:
    """
    Creates institutional portfolio.
    """

    # --------------------------------------------------------

    @staticmethod
    def build(
        optimization:
        OptimizationResult,

        prices:
        pd.Series | None = None,

        portfolio_value:
        float = 1.0,
    ) -> TargetPortfolio:

        weights = (
            optimization
            .weights
        )

        exposure = (

            PortfolioExposureEngine
            .compute(
                weights
            )
        )

        portfolio_weights = (
            PortfolioWeights(

                weights=
                weights,

                cash_weight=
                exposure.cash_weight,

                gross_exposure=
                exposure.gross_exposure,

                net_exposure=
                exposure.net_exposure,
            )
        )

        positions = (

            PositionBuilder
            .build_positions(

                weights=
                weights,

                prices=
                prices,

                portfolio_value=
                portfolio_value,
            )
        )

        return TargetPortfolio(

            weights=
            portfolio_weights,

            positions=
            positions,

            expected_return=
            optimization
            .expected_return,

            expected_volatility=
            optimization
            .expected_volatility,

            expected_sharpe=
            optimization
            .expected_sharpe,
        )


# ============================================================
# PORTFOLIO ASSEMBLY LAYER
# ============================================================


class PortfolioAssemblyLayer:
    """
    Institutional portfolio assembly.
    """

    def __init__(
        self,
        metadata:
        PortfolioBuilderMetadata,

        config:
        PortfolioBuilderConfig
        | None = None,
    ) -> None:

        self.metadata = metadata

        self.config = config

    # --------------------------------------------------------

    def assemble(
        self,

        optimization:
        OptimizationResult,

        inputs:
        PortfolioConstructionInput,
    ) -> (
        PortfolioAssemblyResult
    ):

        prices = None

        if (
            inputs.prices
            is not None
        ):

            prices = (
                inputs.prices
            )

        portfolio = (

            TargetPortfolioBuilder
            .build(

                optimization=
                optimization,

                prices=
                prices,
            )
        )

        diagnostics = (

            PortfolioAssemblyDiagnostics
            .compute(

                optimization
                .weights
            )
        )

        return PortfolioAssemblyResult(

            portfolio=
            portfolio,

            diagnostics=
            diagnostics,
        )


# ============================================================
# FACTORY
# ============================================================


class PortfolioAssemblyFactory:
    """
    Portfolio assembly factory.
    """

    # --------------------------------------------------------

    @staticmethod
    def standard(
        metadata:
        PortfolioBuilderMetadata,

        config:
        PortfolioBuilderConfig
        | None = None,
    ) -> (
        PortfolioAssemblyLayer
    ):

        return (
            PortfolioAssemblyLayer(

                metadata=
                metadata,

                config=
                config,
            )
        )
    
# ============================================================
# PART 8 — REBALANCE GENERATION
# ============================================================

# ============================================================
# REBALANCE RESULTS
# ============================================================


@dataclass(slots=True)
class RebalanceGenerationResult:
    """
    Rebalance output.
    """

    rebalance_plan: RebalancePlan

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# TRADE INSTRUCTION
# ============================================================


@dataclass(slots=True)
class TradeInstruction:
    """
    Institutional trade instruction.
    """

    asset: str

    current_weight: float

    target_weight: float

    trade_weight: float

    side: str

    estimated_notional: float

    priority: int = 1


# ============================================================
# TURNOVER ENGINE
# ============================================================


class RebalanceTurnoverEngine:
    """
    Turnover calculations.
    """

    # --------------------------------------------------------

    @staticmethod
    def turnover(
        current_weights:
        pd.Series,

        target_weights:
        pd.Series,
    ) -> float:

        aligned = pd.concat(

            [
                current_weights,
                target_weights,
            ],

            axis=1,
        ).fillna(0.0)

        return float(

            np.abs(

                aligned.iloc[:, 0]

                -

                aligned.iloc[:, 1]

            ).sum()

            / 2.0

        )


# ============================================================
# TRADE GENERATOR
# ============================================================


class TradeGenerator:
    """
    Creates rebalance trades.
    """

    # --------------------------------------------------------

    @staticmethod
    def generate_orders(
        current_weights:
        pd.Series,

        target_weights:
        pd.Series,

        portfolio_value:
        float = 1.0,
    ) -> list[RebalanceOrder]:

        all_assets = (
            current_weights.index
            .union(
                target_weights.index
            )
        )

        current = (
            current_weights
            .reindex(
                all_assets
            )
            .fillna(0.0)
        )

        target = (
            target_weights
            .reindex(
                all_assets
            )
            .fillna(0.0)
        )

        orders = []

        for asset in all_assets:

            current_weight = float(
                current.loc[asset]
            )

            target_weight = float(
                target.loc[asset]
            )

            trade_weight = (
                target_weight
                -
                current_weight
            )

            if abs(
                trade_weight
            ) < 1e-8:

                continue

            estimated_notional = (

                trade_weight
                *
                portfolio_value

            )

            orders.append(

                RebalanceOrder(

                    asset=
                    str(asset),

                    current_weight=
                    current_weight,

                    target_weight=
                    target_weight,

                    trade_weight=
                    trade_weight,

                    estimated_notional=
                    estimated_notional,
                )
            )

        return orders


# ============================================================
# INSTRUCTION GENERATOR
# ============================================================


class TradeInstructionGenerator:
    """
    Generates institutional instructions.
    """

    # --------------------------------------------------------

    @staticmethod
    def generate(
        orders:
        list[RebalanceOrder],
    ) -> list[
        TradeInstruction
    ]:

        instructions = []

        for order in orders:

            side = (
                "BUY"
                if order.trade_weight > 0
                else "SELL"
            )

            instructions.append(

                TradeInstruction(

                    asset=
                    order.asset,

                    current_weight=
                    order.current_weight,

                    target_weight=
                    order.target_weight,

                    trade_weight=
                    order.trade_weight,

                    side=
                    side,

                    estimated_notional=
                    order.estimated_notional,
                )
            )

        return instructions


# ============================================================
# DRIFT ANALYTICS
# ============================================================


class PortfolioDriftEngine:
    """
    Portfolio drift analysis.
    """

    # --------------------------------------------------------

    @staticmethod
    def drift(
        current_weights:
        pd.Series,

        target_weights:
        pd.Series,
    ) -> pd.Series:

        aligned = pd.concat(

            [
                current_weights,
                target_weights,
            ],

            axis=1,
        ).fillna(0.0)

        return (

            aligned.iloc[:, 1]

            -

            aligned.iloc[:, 0]

        )


# ============================================================
# REBALANCE DIAGNOSTICS
# ============================================================


class RebalanceDiagnosticEngine:
    """
    Rebalance diagnostics.
    """

    # --------------------------------------------------------

    @staticmethod
    def compute(
        orders:
        list[RebalanceOrder],

        turnover:
        float,
    ) -> dict[str, Any]:

        buy_count = sum(

            1

            for o in orders

            if o.trade_weight > 0

        )

        sell_count = sum(

            1

            for o in orders

            if o.trade_weight < 0

        )

        total_notional = float(

            np.sum(

                [
                    abs(
                        o.estimated_notional
                    )

                    for o in orders
                ]

            )

        )

        return {

            "num_orders":
            len(orders),

            "buy_orders":
            buy_count,

            "sell_orders":
            sell_count,

            "turnover":
            turnover,

            "total_notional":
            total_notional,
        }


# ============================================================
# REBALANCE ENGINE
# ============================================================


class RebalanceGenerationEngine:
    """
    Institutional rebalance engine.
    """

    def __init__(
        self,
        metadata:
        PortfolioBuilderMetadata,

        config:
        PortfolioBuilderConfig
        | None = None,
    ) -> None:

        self.metadata = metadata

        self.config = config

    # --------------------------------------------------------

    def generate(
        self,

        current_portfolio:
        CurrentPortfolioInput
        | None,

        target_portfolio:
        TargetPortfolio,

        rebalance_type:
        RebalanceType = (
            RebalanceType
            .CALENDAR
        ),
    ) -> (
        RebalanceGenerationResult
    ):

        target_weights = (

            target_portfolio
            .weights
            .weights

        )

        if (
            current_portfolio
            is None
        ):

            current_weights = pd.Series(

                0.0,

                index=
                target_weights.index,
            )

        else:

            current_weights = (

                current_portfolio
                .current_weights

            )

        orders = (

            TradeGenerator
            .generate_orders(

                current_weights=
                current_weights,

                target_weights=
                target_weights,
            )
        )

        turnover = (

            RebalanceTurnoverEngine
            .turnover(

                current_weights=
                current_weights,

                target_weights=
                target_weights,
            )
        )

        rebalance_plan = (

            RebalancePlan(

                rebalance_type=
                rebalance_type,

                orders=
                orders,

                turnover=
                turnover,
            )
        )

        diagnostics = (

            RebalanceDiagnosticEngine
            .compute(

                orders=
                orders,

                turnover=
                turnover,
            )
        )

        return RebalanceGenerationResult(

            rebalance_plan=
            rebalance_plan,

            diagnostics=
            diagnostics,
        )


# ============================================================
# FACTORY
# ============================================================


class RebalanceGenerationFactory:
    """
    Rebalance factory.
    """

    # --------------------------------------------------------

    @staticmethod
    def standard(
        metadata:
        PortfolioBuilderMetadata,

        config:
        PortfolioBuilderConfig
        | None = None,
    ) -> (
        RebalanceGenerationEngine
    ):

        return (

            RebalanceGenerationEngine(

                metadata=
                metadata,

                config=
                config,
            )

        )
    
# ============================================================
# PART 9 — PORTFOLIO VALIDATION
# ============================================================

# ============================================================
# VALIDATION RESULT OBJECTS
# ============================================================


@dataclass(slots=True)
class ValidationIssue:
    """
    Individual validation issue.
    """

    rule_name: str

    severity: str

    message: str

    value: float | None = None

    limit: float | None = None


@dataclass(slots=True)
class PortfolioValidationResult:
    """
    Institutional validation result.
    """

    passed: bool

    issues: list[ValidationIssue]

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# BASE VALIDATOR
# ============================================================


class BasePortfolioValidator(
    ABC,
):
    """
    Base validator interface.
    """

    @abstractmethod
    def validate(
        self,
        portfolio:
        TargetPortfolio,
    ) -> (
        list[ValidationIssue]
    ):
        pass


# ============================================================
# POSITION LIMIT VALIDATOR
# ============================================================


class PositionLimitValidator(
    BasePortfolioValidator,
):
    """
    Position concentration checks.
    """

    def __init__(
        self,
        constraint_set:
        ConstraintSet,
    ) -> None:

        self.constraint_set = (
            constraint_set
        )

    # --------------------------------------------------------

    def validate(
        self,
        portfolio:
        TargetPortfolio,
    ) -> (
        list[ValidationIssue]
    ):

        issues = []

        weights = (
            portfolio
            .weights
            .weights
        )

        for asset in weights.index:

            weight = float(
                weights.loc[asset]
            )

            upper = float(
                self.constraint_set
                .upper_bounds
                .loc[asset]
            )

            lower = float(
                self.constraint_set
                .lower_bounds
                .loc[asset]
            )

            if weight > upper:

                issues.append(

                    ValidationIssue(

                        rule_name=
                        "POSITION_MAX",

                        severity=
                        "ERROR",

                        message=
                        f"{asset} above max weight",

                        value=
                        weight,

                        limit=
                        upper,
                    )
                )

            if weight < lower:

                issues.append(

                    ValidationIssue(

                        rule_name=
                        "POSITION_MIN",

                        severity=
                        "ERROR",

                        message=
                        f"{asset} below min weight",

                        value=
                        weight,

                        limit=
                        lower,
                    )
                )

        return issues


# ============================================================
# EXPOSURE VALIDATOR
# ============================================================


class ExposureValidator(
    BasePortfolioValidator,
):
    """
    Exposure checks.
    """

    def validate(
        self,
        portfolio:
        TargetPortfolio,
    ) -> (
        list[ValidationIssue]
    ):

        issues = []

        gross = float(
            portfolio
            .weights
            .gross_exposure
        )

        net = float(
            portfolio
            .weights
            .net_exposure
        )

        if gross > 2.0:

            issues.append(

                ValidationIssue(

                    rule_name=
                    "GROSS_EXPOSURE",

                    severity=
                    "WARNING",

                    message=
                    "Gross exposure exceeds 200%",

                    value=
                    gross,

                    limit=
                    2.0,
                )
            )

        if abs(net) > 1.0:

            issues.append(

                ValidationIssue(

                    rule_name=
                    "NET_EXPOSURE",

                    severity=
                    "ERROR",

                    message=
                    "Net exposure exceeds 100%",

                    value=
                    net,

                    limit=
                    1.0,
                )
            )

        return issues


# ============================================================
# TURNOVER VALIDATOR
# ============================================================


class TurnoverValidator:
    """
    Turnover checks.
    """

    # --------------------------------------------------------

    @staticmethod
    def validate(
        rebalance:
        RebalanceGenerationResult,

        constraint_set:
        ConstraintSet,
    ) -> (
        list[ValidationIssue]
    ):

        issues = []

        if (
            constraint_set.max_turnover
            is None
        ):

            return issues

        turnover = float(

            rebalance
            .rebalance_plan
            .turnover

        )

        if (
            turnover
            >
            constraint_set
            .max_turnover
        ):

            issues.append(

                ValidationIssue(

                    rule_name=
                    "TURNOVER",

                    severity=
                    "WARNING",

                    message=
                    "Turnover limit exceeded",

                    value=
                    turnover,

                    limit=
                    constraint_set
                    .max_turnover,
                )
            )

        return issues


# ============================================================
# RISK VALIDATOR
# ============================================================


class RiskValidator:
    """
    Portfolio risk checks.
    """

    # --------------------------------------------------------

    @staticmethod
    def validate(
        portfolio:
        TargetPortfolio,

        risk:
        RiskModelResult,
    ) -> (
        list[ValidationIssue]
    ):

        issues = []

        volatility = float(
            portfolio
            .expected_volatility
        )

        if volatility > 0.50:

            issues.append(

                ValidationIssue(

                    rule_name=
                    "VOLATILITY",

                    severity=
                    "WARNING",

                    message=
                    "Expected volatility too high",

                    value=
                    volatility,

                    limit=
                    0.50,
                )
            )

        return issues


# ============================================================
# LIQUIDITY VALIDATOR
# ============================================================


class LiquidityValidator:
    """
    Liquidity checks.
    """

    # --------------------------------------------------------

    @staticmethod
    def validate(
        portfolio:
        TargetPortfolio,
    ) -> (
        list[ValidationIssue]
    ):

        issues = []

        # Placeholder for future ADV
        # integration.

        return issues


# ============================================================
# VALIDATION ENGINE
# ============================================================


class PortfolioValidationEngine:
    """
    Institutional portfolio validator.
    """

    def __init__(
        self,
        metadata:
        PortfolioBuilderMetadata,

        config:
        PortfolioBuilderConfig
        | None = None,
    ) -> None:

        self.metadata = metadata

        self.config = config

    # --------------------------------------------------------

    def validate(
        self,

        portfolio:
        TargetPortfolio,

        risk:
        RiskModelResult,

        constraint_set:
        ConstraintSet,

        rebalance:
        RebalanceGenerationResult,
    ) -> (
        PortfolioValidationResult
    ):

        issues = []

        # ------------------------------
        # position limits
        # ------------------------------

        issues.extend(

            PositionLimitValidator(
                constraint_set
            ).validate(
                portfolio
            )

        )

        # ------------------------------
        # exposures
        # ------------------------------

        issues.extend(

            ExposureValidator()
            .validate(
                portfolio
            )

        )

        # ------------------------------
        # turnover
        # ------------------------------

        issues.extend(

            TurnoverValidator
            .validate(

                rebalance=
                rebalance,

                constraint_set=
                constraint_set,
            )

        )

        # ------------------------------
        # risk
        # ------------------------------

        issues.extend(

            RiskValidator
            .validate(

                portfolio=
                portfolio,

                risk=
                risk,
            )

        )

        # ------------------------------
        # liquidity
        # ------------------------------

        issues.extend(

            LiquidityValidator
            .validate(
                portfolio
            )

        )

        passed = (

            len(

                [

                    i

                    for i in issues

                    if i.severity
                    == "ERROR"

                ]

            )

            == 0

        )

        diagnostics = {

            "total_issues":
            len(issues),

            "errors":
            sum(

                1

                for i in issues

                if i.severity
                == "ERROR"

            ),

            "warnings":
            sum(

                1

                for i in issues

                if i.severity
                == "WARNING"

            ),
        }

        return PortfolioValidationResult(

            passed=
            passed,

            issues=
            issues,

            diagnostics=
            diagnostics,
        )


# ============================================================
# FACTORY
# ============================================================


class PortfolioValidationFactory:
    """
    Validation factory.
    """

    # --------------------------------------------------------

    @staticmethod
    def standard(
        metadata:
        PortfolioBuilderMetadata,

        config:
        PortfolioBuilderConfig
        | None = None,
    ) -> (
        PortfolioValidationEngine
    ):

        return (
            PortfolioValidationEngine(

                metadata=
                metadata,

                config=
                config,
            )
        )
    
# ============================================================
# PART 10 — INSTITUTIONAL REPORTING LAYER
# ============================================================

import json
from dataclasses import asdict


# ============================================================
# MASTER REPORT OBJECTS
# ============================================================


@dataclass(slots=True)
class PortfolioConstructionSummary:
    """
    Executive summary.
    """

    expected_return: float

    expected_volatility: float

    expected_sharpe: float

    gross_exposure: float

    net_exposure: float

    turnover: float

    validation_passed: bool

    number_of_positions: int


# ============================================================
# INSTITUTIONAL PORTFOLIO CONSTRUCTION REPORT
# ============================================================

@dataclass(slots=True)
class InstitutionalPortfolioConstructionReport:
    """
    Master portfolio construction report.

    Final output of the portfolio construction pipeline.

    Contains:

        Forecast
        Risk
        Constraints
        Optimization
        Portfolio Assembly
        Rebalance
        Validation
        Diagnostics

    Acts as the single source of truth
    for downstream reporting,
    attribution,
    analytics,
    monitoring,
    and audit workflows.
    """

    # --------------------------------------------------------
    # Core Metadata
    # --------------------------------------------------------

    metadata: PortfolioBuilderMetadata

    # --------------------------------------------------------
    # Executive Summary
    # --------------------------------------------------------

    summary: PortfolioConstructionSummary

    # --------------------------------------------------------
    # Pipeline Outputs
    # --------------------------------------------------------

    forecast_result: ForecastResult | None = None
    risk_result: RiskModelResult | None = None
    constraint_set: ConstraintSet | None = None
    optimization_result: OptimizationResult | None = None
    portfolio_result: PortfolioAssemblyResult | None = None
    rebalance_result: RebalanceGenerationResult | None = None
    validation_result: PortfolioValidationResult | None = None

    # --------------------------------------------------------
    # Institutional Diagnostics
    # --------------------------------------------------------

    diagnostics_report: InstitutionalDiagnosticsPackage | None = None

    # --------------------------------------------------------
    # Audit / Runtime Diagnostics
    # --------------------------------------------------------

    runtime_diagnostics: dict[str, Any] = field(
        default_factory=dict
    )
# ============================================================
# SUMMARY BUILDER
# ============================================================


class PortfolioConstructionSummaryBuilder:
    """
    Builds executive summary.
    """

    # --------------------------------------------------------

    @staticmethod
    def build(
        portfolio_result:
        PortfolioAssemblyResult,

        rebalance_result:
        RebalanceGenerationResult,

        validation_result:
        PortfolioValidationResult,
    ) -> (
        PortfolioConstructionSummary
    ):

        portfolio = (
            portfolio_result
            .portfolio
        )

        return PortfolioConstructionSummary(

            expected_return=
            float(
                portfolio
                .expected_return
            ),

            expected_volatility=
            float(
                portfolio
                .expected_volatility
            ),

            expected_sharpe=
            float(
                portfolio
                .expected_sharpe
            ),

            gross_exposure=
            float(
                portfolio
                .weights
                .gross_exposure
            ),

            net_exposure=
            float(
                portfolio
                .weights
                .net_exposure
            ),

            turnover=
            float(
                rebalance_result
                .rebalance_plan
                .turnover
            ),

            validation_passed=
            bool(
                validation_result
                .passed
            ),

            number_of_positions=
            int(
                len(
                    portfolio
                    .positions
                )
            ),
        )


# ============================================================
# DIAGNOSTICS BUILDER
# ============================================================


class PortfolioReportDiagnosticsBuilder:
    """
    Consolidates diagnostics.
    """

    # --------------------------------------------------------

    @staticmethod
    def build(
        forecast_result:
        ForecastResult | None,

        risk_result:
        RiskModelResult | None,

        optimization_result:
        OptimizationResult | None,

        portfolio_result:
        PortfolioAssemblyResult | None,

        rebalance_result:
        RebalanceGenerationResult | None,

        validation_result:
        PortfolioValidationResult | None,
    ) -> dict[str, Any]:

        diagnostics = {}

        if forecast_result is not None:

            diagnostics[
                "forecast"
            ] = (
                forecast_result
                .diagnostics
            )

        if risk_result is not None:

            diagnostics[
                "risk"
            ] = (
                risk_result
                .diagnostics
            )

        if optimization_result is not None:

            diagnostics[
                "optimization"
            ] = (
                optimization_result
                .diagnostics
            )

        if portfolio_result is not None:

            diagnostics[
                "portfolio"
            ] = (
                portfolio_result
                .diagnostics
            )

        if rebalance_result is not None:

            diagnostics[
                "rebalance"
            ] = (
                rebalance_result
                .diagnostics
            )

        if validation_result is not None:

            diagnostics[
                "validation"
            ] = (
                validation_result
                .diagnostics
            )

        return diagnostics


# ============================================================
# REPORT BUILDER
# ============================================================


class InstitutionalPortfolioReportBuilder:
    """
    Master report builder.
    """

    # --------------------------------------------------------

    @staticmethod
    def build(
        metadata:
        PortfolioBuilderMetadata,

        forecast_result:
        ForecastResult | None,

        risk_result:
        RiskModelResult | None,

        constraint_set:
        ConstraintSet | None,

        optimization_result:
        OptimizationResult | None,

        portfolio_result:
        PortfolioAssemblyResult | None,

        rebalance_result:
        RebalanceGenerationResult | None,

        validation_result:
        PortfolioValidationResult | None,
    ) -> (InstitutionalPortfolioConstructionReport):

        if (
            portfolio_result
            is None
        ):

            raise ValueError(
                "Portfolio result required"
            )

        if (
            rebalance_result
            is None
        ):

            raise ValueError(
                "Rebalance result required"
            )

        if (
            validation_result
            is None
        ):

            raise ValueError(
                "Validation result required"
            )

        summary = (

            PortfolioConstructionSummaryBuilder
            .build(

                portfolio_result=
                portfolio_result,

                rebalance_result=
                rebalance_result,

                validation_result=
                validation_result,
            )
        )

        diagnostics = (

            PortfolioReportDiagnosticsBuilder
            .build(

                forecast_result=
                forecast_result,

                risk_result=
                risk_result,

                optimization_result=
                optimization_result,

                portfolio_result=
                portfolio_result,

                rebalance_result=
                rebalance_result,

                validation_result=
                validation_result,
            )
        )

        return (
            InstitutionalPortfolioConstructionReport(

                metadata=
                metadata,

                summary=
                summary,

                forecast_result=
                forecast_result,

                risk_result=
                risk_result,

                constraint_set=
                constraint_set,

                optimization_result=
                optimization_result,

                portfolio_result=
                portfolio_result,

                rebalance_result=
                rebalance_result,

                validation_result=
                validation_result,

                diagnostics=
                diagnostics,
            )
        )


# ============================================================
# REPORT EXPORTERS
# ============================================================


class PortfolioReportExporter:
    """
    Institutional exporters.
    """

    # --------------------------------------------------------

    @staticmethod
    def to_dict(
        report:
        InstitutionalPortfolioConstructionReport,
    ) -> dict[str, Any]:

        return asdict(
            report
        )

    # --------------------------------------------------------

    @staticmethod
    def summary_dataframe(
        report:
        InstitutionalPortfolioConstructionReport,
    ) -> pd.DataFrame:

        return pd.DataFrame(
            [
                asdict(
                    report.summary
                )
            ]
        )

    # --------------------------------------------------------

    @staticmethod
    def validation_dataframe(
        report:
        InstitutionalPortfolioConstructionReport,
    ) -> pd.DataFrame:

        validation = (
            report
            .validation_result
        )

        if (
            validation is None
        ):

            return pd.DataFrame()

        if (
            len(
                validation.issues
            )
            == 0
        ):

            return pd.DataFrame()

        return pd.DataFrame(

            [
                asdict(i)

                for i in
                validation.issues
            ]
        )

    # --------------------------------------------------------

    @staticmethod
    def rebalance_dataframe(
        report:
        InstitutionalPortfolioConstructionReport,
    ) -> pd.DataFrame:

        rebalance = (
            report
            .rebalance_result
        )

        if (
            rebalance is None
        ):

            return pd.DataFrame()

        orders = (
            rebalance
            .rebalance_plan
            .orders
        )

        return pd.DataFrame(

            [
                asdict(o)

                for o in orders
            ]
        )

    # --------------------------------------------------------

    @staticmethod
    def positions_dataframe(
        report:
        InstitutionalPortfolioConstructionReport,
    ) -> pd.DataFrame:

        portfolio = (
            report
            .portfolio_result
        )

        if (
            portfolio is None
        ):

            return pd.DataFrame()

        return pd.DataFrame(

            [
                asdict(p)

                for p in

                portfolio
                .portfolio
                .positions
            ]
        )

    # --------------------------------------------------------

    @staticmethod
    def to_json(
        report:
        InstitutionalPortfolioConstructionReport,
    ) -> str:

        return json.dumps(

            PortfolioReportExporter
            .to_dict(
                report
            ),

            default=str,

            indent=2,
        )


# ============================================================
# REPORT FACTORY
# ============================================================


class PortfolioReportingFactory:
    """
    Reporting factory.
    """

    # --------------------------------------------------------

    @staticmethod
    def builder(
    ) -> (
        InstitutionalPortfolioReportBuilder
    ):

        return (
            InstitutionalPortfolioReportBuilder()
        )

    # --------------------------------------------------------

    @staticmethod
    def exporter(
    ) -> (
        PortfolioReportExporter
    ):

        return (
            PortfolioReportExporter()
        )
    
# ============================================================
# PART 11 — MASTER PORTFOLIO BUILDER ENGINE
# ============================================================

# ============================================================
# ENGINE CONFIGURATION
# ============================================================


@dataclass(slots=True)
class PortfolioBuilderEngineConfig:
    """
    Controls engine execution.
    """

    run_forecast: bool = True

    run_risk: bool = True

    run_constraints: bool = True

    run_optimization: bool = True

    run_portfolio_assembly: bool = True

    run_rebalance: bool = True

    run_validation: bool = True

    run_reporting: bool = True


# ============================================================
# MASTER ENGINE
# ============================================================


class InstitutionalPortfolioBuilderEngine:
    """
    Institutional portfolio construction engine.

    Pipeline:

        Forecast
        Risk
        Constraints
        Optimization
        Portfolio Assembly
        Rebalance
        Validation
        Reporting
    """

    # --------------------------------------------------------

    def __init__(
        self,
        metadata:
        PortfolioBuilderMetadata,

        config:
        PortfolioBuilderConfig
        | None = None,

        engine_config:
        PortfolioBuilderEngineConfig
        | None = None,
    ) -> None:

        self.metadata = metadata

        self.config = config

        self.engine_config = (

            engine_config

            if engine_config
            is not None

            else PortfolioBuilderEngineConfig()

        )

    # ========================================================
    # FORECAST
    # ========================================================

    def run_forecast(
        self,
        inputs:
        PortfolioConstructionInput,
    ) -> (
        ForecastResult
        | None
    ):

        if (
            not self.engine_config
            .run_forecast
        ):

            return None

        try:

            engine = (
                ForecastIntegrationFactory
                .standard(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            return (
                engine
                .build_forecast(
                    inputs
                )
            )

        except Exception:

            return None

    # ========================================================
    # RISK
    # ========================================================

    def run_risk(
        self,
        inputs:
        PortfolioConstructionInput,
    ) -> (
        RiskModelResult
        | None
    ):

        if (
            not self.engine_config
            .run_risk
        ):

            return None

        try:

            engine = (
                RiskModelIntegrationFactory
                .standard(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            return (
                engine
                .build_risk_model(
                    inputs
                )
            )

        except Exception:

            return None

    # ========================================================
    # CONSTRAINTS
    # ========================================================

    def run_constraints(
        self,
        inputs:
        PortfolioConstructionInput,
    ) -> (
        ConstraintSet
        | None
    ):

        if (
            not self.engine_config
            .run_constraints
        ):

            return None

        try:

            engine = (
                ConstraintIntegrationFactory
                .standard(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            return (
                engine
                .build_constraints(
                    inputs
                )
            )

        except Exception:

            return None

    # ========================================================
    # OPTIMIZATION
    # ========================================================

    def run_optimization(
        self,

        forecast:
        ForecastResult,

        risk:
        RiskModelResult,

        constraints:
        ConstraintSet,
    ) -> (
        OptimizationResult
        | None
    ):

        if (
            not self.engine_config
            .run_optimization
        ):

            return None

        try:

            engine = (
                OptimizationIntegrationFactory
                .standard(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            return (
                engine
                .optimize(

                    forecast=
                    forecast,

                    risk=
                    risk,

                    constraints=
                    constraints,
                )
            )

        except Exception:

            return None

    # ========================================================
    # PORTFOLIO ASSEMBLY
    # ========================================================

    def run_portfolio_assembly(
        self,

        optimization:
        OptimizationResult,

        inputs:
        PortfolioConstructionInput,
    ) -> (
        PortfolioAssemblyResult
        | None
    ):

        if (
            not self.engine_config
            .run_portfolio_assembly
        ):

            return None

        try:

            engine = (
                PortfolioAssemblyFactory
                .standard(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            return (
                engine
                .assemble(

                    optimization=
                    optimization,

                    inputs=
                    inputs,
                )
            )

        except Exception:

            return None

    # ========================================================
    # REBALANCE
    # ========================================================

    def run_rebalance(
        self,

        portfolio:
        PortfolioAssemblyResult,

        inputs:
        PortfolioConstructionInput,
    ) -> (
        RebalanceGenerationResult
        | None
    ):

        if (
            not self.engine_config
            .run_rebalance
        ):

            return None

        try:

            engine = (
                RebalanceGenerationFactory
                .standard(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            return (
                engine
                .generate(

                    current_portfolio=
                    inputs.current_portfolio,

                    target_portfolio=
                    portfolio.portfolio,
                )
            )

        except Exception:

            return None

    # ========================================================
    # VALIDATION
    # ========================================================

    def run_validation(
        self,

        portfolio:
        PortfolioAssemblyResult,

        risk:
        RiskModelResult,

        constraints:
        ConstraintSet,

        rebalance:
        RebalanceGenerationResult,
    ) -> (
        PortfolioValidationResult
        | None
    ):

        if (
            not self.engine_config
            .run_validation
        ):

            return None

        try:

            engine = (
                PortfolioValidationFactory
                .standard(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            return (
                engine
                .validate(

                    portfolio=
                    portfolio.portfolio,

                    risk=
                    risk,

                    constraint_set=
                    constraints,

                    rebalance=
                    rebalance,
                )
            )

        except Exception:

            return None

    # ========================================================
    # REPORTING
    # ========================================================

    def run_reporting(
        self,

        forecast:
        ForecastResult
        | None,

        risk:
        RiskModelResult
        | None,

        constraints:
        ConstraintSet
        | None,

        optimization:
        OptimizationResult
        | None,

        portfolio:
        PortfolioAssemblyResult
        | None,

        rebalance:
        RebalanceGenerationResult
        | None,

        validation:
        PortfolioValidationResult
        | None,
    ) -> (InstitutionalPortfolioConstructionReport):

        return (

            InstitutionalPortfolioReportBuilder
            .build(

                metadata=
                self.metadata,

                forecast_result=
                forecast,

                risk_result=
                risk,

                constraint_set=
                constraints,

                optimization_result=
                optimization,

                portfolio_result=
                portfolio,

                rebalance_result=
                rebalance,

                validation_result=
                validation,
            )
        )

    # ========================================================
    # MASTER RUN
    # ========================================================

    def run(
        self,
        inputs:
        PortfolioConstructionInput,
    ) -> (InstitutionalPortfolioConstructionReport):

        forecast_result = (
            self.run_forecast(
                inputs
            )
        )

        risk_result = (
            self.run_risk(
                inputs
            )
        )

        constraint_result = (
            self.run_constraints(
                inputs
            )
        )

        if (
            forecast_result is None
            or
            risk_result is None
            or
            constraint_result is None
        ):

            raise RuntimeError(
                "Forecast/Risk/Constraint stage failed."
            )

        optimization_result = (
            self.run_optimization(

                forecast=
                forecast_result,

                risk=
                risk_result,

                constraints=
                constraint_result,
            )
        )

        if (
            optimization_result
            is None
        ):

            raise RuntimeError(
                "Optimization failed."
            )

        portfolio_result = (
            self.run_portfolio_assembly(

                optimization=
                optimization_result,

                inputs=
                inputs,
            )
        )

        if (
            portfolio_result
            is None
        ):

            raise RuntimeError(
                "Portfolio assembly failed."
            )

        rebalance_result = (
            self.run_rebalance(

                portfolio=
                portfolio_result,

                inputs=
                inputs,
            )
        )

        if (
            rebalance_result
            is None
        ):

            raise RuntimeError(
                "Rebalance generation failed."
            )

        validation_result = (
            self.run_validation(

                portfolio=
                portfolio_result,

                risk=
                risk_result,

                constraints=
                constraint_result,

                rebalance=
                rebalance_result,
            )
        )

        if (
            validation_result
            is None
        ):

            raise RuntimeError(
                "Validation failed."
            )

        report = (
            self.run_reporting(

                forecast=
                forecast_result,

                risk=
                risk_result,

                constraints=
                constraint_result,

                optimization=
                optimization_result,

                portfolio=
                portfolio_result,

                rebalance=
                rebalance_result,

                validation=
                validation_result,
            )
        )

        return report
    
# ============================================================
# PART 12 — FACTORY & CONVENIENCE APIS
# ============================================================

# ============================================================
# ENGINE FACTORY
# ============================================================


class PortfolioBuilderFactory:
    """
    Centralized portfolio builder factory.

    Creates fully configured
    institutional engines.
    """

    # --------------------------------------------------------

    @staticmethod
    def create_engine(
        metadata:
        PortfolioBuilderMetadata,

        config:
        PortfolioBuilderConfig
        | None = None,

        engine_config:
        PortfolioBuilderEngineConfig
        | None = None,
    ) -> (
        InstitutionalPortfolioBuilderEngine
    ):

        return (
            InstitutionalPortfolioBuilderEngine(

                metadata=
                metadata,

                config=
                config,

                engine_config=
                engine_config,
            )
        )

    # --------------------------------------------------------

    @staticmethod
    def default_engine(
        metadata:
        PortfolioBuilderMetadata,
    ) -> (
        InstitutionalPortfolioBuilderEngine
    ):

        return (
            InstitutionalPortfolioBuilderEngine(

                metadata=
                metadata,

                config=
                PortfolioBuilderConfig(),

                engine_config=
                PortfolioBuilderEngineConfig(),
            )
        )


# ============================================================
# PORTFOLIO CONSTRUCTION API
# ============================================================


def build_portfolio(
    *,
    metadata:
    PortfolioBuilderMetadata,

    inputs:
    PortfolioConstructionInput,

    config:
    PortfolioBuilderConfig
    | None = None,
) -> (InstitutionalPortfolioConstructionReport):
    """
    One-line portfolio build.

    Example
    -------
    report = build_portfolio(
        metadata=metadata,
        inputs=inputs,
    )
    """

    engine = (

        PortfolioBuilderFactory
        .create_engine(

            metadata=
            metadata,

            config=
            config,
        )
    )

    return engine.run(
        inputs
    )


# ============================================================
# PORTFOLIO REBUILD API
# ============================================================


def rebuild_portfolio(
    *,
    metadata:
    PortfolioBuilderMetadata,

    inputs:
    PortfolioConstructionInput,

    existing_report:
    InstitutionalPortfolioConstructionReport,

    config:
    PortfolioBuilderConfig
    | None = None,
) -> (InstitutionalPortfolioConstructionReport):
    """
    Re-run full portfolio build.

    Existing report provided
    for audit trail.
    """

    _ = existing_report

    engine = (

        PortfolioBuilderFactory
        .create_engine(

            metadata=
            metadata,

            config=
            config,
        )
    )

    return engine.run(
        inputs
    )


# ============================================================
# VALIDATION ONLY API
# ============================================================


def validate_portfolio(
    *,
    metadata:
    PortfolioBuilderMetadata,

    portfolio:
    TargetPortfolio,

    risk:
    RiskModelResult,

    constraints:
    ConstraintSet,

    rebalance:
    RebalanceGenerationResult,

    config:
    PortfolioBuilderConfig
    | None = None,
) -> (
    PortfolioValidationResult
):
    """
    Run validation only.
    """

    validator = (

        PortfolioValidationFactory
        .standard(

            metadata=
            metadata,

            config=
            config,
        )
    )

    return validator.validate(

        portfolio=
        portfolio,

        risk=
        risk,

        constraint_set=
        constraints,

        rebalance=
        rebalance,
    )


# ============================================================
# REPORT EXPORT API
# ============================================================


def report_to_dict(
    report:
    InstitutionalPortfolioConstructionReport,
) -> dict[str, Any]:

    return (
        PortfolioReportExporter
        .to_dict(
            report
        )
    )


# ------------------------------------------------------------


def report_to_json(
    report:
    InstitutionalPortfolioConstructionReport,
) -> str:

    return (
        PortfolioReportExporter
        .to_json(
            report
        )
    )


# ------------------------------------------------------------


def report_summary_dataframe(
    report:
    InstitutionalPortfolioConstructionReport,
) -> pd.DataFrame:

    return (

        PortfolioReportExporter
        .summary_dataframe(
            report
        )
    )


# ------------------------------------------------------------


def report_positions_dataframe(
    report:
    InstitutionalPortfolioConstructionReport,
) -> pd.DataFrame:

    return (

        PortfolioReportExporter
        .positions_dataframe(
            report
        )
    )


# ------------------------------------------------------------


def report_rebalance_dataframe(
    report:
    InstitutionalPortfolioConstructionReport,
) -> pd.DataFrame:

    return (

        PortfolioReportExporter
        .rebalance_dataframe(
            report
        )
    )


# ------------------------------------------------------------


def report_validation_dataframe(
    report:
    InstitutionalPortfolioConstructionReport,
) -> pd.DataFrame:

    return (

        PortfolioReportExporter
        .validation_dataframe(
            report
        )
    )


# ============================================================
# QUICK BUILD API
# ============================================================


def quick_build(
    metadata:
    PortfolioBuilderMetadata,

    inputs:
    PortfolioConstructionInput,
) -> (
    InstitutionalPortfolioConstructionReport
):
    """
    Simplest entry point.

    Uses all defaults.
    """

    return build_portfolio(

        metadata=
        metadata,

        inputs=
        inputs,
    )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [

    # Factory
    "PortfolioBuilderFactory",

    # Main Engine
    "InstitutionalPortfolioBuilderEngine",

    # APIs
    "build_portfolio",
    "rebuild_portfolio",
    "validate_portfolio",

    # Report Helpers
    "report_to_dict",
    "report_to_json",
    "report_summary_dataframe",
    "report_positions_dataframe",
    "report_rebalance_dataframe",
    "report_validation_dataframe",

    # Quick API
    "quick_build",
]