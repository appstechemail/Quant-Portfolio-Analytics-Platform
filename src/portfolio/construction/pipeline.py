# ============================================================
# PIPELINE.PY
# PART 1 — FRAMEWORK & CORE OBJECTS
# ============================================================

from __future__ import annotations

# ============================================================
# STANDARD LIBRARY
# ============================================================

from dataclasses import (
    dataclass,
    field,
)

from .portfolio_builder import PortfolioBuilderMetadata, InstitutionalPortfolioConstructionReport

from src.portfolio.construction.analytics import (
    AnalyticsEngine,
    AnalyticsMetadata,
)

from enum import (
    Enum,
    auto,
)

from datetime import (
    datetime,timezone
)

import time

from typing import (
    Any,
    Optional,
)

import uuid

# ============================================================
# THIRD PARTY
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ============================================================
# PIPELINE VERSION
# ============================================================

PIPELINE_VERSION = "1.0.0"

# ============================================================
# PIPELINE STAGES
# ============================================================


class PipelineStage(
    Enum,
):
    """
    Institutional pipeline stages.
    """

    FORECAST = auto()

    RISK = auto()

    CONSTRAINTS = auto()

    OPTIMIZATION = auto()

    PORTFOLIO_BUILD = auto()

    REBALANCE = auto()

    EXECUTION = auto()

    DIAGNOSTICS = auto()

    ATTRIBUTION = auto()

    STRESS_TESTING = auto()

    REPORTING = auto()


# ============================================================
# PIPELINE STATUS
# ============================================================


class PipelineStatus(
    Enum,
):
    """
    Pipeline execution state.
    """

    NOT_STARTED = auto()

    RUNNING = auto()

    COMPLETED = auto()

    FAILED = auto()

    SKIPPED = auto()


# ============================================================
# AUDIT EVENT
# ============================================================


@dataclass(slots=True)
class PipelineAuditEvent:
    """
    Immutable audit trail record.
    """

    timestamp: datetime

    stage: str

    event: str

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# STAGE EXECUTION METADATA
# ============================================================


@dataclass(slots=True)
class StageExecutionResult:
    """
    Metadata for a single stage run.
    """

    stage: PipelineStage

    status: PipelineStatus

    start_time: datetime

    end_time: datetime | None = None

    runtime_seconds: float = 0.0

    error_message: str | None = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# PIPELINE METADATA
# ============================================================


@dataclass(slots=True)
class PipelineMetadata:
    """
    Global metadata for an institutional run.
    """

    run_id: str

    strategy_name: str

    universe_name: str

    benchmark_name: str | None = None

    created_at: datetime = field(
        default_factory=datetime.utcnow
    )

    version: str = PIPELINE_VERSION

    owner: str | None = None

    tags: list[str] = field(
        default_factory=list
    )


# ============================================================
# PIPELINE CONFIGURATION
# ============================================================


@dataclass(slots=True)
class PipelineConfig:
    """
    High-level pipeline configuration.
    """

    run_forecast: bool = True

    run_risk: bool = True

    run_constraints: bool = True

    run_optimization: bool = True

    run_portfolio_build: bool = True

    run_rebalance: bool = True

    run_execution: bool = True

    run_analytics: bool = True

    run_diagnostics: bool = True

    run_attribution: bool = True

    run_stress_testing: bool = True

    run_reporting: bool = True

    fail_fast: bool = True

    persist_audit_trail: bool = True


# ============================================================
# PIPELINE CONTEXT
# ============================================================


@dataclass(slots=True)
class PipelineContext:
    """
    Shared runtime state.

    Used internally by the pipeline engine
    to pass objects between stages.
    """

    metadata: PipelineMetadata

    config: PipelineConfig

    stage_results: dict[
        PipelineStage,
        StageExecutionResult
    ] = field(
        default_factory=dict
    )

    audit_log: list[
        PipelineAuditEvent
    ] = field(
        default_factory=list
    )

    shared_objects: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# HELPERS
# ============================================================


def generate_pipeline_run_id() -> str:
    """
    Generates institutional run ID.
    """

    return (
        f"PIPE-"
        f"{uuid.uuid4().hex[:12]}"
        .upper()
    )


# ============================================================
# FACTORY
# ============================================================


class PipelineFrameworkFactory:
    """
    Framework factory helpers.
    """

    @staticmethod
    def create_metadata(
        *,
        strategy_name: str,
        universe_name: str,
        benchmark_name: str | None = None,
        owner: str | None = None,
    ) -> PipelineMetadata:

        return PipelineMetadata(

            run_id=
            generate_pipeline_run_id(),

            strategy_name=
            strategy_name,

            universe_name=
            universe_name,

            benchmark_name=
            benchmark_name,

            owner=
            owner,
        )

    # --------------------------------------------------------

    @staticmethod
    def create_context(
        *,
        metadata:
        PipelineMetadata,

        config:
        PipelineConfig | None = None,
    ) -> PipelineContext:

        return PipelineContext(

            metadata=
            metadata,

            config=
            config
            if config is not None
            else PipelineConfig(),
        )
    
# ============================================================
# PART 2 — PIPELINE INPUTS
# ============================================================

# ============================================================
# MARKET DATA INPUT
# ============================================================


@dataclass(slots=True)
class MarketDataInput:
    """
    Market data supplied to the pipeline.
    """

    prices: pd.DataFrame

    returns: pd.DataFrame | None = None

    benchmark_prices: pd.DataFrame | None = None

    benchmark_returns: pd.DataFrame | None = None

    volumes: pd.DataFrame | None = None

    market_caps: pd.DataFrame | None = None


# ============================================================
# FORECAST INPUT
# ============================================================


@dataclass(slots=True)
class ForecastInput:
    """
    Forecast-related inputs.

    Used when forecasts are produced
    outside the pipeline.
    """

    alpha_scores: pd.DataFrame | None = None

    expected_returns: pd.DataFrame | None = None

    forecast_confidence: pd.DataFrame | None = None


# ============================================================
# FACTOR INPUT
# ============================================================


@dataclass(slots=True)
class FactorInput:
    """
    Factor model inputs.
    """

    factor_exposures: pd.DataFrame | None = None

    factor_returns: pd.DataFrame | None = None

    factor_covariance: pd.DataFrame | None = None


# ============================================================
# PORTFOLIO INPUT
# ============================================================


@dataclass(slots=True)
class PortfolioInput:
    """
    Current portfolio state.
    """

    current_weights: pd.Series | None = None

    current_holdings: pd.DataFrame | None = None

    current_positions: pd.DataFrame | None = None

    cash_weight: float = 0.0


# ============================================================
# LIQUIDITY INPUT
# ============================================================


@dataclass(slots=True)
class LiquidityInput:
    """
    Liquidity and execution data.
    """

    average_daily_volume: pd.Series | None = None

    bid_ask_spread: pd.Series | None = None

    participation_limit: float | None = None

    liquidity_profile: pd.DataFrame | None = None


# ============================================================
# CONSTRAINT INPUT
# ============================================================


@dataclass(slots=True)
class ConstraintInput:
    """
    Constraint data supplied externally.
    """

    sector_map: pd.Series | None = None

    industry_map: pd.Series | None = None

    country_map: pd.Series | None = None

    custom_constraints: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# PIPELINE INPUT
# ============================================================


@dataclass(slots=True)
class PipelineInput:
    """
    Master pipeline input contract.

    Every stage must read from this object.

    This object should remain stable
    across the platform.
    """

    market_data: MarketDataInput

    forecast_data: ForecastInput = field(
        default_factory=ForecastInput
    )

    factor_data: FactorInput = field(
        default_factory=FactorInput
    )

    portfolio_data: PortfolioInput = field(
        default_factory=PortfolioInput
    )

    liquidity_data: LiquidityInput = field(
        default_factory=LiquidityInput
    )

    constraint_data: ConstraintInput = field(
        default_factory=ConstraintInput
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# INPUT VALIDATION
# ============================================================


class PipelineInputValidator:
    """
    Validates pipeline inputs.
    """

    # --------------------------------------------------------

    @staticmethod
    def validate_market_data(
        market_data:
        MarketDataInput,
    ) -> None:

        if (
            market_data.prices
            is None
        ):

            raise ValueError(
                "prices cannot be None."
            )

        if (
            market_data.prices.empty
        ):

            raise ValueError(
                "prices are empty."
            )

    # --------------------------------------------------------

    @staticmethod
    def validate(
        inputs:
        PipelineInput,
    ) -> None:

        PipelineInputValidator\
            .validate_market_data(
                inputs.market_data
            )


# ============================================================
# INPUT FACTORY
# ============================================================


class PipelineInputFactory:
    """
    Convenience constructors.
    """

    # --------------------------------------------------------

    @staticmethod
    def from_prices(
        prices:
        pd.DataFrame,
    ) -> PipelineInput:

        market_data = (
            MarketDataInput(
                prices=prices
            )
        )

        return PipelineInput(
            market_data=
            market_data
        )

    # --------------------------------------------------------

    @staticmethod
    def from_prices_returns(
        *,
        prices:
        pd.DataFrame,

        returns:
        pd.DataFrame,
    ) -> PipelineInput:

        market_data = (
            MarketDataInput(

                prices=
                prices,

                returns=
                returns,
            )
        )

        return PipelineInput(
            market_data=
            market_data
        )

    # --------------------------------------------------------

    @staticmethod
    def full_input(
        *,
        market_data:
        MarketDataInput,

        forecast_data:
        ForecastInput | None = None,

        factor_data:
        FactorInput | None = None,

        portfolio_data:
        PortfolioInput | None = None,

        liquidity_data:
        LiquidityInput | None = None,

        constraint_data:
        ConstraintInput | None = None,
    ) -> PipelineInput:

        return PipelineInput(

            market_data=
            market_data,

            forecast_data=
            forecast_data
            if forecast_data
            is not None
            else ForecastInput(),

            factor_data=
            factor_data
            if factor_data
            is not None
            else FactorInput(),

            portfolio_data=
            portfolio_data
            if portfolio_data
            is not None
            else PortfolioInput(),

            liquidity_data=
            liquidity_data
            if liquidity_data
            is not None
            else LiquidityInput(),

            constraint_data=
            constraint_data
            if constraint_data
            is not None
            else ConstraintInput(),
        )
    


# ============================================================
# PART 3 — PIPELINE OUTPUTS
# ============================================================

# ============================================================
# STAGE OUTPUT WRAPPER
# ============================================================


@dataclass(slots=True)
class PipelineStageOutput:
    stage: PipelineStage
    status: PipelineStatus
    payload: Any = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    error_message: str | None = None


# ============================================================
# FORECAST OUTPUT
# ============================================================


@dataclass(slots=True)
class ForecastStageOutput:
    """
    Forecast stage output.
    """

    result: Any = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# RISK OUTPUT
# ============================================================


@dataclass(slots=True)
class RiskStageOutput:
    """
    Risk stage output.
    """

    result: Any = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# CONSTRAINT OUTPUT
# ============================================================


@dataclass(slots=True)
class ConstraintStageOutput:
    """
    Constraint stage output.
    """

    result: Any = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# OPTIMIZATION OUTPUT
# ============================================================


@dataclass(slots=True)
class OptimizationStageOutput:
    """
    Optimization stage output.
    """

    result: Any = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# PORTFOLIO BUILD OUTPUT
# ============================================================


@dataclass(slots=True)
class PortfolioBuildStageOutput:
    """
    Portfolio construction output.
    """

    result: Any = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# REBALANCE OUTPUT
# ============================================================


@dataclass(slots=True)
class RebalanceStageOutput:
    """
    Rebalance output.
    """

    result: Any = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# EXECUTION OUTPUT
# ============================================================


@dataclass(slots=True)
class ExecutionStageOutput:
    """
    Execution output.
    """

    result: Any = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# DIAGNOSTICS OUTPUT
# ============================================================


@dataclass(slots=True)
class DiagnosticsStageOutput:
    """
    Portfolio diagnostics output.
    """
    result: Any = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ============================================================
# ATTRIBUTION OUTPUT
# ============================================================


@dataclass(slots=True)
class AttributionStageOutput:
    """
    Attribution output.
    """

    report: Any = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# STRESS TEST OUTPUT
# ============================================================


@dataclass(slots=True)
class StressTestingStageOutput:
    """
    Stress testing output.
    """

    report: Any = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# REPORTING OUTPUT
# ============================================================


@dataclass(slots=True)
class ReportingStageOutput:
    """
    Reporting stage output.
    """

    report: Any = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# MASTER PIPELINE OUTPUT
# ============================================================


@dataclass(slots=True)
class PipelineOutput:
    """
    Institutional pipeline output.

    Single source of truth for all
    pipeline results.
    """

    metadata: PipelineMetadata

    # ---------------------------------
    # Stage Outputs
    # ---------------------------------

    forecast: ForecastStageOutput | None = None

    risk: RiskStageOutput | None = None

    constraints: ConstraintStageOutput | None = None

    optimization: OptimizationStageOutput | None = None

    portfolio_build: PortfolioBuildStageOutput | None = None

    rebalance: RebalanceStageOutput | None = None

    execution: ExecutionStageOutput | None = None

    diagnostics: DiagnosticsStageOutput | None = None

    attribution: AttributionStageOutput | None = None

    stress_testing: StressTestingStageOutput | None = None

    reporting: ReportingStageOutput | None = None

    # ---------------------------------
    # Pipeline Audit
    # ---------------------------------

    stage_results: dict[
        PipelineStage,
        StageExecutionResult
    ] = field(
        default_factory=dict
    )

    audit_log: list[
        PipelineAuditEvent
    ] = field(
        default_factory=list
    )

    runtime_seconds: float = 0.0

    success: bool = True

    error_message: str | None = None


# ============================================================
# OUTPUT FACTORY
# ============================================================


class PipelineOutputFactory:
    """
    Output helpers.
    """

    # --------------------------------------------------------

    @staticmethod
    def create_empty(
        metadata:
        PipelineMetadata,
    ) -> PipelineOutput:

        return PipelineOutput(
            metadata=
            metadata
        )

    # --------------------------------------------------------

    @staticmethod
    def create_failure(
        *,
        metadata:
        PipelineMetadata,

        error_message:
        str,
    ) -> PipelineOutput:

        return PipelineOutput(

            metadata=
            metadata,

            success=
            False,

            error_message=
            error_message,
        )

    # --------------------------------------------------------

    @staticmethod
    def create_success(
        *,
        metadata:
        PipelineMetadata,
    ) -> PipelineOutput:

        return PipelineOutput(

            metadata=
            metadata,

            success=
            True,
        )
    

# ============================================================
# PART 4 — FORECAST STAGE
# ============================================================

from abc import (
    ABC,
    abstractmethod,
)

import time


# ============================================================
# FORECAST ENGINE CONTRACT
# ============================================================


class BaseForecastEngine(ABC):
    """
    Institutional forecast contract.

    Any forecasting engine must
    implement this interface.
    """

    @abstractmethod
    def run(
        self,
        inputs:
        PipelineInput,
    ) -> Any:
        """
        Generate forecasts.
        """
        raise NotImplementedError


# ============================================================
# PASS-THROUGH FORECAST ENGINE
# ============================================================


class PassThroughForecastEngine(
    BaseForecastEngine,
):
    """
    Uses forecasts already supplied
    inside PipelineInput.
    """

    def run(
        self,
        inputs:
        PipelineInput,
    ) -> Any:

        if (
            inputs.forecast_data
            .expected_returns
            is not None
        ):
            return (
                inputs.forecast_data
                .expected_returns
            )

        if (
            inputs.forecast_data
            .alpha_scores
            is not None
        ):
            return (
                inputs.forecast_data
                .alpha_scores
            )

        return None


# ============================================================
# FORECAST STAGE CONFIG
# ============================================================


@dataclass(slots=True)
class ForecastStageConfig:
    """
    Forecast stage settings.
    """

    enabled: bool = True

    allow_passthrough: bool = True

    fail_on_missing_forecast: bool = False


# ============================================================
# FORECAST STAGE
# ============================================================


class ForecastStage:
    """
    Institutional Forecast Stage.

    Responsible for:

        Forecast generation
        Forecast validation
        Forecast diagnostics
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        engine:
        BaseForecastEngine | None = None,

        config:
        ForecastStageConfig
        | None = None,
    ) -> None:

        self.engine = engine

        self.config = (
            config
            if config is not None
            else ForecastStageConfig()
        )

    # --------------------------------------------------------

    def validate_forecast(
        self,
        forecast:
        Any,
    ) -> dict[str, Any]:

        diagnostics = {}

        if forecast is None:

            diagnostics[
                "forecast_available"
            ] = False

            return diagnostics

        diagnostics[
            "forecast_available"
        ] = True

        if isinstance(
            forecast,
            pd.DataFrame,
        ):

            diagnostics[
                "rows"
            ] = len(
                forecast
            )

            diagnostics[
                "columns"
            ] = list(
                forecast.columns
            )

        elif isinstance(
            forecast,
            pd.Series,
        ):

            diagnostics[
                "count"
            ] = len(
                forecast
            )

        return diagnostics

    # --------------------------------------------------------

    def run(
        self,
        inputs:
        PipelineInput,
    ) -> ForecastStageOutput:

        start = time.perf_counter()

        try:

            # ----------------------------------
            # Stage Disabled
            # ----------------------------------

            if (
                not self.config
                .enabled
            ):

                return (
                    ForecastStageOutput(

                        result=None,

                        diagnostics={
                            "status":
                            "disabled"
                        },
                    )
                )

            # ----------------------------------
            # Engine
            # ----------------------------------

            forecast = None

            if (
                self.engine
                is not None
            ):

                forecast = (
                    self.engine.run(
                        inputs
                    )
                )

            # ----------------------------------
            # Pass-through mode
            # ----------------------------------

            elif (
                self.config
                .allow_passthrough
            ):

                forecast = (

                    PassThroughForecastEngine()
                    .run(inputs)

                )

            # ----------------------------------
            # Missing Forecast
            # ----------------------------------

            if (
                forecast is None
                and
                self.config
                .fail_on_missing_forecast
            ):

                raise ValueError(
                    "No forecast data available."
                )

            diagnostics = (
                self.validate_forecast(
                    forecast
                )
            )

            diagnostics[
                "runtime_seconds"
            ] = (
                time.perf_counter()
                - start
            )

            return (
                ForecastStageOutput(

                    result=
                    forecast,

                    diagnostics=
                    diagnostics,
                )
            )

        except Exception as exc:

            return (
                ForecastStageOutput(

                    result=None,

                    diagnostics={

                        "error":
                        str(exc),

                        "runtime_seconds":
                        (
                            time.perf_counter()
                            - start
                        ),
                    },
                )
            )


# ============================================================
# FORECAST STAGE FACTORY
# ============================================================


class ForecastStageFactory:
    """
        Forecast stage constructors.
    """
    
    # --------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        metadata=None,
        config=None,
        pipeline_input=None,
        engine=None,
    ) -> ForecastStage:

        # Explicit engine supplied
        if engine is not None:
            return cls.with_engine(engine)

        # Default behaviour
        return cls.passthrough()

    @staticmethod
    def passthrough() -> ForecastStage:

        return ForecastStage(
            engine=None,
            config=ForecastStageConfig(
                enabled=True,
                allow_passthrough=True,
            ),
        )

    @staticmethod
    def with_engine(
        engine: BaseForecastEngine,
    ) -> ForecastStage:

        return ForecastStage(
            engine=engine
        )

# ============================================================
# CONTEXT INTEGRATION
# ============================================================


def run_forecast_stage(
    *,
    context:
    PipelineContext,

    inputs:
    PipelineInput,

    stage:
    ForecastStage,
) -> ForecastStageOutput:
    """
    Executes forecast stage and stores
    output into PipelineContext.
    """

    output = (
        stage.run(
            inputs
        )
    )

    context.shared_objects[
        "forecast_output"
    ] = output

    return output

# ============================================================
# PART 5 — RISK STAGE
# ============================================================

from abc import (
    ABC,
    abstractmethod,
)

import time


# ============================================================
# RISK ENGINE CONTRACT
# ============================================================


class BaseRiskEngine(ABC):
    """
    Institutional risk model contract.
    """

    @abstractmethod
    def run(
        self,
        inputs:
        PipelineInput,
    ) -> Any:
        """
        Generate risk model output.
        """
        raise NotImplementedError


# ============================================================
# PASS-THROUGH RISK ENGINE
# ============================================================


class PassThroughRiskEngine(
    BaseRiskEngine,
):
    """
    Uses externally supplied
    factor covariance information.
    """

    def run(
        self,
        inputs:
        PipelineInput,
    ) -> dict[str, Any]:

        return {

            "factor_covariance":
            inputs.factor_data
            .factor_covariance,

            "factor_exposures":
            inputs.factor_data
            .factor_exposures,
        }


# ============================================================
# RISK STAGE CONFIG
# ============================================================


@dataclass(slots=True)
class RiskStageConfig:
    """
    Risk stage settings.
    """

    enabled: bool = True

    allow_passthrough: bool = True

    fail_on_missing_risk: bool = False

    compute_basic_stats: bool = True


# ============================================================
# RISK DIAGNOSTICS
# ============================================================


@dataclass(slots=True)
class RiskDiagnostics:
    """
    Risk diagnostics summary.
    """

    annualized_volatility: float = 0.0

    average_correlation: float = 0.0

    covariance_available: bool = False

    exposures_available: bool = False

    observations: int = 0


# ============================================================
# RISK STAGE
# ============================================================


class RiskStage:
    """
    Institutional Risk Stage.
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        engine:
        BaseRiskEngine | None = None,

        config:
        RiskStageConfig
        | None = None,
    ) -> None:

        self.engine = engine

        self.config = (
            config
            if config is not None
            else RiskStageConfig()
        )

    # --------------------------------------------------------

    @staticmethod
    def compute_basic_risk_metrics(
        returns:
        pd.DataFrame | None,
    ) -> RiskDiagnostics:

        diagnostics = (
            RiskDiagnostics()
        )

        if (
            returns is None
        ):
            return diagnostics

        if (
            returns.empty
        ):
            return diagnostics

        diagnostics.observations = (
            len(returns)
        )

        # ----------------------------------
        # Annualized Volatility
        # ----------------------------------

        try:

            portfolio_vol = float(

                returns.mean(
                    axis=1
                )
                .std()

                *
                np.sqrt(252)

            )

            diagnostics\
                .annualized_volatility = (
                    portfolio_vol
                )

        except Exception:

            pass

        # ----------------------------------
        # Average Correlation
        # ----------------------------------

        try:

            corr = (
                returns.corr()
            )

            if (
                corr.shape[0]
                > 1
            ):

                mask = np.triu(
                    np.ones(
                        corr.shape
                    ),
                    k=1,
                ).astype(bool)

                values = (
                    corr.where(mask)
                    .stack()
                )

                diagnostics\
                    .average_correlation = (
                        float(
                            values.mean()
                        )
                    )

        except Exception:

            pass

        return diagnostics

    # --------------------------------------------------------

    def validate_risk_result(
        self,
        risk_result:
        Any,
        inputs:
        PipelineInput,
    ) -> dict[str, Any]:

        diagnostics = {}

        basic = (
            self.compute_basic_risk_metrics(

                inputs.market_data
                .returns

            )
        )

        diagnostics.update(

            {
                "annualized_volatility":
                basic
                .annualized_volatility,

                "average_correlation":
                basic
                .average_correlation,

                "observations":
                basic
                .observations,
            }

        )

        factor_cov = (
            inputs.factor_data
            .factor_covariance
        )

        factor_exp = (
            inputs.factor_data
            .factor_exposures
        )

        diagnostics[
            "factor_covariance_available"
        ] = (
            factor_cov
            is not None
        )

        diagnostics[
            "factor_exposures_available"
        ] = (
            factor_exp
            is not None
        )

        return diagnostics

    # --------------------------------------------------------

    def run(
        self,
        inputs:
        PipelineInput,
    ) -> RiskStageOutput:

        start = time.perf_counter()

        try:

            # ----------------------------------
            # Disabled
            # ----------------------------------

            if (
                not self.config
                .enabled
            ):

                return (
                    RiskStageOutput(

                        result=None,

                        diagnostics={
                            "status":
                            "disabled"
                        },
                    )
                )

            # ----------------------------------
            # Run Risk Engine
            # ----------------------------------

            risk_result = None

            if (
                self.engine
                is not None
            ):

                risk_result = (
                    self.engine.run(
                        inputs
                    )
                )

            elif (
                self.config
                .allow_passthrough
            ):

                risk_result = (

                    PassThroughRiskEngine()
                    .run(inputs)

                )

            # ----------------------------------
            # Validation
            # ----------------------------------

            if (
                risk_result is None
                and
                self.config
                .fail_on_missing_risk
            ):

                raise ValueError(
                    "Risk model unavailable."
                )

            diagnostics = (
                self.validate_risk_result(

                    risk_result,

                    inputs,
                )
            )

            diagnostics[
                "runtime_seconds"
            ] = (
                time.perf_counter()
                - start
            )

            return (
                RiskStageOutput(

                    result=
                    risk_result,

                    diagnostics=
                    diagnostics,
                )
            )

        except Exception as exc:

            return (
                RiskStageOutput(

                    result=None,

                    diagnostics={

                        "error":
                        str(exc),

                        "runtime_seconds":
                        (
                            time.perf_counter()
                            - start
                        ),
                    },
                )
            )


# ============================================================
# RISK STAGE FACTORY
# ============================================================


class RiskStageFactory:
    """
    Risk stage constructors.
    """

    @staticmethod
    def passthrough() -> RiskStage:

        return RiskStage(
            engine=None,
            config=RiskStageConfig(
                enabled=True,
                allow_passthrough=True,
            ),
        )

    @staticmethod
    def with_engine(
        engine: BaseRiskEngine,
    ) -> RiskStage:

        return RiskStage(
            engine=engine
        )

    @staticmethod
    def create(
        *,
        metadata: PipelineMetadata,
        config: PipelineConfig,
        pipeline_input: PipelineInput,
    ) -> RiskStage:
        """
        Institutional factory entrypoint.
        """

        # Future:
        #
        # if config.risk.engine == "factor":
        #     return RiskStageFactory.with_engine(
        #         FactorRiskEngine(...)
        #     )
        #
        # if config.risk.engine == "barra":
        #     return RiskStageFactory.with_engine(
        #         BarraRiskEngine(...)
        #     )

        # Reserved for future use
        _ = metadata
        _ = config
        _ = pipeline_input

        return RiskStageFactory.passthrough()

# ============================================================
# CONTEXT INTEGRATION
# ============================================================

def run_risk_stage(
    *,
    context: PipelineContext,
    inputs: PipelineInput,
    stage: RiskStage,
    forecast_output: ForecastStageOutput | None,
) -> RiskStageOutput:

    _ = forecast_output

    output = stage.run(inputs)

    context.shared_objects["risk_output"] = output

    return output


# ============================================================
# PART 6 — CONSTRAINT STAGE
# ============================================================

from abc import (
    ABC,
    abstractmethod,
)

import time


# ============================================================
# CONSTRAINT ENGINE CONTRACT
# ============================================================


class BaseConstraintEngine(
    ABC,
):
    """
    Institutional constraint engine
    contract.
    """

    @abstractmethod
    def run(
        self,
        inputs:
        PipelineInput,
    ) -> Any:
        """
        Build institutional constraint set.
        """
        raise NotImplementedError


# ============================================================
# CONSTRAINT PACKAGE
# ============================================================


@dataclass(slots=True)
class ConstraintPackage:
    """
    Canonical optimizer-ready
    constraint representation.
    """

    sector_map: pd.Series | None = None

    industry_map: pd.Series | None = None

    country_map: pd.Series | None = None

    custom_constraints: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# PASS THROUGH CONSTRAINT ENGINE
# ============================================================


class PassThroughConstraintEngine(
    BaseConstraintEngine,
):
    """
    Uses constraint information
    already supplied in
    PipelineInput.
    """

    def run(
        self,
        inputs:
        PipelineInput,
    ) -> ConstraintPackage:

        return ConstraintPackage(

            sector_map=
            inputs.constraint_data
            .sector_map,

            industry_map=
            inputs.constraint_data
            .industry_map,

            country_map=
            inputs.constraint_data
            .country_map,

            custom_constraints=
            inputs.constraint_data
            .custom_constraints,
        )


# ============================================================
# CONSTRAINT STAGE CONFIG
# ============================================================


@dataclass(slots=True)
class ConstraintStageConfig:
    """
    Constraint stage settings.
    """

    enabled: bool = True

    allow_passthrough: bool = True

    fail_on_missing_constraints: bool = False

    validate_classifications: bool = True


# ============================================================
# CONSTRAINT DIAGNOSTICS
# ============================================================


@dataclass(slots=True)
class ConstraintDiagnostics:
    """
    Constraint diagnostics.
    """

    sectors_available: bool = False

    industries_available: bool = False

    countries_available: bool = False

    custom_constraint_count: int = 0


# ============================================================
# CONSTRAINT STAGE
# ============================================================


class ConstraintStage:
    """
    Institutional Constraint Stage.
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        engine:
        BaseConstraintEngine
        | None = None,

        config:
        ConstraintStageConfig
        | None = None,
    ) -> None:

        self.engine = engine

        self.config = (
            config
            if config is not None
            else ConstraintStageConfig()
        )

    # --------------------------------------------------------

    @staticmethod
    def validate_constraint_package(
        package:
        ConstraintPackage,
    ) -> dict[str, Any]:

        diagnostics = {}

        diagnostics[
            "sector_map_available"
        ] = (
            package.sector_map
            is not None
        )

        diagnostics[
            "industry_map_available"
        ] = (
            package.industry_map
            is not None
        )

        diagnostics[
            "country_map_available"
        ] = (
            package.country_map
            is not None
        )

        diagnostics[
            "custom_constraint_count"
        ] = len(
            package.custom_constraints
        )

        return diagnostics

    # --------------------------------------------------------

    def run(
        self,
        inputs:
        PipelineInput,
    ) -> ConstraintStageOutput:

        start = time.perf_counter()

        try:

            # ----------------------------------
            # Disabled
            # ----------------------------------

            if (
                not self.config
                .enabled
            ):

                return (
                    ConstraintStageOutput(

                        result=None,

                        diagnostics={
                            "status":
                            "disabled"
                        },
                    )
                )

            # ----------------------------------
            # Engine
            # ----------------------------------

            package = None

            if (
                self.engine
                is not None
            ):

                package = (
                    self.engine.run(
                        inputs
                    )
                )

            elif (
                self.config
                .allow_passthrough
            ):

                package = (

                    PassThroughConstraintEngine()
                    .run(inputs)

                )

            # ----------------------------------
            # Validation
            # ----------------------------------

            if (
                package is None
                and
                self.config
                .fail_on_missing_constraints
            ):

                raise ValueError(
                    "Constraint package unavailable."
                )

            diagnostics = {}

            if (
                package is not None
            ):

                diagnostics = (

                    self
                    .validate_constraint_package(
                        package
                    )

                )

            diagnostics[
                "runtime_seconds"
            ] = (
                time.perf_counter()
                - start
            )

            return (
                ConstraintStageOutput(

                    result=
                    package,

                    diagnostics=
                    diagnostics,
                )
            )

        except Exception as exc:

            return (
                ConstraintStageOutput(

                    result=None,

                    diagnostics={

                        "error":
                        str(exc),

                        "runtime_seconds":
                        (
                            time.perf_counter()
                            - start
                        ),
                    },
                )
            )


# ============================================================
# CONSTRAINT STAGE FACTORY
# ============================================================

class ConstraintStageFactory:
    """
    Constraint stage constructors.
    """

    # --------------------------------------------------------

    @staticmethod
    def passthrough() -> ConstraintStage:

        return ConstraintStage(
            engine=None,
            config=ConstraintStageConfig(
                enabled=True,
                allow_passthrough=True,
            ),
        )

    # --------------------------------------------------------

    @staticmethod
    def with_engine(
        engine: BaseConstraintEngine,
    ) -> ConstraintStage:

        return ConstraintStage(
            engine=engine
        )

    # --------------------------------------------------------

    @staticmethod
    def create(
        *,
        metadata: PipelineMetadata,
        config: PipelineConfig,
        pipeline_input: PipelineInput,
    ) -> ConstraintStage:
        """
        Institutional factory entrypoint.

        Future examples:
            - ExposureConstraintEngine
            - SectorConstraintEngine
            - LiquidityConstraintEngine
            - TurnoverConstraintEngine
            - RegulatoryConstraintEngine
            - ConcentrationConstraintEngine
        """

        # Future implementation:
        #
        # if config.constraints.engine == "sector":
        #     return ConstraintStageFactory.with_engine(
        #         SectorConstraintEngine(...)
        #     )
        #
        # if config.constraints.engine == "liquidity":
        #     return ConstraintStageFactory.with_engine(
        #         LiquidityConstraintEngine(...)
        #     )

        # Reserved for future use
        _ = metadata
        _ = config
        _ = pipeline_input

        return ConstraintStageFactory.passthrough()
# ============================================================
# CONTEXT INTEGRATION
# ============================================================

def run_constraint_stage(
    *,
    context: PipelineContext,
    inputs: PipelineInput,
    stage: ConstraintStage,
    forecast_output: ForecastStageOutput | None,
    risk_output: RiskStageOutput | None,
) -> ConstraintStageOutput:

    _ = forecast_output
    _ = risk_output

    output = stage.run(inputs)

    context.shared_objects["constraint_output"] = output

    return output


# ============================================================
# PART 7 — OPTIMIZATION STAGE
# ============================================================

from abc import (
    ABC,
    abstractmethod,
)

import time


# ============================================================
# OPTIMIZER CONTRACT
# ============================================================


class BaseOptimizerEngine(
    ABC,
):
    """
    Institutional optimizer contract.
    """

    @abstractmethod
    def run(
        self,
        *,
        inputs:
        PipelineInput,

        forecast_output:
        ForecastStageOutput
        | None,

        risk_output:
        RiskStageOutput
        | None,

        constraint_output:
        ConstraintStageOutput
        | None,
    ) -> Any:
        """
        Produce optimized portfolio.
        """
        raise NotImplementedError


# ============================================================
# TARGET PORTFOLIO
# ============================================================


@dataclass(slots=True)
class TargetPortfolio:
    """
    Optimizer output.

    Canonical representation used
    throughout the platform.
    """

    weights: pd.Series

    expected_return: float = 0.0

    expected_volatility: float = 0.0

    expected_sharpe: float = 0.0

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# FALLBACK OPTIMIZER
# ============================================================


class EqualWeightOptimizer(
    BaseOptimizerEngine,
):
    """
    Institutional-safe fallback.

    Generates equal weights.
    """

    def run(
        self,
        *,
        inputs:
        PipelineInput,

        forecast_output:
        ForecastStageOutput
        | None,

        risk_output:
        RiskStageOutput
        | None,

        constraint_output:
        ConstraintStageOutput
        | None,
    ) -> TargetPortfolio:

        prices = (
            inputs.market_data
            .prices
        )

        assets = (
            list(
                prices.columns
            )
        )

        n_assets = len(
            assets
        )

        if n_assets == 0:

            raise ValueError(
                "No assets available."
            )

        weights = pd.Series(

            1.0 / n_assets,

            index=assets,

            dtype=float,
        )

        return TargetPortfolio(

            weights=
            weights,

            diagnostics={

                "optimizer":
                "equal_weight",

                "assets":
                n_assets,
            },
        )


# ============================================================
# OPTIMIZATION CONFIG
# ============================================================


@dataclass(slots=True)
class OptimizationStageConfig:
    """
    Optimization settings.
    """

    enabled: bool = True

    allow_fallback: bool = True

    fail_on_optimizer_error: bool = False

    minimum_assets: int = 1


# ============================================================
# OPTIMIZATION DIAGNOSTICS
# ============================================================


@dataclass(slots=True)
class OptimizationDiagnostics:
    """
    Optimizer diagnostics.
    """

    n_assets: int = 0

    gross_exposure: float = 0.0

    net_exposure: float = 0.0

    max_weight: float = 0.0

    min_weight: float = 0.0

    hhi: float = 0.0

    effective_n: float = 0.0


# ============================================================
# OPTIMIZATION STAGE
# ============================================================


class OptimizationStage:
    """
    Institutional optimization stage.
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        optimizer:
        BaseOptimizerEngine
        | None = None,

        config:
        OptimizationStageConfig
        | None = None,
    ) -> None:

        self.optimizer = optimizer

        self.config = (
            config
            if config is not None
            else OptimizationStageConfig()
        )

    # --------------------------------------------------------

    @staticmethod
    def portfolio_diagnostics(
        portfolio:
        TargetPortfolio,
    ) -> dict[str, Any]:

        weights = (
            portfolio.weights
            .fillna(0.0)
        )

        diagnostics = {}

        diagnostics[
            "n_assets"
        ] = len(
            weights
        )

        diagnostics[
            "gross_exposure"
        ] = float(
            weights.abs()
            .sum()
        )

        diagnostics[
            "net_exposure"
        ] = float(
            weights.sum()
        )

        diagnostics[
            "max_weight"
        ] = float(
            weights.max()
        )

        diagnostics[
            "min_weight"
        ] = float(
            weights.min()
        )

        hhi = float(
            (weights ** 2)
            .sum()
        )

        diagnostics[
            "hhi"
        ] = hhi

        diagnostics[
            "effective_n"
        ] = (
            1.0 / hhi
            if hhi > 0
            else 0.0
        )

        return diagnostics

    # --------------------------------------------------------

    def run(
        self,
        *,
        inputs:
        PipelineInput,

        forecast_output:
        ForecastStageOutput
        | None,

        risk_output:
        RiskStageOutput
        | None,

        constraint_output:
        ConstraintStageOutput
        | None,
    ) -> OptimizationStageOutput:

        start = time.perf_counter()

        try:

            # ----------------------------------
            # Disabled
            # ----------------------------------

            if (
                not self.config
                .enabled
            ):

                return (
                    OptimizationStageOutput(

                        result=None,

                        diagnostics={
                            "status":
                            "disabled"
                        },
                    )
                )

            # ----------------------------------
            # Optimizer
            # ----------------------------------

            portfolio = None

            if (
                self.optimizer
                is not None
            ):

                portfolio = (
                    self.optimizer.run(

                        inputs=
                        inputs,

                        forecast_output=
                        forecast_output,

                        risk_output=
                        risk_output,

                        constraint_output=
                        constraint_output,
                    )
                )

            elif (
                self.config
                .allow_fallback
            ):

                portfolio = (

                    EqualWeightOptimizer()
                    .run(

                        inputs=
                        inputs,

                        forecast_output=
                        forecast_output,

                        risk_output=
                        risk_output,

                        constraint_output=
                        constraint_output,
                    )

                )

            # ----------------------------------
            # Failure
            # ----------------------------------

            if (
                portfolio is None
                and
                self.config
                .fail_on_optimizer_error
            ):

                raise ValueError(
                    "Optimizer failed."
                )

            diagnostics = {}

            if (
                portfolio is not None
            ):

                diagnostics = (
                    self
                    .portfolio_diagnostics(
                        portfolio
                    )
                )

            diagnostics[
                "runtime_seconds"
            ] = (
                time.perf_counter()
                - start
            )

            return (
                OptimizationStageOutput(

                    result=
                    portfolio,

                    diagnostics=
                    diagnostics,
                )
            )

        except Exception as exc:

            return (
                OptimizationStageOutput(

                    result=None,

                    diagnostics={

                        "error":
                        str(exc),

                        "runtime_seconds":
                        (
                            time.perf_counter()
                            - start
                        ),
                    },
                )
            )


# ============================================================
# OPTIMIZATION FACTORY
# ============================================================

class OptimizationStageFactory:
    """
    Optimization constructors.
    """

    # --------------------------------------------------------

    @staticmethod
    def equal_weight() -> OptimizationStage:

        return OptimizationStage(
            optimizer=EqualWeightOptimizer()
        )

    # --------------------------------------------------------

    @staticmethod
    def with_optimizer(
        optimizer: BaseOptimizerEngine,
    ) -> OptimizationStage:

        return OptimizationStage(
            optimizer=optimizer
        )

    # --------------------------------------------------------

    @staticmethod
    def create(
        *,
        metadata: PipelineMetadata,
        config: PipelineConfig,
        pipeline_input: PipelineInput,
    ) -> OptimizationStage:
        """
        Institutional factory entrypoint.

        Future examples:
            - EqualWeightOptimizer
            - MeanVarianceOptimizer
            - BlackLittermanOptimizer
            - RiskParityOptimizer
            - HierarchicalRiskParityOptimizer
            - CVaROptimizer
            """

        # Future:
        #
        # if config.optimizer.engine == "mean_variance":
        #     return OptimizationStageFactory.with_optimizer(
        #         MeanVarianceOptimizer(...)
        #     )
        #
        # if config.optimizer.engine == "black_litterman":
        #     return OptimizationStageFactory.with_optimizer(
        #         BlackLittermanOptimizer(...)
        #     )
        #
        # if config.optimizer.engine == "risk_parity":
        #     return OptimizationStageFactory.with_optimizer(
        #         RiskParityOptimizer(...)
        #     )

        # Reserved for future use
        _ = metadata
        _ = config
        _ = pipeline_input

        return OptimizationStageFactory.equal_weight()


# ============================================================
# CONTEXT INTEGRATION
# ============================================================


def run_optimization_stage(
    *,
    context: PipelineContext,
    inputs: PipelineInput,
    stage: OptimizationStage,
    forecast_output: ForecastStageOutput | None,
    risk_output: RiskStageOutput | None,
    constraint_output: ConstraintStageOutput | None,
) -> OptimizationStageOutput:
    
    """
    Execute optimization stage and
    persist result into context.
    """

    output = (
        stage.run(
            inputs= inputs,
            forecast_output= forecast_output,
            risk_output= risk_output,
            constraint_output= constraint_output,
        )
    )

    context.shared_objects["optimization_output"] = output

    return output


# ============================================================
# PART 8 — PORTFOLIO BUILD STAGE
# ============================================================

import time


# ============================================================
# PORTFOLIO HOLDING
# ============================================================


@dataclass(slots=True)
class PortfolioHolding:
    """
    Canonical institutional holding.
    """

    asset_id: str

    target_weight: float

    current_weight: float = 0.0

    active_weight: float = 0.0

    expected_return: float = 0.0

    expected_risk: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# INSTITUTIONAL PORTFOLIO
# ============================================================


@dataclass(slots=True)
class InstitutionalPortfolio:
    """
    Canonical portfolio representation.
    """

    holdings: list[PortfolioHolding]

    weights: pd.Series

    cash_weight: float = 0.0

    gross_exposure: float = 0.0

    net_exposure: float = 0.0

    effective_n: float = 0.0

    expected_return: float = 0.0

    expected_volatility: float = 0.0

    expected_sharpe: float = 0.0

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# PORTFOLIO BUILD CONFIG
# ============================================================


@dataclass(slots=True)
class PortfolioBuildStageConfig:
    """
    Portfolio assembly settings.
    """

    enabled: bool = True

    minimum_weight: float = 0.0

    normalize_weights: bool = True

    drop_zero_weight_assets: bool = True


# ============================================================
# PORTFOLIO BUILD DIAGNOSTICS
# ============================================================


@dataclass(slots=True)
class PortfolioBuildDiagnostics:
    """
    Portfolio diagnostics.
    """

    positions: int = 0

    gross_exposure: float = 0.0

    net_exposure: float = 0.0

    effective_n: float = 0.0

    cash_weight: float = 0.0


# ============================================================
# PORTFOLIO BUILDER
# ============================================================


class InstitutionalPortfolioBuilder:
    """
    Converts optimizer output into
    institutional portfolio object.
    """

    # --------------------------------------------------------

    @staticmethod
    def clean_weights(
        weights: pd.Series,
        config:
        PortfolioBuildStageConfig,
    ) -> pd.Series:

        w = (
            weights.copy()
            .fillna(0.0)
        )

        # -----------------------------
        # Drop tiny weights
        # -----------------------------

        if (
            config.minimum_weight
            > 0
        ):

            w.loc[
                w.abs()
                < config.minimum_weight
            ] = 0.0

        # -----------------------------
        # Remove zeros
        # -----------------------------

        if (
            config
            .drop_zero_weight_assets
        ):

            w = (
                w.loc[
                    w != 0
                ]
            )

        # -----------------------------
        # Normalize
        # -----------------------------

        if (
            config
            .normalize_weights
        ):

            gross = float(
                w.abs()
                .sum()
            )

            if gross > 0:

                w = (
                    w / gross
                )

        return w

    # --------------------------------------------------------

    @staticmethod
    def compute_effective_n(
        weights:
        pd.Series,
    ) -> float:

        hhi = float(
            (weights ** 2)
            .sum()
        )

        if hhi <= 0:

            return 0.0

        return (
            1.0 / hhi
        )

    # --------------------------------------------------------

    def build(
        self,
        *,
        target_portfolio:
        TargetPortfolio,

        current_weights:
        pd.Series | None,

        config:
        PortfolioBuildStageConfig,
    ) -> InstitutionalPortfolio:

        weights = (
            self.clean_weights(
                target_portfolio.weights,
                config,
            )
        )

        holdings = []

        for asset, weight in (
            weights.items()
        ):

            current = 0.0

            if (
                current_weights
                is not None
            ):

                current = float(

                    current_weights.get(
                        asset,
                        0.0,
                    )

                )

            holdings.append(

                PortfolioHolding(

                    asset_id=
                    str(asset),

                    target_weight=
                    float(weight),

                    current_weight=
                    current,

                    active_weight=
                    float(
                        weight
                        - current
                    ),
                )

            )

        gross = float(
            weights.abs()
            .sum()
        )

        net = float(
            weights.sum()
        )

        effective_n = (
            self.compute_effective_n(
                weights
            )
        )

        cash_weight = max(
            0.0,
            1.0 - gross,
        )

        diagnostics = {

            "positions":
            len(weights),

            "gross_exposure":
            gross,

            "net_exposure":
            net,

            "effective_n":
            effective_n,

            "cash_weight":
            cash_weight,
        }

        return InstitutionalPortfolio(

            holdings=
            holdings,

            weights=
            weights,

            cash_weight=
            cash_weight,

            gross_exposure=
            gross,

            net_exposure=
            net,

            effective_n=
            effective_n,

            expected_return=
            target_portfolio
            .expected_return,

            expected_volatility=
            target_portfolio
            .expected_volatility,

            expected_sharpe=
            target_portfolio
            .expected_sharpe,

            diagnostics=
            diagnostics,
        )


# ============================================================
# PORTFOLIO BUILD STAGE
# ============================================================


class PortfolioBuildStage:
    """
    Institutional portfolio assembly stage.
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        config:
        PortfolioBuildStageConfig
        | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else PortfolioBuildStageConfig()
        )

        self.builder = (
            InstitutionalPortfolioBuilder()
        )

    # --------------------------------------------------------

    def run(
        self,
        *,
        inputs:
        PipelineInput,

        optimization_output:
        OptimizationStageOutput
        | None,
    ) -> PortfolioBuildStageOutput:

        start = time.perf_counter()

        try:

            if (
                not self.config
                .enabled
            ):

                return (
                    PortfolioBuildStageOutput(

                        result=None,

                        diagnostics={
                            "status":
                            "disabled"
                        },
                    )
                )

            if (
                optimization_output
                is None
            ):

                raise ValueError(
                    "Missing optimization output."
                )

            target_portfolio = (
                optimization_output
                .result
            )

            if (
                target_portfolio
                is None
            ):

                raise ValueError(
                    "Optimizer returned None."
                )

            portfolio = (
                self.builder.build(

                    target_portfolio=
                    target_portfolio,

                    current_weights=
                    inputs
                    .portfolio_data
                    .current_weights,

                    config=
                    self.config,
                )
            )

            diagnostics = (
                portfolio
                .diagnostics
            )

            diagnostics[
                "runtime_seconds"
            ] = (
                time.perf_counter()
                - start
            )

            return (
                PortfolioBuildStageOutput(

                    result=
                    portfolio,

                    diagnostics=
                    diagnostics,
                )
            )

        except Exception as exc:

            return (
                PortfolioBuildStageOutput(

                    result=None,

                    diagnostics={

                        "error":
                        str(exc),

                        "runtime_seconds":
                        (
                            time.perf_counter()
                            - start
                        ),
                    },
                )
            )


# ============================================================
# FACTORY
# ============================================================


class PortfolioBuildStageFactory:
    """
    Portfolio build constructors.
    """

    @staticmethod
    def create(
        *,
        metadata: PipelineMetadata,
        config: PipelineConfig,
        pipeline_input: PipelineInput,
    ) -> PortfolioBuildStage:

        _ = metadata
        _ = config
        _ = pipeline_input

        return PortfolioBuildStage()


# ============================================================
# CONTEXT INTEGRATION
# ============================================================


def run_portfolio_build_stage(
    *,
    context:
    PipelineContext,

    inputs:
    PipelineInput,

    stage:
    PortfolioBuildStage,

    optimization_output:
    OptimizationStageOutput
    | None,
) -> PortfolioBuildStageOutput:
    """
    Execute portfolio assembly stage.
    """

    output = (
        stage.run(

            inputs=
            inputs,

            optimization_output=
            optimization_output,
        )
    )

    context.shared_objects[
        "portfolio_build_output"
    ] = output

    return output


# ============================================================
# PART 9 — REBALANCE STAGE
# ============================================================

import time


# ============================================================
# TRADE ACTION
# ============================================================


class TradeAction(
    Enum,
):
    """
    Canonical trade action.
    """

    BUY = "BUY"

    SELL = "SELL"

    HOLD = "HOLD"


# ============================================================
# REBALANCE TRADE
# ============================================================


@dataclass(slots=True)
class RebalanceTrade:
    """
    Single rebalance trade.
    """

    asset_id: str

    action: TradeAction

    current_weight: float

    target_weight: float

    weight_change: float

    abs_weight_change: float


# ============================================================
# REBALANCE PACKAGE
# ============================================================


@dataclass(slots=True)
class InstitutionalRebalance:
    """
    Institutional rebalance package.
    """

    trades: list[RebalanceTrade]

    turnover: float

    buy_turnover: float

    sell_turnover: float

    gross_trade_weight: float

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# REBALANCE CONFIG
# ============================================================


@dataclass(slots=True)
class RebalanceStageConfig:
    """
    Rebalance settings.
    """

    enabled: bool = True

    minimum_trade_weight: float = 0.0001

    ignore_small_trades: bool = True


# ============================================================
# REBALANCE DIAGNOSTICS
# ============================================================


@dataclass(slots=True)
class RebalanceDiagnostics:
    """
    Rebalance diagnostics.
    """

    trade_count: int = 0

    turnover: float = 0.0

    buy_turnover: float = 0.0

    sell_turnover: float = 0.0


# ============================================================
# REBALANCE ENGINE
# ============================================================


class InstitutionalRebalanceEngine:
    """
    Institutional rebalance engine.
    """

    # --------------------------------------------------------

    def generate_trades(
        self,
        *,
        target_weights:
        pd.Series,

        current_weights:
        pd.Series | None,

        config:
        RebalanceStageConfig,
    ) -> list[
        RebalanceTrade
    ]:

        trades = []

        if (
            current_weights
            is None
        ):

            current_weights = (
                pd.Series(
                    dtype=float
                )
            )

        all_assets = sorted(

            set(
                target_weights.index
            )

            |

            set(
                current_weights.index
            )
        )

        for asset in all_assets:

            current = float(

                current_weights.get(
                    asset,
                    0.0,
                )

            )

            target = float(

                target_weights.get(
                    asset,
                    0.0,
                )

            )

            change = (
                target
                - current
            )

            abs_change = abs(
                change
            )

            # -------------------------
            # Ignore tiny trades
            # -------------------------

            if (
                config.ignore_small_trades
                and
                abs_change
                < config.minimum_trade_weight
            ):

                continue

            # -------------------------
            # Action
            # -------------------------

            if change > 0:

                action = (
                    TradeAction.BUY
                )

            elif change < 0:

                action = (
                    TradeAction.SELL
                )

            else:

                action = (
                    TradeAction.HOLD
                )

            trades.append(

                RebalanceTrade(

                    asset_id=
                    str(asset),

                    action=
                    action,

                    current_weight=
                    current,

                    target_weight=
                    target,

                    weight_change=
                    change,

                    abs_weight_change=
                    abs_change,
                )

            )

        return trades

    # --------------------------------------------------------

    @staticmethod
    def compute_statistics(
        trades:
        list[
            RebalanceTrade
        ],
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:

        if len(trades) == 0:

            return (
                0.0,
                0.0,
                0.0,
                0.0,
            )

        buy_turnover = sum(

            t.abs_weight_change

            for t in trades

            if t.action
            ==
            TradeAction.BUY
        )

        sell_turnover = sum(

            t.abs_weight_change

            for t in trades

            if t.action
            ==
            TradeAction.SELL
        )

        gross_trade = (
            buy_turnover
            + sell_turnover
        )

        turnover = (
            gross_trade
            / 2.0
        )

        return (
            turnover,
            buy_turnover,
            sell_turnover,
            gross_trade,
        )

    # --------------------------------------------------------

    def run(
        self,
        *,
        portfolio:
        InstitutionalPortfolio,

        current_weights:
        pd.Series | None,

        config:
        RebalanceStageConfig,
    ) -> InstitutionalRebalance:

        trades = (
            self.generate_trades(

                target_weights=
                portfolio.weights,

                current_weights=
                current_weights,

                config=
                config,
            )
        )

        (
            turnover,
            buy_turnover,
            sell_turnover,
            gross_trade,
        ) = (
            self.compute_statistics(
                trades
            )
        )

        diagnostics = {

            "trade_count":
            len(trades),

            "turnover":
            turnover,

            "buy_turnover":
            buy_turnover,

            "sell_turnover":
            sell_turnover,
        }

        return (
            InstitutionalRebalance(

                trades=
                trades,

                turnover=
                turnover,

                buy_turnover=
                buy_turnover,

                sell_turnover=
                sell_turnover,

                gross_trade_weight=
                gross_trade,

                diagnostics=
                diagnostics,
            )
        )


# ============================================================
# REBALANCE STAGE
# ============================================================


class RebalanceStage:
    """
    Institutional rebalance stage.
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        config:
        RebalanceStageConfig
        | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else RebalanceStageConfig()
        )

        self.engine = (
            InstitutionalRebalanceEngine()
        )

    # --------------------------------------------------------

    def run(
        self,
        *,
        inputs:
        PipelineInput,

        portfolio_output:
        PortfolioBuildStageOutput
        | None,
    ) -> RebalanceStageOutput:

        start = time.perf_counter()

        try:

            if (
                not self.config
                .enabled
            ):

                return (
                    RebalanceStageOutput(

                        result=None,

                        diagnostics={
                            "status":
                            "disabled"
                        },
                    )
                )

            if (
                portfolio_output
                is None
            ):

                raise ValueError(
                    "Missing portfolio output."
                )

            portfolio = (
                portfolio_output
                .result
            )

            if (
                portfolio is None
            ):

                raise ValueError(
                    "Portfolio unavailable."
                )

            rebalance = (
                self.engine.run(

                    portfolio=
                    portfolio,

                    current_weights=
                    inputs
                    .portfolio_data
                    .current_weights,

                    config=
                    self.config,
                )
            )

            diagnostics = (
                rebalance
                .diagnostics
            )

            diagnostics[
                "runtime_seconds"
            ] = (
                time.perf_counter()
                - start
            )

            return (
                RebalanceStageOutput(

                    result=
                    rebalance,

                    diagnostics=
                    diagnostics,
                )
            )

        except Exception as exc:

            return (
                RebalanceStageOutput(

                    result=None,

                    diagnostics={

                        "error":
                        str(exc),

                        "runtime_seconds":
                        (
                            time.perf_counter()
                            - start
                        ),
                    },
                )
            )


# ============================================================
# FACTORY
# ============================================================

class RebalanceStageFactory:
    """
    Rebalance constructors.
    """

    @staticmethod
    def create(
        *,
        metadata: PipelineMetadata,
        config: PipelineConfig,
        pipeline_input: PipelineInput,
    ) -> RebalanceStage:

        # Reserved for future use
        _ = metadata
        _ = config
        _ = pipeline_input

        return RebalanceStage()

# ============================================================
# CONTEXT INTEGRATION
# ============================================================


def run_rebalance_stage(
    *,
    context: PipelineContext,
    inputs: PipelineInput,
    stage: RebalanceStage,
    portfolio_output: PortfolioBuildStageOutput | None,
) -> RebalanceStageOutput:

    _ = inputs   # if not used

    output = stage.run(
        inputs=inputs,
        portfolio_output=portfolio_output,
    )

    context.shared_objects["rebalance_output"] = output

    return output


# ============================================================
# PART 10 — EXECUTION STAGE
# ============================================================

import time


# ============================================================
# EXECUTION SIDE
# ============================================================


class ExecutionSide(
    Enum,
):
    """
    Trade direction.
    """

    BUY = "BUY"

    SELL = "SELL"


# ============================================================
# EXECUTION ORDER
# ============================================================


@dataclass(slots=True)
class ExecutionOrder:
    """
    Canonical execution order.
    """

    asset_id: str

    side: ExecutionSide

    target_weight_change: float

    estimated_slippage: float

    estimated_cost: float

    estimated_market_impact: float

    expected_fill_price: float | None = None


# ============================================================
# EXECUTION PACKAGE
# ============================================================


@dataclass(slots=True)
class InstitutionalExecutionPackage:
    """
    Institutional execution package.
    """

    orders: list[ExecutionOrder]

    total_turnover: float

    total_cost: float

    total_slippage: float

    total_market_impact: float

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# EXECUTION CONFIG
# ============================================================


@dataclass(slots=True)
class ExecutionStageConfig:
    """
    Execution assumptions.
    """

    enabled: bool = True

    transaction_cost_bps: float = 5.0

    slippage_bps: float = 10.0

    market_impact_bps: float = 3.0

    ignore_small_orders: bool = True

    minimum_order_weight: float = 0.0001


# ============================================================
# EXECUTION DIAGNOSTICS
# ============================================================


@dataclass(slots=True)
class ExecutionDiagnostics:
    """
    Execution summary.
    """

    order_count: int = 0

    turnover: float = 0.0

    total_cost: float = 0.0

    total_slippage: float = 0.0

    total_market_impact: float = 0.0


# ============================================================
# EXECUTION ENGINE
# ============================================================


class InstitutionalExecutionEngine:
    """
    Institutional execution simulator.
    """

    # --------------------------------------------------------

    @staticmethod
    def bps_to_decimal(
        bps: float,
    ) -> float:

        return (
            bps / 10000.0
        )

    # --------------------------------------------------------

    def create_orders(
        self,
        *,
        rebalance:
        InstitutionalRebalance,

        config:
        ExecutionStageConfig,
    ) -> list[
        ExecutionOrder
    ]:

        orders = []

        tc_rate = (
            self.bps_to_decimal(
                config.transaction_cost_bps
            )
        )

        slippage_rate = (
            self.bps_to_decimal(
                config.slippage_bps
            )
        )

        impact_rate = (
            self.bps_to_decimal(
                config.market_impact_bps
            )
        )

        for trade in rebalance.trades:

            size = (
                trade.abs_weight_change
            )

            # -------------------------
            # Ignore tiny trades
            # -------------------------

            if (
                config.ignore_small_orders
                and
                size
                < config.minimum_order_weight
            ):

                continue

            side = (

                ExecutionSide.BUY

                if trade.action
                ==
                TradeAction.BUY

                else

                ExecutionSide.SELL

            )

            estimated_cost = (
                size
                * tc_rate
            )

            estimated_slippage = (
                size
                * slippage_rate
            )

            estimated_impact = (
                size
                * impact_rate
            )

            orders.append(

                ExecutionOrder(

                    asset_id=
                    trade.asset_id,

                    side=
                    side,

                    target_weight_change=
                    trade.weight_change,

                    estimated_slippage=
                    estimated_slippage,

                    estimated_cost=
                    estimated_cost,

                    estimated_market_impact=
                    estimated_impact,
                )

            )

        return orders

    # --------------------------------------------------------

    def run(
        self,
        *,
        rebalance:
        InstitutionalRebalance,

        config:
        ExecutionStageConfig,
    ) -> (
        InstitutionalExecutionPackage
    ):

        orders = (
            self.create_orders(

                rebalance=
                rebalance,

                config=
                config,
            )
        )

        total_cost = float(

            sum(

                x.estimated_cost

                for x in orders
            )
        )

        total_slippage = float(

            sum(

                x.estimated_slippage

                for x in orders
            )
        )

        total_impact = float(

            sum(

                x
                .estimated_market_impact

                for x in orders
            )
        )

        diagnostics = {

            "order_count":
            len(orders),

            "turnover":
            rebalance.turnover,

            "total_cost":
            total_cost,

            "total_slippage":
            total_slippage,

            "total_market_impact":
            total_impact,
        }

        return (
            InstitutionalExecutionPackage(

                orders=
                orders,

                total_turnover=
                rebalance.turnover,

                total_cost=
                total_cost,

                total_slippage=
                total_slippage,

                total_market_impact=
                total_impact,

                diagnostics=
                diagnostics,
            )
        )


# ============================================================
# EXECUTION STAGE
# ============================================================


class ExecutionStage:
    """
    Institutional execution stage.
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        config:
        ExecutionStageConfig
        | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else ExecutionStageConfig()
        )

        self.engine = (
            InstitutionalExecutionEngine()
        )

    # --------------------------------------------------------

    def run(
        self,
        *,
        rebalance_output:
        RebalanceStageOutput
        | None,
    ) -> ExecutionStageOutput:

        start = (
            time.perf_counter()
        )

        try:

            if (
                not self.config
                .enabled
            ):

                return (
                    ExecutionStageOutput(

                        result=None,

                        diagnostics={
                            "status":
                            "disabled"
                        },
                    )
                )

            if (
                rebalance_output
                is None
            ):

                raise ValueError(
                    "Missing rebalance output."
                )

            rebalance = (
                rebalance_output
                .result
            )

            if (
                rebalance
                is None
            ):

                raise ValueError(
                    "Rebalance unavailable."
                )

            execution = (
                self.engine.run(

                    rebalance=
                    rebalance,

                    config=
                    self.config,
                )
            )

            diagnostics = (
                execution
                .diagnostics
            )

            diagnostics[
                "runtime_seconds"
            ] = (
                time.perf_counter()
                - start
            )

            return (
                ExecutionStageOutput(

                    result=
                    execution,

                    diagnostics=
                    diagnostics,
                )
            )

        except Exception as exc:

            return (
                ExecutionStageOutput(

                    result=None,

                    diagnostics={

                        "error":
                        str(exc),

                        "runtime_seconds":
                        (
                            time.perf_counter()
                            - start
                        ),
                    },
                )
            )


# ============================================================
# FACTORY
# ============================================================

class ExecutionStageFactory:
    """
    Execution constructors.
    """

    @staticmethod
    def create(
        *,
        metadata: PipelineMetadata,
        config: PipelineConfig,
        pipeline_input: PipelineInput,
    ) -> ExecutionStage:

        _ = metadata
        _ = config
        _ = pipeline_input

        return ExecutionStage()


# ============================================================
# CONTEXT INTEGRATION
# ============================================================


def run_execution_stage(
    *,
    context: PipelineContext,
    stage: ExecutionStage,
    rebalance_output: RebalanceStageOutput | None,
) -> ExecutionStageOutput:
    """
    Execute execution stage.
    """

    output = (
        stage.run(

            rebalance_output=
            rebalance_output,
        )
    )

    context.shared_objects[
        "execution_output"
    ] = output

    return output


# ============================================================
# PART 11 — DIAGNOSTICS STAGE
# ============================================================

import time


# ============================================================
# DIAGNOSTIC SOURCE
# ============================================================


class DiagnosticSource(
    Enum,
):
    """
    Origin of diagnostic result.
    """

    PORTFOLIO = "portfolio"

    ANALYTICS = "analytics"

    ATTRIBUTION = "attribution"

    STRESS_TEST = "stress_test"

    EXECUTION = "execution"

    REBALANCE = "rebalance"


# ============================================================
# DIAGNOSTIC RECORD
# ============================================================


@dataclass(slots=True)
class DiagnosticRecord:
    """
    Single diagnostic entry.
    """

    source: DiagnosticSource

    metric_name: str

    metric_value: Any

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# MASTER DIAGNOSTIC PACKAGE
# ============================================================


@dataclass(slots=True)
class InstitutionalDiagnosticsPackage:
    """
    Institutional diagnostics.
    """

    records: list[
        DiagnosticRecord
    ]

    summary: dict[str, Any]

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# DIAGNOSTIC CONFIG
# ============================================================


@dataclass(slots=True)
class DiagnosticsStageConfig:
    """
    Diagnostics stage settings.
    """

    enabled: bool = True

    collect_portfolio: bool = True

    collect_execution: bool = True

    collect_rebalance: bool = True

    collect_analytics: bool = True

    collect_attribution: bool = True

    collect_stress: bool = True


# ============================================================
# DIAGNOSTICS ENGINE
# ============================================================


class InstitutionalDiagnosticsEngine:
    """
    Aggregates institutional
    diagnostics.
    """

    # --------------------------------------------------------

    @staticmethod
    def add_record(
        records:
        list[
            DiagnosticRecord
        ],
        *,
        source:
        DiagnosticSource,
        name:
        str,
        value:
        Any,
    ) -> None:

        records.append(

            DiagnosticRecord(

                source=
                source,

                metric_name=
                name,

                metric_value=
                value,
            )

        )

    # --------------------------------------------------------

    def collect_portfolio(
        self,
        portfolio:
        InstitutionalPortfolio,
        records:
        list[
            DiagnosticRecord
        ],
    ) -> None:

        self.add_record(

            records,

            source=
            DiagnosticSource
            .PORTFOLIO,

            name=
            "gross_exposure",

            value=
            portfolio
            .gross_exposure,
        )

        self.add_record(

            records,

            source=
            DiagnosticSource
            .PORTFOLIO,

            name=
            "net_exposure",

            value=
            portfolio
            .net_exposure,
        )

        self.add_record(

            records,

            source=
            DiagnosticSource
            .PORTFOLIO,

            name=
            "effective_n",

            value=
            portfolio
            .effective_n,
        )

        self.add_record(

            records,

            source=
            DiagnosticSource
            .PORTFOLIO,

            name=
            "expected_return",

            value=
            portfolio
            .expected_return,
        )

        self.add_record(

            records,

            source=
            DiagnosticSource
            .PORTFOLIO,

            name=
            "expected_volatility",

            value=
            portfolio
            .expected_volatility,
        )

    # --------------------------------------------------------

    def collect_rebalance(
        self,
        rebalance:
        InstitutionalRebalance,
        records:
        list[
            DiagnosticRecord
        ],
    ) -> None:

        self.add_record(

            records,

            source=
            DiagnosticSource
            .REBALANCE,

            name=
            "turnover",

            value=
            rebalance
            .turnover,
        )

        self.add_record(

            records,

            source=
            DiagnosticSource
            .REBALANCE,

            name=
            "trade_count",

            value=
            len(
                rebalance
                .trades
            ),
        )

    # --------------------------------------------------------

    def collect_execution(
        self,
        execution:
        InstitutionalExecutionPackage,
        records:
        list[
            DiagnosticRecord
        ],
    ) -> None:

        self.add_record(

            records,

            source=
            DiagnosticSource
            .EXECUTION,

            name=
            "execution_cost",

            value=
            execution
            .total_cost,
        )

        self.add_record(

            records,

            source=
            DiagnosticSource
            .EXECUTION,

            name=
            "slippage",

            value=
            execution
            .total_slippage,
        )

        self.add_record(

            records,

            source=
            DiagnosticSource
            .EXECUTION,

            name=
            "market_impact",

            value=
            execution
            .total_market_impact,
        )

    # --------------------------------------------------------

    def collect_external_result(
        self,
        *,
        result:
        Any,
        source:
        DiagnosticSource,
        records:
        list[
            DiagnosticRecord
        ],
    ) -> None:

        if result is None:

            return

        if hasattr(
            result,
            "diagnostics"
        ):

            diag = getattr(
                result,
                "diagnostics"
            )

            if isinstance(
                diag,
                dict,
            ):

                for (
                    k,
                    v,
                ) in diag.items():

                    self.add_record(

                        records,

                        source=
                        source,

                        name=
                        str(k),

                        value=
                        v,
                    )

    # --------------------------------------------------------

    @staticmethod
    def build_summary(
        records:
        list[
            DiagnosticRecord
        ],
    ) -> dict[str, Any]:

        summary = {}

        summary[
            "record_count"
        ] = len(records)

        summary[
            "sources"
        ] = sorted(

            list(

                {
                    x.source.value
                    for x in records
                }

            )

        )

        return summary

    # --------------------------------------------------------

    def run(
        self,
        *,
        portfolio:
        InstitutionalPortfolio
        | None,

        rebalance:
        InstitutionalRebalance
        | None,

        execution:
        InstitutionalExecutionPackage
        | None,

        analytics_result:
        Any,

        attribution_result:
        Any,

        stress_result:
        Any,
    ) -> (
        InstitutionalDiagnosticsPackage
    ):

        records = []

        if (
            portfolio
            is not None
        ):

            self.collect_portfolio(

                portfolio,
                records,
            )

        if (
            rebalance
            is not None
        ):

            self.collect_rebalance(

                rebalance,
                records,
            )

        if (
            execution
            is not None
        ):

            self.collect_execution(

                execution,
                records,
            )

        self.collect_external_result(

            result=
            analytics_result,

            source=
            DiagnosticSource
            .ANALYTICS,

            records=
            records,
        )

        self.collect_external_result(

            result=
            attribution_result,

            source=
            DiagnosticSource
            .ATTRIBUTION,

            records=
            records,
        )

        self.collect_external_result(

            result=
            stress_result,

            source=
            DiagnosticSource
            .STRESS_TEST,

            records=
            records,
        )

        summary = (
            self.build_summary(
                records
            )
        )

        return (
            InstitutionalDiagnosticsPackage(

                records=
                records,

                summary=
                summary,
            )
        )


# ============================================================
# DIAGNOSTICS STAGE
# ============================================================


class DiagnosticsStage:
    """
    Institutional diagnostics stage.
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        config:
        DiagnosticsStageConfig
        | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else DiagnosticsStageConfig()
        )

        self.engine = (
            InstitutionalDiagnosticsEngine()
        )

    # --------------------------------------------------------

    def run(
        self,
        *,
        portfolio_output:
        PortfolioBuildStageOutput
        | None,

        rebalance_output:
        RebalanceStageOutput
        | None,

        execution_output:
        ExecutionStageOutput
        | None,

        analytics_result:
        Any = None,

        attribution_result:
        Any = None,

        stress_result:
        Any = None,
    ) -> DiagnosticsStageOutput:

        start = (
            time.perf_counter()
        )

        try:

            if (
                not self.config
                .enabled
            ):

                return (
                    DiagnosticsStageOutput(
                        result=None,
                        diagnostics={
                            "status":
                            "disabled"
                        },
                    )
                )

            package = (
                self.engine.run(

                    portfolio=
                    (
                        portfolio_output.result
                        if portfolio_output
                        else None
                    ),

                    rebalance=
                    (
                        rebalance_output.result
                        if rebalance_output
                        else None
                    ),

                    execution=
                    (
                        execution_output.result
                        if execution_output
                        else None
                    ),

                    analytics_result=
                    analytics_result,

                    attribution_result=
                    attribution_result,

                    stress_result=
                    stress_result,
                )
            )

            diagnostics = {

                "record_count":
                len(
                    package.records
                ),

                "runtime_seconds":
                (
                    time.perf_counter()
                    - start
                ),
            }

            return (
                DiagnosticsStageOutput(
                    result= package,
                    diagnostics= diagnostics,
                )
            )

        except Exception as exc:

            return (
                DiagnosticsStageOutput(
                    result= None,
                    diagnostics={
                        "error":
                        str(exc),
                        "runtime_seconds":
                        (
                            time.perf_counter()
                            - start
                        ),
                    },
                )
            )


# ============================================================
# FACTORY
# ============================================================


class DiagnosticsStageFactory:

    @staticmethod
    def create(
        *,
        metadata: PipelineMetadata,
        config: PipelineConfig,
        pipeline_input: PipelineInput,
    ) -> DiagnosticsStage:

        # Reserved for future use:
        # - broker selection
        # - OMS/EMS configuration
        # - execution policies
        # - venue routing
        # - slippage models

        _ = metadata
        _ = config
        _ = pipeline_input

        return (
            DiagnosticsStage()
        )


# ============================================================
# CONTEXT INTEGRATION
# ============================================================


def run_diagnostics_stage(
    *,
    context: PipelineContext,
    stage: DiagnosticsStage,
    portfolio_output: PortfolioBuildStageOutput | None,
    rebalance_output: RebalanceStageOutput | None,
    execution_output: ExecutionStageOutput | None,
    analytics_result: Any = None,
    attribution_result: Any = None,
    stress_result: Any = None,
) -> DiagnosticsStageOutput:
    """
    Execute diagnostics stage.
    """

    output = (
        stage.run(

            portfolio_output= portfolio_output,
            rebalance_output= rebalance_output,
            execution_output= execution_output,
            analytics_result= analytics_result,
            attribution_result= attribution_result,
            stress_result= stress_result,
        )
    )

    context.shared_objects[
        "diagnostics_output"
    ] = output

    return output


# ============================================================
# PART 12 — INSTITUTIONAL PORTFOLIO REPORT STAGE
# ============================================================


# ============================================================
# REPORT CONFIG
# ============================================================


@dataclass(slots=True)
class PortfolioReportStageConfig:
    """
    Reporting stage settings.
    """

    enabled: bool = True
    include_forecast: bool = True
    include_risk: bool = True
    include_constraints: bool = True
    include_optimization: bool = True
    include_portfolio: bool = True
    include_rebalance: bool = True
    include_execution: bool = True
    include_diagnostics: bool = True

# ============================================================
# REPORT SUMMARY
# ============================================================


@dataclass(slots=True)
class PortfolioReportSummary:
    """
    High-level portfolio summary.
    """

    positions: int = 0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    effective_n: float = 0.0
    turnover: float = 0.0
    total_cost: float = 0.0
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    expected_sharpe: float = 0.0

# ============================================================
# REPORT BUILDER
# ============================================================


class InstitutionalPortfolioReportBuilder:
    """
    Builds final institutional report.
    """

    # --------------------------------------------------------

    @staticmethod
    def build_summary(
        *,
        portfolio:
        InstitutionalPortfolio
        | None,

        rebalance:
        InstitutionalRebalance
        | None,

        execution:
        InstitutionalExecutionPackage
        | None,
    ) -> PortfolioReportSummary:

        summary = (
            PortfolioReportSummary()
        )

        # ----------------------------------
        # Portfolio
        # ----------------------------------

        if portfolio is not None:

            summary.positions = (
                len(
                    portfolio.holdings
                )
            )

            summary.gross_exposure = (
                portfolio
                .gross_exposure
            )

            summary.net_exposure = (
                portfolio
                .net_exposure
            )

            summary.effective_n = (
                portfolio
                .effective_n
            )

            summary.expected_return = (
                portfolio
                .expected_return
            )

            summary.expected_volatility = (
                portfolio
                .expected_volatility
            )

            summary.expected_sharpe = (
                portfolio
                .expected_sharpe
            )

        # ----------------------------------
        # Rebalance
        # ----------------------------------

        if rebalance is not None:

            summary.turnover = (
                rebalance
                .turnover
            )

        # ----------------------------------
        # Execution
        # ----------------------------------

        if execution is not None:

            summary.total_cost = (
                execution
                .total_cost
            )

        return summary

    # --------------------------------------------------------

    @staticmethod
    def build(
        *,
        metadata:
        PortfolioBuilderMetadata,

        forecast_result:
        Any,

        risk_result:
        Any,

        constraint_result:
        Any,

        optimization_result:
        Any,

        portfolio_result:
        Any,

        rebalance_result:
        Any,

        execution_result:
        Any,

        diagnostics_result:
        Any,
    ) -> (InstitutionalPortfolioConstructionReport):

        portfolio_obj = None
        rebalance_obj = None
        execution_obj = None

        if (
            portfolio_result
            is not None
        ):

            portfolio_obj = (
                portfolio_result
                .result
            )

        if (
            rebalance_result
            is not None
        ):

            rebalance_obj = (
                rebalance_result
                .result
            )

        if (
            execution_result
            is not None
        ):

            execution_obj = (
                execution_result
                .result
            )

        summary = (
            InstitutionalPortfolioReportBuilder
            .build_summary(

                portfolio=
                portfolio_obj,

                rebalance=
                rebalance_obj,

                execution=
                execution_obj,
            )
        )

        return InstitutionalPortfolioConstructionReport(

            metadata=metadata,

            summary=summary,

            forecast_result=(
                forecast_result.result
                if forecast_result
                else None
            ),

            risk_result=(
                risk_result.result
                if risk_result
                else None
            ),

            constraint_set=(
                constraint_result.result
                if constraint_result
                else None
            ),

            optimization_result=(
                optimization_result.result
                if optimization_result
                else None
            ),

            portfolio_result=portfolio_obj,

            rebalance_result=rebalance_obj,

            validation_result=None,

            diagnostics_report=(
                diagnostics_result.result
                if diagnostics_result
                else None
            ),

            runtime_diagnostics={
                "execution": execution_obj
            },
        )


# ============================================================
# REPORT STAGE
# ============================================================


class PortfolioReportStage:
    """
    Institutional reporting stage.
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        metadata:
        PortfolioBuilderMetadata,

        config:
        PortfolioReportStageConfig
        | None = None,
    ) -> None:

        self.metadata = (
            metadata
        )

        self.config = (
            config
            if config is not None
            else PortfolioReportStageConfig()
        )

    # --------------------------------------------------------

    def run(
        self,
        *,
        forecast_output:
        ForecastStageOutput
        | None,

        risk_output:
        RiskStageOutput
        | None,

        constraint_output:
        ConstraintStageOutput
        | None,

        optimization_output:
        OptimizationStageOutput
        | None,

        portfolio_output:
        PortfolioBuildStageOutput
        | None,

        rebalance_output:
        RebalanceStageOutput
        | None,

        execution_output:
        ExecutionStageOutput
        | None,

        diagnostics_output:
        DiagnosticsStageOutput
        | None,
    ) -> PipelineStageOutput:

        start = (
            time.perf_counter()
        )

        try:

            if (
                not self.config
                .enabled
            ):

                return PipelineStageOutput(
                    stage=PipelineStage.REPORTING,
                    status=PipelineStatus.SKIPPED,
                    payload=None,
                    diagnostics={
                        "status": "disabled",
                    },
                    runtime_seconds=(
                        time.perf_counter() - start
                    ),
                )

            report = (
                InstitutionalPortfolioReportBuilder
                .build(

                    metadata=
                    self.metadata,

                    forecast_result=
                    forecast_output,

                    risk_result=
                    risk_output,

                    constraint_result=
                    constraint_output,

                    optimization_result=
                    optimization_output,

                    portfolio_result=
                    portfolio_output,

                    rebalance_result=
                    rebalance_output,

                    execution_result=
                    execution_output,

                    diagnostics_result=
                    diagnostics_output,
                )
            )

            diagnostics = {

                "positions":
                report.summary
                .positions,

                "runtime_seconds":
                (
                    time.perf_counter()
                    - start
                ),
            }

            return PipelineStageOutput(
                stage=PipelineStage.REPORTING,
                status=PipelineStatus.COMPLETED,
                payload=report,
                diagnostics=diagnostics,
                runtime_seconds=(
                    time.perf_counter() - start
                ),
            )

        except Exception as exc:
            import traceback
            traceback.print_exc()
            
            return PipelineStageOutput(
                stage=PipelineStage.REPORTING,
                status=PipelineStatus.FAILED,
                payload=None,
                diagnostics={
                    "error": str(exc),
                    "traceback":
                    traceback.format_exc(),
                },
                runtime_seconds=(
                    time.perf_counter() - start
                ),
                error_message=str(exc),
            )


# ============================================================
# FACTORY
# ============================================================

class PortfolioReportStageFactory:

    @staticmethod
    def create(
        *,
        metadata: PipelineMetadata,
        config: PipelineConfig,
        pipeline_input: PipelineInput,
    ) -> PortfolioReportStage:

        _ = config
        _ = pipeline_input

        return PortfolioReportStage(
            metadata=metadata
        )

# ============================================================
# CONTEXT INTEGRATION
# ============================================================


def run_report_stage(
    *,
    context:
    PipelineContext,

    stage:
    PortfolioReportStage,

    forecast_output:
    ForecastStageOutput
    | None,

    risk_output:
    RiskStageOutput
    | None,

    constraint_output:
    ConstraintStageOutput
    | None,

    optimization_output:
    OptimizationStageOutput
    | None,

    portfolio_output:
    PortfolioBuildStageOutput
    | None,

    rebalance_output:
    RebalanceStageOutput
    | None,

    execution_output:
    ExecutionStageOutput
    | None,

    diagnostics_output:
    DiagnosticsStageOutput
    | None,
) -> PipelineStageOutput:
    """
    Execute reporting stage.
    """

    output = (
        stage.run(

            forecast_output=
            forecast_output,

            risk_output=
            risk_output,

            constraint_output=
            constraint_output,

            optimization_output=
            optimization_output,

            portfolio_output=
            portfolio_output,

            rebalance_output=
            rebalance_output,

            execution_output=
            execution_output,

            diagnostics_output=
            diagnostics_output,
        )
    )

    context.shared_objects[
        "report_output"
    ] = output

    return output



# ============================================================
# PART 13 — MASTER PIPELINE ENGINE
# ============================================================

import time


# ============================================================
# PIPELINE CONFIG
# ============================================================


@dataclass(slots=True)
class InstitutionalPipelineConfig:
    """
    Master pipeline switches.
    """

    run_forecast: bool = True
    run_risk: bool = True
    run_constraints: bool = True
    run_optimization: bool = True
    run_portfolio_build: bool = True
    run_rebalance: bool = True
    run_execution: bool = True

    run_analytics: bool = True
    run_attribution: bool = True
    run_stress_testing: bool = True
    run_monitoring: bool = True

    run_diagnostics: bool = True
    run_reporting: bool = True

# ============================================================
# PIPELINE RUN STATS
# ============================================================


@dataclass(slots=True)
class PipelineRuntimeStats:
    """
    Runtime monitoring.
    """

    started_at: float

    completed_at: float

    runtime_seconds: float

    stage_runtimes: dict[str, float] = field(
        default_factory=dict
    )


# ============================================================
# PIPELINE RESULT
# ============================================================


@dataclass(slots=True)
class InstitutionalPipelineResult:
    """
    Final pipeline output.
    """

    report: InstitutionalPortfolioConstructionReport

    context: PipelineContext

    runtime: PipelineRuntimeStats

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )

    status: str = "COMPLETED"

    message: str = (
        "Institutional portfolio construction "
        "pipeline completed successfully."
    )


# ============================================================
# MASTER PIPELINE ENGINE
# ============================================================


class InstitutionalPortfolioPipeline:
    """
    Institutional-grade portfolio
    construction orchestrator.
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        metadata:
        PortfolioBuilderMetadata,

        config:
        InstitutionalPipelineConfig
        | None = None,
    ) -> None:

        self.metadata = metadata

        self.config = (
            config
            if config is not None
            else InstitutionalPipelineConfig()
        )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    def run_forecast_stage(
        self,
        *,
        context:
        PipelineContext,

        inputs:
        PipelineInput,
    ) -> (
        ForecastStageOutput
        | None
    ):

        if not (
            self.config
            .run_forecast
        ):

            return None

        stage = ForecastStageFactory.create(
            metadata=context.metadata,
            config=context.config,
            pipeline_input=context.shared_objects["pipeline_input"],
        )

        return (
            run_forecast_stage(

                context=
                context,

                inputs=
                inputs,

                stage=
                stage,
            )
        )

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    def run_risk_stage(
        self,
        *,
        context: PipelineContext,
        inputs: PipelineInput,
        forecast_output: ForecastStageOutput | None,
    ) -> RiskStageOutput | None:

        if not self.config.run_risk:
            return None

        stage = RiskStageFactory.create(
            metadata=context.metadata,
            config=context.config,
            pipeline_input=context.shared_objects["pipeline_input"],
        )

        return run_risk_stage(
            context=context,
            inputs=inputs,
            stage=stage,
            forecast_output=forecast_output,
        )

    # --------------------------------------------------------
    # CONSTRAINTS
    # --------------------------------------------------------

    def run_constraint_stage(
        self,
        *,
        context: PipelineContext,
        inputs: PipelineInput,
        forecast_output: ForecastStageOutput | None,
        risk_output: RiskStageOutput | None,
    ) -> ( ConstraintStageOutput | None ):

        if not (
            self.config
            .run_constraints
        ):

            return None

        stage = ConstraintStageFactory.create(
            metadata=context.metadata,
            config=context.config,
            pipeline_input=context.shared_objects["pipeline_input"],
        )

        return run_constraint_stage(
            context=context,
            inputs=inputs,
            stage=stage,
            forecast_output=forecast_output,
            risk_output=risk_output,
        )

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    def run_optimization_stage(
        self,
        *,
        context: PipelineContext,
        inputs: PipelineInput,
        forecast_output: ForecastStageOutput | None,
        risk_output: RiskStageOutput | None,
        constraint_output: ConstraintStageOutput | None,) -> ( OptimizationStageOutput | None
    ):

        if not (
            self.config
            .run_optimization
        ):

            return None

        stage = OptimizationStageFactory.create(
            metadata=context.metadata,
            config=context.config,
            pipeline_input=context.shared_objects["pipeline_input"],
        )

        return (
            run_optimization_stage(

                context= context,
                inputs= inputs,
                stage= stage,
                forecast_output= forecast_output,
                risk_output= risk_output,
                constraint_output= constraint_output,
            )
        )

    # --------------------------------------------------------
    # PORTFOLIO BUILD
    # --------------------------------------------------------

    def run_portfolio_stage(
        self,
        *,
        context: PipelineContext,
        inputs: PipelineInput,
        optimization_output: OptimizationStageOutput | None,
    ) -> ( PortfolioBuildStageOutput| None):

        if not (
            self.config
            .run_portfolio_build
        ):

            return None

        stage = PortfolioBuildStageFactory.create(
                    metadata=context.metadata,
                    config=context.config,
                    pipeline_input=context.shared_objects["pipeline_input"],
                )

        return (
            run_portfolio_build_stage(
                context= context,
                inputs= inputs,
                stage= stage,
                optimization_output= optimization_output,
            )
        )

    # --------------------------------------------------------
    # REBALANCE
    # --------------------------------------------------------

    def run_rebalance_stage(
        self,
        *,
        context: PipelineContext,
        inputs: PipelineInput,
        portfolio_output: PortfolioBuildStageOutput | None,
    ) -> RebalanceStageOutput | None:

        if not (
            self.config
            .run_rebalance
        ):

            return None

        stage = RebalanceStageFactory.create(
            metadata=context.metadata,
            config=context.config,
            pipeline_input=context.shared_objects["pipeline_input"],
        )

        return (
            run_rebalance_stage(
                context= context,
                inputs= inputs,
                stage= stage,
                portfolio_output= portfolio_output,
            )
        )

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    def run_execution_stage(
        self,
        *,
        context: PipelineContext,
        rebalance_output: RebalanceStageOutput | None,
    ) -> ExecutionStageOutput | None:

        if not (
            self.config
            .run_execution
        ):

            return None

        stage = ExecutionStageFactory.create(
            metadata=context.metadata,
            config=context.config,
            pipeline_input=context.shared_objects["pipeline_input"],
        )

        return (
            run_execution_stage(

                context=
                context,

                stage=
                stage,

                rebalance_output=
                rebalance_output,
            )
        )

    # --------------------------------------------------------
    # ANALYTICS
    # --------------------------------------------------------

    def run_analytics_stage(
        self,
        *,
        context: PipelineContext,
        inputs: PipelineInput,
        portfolio_output: PortfolioBuildStageOutput | None,
        rebalance_output: RebalanceStageOutput | None,
        execution_output: ExecutionStageOutput | None,
    ) -> Any:

        if not self.config.run_analytics:
            return None

        start = time.perf_counter()

        try:

            # ----------------------------------
            # Validate portfolio output
            # ----------------------------------

            if (
                portfolio_output is None
                or portfolio_output.result is None
            ):
                raise ValueError(
                    "Analytics requires a valid portfolio output."
                )

            portfolio_result = (
                portfolio_output.result
            )

            # ----------------------------------
            # Build analytics metadata
            # ----------------------------------

            analytics_metadata = AnalyticsMetadata(
                generated_at=datetime.now(timezone.utc),

                portfolio_name=(
                    getattr(
                        self.metadata,
                        "strategy_name",
                        None,
                    )
                    or "Institutional Portfolio"
                ),

                benchmark_name=(
                    getattr(
                        self.metadata,
                        "benchmark_name",
                        None,
                    )
                    or "NIFTY50"
                ),

                strategy_name=(
                    getattr(
                        self.metadata,
                        "strategy_name",
                        None,
                    )
                    or "StockPredictionV1"
                ),

                universe_name=(
                    getattr(
                        self.metadata,
                        "universe_name",
                        None,
                    )
                    or "NSE500"
                ),
            )

            # ----------------------------------
            # Portfolio DataFrame
            # ----------------------------------

            portfolio = self._build_analytics_portfolio(
                inputs=inputs,
                portfolio_result=portfolio_result,
            )

            # ----------------------------------
            # Portfolio returns
            # ----------------------------------

            portfolio_returns = (
                self._build_analytics_portfolio_returns(
                    inputs=inputs,
                    portfolio=portfolio,
                )
            )

            # ----------------------------------
            # Benchmark returns
            # ----------------------------------

            benchmark_returns = (
                self._build_analytics_benchmark_returns(
                    inputs=inputs,
                )
            )

            # ----------------------------------
            # Execution DataFrame
            # ----------------------------------

            execution_df = (
                self._build_analytics_execution_df(
                    execution_output=execution_output,
                )
            )

            # ----------------------------------
            # Rebalance series
            # ----------------------------------

            turnover_series = (
                self._build_analytics_turnover_series(
                    rebalance_output=rebalance_output,
                )
            )

            trade_count_series = (
                self._build_analytics_trade_count_series(
                    rebalance_output=rebalance_output,
                )
            )

            # ----------------------------------
            # Factor exposures
            # ----------------------------------

            factor_exposures = (
                self._build_analytics_factor_exposures(
                    inputs=inputs,
                    portfolio=portfolio,
                )
            )

            # ----------------------------------
            # Run Analytics Engine
            # ----------------------------------

            engine = AnalyticsEngine(
                metadata=analytics_metadata,
            )

            result = engine.run_all(
                portfolio=portfolio,
                portfolio_returns=portfolio_returns,
                benchmark_returns=benchmark_returns,
                execution_df=execution_df,
                turnover_series=turnover_series,
                trade_count_series=trade_count_series,
                factor_exposures=factor_exposures,
            )

            # ----------------------------------
            # Diagnostics
            # ----------------------------------

            context.shared_objects[
                "analytics_result"
            ] = result

            context.shared_objects[
                "analytics_runtime_seconds"
            ] = (
                time.perf_counter()
                - start
            )

            return result

        except Exception as exc:

            logger.exception(
                "Analytics stage failed: %s",
                exc,
            )

            context.shared_objects[
                "analytics_error"
            ] = str(exc)

            context.shared_objects[
                "analytics_runtime_seconds"
            ] = (
                time.perf_counter()
                - start
            )

            return None

    # --------------------------------

    def _build_analytics_portfolio(
        self,
        *,
        inputs: PipelineInput,
        portfolio_result: Any,
    ) -> pd.DataFrame:

        weights = (
            portfolio_result
            .weights
            .copy()
            .astype(float)
        )

        if weights.empty:
            raise ValueError(
                "Analytics portfolio contains no weights."
            )

        prices = inputs.market_data.prices

        if prices is None or prices.empty:
            raise ValueError(
                "Market prices unavailable for analytics."
            )

        latest_prices = (
            prices
            .sort_index()
            .iloc[-1]
        )

        portfolio = pd.DataFrame({
            "Ticker": weights.index.astype(str),
            "Position_Weight": weights.values,
        })

        portfolio["Close"] = (
            portfolio["Ticker"]
            .map(latest_prices)
        )

        portfolio["Market_Value"] = (
            portfolio["Position_Weight"]
            .abs()
            *
            portfolio["Close"].fillna(0.0)
        )

        volumes = (
            inputs.market_data.volumes
        )

        if (
            volumes is not None
            and not volumes.empty
        ):

            latest_volume = (
                volumes
                .sort_index()
                .iloc[-1]
            )

            portfolio["ADV"] = (
                portfolio["Ticker"]
                .map(latest_volume)
                *
                portfolio["Close"]
            )

        else:

            portfolio["ADV"] = 0.0

        return portfolio

    # --------------------------------

    def _build_analytics_portfolio_returns(
        self,
        *,
        inputs: PipelineInput,
        portfolio: pd.DataFrame,
    ) -> pd.Series:

        returns = inputs.market_data.returns

        if returns is None or returns.empty:
            return pd.Series(dtype=float)

        weights = (
            portfolio
            .set_index("Ticker")[
                "Position_Weight"
            ]
        )

        common_assets = (
            returns.columns
            .intersection(weights.index)
        )

        if len(common_assets) == 0:
            return pd.Series(dtype=float)

        aligned_returns = (
            returns[
                common_assets
            ]
            .copy()
        )

        aligned_weights = (
            weights
            .reindex(common_assets)
            .fillna(0.0)
        )

        portfolio_returns = (
            aligned_returns
            .mul(aligned_weights, axis=1)
            .sum(axis=1)
        )

        return (
            portfolio_returns
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )


    def _build_analytics_benchmark_returns(
        self,
        *,
        inputs: PipelineInput,
    ) -> pd.Series | None:

        benchmark_returns = (
            inputs.market_data.benchmark_returns
        )

        if (
            benchmark_returns is None
            or benchmark_returns.empty
        ):
            return None

        if isinstance(
            benchmark_returns,
            pd.Series,
        ):
            return (
                benchmark_returns
                .astype(float)
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
            )

        if (
            isinstance(
                benchmark_returns,
                pd.DataFrame,
            )
            and not benchmark_returns.empty
        ):

            # Single benchmark column
            if len(
                benchmark_returns.columns
            ) == 1:

                return (
                    benchmark_returns.iloc[:, 0]
                    .astype(float)
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                    .dropna()
                )

            # Prefer benchmark identified
            # in metadata if available.
            benchmark_name = (
                getattr(
                    self.metadata,
                    "benchmark_name",
                    None,
                )
                or getattr(
                    self.metadata,
                    "benchmark",
                    None,
                )
            )

            if (
                benchmark_name is not None
                and benchmark_name
                in benchmark_returns.columns
            ):

                return (
                    benchmark_returns[
                        benchmark_name
                    ]
                    .astype(float)
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                    .dropna()
                )

        return None

    # --------------------------------
    # Analytics Execution DataFrame
    # --------------------------------

    def _build_analytics_execution_df(
        self,
        *,
        execution_output: ExecutionStageOutput | None,
    ) -> pd.DataFrame | None:

        if (
            execution_output is None
            or execution_output.result is None
        ):
            return None

        execution = execution_output.result

        orders = getattr(
            execution,
            "orders",
            None,
        )

        if not orders:
            return None

        rows = []

        for order in orders:

            side = getattr(
                order,
                "side",
                None,
            )

            side_value = getattr(
                side,
                "value",
                side,
            )

            rows.append(
                {
                    "Asset": getattr(
                        order,
                        "asset_id",
                        None,
                    ),

                    "Side": side_value,

                    "Quantity": float(
                        getattr(
                            order,
                            "target_weight_change",
                            0.0,
                        )
                    ),

                    "Slippage": float(
                        getattr(
                            order,
                            "estimated_slippage",
                            0.0,
                        )
                    ),

                    "TransactionCost": float(
                        getattr(
                            order,
                            "estimated_cost",
                            0.0,
                        )
                    ),

                    "MarketImpact": float(
                        getattr(
                            order,
                            "estimated_market_impact",
                            0.0,
                        )
                    ),

                    "ExpectedFillPrice": getattr(
                        order,
                        "expected_fill_price",
                        None,
                    ),
                }
            )

        if not rows:
            return None

        return pd.DataFrame(rows)

    # --------------------------------
    # Analytics Rebalance Series
    # --------------------------------

    def _build_analytics_turnover_series(
        self,
        *,
        rebalance_output: RebalanceStageOutput | None,
    ) -> pd.Series | None:
        """
        Build turnover time series for AnalyticsEngine.

        The current institutional rebalance stage produces a single
        rebalance snapshot per pipeline execution, so the resulting
        series contains one observation indexed by the pipeline
        generation timestamp.
        """

        if (
            rebalance_output is None
            or rebalance_output.result is None
        ):
            return None

        rebalance = rebalance_output.result

        turnover = getattr(
            rebalance,
            "turnover",
            None,
        )

        if turnover is None:

            diagnostics = getattr(
                rebalance,
                "diagnostics",
                None,
            )

            if isinstance(diagnostics, dict):

                turnover = diagnostics.get(
                    "turnover",
                    0.0,
                )

            else:

                turnover = getattr(
                    diagnostics,
                    "turnover",
                    0.0,
                )

        try:

            turnover = float(turnover)

        except (
            TypeError,
            ValueError,
        ):

            turnover = 0.0

        timestamp = pd.Timestamp.now(
            tz="UTC"
        )

        return pd.Series(
            [turnover],
            index=pd.DatetimeIndex(
                [timestamp],
                name="Date",
            ),
            name="Turnover",
            dtype=float,
        )

    # --------------------------------
    # Analytics Trade Count Series
    # --------------------------------

    def _build_analytics_trade_count_series(
        self,
        *,
        rebalance_output: RebalanceStageOutput | None,
    ) -> pd.Series | None:
        """
        Build trade-count time series for AnalyticsEngine.

        The current institutional rebalance stage produces one
        rebalance snapshot per pipeline execution.
        """

        if (
            rebalance_output is None
            or rebalance_output.result is None
        ):
            return None

        rebalance = rebalance_output.result

        diagnostics = getattr(
            rebalance,
            "diagnostics",
            None,
        )

        trade_count = None

        if isinstance(diagnostics, dict):

            trade_count = diagnostics.get(
                "trade_count",
                None,
            )

        else:

            trade_count = getattr(
                diagnostics,
                "trade_count",
                None,
            )

        if trade_count is None:

            trades = getattr(
                rebalance,
                "trades",
                None,
            )

            trade_count = (
                len(trades)
                if trades is not None
                else 0
            )

        try:

            trade_count = int(
                trade_count
            )

        except (
            TypeError,
            ValueError,
        ):

            trade_count = 0

        timestamp = pd.Timestamp.now(
            tz="UTC"
        )

        return pd.Series(
            [trade_count],
            index=pd.DatetimeIndex(
                [timestamp],
                name="Date",
            ),
            name="TradeCount",
            dtype=int,
        )

    # --------------------------------
    # Analytics Factor Exposures
    # --------------------------------

    def _build_analytics_factor_exposures(
        self,
        *,
        inputs: PipelineInput,
        portfolio: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """
        Build factor exposures for AnalyticsEngine.

        Factor exposures are optional at the current pipeline stage.
        If factor exposure data is not available through PipelineInput,
        return None rather than fabricating exposures.

        Parameters
        ----------
        inputs:
            Institutional PipelineInput.

        portfolio:
            Analytics portfolio DataFrame.

        Returns
        -------
        pd.DataFrame | None
            Factor exposure DataFrame when available, otherwise None.
        """

        # ----------------------------------
        # 1. Check PipelineInput directly
        # ----------------------------------

        factor_exposures = getattr(
            inputs,
            "factor_exposures",
            None,
        )

        if (
            factor_exposures is not None
            and isinstance(
                factor_exposures,
                pd.DataFrame,
            )
            and not factor_exposures.empty
        ):
            return factor_exposures.copy()

        # ----------------------------------
        # 2. Check nested market-data object
        # ----------------------------------

        market_data = getattr(
            inputs,
            "market_data",
            None,
        )

        if market_data is not None:

            factor_exposures = getattr(
                market_data,
                "factor_exposures",
                None,
            )

            if (
                factor_exposures is not None
                and isinstance(
                    factor_exposures,
                    pd.DataFrame,
                )
                and not factor_exposures.empty
            ):
                return factor_exposures.copy()

        # ----------------------------------
        # 3. No factor exposures available
        # ----------------------------------

        logger.debug(
            "Analytics factor exposures unavailable; "
            "continuing without factor exposures."
        )

        return None

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------

    def run_diagnostics_stage(
        self,
        *,
        context:
        PipelineContext,

        portfolio_output:
        PortfolioBuildStageOutput
        | None,

        rebalance_output:
        RebalanceStageOutput
        | None,

        execution_output:
        ExecutionStageOutput
        | None,

        analytics_result:
        Any = None,

        attribution_result:
        Any = None,

        stress_result:
        Any = None,
    ) -> (
        DiagnosticsStageOutput
        | None
    ):

        if not (
            self.config
            .run_diagnostics
        ):

            return None

        stage = DiagnosticsStageFactory.create(
            metadata=context.metadata,
            config=context.config,
            pipeline_input=context.shared_objects["pipeline_input"],
        )

        return (
            run_diagnostics_stage(

                context=
                context,

                stage=
                stage,

                portfolio_output=
                portfolio_output,

                rebalance_output=
                rebalance_output,

                execution_output=
                execution_output,

                analytics_result=
                analytics_result,

                attribution_result=
                attribution_result,

                stress_result=
                stress_result,
            )
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    def run_report_stage(
        self,
        *,
        context:
        PipelineContext,

        forecast_output:
        ForecastStageOutput
        | None,

        risk_output:
        RiskStageOutput
        | None,

        constraint_output:
        ConstraintStageOutput
        | None,

        optimization_output:
        OptimizationStageOutput
        | None,

        portfolio_output:
        PortfolioBuildStageOutput
        | None,

        rebalance_output:
        RebalanceStageOutput
        | None,

        execution_output:
        ExecutionStageOutput
        | None,

        diagnostics_output:
        DiagnosticsStageOutput
        | None,
    ) -> (
        PipelineStageOutput
        | None
    ):

        if not (
            self.config
            .run_reporting
        ):

            return None

        stage = PortfolioReportStageFactory.create(
            metadata=context.metadata,
            config=context.config,
            pipeline_input=context.shared_objects["pipeline_input"],
        )

        return (
            run_report_stage(

                context=
                context,

                stage=
                stage,

                forecast_output=
                forecast_output,

                risk_output=
                risk_output,

                constraint_output=
                constraint_output,

                optimization_output=
                optimization_output,

                portfolio_output=
                portfolio_output,

                rebalance_output=
                rebalance_output,

                execution_output=
                execution_output,

                diagnostics_output=
                diagnostics_output,
            )
        )

    # --------------------------------------------------------
    # MASTER RUN
    # --------------------------------------------------------

    def run(
        self,
        *,
        inputs: PipelineInput,
        analytics_result: Any = None,
        attribution_result: Any = None,
        stress_result: Any = None,
    ) -> (
        InstitutionalPipelineResult
    ):

        started_at = (
            time.perf_counter()
        )

        context = PipelineContext(
            metadata=self.metadata,
            config=self.config
        )

        context.shared_objects.update({
            "pipeline_input": inputs,
            "analytics_result": analytics_result,
            "attribution_result": attribution_result,
            "stress_result": stress_result,
        })

        # ----------------------------------
        # Forecast
        # ----------------------------------

        forecast_output = (
            self.run_forecast_stage(

                context=
                context,

                inputs=
                inputs,
            )
        )

        # ----------------------------------
        # Risk
        # ----------------------------------

        risk_output = self.run_risk_stage(
            context=context,
            inputs=inputs,
            forecast_output=forecast_output,
        )

        # ----------------------------------
        # Constraints
        # ----------------------------------

        constraint_output = (
            self.run_constraint_stage(
                context= context,
                inputs= inputs,
                forecast_output= forecast_output,
                risk_output= risk_output,
            )
        )

        # ----------------------------------
        # Optimization
        # ----------------------------------

        optimization_output = (
            self.run_optimization_stage(
                context= context,
                inputs= inputs,
                forecast_output= forecast_output,
                risk_output= risk_output,
                constraint_output= constraint_output,
            )
        )

        # ----------------------------------
        # Portfolio
        # ----------------------------------

        portfolio_output = (
            self.run_portfolio_stage(
                context= context,
                inputs= inputs,
                optimization_output= optimization_output,
            )
        )

        # ----------------------------------
        # Rebalance
        # ----------------------------------

        rebalance_output = (
            self.run_rebalance_stage(
                context= context,
                inputs= inputs,
                portfolio_output= portfolio_output,
            )
        )

        # ----------------------------------
        # Execution
        # ----------------------------------

        execution_output = self.run_execution_stage(
            context=context,
            rebalance_output=rebalance_output,
        )

        # ----------------------------------
        # Analytics
        # ----------------------------------

        analytics_result = self.run_analytics_stage(
            context=context,
            inputs=inputs,
            portfolio_output=portfolio_output,
            rebalance_output=rebalance_output,
            execution_output=execution_output,
        )

        context.shared_objects["analytics_result"] = analytics_result

        # ----------------------------------
        # Diagnostics
        # ----------------------------------

        diagnostics_output = (
            self.run_diagnostics_stage(
                context= context,
                portfolio_output= portfolio_output,
                rebalance_output= rebalance_output,
                execution_output= execution_output,
                analytics_result= analytics_result,
                attribution_result= attribution_result,
                stress_result= stress_result,
            )
        )

        # ----------------------------------
        # Report
        # ----------------------------------

        report_output = (
            self.run_report_stage(

                context=
                context,

                forecast_output=
                forecast_output,

                risk_output=
                risk_output,

                constraint_output=
                constraint_output,

                optimization_output=
                optimization_output,

                portfolio_output=
                portfolio_output,

                rebalance_output=
                rebalance_output,

                execution_output=
                execution_output,

                diagnostics_output=
                diagnostics_output,
            )
        )

        completed_at = (
            time.perf_counter()
        )

        runtime = (
            PipelineRuntimeStats(

                started_at=
                started_at,

                completed_at=
                completed_at,

                runtime_seconds=
                completed_at
                - started_at,
            )
        )

        return InstitutionalPipelineResult(
            report=(
                report_output.payload
                if report_output is not None
                else None
            ),
            context=context,
            runtime=runtime,
            diagnostics={
                "report_stage":
                (
                    report_output.diagnostics
                    if report_output is not None
                    else {}
                )
            },
            status=(
                PipelineStatus.COMPLETED.name
                if report_output is not None
                and report_output.payload is not None
                else PipelineStatus.FAILED.name
            ),
            message=(
                "Pipeline completed."
                if report_output is not None
                and report_output.payload is not None
                else "Reporting stage failed."
            ),
        )
    

# ============================================================
# PART 14 — FACTORY & CONVENIENCE APIS
# ============================================================

from typing import Optional


# ============================================================
# PIPELINE FACTORY
# ============================================================


class InstitutionalPipelineFactory:
    """
    Central factory for creating
    institutional portfolio pipelines.
    """

    # --------------------------------------------------------

    @staticmethod
    def create(
        *,
        metadata:
        PortfolioBuilderMetadata,

        config:
        InstitutionalPipelineConfig
        | None = None,
    ) -> InstitutionalPortfolioPipeline:
        """
        Create fully configured pipeline.
        """

        return (
            InstitutionalPortfolioPipeline(

                metadata=
                metadata,

                config=
                config,
            )
        )

    # --------------------------------------------------------

    @staticmethod
    def create_default(
        *,
        metadata:
        PortfolioBuilderMetadata,
    ) -> InstitutionalPortfolioPipeline:
        """
        Create pipeline using
        default configuration.
        """

        return (
            InstitutionalPortfolioPipeline(

                metadata=
                metadata,

                config=
                InstitutionalPipelineConfig(),
            )
        )


# ============================================================
# CONVENIENCE API
# create_pipeline()
# ============================================================


def create_pipeline(
    *,
    metadata:
    PortfolioBuilderMetadata,

    config:
    InstitutionalPipelineConfig
    | None = None,
) -> InstitutionalPortfolioPipeline:
    """
    Public factory.

    Example
    -------
    pipeline = create_pipeline(
        metadata=metadata
    )
    """

    return (
        InstitutionalPipelineFactory
        .create(

            metadata=
            metadata,

            config=
            config,
        )
    )


# ============================================================
# CONVENIENCE API
# run_pipeline()
# ============================================================


def run_pipeline(
    *,
    inputs:
    PipelineInput,

    metadata:
    PortfolioBuilderMetadata,

    config:
    InstitutionalPipelineConfig
    | None = None,

    analytics_result:
    Any = None,

    attribution_result:
    Any = None,

    stress_result:
    Any = None,
) -> InstitutionalPipelineResult:
    """
    Execute complete institutional pipeline.

    Example
    -------
    result = run_pipeline(
        inputs=inputs,
        metadata=metadata,
    )
    """

    pipeline = (
        create_pipeline(

            metadata=
            metadata,

            config=
            config,
        )
    )

    return (
        pipeline.run(

            inputs=
            inputs,

            analytics_result=
            analytics_result,

            attribution_result=
            attribution_result,

            stress_result=
            stress_result,
        )
    )


# ============================================================
# CONVENIENCE API
# institutional_pipeline()
# ============================================================


def institutional_pipeline(
    *,
    inputs:
    PipelineInput,

    metadata:
    PortfolioBuilderMetadata,
) -> InstitutionalPipelineResult:
    """
    Simplest institutional API.

    Example
    -------
    result = institutional_pipeline(
        inputs=inputs,
        metadata=metadata,
    )
    """

    return (
        run_pipeline(

            inputs=
            inputs,

            metadata=
            metadata,
        )
    )


# ============================================================
# CONVENIENCE API
# build_portfolio()
# ============================================================


def build_portfolio(
    *,
    inputs:
    PipelineInput,

    metadata:
    PortfolioBuilderMetadata,
) -> InstitutionalPortfolio:
    """
    Return only portfolio object.

    Example
    -------
    portfolio = build_portfolio(
        inputs=inputs,
        metadata=metadata,
    )
    """

    result = (
        institutional_pipeline(

            inputs=
            inputs,

            metadata=
            metadata,
        )
    )

    report = result.report

    if report is None:

        raise RuntimeError(
            "Pipeline failed."
        )

    return (
        report
        .portfolio_result
    )


# ============================================================
# CONVENIENCE API
# build_rebalance()
# ============================================================


def build_rebalance(
    *,
    inputs:
    PipelineInput,

    metadata:
    PortfolioBuilderMetadata,
) -> InstitutionalRebalance:
    """
    Return rebalance package only.
    """

    result = (
        institutional_pipeline(

            inputs=
            inputs,

            metadata=
            metadata,
        )
    )

    report = result.report

    if report is None:

        raise RuntimeError(
            "Pipeline failed."
        )

    return (
        report
        .rebalance_result
    )


# ============================================================
# CONVENIENCE API
# build_execution()
# ============================================================


def build_execution(
    *,
    inputs:
    PipelineInput,

    metadata:
    PortfolioBuilderMetadata,
) -> InstitutionalExecutionPackage:
    """
    Return execution package only.
    """

    result = (
        institutional_pipeline(

            inputs=
            inputs,

            metadata=
            metadata,
        )
    )

    report = result.report

    if report is None:

        raise RuntimeError(
            "Pipeline failed."
        )

    execution = (
        report
        .diagnostics
        .get(
            "execution"
        )
    )

    if execution is None:

        raise RuntimeError(
            "Execution unavailable."
        )

    return execution


# ============================================================
# CONVENIENCE API
# diagnostics_report()
# ============================================================


def diagnostics_report(
    *,
    inputs:
    PipelineInput,

    metadata:
    PortfolioBuilderMetadata,

    analytics_result:
    Any = None,

    attribution_result:
    Any = None,

    stress_result:
    Any = None,
) -> InstitutionalDiagnosticsPackage:
    """
    Return diagnostics package only.
    """

    result = (
        run_pipeline(

            inputs=
            inputs,

            metadata=
            metadata,

            analytics_result=
            analytics_result,

            attribution_result=
            attribution_result,

            stress_result=
            stress_result,
        )
    )

    report = result.report

    if report is None:

        raise RuntimeError(
            "Pipeline failed."
        )

    diagnostics = (
        report
        .diagnostics
        .get(
            "diagnostics"
        )
    )

    if diagnostics is None:

        raise RuntimeError(
            "Diagnostics unavailable."
        )

    return diagnostics


# ============================================================
# CONVENIENCE API
# full_report()
# ============================================================


def full_report(
    *,
    inputs:
    PipelineInput,

    metadata:
    PortfolioBuilderMetadata,

    analytics_result:
    Any = None,

    attribution_result:
    Any = None,

    stress_result:
    Any = None,
) -> InstitutionalPortfolioConstructionReport:
    """
    Return institutional report.

    Example
    -------
    report = full_report(
        inputs=inputs,
        metadata=metadata,
    )
    """

    result = (
        run_pipeline(

            inputs=
            inputs,

            metadata=
            metadata,

            analytics_result=
            analytics_result,

            attribution_result=
            attribution_result,

            stress_result=
            stress_result,
        )
    )

    if result.report is None:

        raise RuntimeError(
            "Report unavailable."
        )

    return result.report


# ============================================================
# QUICK TEST HELPER
# ============================================================


def smoke_test_pipeline(
    *,
    inputs:
    PipelineInput,

    metadata:
    PortfolioBuilderMetadata,
) -> bool:
    """
    Simple pipeline validation.

    Returns
    -------
    bool
    """

    try:

        result = (
            institutional_pipeline(

                inputs=
                inputs,

                metadata=
                metadata,
            )
        )

        return (
            result.report
            is not None
        )

    except Exception:

        return False