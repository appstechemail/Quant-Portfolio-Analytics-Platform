"""
============================================================
stress_testing.py

Institutional Grade Quant Platform

PART 1
Framework & Core Objects
============================================================
"""

from __future__ import annotations

# ============================================================
# STANDARD LIBRARIES
# ============================================================

from dataclasses import (
    dataclass,
    field,
    asdict,
)

from abc import (
    ABC,
    abstractmethod,
)
from enum import Enum
from datetime import (
    datetime,
)
from typing import (
    Any,
    Optional,
)

# ============================================================
# THIRD PARTY
# ============================================================

import numpy as np
import pandas as pd
import json


# ============================================================
# GLOBAL CONSTANTS
# ============================================================

EPSILON = 1e-12

TRADING_DAYS = 252

DEFAULT_CONFIDENCE_LEVEL = 0.95

DEFAULT_MONTE_CARLO_PATHS = 10000

DEFAULT_MONTE_CARLO_HORIZON = 252

# ============================================================
# STRESS TEST TYPES
# ============================================================

class StressTestType(
    str,
    Enum,
):

    HISTORICAL = "historical"

    SCENARIO = "scenario"

    FACTOR_SHOCK = "factor_shock"

    CORRELATION_BREAKDOWN = (
        "correlation_breakdown"
    )

    LIQUIDITY = "liquidity"

    VOLATILITY_REGIME = (
        "volatility_regime"
    )

    MONTE_CARLO = "monte_carlo"

    TAIL_RISK = "tail_risk"

    REVERSE_STRESS = (
        "reverse_stress"
    )


# ============================================================
# SCENARIO SEVERITY
# ============================================================

class ScenarioSeverity(
    str,
    Enum,
):

    MILD = "mild"

    MODERATE = "moderate"

    SEVERE = "severe"

    EXTREME = "extreme"


# ============================================================
# STRESS RESULT STATUS
# ============================================================

class StressResultStatus(
    str,
    Enum,
):

    PASSED = "passed"

    WARNING = "warning"

    FAILED = "failed"


# ============================================================
# STRESS CONFIG
# ============================================================

@dataclass(slots=True)
class StressTestingConfig:
    """
    Global stress testing configuration.
    """

    confidence_level: float = (
        DEFAULT_CONFIDENCE_LEVEL
    )

    monte_carlo_paths: int = (
        DEFAULT_MONTE_CARLO_PATHS
    )

    monte_carlo_horizon: int = (
        DEFAULT_MONTE_CARLO_HORIZON
    )

    liquidity_haircut: float = 0.20

    volatility_multiplier: float = 2.0

    correlation_breakdown_level: float = (
        0.90
    )

    tail_percentile: float = 0.01

    reporting_enabled: bool = True


# ============================================================
# METADATA
# ============================================================

@dataclass(slots=True)
class StressMetadata:
    """
    Shared metadata across all stress tests.
    """

    portfolio_name: str

    benchmark_name: str | None = None

    run_timestamp: datetime = field(
        default_factory=datetime.utcnow
    )

    notes: str | None = None


# ============================================================
# BASE RESULT
# ============================================================

@dataclass(slots=True)
class StressTestResult:
    """
    Base stress test result object.
    """

    metadata: StressMetadata

    stress_type: StressTestType

    scenario_name: str

    severity: ScenarioSeverity

    status: StressResultStatus

    portfolio_return: float

    portfolio_pnl: float

    drawdown: float

    var_impact: float

    volatility_impact: float

    diagnostics: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )


# ============================================================
# SCENARIO DEFINITION
# ============================================================

@dataclass(slots=True)
class StressScenario:
    """
    Generic stress scenario definition.
    """

    scenario_name: str

    severity: ScenarioSeverity

    description: str

    factor_shocks: dict[
        str,
        float,
    ] = field(
        default_factory=dict
    )

    volatility_multiplier: float = 1.0

    correlation_shift: float = 0.0

    liquidity_haircut: float = 0.0


# ============================================================
# VALIDATION UTILITIES
# ============================================================

class StressValidation:

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
                f"{name} cannot be empty."
            )

    # --------------------------------------------------------

    @staticmethod
    def validate_series(
        series: pd.Series,
        name: str,
    ) -> None:

        if not isinstance(
            series,
            pd.Series,
        ):

            raise TypeError(
                f"{name} must be Series."
            )

        if series.empty:

            raise ValueError(
                f"{name} cannot be empty."
            )

    # --------------------------------------------------------

    @staticmethod
    def validate_columns(
        df: pd.DataFrame,
        required: list[str],
        name: str,
    ) -> None:

        missing = [

            c

            for c in required

            if c not in df.columns

        ]

        if missing:

            raise ValueError(

                f"{name} missing columns: "
                f"{missing}"

            )

    # --------------------------------------------------------

    @staticmethod
    def validate_same_length(
        *arrays,
    ) -> None:

        lengths = [

            len(x)

            for x in arrays

        ]

        if len(
            set(lengths)
        ) > 1:

            raise ValueError(
                "Inputs have unequal lengths."
            )


# ============================================================
# STRESS UTILITIES
# ============================================================

class StressUtils:

    # --------------------------------------------------------

    @staticmethod
    def annualize_volatility(
        returns: pd.Series,
    ) -> float:

        StressValidation.validate_series(
            returns,
            "returns",
        )

        return float(

            returns.std()

            * np.sqrt(
                TRADING_DAYS
            )

        )

    # --------------------------------------------------------

    @staticmethod
    def cumulative_return(
        returns: pd.Series,
    ) -> float:

        StressValidation.validate_series(
            returns,
            "returns",
        )

        return float(

            np.prod(
                1.0 + returns
            )

            - 1.0

        )

    # --------------------------------------------------------

    @staticmethod
    def max_drawdown(
        returns: pd.Series,
    ) -> float:

        StressValidation.validate_series(
            returns,
            "returns",
        )

        cumulative = (

            1.0 + returns

        ).cumprod()

        running_max = (
            cumulative.cummax()
        )

        drawdown = (

            cumulative

            / running_max

            - 1.0

        )

        return float(
            drawdown.min()
        )

    # --------------------------------------------------------

    @staticmethod
    def portfolio_return(
        weights: pd.Series,
        returns: pd.Series,
    ) -> float:

        StressValidation.validate_same_length(

            weights,
            returns,

        )

        return float(

            (
                weights
                * returns
            ).sum()

        )


# ============================================================
# BASE STRESS TEST
# ============================================================

class BaseStressTest(
    ABC,
):

    def __init__(
        self,
        metadata: StressMetadata,
        config:
        StressTestingConfig | None = None,
    ) -> None:

        self.metadata = metadata

        self.config = (

            config

            if config is not None

            else StressTestingConfig()

        )

    # --------------------------------------------------------

    @abstractmethod
    def run(
        self,
        *args,
        **kwargs,
    ) -> StressTestResult:
        """
        Execute stress test.
        """
        raise NotImplementedError
    

# ============================================================
# PART 2
# STRESS RESULT OBJECTS
# ============================================================

# ============================================================
# HISTORICAL STRESS RESULT
# ============================================================

@dataclass(slots=True)
class HistoricalStressResult(
    StressTestResult,
):
    """
    Historical event replay result.
    """

    event_name: str = ""

    start_date: pd.Timestamp | None = None

    end_date: pd.Timestamp | None = None

    benchmark_return: float = 0.0

    active_return: float = 0.0

    period_length_days: int = 0


# ============================================================
# SCENARIO STRESS RESULT
# ============================================================

@dataclass(slots=True)
class ScenarioStressResult(
    StressTestResult,
):
    """
    Hypothetical scenario stress result.
    """

    shocked_factors: dict[
        str,
        float,
    ] = field(
        default_factory=dict
    )

    factor_contributions: dict[
        str,
        float,
    ] = field(
        default_factory=dict
    )


# ============================================================
# FACTOR SHOCK RESULT
# ============================================================

@dataclass(slots=True)
class FactorShockResult(
    StressTestResult,
):
    """
    Factor shock stress result.
    """

    factor_name: str = ""

    shock_size: float = 0.0

    exposure: float = 0.0

    contribution: float = 0.0


# ============================================================
# CORRELATION BREAKDOWN RESULT
# ============================================================

@dataclass(slots=True)
class CorrelationBreakdownResult(
    StressTestResult,
):
    """
    Correlation stress result.
    """

    original_avg_correlation: float = 0.0

    stressed_avg_correlation: float = 0.0

    diversification_loss: float = 0.0


# ============================================================
# LIQUIDITY STRESS RESULT
# ============================================================

@dataclass(slots=True)
class LiquidityStressResult(
    StressTestResult,
):
    """
    Liquidity shock result.
    """

    liquidity_haircut: float = 0.0

    liquidation_cost: float = 0.0

    days_to_liquidate: float = 0.0

    liquidity_score: float = 0.0


# ============================================================
# VOLATILITY REGIME RESULT
# ============================================================

@dataclass(slots=True)
class VolatilityStressResult(
    StressTestResult,
):
    """
    Volatility regime stress.
    """

    original_volatility: float = 0.0

    stressed_volatility: float = 0.0

    volatility_multiplier: float = 1.0


# ============================================================
# MONTE CARLO RESULT
# ============================================================

@dataclass(slots=True)
class MonteCarloStressResult(
    StressTestResult,
):
    """
    Monte Carlo simulation result.
    """

    num_paths: int = 0

    horizon_days: int = 0

    mean_return: float = 0.0

    median_return: float = 0.0

    worst_case_return: float = 0.0

    best_case_return: float = 0.0

    percentile_1: float = 0.0

    percentile_5: float = 0.0

    percentile_95: float = 0.0

    percentile_99: float = 0.0


# ============================================================
# TAIL RISK RESULT
# ============================================================

@dataclass(slots=True)
class TailRiskStressResult(
    StressTestResult,
):
    """
    Tail-risk stress result.
    """

    value_at_risk: float = 0.0

    expected_shortfall: float = 0.0

    tail_loss_probability: float = 0.0


# ============================================================
# REVERSE STRESS RESULT
# ============================================================

@dataclass(slots=True)
class ReverseStressResult(
    StressTestResult,
):
    """
    Reverse stress result.
    """

    target_loss: float = 0.0

    required_market_shock: float = 0.0

    required_factor_shock: float = 0.0

    scenario_description: str = ""


# ============================================================
# VULNERABILITY RESULT
# ============================================================

@dataclass(slots=True)
class VulnerabilityAnalysisResult(
    StressTestResult,
):
    """
    Portfolio vulnerability analysis.
    """

    vulnerability_score: float = 0.0

    concentration_risk: float = 0.0

    liquidity_risk: float = 0.0

    factor_risk: float = 0.0

    tail_risk: float = 0.0

    top_risk_sources: list[str] = field(
        default_factory=list
    )


# ============================================================
# MASTER STRESS REPORT
# ============================================================

@dataclass(slots=True)
class InstitutionalStressReport:
    """
    Institutional master stress report.
    """

    metadata: StressMetadata

    report_timestamp: datetime

    portfolio_name: str

    # --------------------------------------
    # Component Results
    # --------------------------------------

    historical_results: list[
        HistoricalStressResult
    ] = field(
        default_factory=list
    )

    scenario_results: list[
        ScenarioStressResult
    ] = field(
        default_factory=list
    )

    factor_results: list[
        FactorShockResult
    ] = field(
        default_factory=list
    )

    correlation_results: list[
        CorrelationBreakdownResult
    ] = field(
        default_factory=list
    )

    liquidity_results: list[
        LiquidityStressResult
    ] = field(
        default_factory=list
    )

    volatility_results: list[
        VolatilityStressResult
    ] = field(
        default_factory=list
    )

    monte_carlo_results: list[
        MonteCarloStressResult
    ] = field(
        default_factory=list
    )

    tail_risk_results: list[
        TailRiskStressResult
    ] = field(
        default_factory=list
    )

    reverse_results: list[
        ReverseStressResult
    ] = field(
        default_factory=list
    )

    vulnerability_results: list[
        VulnerabilityAnalysisResult
    ] = field(
        default_factory=list
    )

    # --------------------------------------
    # Summary
    # --------------------------------------

    summary_metrics: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    diagnostics: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )


# ============================================================
# REPORT EXPORT HELPERS
# ============================================================

class StressReportExporter:
    """
    Export institutional stress reports.
    """

    # --------------------------------------------------------

    @staticmethod
    def to_dict(
        report:
        InstitutionalStressReport,
    ) -> dict[str, Any]:

        return asdict(
            report
        )

    # --------------------------------------------------------

    @staticmethod
    def to_json(
        report:
        InstitutionalStressReport,
    ) -> str:

        return json.dumps(

            asdict(report),

            default=str,

            indent=2,

        )

    # --------------------------------------------------------

    @staticmethod
    def summary_dataframe(
        report:
        InstitutionalStressReport,
    ) -> pd.DataFrame:

        return pd.DataFrame(
            [
                report.summary_metrics
            ]
        )
    


# ============================================================
# PART 3
# HISTORICAL STRESS TESTING
# ============================================================

# ============================================================
# HISTORICAL EVENT LIBRARY
# ============================================================

@dataclass(slots=True)
class HistoricalStressEvent:
    """
    Historical market event definition.
    """

    event_name: str

    start_date: str

    end_date: str

    description: str

    severity: ScenarioSeverity


# ------------------------------------------------------------

DEFAULT_HISTORICAL_EVENTS = {

    "GFC_2008":

    HistoricalStressEvent(

        event_name="Global Financial Crisis",

        start_date="2008-09-01",

        end_date="2009-03-31",

        description=
        "Lehman collapse and global recession",

        severity=
        ScenarioSeverity.EXTREME,
    ),

    "COVID_2020":

    HistoricalStressEvent(

        event_name="COVID Crash",

        start_date="2020-02-15",

        end_date="2020-04-15",

        description=
        "Pandemic-driven market crash",

        severity=
        ScenarioSeverity.EXTREME,
    ),

    "DOTCOM":

    HistoricalStressEvent(

        event_name="Dot-Com Bust",

        start_date="2000-03-01",

        end_date="2002-10-01",

        description=
        "Technology bubble collapse",

        severity=
        ScenarioSeverity.SEVERE,
    ),

    "TAPER_TANTRUM":

    HistoricalStressEvent(

        event_name="Taper Tantrum",

        start_date="2013-05-01",

        end_date="2013-09-01",

        description=
        "Fed taper announcement shock",

        severity=
        ScenarioSeverity.MODERATE,
    ),

    "BANKING_CRISIS_2023":

    HistoricalStressEvent(

        event_name="Regional Banking Crisis",

        start_date="2023-03-01",

        end_date="2023-04-15",

        description=
        "SVB and regional bank stress",

        severity=
        ScenarioSeverity.MODERATE,
    ),
}


# ============================================================
# HISTORICAL STRESS TESTER
# ============================================================

class HistoricalStressTester(
    BaseStressTest,
):
    """
    Institutional historical event replay.

    Replays portfolio through:

        Historical crises
        Historical crashes
        Historical volatility spikes

    and computes:

        Return
        Drawdown
        Volatility impact
        Active return
    """

    # --------------------------------------------------------

    def __init__(
        self,
        metadata: StressMetadata,
        config:
        StressTestingConfig | None = None,
    ) -> None:

        super().__init__(
            metadata,
            config,
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    @staticmethod
    def validate_inputs(
        returns: pd.DataFrame,
        weights: pd.Series,
    ) -> None:

        StressValidation.validate_dataframe(
            returns,
            "returns",
        )

        StressValidation.validate_series(
            weights,
            "weights",
        )

    # --------------------------------------------------------
    # Portfolio Return Series
    # --------------------------------------------------------

    @staticmethod
    def portfolio_return_series(
        returns: pd.DataFrame,
        weights: pd.Series,
    ) -> pd.Series:

        common_assets = [

            c

            for c in returns.columns

            if c in weights.index

        ]

        if len(common_assets) == 0:

            raise ValueError(
                "No common assets."
            )

        aligned_returns = (
            returns[
                common_assets
            ]
        )

        aligned_weights = (
            weights.loc[
                common_assets
            ]
        )

        return (
            aligned_returns
            @
            aligned_weights
        )

    # --------------------------------------------------------
    # Slice Event Window
    # --------------------------------------------------------

    @staticmethod
    def event_window(
        returns: pd.Series,
        start_date: str,
        end_date: str,
    ) -> pd.Series:

        return returns.loc[
            start_date:end_date
        ]

    # --------------------------------------------------------
    # Analyze Event
    # --------------------------------------------------------

    def analyze_event(
        self,
        *,
        returns: pd.DataFrame,
        weights: pd.Series,
        event:
        HistoricalStressEvent,
        benchmark_returns:
        pd.Series | None = None,
    ) -> HistoricalStressResult:

        portfolio_returns = (
            self.portfolio_return_series(
                returns,
                weights,
            )
        )

        event_returns = (
            self.event_window(

                portfolio_returns,

                event.start_date,

                event.end_date,

            )
        )

        if len(
            event_returns
        ) == 0:

            raise ValueError(

                f"No data for "
                f"{event.event_name}"

            )

        # ----------------------------------
        # Metrics
        # ----------------------------------

        portfolio_return = (

            StressUtils
            .cumulative_return(
                event_returns
            )

        )

        pnl = (
            portfolio_return
        )

        drawdown = (

            StressUtils
            .max_drawdown(
                event_returns
            )

        )

        stressed_vol = (

            StressUtils
            .annualize_volatility(
                event_returns
            )

        )

        active_return = 0.0

        benchmark_return = 0.0

        if (
            benchmark_returns
            is not None
        ):

            benchmark_window = (
                benchmark_returns.loc[
                    event.start_date:
                    event.end_date
                ]
            )

            benchmark_return = (

                StressUtils
                .cumulative_return(
                    benchmark_window
                )

            )

            active_return = (

                portfolio_return
                -
                benchmark_return

            )

        return HistoricalStressResult(

            metadata=
            self.metadata,

            stress_type=
            StressTestType.HISTORICAL,

            scenario_name=
            event.event_name,

            severity=
            event.severity,

            status=
            StressResultStatus.PASSED,

            portfolio_return=
            portfolio_return,

            portfolio_pnl=
            pnl,

            drawdown=
            drawdown,

            var_impact=
            0.0,

            volatility_impact=
            stressed_vol,

            diagnostics={

                "description":
                event.description,

            },

            event_name=
            event.event_name,

            start_date=
            pd.Timestamp(
                event.start_date
            ),

            end_date=
            pd.Timestamp(
                event.end_date
            ),

            benchmark_return=
            benchmark_return,

            active_return=
            active_return,

            period_length_days=
            len(
                event_returns
            ),
        )

    # --------------------------------------------------------
    # Run Named Event
    # --------------------------------------------------------

    def run_named_event(
        self,
        *,
        returns: pd.DataFrame,
        weights: pd.Series,
        event_key: str,
        benchmark_returns:
        pd.Series | None = None,
    ) -> HistoricalStressResult:

        if (
            event_key
            not in
            DEFAULT_HISTORICAL_EVENTS
        ):

            raise KeyError(
                f"Unknown event: "
                f"{event_key}"
            )

        event = (
            DEFAULT_HISTORICAL_EVENTS[
                event_key
            ]
        )

        return self.analyze_event(

            returns=
            returns,

            weights=
            weights,

            event=
            event,

            benchmark_returns=
            benchmark_returns,
        )

    # --------------------------------------------------------
    # Run All Events
    # --------------------------------------------------------

    def run_all_events(
        self,
        *,
        returns: pd.DataFrame,
        weights: pd.Series,
        benchmark_returns:
        pd.Series | None = None,
    ) -> list[
        HistoricalStressResult
    ]:

        results = []

        for event in (
            DEFAULT_HISTORICAL_EVENTS
            .values()
        ):

            try:

                result = (
                    self.analyze_event(

                        returns=
                        returns,

                        weights=
                        weights,

                        event=
                        event,

                        benchmark_returns=
                        benchmark_returns,

                    )
                )

                results.append(
                    result
                )

            except Exception:

                continue

        return results

    # --------------------------------------------------------
    # Worst Rolling Period
    # --------------------------------------------------------

    def worst_period(
        self,
        *,
        returns: pd.DataFrame,
        weights: pd.Series,
        window_days: int = 63,
    ) -> HistoricalStressResult:

        portfolio_returns = (
            self.portfolio_return_series(
                returns,
                weights,
            )
        )

        rolling_returns = (

            (
                1.0
                +
                portfolio_returns
            )

            .rolling(
                window_days
            )

            .apply(
                np.prod,
                raw=True,
            )

            -
            1.0

        )

        worst_idx = (
            rolling_returns
            .idxmin()
        )

        end_loc = (
            portfolio_returns.index
            .get_loc(
                worst_idx
            )
        )

        start_loc = max(
            0,
            end_loc
            -
            window_days
            +
            1,
        )

        window = (
            portfolio_returns.iloc[
                start_loc:
                end_loc + 1
            ]
        )

        return HistoricalStressResult(

            metadata=
            self.metadata,

            stress_type=
            StressTestType.HISTORICAL,

            scenario_name=
            "Worst Rolling Period",

            severity=
            ScenarioSeverity.SEVERE,

            status=
            StressResultStatus.WARNING,

            portfolio_return=
            StressUtils
            .cumulative_return(
                window
            ),

            portfolio_pnl=
            StressUtils
            .cumulative_return(
                window
            ),

            drawdown=
            StressUtils
            .max_drawdown(
                window
            ),

            var_impact=
            0.0,

            volatility_impact=
            StressUtils
            .annualize_volatility(
                window
            ),

            event_name=
            "Worst Rolling Period",

            start_date=
            window.index[0],

            end_date=
            window.index[-1],

            period_length_days=
            len(window),
        )

    # --------------------------------------------------------
    # Required Abstract
    # --------------------------------------------------------

    def run(
        self,
        *args,
        **kwargs,
    ) -> HistoricalStressResult:

        return self.run_named_event(
            *args,
            **kwargs,
        )


# ============================================================
# CONVENIENCE APIS
# ============================================================

def historical_stress_test(
    *,
    metadata: StressMetadata,
    returns: pd.DataFrame,
    weights: pd.Series,
    event_key: str,
    benchmark_returns:
    pd.Series | None = None,
    config:
    StressTestingConfig | None = None,
) -> HistoricalStressResult:

    tester = HistoricalStressTester(
        metadata=metadata,
        config=config,
    )

    return tester.run_named_event(

        returns=returns,

        weights=weights,

        event_key=event_key,

        benchmark_returns=
        benchmark_returns,
    )


def run_all_historical_events(
    *,
    metadata: StressMetadata,
    returns: pd.DataFrame,
    weights: pd.Series,
    benchmark_returns:
    pd.Series | None = None,
    config:
    StressTestingConfig | None = None,
) -> list[
    HistoricalStressResult
]:

    tester = HistoricalStressTester(
        metadata=metadata,
        config=config,
    )

    return tester.run_all_events(

        returns=returns,

        weights=weights,

        benchmark_returns=
        benchmark_returns,
    )

# ============================================================
# PART 4
# SCENARIO STRESS TESTING
# ============================================================

# ============================================================
# PREDEFINED SCENARIOS
# ============================================================

SCENARIO_LIBRARY: dict[
    str,
    StressScenario,
] = {

    "equity_crash_10":

    StressScenario(

        scenario_name=
        "Equity Crash -10%",

        severity=
        ScenarioSeverity.MODERATE,

        description=
        "Broad market equity decline of 10%",

        factor_shocks={
            "equity_market":
            -0.10,
        },
    ),

    "equity_crash_20":

    StressScenario(

        scenario_name=
        "Equity Crash -20%",

        severity=
        ScenarioSeverity.SEVERE,

        description=
        "Broad market equity decline of 20%",

        factor_shocks={
            "equity_market":
            -0.20,
        },
    ),

    "equity_crash_40":

    StressScenario(

        scenario_name=
        "Equity Crash -40%",

        severity=
        ScenarioSeverity.EXTREME,

        description=
        "Extreme equity market collapse",

        factor_shocks={
            "equity_market":
            -0.40,
        },
    ),

    "rates_up_100bps":

    StressScenario(

        scenario_name=
        "Rates +100bps",

        severity=
        ScenarioSeverity.MODERATE,

        description=
        "Interest rate shock",

        factor_shocks={
            "rates":
            0.01,
        },
    ),

    "rates_up_300bps":

    StressScenario(

        scenario_name=
        "Rates +300bps",

        severity=
        ScenarioSeverity.SEVERE,

        description=
        "Aggressive tightening cycle",

        factor_shocks={
            "rates":
            0.03,
        },
    ),

    "credit_spread_widening":

    StressScenario(

        scenario_name=
        "Credit Spread Shock",

        severity=
        ScenarioSeverity.SEVERE,

        description=
        "Credit spreads widen sharply",

        factor_shocks={
            "credit_spread":
            0.05,
        },
    ),

    "usd_strength":

    StressScenario(

        scenario_name=
        "USD Strength",

        severity=
        ScenarioSeverity.MODERATE,

        description=
        "US dollar appreciation",

        factor_shocks={
            "fx":
            0.10,
        },
    ),

    "commodity_crash":

    StressScenario(

        scenario_name=
        "Commodity Crash",

        severity=
        ScenarioSeverity.SEVERE,

        description=
        "Commodity market decline",

        factor_shocks={
            "commodity":
            -0.30,
        },
    ),
}


# ============================================================
# SCENARIO STRESS TESTER
# ============================================================

class ScenarioStressTester(
    BaseStressTest,
):
    """
    Institutional scenario engine.

    Supports:

        Equity shocks
        Interest-rate shocks
        Credit shocks
        FX shocks
        Commodity shocks

        Custom factor scenarios
    """

    # --------------------------------------------------------

    @staticmethod
    def validate_exposures(
        factor_exposures: pd.Series,
    ) -> None:

        StressValidation.validate_series(

            factor_exposures,

            "factor_exposures",

        )

    # --------------------------------------------------------

    @staticmethod
    def scenario_pnl(
        factor_exposures: pd.Series,
        factor_shocks: dict[
            str,
            float,
        ],
    ) -> tuple[
        float,
        dict[str, float],
    ]:

        contributions = {}

        total_pnl = 0.0

        for factor, shock in (
            factor_shocks.items()
        ):

            exposure = float(

                factor_exposures.get(
                    factor,
                    0.0,
                )

            )

            contribution = (
                exposure
                * shock
            )

            contributions[
                factor
            ] = contribution

            total_pnl += (
                contribution
            )

        return (
            float(total_pnl),
            contributions,
        )

    # --------------------------------------------------------

    def run_scenario(
        self,
        *,
        factor_exposures: pd.Series,
        scenario:
        StressScenario,
        current_volatility:
        float | None = None,
    ) -> ScenarioStressResult:

        self.validate_exposures(
            factor_exposures
        )

        pnl, contributions = (

            self.scenario_pnl(

                factor_exposures=
                factor_exposures,

                factor_shocks=
                scenario.factor_shocks,

            )

        )

        stressed_vol = 0.0

        if (
            current_volatility
            is not None
        ):

            stressed_vol = (

                current_volatility

                *
                scenario
                .volatility_multiplier

            )

        # ----------------------------------
        # status
        # ----------------------------------

        if pnl <= -0.20:

            status = (
                StressResultStatus
                .FAILED
            )

        elif pnl <= -0.10:

            status = (
                StressResultStatus
                .WARNING
            )

        else:

            status = (
                StressResultStatus
                .PASSED
            )

        return ScenarioStressResult(

            metadata=
            self.metadata,

            stress_type=
            StressTestType.SCENARIO,

            scenario_name=
            scenario.scenario_name,

            severity=
            scenario.severity,

            status=
            status,

            portfolio_return=
            pnl,

            portfolio_pnl=
            pnl,

            drawdown=
            pnl,

            var_impact=
            abs(pnl),

            volatility_impact=
            stressed_vol,

            diagnostics={

                "description":
                scenario.description,

            },

            shocked_factors=
            scenario.factor_shocks,

            factor_contributions=
            contributions,
        )

    # --------------------------------------------------------

    def run_library_scenario(
        self,
        *,
        factor_exposures: pd.Series,
        scenario_key: str,
        current_volatility:
        float | None = None,
    ) -> ScenarioStressResult:

        if (
            scenario_key
            not in
            SCENARIO_LIBRARY
        ):

            raise KeyError(

                f"Unknown scenario: "
                f"{scenario_key}"

            )

        return self.run_scenario(

            factor_exposures=
            factor_exposures,

            scenario=
            SCENARIO_LIBRARY[
                scenario_key
            ],

            current_volatility=
            current_volatility,
        )

    # --------------------------------------------------------

    def run_all_library_scenarios(
        self,
        *,
        factor_exposures: pd.Series,
        current_volatility:
        float | None = None,
    ) -> list[
        ScenarioStressResult
    ]:

        results = []

        for scenario in (
            SCENARIO_LIBRARY
            .values()
        ):

            results.append(

                self.run_scenario(

                    factor_exposures=
                    factor_exposures,

                    scenario=
                    scenario,

                    current_volatility=
                    current_volatility,

                )

            )

        return results

    # --------------------------------------------------------

    def create_custom_scenario(
        self,
        *,
        name: str,
        factor_shocks:
        dict[str, float],
        severity:
        ScenarioSeverity =
        ScenarioSeverity.MODERATE,
        description: str = "",
        volatility_multiplier:
        float = 1.0,
        correlation_shift:
        float = 0.0,
        liquidity_haircut:
        float = 0.0,
    ) -> StressScenario:

        return StressScenario(

            scenario_name=
            name,

            severity=
            severity,

            description=
            description,

            factor_shocks=
            factor_shocks,

            volatility_multiplier=
            volatility_multiplier,

            correlation_shift=
            correlation_shift,

            liquidity_haircut=
            liquidity_haircut,
        )

    # --------------------------------------------------------

    def run(
        self,
        *args,
        **kwargs,
    ) -> ScenarioStressResult:

        return self.run_scenario(
            *args,
            **kwargs,
        )


# ============================================================
# CONVENIENCE APIS
# ============================================================

def scenario_stress_test(
    *,
    metadata: StressMetadata,
    factor_exposures: pd.Series,
    scenario_key: str,
    current_volatility:
    float | None = None,
    config:
    StressTestingConfig | None = None,
) -> ScenarioStressResult:

    tester = ScenarioStressTester(

        metadata=
        metadata,

        config=
        config,
    )

    return tester.run_library_scenario(

        factor_exposures=
        factor_exposures,

        scenario_key=
        scenario_key,

        current_volatility=
        current_volatility,
    )


def run_all_scenarios(
    *,
    metadata: StressMetadata,
    factor_exposures: pd.Series,
    current_volatility:
    float | None = None,
    config:
    StressTestingConfig | None = None,
) -> list[
    ScenarioStressResult
]:

    tester = ScenarioStressTester(

        metadata=
        metadata,

        config=
        config,
    )

    return (
        tester
        .run_all_library_scenarios(

            factor_exposures=
            factor_exposures,

            current_volatility=
            current_volatility,

        )
    )


# ============================================================
# PART 5
# FACTOR SHOCK ENGINE
# ============================================================

# ============================================================
# FACTOR DEFINITIONS
# ============================================================

class FactorType(
    str,
    Enum,
):

    MARKET = "market"

    VALUE = "value"

    MOMENTUM = "momentum"

    SIZE = "size"

    QUALITY = "quality"

    LOW_VOL = "low_vol"

    GROWTH = "growth"

    CREDIT = "credit"

    RATES = "rates"

    FX = "fx"

    COMMODITY = "commodity"


# ============================================================
# FACTOR SHOCK DEFINITION
# ============================================================

@dataclass(slots=True)
class FactorShock:

    factor_name: str

    shock_value: float

    description: str = ""

    severity: ScenarioSeverity = (
        ScenarioSeverity.MODERATE
    )


# ============================================================
# FACTOR SHOCK RESULT
# ============================================================

@dataclass(slots=True)
class FactorShockAnalysis:

    factor_name: str

    exposure: float

    shock_value: float

    pnl_impact: float

    contribution_pct: float


# ============================================================
# FACTOR SHOCK TEST RESULT
# ============================================================

@dataclass(slots=True)
class FactorStressResult(
    StressTestResult,
):

    total_factor_pnl: float = 0.0

    factor_breakdown: list[
        FactorShockAnalysis
    ] = field(
        default_factory=list
    )

    vulnerability_rank: dict[
        str,
        int,
    ] = field(
        default_factory=dict
    )


# ============================================================
# FACTOR SHOCK LIBRARY
# ============================================================

DEFAULT_FACTOR_SHOCKS = {

    FactorType.MARKET.value:
    FactorShock(
        factor_name=
        FactorType.MARKET.value,

        shock_value=
        -0.15,

        description=
        "Broad market drawdown",

        severity=
        ScenarioSeverity.SEVERE,
    ),

    FactorType.VALUE.value:
    FactorShock(
        factor_name=
        FactorType.VALUE.value,

        shock_value=
        -0.10,
    ),

    FactorType.MOMENTUM.value:
    FactorShock(
        factor_name=
        FactorType.MOMENTUM.value,

        shock_value=
        -0.12,
    ),

    FactorType.SIZE.value:
    FactorShock(
        factor_name=
        FactorType.SIZE.value,

        shock_value=
        -0.08,
    ),

    FactorType.QUALITY.value:
    FactorShock(
        factor_name=
        FactorType.QUALITY.value,

        shock_value=
        -0.05,
    ),

    FactorType.LOW_VOL.value:
    FactorShock(
        factor_name=
        FactorType.LOW_VOL.value,

        shock_value=
        -0.03,
    ),

    FactorType.CREDIT.value:
    FactorShock(
        factor_name=
        FactorType.CREDIT.value,

        shock_value=
        -0.10,
    ),

    FactorType.RATES.value:
    FactorShock(
        factor_name=
        FactorType.RATES.value,

        shock_value=
        0.02,
    ),

    FactorType.FX.value:
    FactorShock(
        factor_name=
        FactorType.FX.value,

        shock_value=
        0.10,
    ),

    FactorType.COMMODITY.value:
    FactorShock(
        factor_name=
        FactorType.COMMODITY.value,

        shock_value=
        -0.25,
    ),
}


# ============================================================
# FACTOR SHOCK ENGINE
# ============================================================

class FactorShockEngine(
    BaseStressTest,
):
    """
    Institutional factor stress engine.
    """

    # --------------------------------------------------------

    @staticmethod
    def validate_inputs(
        factor_exposures: pd.Series,
    ) -> None:

        StressValidation.validate_series(

            factor_exposures,

            "factor_exposures",

        )

    # --------------------------------------------------------

    @staticmethod
    def compute_factor_impact(
        exposure: float,
        shock: float,
    ) -> float:

        return float(
            exposure * shock
        )

    # --------------------------------------------------------

    def run_factor_shocks(
        self,
        *,
        factor_exposures: pd.Series,
        shocks:
        dict[str, FactorShock]
        | None = None,
    ) -> FactorStressResult:

        self.validate_inputs(
            factor_exposures
        )

        if shocks is None:

            shocks = (
                DEFAULT_FACTOR_SHOCKS
            )

        factor_results = []

        total_pnl = 0.0

        # ----------------------------------
        # Factor level analysis
        # ----------------------------------

        for factor, shock_obj in (
            shocks.items()
        ):

            exposure = float(

                factor_exposures.get(
                    factor,
                    0.0,
                )

            )

            pnl = (
                self.compute_factor_impact(
                    exposure,
                    shock_obj.shock_value,
                )
            )

            total_pnl += pnl

            factor_results.append(

                FactorShockAnalysis(

                    factor_name=
                    factor,

                    exposure=
                    exposure,

                    shock_value=
                    shock_obj.shock_value,

                    pnl_impact=
                    pnl,

                    contribution_pct=
                    0.0,

                )

            )

        # ----------------------------------
        # Contribution %
        # ----------------------------------

        abs_total = max(

            sum(
                abs(
                    x.pnl_impact
                )
                for x in factor_results
            ),

            EPSILON,
        )

        for x in factor_results:

            x.contribution_pct = (

                abs(
                    x.pnl_impact
                )

                / abs_total

            )

        # ----------------------------------
        # Vulnerability ranking
        # ----------------------------------

        sorted_factors = sorted(

            factor_results,

            key=lambda x:
            abs(
                x.pnl_impact
            ),

            reverse=True,
        )

        vulnerability_rank = {

            x.factor_name:
            rank + 1

            for rank, x
            in enumerate(
                sorted_factors
            )
        }

        # ----------------------------------
        # Status
        # ----------------------------------

        if total_pnl <= -0.20:

            status = (
                StressResultStatus
                .FAILED
            )

        elif total_pnl <= -0.10:

            status = (
                StressResultStatus
                .WARNING
            )

        else:

            status = (
                StressResultStatus
                .PASSED
            )

        return FactorStressResult(

            metadata=
            self.metadata,

            stress_type=
            StressTestType
            .FACTOR_SHOCK,

            scenario_name=
            "Factor Shock Analysis",

            severity=
            ScenarioSeverity
            .SEVERE,

            status=
            status,

            portfolio_return=
            total_pnl,

            portfolio_pnl=
            total_pnl,

            drawdown=
            total_pnl,

            var_impact=
            abs(
                total_pnl
            ),

            volatility_impact=
            0.0,

            diagnostics={

                "num_factors":
                len(
                    factor_results
                )

            },

            total_factor_pnl=
            total_pnl,

            factor_breakdown=
            factor_results,

            vulnerability_rank=
            vulnerability_rank,
        )

    # --------------------------------------------------------

    def single_factor_stress(
        self,
        *,
        factor_exposures:
        pd.Series,

        factor_name: str,

        shock_value: float,
    ) -> FactorStressResult:

        shock = {

            factor_name:

            FactorShock(

                factor_name=
                factor_name,

                shock_value=
                shock_value,
            )

        }

        return self.run_factor_shocks(

            factor_exposures=
            factor_exposures,

            shocks=
            shock,
        )

    # --------------------------------------------------------

    def sensitivity_matrix(
        self,
        *,
        factor_exposures:
        pd.Series,

        shock_grid:
        np.ndarray,
    ) -> pd.DataFrame:

        results = []

        for factor in (
            factor_exposures.index
        ):

            exposure = float(
                factor_exposures[
                    factor
                ]
            )

            row = []

            for shock in (
                shock_grid
            ):

                row.append(

                    exposure
                    * shock

                )

            results.append(
                row
            )

        return pd.DataFrame(

            results,

            index=
            factor_exposures.index,

            columns=
            shock_grid,
        )

    # --------------------------------------------------------

    def run(
        self,
        *args,
        **kwargs,
    ) -> FactorStressResult:

        return self.run_factor_shocks(
            *args,
            **kwargs,
        )


# ============================================================
# CONVENIENCE APIS
# ============================================================

def factor_stress_test(
    *,
    metadata:
    StressMetadata,

    factor_exposures:
    pd.Series,

    shocks:
    dict[str, FactorShock]
    | None = None,

    config:
    StressTestingConfig
    | None = None,
) -> FactorStressResult:

    engine = (
        FactorShockEngine(
            metadata=
            metadata,

            config=
            config,
        )
    )

    return engine.run_factor_shocks(

        factor_exposures=
        factor_exposures,

        shocks=
        shocks,
    )


def factor_sensitivity_matrix(
    *,
    metadata:
    StressMetadata,

    factor_exposures:
    pd.Series,

    shock_grid:
    np.ndarray,
) -> pd.DataFrame:

    engine = (
        FactorShockEngine(
            metadata=
            metadata
        )
    )

    return engine.sensitivity_matrix(

        factor_exposures=
        factor_exposures,

        shock_grid=
        shock_grid,
    )


# ============================================================
# PART 6
# CORRELATION BREAKDOWN ENGINE
# ============================================================

# ============================================================
# CORRELATION STRESS CONFIG
# ============================================================

@dataclass(slots=True)
class CorrelationStressConfig:
    """
    Configuration for correlation stress testing.
    """

    stressed_correlation: float = 0.90

    extreme_correlation: float = 0.99

    minimum_correlation: float = -0.99

    maximum_correlation: float = 0.99


# ============================================================
# CORRELATION STRESS RESULT DETAIL
# ============================================================

@dataclass(slots=True)
class CorrelationShockDetail:

    original_correlation: float

    stressed_correlation: float

    correlation_change: float


# ============================================================
# EXTENDED RESULT
# ============================================================

@dataclass(slots=True)
class CorrelationStressAnalysisResult(
    CorrelationBreakdownResult,
):

    original_portfolio_vol: float = 0.0

    stressed_portfolio_vol: float = 0.0

    diversification_ratio_before: float = 0.0

    diversification_ratio_after: float = 0.0

    correlation_details: dict[
        str,
        CorrelationShockDetail,
    ] = field(
        default_factory=dict
    )


# ============================================================
# CORRELATION BREAKDOWN ENGINE
# ============================================================

class CorrelationBreakdownEngine(
    BaseStressTest,
):
    """
    Institutional correlation stress engine.

    Objectives:

        Correlation spikes
        Diversification failure
        Correlation regime stress
        Cross-asset contagion

    Inputs:

        covariance matrix
        correlation matrix
        portfolio weights
    """

    # --------------------------------------------------------

    def __init__(
        self,
        metadata: StressMetadata,
        config:
        StressTestingConfig | None = None,
        correlation_config:
        CorrelationStressConfig | None = None,
    ) -> None:

        super().__init__(
            metadata,
            config,
        )

        self.correlation_config = (

            correlation_config

            if correlation_config
            is not None

            else CorrelationStressConfig()

        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    @staticmethod
    def validate_inputs(
        weights: pd.Series,
        covariance: pd.DataFrame,
    ) -> None:

        StressValidation.validate_series(
            weights,
            "weights",
        )

        StressValidation.validate_dataframe(
            covariance,
            "covariance",
        )

    # --------------------------------------------------------
    # Covariance -> Correlation
    # --------------------------------------------------------

    @staticmethod
    def covariance_to_correlation(
        covariance:
        pd.DataFrame,
    ) -> pd.DataFrame:

        vol = np.sqrt(
            np.diag(
                covariance
            )
        )

        vol_matrix = np.outer(
            vol,
            vol,
        )

        corr = (

            covariance.values

            / np.maximum(
                vol_matrix,
                EPSILON,
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
            covariance.index,

            columns=
            covariance.columns,
        )

    # --------------------------------------------------------
    # Correlation -> Covariance
    # --------------------------------------------------------

    @staticmethod
    def correlation_to_covariance(
        correlation:
        pd.DataFrame,
        volatilities:
        np.ndarray,
    ) -> pd.DataFrame:

        vol_matrix = np.outer(
            volatilities,
            volatilities,
        )

        cov = (

            correlation.values

            * vol_matrix

        )

        return pd.DataFrame(

            cov,

            index=
            correlation.index,

            columns=
            correlation.columns,
        )

    # --------------------------------------------------------
    # Portfolio Volatility
    # --------------------------------------------------------

    @staticmethod
    def portfolio_volatility(
        weights:
        pd.Series,
        covariance:
        pd.DataFrame,
    ) -> float:

        w = (
            weights.values
        )

        cov = (
            covariance.values
        )

        return float(

            np.sqrt(

                w.T
                @ cov
                @ w

            )

        )

    # --------------------------------------------------------
    # Diversification Ratio
    # --------------------------------------------------------

    @staticmethod
    def diversification_ratio(
        weights:
        pd.Series,
        covariance:
        pd.DataFrame,
    ) -> float:

        asset_vol = np.sqrt(
            np.diag(
                covariance
            )
        )

        weighted_vol = float(

            np.sum(

                np.abs(
                    weights.values
                )

                * asset_vol

            )

        )

        portfolio_vol = (

            CorrelationBreakdownEngine
            .portfolio_volatility(

                weights,
                covariance,

            )

        )

        return float(

            weighted_vol

            /

            max(
                portfolio_vol,
                EPSILON,
            )

        )

    # --------------------------------------------------------
    # Stress Correlation Matrix
    # --------------------------------------------------------

    def stress_correlation_matrix(
        self,
        correlation:
        pd.DataFrame,
        stressed_level:
        float | None = None,
    ) -> pd.DataFrame:

        level = (

            stressed_level

            if stressed_level
            is not None

            else
            self.correlation_config
            .stressed_correlation

        )

        stressed = (
            correlation.copy()
        )

        for i in range(
            len(stressed)
        ):

            for j in range(
                len(stressed)
            ):

                if i != j:

                    stressed.iloc[
                        i,
                        j,
                    ] = level

        np.fill_diagonal(
            stressed.values,
            1.0,
        )

        return stressed

    # --------------------------------------------------------
    # Run Breakdown Stress
    # --------------------------------------------------------

    def run_breakdown_stress(
        self,
        *,
        weights:
        pd.Series,
        covariance:
        pd.DataFrame,
        stressed_level:
        float | None = None,
    ) -> (
        CorrelationStressAnalysisResult
    ):

        self.validate_inputs(

            weights,
            covariance,
        )

        corr = (
            self
            .covariance_to_correlation(
                covariance
            )
        )

        vols = np.sqrt(
            np.diag(
                covariance
            )
        )

        stressed_corr = (

            self
            .stress_correlation_matrix(

                corr,

                stressed_level,

            )

        )

        stressed_cov = (

            self
            .correlation_to_covariance(

                stressed_corr,

                vols,

            )

        )

        original_vol = (

            self
            .portfolio_volatility(

                weights,
                covariance,

            )

        )

        stressed_vol = (

            self
            .portfolio_volatility(

                weights,
                stressed_cov,

            )

        )

        diversification_before = (

            self
            .diversification_ratio(

                weights,
                covariance,

            )

        )

        diversification_after = (

            self
            .diversification_ratio(

                weights,
                stressed_cov,

            )

        )

        diversification_loss = (

            diversification_before

            -
            diversification_after

        )

        avg_corr_before = float(

            corr.where(
                ~np.eye(
                    len(corr),
                    dtype=bool,
                )
            )

            .stack()

            .mean()

        )

        avg_corr_after = float(

            stressed_corr.where(
                ~np.eye(
                    len(corr),
                    dtype=bool,
                )
            )

            .stack()

            .mean()

        )

        details = {}

        for i in corr.index:

            for j in corr.columns:

                if i >= j:
                    continue

                key = (
                    f"{i}|{j}"
                )

                details[key] = (

                    CorrelationShockDetail(

                        original_correlation=
                        float(
                            corr.loc[
                                i,
                                j,
                            ]
                        ),

                        stressed_correlation=
                        float(
                            stressed_corr.loc[
                                i,
                                j,
                            ]
                        ),

                        correlation_change=
                        float(

                            stressed_corr.loc[
                                i,
                                j,
                            ]

                            -

                            corr.loc[
                                i,
                                j,
                            ]

                        ),

                    )

                )

        stress_return = (

            -abs(
                stressed_vol
                -
                original_vol
            )

        )

        status = (
            StressResultStatus
            .PASSED
        )

        if (
            stressed_vol
            >
            original_vol
            * 1.50
        ):

            status = (
                StressResultStatus
                .FAILED
            )

        elif (
            stressed_vol
            >
            original_vol
            * 1.20
        ):

            status = (
                StressResultStatus
                .WARNING
            )

        return (
            CorrelationStressAnalysisResult(

                metadata=
                self.metadata,

                stress_type=
                StressTestType
                .CORRELATION_BREAKDOWN,

                scenario_name=
                "Correlation Breakdown",

                severity=
                ScenarioSeverity
                .SEVERE,

                status=
                status,

                portfolio_return=
                stress_return,

                portfolio_pnl=
                stress_return,

                drawdown=
                stress_return,

                var_impact=
                stressed_vol,

                volatility_impact=
                (
                    stressed_vol
                    -
                    original_vol
                ),

                diagnostics={

                    "stress_level":
                    avg_corr_after,

                },

                original_avg_correlation=
                avg_corr_before,

                stressed_avg_correlation=
                avg_corr_after,

                diversification_loss=
                diversification_loss,

                original_portfolio_vol=
                original_vol,

                stressed_portfolio_vol=
                stressed_vol,

                diversification_ratio_before=
                diversification_before,

                diversification_ratio_after=
                diversification_after,

                correlation_details=
                details,

            )
        )

    # --------------------------------------------------------

    def run(
        self,
        *args,
        **kwargs,
    ) -> (
        CorrelationStressAnalysisResult
    ):

        return (
            self
            .run_breakdown_stress(
                *args,
                **kwargs,
            )
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def correlation_breakdown_stress(
    *,
    metadata:
    StressMetadata,
    weights:
    pd.Series,
    covariance:
    pd.DataFrame,
    stressed_level:
    float | None = None,
    config:
    StressTestingConfig | None = None,
) -> (
    CorrelationStressAnalysisResult
):

    engine = (
        CorrelationBreakdownEngine(

            metadata=
            metadata,

            config=
            config,

        )
    )

    return (
        engine
        .run_breakdown_stress(

            weights=
            weights,

            covariance=
            covariance,

            stressed_level=
            stressed_level,

        )
    )


# ============================================================
# PART 7
# LIQUIDITY STRESS ENGINE
# ============================================================

# ============================================================
# LIQUIDITY CONFIG
# ============================================================

@dataclass(slots=True)
class LiquidityStressConfig:
    """
    Institutional liquidity stress configuration.
    """

    max_participation_rate: float = 0.10

    stress_participation_rate: float = 0.05

    liquidity_haircut: float = 0.20

    extreme_liquidity_haircut: float = 0.40

    market_depth_shock: float = 0.50

    min_daily_volume_buffer: float = 0.25


# ============================================================
# LIQUIDITY POSITION OBJECT
# ============================================================

@dataclass(slots=True)
class LiquidityPosition:

    asset: str

    position_value: float

    daily_volume: float

    avg_daily_dollar_volume: float

    participation_rate: float = 0.0


# ============================================================
# LIQUIDITY ANALYSIS DETAIL
# ============================================================

@dataclass(slots=True)
class LiquidityAnalysisDetail:

    asset: str

    position_value: float

    stressed_adv: float

    liquidation_days: float

    haircut_cost: float

    participation_rate: float


# ============================================================
# EXTENDED RESULT
# ============================================================

@dataclass(slots=True)
class LiquidityStressAnalysisResult(
    LiquidityStressResult,
):

    total_haircut_cost: float = 0.0

    max_liquidation_days: float = 0.0

    average_liquidation_days: float = 0.0

    portfolio_capacity: float = 0.0

    stressed_capacity: float = 0.0

    details: list[
        LiquidityAnalysisDetail
    ] = field(
        default_factory=list
    )


# ============================================================
# LIQUIDITY ENGINE
# ============================================================

class LiquidityStressEngine(
    BaseStressTest,
):
    """
    Institutional liquidity stress engine.

    Measures:

        liquidation horizon
        participation constraints
        liquidity haircuts
        capacity reduction
        forced liquidation impact
    """

    # --------------------------------------------------------

    def __init__(
        self,
        metadata: StressMetadata,
        config:
        StressTestingConfig | None = None,
        liquidity_config:
        LiquidityStressConfig | None = None,
    ) -> None:

        super().__init__(
            metadata,
            config,
        )

        self.liquidity_config = (

            liquidity_config

            if liquidity_config
            is not None

            else LiquidityStressConfig()

        )

    # --------------------------------------------------------

    @staticmethod
    def validate_positions(
        positions: pd.DataFrame,
    ) -> None:

        StressValidation.validate_dataframe(
            positions,
            "positions",
        )

        StressValidation.validate_columns(

            positions,

            [
                "asset",
                "position_value",
                "avg_daily_dollar_volume",
            ],

            "positions",
        )

    # --------------------------------------------------------
    # Liquidation Days
    # --------------------------------------------------------

    @staticmethod
    def liquidation_days(
        position_value: float,
        adv: float,
        participation_rate: float,
    ) -> float:

        tradable_per_day = (

            adv
            *
            participation_rate

        )

        return float(

            position_value

            /

            max(
                tradable_per_day,
                EPSILON,
            )

        )

    # --------------------------------------------------------
    # Haircut Cost
    # --------------------------------------------------------

    @staticmethod
    def haircut_cost(
        position_value: float,
        haircut: float,
    ) -> float:

        return float(
            position_value
            * haircut
        )

    # --------------------------------------------------------
    # Capacity
    # --------------------------------------------------------

    @staticmethod
    def capacity_estimate(
        adv: pd.Series,
        participation_rate: float,
    ) -> float:

        return float(

            np.sum(
                adv
            )

            *
            participation_rate

        )

    # --------------------------------------------------------
    # Position Level Analysis
    # --------------------------------------------------------

    def analyze_position(
        self,
        row: pd.Series,
    ) -> LiquidityAnalysisDetail:

        stressed_adv = (

            row[
                "avg_daily_dollar_volume"
            ]

            *
            (
                1.0
                -
                self.liquidity_config
                .market_depth_shock
            )

        )

        liquidation_days = (

            self.liquidation_days(

                position_value=
                row[
                    "position_value"
                ],

                adv=
                stressed_adv,

                participation_rate=
                self.liquidity_config
                .stress_participation_rate,

            )

        )

        haircut_cost = (

            self.haircut_cost(

                position_value=
                row[
                    "position_value"
                ],

                haircut=
                self.liquidity_config
                .liquidity_haircut,

            )

        )

        participation_rate = (

            row[
                "position_value"
            ]

            /

            max(
                stressed_adv,
                EPSILON,
            )

        )

        return (
            LiquidityAnalysisDetail(

                asset=
                row["asset"],

                position_value=
                float(
                    row[
                        "position_value"
                    ]
                ),

                stressed_adv=
                float(
                    stressed_adv
                ),

                liquidation_days=
                liquidation_days,

                haircut_cost=
                haircut_cost,

                participation_rate=
                float(
                    participation_rate
                ),
            )
        )

    # --------------------------------------------------------
    # Full Liquidity Stress
    # --------------------------------------------------------

    def run_liquidity_stress(
        self,
        *,
        positions:
        pd.DataFrame,
    ) -> (
        LiquidityStressAnalysisResult
    ):

        self.validate_positions(
            positions
        )

        details = []

        for _, row in (
            positions.iterrows()
        ):

            details.append(
                self.analyze_position(
                    row
                )
            )

        total_haircut = float(

            np.sum(

                [
                    x.haircut_cost
                    for x
                    in details
                ]

            )

        )

        max_days = float(

            np.max(

                [
                    x.liquidation_days
                    for x
                    in details
                ]

            )

        )

        avg_days = float(

            np.mean(

                [
                    x.liquidation_days
                    for x
                    in details
                ]

            )

        )

        portfolio_capacity = (

            self.capacity_estimate(

                positions[
                    "avg_daily_dollar_volume"
                ],

                self.liquidity_config
                .max_participation_rate,

            )

        )

        stressed_capacity = (

            portfolio_capacity

            *
            (
                1.0
                -
                self.liquidity_config
                .market_depth_shock
            )

        )

        liquidity_score = (

            1.0

            /

            max(
                avg_days,
                1.0,
            )

        )

        stress_return = (

            -total_haircut

            /

            max(

                positions[
                    "position_value"
                ].sum(),

                EPSILON,

            )

        )

        status = (
            StressResultStatus
            .PASSED
        )

        if max_days > 30:

            status = (
                StressResultStatus
                .FAILED
            )

        elif max_days > 10:

            status = (
                StressResultStatus
                .WARNING
            )

        return (
            LiquidityStressAnalysisResult(

                metadata=
                self.metadata,

                stress_type=
                StressTestType
                .LIQUIDITY,

                scenario_name=
                "Liquidity Stress",

                severity=
                ScenarioSeverity
                .SEVERE,

                status=
                status,

                portfolio_return=
                stress_return,

                portfolio_pnl=
                stress_return,

                drawdown=
                stress_return,

                var_impact=
                abs(
                    stress_return
                ),

                volatility_impact=
                0.0,

                diagnostics={

                    "num_positions":
                    len(
                        details
                    )

                },

                liquidity_haircut=
                self.liquidity_config
                .liquidity_haircut,

                liquidation_cost=
                total_haircut,

                days_to_liquidate=
                avg_days,

                liquidity_score=
                liquidity_score,

                total_haircut_cost=
                total_haircut,

                max_liquidation_days=
                max_days,

                average_liquidation_days=
                avg_days,

                portfolio_capacity=
                portfolio_capacity,

                stressed_capacity=
                stressed_capacity,

                details=
                details,
            )
        )

    # --------------------------------------------------------

    def run(
        self,
        *args,
        **kwargs,
    ) -> (
        LiquidityStressAnalysisResult
    ):

        return (
            self.run_liquidity_stress(
                *args,
                **kwargs,
            )
        )


# ============================================================
# FORCED LIQUIDATION SIMULATOR
# ============================================================

class ForcedLiquidationSimulator:
    """
    Simulates portfolio liquidation
    under stressed market depth.
    """

    @staticmethod
    def liquidation_schedule(
        positions:
        pd.DataFrame,
        participation_rate: float,
    ) -> pd.DataFrame:

        schedule = []

        for _, row in (
            positions.iterrows()
        ):

            adv = row[
                "avg_daily_dollar_volume"
            ]

            liquidation_days = (

                row[
                    "position_value"
                ]

                /

                max(
                    adv
                    *
                    participation_rate,
                    EPSILON,
                )

            )

            schedule.append(

                {

                    "asset":
                    row["asset"],

                    "liquidation_days":
                    liquidation_days,

                }

            )

        return pd.DataFrame(
            schedule
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def liquidity_stress_test(
    *,
    metadata:
    StressMetadata,

    positions:
    pd.DataFrame,

    config:
    StressTestingConfig | None = None,
) -> (
    LiquidityStressAnalysisResult
):

    engine = (
        LiquidityStressEngine(

            metadata=
            metadata,

            config=
            config,

        )
    )

    return (
        engine
        .run_liquidity_stress(

            positions=
            positions,

        )
    )


# ============================================================
# PART 8
# VOLATILITY REGIME STRESS ENGINE
# ============================================================

# ============================================================
# REGIME ENUM
# ============================================================

class VolatilityRegime(
    str,
    Enum,
):

    LOW_VOL = "low_vol"

    NORMAL_VOL = "normal_vol"

    HIGH_VOL = "high_vol"

    CRISIS_VOL = "crisis_vol"


# ============================================================
# VOLATILITY REGIME CONFIG
# ============================================================

@dataclass(slots=True)
class VolatilityRegimeConfig:

    low_vol_threshold: float = 0.10

    normal_vol_threshold: float = 0.20

    high_vol_threshold: float = 0.35

    crisis_vol_threshold: float = 0.60

    stress_multiplier_high: float = 1.50

    stress_multiplier_crisis: float = 2.50

    lookback_days: int = 63


# ============================================================
# REGIME TRANSITION
# ============================================================

@dataclass(slots=True)
class RegimeTransition:

    from_regime: VolatilityRegime

    to_regime: VolatilityRegime

    volatility_multiplier: float


# ============================================================
# VOLATILITY REGIME DETAIL
# ============================================================

@dataclass(slots=True)
class VolatilityRegimeDetail:

    current_regime: VolatilityRegime

    stressed_regime: VolatilityRegime

    current_volatility: float

    stressed_volatility: float

    volatility_change: float


# ============================================================
# EXTENDED RESULT
# ============================================================

@dataclass(slots=True)
class VolatilityRegimeStressResult(
    VolatilityStressResult,
):

    regime_detail: (
        VolatilityRegimeDetail
        | None
    ) = None

    volatility_shock_loss: float = 0.0

    expected_drawdown: float = 0.0

    volatility_multiplier_used: float = 1.0


# ============================================================
# VOLATILITY REGIME ENGINE
# ============================================================

class VolatilityRegimeStressEngine(
    BaseStressTest,
):
    """
    Institutional volatility regime stress.

    Features:

        Volatility regime classification
        Regime transitions
        Crisis-volatility shocks
        Volatility clustering stress
        VIX-style stress scenarios
    """

    # --------------------------------------------------------

    def __init__(
        self,
        metadata: StressMetadata,
        config:
        StressTestingConfig | None = None,
        regime_config:
        VolatilityRegimeConfig | None = None,
    ) -> None:

        super().__init__(
            metadata,
            config,
        )

        self.regime_config = (

            regime_config

            if regime_config
            is not None

            else VolatilityRegimeConfig()

        )

    # --------------------------------------------------------
    # Realized Vol
    # --------------------------------------------------------

    @staticmethod
    def realized_volatility(
        returns: pd.Series,
    ) -> float:

        StressValidation.validate_series(
            returns,
            "returns",
        )

        return float(

            returns.std()

            * np.sqrt(
                TRADING_DAYS
            )

        )

    # --------------------------------------------------------
    # Regime Classification
    # --------------------------------------------------------

    def classify_regime(
        self,
        volatility: float,
    ) -> VolatilityRegime:

        cfg = (
            self.regime_config
        )

        if (
            volatility
            <
            cfg.low_vol_threshold
        ):

            return (
                VolatilityRegime
                .LOW_VOL
            )

        if (
            volatility
            <
            cfg.normal_vol_threshold
        ):

            return (
                VolatilityRegime
                .NORMAL_VOL
            )

        if (
            volatility
            <
            cfg.high_vol_threshold
        ):

            return (
                VolatilityRegime
                .HIGH_VOL
            )

        return (
            VolatilityRegime
            .CRISIS_VOL
        )

    # --------------------------------------------------------
    # Shock Multiplier
    # --------------------------------------------------------

    def regime_multiplier(
        self,
        target_regime:
        VolatilityRegime,
    ) -> float:

        if (
            target_regime
            ==
            VolatilityRegime
            .HIGH_VOL
        ):

            return (
                self.regime_config
                .stress_multiplier_high
            )

        if (
            target_regime
            ==
            VolatilityRegime
            .CRISIS_VOL
        ):

            return (
                self.regime_config
                .stress_multiplier_crisis
            )

        return 1.0

    # --------------------------------------------------------
    # Expected Drawdown
    # --------------------------------------------------------

    @staticmethod
    def expected_drawdown(
        stressed_volatility:
        float,
    ) -> float:

        return float(
            -2.0
            * stressed_volatility
        )

    # --------------------------------------------------------
    # Volatility Shock Loss
    # --------------------------------------------------------

    @staticmethod
    def volatility_shock_loss(
        current_vol: float,
        stressed_vol: float,
    ) -> float:

        return float(

            -abs(

                stressed_vol

                -
                current_vol

            )

        )

    # --------------------------------------------------------
    # Run Regime Stress
    # --------------------------------------------------------

    def run_regime_stress(
        self,
        *,
        returns: pd.Series,
        target_regime:
        VolatilityRegime =
        VolatilityRegime.CRISIS_VOL,
    ) -> (
        VolatilityRegimeStressResult
    ):

        current_vol = (
            self.realized_volatility(
                returns
            )
        )

        current_regime = (
            self.classify_regime(
                current_vol
            )
        )

        multiplier = (
            self.regime_multiplier(
                target_regime
            )
        )

        stressed_vol = (
            current_vol
            * multiplier
        )

        vol_change = (
            stressed_vol
            - current_vol
        )

        expected_dd = (
            self.expected_drawdown(
                stressed_vol
            )
        )

        shock_loss = (
            self.volatility_shock_loss(

                current_vol,

                stressed_vol,

            )
        )

        if (
            target_regime
            ==
            VolatilityRegime
            .CRISIS_VOL
        ):

            severity = (
                ScenarioSeverity
                .EXTREME
            )

        else:

            severity = (
                ScenarioSeverity
                .SEVERE
            )

        status = (
            StressResultStatus
            .PASSED
        )

        if (
            stressed_vol
            >
            current_vol
            * 2.0
        ):

            status = (
                StressResultStatus
                .FAILED
            )

        elif (
            stressed_vol
            >
            current_vol
            * 1.3
        ):

            status = (
                StressResultStatus
                .WARNING
            )

        detail = (
            VolatilityRegimeDetail(

                current_regime=
                current_regime,

                stressed_regime=
                target_regime,

                current_volatility=
                current_vol,

                stressed_volatility=
                stressed_vol,

                volatility_change=
                vol_change,
            )
        )

        return (
            VolatilityRegimeStressResult(

                metadata=
                self.metadata,

                stress_type=
                StressTestType
                .VOLATILITY_REGIME,

                scenario_name=
                f"{current_regime.value}"
                f"_to_"
                f"{target_regime.value}",

                severity=
                severity,

                status=
                status,

                portfolio_return=
                shock_loss,

                portfolio_pnl=
                shock_loss,

                drawdown=
                expected_dd,

                var_impact=
                stressed_vol,

                volatility_impact=
                vol_change,

                diagnostics={

                    "lookback_days":
                    self.regime_config
                    .lookback_days,

                },

                original_volatility=
                current_vol,

                stressed_volatility=
                stressed_vol,

                volatility_multiplier=
                multiplier,

                regime_detail=
                detail,

                volatility_shock_loss=
                shock_loss,

                expected_drawdown=
                expected_dd,

                volatility_multiplier_used=
                multiplier,
            )
        )

    # --------------------------------------------------------
    # VIX Shock Scenario
    # --------------------------------------------------------

    def run_vix_shock(
        self,
        *,
        returns: pd.Series,
        vix_multiplier:
        float = 2.0,
    ) -> (
        VolatilityRegimeStressResult
    ):

        current_vol = (
            self.realized_volatility(
                returns
            )
        )

        stressed_vol = (
            current_vol
            * vix_multiplier
        )

        target_regime = (
            VolatilityRegime
            .CRISIS_VOL
        )

        return self.run_regime_stress(

            returns=
            returns,

            target_regime=
            target_regime,
        )

    # --------------------------------------------------------
    # Volatility Clustering Stress
    # --------------------------------------------------------

    def run_clustering_stress(
        self,
        *,
        returns: pd.Series,
        cluster_multiplier:
        float = 1.75,
    ) -> (
        VolatilityRegimeStressResult
    ):

        current_vol = (
            self.realized_volatility(
                returns
            )
        )

        stressed_vol = (
            current_vol
            * cluster_multiplier
        )

        return (
            VolatilityRegimeStressResult(

                metadata=
                self.metadata,

                stress_type=
                StressTestType
                .VOLATILITY_REGIME,

                scenario_name=
                "volatility_clustering",

                severity=
                ScenarioSeverity
                .SEVERE,

                status=
                StressResultStatus
                .WARNING,

                portfolio_return=
                -(
                    stressed_vol
                    -
                    current_vol
                ),

                portfolio_pnl=
                -(
                    stressed_vol
                    -
                    current_vol
                ),

                drawdown=
                -2.0
                * stressed_vol,

                var_impact=
                stressed_vol,

                volatility_impact=
                stressed_vol
                -
                current_vol,

                original_volatility=
                current_vol,

                stressed_volatility=
                stressed_vol,

                volatility_multiplier=
                cluster_multiplier,

                volatility_multiplier_used=
                cluster_multiplier,
            )
        )

    # --------------------------------------------------------

    def run(
        self,
        *args,
        **kwargs,
    ) -> (
        VolatilityRegimeStressResult
    ):

        return self.run_regime_stress(
            *args,
            **kwargs,
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def volatility_regime_stress_test(
    *,
    metadata:
    StressMetadata,
    returns:
    pd.Series,
    target_regime:
    VolatilityRegime =
    VolatilityRegime.CRISIS_VOL,
    config:
    StressTestingConfig | None = None,
) -> (
    VolatilityRegimeStressResult
):

    engine = (
        VolatilityRegimeStressEngine(

            metadata=
            metadata,

            config=
            config,

        )
    )

    return (
        engine
        .run_regime_stress(

            returns=
            returns,

            target_regime=
            target_regime,

        )
    )


# ============================================================
# PART 9
# MONTE CARLO STRESS TESTING
# ============================================================

# ============================================================
# MONTE CARLO CONFIG
# ============================================================

@dataclass(slots=True)
class MonteCarloConfig:

    num_simulations: int = 10000

    horizon_days: int = 252

    confidence_level: float = 0.95

    random_seed: int = 42

    use_antithetic: bool = True

    use_regime_scaling: bool = False

    volatility_multiplier: float = 1.0


# ============================================================
# PATH RESULT
# ============================================================

@dataclass(slots=True)
class SimulationPathResult:

    path_id: int

    cumulative_return: float

    annualized_return: float

    annualized_volatility: float

    max_drawdown: float

    terminal_value: float


# ============================================================
# MONTE CARLO RESULT
# ============================================================

@dataclass(slots=True)
class MonteCarloStressResult(
    MonteCarloStressResult,
):

    var_95: float = 0.0

    var_99: float = 0.0

    expected_shortfall_95: float = 0.0

    expected_shortfall_99: float = 0.0

    worst_case_return: float = 0.0

    best_case_return: float = 0.0

    median_return: float = 0.0

    mean_return: float = 0.0

    mean_drawdown: float = 0.0

    tail_paths: list[
        SimulationPathResult
    ] = field(
        default_factory=list
    )


# ============================================================
# MONTE CARLO ENGINE
# ============================================================

class MonteCarloStressEngine(
    BaseStressTest,
):
    """
    Institutional Monte Carlo engine.

    Capabilities:

        Multivariate simulation
        Distribution estimation
        VaR / ES
        Tail extraction
        Drawdown analysis
    """

    # --------------------------------------------------------

    def __init__(
        self,
        metadata: StressMetadata,
        config:
        StressTestingConfig | None = None,
        monte_carlo_config:
        MonteCarloConfig | None = None,
    ) -> None:

        super().__init__(
            metadata,
            config,
        )

        self.mc_config = (

            monte_carlo_config

            if monte_carlo_config
            is not None

            else MonteCarloConfig()

        )

    # --------------------------------------------------------

    @staticmethod
    def validate_returns(
        returns: pd.Series,
    ) -> None:

        StressValidation.validate_series(
            returns,
            "returns",
        )

    # --------------------------------------------------------
    # Historical Statistics
    # --------------------------------------------------------

    @staticmethod
    def estimate_parameters(
        returns: pd.Series,
    ) -> tuple[
        float,
        float,
    ]:

        mu = float(
            returns.mean()
        )

        sigma = float(
            returns.std()
        )

        return (
            mu,
            sigma,
        )

    # --------------------------------------------------------
    # Simulate Paths
    # --------------------------------------------------------

    def generate_paths(
        self,
        *,
        mu: float,
        sigma: float,
    ) -> np.ndarray:

        cfg = self.mc_config

        np.random.seed(
            cfg.random_seed
        )

        simulations = (
            cfg.num_simulations
        )

        horizon = (
            cfg.horizon_days
        )

        sigma *= (
            cfg.volatility_multiplier
        )

        shocks = np.random.normal(

            loc=mu,

            scale=sigma,

            size=(
                simulations,
                horizon,
            ),

        )

        if (
            cfg.use_antithetic
        ):

            anti = -shocks

            shocks = np.vstack(
                [
                    shocks,
                    anti,
                ]
            )

        return shocks

    # --------------------------------------------------------
    # Path Analytics
    # --------------------------------------------------------

    @staticmethod
    def analyze_path(
        path_id: int,
        returns: np.ndarray,
    ) -> SimulationPathResult:

        cumulative_curve = np.cumprod(
            1.0 + returns
        )

        terminal_value = float(
            cumulative_curve[-1]
        )

        cumulative_return = (
            terminal_value
            - 1.0
        )

        annualized_return = float(

            (
                terminal_value
                **
                (
                    TRADING_DAYS
                    /
                    max(
                        len(
                            returns
                        ),
                        1,
                    )
                )
            )

            - 1.0

        )

        annualized_vol = float(

            np.std(
                returns
            )

            * np.sqrt(
                TRADING_DAYS
            )

        )

        running_max = np.maximum.accumulate(
            cumulative_curve
        )

        drawdowns = (

            cumulative_curve
            /
            running_max

            - 1.0

        )

        max_dd = float(
            np.min(
                drawdowns
            )
        )

        return (
            SimulationPathResult(

                path_id=
                path_id,

                cumulative_return=
                cumulative_return,

                annualized_return=
                annualized_return,

                annualized_volatility=
                annualized_vol,

                max_drawdown=
                max_dd,

                terminal_value=
                terminal_value,
            )
        )

    # --------------------------------------------------------
    # VaR
    # --------------------------------------------------------

    @staticmethod
    def compute_var(
        returns:
        np.ndarray,
        confidence:
        float,
    ) -> float:

        percentile = (
            (
                1.0
                -
                confidence
            )
            * 100.0
        )

        return float(

            np.percentile(

                returns,

                percentile,

            )

        )

    # --------------------------------------------------------
    # ES
    # --------------------------------------------------------

    @staticmethod
    def compute_expected_shortfall(
        returns:
        np.ndarray,
        confidence:
        float,
    ) -> float:

        var = (
            MonteCarloStressEngine
            .compute_var(

                returns,

                confidence,

            )
        )

        tail = returns[
            returns <= var
        ]

        if len(
            tail
        ) == 0:

            return var

        return float(
            np.mean(
                tail
            )
        )

    # --------------------------------------------------------
    # Run Simulation
    # --------------------------------------------------------

    def run_monte_carlo(
        self,
        *,
        returns: pd.Series,
    ) -> (
        MonteCarloStressResult
    ):

        self.validate_returns(
            returns
        )

        mu, sigma = (
            self.estimate_parameters(
                returns
            )
        )

        paths = (
            self.generate_paths(

                mu=mu,

                sigma=sigma,

            )
        )

        path_results = []

        cumulative_returns = []

        drawdowns = []

        for idx in range(
            len(paths)
        ):

            result = (
                self.analyze_path(

                    idx,

                    paths[idx],

                )
            )

            path_results.append(
                result
            )

            cumulative_returns.append(

                result
                .cumulative_return

            )

            drawdowns.append(

                result
                .max_drawdown

            )

        cumulative_returns = np.asarray(
            cumulative_returns
        )

        drawdowns = np.asarray(
            drawdowns
        )

        var95 = (
            self.compute_var(

                cumulative_returns,

                0.95,

            )
        )

        var99 = (
            self.compute_var(

                cumulative_returns,

                0.99,

            )
        )

        es95 = (
            self.compute_expected_shortfall(

                cumulative_returns,

                0.95,

            )
        )

        es99 = (
            self.compute_expected_shortfall(

                cumulative_returns,

                0.99,

            )
        )

        tail_cutoff = np.percentile(

            cumulative_returns,

            1.0,

        )

        tail_paths = [

            p

            for p in path_results

            if p.cumulative_return
            <= tail_cutoff

        ]

        status = (
            StressResultStatus
            .PASSED
        )

        if (
            var95
            < -0.40
        ):

            status = (
                StressResultStatus
                .FAILED
            )

        elif (
            var95
            < -0.20
        ):

            status = (
                StressResultStatus
                .WARNING
            )

        return (
            MonteCarloStressResult(

                metadata=
                self.metadata,

                stress_type=
                StressTestType
                .MONTE_CARLO,

                scenario_name=
                "Monte Carlo",

                severity=
                ScenarioSeverity
                .SEVERE,

                status=
                status,

                portfolio_return=
                float(
                    np.mean(
                        cumulative_returns
                    )
                ),

                portfolio_pnl=
                float(
                    np.mean(
                        cumulative_returns
                    )
                ),

                drawdown=
                float(
                    np.mean(
                        drawdowns
                    )
                ),

                var_impact=
                abs(
                    var95
                ),

                volatility_impact=
                sigma,

                diagnostics={

                    "simulations":
                    len(
                        path_results
                    ),

                },

                simulation_count=
                len(
                    path_results
                ),

                horizon_days=
                self.mc_config
                .horizon_days,

                confidence_level=
                self.mc_config
                .confidence_level,

                var_95=
                var95,

                var_99=
                var99,

                expected_shortfall_95=
                es95,

                expected_shortfall_99=
                es99,

                worst_case_return=
                float(
                    np.min(
                        cumulative_returns
                    )
                ),

                best_case_return=
                float(
                    np.max(
                        cumulative_returns
                    )
                ),

                median_return=
                float(
                    np.median(
                        cumulative_returns
                    )
                ),

                mean_return=
                float(
                    np.mean(
                        cumulative_returns
                    )
                ),

                mean_drawdown=
                float(
                    np.mean(
                        drawdowns
                    )
                ),

                tail_paths=
                tail_paths,
            )
        )

    # --------------------------------------------------------

    def run(
        self,
        *args,
        **kwargs,
    ) -> (
        MonteCarloStressResult
    ):

        return (
            self.run_monte_carlo(
                *args,
                **kwargs,
            )
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def monte_carlo_stress_test(
    *,
    metadata:
    StressMetadata,

    returns:
    pd.Series,

    config:
    StressTestingConfig | None = None,

    monte_carlo_config:
    MonteCarloConfig | None = None,
) -> (
    MonteCarloStressResult
):

    engine = (
        MonteCarloStressEngine(

            metadata=
            metadata,

            config=
            config,

            monte_carlo_config=
            monte_carlo_config,

        )
    )

    return (
        engine.run_monte_carlo(

            returns=
            returns,

        )
    )


# ============================================================
# PART 10
# TAIL RISK & EXTREME EVENT ENGINE
# ============================================================

# ============================================================
# TAIL RISK CONFIG
# ============================================================

@dataclass(slots=True)
class TailRiskConfig:

    tail_percentile: float = 0.01

    extreme_percentile: float = 0.005

    crash_threshold: float = -0.10

    severe_crash_threshold: float = -0.20

    black_swan_threshold: float = -0.40

    jump_multiplier: float = 3.0

    min_tail_observations: int = 20


# ============================================================
# EXTREME EVENT
# ============================================================

@dataclass(slots=True)
class ExtremeEvent:

    event_name: str

    shock_return: float

    probability: float

    severity: ScenarioSeverity

    description: str = ""


# ============================================================
# TAIL EVENT DETAIL
# ============================================================

@dataclass(slots=True)
class TailEventDetail:

    threshold: float

    count: int

    probability: float

    average_loss: float

    worst_loss: float


# ============================================================
# RESULT
# ============================================================

@dataclass(slots=True)
class TailRiskStressResult(
    TailRiskStressResult,
):

    tail_var: float = 0.0

    tail_expected_shortfall: float = 0.0

    crash_probability: float = 0.0

    severe_crash_probability: float = 0.0

    black_swan_probability: float = 0.0

    tail_index: float = 0.0

    jump_loss_estimate: float = 0.0

    event_details: list[
        TailEventDetail
    ] = field(
        default_factory=list
    )


# ============================================================
# DEFAULT EXTREME EVENTS
# ============================================================

DEFAULT_EXTREME_EVENTS = [

    ExtremeEvent(
        event_name="Flash Crash",
        shock_return=-0.10,
        probability=0.01,
        severity=
        ScenarioSeverity.SEVERE,
    ),

    ExtremeEvent(
        event_name="Liquidity Freeze",
        shock_return=-0.15,
        probability=0.005,
        severity=
        ScenarioSeverity.SEVERE,
    ),

    ExtremeEvent(
        event_name="Financial Crisis",
        shock_return=-0.35,
        probability=0.001,
        severity=
        ScenarioSeverity.EXTREME,
    ),

    ExtremeEvent(
        event_name="Black Swan",
        shock_return=-0.50,
        probability=0.0005,
        severity=
        ScenarioSeverity.EXTREME,
    ),
]


# ============================================================
# TAIL RISK ENGINE
# ============================================================

class TailRiskEngine(
    BaseStressTest,
):
    """
    Institutional tail-risk engine.

    Measures:

        Tail VaR
        Tail ES
        Crash probabilities
        Jump risk
        Black swan exposure
    """

    # --------------------------------------------------------

    def __init__(
        self,
        metadata: StressMetadata,
        config:
        StressTestingConfig | None = None,
        tail_config:
        TailRiskConfig | None = None,
    ) -> None:

        super().__init__(
            metadata,
            config,
        )

        self.tail_config = (

            tail_config

            if tail_config
            is not None

            else TailRiskConfig()

        )

    # --------------------------------------------------------

    @staticmethod
    def validate_returns(
        returns: pd.Series,
    ) -> None:

        StressValidation.validate_series(
            returns,
            "returns",
        )

    # --------------------------------------------------------
    # Tail VaR
    # --------------------------------------------------------

    def tail_var(
        self,
        returns: pd.Series,
    ) -> float:

        percentile = (

            self.tail_config
            .tail_percentile

            * 100.0

        )

        return float(

            np.percentile(
                returns,
                percentile,
            )

        )

    # --------------------------------------------------------
    # Tail ES
    # --------------------------------------------------------

    def tail_expected_shortfall(
        self,
        returns: pd.Series,
    ) -> float:

        var = (
            self.tail_var(
                returns
            )
        )

        tail = returns[
            returns <= var
        ]

        if len(
            tail
        ) == 0:

            return var

        return float(
            tail.mean()
        )

    # --------------------------------------------------------
    # Crash Probability
    # --------------------------------------------------------

    @staticmethod
    def probability_below(
        returns: pd.Series,
        threshold: float,
    ) -> float:

        return float(

            np.mean(
                returns
                <= threshold
            )

        )

    # --------------------------------------------------------
    # Tail Index
    # --------------------------------------------------------

    def estimate_tail_index(
        self,
        returns: pd.Series,
    ) -> float:
        """
        Simplified Hill estimator.
        """

        losses = np.sort(

            np.abs(

                returns[
                    returns < 0
                ]

            )

        )

        if len(
            losses
        ) < max(
            self.tail_config
            .min_tail_observations,
            5,
        ):

            return 0.0

        k = min(
            len(losses) // 5,
            50,
        )

        tail = losses[-k:]

        threshold = tail[0]

        hill = np.mean(

            np.log(
                tail
                /
                threshold
            )

        )

        if hill <= 0:

            return 0.0

        return float(
            1.0 / hill
        )

    # --------------------------------------------------------
    # Jump Loss
    # --------------------------------------------------------

    def jump_loss_estimate(
        self,
        returns: pd.Series,
    ) -> float:

        sigma = float(
            returns.std()
        )

        return float(

            -self.tail_config
            .jump_multiplier

            * sigma

        )

    # --------------------------------------------------------
    # Tail Details
    # --------------------------------------------------------

    def tail_event_details(
        self,
        returns: pd.Series,
    ) -> list[
        TailEventDetail
    ]:

        thresholds = [

            self.tail_config
            .crash_threshold,

            self.tail_config
            .severe_crash_threshold,

            self.tail_config
            .black_swan_threshold,
        ]

        results = []

        for threshold in thresholds:

            subset = returns[
                returns
                <= threshold
            ]

            count = len(
                subset
            )

            prob = float(
                count
                /
                max(
                    len(
                        returns
                    ),
                    1,
                )
            )

            results.append(

                TailEventDetail(

                    threshold=
                    threshold,

                    count=
                    count,

                    probability=
                    prob,

                    average_loss=
                    float(
                        subset.mean()
                    )
                    if count > 0
                    else 0.0,

                    worst_loss=
                    float(
                        subset.min()
                    )
                    if count > 0
                    else 0.0,
                )

            )

        return results

    # --------------------------------------------------------
    # Extreme Event Stress
    # --------------------------------------------------------

    def extreme_event_loss(
        self,
        events:
        list[
            ExtremeEvent
        ]
        | None = None,
    ) -> float:

        if events is None:

            events = (
                DEFAULT_EXTREME_EVENTS
            )

        expected_loss = 0.0

        for event in events:

            expected_loss += (

                event.shock_return

                *
                event.probability

            )

        return float(
            expected_loss
        )

    # --------------------------------------------------------
    # Run Tail Stress
    # --------------------------------------------------------

    def run_tail_stress(
        self,
        *,
        returns:
        pd.Series,
        events:
        list[
            ExtremeEvent
        ]
        | None = None,
    ) -> (
        TailRiskStressResult
    ):

        self.validate_returns(
            returns
        )

        tail_var = (
            self.tail_var(
                returns
            )
        )

        tail_es = (
            self.tail_expected_shortfall(
                returns
            )
        )

        crash_prob = (

            self.probability_below(

                returns,

                self.tail_config
                .crash_threshold,

            )

        )

        severe_prob = (

            self.probability_below(

                returns,

                self.tail_config
                .severe_crash_threshold,

            )

        )

        black_swan_prob = (

            self.probability_below(

                returns,

                self.tail_config
                .black_swan_threshold,

            )

        )

        tail_index = (

            self.estimate_tail_index(
                returns
            )
        )

        jump_loss = (

            self.jump_loss_estimate(
                returns
            )
        )

        expected_extreme_loss = (

            self.extreme_event_loss(
                events
            )
        )

        details = (
            self.tail_event_details(
                returns
            )
        )

        status = (
            StressResultStatus
            .PASSED
        )

        if (
            black_swan_prob
            >
            0.01
        ):

            status = (
                StressResultStatus
                .FAILED
            )

        elif (
            crash_prob
            >
            0.05
        ):

            status = (
                StressResultStatus
                .WARNING
            )

        return (
            TailRiskStressResult(

                metadata=
                self.metadata,

                stress_type=
                StressTestType
                .TAIL_RISK,

                scenario_name=
                "Tail Risk",

                severity=
                ScenarioSeverity
                .EXTREME,

                status=
                status,

                portfolio_return=
                expected_extreme_loss,

                portfolio_pnl=
                expected_extreme_loss,

                drawdown=
                tail_es,

                var_impact=
                abs(
                    tail_var
                ),

                volatility_impact=
                abs(
                    jump_loss
                ),

                diagnostics={

                    "tail_observations":
                    len(
                        returns
                    )

                },

                tail_var=
                tail_var,

                tail_expected_shortfall=
                tail_es,

                crash_probability=
                crash_prob,

                severe_crash_probability=
                severe_prob,

                black_swan_probability=
                black_swan_prob,

                tail_index=
                tail_index,

                jump_loss_estimate=
                jump_loss,

                event_details=
                details,
            )
        )

    # --------------------------------------------------------

    def run(
        self,
        *args,
        **kwargs,
    ) -> (
        TailRiskStressResult
    ):

        return self.run_tail_stress(
            *args,
            **kwargs,
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def tail_risk_stress_test(
    *,
    metadata:
    StressMetadata,
    returns:
    pd.Series,
    events:
    list[
        ExtremeEvent
    ]
    | None = None,
    config:
    StressTestingConfig
    | None = None,
) -> (
    TailRiskStressResult
):

    engine = TailRiskEngine(

        metadata=
        metadata,

        config=
        config,
    )

    return engine.run_tail_stress(

        returns=
        returns,

        events=
        events,
    )

# ============================================================
# PART 11
# REVERSE STRESS TESTING
# ============================================================

# ============================================================
# CONFIG
# ============================================================

@dataclass(slots=True)
class ReverseStressConfig:

    target_drawdown: float = -0.20

    target_loss: float = -0.15

    target_var_breach: float = 0.25

    max_search_multiplier: float = 10.0

    search_steps: int = 500

    tolerance: float = 1e-4


# ============================================================
# BREAKPOINT RESULT
# ============================================================

@dataclass(slots=True)
class ReverseStressBreakpoint:

    stress_type: str

    required_shock: float

    resulting_loss: float

    description: str = ""


# ============================================================
# RESULT
# ============================================================

@dataclass(slots=True)
class ReverseStressResult(
    StressTestResult,
):

    required_market_shock: float = 0.0

    required_volatility_multiplier: float = 0.0

    required_correlation_level: float = 0.0

    required_liquidity_haircut: float = 0.0

    breakpoints: list[
        ReverseStressBreakpoint
    ] = field(
        default_factory=list
    )


# ============================================================
# ENGINE
# ============================================================

class ReverseStressEngine(
    BaseStressTest,
):
    """
    Institutional reverse stress testing.

    Answers:

        What market decline breaks us?
        What volatility regime breaks us?
        What liquidity collapse breaks us?
        What correlation spike breaks us?
    """

    # --------------------------------------------------------

    def __init__(
        self,
        metadata: StressMetadata,
        config:
        StressTestingConfig | None = None,
        reverse_config:
        ReverseStressConfig | None = None,
    ) -> None:

        super().__init__(
            metadata,
            config,
        )

        self.reverse_config = (

            reverse_config

            if reverse_config
            is not None

            else ReverseStressConfig()

        )

    # --------------------------------------------------------
    # Binary Search Utility
    # --------------------------------------------------------

    @staticmethod
    def solve_threshold(
        fn,
        target_value: float,
        low: float,
        high: float,
        steps: int = 200,
    ) -> float:

        for _ in range(
            steps
        ):

            mid = (
                low + high
            ) / 2.0

            value = fn(
                mid
            )

            if (
                value
                < target_value
            ):

                high = mid

            else:

                low = mid

        return float(
            (low + high)
            / 2.0
        )

    # --------------------------------------------------------
    # Market Shock Breakpoint
    # --------------------------------------------------------

    def market_shock_breakpoint(
        self,
        portfolio_beta: float,
    ) -> float:

        target_loss = abs(

            self.reverse_config
            .target_loss

        )

        return (

            target_loss

            /

            max(
                abs(
                    portfolio_beta
                ),
                EPSILON,
            )

        )

    # --------------------------------------------------------
    # Volatility Breakpoint
    # --------------------------------------------------------

    def volatility_breakpoint(
        self,
        current_volatility:
        float,
    ) -> float:

        target_var = (

            self.reverse_config
            .target_var_breach

        )

        return (

            target_var

            /

            max(
                current_volatility,
                EPSILON,
            )

        )

    # --------------------------------------------------------
    # Correlation Breakpoint
    # --------------------------------------------------------

    @staticmethod
    def correlation_breakpoint(
        diversification_ratio:
        float,
    ) -> float:

        return float(

            min(

                0.99,

                1.0

                -

                (
                    diversification_ratio
                    /
                    10.0
                ),

            )

        )

    # --------------------------------------------------------
    # Liquidity Breakpoint
    # --------------------------------------------------------

    def liquidity_breakpoint(
        self,
        current_haircut:
        float,
    ) -> float:

        target_loss = abs(

            self.reverse_config
            .target_loss

        )

        return float(

            min(

                1.0,

                current_haircut

                + target_loss,

            )

        )

    # --------------------------------------------------------
    # Run Reverse Stress
    # --------------------------------------------------------

    def run_reverse_stress(
        self,
        *,
        portfolio_beta: float,
        current_volatility:
        float,
        diversification_ratio:
        float,
        liquidity_haircut:
        float = 0.20,
    ) -> ReverseStressResult:

        market_shock = (

            self.market_shock_breakpoint(
                portfolio_beta
            )

        )

        vol_multiplier = (

            self.volatility_breakpoint(
                current_volatility
            )

        )

        correlation_level = (

            self.correlation_breakpoint(
                diversification_ratio
            )

        )

        liquidity_break = (

            self.liquidity_breakpoint(
                liquidity_haircut
            )

        )

        breakpoints = [

            ReverseStressBreakpoint(

                stress_type=
                "market_shock",

                required_shock=
                market_shock,

                resulting_loss=
                self.reverse_config
                .target_loss,

                description=
                "Market decline required to breach loss target",

            ),

            ReverseStressBreakpoint(

                stress_type=
                "volatility_regime",

                required_shock=
                vol_multiplier,

                resulting_loss=
                self.reverse_config
                .target_loss,

                description=
                "Volatility increase multiplier",

            ),

            ReverseStressBreakpoint(

                stress_type=
                "correlation_spike",

                required_shock=
                correlation_level,

                resulting_loss=
                self.reverse_config
                .target_loss,

                description=
                "Average correlation required",

            ),

            ReverseStressBreakpoint(

                stress_type=
                "liquidity_collapse",

                required_shock=
                liquidity_break,

                resulting_loss=
                self.reverse_config
                .target_loss,

                description=
                "Liquidity haircut required",

            ),

        ]

        severity = (
            ScenarioSeverity
            .EXTREME
        )

        return ReverseStressResult(

            metadata=
            self.metadata,

            stress_type=
            StressTestType
            .REVERSE_STRESS,

            scenario_name=
            "Reverse Stress Test",

            severity=
            severity,

            status=
            StressResultStatus
            .PASSED,

            portfolio_return=
            self.reverse_config
            .target_loss,

            portfolio_pnl=
            self.reverse_config
            .target_loss,

            drawdown=
            self.reverse_config
            .target_drawdown,

            var_impact=
            self.reverse_config
            .target_var_breach,

            volatility_impact=
            vol_multiplier,

            diagnostics={

                "target_loss":
                self.reverse_config
                .target_loss,

                "target_drawdown":
                self.reverse_config
                .target_drawdown,

            },

            required_market_shock=
            market_shock,

            required_volatility_multiplier=
            vol_multiplier,

            required_correlation_level=
            correlation_level,

            required_liquidity_haircut=
            liquidity_break,

            breakpoints=
            breakpoints,
        )

    # --------------------------------------------------------

    def run(
        self,
        *args,
        **kwargs,
    ) -> ReverseStressResult:

        return self.run_reverse_stress(
            *args,
            **kwargs,
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def reverse_stress_test(
    *,
    metadata:
    StressMetadata,

    portfolio_beta:
    float,

    current_volatility:
    float,

    diversification_ratio:
    float,

    liquidity_haircut:
    float = 0.20,

    config:
    StressTestingConfig
    | None = None,
) -> ReverseStressResult:

    engine = ReverseStressEngine(

        metadata=
        metadata,

        config=
        config,

    )

    return engine.run_reverse_stress(

        portfolio_beta=
        portfolio_beta,

        current_volatility=
        current_volatility,

        diversification_ratio=
        diversification_ratio,

        liquidity_haircut=
        liquidity_haircut,
    )


# ============================================================
# PART 12
# PORTFOLIO VULNERABILITY ANALYTICS
# ============================================================

# ============================================================
# VULNERABILITY CONFIG
# ============================================================

@dataclass(slots=True)
class VulnerabilityConfig:
    """
    Composite vulnerability scoring.
    """

    factor_weight: float = 0.20

    liquidity_weight: float = 0.20

    volatility_weight: float = 0.20

    correlation_weight: float = 0.15

    tail_weight: float = 0.15

    concentration_weight: float = 0.10

    max_score: float = 100.0


# ============================================================
# VULNERABILITY COMPONENT
# ============================================================

@dataclass(slots=True)
class VulnerabilityComponent:

    component_name: str

    raw_value: float

    normalized_score: float

    weight: float

    weighted_score: float


# ============================================================
# VULNERABILITY REPORT
# ============================================================

@dataclass(slots=True)
class VulnerabilityReport:

    overall_score: float

    vulnerability_level: str

    components: list[
        VulnerabilityComponent
    ]

    diagnostics: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )


# ============================================================
# EXTENDED RESULT
# ============================================================

@dataclass(slots=True)
class PortfolioVulnerabilityResult(
    StressTestResult,
):

    overall_score: float = 0.0

    vulnerability_level: str = ""

    factor_score: float = 0.0

    liquidity_score: float = 0.0

    volatility_score: float = 0.0

    correlation_score: float = 0.0

    tail_score: float = 0.0

    concentration_score: float = 0.0

    report: VulnerabilityReport | None = None


# ============================================================
# ENGINE
# ============================================================

class PortfolioVulnerabilityEngine(
    BaseStressTest,
):
    """
    Institutional vulnerability engine.

    Aggregates all stress outputs.

    Produces:

        Overall vulnerability score
        Risk ranking
        Weakest portfolio dimension
    """

    # --------------------------------------------------------

    def __init__(
        self,
        metadata: StressMetadata,
        config:
        StressTestingConfig | None = None,
        vulnerability_config:
        VulnerabilityConfig | None = None,
    ) -> None:

        super().__init__(
            metadata,
            config,
        )

        self.vulnerability_config = (

            vulnerability_config

            if vulnerability_config
            is not None

            else VulnerabilityConfig()

        )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    @staticmethod
    def normalize(
        value: float,
        min_value: float,
        max_value: float,
    ) -> float:

        if (
            max_value
            <= min_value
        ):

            return 0.0

        score = (

            value
            -
            min_value

        ) / (

            max_value
            -
            min_value

        )

        return float(

            np.clip(
                score,
                0.0,
                1.0,
            )

        )

    # --------------------------------------------------------
    # Factor Vulnerability
    # --------------------------------------------------------

    @staticmethod
    def factor_score(
        factor_result:
        FactorStressResult
        | None,
    ) -> float:

        if factor_result is None:

            return 0.0

        return float(

            min(

                abs(
                    factor_result
                    .total_factor_pnl
                ),

                1.0,

            )

        )

    # --------------------------------------------------------
    # Liquidity Vulnerability
    # --------------------------------------------------------

    @staticmethod
    def liquidity_score(
        liquidity_result:
        LiquidityStressAnalysisResult
        | None,
    ) -> float:

        if liquidity_result is None:

            return 0.0

        return float(

            min(

                liquidity_result
                .max_liquidation_days

                / 30.0,

                1.0,

            )

        )

    # --------------------------------------------------------
    # Volatility Vulnerability
    # --------------------------------------------------------

    @staticmethod
    def volatility_score(
        vol_result:
        VolatilityRegimeStressResult
        | None,
    ) -> float:

        if vol_result is None:

            return 0.0

        return float(

            min(

                abs(
                    vol_result
                    .volatility_multiplier_used
                )

                / 5.0,

                1.0,

            )

        )

    # --------------------------------------------------------
    # Correlation Vulnerability
    # --------------------------------------------------------

    @staticmethod
    def correlation_score(
        corr_result:
        CorrelationStressAnalysisResult
        | None,
    ) -> float:

        if corr_result is None:

            return 0.0

        return float(

            min(

                corr_result
                .stressed_avg_correlation,

                1.0,

            )

        )

    # --------------------------------------------------------
    # Tail Vulnerability
    # --------------------------------------------------------

    @staticmethod
    def tail_score(
        tail_result:
        TailRiskStressResult
        | None,
    ) -> float:

        if tail_result is None:

            return 0.0

        return float(

            min(

                abs(
                    tail_result
                    .tail_expected_shortfall
                ),

                1.0,

            )

        )

    # --------------------------------------------------------
    # Concentration Vulnerability
    # --------------------------------------------------------

    @staticmethod
    def concentration_score(
        concentration_metric:
        float | None,
    ) -> float:

        if (
            concentration_metric
            is None
        ):

            return 0.0

        return float(

            min(

                concentration_metric,

                1.0,

            )

        )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    @staticmethod
    def classify(
        score: float,
    ) -> str:

        if score < 20:

            return "LOW"

        if score < 40:

            return "MODERATE"

        if score < 60:

            return "ELEVATED"

        if score < 80:

            return "HIGH"

        return "CRITICAL"

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    def run_vulnerability_analysis(
        self,
        *,
        factor_result:
        FactorStressResult | None = None,

        liquidity_result:
        LiquidityStressAnalysisResult
        | None = None,

        volatility_result:
        VolatilityRegimeStressResult
        | None = None,

        correlation_result:
        CorrelationStressAnalysisResult
        | None = None,

        tail_result:
        TailRiskStressResult
        | None = None,

        concentration_metric:
        float | None = None,
    ) -> (
        PortfolioVulnerabilityResult
    ):

        cfg = (
            self
            .vulnerability_config
        )

        factor_score = (
            self.factor_score(
                factor_result
            )
        )

        liquidity_score = (
            self.liquidity_score(
                liquidity_result
            )
        )

        volatility_score = (
            self.volatility_score(
                volatility_result
            )
        )

        correlation_score = (
            self.correlation_score(
                correlation_result
            )
        )

        tail_score = (
            self.tail_score(
                tail_result
            )
        )

        concentration_score = (
            self.concentration_score(
                concentration_metric
            )
        )

        components = []

        def add_component(
            name: str,
            raw: float,
            weight: float,
        ):

            weighted = (
                raw
                * weight
                * cfg.max_score
            )

            components.append(

                VulnerabilityComponent(

                    component_name=
                    name,

                    raw_value=
                    raw,

                    normalized_score=
                    raw,

                    weight=
                    weight,

                    weighted_score=
                    weighted,
                )

            )

        add_component(
            "factor",
            factor_score,
            cfg.factor_weight,
        )

        add_component(
            "liquidity",
            liquidity_score,
            cfg.liquidity_weight,
        )

        add_component(
            "volatility",
            volatility_score,
            cfg.volatility_weight,
        )

        add_component(
            "correlation",
            correlation_score,
            cfg.correlation_weight,
        )

        add_component(
            "tail",
            tail_score,
            cfg.tail_weight,
        )

        add_component(
            "concentration",
            concentration_score,
            cfg.concentration_weight,
        )

        overall_score = float(

            np.sum(

                [
                    c.weighted_score
                    for c
                    in components
                ]

            )

        )

        vulnerability_level = (
            self.classify(
                overall_score
            )
        )

        report = (
            VulnerabilityReport(

                overall_score=
                overall_score,

                vulnerability_level=
                vulnerability_level,

                components=
                components,

                diagnostics={

                    "component_count":
                    len(
                        components
                    )

                },
            )
        )

        return (
            PortfolioVulnerabilityResult(

                metadata=
                self.metadata,

                stress_type=
                StressTestType
                .VULNERABILITY,

                scenario_name=
                "Portfolio Vulnerability",

                severity=
                ScenarioSeverity
                .SEVERE,

                status=
                StressResultStatus
                .PASSED,

                portfolio_return=
                -overall_score
                / 100.0,

                portfolio_pnl=
                -overall_score
                / 100.0,

                drawdown=
                -overall_score
                / 100.0,

                var_impact=
                overall_score
                / 100.0,

                volatility_impact=
                overall_score
                / 100.0,

                diagnostics={

                    "vulnerability_level":
                    vulnerability_level,

                },

                overall_score=
                overall_score,

                vulnerability_level=
                vulnerability_level,

                factor_score=
                factor_score,

                liquidity_score=
                liquidity_score,

                volatility_score=
                volatility_score,

                correlation_score=
                correlation_score,

                tail_score=
                tail_score,

                concentration_score=
                concentration_score,

                report=
                report,
            )
        )

    # --------------------------------------------------------

    def run(
        self,
        *args,
        **kwargs,
    ) -> (
        PortfolioVulnerabilityResult
    ):

        return (
            self
            .run_vulnerability_analysis(
                *args,
                **kwargs,
            )
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def portfolio_vulnerability_analysis(
    *,
    metadata:
    StressMetadata,

    factor_result:
    FactorStressResult | None = None,

    liquidity_result:
    LiquidityStressAnalysisResult
    | None = None,

    volatility_result:
    VolatilityRegimeStressResult
    | None = None,

    correlation_result:
    CorrelationStressAnalysisResult
    | None = None,

    tail_result:
    TailRiskStressResult
    | None = None,

    concentration_metric:
    float | None = None,

    config:
    StressTestingConfig
    | None = None,
) -> (
    PortfolioVulnerabilityResult
):

    engine = (
        PortfolioVulnerabilityEngine(

            metadata=
            metadata,

            config=
            config,

        )
    )

    return (
        engine
        .run_vulnerability_analysis(

            factor_result=
            factor_result,

            liquidity_result=
            liquidity_result,

            volatility_result=
            volatility_result,

            correlation_result=
            correlation_result,

            tail_result=
            tail_result,

            concentration_metric=
            concentration_metric,
        )
    )


# ============================================================
# PART 13
# INSTITUTIONAL MASTER STRESS REPORT
# ============================================================

# ============================================================
# REPORT SUMMARY
# ============================================================

@dataclass(slots=True)
class StressSummary:

    overall_risk_score: float

    worst_scenario_name: str

    worst_scenario_loss: float

    average_stress_loss: float

    average_drawdown: float

    vulnerability_level: str

    report_timestamp: datetime


# ============================================================
# REPORT SECTION
# ============================================================

@dataclass(slots=True)
class StressReportSection:

    section_name: str

    metrics: dict[
        str,
        Any,
    ]

    comments: str = ""


# ============================================================
# MASTER REPORT
# ============================================================

@dataclass(slots=True)
class InstitutionalStressReport:

    metadata: StressMetadata

    summary: StressSummary

    sections: list[
        StressReportSection
    ] = field(
        default_factory=list
    )

    historical_result: HistoricalStressResult | None = None

    scenario_result: ScenarioStressResult | None = None

    factor_result: FactorStressResult | None = None

    correlation_result: CorrelationStressAnalysisResult | None = None

    liquidity_result: LiquidityStressAnalysisResult | None = None

    volatility_result: VolatilityRegimeStressResult | None = None

    monte_carlo_result: MonteCarloStressResult | None = None

    tail_result: TailRiskStressResult | None = None

    reverse_result: ReverseStressResult | None = None

    vulnerability_result: PortfolioVulnerabilityResult | None = None


# ============================================================
# REPORT BUILDER
# ============================================================

class InstitutionalStressReportBuilder:
    """
    Institutional reporting layer.

    Combines all stress outputs into a
    single enterprise report.
    """

    # --------------------------------------------------------

    @staticmethod
    def collect_losses(
        results: list[
            StressTestResult
            | None
        ],
    ) -> np.ndarray:

        losses = []

        for result in results:

            if result is None:

                continue

            losses.append(
                float(
                    result.portfolio_return
                )
            )

        return np.asarray(
            losses,
            dtype=float,
        )

    # --------------------------------------------------------

    @staticmethod
    def collect_drawdowns(
        results: list[
            StressTestResult
            | None
        ],
    ) -> np.ndarray:

        values = []

        for result in results:

            if result is None:

                continue

            values.append(
                float(
                    result.drawdown
                )
            )

        return np.asarray(
            values,
            dtype=float,
        )

    # --------------------------------------------------------

    @staticmethod
    def determine_worst_scenario(
        results: list[
            StressTestResult
            | None
        ],
    ) -> tuple[
        str,
        float,
    ]:

        worst_name = ""

        worst_loss = 0.0

        for result in results:

            if result is None:

                continue

            pnl = float(
                result.portfolio_return
            )

            if pnl < worst_loss:

                worst_loss = pnl

                worst_name = (
                    result.scenario_name
                )

        return (
            worst_name,
            worst_loss,
        )

    # --------------------------------------------------------

    @staticmethod
    def build_summary(
        *,
        results: list[
            StressTestResult
            | None
        ],
        vulnerability_result:
        PortfolioVulnerabilityResult
        | None,
    ) -> StressSummary:

        losses = (
            InstitutionalStressReportBuilder
            .collect_losses(
                results
            )
        )

        drawdowns = (
            InstitutionalStressReportBuilder
            .collect_drawdowns(
                results
            )
        )

        worst_name, worst_loss = (

            InstitutionalStressReportBuilder
            .determine_worst_scenario(
                results
            )

        )

        avg_loss = float(

            np.mean(
                losses
            )

        ) if len(
            losses
        ) else 0.0

        avg_dd = float(

            np.mean(
                drawdowns
            )

        ) if len(
            drawdowns
        ) else 0.0

        vulnerability_score = (

            vulnerability_result
            .overall_score

            if vulnerability_result
            is not None

            else 0.0

        )

        vulnerability_level = (

            vulnerability_result
            .vulnerability_level

            if vulnerability_result
            is not None

            else "UNKNOWN"

        )

        return StressSummary(

            overall_risk_score=
            vulnerability_score,

            worst_scenario_name=
            worst_name,

            worst_scenario_loss=
            worst_loss,

            average_stress_loss=
            avg_loss,

            average_drawdown=
            avg_dd,

            vulnerability_level=
            vulnerability_level,

            report_timestamp=
            datetime.utcnow(),
        )

    # --------------------------------------------------------

    @staticmethod
    def historical_section(
        result:
        HistoricalStressResult
        | None,
    ) -> (
        StressReportSection
        | None
    ):

        if result is None:

            return None

        return StressReportSection(

            section_name=
            "Historical Stress",

            metrics={

                "return":
                result.portfolio_return,

                "drawdown":
                result.drawdown,

                "status":
                result.status.value,
            },
        )

    # --------------------------------------------------------

    @staticmethod
    def factor_section(
        result:
        FactorStressResult
        | None,
    ) -> (
        StressReportSection
        | None
    ):

        if result is None:

            return None

        return StressReportSection(

            section_name=
            "Factor Stress",

            metrics={

                "factor_pnl":
                result.total_factor_pnl,

                "status":
                result.status.value,
            },
        )

    # --------------------------------------------------------

    @staticmethod
    def liquidity_section(
        result:
        LiquidityStressAnalysisResult
        | None,
    ) -> (
        StressReportSection
        | None
    ):

        if result is None:

            return None

        return StressReportSection(

            section_name=
            "Liquidity Stress",

            metrics={

                "days_to_liquidate":
                result.average_liquidation_days,

                "capacity":
                result.stressed_capacity,

                "haircut_cost":
                result.total_haircut_cost,
            },
        )

    # --------------------------------------------------------

    @staticmethod
    def tail_section(
        result:
        TailRiskStressResult
        | None,
    ) -> (
        StressReportSection
        | None
    ):

        if result is None:

            return None

        return StressReportSection(

            section_name=
            "Tail Risk",

            metrics={

                "tail_var":
                result.tail_var,

                "tail_es":
                result.tail_expected_shortfall,

                "crash_prob":
                result.crash_probability,

                "black_swan_prob":
                result.black_swan_probability,
            },
        )

    # --------------------------------------------------------

    @staticmethod
    def vulnerability_section(
        result:
        PortfolioVulnerabilityResult
        | None,
    ) -> (
        StressReportSection
        | None
    ):

        if result is None:

            return None

        return StressReportSection(

            section_name=
            "Vulnerability",

            metrics={

                "score":
                result.overall_score,

                "level":
                result.vulnerability_level,

                "factor":
                result.factor_score,

                "liquidity":
                result.liquidity_score,

                "tail":
                result.tail_score,
            },
        )

    # --------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        metadata:
        StressMetadata,

        historical_result:
        HistoricalStressResult
        | None = None,

        scenario_result:
        ScenarioStressResult
        | None = None,

        factor_result:
        FactorStressResult
        | None = None,

        correlation_result:
        CorrelationStressAnalysisResult
        | None = None,

        liquidity_result:
        LiquidityStressAnalysisResult
        | None = None,

        volatility_result:
        VolatilityRegimeStressResult
        | None = None,

        monte_carlo_result:
        MonteCarloStressResult
        | None = None,

        tail_result:
        TailRiskStressResult
        | None = None,

        reverse_result:
        ReverseStressResult
        | None = None,

        vulnerability_result:
        PortfolioVulnerabilityResult
        | None = None,
    ) -> InstitutionalStressReport:

        all_results = [

            historical_result,
            scenario_result,
            factor_result,
            correlation_result,
            liquidity_result,
            volatility_result,
            monte_carlo_result,
            tail_result,
            reverse_result,
            vulnerability_result,
        ]

        summary = cls.build_summary(

            results=
            all_results,

            vulnerability_result=
            vulnerability_result,
        )

        sections = []

        for section in [

            cls.historical_section(
                historical_result
            ),

            cls.factor_section(
                factor_result
            ),

            cls.liquidity_section(
                liquidity_result
            ),

            cls.tail_section(
                tail_result
            ),

            cls.vulnerability_section(
                vulnerability_result
            ),

        ]:

            if section is not None:

                sections.append(
                    section
                )

        return InstitutionalStressReport(

            metadata=
            metadata,

            summary=
            summary,

            sections=
            sections,

            historical_result=
            historical_result,

            scenario_result=
            scenario_result,

            factor_result=
            factor_result,

            correlation_result=
            correlation_result,

            liquidity_result=
            liquidity_result,

            volatility_result=
            volatility_result,

            monte_carlo_result=
            monte_carlo_result,

            tail_result=
            tail_result,

            reverse_result=
            reverse_result,

            vulnerability_result=
            vulnerability_result,
        )


# ============================================================
# EXPORTER
# ============================================================

class StressReportExporter:

    # --------------------------------------------------------

    @staticmethod
    def to_dict(
        report:
        InstitutionalStressReport,
    ) -> dict[
        str,
        Any,
    ]:

        return asdict(
            report
        )

    # --------------------------------------------------------

    @staticmethod
    def summary_dataframe(
        report:
        InstitutionalStressReport,
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
    def sections_dataframe(
        report:
        InstitutionalStressReport,
    ) -> pd.DataFrame:

        rows = []

        for section in (
            report.sections
        ):

            row = {

                "section":
                section.section_name
            }

            row.update(
                section.metrics
            )

            rows.append(
                row
            )

        return pd.DataFrame(
            rows
        )

    # --------------------------------------------------------

    @staticmethod
    def to_json(
        report:
        InstitutionalStressReport,
    ) -> str:

        return json.dumps(

            StressReportExporter
            .to_dict(
                report
            ),

            default=str,

            indent=2,
        )


# ============================================================
# PART 14
# STRESS TESTING ENGINE
# ============================================================

# ============================================================
# ENGINE CONFIG
# ============================================================

@dataclass(slots=True)
class StressTestingEngineConfig:

    run_historical: bool = True

    run_scenario: bool = True

    run_factor: bool = True

    run_correlation: bool = True

    run_liquidity: bool = True

    run_volatility: bool = True

    run_monte_carlo: bool = True

    run_tail: bool = True

    run_reverse: bool = True

    run_vulnerability: bool = True


# ============================================================
# INPUT OBJECT
# ============================================================

@dataclass(slots=True)
class StressTestingInput:

    returns: pd.Series

    portfolio_weights: (
        pd.Series
        | None
    ) = None

    factor_exposures: (
        pd.DataFrame
        | None
    ) = None

    factor_returns: (
        pd.DataFrame
        | None
    ) = None

    correlation_matrix: (
        pd.DataFrame
        | None
    ) = None

    liquidity_profile: (
        pd.DataFrame
        | None
    ) = None

    portfolio_beta: float = 1.0

    diversification_ratio: float = 2.0

    concentration_metric: float = 0.20


# ============================================================
# MASTER ENGINE
# ============================================================

class InstitutionalStressTestingEngine:
    """
    Institutional-grade stress platform.

    Runs:

        Historical Stress
        Scenario Stress
        Factor Stress
        Correlation Stress
        Liquidity Stress
        Volatility Stress
        Monte Carlo
        Tail Risk
        Reverse Stress
        Vulnerability Analytics

    Produces:

        InstitutionalStressReport
    """

    # --------------------------------------------------------

    def __init__(
        self,
        metadata:
        StressMetadata,
        config:
        StressTestingConfig | None = None,
        engine_config:
        StressTestingEngineConfig
        | None = None,
    ) -> None:

        self.metadata = metadata

        self.config = config

        self.engine_config = (

            engine_config

            if engine_config
            is not None

            else StressTestingEngineConfig()

        )

    # --------------------------------------------------------
    # Historical Stress
    # --------------------------------------------------------

    def run_historical(
        self,
        inputs:
        StressTestingInput,
    ) -> (
        HistoricalStressResult
        | None
    ):

        if not (
            self.engine_config
            .run_historical
        ):

            return None

        try:

            engine = (
                HistoricalStressTester(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            return engine.run(

                returns=
                inputs.returns
            )

        except Exception:

            return None

    # --------------------------------------------------------
    # Scenario Stress
    # --------------------------------------------------------

    def run_scenario(
        self,
        inputs:
        StressTestingInput,
    ) -> (
        ScenarioStressResult
        | None
    ):

        if not (
            self.engine_config
            .run_scenario
        ):

            return None

        try:

            engine = (
                ScenarioStressTester(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            return engine.run(
                returns=
                inputs.returns
            )

        except Exception:

            return None

    # --------------------------------------------------------
    # Factor Stress
    # --------------------------------------------------------

    def run_factor(
        self,
        inputs:
        StressTestingInput,
    ) -> (
        FactorStressResult
        | None
    ):

        if (
            not self.engine_config
            .run_factor
        ):

            return None

        if (
            inputs.factor_exposures
            is None
        ):

            return None

        try:

            engine = (
                FactorShockEngine(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            return engine.run(
                factor_exposures=
                inputs.factor_exposures
            )

        except Exception:

            return None

    # --------------------------------------------------------
    # Correlation Stress
    # --------------------------------------------------------

    def run_correlation(
        self,
        inputs:
        StressTestingInput,
    ) -> (
        CorrelationStressAnalysisResult
        | None
    ):

        if (
            not self.engine_config
            .run_correlation
        ):

            return None

        if (
            inputs.correlation_matrix
            is None
        ):

            return None

        try:

            engine = (
                CorrelationBreakdownEngine(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            return engine.run(
                correlation_matrix=
                inputs.correlation_matrix
            )

        except Exception:

            return None

    # --------------------------------------------------------
    # Liquidity Stress
    # --------------------------------------------------------

    def run_liquidity(
        self,
        inputs:
        StressTestingInput,
    ) -> (
        LiquidityStressAnalysisResult
        | None
    ):

        if (
            not self.engine_config
            .run_liquidity
        ):

            return None

        if (
            inputs.liquidity_profile
            is None
        ):

            return None

        try:

            engine = (
                LiquidityStressEngine(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            return engine.run(
                liquidity_profile=
                inputs.liquidity_profile
            )

        except Exception:

            return None

    # --------------------------------------------------------
    # Volatility Stress
    # --------------------------------------------------------

    def run_volatility(
        self,
        inputs:
        StressTestingInput,
    ) -> (
        VolatilityRegimeStressResult
        | None
    ):

        if (
            not self.engine_config
            .run_volatility
        ):

            return None

        try:

            engine = (
                VolatilityRegimeStressEngine(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            return engine.run(
                returns=
                inputs.returns
            )

        except Exception:

            return None

    # --------------------------------------------------------
    # Monte Carlo
    # --------------------------------------------------------

    def run_monte_carlo(
        self,
        inputs:
        StressTestingInput,
    ) -> (
        MonteCarloStressResult
        | None
    ):

        if (
            not self.engine_config
            .run_monte_carlo
        ):

            return None

        try:

            engine = (
                MonteCarloStressEngine(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            return engine.run(
                returns=
                inputs.returns
            )

        except Exception:

            return None

    # --------------------------------------------------------
    # Tail Risk
    # --------------------------------------------------------

    def run_tail(
        self,
        inputs:
        StressTestingInput,
    ) -> (
        TailRiskStressResult
        | None
    ):

        if (
            not self.engine_config
            .run_tail
        ):

            return None

        try:

            engine = (
                TailRiskEngine(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            return engine.run(
                returns=
                inputs.returns
            )

        except Exception:

            return None

    # --------------------------------------------------------
    # Reverse Stress
    # --------------------------------------------------------

    def run_reverse(
        self,
        inputs:
        StressTestingInput,
    ) -> (
        ReverseStressResult
        | None
    ):

        if (
            not self.engine_config
            .run_reverse
        ):

            return None

        try:

            engine = (
                ReverseStressEngine(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            current_vol = float(
                inputs.returns.std()
                * np.sqrt(
                    TRADING_DAYS
                )
            )

            return engine.run(

                portfolio_beta=
                inputs.portfolio_beta,

                current_volatility=
                current_vol,

                diversification_ratio=
                inputs
                .diversification_ratio,
            )

        except Exception:

            return None

    # --------------------------------------------------------
    # Vulnerability
    # --------------------------------------------------------

    def run_vulnerability(
        self,
        *,
        factor_result:
        FactorStressResult
        | None,

        liquidity_result:
        LiquidityStressAnalysisResult
        | None,

        volatility_result:
        VolatilityRegimeStressResult
        | None,

        correlation_result:
        CorrelationStressAnalysisResult
        | None,

        tail_result:
        TailRiskStressResult
        | None,

        concentration_metric:
        float,
    ) -> (
        PortfolioVulnerabilityResult
        | None
    ):

        if (
            not self.engine_config
            .run_vulnerability
        ):

            return None

        try:

            engine = (
                PortfolioVulnerabilityEngine(

                    metadata=
                    self.metadata,

                    config=
                    self.config,
                )
            )

            return engine.run(

                factor_result=
                factor_result,

                liquidity_result=
                liquidity_result,

                volatility_result=
                volatility_result,

                correlation_result=
                correlation_result,

                tail_result=
                tail_result,

                concentration_metric=
                concentration_metric,
            )

        except Exception:

            return None

    # --------------------------------------------------------
    # MASTER RUN
    # --------------------------------------------------------

    def run(
        self,
        inputs:
        StressTestingInput,
    ) -> (
        InstitutionalStressReport
    ):

        historical_result = (
            self.run_historical(
                inputs
            )
        )

        scenario_result = (
            self.run_scenario(
                inputs
            )
        )

        factor_result = (
            self.run_factor(
                inputs
            )
        )

        correlation_result = (
            self.run_correlation(
                inputs
            )
        )

        liquidity_result = (
            self.run_liquidity(
                inputs
            )
        )

        volatility_result = (
            self.run_volatility(
                inputs
            )
        )

        monte_carlo_result = (
            self.run_monte_carlo(
                inputs
            )
        )

        tail_result = (
            self.run_tail(
                inputs
            )
        )

        reverse_result = (
            self.run_reverse(
                inputs
            )
        )

        vulnerability_result = (
            self.run_vulnerability(

                factor_result=
                factor_result,

                liquidity_result=
                liquidity_result,

                volatility_result=
                volatility_result,

                correlation_result=
                correlation_result,

                tail_result=
                tail_result,

                concentration_metric=
                inputs
                .concentration_metric,
            )
        )

        report = (
            InstitutionalStressReportBuilder
            .build(

                metadata=
                self.metadata,

                historical_result=
                historical_result,

                scenario_result=
                scenario_result,

                factor_result=
                factor_result,

                correlation_result=
                correlation_result,

                liquidity_result=
                liquidity_result,

                volatility_result=
                volatility_result,

                monte_carlo_result=
                monte_carlo_result,

                tail_result=
                tail_result,

                reverse_result=
                reverse_result,

                vulnerability_result=
                vulnerability_result,
            )
        )

        return report
    

# ============================================================
# PART 15
# FACTORY & CONVENIENCE APIS
# ============================================================

# ============================================================
# ENGINE FACTORY
# ============================================================

class StressTestingFactory:
    """
    Centralized factory for all
    stress-testing engines.
    """

    # --------------------------------------------------------

    @staticmethod
    def historical_engine(
        metadata:
        StressMetadata,
        config:
        StressTestingConfig
        | None = None,
    ) -> HistoricalStressTester:

        return HistoricalStressTester(
            metadata=metadata,
            config=config,
        )

    # --------------------------------------------------------

    @staticmethod
    def scenario_engine(
        metadata:
        StressMetadata,
        config:
        StressTestingConfig
        | None = None,
    ) -> ScenarioStressTester:

        return ScenarioStressTester(
            metadata=metadata,
            config=config,
        )

    # --------------------------------------------------------

    @staticmethod
    def factor_engine(
        metadata:
        StressMetadata,
        config:
        StressTestingConfig
        | None = None,
    ) -> FactorShockEngine:

        return FactorShockEngine(
            metadata=metadata,
            config=config,
        )

    # --------------------------------------------------------

    @staticmethod
    def correlation_engine(
        metadata:
        StressMetadata,
        config:
        StressTestingConfig
        | None = None,
    ) -> CorrelationBreakdownEngine:

        return CorrelationBreakdownEngine(
            metadata=metadata,
            config=config,
        )

    # --------------------------------------------------------

    @staticmethod
    def liquidity_engine(
        metadata:
        StressMetadata,
        config:
        StressTestingConfig
        | None = None,
    ) -> LiquidityStressEngine:

        return LiquidityStressEngine(
            metadata=metadata,
            config=config,
        )

    # --------------------------------------------------------

    @staticmethod
    def volatility_engine(
        metadata:
        StressMetadata,
        config:
        StressTestingConfig
        | None = None,
    ) -> VolatilityRegimeStressEngine:

        return VolatilityRegimeStressEngine(
            metadata=metadata,
            config=config,
        )

    # --------------------------------------------------------

    @staticmethod
    def monte_carlo_engine(
        metadata:
        StressMetadata,
        config:
        StressTestingConfig
        | None = None,
    ) -> MonteCarloStressEngine:

        return MonteCarloStressEngine(
            metadata=metadata,
            config=config,
        )

    # --------------------------------------------------------

    @staticmethod
    def tail_risk_engine(
        metadata:
        StressMetadata,
        config:
        StressTestingConfig
        | None = None,
    ) -> TailRiskEngine:

        return TailRiskEngine(
            metadata=metadata,
            config=config,
        )

    # --------------------------------------------------------

    @staticmethod
    def reverse_stress_engine(
        metadata:
        StressMetadata,
        config:
        StressTestingConfig
        | None = None,
    ) -> ReverseStressEngine:

        return ReverseStressEngine(
            metadata=metadata,
            config=config,
        )

    # --------------------------------------------------------

    @staticmethod
    def vulnerability_engine(
        metadata:
        StressMetadata,
        config:
        StressTestingConfig
        | None = None,
    ) -> PortfolioVulnerabilityEngine:

        return PortfolioVulnerabilityEngine(
            metadata=metadata,
            config=config,
        )

    # --------------------------------------------------------

    @staticmethod
    def institutional_engine(
        metadata:
        StressMetadata,
        config:
        StressTestingConfig
        | None = None,
        engine_config:
        StressTestingEngineConfig
        | None = None,
    ) -> InstitutionalStressTestingEngine:

        return InstitutionalStressTestingEngine(
            metadata=metadata,
            config=config,
            engine_config=engine_config,
        )


# ============================================================
# MASTER RUNNER
# ============================================================

def run_full_stress_suite(
    *,
    metadata:
    StressMetadata,

    inputs:
    StressTestingInput,

    config:
    StressTestingConfig
    | None = None,

    engine_config:
    StressTestingEngineConfig
    | None = None,
) -> InstitutionalStressReport:
    """
    Run complete institutional
    stress testing suite.
    """

    engine = (
        StressTestingFactory
        .institutional_engine(
            metadata=metadata,
            config=config,
            engine_config=
            engine_config,
        )
    )

    return engine.run(
        inputs
    )


# ============================================================
# TAIL-RISK SUITE
# ============================================================

def run_tail_risk_suite(
    *,
    metadata:
    StressMetadata,

    returns:
    pd.Series,

    config:
    StressTestingConfig
    | None = None,
) -> TailRiskStressResult:

    engine = (
        StressTestingFactory
        .tail_risk_engine(
            metadata,
            config,
        )
    )

    return engine.run(
        returns=returns
    )


# ============================================================
# MONTE CARLO SUITE
# ============================================================

def run_monte_carlo_suite(
    *,
    metadata:
    StressMetadata,

    returns:
    pd.Series,

    config:
    StressTestingConfig
    | None = None,
) -> MonteCarloStressResult:

    engine = (
        StressTestingFactory
        .monte_carlo_engine(
            metadata,
            config,
        )
    )

    return engine.run(
        returns=returns
    )


# ============================================================
# LIQUIDITY SUITE
# ============================================================

def run_liquidity_suite(
    *,
    metadata:
    StressMetadata,

    liquidity_profile:
    pd.DataFrame,

    config:
    StressTestingConfig
    | None = None,
) -> LiquidityStressAnalysisResult:

    engine = (
        StressTestingFactory
        .liquidity_engine(
            metadata,
            config,
        )
    )

    return engine.run(
        liquidity_profile=
        liquidity_profile
    )


# ============================================================
# VOLATILITY SUITE
# ============================================================

def run_volatility_suite(
    *,
    metadata:
    StressMetadata,

    returns:
    pd.Series,

    config:
    StressTestingConfig
    | None = None,
) -> VolatilityRegimeStressResult:

    engine = (
        StressTestingFactory
        .volatility_engine(
            metadata,
            config,
        )
    )

    return engine.run(
        returns=returns
    )


# ============================================================
# BOARD REPORT
# ============================================================

def run_board_report(
    *,
    metadata:
    StressMetadata,

    inputs:
    StressTestingInput,

    config:
    StressTestingConfig
    | None = None,
) -> str:
    """
    One-line API for CRO/CIO report.
    """

    report = (
        run_full_stress_suite(

            metadata=
            metadata,

            inputs=
            inputs,

            config=
            config,
        )
    )

    return (
        StressReportExporter
        .to_json(
            report
        )
    )


# ============================================================
# REPORT DATAFRAME
# ============================================================

def stress_report_dataframe(
    report:
    InstitutionalStressReport,
) -> pd.DataFrame:

    return (
        StressReportExporter
        .sections_dataframe(
            report
        )
    )


# ============================================================
# QUICK REPORT SUMMARY
# ============================================================

def stress_summary_dataframe(
    report:
    InstitutionalStressReport,
) -> pd.DataFrame:

    return (
        StressReportExporter
        .summary_dataframe(
            report
        )
    )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [

    # Core

    "StressTestingFactory",

    "InstitutionalStressTestingEngine",

    "StressTestingInput",

    "StressTestingEngineConfig",

    # Reporting

    "InstitutionalStressReport",

    "StressSummary",

    "StressReportExporter",

    # Convenience

    "run_full_stress_suite",

    "run_tail_risk_suite",

    "run_monte_carlo_suite",

    "run_liquidity_suite",

    "run_volatility_suite",

    "run_board_report",

    "stress_report_dataframe",

    "stress_summary_dataframe",
]