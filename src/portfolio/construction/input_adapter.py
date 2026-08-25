# ============================================================
# INPUT_ADAPTER.PY
# PART 1 — IMPORTS & CONTRACTS
# ============================================================

from __future__ import annotations

# ============================================================
# STANDARD LIBRARY
# ============================================================

from abc import ABC, abstractmethod

from dataclasses import dataclass, field

from datetime import datetime

from enum import Enum, auto

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Mapping,
    
)

from src.portfolio.construction.pipeline import (
    MarketDataInput,
    PortfolioInput,
    ForecastInput,
    FactorInput,
    LiquidityInput,
    ConstraintInput,
    PipelineInput,
    PipelineInputValidator
)

# ============================================================
# THIRD PARTY
# ============================================================

import numpy as np

import pandas as pd

# ============================================================
# INPUT ADAPTER VERSION
# ============================================================

INPUT_ADAPTER_VERSION = "1.0.0"

# ============================================================
# ADAPTER SOURCE TYPES
# ============================================================


class AdapterSourceType(Enum):
    """
    Supported upstream sources.
    """

    ALPHA_PIPELINE = auto()

    FORECAST_ENGINE = auto()

    RISK_MODEL = auto()

    PORTFOLIO_SYSTEM = auto()

    EXECUTION_SYSTEM = auto()

    DATA_WAREHOUSE = auto()

    CSV = auto()

    DATABASE = auto()

    API = auto()

    CUSTOM = auto()


# ============================================================
# ADAPTER STATUS
# ============================================================


class AdapterStatus(Enum):
    """
    Adapter execution status.
    """

    NOT_STARTED = auto()

    RUNNING = auto()

    COMPLETED = auto()

    FAILED = auto()


# ============================================================
# ADAPTER METADATA
# ============================================================


@dataclass(slots=True)
class AdapterMetadata:
    """
    Metadata attached to every adapter run.
    """

    source_name: str

    source_type: AdapterSourceType

    created_at: datetime = field(
        default_factory=datetime.utcnow
    )

    version: str = INPUT_ADAPTER_VERSION

    owner: str | None = None

    tags: list[str] = field(
        default_factory=list
    )


# ============================================================
# ADAPTER RESULT
# ============================================================


@dataclass(slots=True)
class AdapterResult:
    """
    Generic adapter output container.
    """

    status: AdapterStatus

    payload: Any = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: AdapterMetadata | None = None

    error_message: str | None = None


# ============================================================
# INPUT CONTRACT
# ============================================================


class InputAdapterContract(ABC):
    """
    Base contract for all institutional
    input adapters.
    """

    @abstractmethod
    def validate(
        self,
        data: Any,
    ) -> None:
        """
        Validate upstream data.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        data: Any,
    ) -> Any:
        """
        Transform upstream data into
        pipeline-compatible format.
        """
        raise NotImplementedError

    @abstractmethod
    def build(
        self,
        data: Any,
    ) -> AdapterResult:
        """
        Produce adapter output.
        """
        raise NotImplementedError


# ============================================================
# PIPELINE INPUT PROTOCOL
# ============================================================


class PipelineInputLike(Protocol):
    """
    Structural typing only.

    Avoids importing PipelineInput
    and creating circular dependencies.
    """

    market_data: Any

    forecast_data: Any

    factor_data: Any

    portfolio_data: Any

    liquidity_data: Any

    constraint_data: Any


# ============================================================
# DATAFRAME CONTRACT
# ============================================================


class DataFrameProvider(Protocol):
    """
    Any object exposing a dataframe.
    """

    def to_dataframe(
        self,
    ) -> pd.DataFrame:
        ...


# ============================================================
# NUMERIC FRAME CONTRACT
# ============================================================


class NumericFrameProvider(Protocol):
    """
    Any object exposing numeric matrix.
    """

    def to_numpy(
        self,
    ) -> np.ndarray:
        ...


# ============================================================
# SHARED HELPER TYPES
# ============================================================

SeriesLike = (
    pd.Series
    | np.ndarray
    | Sequence[float]
)

FrameLike = (
    pd.DataFrame
    | np.ndarray
)

MappingLike = (
    Dict[str, Any]
    | Mapping[str, Any]
)

# ============================================================
# END PART 1
# ============================================================



# ============================================================
# PART 2 — ADAPTER RESULT OBJECTS
# ============================================================

# ============================================================
# VALIDATION RESULT
# ============================================================


@dataclass(slots=True)
class ValidationResult:
    """
    Validation outcome produced
    before adaptation.
    """

    is_valid: bool

    warnings: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# SCHEMA INSPECTION RESULT
# ============================================================


@dataclass(slots=True)
class SchemaInspectionResult:
    """
    Structure inspection of incoming data.
    """

    column_count: int = 0

    row_count: int = 0

    columns: list[str] = field(
        default_factory=list
    )

    dtypes: dict[str, str] = field(
        default_factory=dict
    )

    missing_values: dict[str, int] = field(
        default_factory=dict
    )


# ============================================================
# DATA QUALITY RESULT
# ============================================================


@dataclass(slots=True)
class DataQualityResult:
    """
    Data quality diagnostics.
    """

    completeness_score: float = 1.0

    consistency_score: float = 1.0

    duplicate_rows: int = 0

    missing_cells: int = 0

    outlier_count: int = 0

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# ADAPTATION STATISTICS
# ============================================================


@dataclass(slots=True)
class AdaptationStatistics:
    """
    Statistics generated during adaptation.
    """

    input_rows: int = 0

    output_rows: int = 0

    input_columns: int = 0

    output_columns: int = 0

    dropped_rows: int = 0

    dropped_columns: int = 0

    created_columns: list[str] = field(
        default_factory=list
    )

    renamed_columns: dict[str, str] = field(
        default_factory=dict
    )


# ============================================================
# ADAPTED DATA OBJECT
# ============================================================


@dataclass(slots=True)
class AdaptedDataset:
    """
    Canonical adapted dataset.
    """

    data: pd.DataFrame

    source_name: str

    source_type: AdapterSourceType

    adapted_at: datetime = field(
        default_factory=datetime.utcnow
    )

    statistics: AdaptationStatistics = field(
        default_factory=AdaptationStatistics
    )

    quality: DataQualityResult = field(
        default_factory=DataQualityResult
    )


# ============================================================
# MARKET DATA ADAPTER RESULT
# ============================================================


@dataclass(slots=True)
class MarketDataAdapterResult:
    """
    Market-data adaptation output.
    """

    prices: pd.DataFrame

    returns: pd.DataFrame | None = None

    benchmark_prices: pd.DataFrame | None = None

    benchmark_returns: pd.DataFrame | None = None

    volumes: pd.DataFrame | None = None

    market_caps: pd.DataFrame | None = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# FORECAST ADAPTER RESULT
# ============================================================


@dataclass(slots=True)
class ForecastAdapterResult:
    """
    Forecast adaptation output.
    """

    alpha_scores: pd.DataFrame | None = None

    expected_returns: pd.DataFrame | None = None

    forecast_confidence: pd.DataFrame | None = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# FACTOR ADAPTER RESULT
# ============================================================


@dataclass(slots=True)
class FactorAdapterResult:
    """
    Factor-model adaptation output.
    """

    factor_exposures: pd.DataFrame | None = None

    factor_returns: pd.DataFrame | None = None

    factor_covariance: pd.DataFrame | None = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# PORTFOLIO ADAPTER RESULT
# ============================================================


@dataclass(slots=True)
class PortfolioAdapterResult:
    """
    Existing portfolio adaptation output.
    """

    current_weights: pd.Series | None = None

    current_holdings: pd.DataFrame | None = None

    current_positions: pd.DataFrame | None = None

    cash_weight: float = 0.0

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# LIQUIDITY ADAPTER RESULT
# ============================================================


@dataclass(slots=True)
class LiquidityAdapterResult:
    """
    Liquidity adaptation output.
    """

    average_daily_volume: pd.Series | None = None

    bid_ask_spread: pd.Series | None = None

    participation_limit: float | None = None

    liquidity_profile: pd.DataFrame | None = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# CONSTRAINT ADAPTER RESULT
# ============================================================


@dataclass(slots=True)
class ConstraintAdapterResult:
    """
    Constraint adaptation output.
    """

    sector_map: pd.Series | None = None

    industry_map: pd.Series | None = None

    country_map: pd.Series | None = None

    custom_constraints: dict[str, Any] = field(
        default_factory=dict
    )

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# MASTER ADAPTER OUTPUT
# ============================================================


@dataclass(slots=True)
class InstitutionalAdapterResult:
    """
    Unified adapter result used by
    all downstream pipeline stages.
    """

    status: AdapterStatus

    validation: ValidationResult

    dataset: AdaptedDataset | None = None

    payload: Any = None

    metadata: AdapterMetadata | None = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=datetime.utcnow
    )

    error_message: str | None = None


# ============================================================
# END PART 2
# ============================================================


# ============================================================
# PART 3 — MARKET DATA ADAPTER
# ============================================================

class MarketDataAdapter:
    """
    Converts raw market data into
    institutional MarketDataInput objects.

    Supported Inputs
    ----------------
    1. Wide price dataframe
       Index = Date
       Columns = Tickers

    2. Long dataframe
       Date | Ticker | Close

    3. Existing returns dataframe

    Produces
    --------
    MarketDataInput
    """

    REQUIRED_LONG_COLUMNS = {
        "Date",
        "Ticker",
        "Close",
    }

    # --------------------------------------------------------

    @staticmethod
    def _validate_prices(
        prices: pd.DataFrame,
    ) -> None:

        if prices is None:

            raise ValueError(
                "prices cannot be None."
            )

        if prices.empty:

            raise ValueError(
                "prices dataframe is empty."
            )

    # --------------------------------------------------------

    @staticmethod
    def _ensure_datetime_index(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        if not isinstance(
            df.index,
            pd.DatetimeIndex,
        ):

            try:

                df.index = pd.to_datetime(
                    df.index
                )

            except Exception as exc:

                raise ValueError(
                    "Unable to convert index "
                    "to DatetimeIndex."
                ) from exc

        df = df.sort_index()

        return df

    # --------------------------------------------------------

    @staticmethod
    def _compute_returns(
        prices: pd.DataFrame,
    ) -> pd.DataFrame:

        returns = (
            prices
            .pct_change()
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        return returns

    # --------------------------------------------------------

    @classmethod
    def from_wide_prices(
        cls,
        prices: pd.DataFrame,
        *,
        benchmark_prices:
        pd.DataFrame | None = None,
        volumes:
        pd.DataFrame | None = None,
        market_caps:
        pd.DataFrame | None = None,
    ) -> MarketDataInput:

        cls._validate_prices(
            prices
        )

        prices = (
            cls
            ._ensure_datetime_index(
                prices
            )
        )

        returns = (
            cls
            ._compute_returns(
                prices
            )
        )

        benchmark_returns = None

        if benchmark_prices is not None:

            benchmark_prices = (
                cls
                ._ensure_datetime_index(
                    benchmark_prices
                )
            )

            benchmark_returns = (
                cls
                ._compute_returns(
                    benchmark_prices
                )
            )

        return MarketDataInput(

            prices=prices,

            returns=returns,

            benchmark_prices=
            benchmark_prices,

            benchmark_returns=
            benchmark_returns,

            volumes=volumes,

            market_caps=
            market_caps,
        )

    # --------------------------------------------------------

    @classmethod
    def from_long_prices(
        cls,
        raw_data: pd.DataFrame,
        *,
        price_column: str = "Close",
        ticker_column: str = "Ticker",
        date_column: str = "Date",
    ) -> MarketDataInput:

        if raw_data is None:

            raise ValueError(
                "raw_data cannot be None."
            )

        if raw_data.empty:

            raise ValueError(
                "raw_data is empty."
            )

        required = {
            date_column,
            ticker_column,
            price_column,
        }

        missing = (
            required
            - set(raw_data.columns)
        )

        if missing:

            raise ValueError(
                f"Missing columns: {missing}"
            )

        df = raw_data.copy()

        df[date_column] = (
            pd.to_datetime(
                df[date_column]
            )
        )

        prices = (
            df
            .pivot(
                index=date_column,
                columns=ticker_column,
                values=price_column,
            )
            .sort_index()
        )

        return (
            cls
            .from_wide_prices(
                prices
            )
        )

    # --------------------------------------------------------

    @classmethod
    def from_prices_and_returns(
        cls,
        *,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> MarketDataInput:

        cls._validate_prices(
            prices
        )

        prices = (
            cls
            ._ensure_datetime_index(
                prices
            )
        )

        returns = (
            cls
            ._ensure_datetime_index(
                returns
            )
        )

        return MarketDataInput(

            prices=prices,

            returns=returns,
        )

    # --------------------------------------------------------

    @classmethod
    def adapt(
        cls,
        data: pd.DataFrame,
    ) -> MarketDataInput:
        """
        Auto-detect format.

        Wide:
            Date index
            Columns=tickers

        Long:
            Date/Ticker/Close columns
        """

        if {
            "Date",
            "Ticker",
            "Close",
        }.issubset(
            set(data.columns)
        ):

            return (
                cls
                .from_long_prices(
                    data
                )
            )

        return (
            cls
            .from_wide_prices(
                data
            )
        )
    
# ============================================================
# PART 4 — FORECAST ADAPTER
# ============================================================

class ForecastAdapter:
    """
    Converts Alpha Engine outputs into
    institutional forecast objects.

    Supported inputs:

    1. Alpha scores
    2. Expected returns
    3. Forecast confidence
    4. Composite scores

    Output:

    ForecastInput
    """

    # --------------------------------------------------------

    @staticmethod
    def from_alpha_scores(
        alpha_scores: pd.DataFrame,
    ) -> ForecastInput:
        """
        Alpha score dataframe
        -> ForecastInput
        """

        if alpha_scores is None:
            raise ValueError(
                "alpha_scores cannot be None."
            )

        if alpha_scores.empty:
            raise ValueError(
                "alpha_scores are empty."
            )

        return ForecastInput(
            alpha_scores=alpha_scores.copy()
        )

    # --------------------------------------------------------

    @staticmethod
    def from_expected_returns(
        expected_returns: pd.DataFrame,
    ) -> ForecastInput:
        """
        Expected return forecasts
        -> ForecastInput
        """

        if expected_returns is None:
            raise ValueError(
                "expected_returns cannot be None."
            )

        if expected_returns.empty:
            raise ValueError(
                "expected_returns are empty."
            )

        return ForecastInput(
            expected_returns=
            expected_returns.copy()
        )

    # --------------------------------------------------------

    @staticmethod
    def from_alpha_and_confidence(
        *,
        alpha_scores: pd.DataFrame,
        confidence: pd.DataFrame,
    ) -> ForecastInput:
        """
        Alpha + confidence.
        """

        if alpha_scores is None:
            raise ValueError(
                "alpha_scores cannot be None."
            )

        if confidence is None:
            raise ValueError(
                "confidence cannot be None."
            )

        return ForecastInput(
            alpha_scores=
            alpha_scores.copy(),

            forecast_confidence=
            confidence.copy(),
        )

    # --------------------------------------------------------

    @staticmethod
    def from_composite_score_engine(
        score_table: pd.DataFrame,
    ) -> ForecastInput:
        """
        Output of feature_category_budget.py

        Expected columns:

        Asset
        CompositeScore
        """

        if score_table is None:
            raise ValueError(
                "score_table cannot be None."
            )

        if score_table.empty:
            raise ValueError(
                "score_table is empty."
            )

        required = {
            "Asset",
            "CompositeScore",
        }

        missing = (
            required
            - set(score_table.columns)
        )

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        alpha_scores = (
            score_table[
                ["Asset", "CompositeScore"]
            ]
            .rename(
                columns={
                    "CompositeScore":
                    "AlphaScore"
                }
            )
        )

        return ForecastInput(
            alpha_scores=
            alpha_scores
        )

    # --------------------------------------------------------

    @staticmethod
    def build(
        *,
        alpha_scores:
        pd.DataFrame | None = None,

        expected_returns:
        pd.DataFrame | None = None,

        forecast_confidence:
        pd.DataFrame | None = None,
    ) -> ForecastInput:
        """
        Generic constructor.
        """

        return ForecastInput(
            alpha_scores=
            alpha_scores,

            expected_returns=
            expected_returns,

            forecast_confidence=
            forecast_confidence,
        )
    


# ============================================================
# PART 5 — FACTOR ADAPTER
# ============================================================


class FactorAdapter:
    """
    Converts factor model outputs into
    standardized FactorInput objects.

    Supports:

    - Factor exposures
    - Factor returns
    - Factor covariance
    - Full factor models
    """

    # --------------------------------------------------------
    # EXPOSURES ONLY
    # --------------------------------------------------------

    @staticmethod
    def from_factor_exposures(
        factor_exposures: pd.DataFrame,
    ) -> FactorInput:

        if factor_exposures is None:
            raise ValueError(
                "factor_exposures cannot be None."
            )

        if factor_exposures.empty:
            raise ValueError(
                "factor_exposures are empty."
            )

        return FactorInput(
            factor_exposures=
            factor_exposures.copy()
        )

    # --------------------------------------------------------
    # RETURNS ONLY
    # --------------------------------------------------------

    @staticmethod
    def from_factor_returns(
        factor_returns: pd.DataFrame,
    ) -> FactorInput:

        if factor_returns is None:
            raise ValueError(
                "factor_returns cannot be None."
            )

        if factor_returns.empty:
            raise ValueError(
                "factor_returns are empty."
            )

        return FactorInput(
            factor_returns=
            factor_returns.copy()
        )

    # --------------------------------------------------------
    # COVARIANCE ONLY
    # --------------------------------------------------------

    @staticmethod
    def from_factor_covariance(
        factor_covariance: pd.DataFrame,
    ) -> FactorInput:

        if factor_covariance is None:
            raise ValueError(
                "factor_covariance cannot be None."
            )

        if factor_covariance.empty:
            raise ValueError(
                "factor_covariance is empty."
            )

        return FactorInput(
            factor_covariance=
            factor_covariance.copy()
        )

    # --------------------------------------------------------
    # EXPOSURES + RETURNS
    # --------------------------------------------------------

    @staticmethod
    def from_exposures_and_returns(
        *,
        factor_exposures: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> FactorInput:

        if factor_exposures is None:
            raise ValueError(
                "factor_exposures cannot be None."
            )

        if factor_returns is None:
            raise ValueError(
                "factor_returns cannot be None."
            )

        return FactorInput(
            factor_exposures=
            factor_exposures.copy(),

            factor_returns=
            factor_returns.copy(),
        )

    # --------------------------------------------------------
    # FULL FACTOR MODEL
    # --------------------------------------------------------

    @staticmethod
    def from_full_factor_model(
        *,
        factor_exposures: pd.DataFrame,
        factor_returns: pd.DataFrame,
        factor_covariance: pd.DataFrame,
    ) -> FactorInput:

        if factor_exposures is None:
            raise ValueError(
                "factor_exposures cannot be None."
            )

        if factor_returns is None:
            raise ValueError(
                "factor_returns cannot be None."
            )

        if factor_covariance is None:
            raise ValueError(
                "factor_covariance cannot be None."
            )

        return FactorInput(

            factor_exposures=
            factor_exposures.copy(),

            factor_returns=
            factor_returns.copy(),

            factor_covariance=
            factor_covariance.copy(),
        )

    # --------------------------------------------------------
    # PCA FACTOR MODEL
    # --------------------------------------------------------

    @staticmethod
    def from_pca_model(
        *,
        exposures: pd.DataFrame,
        factor_covariance: pd.DataFrame,
    ) -> FactorInput:

        return FactorInput(

            factor_exposures=
            exposures.copy(),

            factor_covariance=
            factor_covariance.copy(),
        )

    # --------------------------------------------------------
    # RISK MODEL OUTPUT
    # --------------------------------------------------------

    @staticmethod
    def from_risk_model(
        risk_model_result: Any,
    ) -> FactorInput:
        """
        Generic adapter for
        risk_model.py outputs.
        """

        exposures = getattr(
            risk_model_result,
            "factor_exposures",
            None,
        )

        returns = getattr(
            risk_model_result,
            "factor_returns",
            None,
        )

        covariance = getattr(
            risk_model_result,
            "factor_covariance",
            None,
        )

        return FactorInput(

            factor_exposures=
            exposures.copy()
            if isinstance(
                exposures,
                pd.DataFrame,
            )
            else None,

            factor_returns=
            returns.copy()
            if isinstance(
                returns,
                pd.DataFrame,
            )
            else None,

            factor_covariance=
            covariance.copy()
            if isinstance(
                covariance,
                pd.DataFrame,
            )
            else None,
        )

    # --------------------------------------------------------
    # GENERIC BUILDER
    # --------------------------------------------------------

    @staticmethod
    def build(
        *,
        factor_exposures:
        pd.DataFrame | None = None,

        factor_returns:
        pd.DataFrame | None = None,

        factor_covariance:
        pd.DataFrame | None = None,
    ) -> FactorInput:

        return FactorInput(

            factor_exposures=
            factor_exposures,

            factor_returns=
            factor_returns,

            factor_covariance=
            factor_covariance,
        )
    
# ============================================================
# PART 6 — PORTFOLIO STATE ADAPTER
# ============================================================


class PortfolioStateAdapter:
    """
    Converts portfolio state data into
    standardized PortfolioInput objects.

    Supported sources:

    - Holdings
    - Positions
    - Portfolio weights
    - Broker snapshots
    - Previous rebalance outputs
    - Portfolio builder outputs
    """

    # --------------------------------------------------------
    # HOLDINGS ONLY
    # --------------------------------------------------------

    @staticmethod
    def from_holdings(
        holdings: pd.DataFrame,
        *,
        cash_weight: float = 0.0,
    ) -> PortfolioInput:

        if holdings is None:
            raise ValueError(
                "holdings cannot be None."
            )

        if holdings.empty:
            raise ValueError(
                "holdings are empty."
            )

        return PortfolioInput(

            current_holdings=
            holdings.copy(),

            cash_weight=
            float(cash_weight),
        )

    # --------------------------------------------------------
    # POSITIONS ONLY
    # --------------------------------------------------------

    @staticmethod
    def from_positions(
        positions: pd.DataFrame,
        *,
        cash_weight: float = 0.0,
    ) -> PortfolioInput:

        if positions is None:
            raise ValueError(
                "positions cannot be None."
            )

        if positions.empty:
            raise ValueError(
                "positions are empty."
            )

        return PortfolioInput(

            current_positions=
            positions.copy(),

            cash_weight=
            float(cash_weight),
        )

    # --------------------------------------------------------
    # WEIGHTS ONLY
    # --------------------------------------------------------

    @staticmethod
    def from_weights(
        weights: pd.Series,
        *,
        cash_weight: float = 0.0,
    ) -> PortfolioInput:

        if weights is None:
            raise ValueError(
                "weights cannot be None."
            )

        if len(weights) == 0:
            raise ValueError(
                "weights are empty."
            )

        return PortfolioInput(

            current_weights=
            weights.copy(),

            cash_weight=
            float(cash_weight),
        )

    # --------------------------------------------------------
    # HOLDINGS + WEIGHTS
    # --------------------------------------------------------

    @staticmethod
    def from_holdings_and_weights(
        *,
        holdings: pd.DataFrame,
        weights: pd.Series,
        cash_weight: float = 0.0,
    ) -> PortfolioInput:

        return PortfolioInput(

            current_holdings=
            holdings.copy(),

            current_weights=
            weights.copy(),

            cash_weight=
            float(cash_weight),
        )

    # --------------------------------------------------------
    # FULL PORTFOLIO STATE
    # --------------------------------------------------------

    @staticmethod
    def from_full_state(
        *,
        holdings: pd.DataFrame | None = None,
        positions: pd.DataFrame | None = None,
        weights: pd.Series | None = None,
        cash_weight: float = 0.0,
    ) -> PortfolioInput:

        return PortfolioInput(

            current_holdings=
            holdings.copy()
            if holdings is not None
            else None,

            current_positions=
            positions.copy()
            if positions is not None
            else None,

            current_weights=
            weights.copy()
            if weights is not None
            else None,

            cash_weight=
            float(cash_weight),
        )

    # --------------------------------------------------------
    # BROKER SNAPSHOT
    # --------------------------------------------------------

    @staticmethod
    def from_broker_snapshot(
        snapshot: pd.DataFrame,
        *,
        ticker_col: str = "Ticker",
        quantity_col: str = "Quantity",
        market_value_col: str = "Market_Value",
        cash_weight: float = 0.0,
    ) -> PortfolioInput:
        """
        Converts broker export
        into portfolio holdings.
        """

        if snapshot is None:
            raise ValueError(
                "snapshot cannot be None."
            )

        holdings = snapshot.copy()

        required = [
            ticker_col,
            quantity_col,
            market_value_col,
        ]

        missing = [
            c
            for c in required
            if c not in holdings.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        return PortfolioInput(

            current_holdings=
            holdings,

            cash_weight=
            float(cash_weight),
        )

    # --------------------------------------------------------
    # PREVIOUS REBALANCE OUTPUT
    # --------------------------------------------------------

    @staticmethod
    def from_rebalance_output(
        rebalance_result: Any,
    ) -> PortfolioInput:
        """
        Converts rebalance output
        into next-period portfolio state.
        """

        holdings = getattr(
            rebalance_result,
            "holdings",
            None,
        )

        weights = getattr(
            rebalance_result,
            "weights",
            None,
        )

        positions = getattr(
            rebalance_result,
            "positions",
            None,
        )

        cash_weight = getattr(
            rebalance_result,
            "cash_weight",
            0.0,
        )

        return PortfolioInput(

            current_holdings=
            holdings.copy()
            if isinstance(
                holdings,
                pd.DataFrame,
            )
            else None,

            current_positions=
            positions.copy()
            if isinstance(
                positions,
                pd.DataFrame,
            )
            else None,

            current_weights=
            weights.copy()
            if isinstance(
                weights,
                pd.Series,
            )
            else None,

            cash_weight=
            float(cash_weight),
        )

    # --------------------------------------------------------
    # PORTFOLIO BUILDER OUTPUT
    # --------------------------------------------------------

    @staticmethod
    def from_portfolio_builder(
        portfolio: Any,
    ) -> PortfolioInput:
        """
        Converts portfolio_builder.py
        output into PortfolioInput.
        """

        holdings = getattr(
            portfolio,
            "holdings",
            None,
        )

        weights = getattr(
            portfolio,
            "weights",
            None,
        )

        positions = getattr(
            portfolio,
            "positions",
            None,
        )

        cash_weight = getattr(
            portfolio,
            "cash_weight",
            0.0,
        )

        return PortfolioInput(

            current_holdings=
            holdings.copy()
            if isinstance(
                holdings,
                pd.DataFrame,
            )
            else None,

            current_positions=
            positions.copy()
            if isinstance(
                positions,
                pd.DataFrame,
            )
            else None,

            current_weights=
            weights.copy()
            if isinstance(
                weights,
                pd.Series,
            )
            else None,

            cash_weight=
            float(cash_weight),
        )

    # --------------------------------------------------------
    # GENERIC BUILDER
    # --------------------------------------------------------

    @staticmethod
    def build(
        *,
        holdings: pd.DataFrame | None = None,
        positions: pd.DataFrame | None = None,
        weights: pd.Series | None = None,
        cash_weight: float = 0.0,
    ) -> PortfolioInput:

        return PortfolioInput(

            current_holdings=
            holdings,

            current_positions=
            positions,

            current_weights=
            weights,

            cash_weight=
            float(cash_weight),
        )
    
# ============================================================
# PART 7 — LIQUIDITY ADAPTER
# ============================================================


class LiquidityAdapter:
    """
    Converts liquidity-related data
    into standardized LiquidityInput objects.

    Supported inputs:

    - ADV
    - Dollar ADV
    - Bid/Ask Spread
    - Participation Limits
    - Liquidity Profiles
    """

    # --------------------------------------------------------
    # ADV ONLY
    # --------------------------------------------------------

    @staticmethod
    def from_adv(
        average_daily_volume: pd.Series,
    ) -> LiquidityInput:

        if average_daily_volume is None:

            raise ValueError(
                "average_daily_volume cannot be None."
            )

        if len(average_daily_volume) == 0:

            raise ValueError(
                "average_daily_volume is empty."
            )

        return LiquidityInput(

            average_daily_volume=
            average_daily_volume.copy()
        )

    # --------------------------------------------------------
    # BID / ASK SPREAD ONLY
    # --------------------------------------------------------

    @staticmethod
    def from_bid_ask_spread(
        bid_ask_spread: pd.Series,
    ) -> LiquidityInput:

        if bid_ask_spread is None:

            raise ValueError(
                "bid_ask_spread cannot be None."
            )

        if len(bid_ask_spread) == 0:

            raise ValueError(
                "bid_ask_spread is empty."
            )

        return LiquidityInput(

            bid_ask_spread=
            bid_ask_spread.copy()
        )

    # --------------------------------------------------------
    # PARTICIPATION LIMIT ONLY
    # --------------------------------------------------------

    @staticmethod
    def from_participation_limit(
        participation_limit: float,
    ) -> LiquidityInput:

        if participation_limit <= 0:

            raise ValueError(
                "participation_limit must be positive."
            )

        return LiquidityInput(

            participation_limit=
            float(participation_limit)
        )

    # --------------------------------------------------------
    # LIQUIDITY PROFILE ONLY
    # --------------------------------------------------------

    @staticmethod
    def from_liquidity_profile(
        liquidity_profile: pd.DataFrame,
    ) -> LiquidityInput:

        if liquidity_profile is None:

            raise ValueError(
                "liquidity_profile cannot be None."
            )

        if liquidity_profile.empty:

            raise ValueError(
                "liquidity_profile is empty."
            )

        return LiquidityInput(

            liquidity_profile=
            liquidity_profile.copy()
        )

    # --------------------------------------------------------
    # ADV + SPREAD
    # --------------------------------------------------------

    @staticmethod
    def from_adv_and_spread(
        *,
        average_daily_volume: pd.Series,
        bid_ask_spread: pd.Series,
    ) -> LiquidityInput:

        return LiquidityInput(

            average_daily_volume=
            average_daily_volume.copy(),

            bid_ask_spread=
            bid_ask_spread.copy()
        )

    # --------------------------------------------------------
    # ADV + SPREAD + PARTICIPATION
    # --------------------------------------------------------

    @staticmethod
    def from_execution_inputs(
        *,
        average_daily_volume: pd.Series,
        bid_ask_spread: pd.Series,
        participation_limit: float,
    ) -> LiquidityInput:

        return LiquidityInput(

            average_daily_volume=
            average_daily_volume.copy(),

            bid_ask_spread=
            bid_ask_spread.copy(),

            participation_limit=
            float(participation_limit)
        )

    # --------------------------------------------------------
    # DOLLAR ADV
    # --------------------------------------------------------

    @staticmethod
    def from_dollar_adv(
        *,
        volume: pd.Series,
        price: pd.Series,
    ) -> LiquidityInput:
        """
        Convert volume + price
        into Dollar ADV.
        """

        dollar_adv = volume * price

        return LiquidityInput(

            average_daily_volume=
            dollar_adv.copy()
        )

    # --------------------------------------------------------
    # BROKER EXECUTION FILE
    # --------------------------------------------------------

    @staticmethod
    def from_broker_execution_snapshot(
        snapshot: pd.DataFrame,
        *,
        adv_column: str = "ADV",
        spread_column: str = "Spread",
    ) -> LiquidityInput:

        if snapshot is None:

            raise ValueError(
                "snapshot cannot be None."
            )

        missing = []

        if adv_column not in snapshot.columns:

            missing.append(
                adv_column
            )

        if spread_column not in snapshot.columns:

            missing.append(
                spread_column
            )

        if missing:

            raise ValueError(
                f"Missing columns: {missing}"
            )

        adv = snapshot[
            adv_column
        ]

        spread = snapshot[
            spread_column
        ]

        if "Ticker" in snapshot.columns:

            adv.index = snapshot[
                "Ticker"
            ]

            spread.index = snapshot[
                "Ticker"
            ]

        return LiquidityInput(

            average_daily_volume=
            adv.copy(),

            bid_ask_spread=
            spread.copy()
        )

    # --------------------------------------------------------
    # EXECUTION MODEL OUTPUT
    # --------------------------------------------------------

    @staticmethod
    def from_execution_model(
        execution_result: Any,
    ) -> LiquidityInput:
        """
        Convert execution.py output.
        """

        adv = getattr(
            execution_result,
            "average_daily_volume",
            None,
        )

        spread = getattr(
            execution_result,
            "bid_ask_spread",
            None,
        )

        participation = getattr(
            execution_result,
            "participation_limit",
            None,
        )

        profile = getattr(
            execution_result,
            "liquidity_profile",
            None,
        )

        return LiquidityInput(

            average_daily_volume=
            adv.copy()
            if isinstance(
                adv,
                pd.Series,
            )
            else adv,

            bid_ask_spread=
            spread.copy()
            if isinstance(
                spread,
                pd.Series,
            )
            else spread,

            participation_limit=
            float(participation)
            if participation is not None
            else None,

            liquidity_profile=
            profile.copy()
            if isinstance(
                profile,
                pd.DataFrame,
            )
            else profile,
        )

    # --------------------------------------------------------
    # GENERIC BUILDER
    # --------------------------------------------------------

    @staticmethod
    def build(
        *,
        average_daily_volume:
        pd.Series | None = None,

        bid_ask_spread:
        pd.Series | None = None,

        participation_limit:
        float | None = None,

        liquidity_profile:
        pd.DataFrame | None = None,
    ) -> LiquidityInput:

        return LiquidityInput(

            average_daily_volume=
            average_daily_volume,

            bid_ask_spread=
            bid_ask_spread,

            participation_limit=
            participation_limit,

            liquidity_profile=
            liquidity_profile,
        )
    

# ============================================================
# PART 8 — CONSTRAINT ADAPTER
# ============================================================


class ConstraintAdapter:
    """
    Converts constraint-related inputs
    into standardized ConstraintInput objects.

    Supported:

    - sector map
    - industry map
    - country map
    - custom constraints

    Output:

    ConstraintInput
    """

    # --------------------------------------------------------
    # SECTOR MAP
    # --------------------------------------------------------

    @staticmethod
    def from_sector_map(
        sector_map: pd.Series,
    ) -> ConstraintInput:

        if sector_map is None:

            raise ValueError(
                "sector_map cannot be None."
            )

        if len(sector_map) == 0:

            raise ValueError(
                "sector_map is empty."
            )

        return ConstraintInput(

            sector_map=
            sector_map.copy()
        )

    # --------------------------------------------------------
    # INDUSTRY MAP
    # --------------------------------------------------------

    @staticmethod
    def from_industry_map(
        industry_map: pd.Series,
    ) -> ConstraintInput:

        if industry_map is None:

            raise ValueError(
                "industry_map cannot be None."
            )

        if len(industry_map) == 0:

            raise ValueError(
                "industry_map is empty."
            )

        return ConstraintInput(

            industry_map=
            industry_map.copy()
        )

    # --------------------------------------------------------
    # COUNTRY MAP
    # --------------------------------------------------------

    @staticmethod
    def from_country_map(
        country_map: pd.Series,
    ) -> ConstraintInput:

        if country_map is None:

            raise ValueError(
                "country_map cannot be None."
            )

        if len(country_map) == 0:

            raise ValueError(
                "country_map is empty."
            )

        return ConstraintInput(

            country_map=
            country_map.copy()
        )

    # --------------------------------------------------------
    # CUSTOM CONSTRAINTS
    # --------------------------------------------------------

    @staticmethod
    def from_custom_constraints(
        custom_constraints:
        dict[str, Any],
    ) -> ConstraintInput:

        if custom_constraints is None:

            custom_constraints = {}

        return ConstraintInput(

            custom_constraints=
            dict(custom_constraints)
        )

    # --------------------------------------------------------
    # SECTOR + INDUSTRY
    # --------------------------------------------------------

    @staticmethod
    def from_sector_industry(
        *,
        sector_map:
        pd.Series,

        industry_map:
        pd.Series,
    ) -> ConstraintInput:

        return ConstraintInput(

            sector_map=
            sector_map.copy(),

            industry_map=
            industry_map.copy(),
        )

    # --------------------------------------------------------
    # SECTOR + INDUSTRY + COUNTRY
    # --------------------------------------------------------

    @staticmethod
    def from_classification_maps(
        *,
        sector_map:
        pd.Series | None = None,

        industry_map:
        pd.Series | None = None,

        country_map:
        pd.Series | None = None,
    ) -> ConstraintInput:

        return ConstraintInput(

            sector_map=
            sector_map.copy()
            if sector_map is not None
            else None,

            industry_map=
            industry_map.copy()
            if industry_map is not None
            else None,

            country_map=
            country_map.copy()
            if country_map is not None
            else None,
        )

    # --------------------------------------------------------
    # FULL CONSTRAINT UNIVERSE
    # --------------------------------------------------------

    @staticmethod
    def from_full_constraint_set(
        *,
        sector_map:
        pd.Series | None = None,

        industry_map:
        pd.Series | None = None,

        country_map:
        pd.Series | None = None,

        custom_constraints:
        dict[str, Any] | None = None,
    ) -> ConstraintInput:

        return ConstraintInput(

            sector_map=
            sector_map.copy()
            if sector_map is not None
            else None,

            industry_map=
            industry_map.copy()
            if industry_map is not None
            else None,

            country_map=
            country_map.copy()
            if country_map is not None
            else None,

            custom_constraints=
            dict(custom_constraints)
            if custom_constraints is not None
            else {},
        )

    # --------------------------------------------------------
    # FROM SECURITY MASTER
    # --------------------------------------------------------

    @staticmethod
    def from_security_master(
        security_master:
        pd.DataFrame,
        *,
        ticker_column:
        str = "Ticker",

        sector_column:
        str = "Sector",

        industry_column:
        str = "Industry",

        country_column:
        str = "Country",
    ) -> ConstraintInput:

        if security_master is None:

            raise ValueError(
                "security_master cannot be None."
            )

        if security_master.empty:

            raise ValueError(
                "security_master is empty."
            )

        required = [
            ticker_column,
        ]

        missing = [
            c
            for c in required
            if c not in security_master.columns
        ]

        if missing:

            raise ValueError(
                f"Missing columns: {missing}"
            )

        df = security_master.copy()

        ticker_index = df[
            ticker_column
        ]

        sector_map = None
        industry_map = None
        country_map = None

        if sector_column in df.columns:

            sector_map = pd.Series(
                df[sector_column].values,
                index=ticker_index,
                name="Sector",
            )

        if industry_column in df.columns:

            industry_map = pd.Series(
                df[industry_column].values,
                index=ticker_index,
                name="Industry",
            )

        if country_column in df.columns:

            country_map = pd.Series(
                df[country_column].values,
                index=ticker_index,
                name="Country",
            )

        return ConstraintInput(

            sector_map=
            sector_map,

            industry_map=
            industry_map,

            country_map=
            country_map,
        )

    # --------------------------------------------------------
    # FROM PORTFOLIO BUILDER OUTPUT
    # --------------------------------------------------------

    @staticmethod
    def from_portfolio_builder(
        builder_result: Any,
    ) -> ConstraintInput:

        sector_map = getattr(
            builder_result,
            "sector_map",
            None,
        )

        industry_map = getattr(
            builder_result,
            "industry_map",
            None,
        )

        country_map = getattr(
            builder_result,
            "country_map",
            None,
        )

        custom_constraints = getattr(
            builder_result,
            "custom_constraints",
            {},
        )

        return ConstraintInput(

            sector_map=
            sector_map.copy()
            if isinstance(
                sector_map,
                pd.Series,
            )
            else sector_map,

            industry_map=
            industry_map.copy()
            if isinstance(
                industry_map,
                pd.Series,
            )
            else industry_map,

            country_map=
            country_map.copy()
            if isinstance(
                country_map,
                pd.Series,
            )
            else country_map,

            custom_constraints=
            dict(custom_constraints),
        )

    # --------------------------------------------------------
    # GENERIC BUILDER
    # --------------------------------------------------------

    @staticmethod
    def build(
        *,
        sector_map:
        pd.Series | None = None,

        industry_map:
        pd.Series | None = None,

        country_map:
        pd.Series | None = None,

        custom_constraints:
        dict[str, Any] | None = None,
    ) -> ConstraintInput:

        return ConstraintInput(

            sector_map=
            sector_map,

            industry_map=
            industry_map,

            country_map=
            country_map,

            custom_constraints=
            custom_constraints
            if custom_constraints is not None
            else {},
        )
    

# ============================================================
# PART 9 — ALPHA ADAPTER
#
# Bridges Alpha Research Layer
# →
# ForecastInput
#
# Supports:
#   - feature_category_budget.py
#   - alpha scores
#   - expected returns
#   - confidence scores
#   - composite alpha outputs
#
# Output:
#   ForecastInput
# ============================================================


class AlphaAdapter:
    """
    Institutional Alpha Adapter.

    Converts outputs from the alpha
    research stack into ForecastInput.

    Compatible with:

    - ic_engine.py
    - ic_stability.py
    - feature_clustering.py
    - feature_decay.py
    - feature_category_budget.py
    - composite alpha engine
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_dataframe(
        df: pd.DataFrame | None,
        name: str,
    ) -> None:

        if df is None:

            raise ValueError(
                f"{name} cannot be None."
            )

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

    # ========================================================
    # ALPHA SCORES ONLY
    # ========================================================

    @staticmethod
    def from_alpha_scores(
        alpha_scores:
        pd.DataFrame,
    ) -> ForecastInput:

        AlphaAdapter._validate_dataframe(
            alpha_scores,
            "alpha_scores",
        )

        return ForecastInput(

            alpha_scores=
            alpha_scores.copy()
        )

    # ========================================================
    # EXPECTED RETURNS ONLY
    # ========================================================

    @staticmethod
    def from_expected_returns(
        expected_returns:
        pd.DataFrame,
    ) -> ForecastInput:

        AlphaAdapter._validate_dataframe(
            expected_returns,
            "expected_returns",
        )

        return ForecastInput(

            expected_returns=
            expected_returns.copy()
        )

    # ========================================================
    # CONFIDENCE ONLY
    # ========================================================

    @staticmethod
    def from_confidence(
        confidence:
        pd.DataFrame,
    ) -> ForecastInput:

        AlphaAdapter._validate_dataframe(
            confidence,
            "forecast_confidence",
        )

        return ForecastInput(

            forecast_confidence=
            confidence.copy()
        )

    # ========================================================
    # ALPHA + EXPECTED RETURNS
    # ========================================================

    @staticmethod
    def from_alpha_and_returns(
        *,
        alpha_scores:
        pd.DataFrame,

        expected_returns:
        pd.DataFrame,
    ) -> ForecastInput:

        AlphaAdapter._validate_dataframe(
            alpha_scores,
            "alpha_scores",
        )

        AlphaAdapter._validate_dataframe(
            expected_returns,
            "expected_returns",
        )

        return ForecastInput(

            alpha_scores=
            alpha_scores.copy(),

            expected_returns=
            expected_returns.copy(),
        )

    # ========================================================
    # FULL FORECAST OBJECT
    # ========================================================

    @staticmethod
    def build(
        *,
        alpha_scores:
        pd.DataFrame | None = None,

        expected_returns:
        pd.DataFrame | None = None,

        forecast_confidence:
        pd.DataFrame | None = None,
    ) -> ForecastInput:

        return ForecastInput(

            alpha_scores=
            alpha_scores.copy()
            if alpha_scores is not None
            else None,

            expected_returns=
            expected_returns.copy()
            if expected_returns is not None
            else None,

            forecast_confidence=
            forecast_confidence.copy()
            if forecast_confidence is not None
            else None,
        )

    # ========================================================
    # FEATURE CATEGORY BUDGET
    # ========================================================

    @staticmethod
    def from_feature_category_budget(
        category_budget_result:
        Any,
    ) -> ForecastInput:
        """
        Consumes output of:

        feature_category_budget.py

        Expected attributes:

        final_feature_weights
        expected_returns
        confidence_scores
        """

        alpha_scores = getattr(
            category_budget_result,
            "final_feature_weights",
            None,
        )

        expected_returns = getattr(
            category_budget_result,
            "expected_returns",
            None,
        )

        confidence = getattr(
            category_budget_result,
            "confidence_scores",
            None,
        )

        return ForecastInput(

            alpha_scores=
            alpha_scores.copy()
            if isinstance(
                alpha_scores,
                pd.DataFrame,
            )
            else alpha_scores,

            expected_returns=
            expected_returns.copy()
            if isinstance(
                expected_returns,
                pd.DataFrame,
            )
            else expected_returns,

            forecast_confidence=
            confidence.copy()
            if isinstance(
                confidence,
                pd.DataFrame,
            )
            else confidence,
        )

    # ========================================================
    # COMPOSITE SCORE ENGINE
    # ========================================================

    @staticmethod
    def from_composite_scores(
        composite_scores:
        pd.DataFrame,
        *,
        confidence:
        pd.DataFrame | None = None,
    ) -> ForecastInput:
        """
        Converts composite score engine
        output into ForecastInput.
        """

        AlphaAdapter._validate_dataframe(
            composite_scores,
            "composite_scores",
        )

        return ForecastInput(

            alpha_scores=
            composite_scores.copy(),

            forecast_confidence=
            confidence.copy()
            if confidence is not None
            else None,
        )

    # ========================================================
    # IC ENGINE
    # ========================================================

    @staticmethod
    def from_ic_engine(
        ic_result:
        Any,
    ) -> ForecastInput:
        """
        Consumes IC engine output.

        Supports:

        expected_returns
        alpha_scores
        confidence
        """

        alpha_scores = getattr(
            ic_result,
            "alpha_scores",
            None,
        )

        expected_returns = getattr(
            ic_result,
            "expected_returns",
            None,
        )

        confidence = getattr(
            ic_result,
            "forecast_confidence",
            None,
        )

        return ForecastInput(

            alpha_scores=
            alpha_scores,

            expected_returns=
            expected_returns,

            forecast_confidence=
            confidence,
        )

    # ========================================================
    # ALPHA PIPELINE RESULT
    # ========================================================

    @staticmethod
    def from_alpha_pipeline(
        alpha_results:
        dict[str, Any],
    ) -> ForecastInput:
        """
        Compatible with:

        run_alpha_pipeline()

        Example keys:

        final_feature_weights
        expected_returns
        confidence_scores
        composite_scores
        """

        alpha_scores = (
            alpha_results.get(
                "composite_scores"
            )
            or
            alpha_results.get(
                "final_feature_weights"
            )
        )

        expected_returns = (
            alpha_results.get(
                "expected_returns"
            )
        )

        confidence = (
            alpha_results.get(
                "confidence_scores"
            )
        )

        return ForecastInput(

            alpha_scores=
            alpha_scores.copy()
            if isinstance(
                alpha_scores,
                pd.DataFrame,
            )
            else alpha_scores,

            expected_returns=
            expected_returns.copy()
            if isinstance(
                expected_returns,
                pd.DataFrame,
            )
            else expected_returns,

            forecast_confidence=
            confidence.copy()
            if isinstance(
                confidence,
                pd.DataFrame,
            )
            else confidence,
        )

    # ========================================================
    # GENERIC UNIVERSAL ADAPTER
    # ========================================================

    @staticmethod
    def from_object(
        obj: Any,
    ) -> ForecastInput:
        """
        Generic institutional adapter.

        Searches common alpha names.
        """

        alpha_scores = None

        expected_returns = None

        confidence = None

        for attr in [
            "alpha_scores",
            "composite_scores",
            "final_feature_weights",
            "scores",
        ]:

            if hasattr(obj, attr):

                alpha_scores = getattr(
                    obj,
                    attr,
                )

                break

        for attr in [
            "expected_returns",
            "forecast_returns",
            "predicted_returns",
        ]:

            if hasattr(obj, attr):

                expected_returns = getattr(
                    obj,
                    attr,
                )

                break

        for attr in [
            "forecast_confidence",
            "confidence_scores",
            "confidence",
        ]:

            if hasattr(obj, attr):

                confidence = getattr(
                    obj,
                    attr,
                )

                break

        return ForecastInput(

            alpha_scores=
            alpha_scores,

            expected_returns=
            expected_returns,

            forecast_confidence=
            confidence,
        )
    

# ============================================================
# PART 10 — MAIN.PY ADAPTER
#
# Bridges Existing Platform
# →
# PipelineInput
#
# Converts:
#   final_df
#   ensemble probabilities
#   alpha outputs
#   portfolio selection outputs
#
# into:
#   PipelineInput
#
# This is the critical adapter that allows your
# current platform to plug directly into the new
# institutional portfolio construction engine.
# ============================================================


class MainPipelineAdapter:
    """
    Adapter between existing platform and
    institutional construction framework.
    """

    # ========================================================
    # INTERNAL VALIDATION
    # ========================================================

    @staticmethod
    def _require_columns(
        df: pd.DataFrame,
        cols: list[str],
        name: str,
    ) -> None:

        missing = [
            c
            for c in cols
            if c not in df.columns
        ]

        if missing:

            raise ValueError(
                f"{name} missing columns: {missing}"
            )

    # ========================================================
    # MARKET DATA
    # ========================================================

    @staticmethod
    def build_market_input(
        final_df: pd.DataFrame,
    ) -> MarketDataInput:
        """
        Extract market data from final_df.
        """

        MainPipelineAdapter._require_columns(
            final_df,
            [
                "Date",
                "Ticker",
                "Close",
            ],
            "final_df",
        )

        prices = (
            final_df
            .pivot_table(
                index="Date",
                columns="Ticker",
                values="Close",
            )
            .sort_index()
        )

        returns = (
            prices
            .pct_change()
        )

        volumes = None

        if "Volume" in final_df.columns:

            volumes = (
                final_df
                .pivot_table(
                    index="Date",
                    columns="Ticker",
                    values="Volume",
                )
                .sort_index()
            )

        market_caps = None

        if "MarketCap" in final_df.columns:

            market_caps = (
                final_df
                .pivot_table(
                    index="Date",
                    columns="Ticker",
                    values="MarketCap",
                )
                .sort_index()
            )

        return MarketDataInput(

            prices=prices,

            returns=returns,

            volumes=volumes,

            market_caps=market_caps,
        )

    # ========================================================
    # FORECAST INPUT
    # ========================================================

    @staticmethod
    def build_forecast_input(
        *,
        alpha_results: dict[str, Any] | None = None,

        ensemble_proba: np.ndarray | None = None,

        latest_universe: pd.DataFrame | None = None,
    ) -> ForecastInput:
        """
        Create ForecastInput from:

        run_alpha_pipeline()

        ensemble probabilities

        latest universe snapshot
        """

        alpha_scores = None
        expected_returns = None
        confidence = None

        # ----------------------------------
        # ALPHA PIPELINE
        # ----------------------------------

        if alpha_results is not None:

            alpha_scores = (
                alpha_results.get(
                    "final_feature_weights"
                )
            )

            expected_returns = (
                alpha_results.get(
                    "expected_returns"
                )
            )

            confidence = (
                alpha_results.get(
                    "confidence_scores"
                )
            )

        # ----------------------------------
        # ENSEMBLE PROBABILITIES
        # ----------------------------------

        if latest_universe is not None:

            expected_returns = (
                latest_universe[
                    ["Company", "Probability", "Confidence"]
                ]
                .copy()
                .rename(
                    columns={
                        "Prediction_Prob": "ensemble_score"
                    }
                )
            )

        return ForecastInput(
            alpha_scores=alpha_scores,
            expected_returns=expected_returns,
            forecast_confidence=confidence,
        )
    # ========================================================
    # FACTOR INPUT
    # ========================================================

    @staticmethod
    def build_factor_input(
        final_df: pd.DataFrame,
    ) -> FactorInput:
        """
        Create factor model input.

        Safe fallback if factors absent.
        """

        exposures = None

        factor_cols = [

            c

            for c in final_df.columns

            if c.startswith("Factor_")
        ]

        if len(factor_cols) > 0:

            exposures = final_df[
                factor_cols
            ].copy()

        return FactorInput(
            factor_exposures=exposures
        )

    # ========================================================
    # PORTFOLIO STATE INPUT
    # ========================================================

    @staticmethod
    def build_portfolio_input(
        portfolio: pd.DataFrame | None,
    ) -> PortfolioInput:
        """
        Convert current portfolio.
        """

        if portfolio is None:

            return PortfolioInput()

        weights = None

        if (
            "Ticker" in portfolio.columns
            and
            "Position_Weight"
            in portfolio.columns
        ):

            weights = pd.Series(
                portfolio[
                    "Position_Weight"
                ].values,
                index=portfolio[
                    "Ticker"
                ].values,
            )

        return PortfolioInput(

            current_weights=
            weights,

            current_holdings=
            portfolio.copy(),
        )

    # ========================================================
    # LIQUIDITY INPUT
    # ========================================================

    @staticmethod
    def build_liquidity_input(
        final_df: pd.DataFrame,
    ) -> LiquidityInput:
        """
        Build liquidity information.
        """

        adv = None

        if "Volume" in final_df.columns:

            adv = (

                final_df
                .groupby("Ticker")[
                    "Volume"
                ]
                .mean()
            )

        spreads = None

        if (
            "BidAskSpread"
            in final_df.columns
        ):

            spreads = (

                final_df
                .groupby("Ticker")[
                    "BidAskSpread"
                ]
                .mean()
            )

        return LiquidityInput(

            average_daily_volume=
            adv,

            bid_ask_spread=
            spreads,

            participation_limit=
            0.10,
        )

    # ========================================================
    # CONSTRAINT INPUT
    # ========================================================

    @staticmethod
    def build_constraint_input(
        final_df: pd.DataFrame,
    ) -> ConstraintInput:
        """
        Build mappings.
        """

        sector_map = None

        if "Sector" in final_df.columns:

            sector_map = (

                final_df
                .drop_duplicates(
                    "Ticker"
                )
                .set_index(
                    "Ticker"
                )["Sector"]
            )

        industry_map = None

        if "Industry" in final_df.columns:

            industry_map = (

                final_df
                .drop_duplicates(
                    "Ticker"
                )
                .set_index(
                    "Ticker"
                )["Industry"]
            )

        country_map = None

        if "Country" in final_df.columns:

            country_map = (

                final_df
                .drop_duplicates(
                    "Ticker"
                )
                .set_index(
                    "Ticker"
                )["Country"]
            )

        return ConstraintInput(

            sector_map=
            sector_map,

            industry_map=
            industry_map,

            country_map=
            country_map,
        )

    # ========================================================
    # MASTER CONVERSION
    # ========================================================

    @staticmethod
    def build_pipeline_input(
        *,
        final_df: pd.DataFrame,

        alpha_results: dict[str, Any]
        | None = None,

        ensemble_proba: np.ndarray
        | None = None,

        latest_universe: pd.DataFrame
        | None = None,

        portfolio: pd.DataFrame
        | None = None,
    ) -> PipelineInput:
        """
        Main entry point.

        Converts current platform outputs
        into PipelineInput.
        """

        market_data = (
            MainPipelineAdapter
            .build_market_input(
                final_df
            )
        )

        forecast_data = (
            MainPipelineAdapter
            .build_forecast_input(

                alpha_results=
                alpha_results,

                ensemble_proba=
                ensemble_proba,

                latest_universe=
                latest_universe,
            )
        )

        factor_data = (
            MainPipelineAdapter
            .build_factor_input(
                final_df
            )
        )

        portfolio_data = (
            MainPipelineAdapter
            .build_portfolio_input(
                portfolio
            )
        )

        liquidity_data = (
            MainPipelineAdapter
            .build_liquidity_input(
                final_df
            )
        )

        constraint_data = (
            MainPipelineAdapter
            .build_constraint_input(
                final_df
            )
        )

        return PipelineInput(

            market_data=
            market_data,

            forecast_data=
            forecast_data,

            factor_data=
            factor_data,

            portfolio_data=
            portfolio_data,

            liquidity_data=
            liquidity_data,

            constraint_data=
            constraint_data,

            metadata={
                "source":
                "main.py",

                "created_by":
                "MainPipelineAdapter",
            },
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def build_pipeline_input_from_main(
    *,
    final_df: pd.DataFrame,

    alpha_results: dict[str, Any]
    | None = None,

    ensemble_proba: np.ndarray
    | None = None,

    latest_universe: pd.DataFrame
    | None = None,

    portfolio: pd.DataFrame
    | None = None,
) -> PipelineInput:
    """
    Convenience wrapper for main.py

    Example
    -------
    pipeline_input =
        build_pipeline_input_from_main(
            final_df=final_df,
            alpha_results=alpha_results,
            ensemble_proba=final_proba,
            latest_universe=latest_data,
            portfolio=portfolio
        )
    """

    return (
        MainPipelineAdapter
        .build_pipeline_input(

            final_df=
            final_df,

            alpha_results=
            alpha_results,

            ensemble_proba=
            ensemble_proba,

            latest_universe=
            latest_universe,

            portfolio=
            portfolio,
        )
    )


# ============================================================
# PART 11 — INSTITUTIONAL INPUT ADAPTER ENGINE
#
# Master orchestration layer for all adapters.
#
# Purpose:
#
# Existing Platform
#       │
#       ▼
#  MainPipelineAdapter
#       │
#       ▼
# InstitutionalInputAdapterEngine
#       │
#       ▼
# PipelineInput
#
# Used by:
#
# portfolio_builder.py
# pipeline.py
# portfolio_manager.py
# workflow.py
#
# ============================================================


# ============================================================
# ENGINE CONFIG
# ============================================================


@dataclass(slots=True)
class InstitutionalInputAdapterConfig:
    """
    Controls adapter behavior.
    """

    validate_inputs: bool = True

    build_market_data: bool = True

    build_forecasts: bool = True

    build_factor_data: bool = True

    build_portfolio_state: bool = True

    build_liquidity_data: bool = True

    build_constraints: bool = True


# ============================================================
# ENGINE DIAGNOSTICS
# ============================================================


@dataclass(slots=True)
class AdapterEngineDiagnostics:
    """
    Adapter diagnostics.
    """

    market_rows: int = 0

    universe_size: int = 0

    forecast_assets: int = 0

    liquidity_assets: int = 0

    constraint_assets: int = 0

    validation_passed: bool = False

    warnings: list[str] = field(
        default_factory=list
    )


# ============================================================
# ENGINE RESULT
# ============================================================


@dataclass(slots=True)
class InstitutionalInputAdapterResult:
    """
    Final adapter result.
    """

    pipeline_input: PipelineInput

    diagnostics: AdapterEngineDiagnostics


# ============================================================
# INSTITUTIONAL INPUT ADAPTER ENGINE
# ============================================================


class InstitutionalInputAdapterEngine:
    """
    Institutional-grade adapter engine.

    Central adapter orchestration layer.

    Converts all platform outputs into a
    fully populated PipelineInput object.
    """

    # --------------------------------------------------------

    def __init__(
        self,
        config:
        InstitutionalInputAdapterConfig
        | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else InstitutionalInputAdapterConfig()
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(
        self,
        final_df: pd.DataFrame,
    ) -> None:

        if final_df is None:

            raise ValueError(
                "final_df cannot be None."
            )

        if final_df.empty:

            raise ValueError(
                "final_df is empty."
            )

        required_cols = [
            "Date",
            "Ticker",
            "Close",
        ]

        missing = [

            c

            for c in required_cols

            if c not in final_df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing columns: {missing}"
            )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    def build_diagnostics(
        self,
        *,
        final_df: pd.DataFrame,

        pipeline_input:
        PipelineInput,
    ) -> AdapterEngineDiagnostics:

        diagnostics = (
            AdapterEngineDiagnostics()
        )

        diagnostics.market_rows = (
            len(final_df)
        )

        if "Ticker" in final_df.columns:

            diagnostics.universe_size = (
                final_df["Ticker"]
                .nunique()
            )

        # -----------------------------
        # Forecast
        # -----------------------------

        forecast = (
            pipeline_input
            .forecast_data
        )

        if (
            forecast.expected_returns
            is not None
            and
            isinstance(
                forecast.expected_returns,
                pd.DataFrame,
            )
        ):

            diagnostics.forecast_assets = (
                len(
                    forecast.expected_returns
                )
            )

        # -----------------------------
        # Liquidity
        # -----------------------------

        liquidity = (
            pipeline_input
            .liquidity_data
        )

        if (
            liquidity.average_daily_volume
            is not None
        ):

            diagnostics.liquidity_assets = (
                len(
                    liquidity
                    .average_daily_volume
                )
            )

        # -----------------------------
        # Constraints
        # -----------------------------

        constraints = (
            pipeline_input
            .constraint_data
        )

        if (
            constraints.sector_map
            is not None
        ):

            diagnostics.constraint_assets = (
                len(
                    constraints.sector_map
                )
            )

        diagnostics.validation_passed = True

        return diagnostics

    # ========================================================
    # BUILD
    # ========================================================

    def build(
        self,
        *,
        final_df: pd.DataFrame,

        alpha_results:
        dict[str, Any]
        | None = None,

        ensemble_proba:
        np.ndarray
        | None = None,

        latest_universe:
        pd.DataFrame
        | None = None,

        portfolio:
        pd.DataFrame
        | None = None,
    ) -> InstitutionalInputAdapterResult:
        """
        Main engine entry.
        """

        # ----------------------------------
        # VALIDATION
        # ----------------------------------

        if self.config.validate_inputs:

            self.validate(
                final_df
            )

        # ----------------------------------
        # ADAPTER
        # ----------------------------------

        pipeline_input = (
            MainPipelineAdapter
            .build_pipeline_input(

                final_df=
                final_df,

                alpha_results=
                alpha_results,

                ensemble_proba=
                ensemble_proba,

                latest_universe=
                latest_universe,

                portfolio=
                portfolio,
            )
        )

        # ----------------------------------
        # PIPELINE INPUT VALIDATION
        # ----------------------------------

        if self.config.validate_inputs:

            PipelineInputValidator.validate(
                pipeline_input
            )

        # ----------------------------------
        # DIAGNOSTICS
        # ----------------------------------

        diagnostics = (
            self.build_diagnostics(

                final_df=
                final_df,

                pipeline_input=
                pipeline_input,
            )
        )

        return (
            InstitutionalInputAdapterResult(

                pipeline_input=
                pipeline_input,

                diagnostics=
                diagnostics,
            )
        )


# ============================================================
# FACTORY
# ============================================================


class InstitutionalInputAdapterFactory:
    """
    Factory helpers.
    """

    @staticmethod
    def create(
        *,
        config:
        InstitutionalInputAdapterConfig
        | None = None,
    ) -> InstitutionalInputAdapterEngine:

        return (
            InstitutionalInputAdapterEngine(
                config=config
            )
        )


# ============================================================
# CONVENIENCE API
# ============================================================


def build_pipeline_input_engine(
    *,
    final_df: pd.DataFrame,

    alpha_results:
    dict[str, Any]
    | None = None,

    ensemble_proba:
    np.ndarray
    | None = None,

    latest_universe:
    pd.DataFrame
    | None = None,

    portfolio:
    pd.DataFrame
    | None = None,
) -> PipelineInput:
    """
    Returns only PipelineInput.

    Most common usage.
    """

    engine = (
        InstitutionalInputAdapterFactory
        .create()
    )

    result = (
        engine.build(

            final_df=
            final_df,

            alpha_results=
            alpha_results,

            ensemble_proba=
            ensemble_proba,

            latest_universe=
            latest_universe,

            portfolio=
            portfolio,
        )
    )

    return result.pipeline_input


# ============================================================
# CONVENIENCE API
# ============================================================


def build_pipeline_input_with_diagnostics(
    *,
    final_df: pd.DataFrame,

    alpha_results:
    dict[str, Any]
    | None = None,

    ensemble_proba:
    np.ndarray
    | None = None,

    latest_universe:
    pd.DataFrame
    | None = None,

    portfolio:
    pd.DataFrame
    | None = None,
) -> InstitutionalInputAdapterResult:
    """
    Returns:

    PipelineInput
    +
    Diagnostics
    """

    engine = (
        InstitutionalInputAdapterFactory
        .create()
    )

    return (
        engine.build(

            final_df=
            final_df,

            alpha_results=
            alpha_results,

            ensemble_proba=
            ensemble_proba,

            latest_universe=
            latest_universe,

            portfolio=
            portfolio,
        )
    )


# ============================================================
# SMOKE TEST
# ============================================================


def smoke_test_input_adapter(
    *,
    final_df: pd.DataFrame,
) -> bool:
    """
    Quick validation helper.
    """

    try:

        result = (
            build_pipeline_input_with_diagnostics(
                final_df=final_df
            )
        )

        return (
            result.diagnostics
            .validation_passed
        )

    except Exception:

        return False
    

# ============================================================
# PART 12 — FACTORY APIS
# ============================================================


def create_input_engine(
    *,
    config:
    InstitutionalInputAdapterConfig
    | None = None,
) -> InstitutionalInputAdapterEngine:

    return (
        InstitutionalInputAdapterFactory
        .create(
            config=config
        )
    )


# ------------------------------------------------------------


def build_pipeline_input(
    *,
    final_df: pd.DataFrame,

    alpha_results:
    dict[str, Any]
    | None = None,

    ensemble_proba:
    np.ndarray
    | None = None,

    latest_universe:
    pd.DataFrame
    | None = None,

    portfolio:
    pd.DataFrame
    | None = None,
) -> PipelineInput:

    return (
        build_pipeline_input_engine(

            final_df=
            final_df,

            alpha_results=
            alpha_results,

            ensemble_proba=
            ensemble_proba,

            latest_universe=
            latest_universe,

            portfolio=
            portfolio,
        )
    )


# ------------------------------------------------------------


def build_pipeline_input_diagnostics(
    *,
    final_df: pd.DataFrame,

    alpha_results:
    dict[str, Any]
    | None = None,

    ensemble_proba:
    np.ndarray
    | None = None,

    latest_universe:
    pd.DataFrame
    | None = None,

    portfolio:
    pd.DataFrame
    | None = None,
) -> InstitutionalInputAdapterResult:

    return (
        build_pipeline_input_with_diagnostics(

            final_df=
            final_df,

            alpha_results=
            alpha_results,

            ensemble_proba=
            ensemble_proba,

            latest_universe=
            latest_universe,

            portfolio=
            portfolio,
        )
    )


# ------------------------------------------------------------


__all__ = [
    "create_input_engine",
    "build_pipeline_input",
    "build_pipeline_input_diagnostics",
    "smoke_test_input_adapter",
]