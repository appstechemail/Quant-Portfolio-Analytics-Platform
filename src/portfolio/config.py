"""
============================================================
PORTFOLIO CONSTRUCTION CONFIGURATION
============================================================
Institutional-grade portfolio construction configuration.
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ============================================================
# SELECTION METHODS
# ============================================================

class SelectionMethod(str, Enum):

    LONG_ONLY = "long_only"

    LONG_SHORT = "long_short"


# ============================================================
# WEIGHTING METHODS
# ============================================================

class WeightMethod(str, Enum):

    EQUAL = "equal"

    SCORE = "score"

    SOFTMAX = "softmax"

    RISK_PARITY = "risk_parity"

    INVERSE_VOLATILITY = "inverse_volatility"

    KELLY = "kelly"


# ============================================================
# REBALANCE FREQUENCY
# ============================================================

class RebalanceFrequency(str, Enum):

    DAILY = "daily"

    WEEKLY = "weekly"

    MONTHLY = "monthly"

    QUARTERLY = "quarterly"


# ============================================================
# CONFIG
# ============================================================

@dataclass(slots=True)
class PortfolioSelectionConfig:

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    selection_method: SelectionMethod = (
        SelectionMethod.LONG_ONLY
    )

    weight_method: WeightMethod = (
        WeightMethod.SOFTMAX
    )

    # --------------------------------------------------------
    # Data Columns
    # --------------------------------------------------------

    score_column: str = "Composite_Score"

    rank_column: str = "Composite_Rank"

    percentile_column: str = (
        "Composite_Percentile"
    )

    selected_column: str = (
        "Portfolio_Selected"
    )

    weight_column: str = (
        "Portfolio_Weight"
    )

    # --------------------------------------------------------
    # Portfolio Size
    # --------------------------------------------------------

    top_n: int = 30

    bottom_n: int = 0

    selection_percentile: float | None = None

    softmax_temperature: float = 1.0

    min_positions: int = 10

    max_positions: int = 100

    long_book_size: int = 30

    short_book_size: int = 30

    # --------------------------------------------------------
    # Weight Constraints
    # --------------------------------------------------------

    max_weight: float = 0.10

    min_weight: float = 0.00

    min_score: float = 0.0

    # --------------------------------------------------------
    # Exposure Constraints
    # --------------------------------------------------------

    gross_exposure_target: float = 1.0

    net_exposure_target: float = 1.0

    target_volatility: float | None = None

    tracking_error_target: float | None = None

    risk_aversion: float = 1.0

    # --------------------------------------------------------
    # Concentration Constraints
    # --------------------------------------------------------

    max_sector_weight: float = 0.30

    max_industry_weight: float = 0.20

    max_country_weight: float = 1.00

    # --------------------------------------------------------
    # Liquidity
    # --------------------------------------------------------

    min_adv: float = 0.0

    min_dollar_volume: float = 0.0

    # --------------------------------------------------------
    # Turnover Controls
    # --------------------------------------------------------

    max_turnover: float = 0.25

    min_trade_weight: float = 0.001

    # --------------------------------------------------------
    # Cash
    # --------------------------------------------------------

    cash_buffer: float = 0.01

    # --------------------------------------------------------
    # Neutrality
    # --------------------------------------------------------

    sector_neutral: bool = False

    beta_neutral: bool = False

    # --------------------------------------------------------
    # Rebalance
    # --------------------------------------------------------

    rebalance_frequency: (
        RebalanceFrequency
    ) = RebalanceFrequency.MONTHLY

    # --------------------------------------------------------
    # Misc
    # --------------------------------------------------------

    allow_fractional_weights: bool = True


# ============================================================
# VALIDATION
# ============================================================

def validate_portfolio_config(
    config: PortfolioSelectionConfig,
) -> None:

    # Selection

    if config.top_n <= 0:
        raise ValueError(
            "top_n must be > 0."
        )

    if config.bottom_n < 0:
        raise ValueError(
            "bottom_n cannot be negative."
        )

    if (
        config.selection_percentile
        is not None
        and not (
            0
            < config.selection_percentile
            <= 1
        )
    ):
        raise ValueError(
            "selection_percentile must lie in (0,1]."
        )

    # Position Counts

    if config.min_positions <= 0:
        raise ValueError(
            "min_positions must be positive."
        )

    if (
        config.max_positions
        < config.min_positions
    ):
        raise ValueError(
            "max_positions must be >= min_positions."
        )

    # Weights

    if config.max_weight <= 0:
        raise ValueError(
            "max_weight must be positive."
        )

    if config.max_weight > 1:
        raise ValueError(
            "max_weight cannot exceed 1."
        )

    if config.min_weight < 0:
        raise ValueError(
            "min_weight cannot be negative."
        )

    if (
        config.min_weight
        > config.max_weight
    ):
        raise ValueError(
            "min_weight cannot exceed max_weight."
        )

    # Turnover

    if config.max_turnover < 0:
        raise ValueError(
            "max_turnover cannot be negative."
        )

    if config.max_turnover > 1:
        raise ValueError(
            "max_turnover cannot exceed 1."
        )

    if config.min_trade_weight < 0:
        raise ValueError(
            "min_trade_weight cannot be negative."
        )

    # Cash

    if not (
        0 <= config.cash_buffer < 1
    ):
        raise ValueError(
            "cash_buffer must lie in [0,1)."
        )

    # Exposure

    if (
        config.gross_exposure_target
        <= 0
    ):
        raise ValueError(
            "gross_exposure_target must be positive."
        )

    if (
        abs(
            config.net_exposure_target
        )
        > config.gross_exposure_target
    ):
        raise ValueError(
            "net exposure cannot exceed gross exposure."
        )

    # Long Only Rules

    if (
        config.selection_method
        == SelectionMethod.LONG_ONLY
    ):

        if config.bottom_n != 0:

            raise ValueError(
                "bottom_n must be zero for LONG_ONLY portfolios."
            )

        if (
            config.net_exposure_target
            != 1.0
        ):

            raise ValueError(
                "LONG_ONLY portfolios must have net exposure = 1."
            )

    # Long Short Rules

    if (
        config.selection_method
        == SelectionMethod.LONG_SHORT
    ):

        if config.bottom_n <= 0:

            raise ValueError(
                "LONG_SHORT portfolios require bottom_n > 0."
            )

        if config.short_book_size <= 0:

            raise ValueError(
                "short_book_size must be positive."
            )


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_PORTFOLIO_CONFIG = (
    PortfolioSelectionConfig()
)

validate_portfolio_config(
    DEFAULT_PORTFOLIO_CONFIG
)

# ============================================================
# EXPORTS
# ============================================================

__all__ = [

    "SelectionMethod",

    "WeightMethod",

    "RebalanceFrequency",

    "PortfolioSelectionConfig",

    "DEFAULT_PORTFOLIO_CONFIG",

    "validate_portfolio_config",
]