"""
==========================================================
PORTFOLIO CONSTRUCTION CONFIGURATION
==========================================================

Production configuration for portfolio construction.

Portfolio Selection chooses WHAT to buy.

Portfolio Construction decides

• how much
• whether constraints are violated
• whether positions must be reduced

==========================================================
"""

from dataclasses import dataclass


@dataclass(slots=True)
class ConstructionConfig:
    """
    Portfolio construction constraints.
    """

    # --------------------------------------------------
    # Portfolio Size
    # --------------------------------------------------

    target_positions: int = 20

    min_positions: int = 10

    max_positions: int = 30

    # --------------------------------------------------
    # Position Constraints
    # --------------------------------------------------

    max_position_weight: float = 0.10

    min_position_weight: float = 0.01

    # --------------------------------------------------
    # Sector Constraints
    # --------------------------------------------------

    sector_cap: float = 0.30

    industry_cap: float = 0.20

    # --------------------------------------------------
    # Liquidity
    # --------------------------------------------------

    minimum_average_volume: float = 500000

    minimum_price: float = 20.0

    maximum_spread: float = 0.02

    # --------------------------------------------------
    # Turnover
    # --------------------------------------------------

    maximum_turnover: float = 0.25

    rebalance_buffer: int = 3

    # --------------------------------------------------
    # Cash
    # --------------------------------------------------

    allow_cash: bool = False

    target_cash_weight: float = 0.0

    # --------------------------------------------------
    # Long / Short
    # --------------------------------------------------

    allow_short: bool = False

    gross_exposure: float = 1.0

    net_exposure: float = 1.0

    # --------------------------------------------------
    # Style
    # --------------------------------------------------

    market_neutral: bool = False

    beta_neutral: bool = False


# --------------------------------------------------
# PORTFOLIO COLUMNS
# --------------------------------------------------


PORTFOLIO_COLUMNS = {
    "date": "Date",
    "ticker": "Ticker",
    "weight": "Position_Weight",
    "sector": "Sector",
    "industry": "Industry",
}



    