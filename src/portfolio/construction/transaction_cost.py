
"""
==============================================================
TRANSACTION COST ENGINE
==============================================================

File
----
transaction_cost.py

Purpose
-------
Estimate realistic transaction costs for portfolio trades.

The module is completely independent from

    • Optimizer
    • Portfolio
    • Execution
    • Broker

Responsibilities
----------------
• Commission estimation
• Spread estimation
• Slippage estimation
• Market impact
• Taxes
• Borrow cost
• Exchange fees
• Cost diagnostics

==============================================================
"""

from __future__ import annotations

import logging

from abc import ABC
from abc import abstractmethod

from dataclasses import dataclass
from dataclasses import field

from enum import Enum

import numpy as np
import pandas as pd

from .rebalance import Trade
from .rebalance import TradeSide


LOGGER = logging.getLogger(__name__)


# ==============================================================
# ENUMS
# ==============================================================


class CostComponent(str, Enum):
    """
    Individual cost components.
    """

    COMMISSION = "commission"

    SPREAD = "spread"

    SLIPPAGE = "slippage"

    MARKET_IMPACT = "market_impact"

    TAX = "tax"

    BORROW = "borrow"

    EXCHANGE = "exchange"


class CostModelType(str, Enum):
    """
    Cost model families.
    """

    FIXED = "fixed"

    PERCENTAGE = "percentage"

    ADV = "adv"

    SQUARE_ROOT = "square_root"

    CUSTOM = "custom"


# ==============================================================
# COST BREAKDOWN
# ==============================================================


@dataclass(slots=True)
class CostBreakdown:
    """
    Cost decomposition for one trade.
    """

    commission: float = 0.0

    spread: float = 0.0

    slippage: float = 0.0

    market_impact: float = 0.0

    tax: float = 0.0

    borrow: float = 0.0

    exchange: float = 0.0

    # ----------------------------------------------------------

    @property
    def total_cost(
        self,
    ) -> float:

        return (

            self.commission

            + self.spread

            + self.slippage

            + self.market_impact

            + self.tax

            + self.borrow

            + self.exchange

        )


# ==============================================================
# COST RESULT
# ==============================================================


@dataclass(slots=True)
class TransactionCostResult:
    """
    Output for one trade.
    """

    trade: Trade

    breakdown: CostBreakdown

    metadata: dict = field(
        default_factory=dict
    )


# ==============================================================
# BASE COST MODEL
# ==============================================================


class BaseCostModel(ABC):
    """
    Abstract transaction-cost model.
    """

    @abstractmethod
    def estimate(
        self,
        trade: Trade,
    ) -> float:
        """
        Estimate one component.
        """


# ==============================================================
# VALIDATION
# ==============================================================


def validate_trade(
    trade: Trade,
) -> None:
    """
    Validate trade.
    """

    if not isinstance(
        trade,
        Trade,
    ):

        raise TypeError(
            "Expected Trade object."
        )

    if trade.side not in (

        TradeSide.BUY,

        TradeSide.SELL,

        TradeSide.HOLD,

    ):

        raise ValueError(
            "Invalid trade side."
        )


# ==============================================================
# HELPERS
# ==============================================================


def trade_notional(
    trade: Trade,
) -> float:
    """
    Absolute trade size.
    """

    return float(
        abs(
            trade.trade_weight
        )
    )


def is_buy(
    trade: Trade,
) -> bool:

    return (
        trade.side
        == TradeSide.BUY
    )


def is_sell(
    trade: Trade,
) -> bool:

    return (
        trade.side
        == TradeSide.SELL
    )


# ==============================================================
# EXPORTS
# ==============================================================


__all__ = [
    "CostComponent",
    "CostModelType",
    "CostBreakdown",
    "TransactionCostResult",
    "BaseCostModel",
    "validate_trade",
    "trade_notional",
    "is_buy",
    "is_sell",
]


# Part 2

# ==============================================================
# COMMISSION MODELS
# ==============================================================


class FixedCommissionModel(
    BaseCostModel,
):
    """
    Fixed commission per trade.

    Example
    -------
    Every executed trade costs
    20 currency units.
    """

    def __init__(
        self,
        commission: float,
    ) -> None:

        if commission < 0:

            raise ValueError(
                "Commission must be non-negative."
            )

        self.commission = float(
            commission
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(
            trade
        )

        if trade.side == TradeSide.HOLD:

            return 0.0

        return self.commission


# ==============================================================
# PERCENTAGE COMMISSION
# ==============================================================


class PercentageCommissionModel(
    BaseCostModel,
):
    """
    Commission expressed as a percentage
    of trade notional.

    Example
    -------
    0.10%

    rate = 0.001
    """

    def __init__(
        self,
        rate: float,
    ) -> None:

        if rate < 0:

            raise ValueError(
                "Rate must be non-negative."
            )

        self.rate = float(
            rate
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(
            trade
        )

        if trade.side == TradeSide.HOLD:

            return 0.0

        return (

            trade_notional(
                trade
            )

            * self.rate

        )


# ==============================================================
# MINIMUM COMMISSION
# ==============================================================


class MinimumCommissionModel(
    BaseCostModel,
):
    """
    Percentage commission
    with broker minimum.
    """

    def __init__(
        self,
        *,
        rate: float,
        minimum: float,
    ) -> None:

        self.rate = float(
            rate
        )

        self.minimum = float(
            minimum
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(
            trade
        )

        if trade.side == TradeSide.HOLD:

            return 0.0

        commission = (

            trade_notional(
                trade
            )

            * self.rate

        )

        return max(

            commission,

            self.minimum,

        )


# ==============================================================
# TIERED COMMISSION
# ==============================================================


class TieredCommissionModel(
    BaseCostModel,
):
    """
    Tiered commission schedule.

    Example

        < 2%

            20 bps

        >= 2%

            10 bps
    """

    def __init__(
        self,
        tiers: list[
            tuple[
                float,
                float,
            ]
        ],
    ) -> None:

        if len(tiers) == 0:

            raise ValueError(
                "No tiers provided."
            )

        self.tiers = sorted(
            tiers,
            key=lambda x: x[0],
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(
            trade
        )

        notional = trade_notional(
            trade
        )

        rate = self.tiers[-1][1]

        for threshold, tier_rate in self.tiers:

            if notional <= threshold:

                rate = tier_rate

                break

        return (

            notional

            * rate

        )


# ==============================================================
# COMMISSION FACTORY
# ==============================================================


def build_commission_model(
    model_type: CostModelType,
    **kwargs,
) -> BaseCostModel:
    """
    Factory.
    """

    if model_type == CostModelType.FIXED:

        return FixedCommissionModel(
            **kwargs
        )

    if model_type == CostModelType.PERCENTAGE:

        return PercentageCommissionModel(
            **kwargs
        )

    raise ValueError(

        f"Unsupported model: {model_type}"

    )


# ==============================================================
# COMMISSION UTILITIES
# ==============================================================


def estimate_commission(
    trade: Trade,
    model: BaseCostModel,
) -> float:
    """
    Convenience wrapper.
    """

    return model.estimate(
        trade
    )


# ==============================================================
# EXPORTS
# ==============================================================


__all__.extend(

    [

        "FixedCommissionModel",

        "PercentageCommissionModel",

        "MinimumCommissionModel",

        "TieredCommissionModel",

        "build_commission_model",

        "estimate_commission",

    ]

)


# Part 3

# ==============================================================
# SPREAD MODELS
# ==============================================================


class FixedSpreadModel(
    BaseCostModel,
):
    """
    Fixed bid-ask spread.

    Example
    -------
    Every trade pays

        5 bps

    regardless of security.
    """

    def __init__(
        self,
        spread: float,
    ) -> None:

        if spread < 0:

            raise ValueError(
                "Spread must be non-negative."
            )

        self.spread = float(
            spread
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(
            trade
        )

        if trade.side == TradeSide.HOLD:

            return 0.0

        return (

            trade_notional(
                trade
            )

            * self.spread

        )


# ==============================================================
# HALF-SPREAD MODEL
# ==============================================================


class HalfSpreadModel(
    BaseCostModel,
):
    """
    Crossing the spread.

    Assumes execution occurs
    at one side of the market.

    Cost = 1/2 spread.
    """

    def __init__(
        self,
        spread: float,
    ) -> None:

        self.spread = float(
            spread
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(
            trade
        )

        if trade.side == TradeSide.HOLD:

            return 0.0

        return (

            trade_notional(
                trade
            )

            * self.spread

            * 0.5

        )


# ==============================================================
# VARIABLE SPREAD MODEL
# ==============================================================


class VariableSpreadModel(
    BaseCostModel,
):
    """
    Security-specific spreads.

    Parameters
    ----------
    spread_map

        ticker -> spread
    """

    def __init__(
        self,
        spread_map: dict[
            str,
            float,
        ],
        *,
        default_spread: float = 0.0005,
    ) -> None:

        self.spread_map = spread_map

        self.default_spread = float(
            default_spread
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(
            trade
        )

        if trade.side == TradeSide.HOLD:

            return 0.0

        spread = self.spread_map.get(

            trade.ticker,

            self.default_spread,

        )

        return (

            trade_notional(
                trade
            )

            * spread

        )


# ==============================================================
# PERCENTILE SPREAD MODEL
# ==============================================================


class PercentileSpreadModel(
    BaseCostModel,
):
    """
    Liquidity-adjusted spread.

    Large trades pay larger spreads.
    """

    def __init__(
        self,
        *,
        low_spread: float = 0.0005,
        high_spread: float = 0.0030,
        threshold: float = 0.02,
    ) -> None:

        self.low_spread = float(
            low_spread
        )

        self.high_spread = float(
            high_spread
        )

        self.threshold = float(
            threshold
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(
            trade
        )

        if trade.side == TradeSide.HOLD:

            return 0.0

        notional = trade_notional(
            trade
        )

        spread = (

            self.high_spread

            if notional >= self.threshold

            else self.low_spread

        )

        return (

            notional

            * spread

        )


# ==============================================================
# SPREAD FACTORY
# ==============================================================


def build_spread_model(
    model_type: CostModelType,
    **kwargs,
) -> BaseCostModel:
    """
    Factory.
    """

    if model_type == CostModelType.FIXED:

        return FixedSpreadModel(
            **kwargs
        )

    if model_type == CostModelType.PERCENTAGE:

        return VariableSpreadModel(
            **kwargs
        )

    raise ValueError(

        f"Unsupported spread model: {model_type}"

    )


# ==============================================================
# UTILITIES
# ==============================================================


def estimate_spread(
    trade: Trade,
    model: BaseCostModel,
) -> float:
    """
    Convenience wrapper.
    """

    return model.estimate(
        trade
    )


# ==============================================================
# EXPORTS
# ==============================================================


__all__.extend(

    [

        "FixedSpreadModel",

        "HalfSpreadModel",

        "VariableSpreadModel",

        "PercentileSpreadModel",

        "build_spread_model",

        "estimate_spread",

    ]

)


# Part 4

# ==============================================================
# SLIPPAGE MODELS
# ==============================================================


class FixedSlippageModel(
    BaseCostModel,
):
    """
    Fixed slippage.

    Example
    -------
    Every trade pays

        10 bps
    """

    def __init__(
        self,
        slippage: float,
    ) -> None:

        if slippage < 0:

            raise ValueError(
                "Slippage must be non-negative."
            )

        self.slippage = float(
            slippage
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(
            trade
        )

        if trade.side == TradeSide.HOLD:

            return 0.0

        return (

            trade_notional(
                trade
            )

            * self.slippage

        )


# ==============================================================
# PERCENTAGE SLIPPAGE
# ==============================================================


class PercentageSlippageModel(
    BaseCostModel,
):
    """
    Slippage proportional to
    trade size.
    """

    def __init__(
        self,
        rate: float,
    ) -> None:

        self.rate = float(
            rate
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(
            trade
        )

        if trade.side == TradeSide.HOLD:

            return 0.0

        return (

            trade_notional(
                trade
            )

            * self.rate

        )


# ==============================================================
# VOLATILITY SLIPPAGE
# ==============================================================


class VolatilitySlippageModel(
    BaseCostModel,
):
    """
    Volatility-adjusted slippage.

    Cost

        volatility × multiplier
    """

    def __init__(
        self,
        volatility_map: dict[
            str,
            float,
        ],
        *,
        multiplier: float = 0.50,
        default_volatility: float = 0.02,
    ) -> None:

        self.volatility_map = volatility_map

        self.multiplier = float(
            multiplier
        )

        self.default_volatility = float(
            default_volatility
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(
            trade
        )

        if trade.side == TradeSide.HOLD:

            return 0.0

        volatility = self.volatility_map.get(

            trade.ticker,

            self.default_volatility,

        )

        rate = (

            volatility

            * self.multiplier

        )

        return (

            trade_notional(
                trade
            )

            * rate

        )


# ==============================================================
# LIQUIDITY SLIPPAGE
# ==============================================================


class LiquiditySlippageModel(
    BaseCostModel,
):
    """
    Liquidity-based slippage.

    Lower ADV

        Higher slippage.
    """

    def __init__(
        self,
        adv_map: dict[
            str,
            float,
        ],
        *,
        multiplier: float = 0.01,
        default_adv: float = 1_000_000.0,
    ) -> None:

        self.adv_map = adv_map

        self.multiplier = float(
            multiplier
        )

        self.default_adv = float(
            default_adv
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(
            trade
        )

        if trade.side == TradeSide.HOLD:

            return 0.0

        adv = self.adv_map.get(

            trade.ticker,

            self.default_adv,

        )

        adv = max(
            adv,
            1.0,
        )

        rate = (

            self.multiplier

            / np.sqrt(adv)

        )

        return (

            trade_notional(
                trade
            )

            * rate

        )


# ==============================================================
# COMPOSITE SLIPPAGE
# ==============================================================


class CompositeSlippageModel(
    BaseCostModel,
):
    """
    Combine multiple slippage models.
    """

    def __init__(
        self,
        models: list[
            BaseCostModel
        ],
    ) -> None:

        self.models = models

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        total = 0.0

        for model in self.models:

            total += model.estimate(
                trade
            )

        return total


# ==============================================================
# FACTORY
# ==============================================================


def build_slippage_model(
    model_type: CostModelType,
    **kwargs,
) -> BaseCostModel:
    """
    Factory.
    """

    if model_type == CostModelType.FIXED:

        return FixedSlippageModel(
            **kwargs
        )

    if model_type == CostModelType.PERCENTAGE:

        return PercentageSlippageModel(
            **kwargs
        )

    raise ValueError(

        f"Unsupported slippage model: {model_type}"

    )


# ==============================================================
# UTILITIES
# ==============================================================


def estimate_slippage(
    trade: Trade,
    model: BaseCostModel,
) -> float:
    """
    Convenience wrapper.
    """

    return model.estimate(
        trade
    )


# ==============================================================
# EXPORTS
# ==============================================================


__all__.extend(

    [

        "FixedSlippageModel",

        "PercentageSlippageModel",

        "VolatilitySlippageModel",

        "LiquiditySlippageModel",

        "CompositeSlippageModel",

        "build_slippage_model",

        "estimate_slippage",

    ]

)


# Part 5

# ==============================================================
# MARKET IMPACT MODELS
# ==============================================================


class LinearMarketImpactModel(
    BaseCostModel,
):
    """
    Linear market impact.

    Cost

        rate × trade size
    """

    def __init__(
        self,
        rate: float,
    ) -> None:

        if rate < 0:

            raise ValueError(
                "Rate must be non-negative."
            )

        self.rate = float(
            rate
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(
            trade
        )

        if trade.side == TradeSide.HOLD:

            return 0.0

        return (

            trade_notional(
                trade
            )

            * self.rate

        )


# ==============================================================
# SQUARE-ROOT IMPACT
# ==============================================================


class SquareRootImpactModel(
    BaseCostModel,
):
    """
    Institutional square-root impact.

    Cost

        coefficient × √trade_size
    """

    def __init__(
        self,
        coefficient: float = 0.01,
    ) -> None:

        self.coefficient = float(
            coefficient
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(
            trade
        )

        if trade.side == TradeSide.HOLD:

            return 0.0

        notional = trade_notional(
            trade
        )

        return (

            self.coefficient

            * np.sqrt(
                notional
            )

        )


# ==============================================================
# ADV PARTICIPATION MODEL
# ==============================================================


class ADVParticipationImpactModel(
    BaseCostModel,
):
    """
    Market impact based on
    Average Daily Volume participation.

    Higher participation

        Higher impact.
    """

    def __init__(
        self,
        adv_map: dict[str, float],
        *,
        coefficient: float = 0.10,
        default_adv: float = 1_000_000.0,
    ) -> None:

        self.adv_map = adv_map

        self.coefficient = float(
            coefficient
        )

        self.default_adv = float(
            default_adv
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(
            trade
        )

        if trade.side == TradeSide.HOLD:

            return 0.0

        adv = self.adv_map.get(

            trade.ticker,

            self.default_adv,

        )

        adv = max(
            adv,
            1.0,
        )

        participation = (

            trade_notional(
                trade
            )

            / adv

        )

        return (

            self.coefficient

            * participation

        )


# ==============================================================
# NONLINEAR IMPACT
# ==============================================================


class NonLinearImpactModel(
    BaseCostModel,
):
    """
    General nonlinear impact.

    Cost

        coefficient × size^exponent
    """

    def __init__(
        self,
        *,
        coefficient: float = 0.01,
        exponent: float = 0.75,
    ) -> None:

        self.coefficient = float(
            coefficient
        )

        self.exponent = float(
            exponent
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(
            trade
        )

        if trade.side == TradeSide.HOLD:

            return 0.0

        return (

            self.coefficient

            * (

                trade_notional(
                    trade
                )

                ** self.exponent

            )

        )


# ==============================================================
# COMPOSITE IMPACT MODEL
# ==============================================================


class CompositeMarketImpactModel(
    BaseCostModel,
):
    """
    Combine several
    market impact models.
    """

    def __init__(
        self,
        models: list[
            BaseCostModel
        ],
    ) -> None:

        self.models = models

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        total = 0.0

        for model in self.models:

            total += model.estimate(
                trade
            )

        return total


# ==============================================================
# FACTORY
# ==============================================================


def build_market_impact_model(
    model_type: CostModelType,
    **kwargs,
) -> BaseCostModel:
    """
    Factory.
    """

    if model_type == CostModelType.FIXED:

        return LinearMarketImpactModel(
            **kwargs
        )

    if model_type == CostModelType.SQUARE_ROOT:

        return SquareRootImpactModel(
            **kwargs
        )

    if model_type == CostModelType.ADV:

        return ADVParticipationImpactModel(
            **kwargs
        )

    raise ValueError(

        f"Unsupported market impact model: {model_type}"

    )


# ==============================================================
# UTILITIES
# ==============================================================


def estimate_market_impact(
    trade: Trade,
    model: BaseCostModel,
) -> float:
    """
    Convenience wrapper.
    """

    return model.estimate(
        trade
    )


# ==============================================================
# EXPORTS
# ==============================================================


__all__.extend(

    [

        "LinearMarketImpactModel",

        "SquareRootImpactModel",

        "ADVParticipationImpactModel",

        "NonLinearImpactModel",

        "CompositeMarketImpactModel",

        "build_market_impact_model",

        "estimate_market_impact",

    ]

)


# Part 6

# ==============================================================
# TAX MODELS
# ==============================================================


class PercentageTaxModel(
    BaseCostModel,
):
    """
    Tax expressed as a percentage
    of trade notional.

    Examples
    --------
    Stamp duty
    Securities transaction tax
    """

    def __init__(
        self,
        rate: float,
    ) -> None:

        if rate < 0:

            raise ValueError(
                "Tax rate must be non-negative."
            )

        self.rate = float(rate)

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(trade)

        if trade.side == TradeSide.HOLD:

            return 0.0

        return (

            trade_notional(trade)

            * self.rate

        )


# ==============================================================
# BUY-ONLY TAX
# ==============================================================


class BuyOnlyTaxModel(
    BaseCostModel,
):
    """
    Tax applied only to BUY trades.

    Example
    -------
    Stamp duty in some markets.
    """

    def __init__(
        self,
        rate: float,
    ) -> None:

        self.rate = float(rate)

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(trade)

        if trade.side != TradeSide.BUY:

            return 0.0

        return (

            trade_notional(trade)

            * self.rate

        )


# ==============================================================
# SELL-ONLY TAX
# ==============================================================


class SellOnlyTaxModel(
    BaseCostModel,
):
    """
    Tax applied only to SELL trades.
    """

    def __init__(
        self,
        rate: float,
    ) -> None:

        self.rate = float(rate)

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(trade)

        if trade.side != TradeSide.SELL:

            return 0.0

        return (

            trade_notional(trade)

            * self.rate

        )


# ==============================================================
# BORROW COST
# ==============================================================


class BorrowCostModel(
    BaseCostModel,
):
    """
    Short borrow fee.

    Applied only when
    increasing short exposure.

    Current implementation assumes

        SELL = borrow

    Future versions can inspect
    long/short exposure directly.
    """

    def __init__(
        self,
        annual_rate: float,
        holding_days: int = 1,
    ) -> None:

        self.annual_rate = float(
            annual_rate
        )

        self.holding_days = int(
            holding_days
        )

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(trade)

        if trade.side != TradeSide.SELL:

            return 0.0

        daily_rate = (

            self.annual_rate

            / 252.0

        )

        return (

            trade_notional(trade)

            * daily_rate

            * self.holding_days

        )


# ==============================================================
# EXCHANGE FEES
# ==============================================================


class ExchangeFeeModel(
    BaseCostModel,
):
    """
    Exchange / clearing fee.

    Usually very small.
    """

    def __init__(
        self,
        rate: float,
    ) -> None:

        self.rate = float(rate)

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(trade)

        if trade.side == TradeSide.HOLD:

            return 0.0

        return (

            trade_notional(trade)

            * self.rate

        )


# ==============================================================
# FIXED EXCHANGE FEE
# ==============================================================


class FixedExchangeFeeModel(
    BaseCostModel,
):
    """
    Flat fee per trade.
    """

    def __init__(
        self,
        fee: float,
    ) -> None:

        self.fee = float(fee)

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        validate_trade(trade)

        if trade.side == TradeSide.HOLD:

            return 0.0

        return self.fee


# ==============================================================
# COMPOSITE FEES
# ==============================================================


class CompositeFeeModel(
    BaseCostModel,
):
    """
    Combine taxes,
    borrow,
    exchange fees.
    """

    def __init__(
        self,
        models: list[
            BaseCostModel
        ],
    ) -> None:

        self.models = models

    # ----------------------------------------------------------

    def estimate(
        self,
        trade: Trade,
    ) -> float:

        total = 0.0

        for model in self.models:

            total += model.estimate(
                trade
            )

        return total


# ==============================================================
# FACTORY
# ==============================================================


def build_fee_model(
    component: CostComponent,
    **kwargs,
) -> BaseCostModel:
    """
    Generic fee factory.
    """

    if component == CostComponent.TAX:

        return PercentageTaxModel(
            **kwargs
        )

    if component == CostComponent.BORROW:

        return BorrowCostModel(
            **kwargs
        )

    if component == CostComponent.EXCHANGE:

        return ExchangeFeeModel(
            **kwargs
        )

    raise ValueError(

        f"Unsupported fee component: {component}"

    )


# ==============================================================
# UTILITIES
# ==============================================================


def estimate_fee(
    trade: Trade,
    model: BaseCostModel,
) -> float:
    """
    Convenience wrapper.
    """

    return model.estimate(
        trade
    )


# ==============================================================
# EXPORTS
# ==============================================================


__all__.extend(

    [

        "PercentageTaxModel",

        "BuyOnlyTaxModel",

        "SellOnlyTaxModel",

        "BorrowCostModel",

        "ExchangeFeeModel",

        "FixedExchangeFeeModel",

        "CompositeFeeModel",

        "build_fee_model",

        "estimate_fee",

    ]

)


# Part 7

# ==============================================================
# TRANSACTION COST ENGINE
# ==============================================================


class TransactionCostEngine:
    """
    Production transaction cost engine.

    The engine combines all individual
    transaction cost models into one
    unified estimate.
    """

    def __init__(
        self,
        *,
        commission_model: BaseCostModel | None = None,
        spread_model: BaseCostModel | None = None,
        slippage_model: BaseCostModel | None = None,
        market_impact_model: BaseCostModel | None = None,
        tax_model: BaseCostModel | None = None,
        borrow_model: BaseCostModel | None = None,
        exchange_model: BaseCostModel | None = None,
    ) -> None:

        self.commission_model = commission_model

        self.spread_model = spread_model

        self.slippage_model = slippage_model

        self.market_impact_model = market_impact_model

        self.tax_model = tax_model

        self.borrow_model = borrow_model

        self.exchange_model = exchange_model

    # ----------------------------------------------------------
    # SINGLE TRADE
    # ----------------------------------------------------------

    def estimate_trade(
        self,
        trade: Trade,
    ) -> TransactionCostResult:
        """
        Estimate total cost
        for one trade.
        """

        validate_trade(
            trade
        )

        breakdown = CostBreakdown()

        if self.commission_model is not None:

            breakdown.commission = (

                self.commission_model.estimate(
                    trade
                )

            )

        if self.spread_model is not None:

            breakdown.spread = (

                self.spread_model.estimate(
                    trade
                )

            )

        if self.slippage_model is not None:

            breakdown.slippage = (

                self.slippage_model.estimate(
                    trade
                )

            )

        if self.market_impact_model is not None:

            breakdown.market_impact = (

                self.market_impact_model.estimate(
                    trade
                )

            )

        if self.tax_model is not None:

            breakdown.tax = (

                self.tax_model.estimate(
                    trade
                )

            )

        if self.borrow_model is not None:

            breakdown.borrow = (

                self.borrow_model.estimate(
                    trade
                )

            )

        if self.exchange_model is not None:

            breakdown.exchange = (

                self.exchange_model.estimate(
                    trade
                )

            )

        return TransactionCostResult(

            trade=trade,

            breakdown=breakdown,

            metadata={

                "total_cost":

                    breakdown.total_cost,

            },

        )

    # ----------------------------------------------------------
    # MULTIPLE TRADES
    # ----------------------------------------------------------

    def estimate_trades(
        self,
        trades: list[Trade],
    ) -> list[TransactionCostResult]:
        """
        Estimate costs for
        multiple trades.
        """

        return [

            self.estimate_trade(
                trade
            )

            for trade in trades

        ]

    # ----------------------------------------------------------
    # AGGREGATE COST
    # ----------------------------------------------------------

    def total_cost(
        self,
        trades: list[Trade],
    ) -> float:
        """
        Total transaction cost.
        """

        return sum(

            result.breakdown.total_cost

            for result in self.estimate_trades(
                trades
            )

        )

    # ----------------------------------------------------------
    # COST DATAFRAME
    # ----------------------------------------------------------

    def to_dataframe(
        self,
        trades: list[Trade],
    ) -> pd.DataFrame:
        """
        Convert estimates to DataFrame.
        """

        results = self.estimate_trades(
            trades
        )

        rows = []

        for result in results:

            b = result.breakdown

            rows.append(

                {

                    "Ticker":
                        result.trade.ticker,

                    "Side":
                        result.trade.side.value,

                    "Trade_Weight":
                        result.trade.trade_weight,

                    "Commission":
                        b.commission,

                    "Spread":
                        b.spread,

                    "Slippage":
                        b.slippage,

                    "Market_Impact":
                        b.market_impact,

                    "Tax":
                        b.tax,

                    "Borrow":
                        b.borrow,

                    "Exchange":
                        b.exchange,

                    "Total_Cost":
                        b.total_cost,

                }

            )

        return pd.DataFrame(rows)

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------

    def summary(
        self,
        trades: list[Trade],
    ) -> dict:
        """
        Aggregate transaction cost summary.
        """

        df = self.to_dataframe(
            trades
        )

        if df.empty:

            return {}

        return {

            "number_of_trades":

                len(df),

            "commission":

                float(
                    df["Commission"].sum()
                ),

            "spread":

                float(
                    df["Spread"].sum()
                ),

            "slippage":

                float(
                    df["Slippage"].sum()
                ),

            "market_impact":

                float(
                    df["Market_Impact"].sum()
                ),

            "tax":

                float(
                    df["Tax"].sum()
                ),

            "borrow":

                float(
                    df["Borrow"].sum()
                ),

            "exchange":

                float(
                    df["Exchange"].sum()
                ),

            "total_cost":

                float(
                    df["Total_Cost"].sum()
                ),

        }


# ==============================================================
# DEFAULT ENGINE
# ==============================================================


def build_default_transaction_cost_engine(
) -> TransactionCostEngine:
    """
    Build a default engine
    with conservative assumptions.
    """

    return TransactionCostEngine(

        commission_model=PercentageCommissionModel(
            rate=0.001,
        ),

        spread_model=HalfSpreadModel(
            spread=0.0005,
        ),

        slippage_model=FixedSlippageModel(
            slippage=0.0005,
        ),

        market_impact_model=SquareRootImpactModel(),

        tax_model=PercentageTaxModel(
            rate=0.0001,
        ),

        exchange_model=ExchangeFeeModel(
            rate=0.00005,
        ),

    )


# ==============================================================
# EXPORTS
# ==============================================================


__all__.extend(

    [

        "TransactionCostEngine",

        "build_default_transaction_cost_engine",

    ]

)


# Part 8

# ==============================================================
# DIAGNOSTICS & REPORTING
# ==============================================================


class TransactionCostDiagnostics:
    """
    Diagnostics for transaction costs.

    Pure reporting layer.

    No cost estimation logic belongs here.
    """

    # ----------------------------------------------------------

    @staticmethod
    def component_breakdown(
        results: list[
            TransactionCostResult
        ],
    ) -> pd.DataFrame:
        """
        Cost breakdown by trade.
        """

        rows = []

        for result in results:

            b = result.breakdown

            rows.append(

                {

                    "Ticker":
                        result.trade.ticker,

                    "Side":
                        result.trade.side.value,

                    "Commission":
                        b.commission,

                    "Spread":
                        b.spread,

                    "Slippage":
                        b.slippage,

                    "Market_Impact":
                        b.market_impact,

                    "Tax":
                        b.tax,

                    "Borrow":
                        b.borrow,

                    "Exchange":
                        b.exchange,

                    "Total":
                        b.total_cost,

                }

            )

        return pd.DataFrame(rows)

    # ----------------------------------------------------------

    @staticmethod
    def aggregate(
        results: list[
            TransactionCostResult
        ],
    ) -> dict:
        """
        Aggregate statistics.
        """

        if len(results) == 0:

            return {}

        df = TransactionCostDiagnostics.component_breakdown(
            results
        )

        totals = df.sum(
            numeric_only=True
        )

        total_cost = float(
            totals["Total"]
        )

        return {

            "number_of_trades":

                len(df),

            "total_cost":

                total_cost,

            "average_cost":

                total_cost / len(df),

            "commission":

                float(
                    totals["Commission"]
                ),

            "spread":

                float(
                    totals["Spread"]
                ),

            "slippage":

                float(
                    totals["Slippage"]
                ),

            "market_impact":

                float(
                    totals["Market_Impact"]
                ),

            "tax":

                float(
                    totals["Tax"]
                ),

            "borrow":

                float(
                    totals["Borrow"]
                ),

            "exchange":

                float(
                    totals["Exchange"]
                ),

        }

    # ----------------------------------------------------------

    @staticmethod
    def largest_costs(
        results: list[
            TransactionCostResult
        ],
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Largest transaction costs.
        """

        df = TransactionCostDiagnostics.component_breakdown(
            results
        )

        if df.empty:

            return df

        return (

            df

            .sort_values(

                "Total",

                ascending=False,

            )

            .head(top_n)

            .reset_index(
                drop=True
            )

        )

    # ----------------------------------------------------------

    @staticmethod
    def cost_percentages(
        results: list[
            TransactionCostResult
        ],
    ) -> dict:
        """
        Percentage contribution
        of each cost component.
        """

        summary = TransactionCostDiagnostics.aggregate(
            results
        )

        if not summary:

            return {}

        total = summary["total_cost"]

        if total <= 0:

            return {}

        components = {}

        for key in (

            "commission",

            "spread",

            "slippage",

            "market_impact",

            "tax",

            "borrow",

            "exchange",

        ):

            components[key] = (

                summary[key]

                / total

            )

        return components

    # ----------------------------------------------------------

    @staticmethod
    def print_report(
        results: list[
            TransactionCostResult
        ],
    ) -> None:
        """
        Console report.
        """

        summary = TransactionCostDiagnostics.aggregate(
            results
        )

        if not summary:

            LOGGER.info(
                "No transaction costs."
            )

            return

        LOGGER.info("")

        LOGGER.info(
            "=" * 60
        )

        LOGGER.info(
            "TRANSACTION COST REPORT"
        )

        LOGGER.info(
            "=" * 60
        )

        LOGGER.info(
            "Trades          : %d",
            summary["number_of_trades"],
        )

        LOGGER.info(
            "Total Cost      : %.6f",
            summary["total_cost"],
        )

        LOGGER.info(
            "Average Cost    : %.6f",
            summary["average_cost"],
        )

        LOGGER.info("")

        LOGGER.info(
            "Commission      : %.6f",
            summary["commission"],
        )

        LOGGER.info(
            "Spread          : %.6f",
            summary["spread"],
        )

        LOGGER.info(
            "Slippage        : %.6f",
            summary["slippage"],
        )

        LOGGER.info(
            "Market Impact   : %.6f",
            summary["market_impact"],
        )

        LOGGER.info(
            "Tax             : %.6f",
            summary["tax"],
        )

        LOGGER.info(
            "Borrow          : %.6f",
            summary["borrow"],
        )

        LOGGER.info(
            "Exchange        : %.6f",
            summary["exchange"],
        )

        LOGGER.info(
            "=" * 60
        )


# ==============================================================
# CONVENIENCE
# ==============================================================


def generate_transaction_cost_report(
    engine: TransactionCostEngine,
    trades: list[
        Trade
    ],
) -> dict:
    """
    Generate diagnostics directly
    from an engine.
    """

    results = engine.estimate_trades(
        trades
    )

    TransactionCostDiagnostics.print_report(
        results
    )

    return TransactionCostDiagnostics.aggregate(
        results
    )


# ==============================================================
# EXPORTS
# ==============================================================


__all__.extend(

    [

        "TransactionCostDiagnostics",

        "generate_transaction_cost_report",

    ]

)


# ==============================================================
# SELF TEST
# ==============================================================


if __name__ == "__main__":

    LOGGER.info(

        "Transaction Cost Engine"

    )

