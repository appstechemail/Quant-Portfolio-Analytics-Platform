"""
==============================================================
EXECUTION ENGINE
Part 1: Framework & Core Objects
==============================================================

Institutional-grade execution framework.

Responsibilities
----------------
• Order representation
• Execution requests
• Execution results
• Broker abstraction
• Cost/slippage interfaces
• Market impact interfaces
• Execution reporting

Future Parts
------------
Part 2  Order Models
Part 3  Cost Models
Part 4  Slippage Models
Part 5  Liquidity Models
Part 6  Market Impact
Part 7  Simulator
Part 8  OMS/Broker Integration
Part 9  Analytics
Part 10 Reporting
Part 11 Convenience APIs
==============================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# GLOBAL CONSTANTS
# ============================================================

EPSILON = 1e-12


# ============================================================
# ENUMS
# ============================================================

class OrderSide(str, Enum):

    BUY = "BUY"

    SELL = "SELL"


class OrderType(str, Enum):

    MARKET = "MARKET"

    LIMIT = "LIMIT"

    VWAP = "VWAP"

    TWAP = "TWAP"

    ARRIVAL_PRICE = "ARRIVAL_PRICE"

    IMPLEMENTATION_SHORTFALL = (
        "IMPLEMENTATION_SHORTFALL"
    )


class OrderStatus(str, Enum):

    CREATED = "CREATED"

    SUBMITTED = "SUBMITTED"

    PARTIALLY_FILLED = (
        "PARTIALLY_FILLED"
    )

    FILLED = "FILLED"

    CANCELLED = "CANCELLED"

    REJECTED = "REJECTED"


class ExecutionVenue(str, Enum):

    SIMULATED = "SIMULATED"

    EXCHANGE = "EXCHANGE"

    DARK_POOL = "DARK_POOL"

    BROKER = "BROKER"


# ============================================================
# EXECUTION CONFIG
# ============================================================

@dataclass(slots=True)
class ExecutionConfig:
    """
    Global execution settings.
    """

    default_order_type:OrderType = OrderType.MARKET

    max_participation_rate:float = 0.10

    use_market_impact:bool = True

    use_slippage:bool = True

    use_transaction_costs:bool = True

    random_seed:int = 42


# ============================================================
# ORDER OBJECT
# ============================================================

@dataclass(slots=True)
class Order:
    """
    Institutional order object.
    """

    ticker: str

    side: OrderSide

    quantity: float

    order_type: OrderType

    creation_time: datetime

    target_weight:float | None = None

    limit_price:float | None = None

    notes:dict[str, Any] = field(
            default_factory=dict
        )

    @property
    def signed_quantity(
        self,
    ) -> float:

        if self.side == OrderSide.BUY:

            return float(
                self.quantity
            )

        return float(
            -self.quantity
        )


# ============================================================
# EXECUTION REQUEST
# ============================================================

@dataclass(slots=True)
class ExecutionRequest:
    """
    Batch of orders submitted
    for execution.
    """

    orders: list[Order]

    request_time: datetime

    portfolio_id:str | None = None

    rebalance_id:str | None = None


# ============================================================
# FILL OBJECT
# ============================================================

@dataclass(slots=True)
class Fill:
    """
    Individual execution fill.
    """

    ticker: str

    side: OrderSide

    quantity: float

    fill_price: float

    execution_time:datetime

    venue: ExecutionVenue

    commission:float = 0.0

    fees:float = 0.0

    slippage:float = 0.0

    market_impact:float = 0.0

    @property
    def total_cost(
        self,
    ) -> float:

        return float(

            self.commission

            + self.fees

        )


# ============================================================
# EXECUTION RESULT
# ============================================================

@dataclass(slots=True)
class ExecutionResult:
    """
    Result of executing a batch.
    """

    fills: list[Fill]

    status: OrderStatus

    success: bool

    message: str

    execution_time_seconds: float

    @property
    def total_commission(
        self,
    ) -> float:

        return float(

            sum(
                f.commission
                for f in self.fills
            )

        )

    @property
    def total_fees(
        self,
    ) -> float:

        return float(

            sum(
                f.fees
                for f in self.fills
            )

        )

    @property
    def total_slippage(
        self,
    ) -> float:

        return float(

            sum(
                f.slippage
                for f in self.fills
            )

        )

    @property
    def total_market_impact(
        self,
    ) -> float:

        return float(

            sum(
                f.market_impact
                for f in self.fills
            )

        )


# ============================================================
# EXECUTION REPORT
# ============================================================

@dataclass(slots=True)
class ExecutionReport:
    """
    Institutional reporting object.
    """

    order_count: int

    fill_count: int

    total_notional: float

    total_commission: float

    total_fees: float

    total_slippage: float

    total_market_impact: float

    average_fill_price: float

    success_rate: float


# ============================================================
# COST MODEL INTERFACE
# ============================================================

class BaseCostModel(ABC):
    """
    Transaction cost model.
    """

    @abstractmethod
    def estimate(
        self,
        order: Order,
        fill_price: float,
    ) -> float:

        raise NotImplementedError


# ============================================================
# SLIPPAGE MODEL INTERFACE
# ============================================================

class BaseSlippageModel(ABC):
    """
    Slippage model.
    """

    @abstractmethod
    def estimate(
        self,
        order: Order,
        market_price: float,
    ) -> float:

        raise NotImplementedError


# ============================================================
# MARKET IMPACT MODEL
# ============================================================

class BaseMarketImpactModel(
    ABC,
):
    """
    Market impact model.
    """

    @abstractmethod
    def estimate(
        self,
        order: Order,
        adv: float,
    ) -> float:

        raise NotImplementedError


# ============================================================
# EXECUTION ENGINE
# ============================================================

class BaseExecutionEngine(
    ABC,
):
    """
    Abstract execution engine.
    """

    def __init__(
        self,
        config:
        ExecutionConfig,
    ) -> None:

        self.config = config

    @abstractmethod
    def execute(
        self,
        request:
        ExecutionRequest,
    ) -> ExecutionResult:

        raise NotImplementedError


# ============================================================
# BROKER INTERFACE
# ============================================================

class BaseBroker(
    ABC,
):
    """
    Broker / OMS abstraction.
    """

    @abstractmethod
    def submit_order(
        self,
        order: Order,
    ) -> str:

        raise NotImplementedError

    @abstractmethod
    def cancel_order(
        self,
        order_id: str,
    ) -> bool:

        raise NotImplementedError

    @abstractmethod
    def get_order_status(
        self,
        order_id: str,
    ) -> OrderStatus:

        raise NotImplementedError


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_order(
    order: Order,
) -> None:

    if not order.ticker:

        raise ValueError(
            "Ticker missing."
        )

    if (
        order.quantity
        <= 0
    ):
        raise ValueError(
            "Quantity must be positive."
        )

    if (
        order.order_type
        == OrderType.LIMIT
        and order.limit_price is None
    ):
        raise ValueError(
            "Limit order requires limit_price."
        )


def validate_request(
    request:
    ExecutionRequest,
) -> None:

    if (
        not request.orders
    ):
        raise ValueError(
            "ExecutionRequest contains no orders."
        )

    for order in request.orders:

        validate_order(
            order
        )


# ============================================================
# REPORT BUILDER
# ============================================================

def build_execution_report(
    result:
    ExecutionResult,
) -> ExecutionReport:

    fills = result.fills

    if len(fills) == 0:

        return ExecutionReport(
            order_count=0,
            fill_count=0,
            total_notional=0.0,
            total_commission=0.0,
            total_fees=0.0,
            total_slippage=0.0,
            total_market_impact=0.0,
            average_fill_price=0.0,
            success_rate=0.0,
        )

    notionals = [

        f.quantity
        * f.fill_price

        for f in fills

    ]

    return ExecutionReport(

        order_count=
        len(fills),

        fill_count=
        len(fills),

        total_notional=
        float(
            np.sum(
                notionals
            )
        ),

        total_commission=
        result.total_commission,

        total_fees=
        result.total_fees,

        total_slippage=
        result.total_slippage,

        total_market_impact=
        result.total_market_impact,

        average_fill_price=
        float(
            np.mean(
                [
                    f.fill_price
                    for f in fills
                ]
            )
        ),

        success_rate=
        1.0
        if result.success
        else 0.0,
    )


# ============================================================
# PART 2: EXECUTION INSTRUCTIONS
# ============================================================

@dataclass(slots=True)
class ExecutionInstruction:
    """
    Institutional execution instructions.

    Controls how an order
    should be executed.
    """

    urgency: str = "NORMAL"

    participation_rate:float = 0.10

    start_time:datetime | None = None

    end_time:datetime | None = None

    allow_partial_fills:bool = True

    venue_preference: list[ExecutionVenue] = field(
            default_factory=list
        )

    notes: dict[str, Any] = field(
            default_factory=dict
        )


# ============================================================
# MARKET ORDER
# ============================================================

@dataclass(slots=True)
class MarketOrder(Order):
    """
    Immediate execution.
    """

    def __post_init__(
        self,
    ) -> None:

        self.order_type = (
            OrderType.MARKET
        )


# ============================================================
# LIMIT ORDER
# ============================================================

@dataclass(slots=True)
class LimitOrder(Order):
    """
    Price-constrained order.
    """

    def __post_init__(
        self,
    ) -> None:

        self.order_type = (
            OrderType.LIMIT
        )

        if (
            self.limit_price
            is None
        ):
            raise ValueError(
                "LimitOrder requires "
                "limit_price."
            )


# ============================================================
# VWAP ORDER
# ============================================================

@dataclass(slots=True)
class VWAPOrder(Order):
    """
    Execute close to VWAP.
    """

    benchmark_volume:pd.Series | None = None

    def __post_init__(
        self,
    ) -> None:

        self.order_type = (
            OrderType.VWAP
        )


# ============================================================
# TWAP ORDER
# ============================================================

@dataclass(slots=True)
class TWAPOrder(Order):
    """
    Execute evenly through time.
    """

    slices: int = 10

    def __post_init__(
        self,
    ) -> None:

        self.order_type = (
            OrderType.TWAP
        )

        self.slices = max(
            int(self.slices),
            1,
        )


# ============================================================
# ARRIVAL PRICE ORDER
# ============================================================

@dataclass(slots=True)
class ArrivalPriceOrder(Order):
    """
    Minimize deviation from
    arrival price.
    """

    arrival_price:float | None = None

    def __post_init__(
        self,
    ) -> None:

        self.order_type = (
            OrderType.ARRIVAL_PRICE
        )


# ============================================================
# IMPLEMENTATION SHORTFALL
# ============================================================

@dataclass(slots=True)
class ImplementationShortfallOrder(
    Order,
):
    """
    Optimize urgency vs cost.
    """

    risk_aversion:float = 1.0

    def __post_init__(
        self,
    ) -> None:

        self.order_type = (
            OrderType
            .IMPLEMENTATION_SHORTFALL
        )


# ============================================================
# BASKET ORDER
# ============================================================

@dataclass(slots=True)
class BasketOrder:
    """
    Portfolio-level order.
    """

    orders: list[Order]

    creation_time: datetime

    rebalance_id:str | None = None

    @property
    def total_orders(
        self,
    ) -> int:

        return len(
            self.orders
        )

    @property
    def tickers(
        self,
    ) -> list[str]:

        return [
            o.ticker
            for o in self.orders
        ]


# ============================================================
# CHILD ORDER
# ============================================================

@dataclass(slots=True)
class ChildOrder(Order):
    """
    Slice of parent order.
    """

    parent_order_id:str | None = None

    slice_number:int = 0

    total_slices:int = 1


# ============================================================
# ORDER SCHEDULER
# ============================================================

class OrderScheduler:
    """
    Generate execution schedule.
    """

    @staticmethod
    def twap_schedule(
        *,
        start_time: datetime,
        end_time: datetime,
        slices: int,
    ) -> list[datetime]:
        """
        Uniform schedule.
        """

        slices = max(
            slices,
            1,
        )

        if slices == 1:

            return [
                start_time
            ]

        total_seconds = (
            end_time
            - start_time
        ).total_seconds()

        step = (
            total_seconds
            / (slices - 1)
        )

        return [

            start_time
            + pd.Timedelta(
                seconds=i * step
            )

            for i in range(
                slices
            )

        ]

    # --------------------------------------------------------

    @staticmethod
    def vwap_schedule(
        *,
        volume_profile:
        pd.Series,
    ) -> pd.Series:
        """
        Volume-weighted schedule.

        Returns
        -------
        Fraction of order
        executed per bucket.
        """

        volume_profile = (
            volume_profile
            .astype(float)
            .clip(lower=0)
        )

        total = (
            volume_profile.sum()
        )

        if total <= EPSILON:

            raise ValueError(
                "Invalid volume profile."
            )

        return (
            volume_profile
            / total
        )


# ============================================================
# ORDER SLICER
# ============================================================

class OrderSlicer:
    """
    Convert parent order
    into child orders.
    """

    @staticmethod
    def slice_twap_order(
        order: TWAPOrder,
    ) -> list[ChildOrder]:
        """
        Equal-size slices.
        """

        quantity_per_slice = (
            order.quantity
            / order.slices
        )

        children = []

        for i in range(
            order.slices
        ):

            children.append(

                ChildOrder(

                    ticker=
                    order.ticker,

                    side=
                    order.side,

                    quantity=
                    quantity_per_slice,

                    order_type=
                    order.order_type,

                    creation_time=
                    order.creation_time,

                    parent_order_id=
                    str(id(order)),

                    slice_number=
                    i + 1,

                    total_slices=
                    order.slices,
                )

            )

        return children

    # --------------------------------------------------------

    @staticmethod
    def slice_vwap_order(
        order: VWAPOrder,
        volume_profile:
        pd.Series,
    ) -> list[ChildOrder]:
        """
        Volume-weighted slices.
        """

        weights = (
            OrderScheduler
            .vwap_schedule(
                volume_profile=
                volume_profile
            )
        )

        children = []

        total_slices = len(
            weights
        )

        for i, weight in enumerate(
            weights.values,
            start=1,
        ):

            children.append(

                ChildOrder(

                    ticker=
                    order.ticker,

                    side=
                    order.side,

                    quantity=
                    float(
                        order.quantity
                        * weight
                    ),

                    order_type=
                    order.order_type,

                    creation_time=
                    order.creation_time,

                    parent_order_id=
                    str(id(order)),

                    slice_number=i,

                    total_slices=
                    total_slices,
                )

            )

        return children


# ============================================================
# EXECUTION PRIORITY
# ============================================================

class ExecutionPriority:
    """
    Institutional urgency rules.
    """

    LOW = 1

    NORMAL = 2

    HIGH = 3

    CRITICAL = 4


# ============================================================
# ORDER UTILITIES
# ============================================================

def estimate_order_notional(
    order: Order,
    price: float,
) -> float:
    """
    Dollar value.
    """

    return float(
        order.quantity
        * price
    )


def basket_notional(
    basket: BasketOrder,
    prices:
    dict[str, float],
) -> float:
    """
    Basket notional.
    """

    total = 0.0

    for order in basket.orders:

        if (
            order.ticker
            not in prices
        ):
            continue

        total += (
            estimate_order_notional(
                order,
                prices[
                    order.ticker
                ],
            )
        )

    return float(total)

# ============================================================
# PART 3: EXECUTION COST MODELS
# ============================================================


# ============================================================
# COST ESTIMATE
# ============================================================

@dataclass(slots=True)
class CostEstimate:
    """
    Institutional transaction-cost estimate.
    """

    commission: float

    exchange_fee: float

    regulatory_fee: float

    broker_fee: float

    taxes: float

    total_cost: float


# ============================================================
# COST CONFIGURATION
# ============================================================

@dataclass(slots=True)
class CostModelConfig:
    """
    Global cost assumptions.
    """

    commission_bps: float = 2.0

    exchange_fee_bps: float = 0.10

    regulatory_fee_bps: float = 0.02

    broker_fee_bps: float = 0.50

    tax_bps: float = 0.0

    minimum_commission: float = 0.0


# ============================================================
# BASE COST MODEL
# ============================================================

class ExecutionCostModel(
    BaseCostModel,
):
    """
    Base institutional cost model.
    """

    def __init__(
        self,
        config:
        CostModelConfig,
    ) -> None:

        self.config = config

    # --------------------------------------------------------

    @staticmethod
    def bps_to_decimal(
        bps: float,
    ) -> float:

        return (
            float(bps)
            / 10000.0
        )

    # --------------------------------------------------------

    @staticmethod
    def notional(
        quantity: float,
        price: float,
    ) -> float:

        return float(
            abs(quantity)
            * price
        )

    # --------------------------------------------------------

    def estimate(
        self,
        order: Order,
        fill_price: float,
    ) -> float:

        estimate = (
            self.estimate_costs(
                order,
                fill_price,
            )
        )

        return (
            estimate.total_cost
        )

    # --------------------------------------------------------

    def estimate_costs(
        self,
        order: Order,
        fill_price: float,
    ) -> CostEstimate:

        notional = (
            self.notional(
                order.quantity,
                fill_price,
            )
        )

        commission = max(

            notional
            * self.bps_to_decimal(
                self.config
                .commission_bps
            ),

            self.config
            .minimum_commission,
        )

        exchange_fee = (

            notional
            * self.bps_to_decimal(
                self.config
                .exchange_fee_bps
            )

        )

        regulatory_fee = (

            notional
            * self.bps_to_decimal(
                self.config
                .regulatory_fee_bps
            )

        )

        broker_fee = (

            notional
            * self.bps_to_decimal(
                self.config
                .broker_fee_bps
            )

        )

        taxes = (

            notional
            * self.bps_to_decimal(
                self.config
                .tax_bps
            )

        )

        total = (

            commission

            + exchange_fee

            + regulatory_fee

            + broker_fee

            + taxes

        )

        return CostEstimate(

            commission=
            float(
                commission
            ),

            exchange_fee=
            float(
                exchange_fee
            ),

            regulatory_fee=
            float(
                regulatory_fee
            ),

            broker_fee=
            float(
                broker_fee
            ),

            taxes=
            float(
                taxes
            ),

            total_cost=
            float(
                total
            ),
        )


# ============================================================
# FIXED COMMISSION MODEL
# ============================================================

class FixedCommissionModel(
    ExecutionCostModel,
):
    """
    Fixed commission per trade.
    """

    def __init__(
        self,
        commission_per_trade:
        float,
    ) -> None:

        self.fixed_commission = (
            float(
                commission_per_trade
            )
        )

    # --------------------------------------------------------

    def estimate(
        self,
        order: Order,
        fill_price: float,
    ) -> float:

        return float(
            self.fixed_commission
        )


# ============================================================
# PERCENTAGE COMMISSION MODEL
# ============================================================

class PercentageCommissionModel(
    ExecutionCostModel,
):
    """
    Commission based on trade notional.
    """

    def __init__(
        self,
        commission_bps:
        float,
    ) -> None:

        self.commission_bps = (
            float(
                commission_bps
            )
        )

    # --------------------------------------------------------

    def estimate(
        self,
        order: Order,
        fill_price: float,
    ) -> float:

        notional = (
            self.notional(
                order.quantity,
                fill_price,
            )
        )

        return float(

            notional

            * self.bps_to_decimal(
                self.commission_bps
            )

        )


# ============================================================
# TIERED COMMISSION MODEL
# ============================================================

class TieredCommissionModel(
    ExecutionCostModel,
):
    """
    Institutional tier schedule.

    Example
    -------
    [
        (1e6, 2.0),
        (10e6, 1.0),
        (np.inf, 0.5)
    ]
    """

    def __init__(
        self,
        tiers:
        list[
            tuple[
                float,
                float,
            ]
        ],
    ) -> None:

        self.tiers = tiers

    # --------------------------------------------------------

    def estimate(
        self,
        order: Order,
        fill_price: float,
    ) -> float:

        notional = (
            self.notional(
                order.quantity,
                fill_price,
            )
        )

        for limit, bps in self.tiers:

            if notional <= limit:

                return float(

                    notional

                    * self.bps_to_decimal(
                        bps
                    )

                )

        return 0.0


# ============================================================
# BROKER FEE MODEL
# ============================================================

class BrokerFeeModel:
    """
    Separate broker charge model.
    """

    def __init__(
        self,
        broker_bps:
        float,
    ) -> None:

        self.broker_bps = (
            float(
                broker_bps
            )
        )

    # --------------------------------------------------------

    def estimate(
        self,
        notional:
        float,
    ) -> float:

        return float(

            notional

            * (
                self.broker_bps
                / 10000.0
            )

        )


# ============================================================
# EXCHANGE FEE MODEL
# ============================================================

class ExchangeFeeModel:
    """
    Exchange fee schedule.
    """

    def __init__(
        self,
        exchange_bps:
        float,
    ) -> None:

        self.exchange_bps = (
            float(
                exchange_bps
            )
        )

    # --------------------------------------------------------

    def estimate(
        self,
        notional:
        float,
    ) -> float:

        return float(

            notional

            * (
                self.exchange_bps
                / 10000.0
            )

        )


# ============================================================
# TAX MODEL
# ============================================================

class TaxModel:
    """
    Transaction tax model.
    """

    def __init__(
        self,
        tax_bps:
        float,
    ) -> None:

        self.tax_bps = (
            float(
                tax_bps
            )
        )

    # --------------------------------------------------------

    def estimate(
        self,
        notional:
        float,
    ) -> float:

        return float(

            notional

            * (
                self.tax_bps
                / 10000.0
            )

        )


# ============================================================
# INSTITUTIONAL COST ENGINE
# ============================================================

class InstitutionalCostEngine:
    """
    Aggregates all execution costs.
    """

    def __init__(
        self,
        cost_model:
        ExecutionCostModel,
    ) -> None:

        self.cost_model = (
            cost_model
        )

    # --------------------------------------------------------

    def estimate_fill(
        self,
        order: Order,
        fill_price:
        float,
    ) -> CostEstimate:

        return (
            self.cost_model
            .estimate_costs(
                order,
                fill_price,
            )
        )

    # --------------------------------------------------------

    def estimate_batch(
        self,
        orders:
        list[Order],
        prices:
        dict[str, float],
    ) -> CostEstimate:

        commission = 0.0

        exchange_fee = 0.0

        regulatory_fee = 0.0

        broker_fee = 0.0

        taxes = 0.0

        for order in orders:

            if (
                order.ticker
                not in prices
            ):
                continue

            estimate = (
                self.cost_model
                .estimate_costs(
                    order,
                    prices[
                        order.ticker
                    ],
                )
            )

            commission += (
                estimate
                .commission
            )

            exchange_fee += (
                estimate
                .exchange_fee
            )

            regulatory_fee += (
                estimate
                .regulatory_fee
            )

            broker_fee += (
                estimate
                .broker_fee
            )

            taxes += (
                estimate
                .taxes
            )

        total = (

            commission

            + exchange_fee

            + regulatory_fee

            + broker_fee

            + taxes

        )

        return CostEstimate(

            commission=
            float(
                commission
            ),

            exchange_fee=
            float(
                exchange_fee
            ),

            regulatory_fee=
            float(
                regulatory_fee
            ),

            broker_fee=
            float(
                broker_fee
            ),

            taxes=
            float(
                taxes
            ),

            total_cost=
            float(
                total
            ),
        )


# ============================================================
# REPORTING HELPERS
# ============================================================

def cost_summary(
    estimate:
    CostEstimate,
) -> pd.Series:

    return pd.Series({

        "Commission":
            estimate
            .commission,

        "Exchange_Fee":
            estimate
            .exchange_fee,

        "Regulatory_Fee":
            estimate
            .regulatory_fee,

        "Broker_Fee":
            estimate
            .broker_fee,

        "Taxes":
            estimate
            .taxes,

        "Total_Cost":
            estimate
            .total_cost,
    })


# ============================================================
# PART 4: SLIPPAGE MODELS
# ============================================================

from dataclasses import dataclass


# ============================================================
# SLIPPAGE ESTIMATE
# ============================================================

@dataclass(slots=True)
class SlippageEstimate:
    """
    Institutional slippage estimate.
    """

    slippage_bps: float

    slippage_price: float

    slippage_cost: float


# ============================================================
# SLIPPAGE CONFIG
# ============================================================

@dataclass(slots=True)
class SlippageConfig:
    """
    Global slippage settings.
    """

    base_bps: float = 2.0

    spread_multiplier: float = 0.50

    volatility_multiplier: float = 1.00

    participation_multiplier: float = 5.00

    minimum_bps: float = 0.0

    maximum_bps: float = 200.0


# ============================================================
# BASE SLIPPAGE MODEL
# ============================================================

class ExecutionSlippageModel(
    BaseSlippageModel,
):
    """
    Base slippage model.
    """

    def __init__(
        self,
        config:
        SlippageConfig,
    ) -> None:

        self.config = config

    # --------------------------------------------------------

    @staticmethod
    def bps_to_decimal(
        bps: float,
    ) -> float:

        return (
            float(bps)
            / 10000.0
        )

    # --------------------------------------------------------

    @staticmethod
    def trade_notional(
        quantity: float,
        price: float,
    ) -> float:

        return float(
            abs(quantity)
            * price
        )

    # --------------------------------------------------------

    @staticmethod
    def apply_side(
        side: OrderSide,
        market_price: float,
        slippage_bps: float,
    ) -> float:
        """
        Convert slippage bps
        into execution price.
        """

        shift = (

            market_price

            * (
                slippage_bps
                / 10000.0
            )

        )

        if side == OrderSide.BUY:

            return float(
                market_price
                + shift
            )

        return float(
            market_price
            - shift
        )

    # --------------------------------------------------------

    def estimate(
        self,
        order: Order,
        market_price: float,
    ) -> float:

        return (
            self.estimate_details(
                order,
                market_price,
            )
            .slippage_cost
        )

    # --------------------------------------------------------

    def estimate_details(
        self,
        order: Order,
        market_price: float,
    ) -> SlippageEstimate:

        bps = (
            self.config
            .base_bps
        )

        execution_price = (
            self.apply_side(
                order.side,
                market_price,
                bps,
            )
        )

        cost = abs(

            execution_price
            - market_price

        ) * abs(
            order.quantity
        )

        return SlippageEstimate(

            slippage_bps=
            float(bps),

            slippage_price=
            execution_price,

            slippage_cost=
            float(cost),
        )


# ============================================================
# FIXED SLIPPAGE
# ============================================================

class FixedSlippageModel(
    ExecutionSlippageModel,
):
    """
    Constant slippage.
    """

    def __init__(
        self,
        bps: float,
    ) -> None:

        self.fixed_bps = (
            float(bps)
        )

    # --------------------------------------------------------

    def estimate_details(
        self,
        order: Order,
        market_price: float,
    ) -> SlippageEstimate:

        execution_price = (
            self.apply_side(
                order.side,
                market_price,
                self.fixed_bps,
            )
        )

        cost = abs(

            execution_price
            - market_price

        ) * abs(
            order.quantity
        )

        return SlippageEstimate(

            slippage_bps=
            self.fixed_bps,

            slippage_price=
            execution_price,

            slippage_cost=
            float(cost),
        )


# ============================================================
# SPREAD MODEL
# ============================================================

class SpreadSlippageModel(
    ExecutionSlippageModel,
):
    """
    Spread-based slippage.

    Input
    -----
    spread_bps
    """

    def estimate_details(
        self,
        order: Order,
        market_price: float,
        *,
        spread_bps: float,
    ) -> SlippageEstimate:

        bps = (

            spread_bps

            * self.config
            .spread_multiplier

        )

        execution_price = (
            self.apply_side(
                order.side,
                market_price,
                bps,
            )
        )

        cost = abs(

            execution_price
            - market_price

        ) * abs(
            order.quantity
        )

        return SlippageEstimate(

            slippage_bps=
            float(bps),

            slippage_price=
            execution_price,

            slippage_cost=
            float(cost),
        )


# ============================================================
# VOLATILITY MODEL
# ============================================================

class VolatilitySlippageModel(
    ExecutionSlippageModel,
):
    """
    Volatility-driven slippage.

    Higher volatility
    ⇒ larger slippage.
    """

    def estimate_details(
        self,
        order: Order,
        market_price: float,
        *,
        volatility: float,
    ) -> SlippageEstimate:

        bps = (

            self.config.base_bps

            + (

                volatility
                * 10000

                * self.config
                .volatility_multiplier

            )

        )

        bps = np.clip(

            bps,

            self.config
            .minimum_bps,

            self.config
            .maximum_bps,
        )

        execution_price = (
            self.apply_side(
                order.side,
                market_price,
                bps,
            )
        )

        cost = abs(

            execution_price
            - market_price

        ) * abs(
            order.quantity
        )

        return SlippageEstimate(

            slippage_bps=
            float(bps),

            slippage_price=
            execution_price,

            slippage_cost=
            float(cost),
        )


# ============================================================
# PARTICIPATION MODEL
# ============================================================

class ParticipationSlippageModel(
    ExecutionSlippageModel,
):
    """
    Slippage increases
    with ADV participation.
    """

    def estimate_details(
        self,
        order: Order,
        market_price: float,
        *,
        participation_rate:
        float,
    ) -> SlippageEstimate:

        bps = (

            self.config.base_bps

            + (

                participation_rate
                * 100

                * self.config
                .participation_multiplier

            )

        )

        execution_price = (
            self.apply_side(
                order.side,
                market_price,
                bps,
            )
        )

        cost = abs(

            execution_price
            - market_price

        ) * abs(
            order.quantity
        )

        return SlippageEstimate(

            slippage_bps=
            float(bps),

            slippage_price=
            execution_price,

            slippage_cost=
            float(cost),
        )


# ============================================================
# BUY / SELL ASYMMETRIC MODEL
# ============================================================

class AsymmetricSlippageModel(
    ExecutionSlippageModel,
):
    """
    Different slippage
    for buys and sells.
    """

    def __init__(
        self,
        buy_bps: float,
        sell_bps: float,
    ) -> None:

        self.buy_bps = (
            float(buy_bps)
        )

        self.sell_bps = (
            float(sell_bps)
        )

    # --------------------------------------------------------

    def estimate_details(
        self,
        order: Order,
        market_price: float,
    ) -> SlippageEstimate:

        if (
            order.side
            == OrderSide.BUY
        ):

            bps = self.buy_bps

        else:

            bps = self.sell_bps

        execution_price = (
            self.apply_side(
                order.side,
                market_price,
                bps,
            )
        )

        cost = abs(

            execution_price
            - market_price

        ) * abs(
            order.quantity
        )

        return SlippageEstimate(

            slippage_bps=
            float(bps),

            slippage_price=
            execution_price,

            slippage_cost=
            float(cost),
        )


# ============================================================
# COMPOSITE MODEL
# ============================================================

class CompositeSlippageModel(
    ExecutionSlippageModel,
):
    """
    Combine spread,
    volatility,
    participation.
    """

    def estimate_details(
        self,
        order: Order,
        market_price: float,
        *,
        spread_bps: float,
        volatility: float,
        participation_rate:
        float,
    ) -> SlippageEstimate:

        spread_component = (

            spread_bps

            * self.config
            .spread_multiplier

        )

        volatility_component = (

            volatility
            * 10000

            * self.config
            .volatility_multiplier

        )

        participation_component = (

            participation_rate
            * 100

            * self.config
            .participation_multiplier

        )

        total_bps = (

            self.config
            .base_bps

            + spread_component

            + volatility_component

            + participation_component

        )

        total_bps = np.clip(

            total_bps,

            self.config
            .minimum_bps,

            self.config
            .maximum_bps,
        )

        execution_price = (
            self.apply_side(
                order.side,
                market_price,
                total_bps,
            )
        )

        cost = abs(

            execution_price
            - market_price

        ) * abs(
            order.quantity
        )

        return SlippageEstimate(

            slippage_bps=
            float(total_bps),

            slippage_price=
            execution_price,

            slippage_cost=
            float(cost),
        )


# ============================================================
# BATCH ENGINE
# ============================================================

class InstitutionalSlippageEngine:
    """
    Portfolio-level
    slippage estimation.
    """

    def __init__(
        self,
        model:
        ExecutionSlippageModel,
    ) -> None:

        self.model = model

    # --------------------------------------------------------

    def estimate_order(
        self,
        order: Order,
        market_price:
        float,
    ) -> SlippageEstimate:

        return (
            self.model
            .estimate_details(
                order,
                market_price,
            )
        )

    # --------------------------------------------------------

    def estimate_batch(
        self,
        orders:
        list[Order],
        prices:
        dict[str, float],
    ) -> float:

        total = 0.0

        for order in orders:

            if (
                order.ticker
                not in prices
            ):
                continue

            estimate = (
                self.model
                .estimate_details(
                    order,
                    prices[
                        order.ticker
                    ],
                )
            )

            total += (
                estimate
                .slippage_cost
            )

        return float(total)


# ============================================================
# REPORTING
# ============================================================

def slippage_summary(
    estimate:
    SlippageEstimate,
) -> pd.Series:

    return pd.Series({

        "Slippage_Bps":
            estimate
            .slippage_bps,

        "Execution_Price":
            estimate
            .slippage_price,

        "Slippage_Cost":
            estimate
            .slippage_cost,
    })

# ============================================================
# PART 5: LIQUIDITY & PARTICIPATION MODELS
# ============================================================

from dataclasses import dataclass


# ============================================================
# LIQUIDITY ESTIMATE
# ============================================================

@dataclass(slots=True)
class LiquidityEstimate:
    """
    Liquidity diagnostics for a trade.
    """

    adv: float

    dollar_volume: float

    participation_rate: float

    liquidity_score: float

    trade_capacity: float

    feasible: bool


# ============================================================
# LIQUIDITY CONFIG
# ============================================================

@dataclass(slots=True)
class LiquidityConfig:
    """
    Institutional liquidity settings.
    """

    max_participation_rate: float = 0.10

    warning_participation_rate: float = 0.05

    min_adv_dollars: float = 1_000_000.0

    liquidity_score_floor: float = 0.25

    capacity_multiplier: float = 1.0


# ============================================================
# ADV MODEL
# ============================================================

class ADVModel:
    """
    Average Daily Volume model.
    """

    @staticmethod
    def estimate_adv(
        volume: pd.Series,
        price: pd.Series | None = None,
    ) -> float:
        """
        Estimate ADV.

        If price is supplied:
            dollar ADV

        otherwise:
            share ADV
        """

        if volume.empty:

            return 0.0

        if price is None:

            return float(
                volume.mean()
            )

        return float(
            (
                volume * price
            ).mean()
        )


# ============================================================
# PARTICIPATION MODEL
# ============================================================

class ParticipationModel:
    """
    Institutional participation model.
    """

    @staticmethod
    def participation_rate(
        *,
        trade_notional: float,
        adv_dollars: float,
    ) -> float:

        if adv_dollars <= EPSILON:

            return np.inf

        return float(
            trade_notional
            / adv_dollars
        )

    # --------------------------------------------------------

    @staticmethod
    def participation_flag(
        *,
        participation_rate: float,
        threshold: float,
    ) -> bool:

        return (
            participation_rate
            <= threshold
        )


# ============================================================
# TRADE CAPACITY MODEL
# ============================================================

class CapacityModel:
    """
    Capacity estimation.

    Determines maximum
    deployable capital.
    """

    def __init__(
        self,
        config:
        LiquidityConfig,
    ) -> None:

        self.config = config

    # --------------------------------------------------------

    def trade_capacity(
        self,
        adv_dollars: float,
    ) -> float:

        return float(

            adv_dollars

            * self.config
            .max_participation_rate

            * self.config
            .capacity_multiplier

        )


# ============================================================
# LIQUIDITY SCORE
# ============================================================

class LiquidityScoreModel:
    """
    Convert liquidity metrics
    into normalized score.
    """

    @staticmethod
    def score(
        *,
        adv_dollars: float,
        participation_rate:
        float,
        config:
        LiquidityConfig,
    ) -> float:

        adv_component = min(

            adv_dollars
            /
            max(
                config
                .min_adv_dollars,
                EPSILON,
            ),

            1.0,
        )

        participation_component = (

            1.0

            - min(
                participation_rate
                /
                max(
                    config
                    .max_participation_rate,
                    EPSILON,
                ),
                1.0,
            )

        )

        score = (

            0.50
            * adv_component

            + 0.50
            * participation_component

        )

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )


# ============================================================
# LIQUIDITY FEASIBILITY
# ============================================================

class LiquidityFeasibility:
    """
    Determines whether
    trade can realistically
    be executed.
    """

    @staticmethod
    def feasible(
        *,
        liquidity_score:
        float,
        participation_rate:
        float,
        config:
        LiquidityConfig,
    ) -> bool:

        if (
            liquidity_score
            <
            config
            .liquidity_score_floor
        ):
            return False

        if (
            participation_rate
            >
            config
            .max_participation_rate
        ):
            return False

        return True


# ============================================================
# LIQUIDITY MODEL
# ============================================================

class LiquidityModel:
    """
    Full liquidity engine.
    """

    def __init__(
        self,
        config:
        LiquidityConfig,
    ) -> None:

        self.config = config

        self.capacity_model = (
            CapacityModel(
                config
            )
        )

    # --------------------------------------------------------

    def estimate(
        self,
        *,
        trade_notional:
        float,
        adv_dollars:
        float,
    ) -> LiquidityEstimate:

        participation = (

            ParticipationModel
            .participation_rate(

                trade_notional=
                trade_notional,

                adv_dollars=
                adv_dollars,
            )

        )

        score = (

            LiquidityScoreModel
            .score(

                adv_dollars=
                adv_dollars,

                participation_rate=
                participation,

                config=
                self.config,
            )

        )

        capacity = (

            self.capacity_model
            .trade_capacity(
                adv_dollars
            )

        )

        feasible = (

            LiquidityFeasibility
            .feasible(

                liquidity_score=
                score,

                participation_rate=
                participation,

                config=
                self.config,
            )

        )

        return LiquidityEstimate(

            adv=
            float(
                adv_dollars
            ),

            dollar_volume=
            float(
                adv_dollars
            ),

            participation_rate=
            float(
                participation
            ),

            liquidity_score=
            float(
                score
            ),

            trade_capacity=
            float(
                capacity
            ),

            feasible=
            bool(
                feasible
            ),
        )


# ============================================================
# BATCH LIQUIDITY ENGINE
# ============================================================

class InstitutionalLiquidityEngine:
    """
    Portfolio-level
    liquidity diagnostics.
    """

    def __init__(
        self,
        config:
        LiquidityConfig,
    ) -> None:

        self.model = (
            LiquidityModel(
                config
            )
        )

    # --------------------------------------------------------

    def estimate_trade(
        self,
        *,
        trade_notional:
        float,
        adv_dollars:
        float,
    ) -> LiquidityEstimate:

        return (
            self.model
            .estimate(

                trade_notional=
                trade_notional,

                adv_dollars=
                adv_dollars,
            )
        )

    # --------------------------------------------------------

    def estimate_portfolio(
        self,
        trades:
        pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Required columns

            Trade_Notional
            ADV_Dollars
        """

        out = trades.copy()

        scores = []

        participations = []

        capacities = []

        feasible_flags = []

        for row in out.itertuples():

            estimate = (

                self.model
                .estimate(

                    trade_notional=
                    float(
                        row
                        .Trade_Notional
                    ),

                    adv_dollars=
                    float(
                        row
                        .ADV_Dollars
                    ),
                )

            )

            scores.append(
                estimate
                .liquidity_score
            )

            participations.append(
                estimate
                .participation_rate
            )

            capacities.append(
                estimate
                .trade_capacity
            )

            feasible_flags.append(
                estimate
                .feasible
            )

        out[
            "Liquidity_Score"
        ] = scores

        out[
            "Participation_Rate"
        ] = participations

        out[
            "Trade_Capacity"
        ] = capacities

        out[
            "Liquidity_Feasible"
        ] = feasible_flags

        return out


# ============================================================
# REPORTING HELPERS
# ============================================================

def liquidity_summary(
    estimate:
    LiquidityEstimate,
) -> pd.Series:

    return pd.Series({

        "ADV":
            estimate.adv,

        "Participation_Rate":
            estimate
            .participation_rate,

        "Liquidity_Score":
            estimate
            .liquidity_score,

        "Trade_Capacity":
            estimate
            .trade_capacity,

        "Feasible":
            estimate
            .feasible,
    })


# ============================================================
# PART 6: MARKET IMPACT MODELS
# ============================================================

from dataclasses import dataclass


# ============================================================
# MARKET IMPACT ESTIMATE
# ============================================================

@dataclass(slots=True)
class MarketImpactEstimate:
    """
    Institutional market impact estimate.
    """

    temporary_impact_bps: float

    permanent_impact_bps: float

    total_impact_bps: float

    impact_cost: float

    impact_price: float


# ============================================================
# IMPACT CONFIG
# ============================================================

@dataclass(slots=True)
class MarketImpactConfig:
    """
    Global impact assumptions.
    """

    square_root_coefficient: float = 15.0

    temporary_coefficient: float = 10.0

    permanent_coefficient: float = 5.0

    max_impact_bps: float = 500.0

    min_impact_bps: float = 0.0


# ============================================================
# BASE IMPACT MODEL
# ============================================================

class MarketImpactModel(
    BaseMarketImpactModel,
):
    """
    Base institutional impact model.
    """

    def __init__(
        self,
        config:
        MarketImpactConfig,
    ) -> None:

        self.config = config

    # --------------------------------------------------------

    @staticmethod
    def bps_to_decimal(
        bps: float,
    ) -> float:

        return float(
            bps / 10000.0
        )

    # --------------------------------------------------------

    @staticmethod
    def trade_notional(
        quantity: float,
        price: float,
    ) -> float:

        return float(
            abs(quantity)
            * price
        )

    # --------------------------------------------------------

    @staticmethod
    def participation_rate(
        *,
        trade_notional: float,
        adv_dollars: float,
    ) -> float:

        if adv_dollars <= EPSILON:

            return np.inf

        return float(
            trade_notional
            / adv_dollars
        )

    # --------------------------------------------------------

    @staticmethod
    def apply_side(
        side: OrderSide,
        market_price: float,
        impact_bps: float,
    ) -> float:

        shift = (
            market_price
            * impact_bps
            / 10000.0
        )

        if side == OrderSide.BUY:

            return float(
                market_price
                + shift
            )

        return float(
            market_price
            - shift
        )

    # --------------------------------------------------------

    def estimate(
        self,
        order: Order,
        adv: float,
    ) -> float:

        return float(
            self.config
            .temporary_coefficient
        )

    # --------------------------------------------------------

    def estimate_details(
        self,
        *,
        order: Order,
        market_price: float,
        adv_dollars: float,
    ) -> MarketImpactEstimate:

        participation = (
            self.participation_rate(
                trade_notional=
                self.trade_notional(
                    order.quantity,
                    market_price,
                ),
                adv_dollars=
                adv_dollars,
            )
        )

        temp = (
            self.config
            .temporary_coefficient
            * np.sqrt(
                max(
                    participation,
                    0.0,
                )
            )
        )

        perm = (
            self.config
            .permanent_coefficient
            * participation
        )

        total = temp + perm

        total = float(
            np.clip(
                total,
                self.config
                .min_impact_bps,
                self.config
                .max_impact_bps,
            )
        )

        impact_price = (
            self.apply_side(
                order.side,
                market_price,
                total,
            )
        )

        impact_cost = abs(

            impact_price
            - market_price

        ) * abs(
            order.quantity
        )

        return MarketImpactEstimate(

            temporary_impact_bps=
            float(temp),

            permanent_impact_bps=
            float(perm),

            total_impact_bps=
            float(total),

            impact_cost=
            float(impact_cost),

            impact_price=
            float(
                impact_price
            ),
        )


# ============================================================
# SQUARE ROOT IMPACT
# ============================================================

class SquareRootImpactModel(
    MarketImpactModel,
):
    """
    Industry-standard model.

    Impact ∝ sqrt(participation)
    """

    def estimate_details(
        self,
        *,
        order: Order,
        market_price: float,
        adv_dollars: float,
    ) -> MarketImpactEstimate:

        participation = (
            self.participation_rate(
                trade_notional=
                self.trade_notional(
                    order.quantity,
                    market_price,
                ),
                adv_dollars=
                adv_dollars,
            )
        )

        total = (

            self.config
            .square_root_coefficient

            * np.sqrt(
                max(
                    participation,
                    0.0,
                )
            )

        )

        total = float(
            np.clip(
                total,
                self.config
                .min_impact_bps,
                self.config
                .max_impact_bps,
            )
        )

        impact_price = (
            self.apply_side(
                order.side,
                market_price,
                total,
            )
        )

        impact_cost = abs(

            impact_price
            - market_price

        ) * abs(
            order.quantity
        )

        return MarketImpactEstimate(

            temporary_impact_bps=
            float(total),

            permanent_impact_bps=
            0.0,

            total_impact_bps=
            float(total),

            impact_cost=
            float(
                impact_cost
            ),

            impact_price=
            float(
                impact_price
            ),
        )


# ============================================================
# TEMPORARY IMPACT MODEL
# ============================================================

class TemporaryImpactModel(
    MarketImpactModel,
):
    """
    Only temporary impact.
    """

    def estimate_details(
        self,
        *,
        order: Order,
        market_price: float,
        adv_dollars: float,
    ) -> MarketImpactEstimate:

        participation = (
            self.participation_rate(
                trade_notional=
                self.trade_notional(
                    order.quantity,
                    market_price,
                ),
                adv_dollars=
                adv_dollars,
            )
        )

        temp = (

            self.config
            .temporary_coefficient

            * np.sqrt(
                participation
            )

        )

        impact_price = (
            self.apply_side(
                order.side,
                market_price,
                temp,
            )
        )

        impact_cost = abs(

            impact_price
            - market_price

        ) * abs(
            order.quantity
        )

        return MarketImpactEstimate(

            temporary_impact_bps=
            float(temp),

            permanent_impact_bps=
            0.0,

            total_impact_bps=
            float(temp),

            impact_cost=
            float(
                impact_cost
            ),

            impact_price=
            float(
                impact_price
            ),
        )


# ============================================================
# PERMANENT IMPACT MODEL
# ============================================================

class PermanentImpactModel(
    MarketImpactModel,
):
    """
    Permanent price impact.
    """

    def estimate_details(
        self,
        *,
        order: Order,
        market_price: float,
        adv_dollars: float,
    ) -> MarketImpactEstimate:

        participation = (
            self.participation_rate(
                trade_notional=
                self.trade_notional(
                    order.quantity,
                    market_price,
                ),
                adv_dollars=
                adv_dollars,
            )
        )

        perm = (

            self.config
            .permanent_coefficient

            * participation

        )

        impact_price = (
            self.apply_side(
                order.side,
                market_price,
                perm,
            )
        )

        impact_cost = abs(

            impact_price
            - market_price

        ) * abs(
            order.quantity
        )

        return MarketImpactEstimate(

            temporary_impact_bps=
            0.0,

            permanent_impact_bps=
            float(perm),

            total_impact_bps=
            float(perm),

            impact_cost=
            float(
                impact_cost
            ),

            impact_price=
            float(
                impact_price
            ),
        )


# ============================================================
# ALMGREN-CHRISS STYLE MODEL
# ============================================================

class AlmgrenChrissImpactModel(
    MarketImpactModel,
):
    """
    Simplified institutional model.

    Impact =
        Temporary + Permanent
    """

    pass


# ============================================================
# PORTFOLIO IMPACT ENGINE
# ============================================================

class InstitutionalImpactEngine:
    """
    Portfolio-level impact engine.
    """

    def __init__(
        self,
        model:
        MarketImpactModel,
    ) -> None:

        self.model = model

    # --------------------------------------------------------

    def estimate_trade(
        self,
        *,
        order: Order,
        market_price: float,
        adv_dollars: float,
    ) -> MarketImpactEstimate:

        return (
            self.model
            .estimate_details(
                order=order,
                market_price=
                market_price,
                adv_dollars=
                adv_dollars,
            )
        )

    # --------------------------------------------------------

    def estimate_batch(
        self,
        orders:
        list[Order],
        prices:
        dict[str, float],
        adv:
        dict[str, float],
    ) -> float:

        total_cost = 0.0

        for order in orders:

            if (
                order.ticker
                not in prices
            ):
                continue

            if (
                order.ticker
                not in adv
            ):
                continue

            estimate = (
                self.model
                .estimate_details(
                    order=order,
                    market_price=
                    prices[
                        order.ticker
                    ],
                    adv_dollars=
                    adv[
                        order.ticker
                    ],
                )
            )

            total_cost += (
                estimate
                .impact_cost
            )

        return float(
            total_cost
        )


# ============================================================
# REPORTING HELPERS
# ============================================================

def market_impact_summary(
    estimate:
    MarketImpactEstimate,
) -> pd.Series:

    return pd.Series({

        "Temporary_Impact_Bps":
            estimate
            .temporary_impact_bps,

        "Permanent_Impact_Bps":
            estimate
            .permanent_impact_bps,

        "Total_Impact_Bps":
            estimate
            .total_impact_bps,

        "Impact_Cost":
            estimate
            .impact_cost,

        "Impact_Price":
            estimate
            .impact_price,
    })

# ============================================================
# PART 7: EXECUTION SIMULATOR
# ============================================================

from dataclasses import dataclass, field


# ============================================================
# EXECUTION FILL
# ============================================================

@dataclass(slots=True)
class ExecutionFill:
    """
    Single executed order fill.
    """

    ticker: str

    side: str

    quantity: float

    requested_price: float

    executed_price: float

    execution_cost: float

    slippage_cost: float

    impact_cost: float

    total_cost: float

    participation_rate: float

    liquidity_score: float

    success: bool


# ============================================================
# EXECUTION REPORT
# ============================================================

@dataclass(slots=True)
class ExecutionReport:
    """
    Portfolio execution report.
    """

    fills: list[ExecutionFill] = field(
        default_factory=list
    )

    gross_notional: float = 0.0

    execution_cost: float = 0.0

    slippage_cost: float = 0.0

    impact_cost: float = 0.0

    total_cost: float = 0.0

    successful_orders: int = 0

    failed_orders: int = 0

    average_liquidity_score: float = 0.0

    average_participation_rate: float = 0.0


# ============================================================
# EXECUTION CONFIG
# ============================================================

@dataclass(slots=True)
class ExecutionSimulatorConfig:
    """
    Master execution settings.
    """

    reject_illiquid_orders: bool = True

    reject_high_participation_orders: bool = True

    max_participation_rate: float = 0.10

    min_liquidity_score: float = 0.25

    allow_partial_fills: bool = False

    round_lots: bool = False


# ============================================================
# EXECUTION SIMULATOR
# ============================================================

class ExecutionSimulator:
    """
    Institutional execution simulator.

    Integrates:

        Costs
        Slippage
        Liquidity
        Market Impact

    Produces realistic execution fills.
    """
    
    def __init__(
        self,
        *,
        cost_model:
        ExecutionCostModel,

        slippage_model:
        BaseSlippageModel,

        liquidity_engine:
        InstitutionalLiquidityEngine,

        impact_engine:
        InstitutionalImpactEngine,

        config:
        ExecutionSimulatorConfig,
    ) -> None:

        self.cost_model = (
            cost_model
        )

        self.slippage_model = (
            slippage_model
        )

        self.liquidity_engine = (
            liquidity_engine
        )

        self.impact_engine = (
            impact_engine
        )

        self.config = config

    # --------------------------------------------------------

    def _check_liquidity(
        self,
        liquidity:
        LiquidityEstimate,
    ) -> bool:

        if (
            self.config
            .reject_illiquid_orders
        ):

            if (
                liquidity
                .liquidity_score
                <
                self.config
                .min_liquidity_score
            ):
                return False

        if (
            self.config
            .reject_high_participation_orders
        ):

            if (
                liquidity
                .participation_rate
                >
                self.config
                .max_participation_rate
            ):
                return False

        return True

    # --------------------------------------------------------

    def simulate_order(
        self,
        *,
        order: Order,
        market_price: float,
        spread_bps: float,
        volatility: float,
        adv_dollars: float,
    ) -> ExecutionFill:

        # -----------------------------------
        # Liquidity
        # -----------------------------------

        liquidity = (

            self.liquidity_engine
            .estimate_trade(

                trade_notional=
                abs(
                    order.quantity
                )
                * market_price,

                adv_dollars=
                adv_dollars,
            )

        )

        success = (
            self._check_liquidity(
                liquidity
            )
        )

        # -----------------------------------
        # Costs
        # -----------------------------------

        cost_estimate = (

            self.cost_model
            .estimate(

                order=order,

                price=
                market_price,
            )

        )

        # -----------------------------------
        # Slippage
        # -----------------------------------

        slippage_estimate = (

            self.slippage_model
            .estimate_details(

                order=order,

                market_price=
                market_price,

                spread_bps=
                spread_bps,

                volatility=
                volatility,
            )

        )

        # -----------------------------------
        # Impact
        # -----------------------------------

        impact_estimate = (

            self.impact_engine
            .estimate_trade(

                order=order,

                market_price=
                market_price,

                adv_dollars=
                adv_dollars,
            )

        )

        # -----------------------------------
        # Final Execution Price
        # -----------------------------------

        executed_price = (
            impact_estimate
            .impact_price
        )

        # -----------------------------------
        # Total Cost
        # -----------------------------------

        total_cost = (

            cost_estimate
            .total_cost

            + slippage_estimate
            .slippage_cost

            + impact_estimate
            .impact_cost

        )

        return ExecutionFill(

            ticker=
            order.ticker,

            side=
            order.side.value,

            quantity=
            float(
                order.quantity
            ),

            requested_price=
            float(
                market_price
            ),

            executed_price=
            float(
                executed_price
            ),

            execution_cost=
            float(
                cost_estimate
                .total_cost
            ),

            slippage_cost=
            float(
                slippage_estimate
                .slippage_cost
            ),

            impact_cost=
            float(
                impact_estimate
                .impact_cost
            ),

            total_cost=
            float(
                total_cost
            ),

            participation_rate=
            float(
                liquidity
                .participation_rate
            ),

            liquidity_score=
            float(
                liquidity
                .liquidity_score
            ),

            success=
            bool(
                success
            ),
        )

    # --------------------------------------------------------

    def simulate_portfolio(
        self,
        *,
        orders:
        list[Order],

        market_prices:
        dict[str, float],

        spreads:
        dict[str, float],

        volatility:
        dict[str, float],

        adv:
        dict[str, float],
    ) -> ExecutionReport:

        report = (
            ExecutionReport()
        )

        liquidity_scores = []

        participation_rates = []

        for order in orders:

            if (
                order.ticker
                not in market_prices
            ):
                continue

            fill = (
                self.simulate_order(

                    order=order,

                    market_price=
                    market_prices[
                        order.ticker
                    ],

                    spread_bps=
                    spreads.get(
                        order.ticker,
                        10.0,
                    ),

                    volatility=
                    volatility.get(
                        order.ticker,
                        0.02,
                    ),

                    adv_dollars=
                    adv.get(
                        order.ticker,
                        1e8,
                    ),
                )
            )

            report.fills.append(
                fill
            )

            report.gross_notional += (

                abs(
                    fill.quantity
                )

                * fill.executed_price

            )

            report.execution_cost += (
                fill.execution_cost
            )

            report.slippage_cost += (
                fill.slippage_cost
            )

            report.impact_cost += (
                fill.impact_cost
            )

            report.total_cost += (
                fill.total_cost
            )

            liquidity_scores.append(
                fill.liquidity_score
            )

            participation_rates.append(
                fill.participation_rate
            )

            if fill.success:

                report.successful_orders += 1

            else:

                report.failed_orders += 1

        if liquidity_scores:

            report.average_liquidity_score = (
                float(
                    np.mean(
                        liquidity_scores
                    )
                )
            )

        if participation_rates:

            report.average_participation_rate = (
                float(
                    np.mean(
                        participation_rates
                    )
                )
            )

        return report


# ============================================================
# PORTFOLIO EXECUTION HELPERS
# ============================================================

def execution_report_to_frame(
    report:
    ExecutionReport,
) -> pd.DataFrame:
    """
    Convert fills into dataframe.
    """

    rows = []

    for fill in report.fills:

        rows.append({

            "Ticker":
                fill.ticker,

            "Side":
                fill.side,

            "Quantity":
                fill.quantity,

            "Requested_Price":
                fill.requested_price,

            "Executed_Price":
                fill.executed_price,

            "Execution_Cost":
                fill.execution_cost,

            "Slippage_Cost":
                fill.slippage_cost,

            "Impact_Cost":
                fill.impact_cost,

            "Total_Cost":
                fill.total_cost,

            "Participation_Rate":
                fill.participation_rate,

            "Liquidity_Score":
                fill.liquidity_score,

            "Success":
                fill.success,
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# SUMMARY REPORT
# ============================================================

def execution_summary(
    report:
    ExecutionReport,
) -> pd.Series:

    return pd.Series({

        "Gross_Notional":
            report
            .gross_notional,

        "Execution_Cost":
            report
            .execution_cost,

        "Slippage_Cost":
            report
            .slippage_cost,

        "Impact_Cost":
            report
            .impact_cost,

        "Total_Cost":
            report
            .total_cost,

        "Successful_Orders":
            report
            .successful_orders,

        "Failed_Orders":
            report
            .failed_orders,

        "Average_Liquidity":
            report
            .average_liquidity_score,

        "Average_Participation":
            report
            .average_participation_rate,
    })


# ============================================================
# BROKER / OMS INTEGRATION LAYER
# Part 8
# ============================================================

from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


# ============================================================
# EXECUTION STATUS
# ============================================================

class ExecutionStatus(
    str,
    Enum,
):
    """
    Institutional order lifecycle.
    """

    CREATED = "CREATED"

    ROUTED = "ROUTED"

    ACKNOWLEDGED = "ACKNOWLEDGED"

    PARTIALLY_FILLED = "PARTIALLY_FILLED"

    FILLED = "FILLED"

    REJECTED = "REJECTED"

    CANCELLED = "CANCELLED"

    EXPIRED = "EXPIRED"


# ============================================================
# VENUE TYPE
# ============================================================

class VenueType(
    str,
    Enum,
):
    """
    Execution destination.
    """

    EXCHANGE = "EXCHANGE"

    BROKER = "BROKER"

    DARK_POOL = "DARK_POOL"

    ECN = "ECN"

    INTERNALIZER = "INTERNALIZER"


# ============================================================
# EXECUTION VENUE
# ============================================================

@dataclass(slots=True)
class ExecutionVenue:
    """
    Trading venue definition.
    """

    venue_id: str

    venue_name: str

    venue_type: VenueType

    country: str

    currency: str

    supports_market_orders: bool = True

    supports_limit_orders: bool = True

    supports_stop_orders: bool = True

    average_latency_ms: float = 5.0


# ============================================================
# OMS ACKNOWLEDGEMENT
# ============================================================

@dataclass(slots=True)
class ExecutionAcknowledgement:
    """
    OMS/Broker response.
    """

    broker_order_id: str

    status: ExecutionStatus

    timestamp: datetime

    message: str = ""


# ============================================================
# EXECUTION UPDATE
# ============================================================

@dataclass(slots=True)
class ExecutionUpdate:
    """
    Real-time order update.
    """

    broker_order_id: str

    status: ExecutionStatus

    filled_quantity: float

    remaining_quantity: float

    average_fill_price: float

    timestamp: datetime


# ============================================================
# OMS ORDER RECORD
# ============================================================

@dataclass(slots=True)
class OMSOrder:
    """
    OMS representation of an order.
    """

    order: Order

    broker_order_id: str

    venue_id: str

    status: ExecutionStatus

    create_time: datetime

    last_update_time: datetime

    filled_quantity: float = 0.0

    average_fill_price: float = 0.0


# ============================================================
# BROKER ADAPTER
# ============================================================

class BrokerAdapter(
    ABC,
):
    """
    Generic broker interface.

    All brokers must implement.
    """

    @property
    @abstractmethod
    def broker_name(
        self,
    ) -> str:
        pass

    # --------------------------------------------------------

    @abstractmethod
    def submit_order(
        self,
        order: Order,
    ) -> ExecutionAcknowledgement:
        pass

    # --------------------------------------------------------

    @abstractmethod
    def cancel_order(
        self,
        broker_order_id: str,
    ) -> bool:
        pass

    # --------------------------------------------------------

    @abstractmethod
    def get_order_status(
        self,
        broker_order_id: str,
    ) -> ExecutionUpdate:
        pass

    # --------------------------------------------------------

    @abstractmethod
    def get_positions(
        self,
    ) -> pd.DataFrame:
        pass


# ============================================================
# PAPER BROKER
# ============================================================

class PaperBrokerAdapter(
    BrokerAdapter,
):
    """
    Institutional paper broker.

    Useful for:

        Research
        UAT
        Backtests
        CI testing
    """

    def __init__(
        self,
    ) -> None:

        self.orders: dict[
            str,
            OMSOrder
        ] = {}

        self.counter = 0

    @property
    def broker_name(
        self,
    ) -> str:

        return "PaperBroker"

    # --------------------------------------------------------

    def submit_order(
        self,
        order: Order,
    ) -> ExecutionAcknowledgement:

        self.counter += 1

        broker_order_id = (
            f"PB-{self.counter:08d}"
        )

        now = datetime.utcnow()

        record = OMSOrder(

            order=order,

            broker_order_id=
            broker_order_id,

            venue_id="PAPER",

            status=
            ExecutionStatus.ACKNOWLEDGED,

            create_time=
            now,

            last_update_time=
            now,
        )

        self.orders[
            broker_order_id
        ] = record

        return ExecutionAcknowledgement(

            broker_order_id=
            broker_order_id,

            status=
            ExecutionStatus.ACKNOWLEDGED,

            timestamp=
            now,

            message=
            "Paper order accepted",
        )

    # --------------------------------------------------------

    def cancel_order(
        self,
        broker_order_id: str,
    ) -> bool:

        if (
            broker_order_id
            not in self.orders
        ):
            return False

        self.orders[
            broker_order_id
        ].status = (
            ExecutionStatus.CANCELLED
        )

        return True

    # --------------------------------------------------------

    def get_order_status(
        self,
        broker_order_id: str,
    ) -> ExecutionUpdate:

        order = self.orders[
            broker_order_id
        ]

        return ExecutionUpdate(

            broker_order_id=
            broker_order_id,

            status=
            order.status,

            filled_quantity=
            order.filled_quantity,

            remaining_quantity=
            max(
                0.0,
                order.order.quantity
                -
                order.filled_quantity,
            ),

            average_fill_price=
            order.average_fill_price,

            timestamp=
            datetime.utcnow(),
        )

    # --------------------------------------------------------

    def get_positions(
        self,
    ) -> pd.DataFrame:

        rows = []

        for order in self.orders.values():

            rows.append({

                "Ticker":
                    order.order.ticker,

                "Quantity":
                    order.filled_quantity,

                "AveragePrice":
                    order.average_fill_price,
            })

        return pd.DataFrame(
            rows
        )


# ============================================================
# IBKR SCAFFOLD
# ============================================================

class IBKRAdapter(
    BrokerAdapter,
):
    """
    Interactive Brokers scaffold.

    Production implementation
    can connect through:

        ib_insync
        TWS API
        Gateway API
    """

    @property
    def broker_name(
        self,
    ) -> str:

        return "IBKR"

    def submit_order(
        self,
        order: Order,
    ):
        raise NotImplementedError

    def cancel_order(
        self,
        broker_order_id: str,
    ):
        raise NotImplementedError

    def get_order_status(
        self,
        broker_order_id: str,
    ):
        raise NotImplementedError

    def get_positions(
        self,
    ):
        raise NotImplementedError


# ============================================================
# FIX ADAPTER SCAFFOLD
# ============================================================

class FIXAdapter(
    BrokerAdapter,
):
    """
    FIX protocol adapter scaffold.

    Future:

        FIX 4.2
        FIX 4.4
        FIXT 1.1
    """

    @property
    def broker_name(
        self,
    ) -> str:

        return "FIX"

    def submit_order(
        self,
        order: Order,
    ):
        raise NotImplementedError

    def cancel_order(
        self,
        broker_order_id: str,
    ):
        raise NotImplementedError

    def get_order_status(
        self,
        broker_order_id: str,
    ):
        raise NotImplementedError

    def get_positions(
        self,
    ):
        raise NotImplementedError


# ============================================================
# ORDER ROUTER
# ============================================================

class OrderRouter:
    """
    Simple execution venue router.
    """

    def __init__(
        self,
        venues:
        list[ExecutionVenue],
    ) -> None:

        self.venues = venues

    # --------------------------------------------------------

    def select_venue(
        self,
        order: Order,
    ) -> ExecutionVenue:

        if not self.venues:

            raise ValueError(
                "No execution venues configured."
            )

        return self.venues[0]


# ============================================================
# SMART ORDER ROUTER
# ============================================================

class SmartOrderRouter(
    OrderRouter,
):
    """
    Institutional SOR.

    Future:

        Liquidity-aware
        Fee-aware
        Dark pool aware
        Market impact aware
    """

    def select_venue(
        self,
        order: Order,
    ) -> ExecutionVenue:

        ranked = sorted(

            self.venues,

            key=lambda v:
            v.average_latency_ms,
        )

        return ranked[0]


# ============================================================
# OMS
# ============================================================

class OrderManagementSystem:
    """
    Institutional OMS layer.
    """

    def __init__(
        self,
        broker:
        BrokerAdapter,
    ) -> None:

        self.broker = broker

    # --------------------------------------------------------

    def submit(
        self,
        order: Order,
    ) -> ExecutionAcknowledgement:

        return (
            self.broker
            .submit_order(
                order
            )
        )

    # --------------------------------------------------------

    def cancel(
        self,
        broker_order_id: str,
    ) -> bool:

        return (
            self.broker
            .cancel_order(
                broker_order_id
            )
        )

    # --------------------------------------------------------

    def status(
        self,
        broker_order_id: str,
    ) -> ExecutionUpdate:

        return (
            self.broker
            .get_order_status(
                broker_order_id
            )
        )

    # --------------------------------------------------------

    def positions(
        self,
    ) -> pd.DataFrame:

        return (
            self.broker
            .get_positions()
        )
    
# ============================================================
# EXECUTION ANALYTICS
# Part 9
# ============================================================

from dataclasses import dataclass
import numpy as np
import pandas as pd


# ============================================================
# IMPLEMENTATION SHORTFALL RESULT
# ============================================================

@dataclass(slots=True)
class ImplementationShortfallResult:
    """
    Implementation shortfall decomposition.
    """

    arrival_price: float

    execution_price: float

    quantity: float

    shortfall_per_share: float

    total_shortfall: float

    shortfall_bps: float


# ============================================================
# VWAP RESULT
# ============================================================

@dataclass(slots=True)
class VWAPBenchmarkResult:
    """
    VWAP benchmark comparison.
    """

    execution_price: float

    vwap_price: float

    slippage_bps: float

    outperformed_vwap: bool


# ============================================================
# TWAP RESULT
# ============================================================

@dataclass(slots=True)
class TWAPBenchmarkResult:
    """
    TWAP benchmark comparison.
    """

    execution_price: float

    twap_price: float

    slippage_bps: float

    outperformed_twap: bool


# ============================================================
# PARTICIPATION RESULT
# ============================================================

@dataclass(slots=True)
class ParticipationResult:
    """
    Participation analytics.
    """

    participation_rate: float

    average_adv_participation: float

    max_adv_participation: float


# ============================================================
# COST ATTRIBUTION RESULT
# ============================================================

@dataclass(slots=True)
class CostAttributionResult:
    """
    Cost decomposition.
    """

    commissions: float

    slippage: float

    market_impact: float

    fees: float

    total_cost: float


# ============================================================
# FILL ANALYTICS
# ============================================================

@dataclass(slots=True)
class FillAnalyticsResult:
    """
    Fill quality statistics.
    """

    total_orders: int

    filled_orders: int

    partial_fills: int

    fill_rate: float

    average_fill_price: float


# ============================================================
# EXECUTION ANALYTICS ENGINE
# ============================================================

class ExecutionAnalytics:
    """
    Institutional execution analytics engine.

    Supports:

        Implementation Shortfall
        VWAP Comparison
        TWAP Comparison
        Participation Analysis
        Cost Attribution
        Fill Analysis
    """

    # --------------------------------------------------------
    # Implementation Shortfall
    # --------------------------------------------------------

    @staticmethod
    def implementation_shortfall(
        *,
        arrival_price: float,
        execution_price: float,
        quantity: float,
    ) -> ImplementationShortfallResult:

        shortfall = (
            execution_price
            -
            arrival_price
        )

        total_shortfall = (
            shortfall
            * quantity
        )

        shortfall_bps = (
            shortfall
            /
            arrival_price
        ) * 10000

        return (
            ImplementationShortfallResult(
                arrival_price=
                float(arrival_price),

                execution_price=
                float(execution_price),

                quantity=
                float(quantity),

                shortfall_per_share=
                float(shortfall),

                total_shortfall=
                float(total_shortfall),

                shortfall_bps=
                float(shortfall_bps),
            )
        )

    # --------------------------------------------------------
    # VWAP Benchmark
    # --------------------------------------------------------

    @staticmethod
    def compare_to_vwap(
        *,
        execution_price: float,
        market_data: pd.DataFrame,
    ) -> VWAPBenchmarkResult:

        required = {
            "Price",
            "Volume",
        }

        missing = (
            required
            - set(
                market_data.columns
            )
        )

        if missing:

            raise ValueError(
                f"Missing columns: {missing}"
            )

        vwap = (

            (
                market_data["Price"]
                *
                market_data["Volume"]
            ).sum()

            /

            market_data["Volume"]
            .sum()

        )

        slippage_bps = (

            (
                execution_price
                -
                vwap
            )

            /

            vwap

        ) * 10000

        return (
            VWAPBenchmarkResult(
                execution_price=
                float(execution_price),

                vwap_price=
                float(vwap),

                slippage_bps=
                float(slippage_bps),

                outperformed_vwap=
                execution_price
                <= vwap,
            )
        )

    # --------------------------------------------------------
    # TWAP Benchmark
    # --------------------------------------------------------

    @staticmethod
    def compare_to_twap(
        *,
        execution_price: float,
        market_data: pd.DataFrame,
    ) -> TWAPBenchmarkResult:

        if "Price" not in market_data:

            raise ValueError(
                "Price column missing."
            )

        twap = (
            market_data["Price"]
            .mean()
        )

        slippage_bps = (

            (
                execution_price
                -
                twap
            )

            /

            twap

        ) * 10000

        return (
            TWAPBenchmarkResult(
                execution_price=
                float(execution_price),

                twap_price=
                float(twap),

                slippage_bps=
                float(slippage_bps),

                outperformed_twap=
                execution_price
                <= twap,
            )
        )

    # --------------------------------------------------------
    # Participation Analysis
    # --------------------------------------------------------

    @staticmethod
    def participation_analysis(
        *,
        dollar_volume: float,
        adv: float,
    ) -> ParticipationResult:

        participation = (
            dollar_volume
            /
            max(
                adv,
                1e-9,
            )
        )

        return (
            ParticipationResult(
                participation_rate=
                float(participation),

                average_adv_participation=
                float(participation),

                max_adv_participation=
                float(participation),
            )
        )

    # --------------------------------------------------------
    # Cost Attribution
    # --------------------------------------------------------

    @staticmethod
    def cost_attribution(
        *,
        commissions: float,
        slippage: float,
        market_impact: float,
        fees: float,
    ) -> CostAttributionResult:

        total = (

            commissions
            +
            slippage
            +
            market_impact
            +
            fees

        )

        return (
            CostAttributionResult(
                commissions=
                float(commissions),

                slippage=
                float(slippage),

                market_impact=
                float(market_impact),

                fees=
                float(fees),

                total_cost=
                float(total),
            )
        )

    # --------------------------------------------------------
    # Fill Analysis
    # --------------------------------------------------------

    @staticmethod
    def fill_analysis(
        fills: pd.DataFrame,
    ) -> FillAnalyticsResult:

        if fills.empty:

            return (
                FillAnalyticsResult(
                    total_orders=0,
                    filled_orders=0,
                    partial_fills=0,
                    fill_rate=0.0,
                    average_fill_price=0.0,
                )
            )

        total_orders = len(fills)

        filled = (
            fills["Filled"]
            .sum()
        )

        partial = (
            fills["PartialFill"]
            .sum()
            if "PartialFill"
            in fills.columns
            else 0
        )

        avg_price = (
            fills["FillPrice"]
            .mean()
        )

        fill_rate = (
            filled
            /
            total_orders
        )

        return (
            FillAnalyticsResult(
                total_orders=
                int(total_orders),

                filled_orders=
                int(filled),

                partial_fills=
                int(partial),

                fill_rate=
                float(fill_rate),

                average_fill_price=
                float(avg_price),
            )
        )


# ============================================================
# EXECUTION ANALYTICS REPORT
# ============================================================

@dataclass(slots=True)
class ExecutionAnalyticsReport:
    """
    Consolidated analytics report.
    """

    implementation_shortfall: ImplementationShortfallResult | None

    vwap: VWAPBenchmarkResult | None

    twap: TWAPBenchmarkResult | None

    participation: ParticipationResult | None

    cost: CostAttributionResult | None

    fills: FillAnalyticsResult | None


# ============================================================
# REPORT BUILDER
# ============================================================

class ExecutionReportBuilder:
    """
    Aggregates all analytics
    into one report object.
    """

    @staticmethod
    def build(
        *,
        implementation_shortfall=None,
        vwap=None,
        twap=None,
        participation=None,
        cost=None,
        fills=None,
    ) -> ExecutionAnalyticsReport:

        return (
            ExecutionAnalyticsReport(
                implementation_shortfall=
                implementation_shortfall,

                vwap=vwap,

                twap=twap,

                participation=
                participation,

                cost=cost,

                fills=fills,
            )
        )
    
# ============================================================
# INSTITUTIONAL REPORTING
# Part 10
# ============================================================

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pandas as pd
from datetime import datetime


# ============================================================
# REPORT METADATA
# ============================================================

@dataclass(slots=True)
class ReportMetadata:
    """
    Report metadata.
    """

    report_name: str

    generated_at: datetime

    strategy_name: str

    portfolio_name: str | None = None

    benchmark_name: str | None = None

    version: str = "1.0"


# ============================================================
# TRADE BLOTTER REPORT
# ============================================================

@dataclass(slots=True)
class TradeBlotterReport:
    """
    Institutional trade blotter.
    """

    metadata: ReportMetadata

    trades: pd.DataFrame


# ============================================================
# EXECUTION SUMMARY REPORT
# ============================================================

@dataclass(slots=True)
class ExecutionSummaryReport:
    """
    High-level execution summary.
    """

    metadata: ReportMetadata

    total_orders: int

    filled_orders: int

    fill_rate: float

    total_notional: float

    average_fill_price: float

    total_cost: float


# ============================================================
# TCA REPORT
# ============================================================

@dataclass(slots=True)
class TransactionCostAnalysisReport:
    """
    Transaction Cost Analysis report.
    """

    metadata: ReportMetadata

    implementation_shortfall_bps: float

    vwap_slippage_bps: float

    twap_slippage_bps: float

    total_cost_bps: float

    commissions: float

    slippage: float

    market_impact: float


# ============================================================
# LIQUIDITY REPORT
# ============================================================

@dataclass(slots=True)
class LiquidityReport:
    """
    Liquidity utilization report.
    """

    metadata: ReportMetadata

    average_adv_participation: float

    max_adv_participation: float

    average_liquidity_score: float | None = None


# ============================================================
# BROKER REPORT
# ============================================================

@dataclass(slots=True)
class BrokerReport:
    """
    Broker execution report.
    """

    metadata: ReportMetadata

    broker_name: str

    total_orders: int

    total_fills: int

    total_notional: float

    average_execution_cost: float


# ============================================================
# REPORT WRITER
# ============================================================

class ReportWriter:
    """
    Institutional report export engine.

    Supports:

        CSV
        JSON

    Future:

        Excel
        PDF
        PowerBI
        Tableau
    """

    # --------------------------------------------------------

    @staticmethod
    def to_csv(
        df: pd.DataFrame,
        path: str | Path,
    ) -> None:

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            path,
            index=False,
        )

    # --------------------------------------------------------

    @staticmethod
    def to_json(
        report,
        path: str | Path,
    ) -> None:

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {}

        for key, value in asdict(report).items():

            if isinstance(
                value,
                pd.DataFrame,
            ):
                payload[key] = (
                    value.to_dict(
                        orient="records"
                    )
                )

            else:
                payload[key] = value

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                payload,
                f,
                indent=4,
                default=str,
            )


# ============================================================
# REPORT FACTORY
# ============================================================

class InstitutionalReportFactory:
    """
    Factory creating all execution reports.
    """

    # --------------------------------------------------------
    # Trade Blotter
    # --------------------------------------------------------

    @staticmethod
    def trade_blotter(
        *,
        trades: pd.DataFrame,
        strategy_name: str,
    ) -> TradeBlotterReport:

        meta = ReportMetadata(

            report_name=
            "Trade Blotter",

            generated_at=
            datetime.utcnow(),

            strategy_name=
            strategy_name,
        )

        return (
            TradeBlotterReport(
                metadata=meta,
                trades=trades,
            )
        )

    # --------------------------------------------------------
    # Execution Summary
    # --------------------------------------------------------

    @staticmethod
    def execution_summary(
        *,
        fills: pd.DataFrame,
        strategy_name: str,
    ) -> ExecutionSummaryReport:

        meta = ReportMetadata(

            report_name=
            "Execution Summary",

            generated_at=
            datetime.utcnow(),

            strategy_name=
            strategy_name,
        )

        total_orders = len(fills)

        filled_orders = (
            int(
                fills["Filled"]
                .sum()
            )
            if "Filled"
            in fills.columns
            else total_orders
        )

        fill_rate = (
            filled_orders
            /
            max(
                total_orders,
                1,
            )
        )

        total_notional = (

            fills["Notional"]
            .sum()

            if "Notional"
            in fills.columns

            else 0.0
        )

        avg_fill_price = (

            fills["FillPrice"]
            .mean()

            if "FillPrice"
            in fills.columns

            else 0.0
        )

        total_cost = (

            fills["ExecutionCost"]
            .sum()

            if "ExecutionCost"
            in fills.columns

            else 0.0
        )

        return (
            ExecutionSummaryReport(

                metadata=meta,

                total_orders=
                int(total_orders),

                filled_orders=
                int(filled_orders),

                fill_rate=
                float(fill_rate),

                total_notional=
                float(total_notional),

                average_fill_price=
                float(avg_fill_price),

                total_cost=
                float(total_cost),
            )
        )

    # --------------------------------------------------------
    # TCA Report
    # --------------------------------------------------------

    @staticmethod
    def tca_report(
        *,
        analytics_report,
        strategy_name: str,
    ) -> TransactionCostAnalysisReport:

        meta = ReportMetadata(

            report_name=
            "Transaction Cost Analysis",

            generated_at=
            datetime.utcnow(),

            strategy_name=
            strategy_name,
        )

        return (
            TransactionCostAnalysisReport(

                metadata=meta,

                implementation_shortfall_bps=
                float(
                    analytics_report
                    .implementation_shortfall
                    .shortfall_bps
                )
                if analytics_report
                .implementation_shortfall
                else 0.0,

                vwap_slippage_bps=
                float(
                    analytics_report
                    .vwap
                    .slippage_bps
                )
                if analytics_report.vwap
                else 0.0,

                twap_slippage_bps=
                float(
                    analytics_report
                    .twap
                    .slippage_bps
                )
                if analytics_report.twap
                else 0.0,

                total_cost_bps=
                float(
                    analytics_report
                    .cost
                    .total_cost
                )
                if analytics_report.cost
                else 0.0,

                commissions=
                float(
                    analytics_report
                    .cost
                    .commissions
                )
                if analytics_report.cost
                else 0.0,

                slippage=
                float(
                    analytics_report
                    .cost
                    .slippage
                )
                if analytics_report.cost
                else 0.0,

                market_impact=
                float(
                    analytics_report
                    .cost
                    .market_impact
                )
                if analytics_report.cost
                else 0.0,
            )
        )

    # --------------------------------------------------------
    # Liquidity Report
    # --------------------------------------------------------

    @staticmethod
    def liquidity_report(
        *,
        participation_result,
        strategy_name: str,
    ) -> LiquidityReport:

        meta = ReportMetadata(

            report_name=
            "Liquidity Report",

            generated_at=
            datetime.utcnow(),

            strategy_name=
            strategy_name,
        )

        return (
            LiquidityReport(

                metadata=meta,

                average_adv_participation=
                float(
                    participation_result
                    .average_adv_participation
                ),

                max_adv_participation=
                float(
                    participation_result
                    .max_adv_participation
                ),
            )
        )

    # --------------------------------------------------------
    # Broker Report
    # --------------------------------------------------------

    @staticmethod
    def broker_report(
        *,
        broker_name: str,
        fills: pd.DataFrame,
        strategy_name: str,
    ) -> BrokerReport:

        meta = ReportMetadata(

            report_name=
            "Broker Report",

            generated_at=
            datetime.utcnow(),

            strategy_name=
            strategy_name,
        )

        total_orders = len(fills)

        total_fills = (
            fills["Filled"]
            .sum()
            if "Filled"
            in fills.columns
            else total_orders
        )

        total_notional = (
            fills["Notional"]
            .sum()
            if "Notional"
            in fills.columns
            else 0.0
        )

        avg_cost = (
            fills["ExecutionCost"]
            .mean()
            if "ExecutionCost"
            in fills.columns
            else 0.0
        )

        return (
            BrokerReport(

                metadata=meta,

                broker_name=
                broker_name,

                total_orders=
                int(total_orders),

                total_fills=
                int(total_fills),

                total_notional=
                float(total_notional),

                average_execution_cost=
                float(avg_cost),
            )
        )

# ============================================================
# FACTORY & CONVENIENCE APIs
# Part 11
# ============================================================

from dataclasses import dataclass


# ============================================================
# EXECUTION STACK
# ============================================================

@dataclass(slots=True)
class ExecutionStack:
    """
    Fully wired execution stack.

    Used by:

        Backtests
        Research
        Paper Trading
        Live Trading
    """

    cost_model: ExecutionCostModel

    slippage_model: BaseSlippageModel

    liquidity_engine: InstitutionalLiquidityEngine

    impact_engine: InstitutionalImpactEngine

    simulator: ExecutionSimulator

    broker: BrokerAdapter | None = None

    oms: OrderManagementSystem | None = None


# ============================================================
# DEFAULT BUILDERS
# ============================================================

def build_backtest_execution_engine(
    *,
    cost_model: ExecutionCostModel,
    slippage_model: BaseSlippageModel,
    liquidity_engine: InstitutionalLiquidityEngine,
    impact_engine: InstitutionalImpactEngine,
    simulator_config: ExecutionSimulatorConfig,
) -> ExecutionStack:
    """
    Build execution stack for backtesting.
    """

    simulator = ExecutionSimulator(
        cost_model=cost_model,
        slippage_model=slippage_model,
        liquidity_engine=liquidity_engine,
        impact_engine=impact_engine,
        config=simulator_config,
    )

    return ExecutionStack(
        cost_model=cost_model,
        slippage_model=slippage_model,
        liquidity_engine=liquidity_engine,
        impact_engine=impact_engine,
        simulator=simulator,
    )


# ============================================================
# PAPER TRADING ENGINE
# ============================================================

def build_paper_execution_engine(
    *,
    cost_model: ExecutionCostModel,
    slippage_model: BaseSlippageModel,
    liquidity_engine: InstitutionalLiquidityEngine,
    impact_engine: InstitutionalImpactEngine,
    simulator_config: ExecutionSimulatorConfig,
) -> ExecutionStack:
    """
    Build paper-trading execution stack.
    """

    broker = PaperBrokerAdapter()

    oms = OrderManagementSystem(
        broker=broker,
    )

    simulator = ExecutionSimulator(
        cost_model=cost_model,
        slippage_model=slippage_model,
        liquidity_engine=liquidity_engine,
        impact_engine=impact_engine,
        config=simulator_config,
    )

    return ExecutionStack(
        cost_model=cost_model,
        slippage_model=slippage_model,
        liquidity_engine=liquidity_engine,
        impact_engine=impact_engine,
        simulator=simulator,
        broker=broker,
        oms=oms,
    )


# ============================================================
# LIVE IBKR ENGINE
# ============================================================

def build_ibkr_execution_engine(
    *,
    cost_model: ExecutionCostModel,
    slippage_model: BaseSlippageModel,
    liquidity_engine: InstitutionalLiquidityEngine,
    impact_engine: InstitutionalImpactEngine,
    simulator_config: ExecutionSimulatorConfig,
) -> ExecutionStack:
    """
    Build Interactive Brokers execution stack.

    Production adapter implementation
    plugged later.
    """

    broker = IBKRAdapter()

    oms = OrderManagementSystem(
        broker=broker,
    )

    simulator = ExecutionSimulator(
        cost_model=cost_model,
        slippage_model=slippage_model,
        liquidity_engine=liquidity_engine,
        impact_engine=impact_engine,
        config=simulator_config,
    )

    return ExecutionStack(
        cost_model=cost_model,
        slippage_model=slippage_model,
        liquidity_engine=liquidity_engine,
        impact_engine=impact_engine,
        simulator=simulator,
        broker=broker,
        oms=oms,
    )


# ============================================================
# LIVE FIX ENGINE
# ============================================================

def build_fix_execution_engine(
    *,
    cost_model: ExecutionCostModel,
    slippage_model: BaseSlippageModel,
    liquidity_engine: InstitutionalLiquidityEngine,
    impact_engine: InstitutionalImpactEngine,
    simulator_config: ExecutionSimulatorConfig,
) -> ExecutionStack:
    """
    Build FIX execution stack.
    """

    broker = FIXAdapter()

    oms = OrderManagementSystem(
        broker=broker,
    )

    simulator = ExecutionSimulator(
        cost_model=cost_model,
        slippage_model=slippage_model,
        liquidity_engine=liquidity_engine,
        impact_engine=impact_engine,
        config=simulator_config,
    )

    return ExecutionStack(
        cost_model=cost_model,
        slippage_model=slippage_model,
        liquidity_engine=liquidity_engine,
        impact_engine=impact_engine,
        simulator=simulator,
        broker=broker,
        oms=oms,
    )


# ============================================================
# GENERIC FACTORY
# ============================================================

def build_execution_engine(
    *,
    mode: str,
    cost_model: ExecutionCostModel,
    slippage_model: BaseSlippageModel,
    liquidity_engine: InstitutionalLiquidityEngine,
    impact_engine: InstitutionalImpactEngine,
    simulator_config: ExecutionSimulatorConfig,
) -> ExecutionStack:
    """
    Unified execution engine factory.

    Modes
    -----
    backtest
    paper
    ibkr
    fix
    """

    mode = mode.lower()

    if mode == "backtest":

        return build_backtest_execution_engine(
            cost_model=cost_model,
            slippage_model=slippage_model,
            liquidity_engine=liquidity_engine,
            impact_engine=impact_engine,
            simulator_config=simulator_config,
        )

    if mode == "paper":

        return build_paper_execution_engine(
            cost_model=cost_model,
            slippage_model=slippage_model,
            liquidity_engine=liquidity_engine,
            impact_engine=impact_engine,
            simulator_config=simulator_config,
        )

    if mode == "ibkr":

        return build_ibkr_execution_engine(
            cost_model=cost_model,
            slippage_model=slippage_model,
            liquidity_engine=liquidity_engine,
            impact_engine=impact_engine,
            simulator_config=simulator_config,
        )

    if mode == "fix":

        return build_fix_execution_engine(
            cost_model=cost_model,
            slippage_model=slippage_model,
            liquidity_engine=liquidity_engine,
            impact_engine=impact_engine,
            simulator_config=simulator_config,
        )

    raise ValueError(
        f"Unknown execution mode: {mode}"
    )


# ============================================================
# SIMPLE EXECUTION API
# ============================================================

def simulate_execution(
    *,
    orders: pd.DataFrame,
    execution_stack: ExecutionStack,
    market_data: pd.DataFrame,
):
    """
    Convenience wrapper.

    Used by:

        portfolio pipeline
        backtests
        execution research
    """

    return execution_stack.simulator.run(
        orders=orders,
        market_data=market_data,
    )


# ============================================================
# OMS API
# ============================================================

def submit_order(
    *,
    order: Order,
    execution_stack: ExecutionStack,
):
    """
    Submit order through OMS.
    """

    if execution_stack.oms is None:

        raise RuntimeError(
            "OMS not available."
        )

    return (
        execution_stack.oms.submit(
            order
        )
    )


# ============================================================
# OMS STATUS API
# ============================================================

def get_order_status(
    *,
    broker_order_id: str,
    execution_stack: ExecutionStack,
):
    """
    Query OMS order status.
    """

    if execution_stack.oms is None:

        raise RuntimeError(
            "OMS not available."
        )

    return (
        execution_stack.oms.status(
            broker_order_id
        )
    )


# ============================================================
# OMS POSITIONS API
# ============================================================

def get_live_positions(
    *,
    execution_stack: ExecutionStack,
):
    """
    Retrieve broker positions.
    """

    if execution_stack.oms is None:

        raise RuntimeError(
            "OMS not available."
        )

    return (
        execution_stack.oms.positions()
    )