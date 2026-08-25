"""
Why rebalance.py is important

This is the module that decides:

Current Portfolio
        │
        ▼
New Target Portfolio
        │
        ▼
Need Rebalance?
        │
        ▼
Generate Trades
        │
        ▼
Apply Costs
        │
        ▼
Final Executable Orders

"""


"""
==============================================================
REBALANCE ENGINE
==============================================================

Institutional-grade portfolio rebalancing engine.

Responsibilities
----------------
• Rebalance scheduling
• Drift detection
• Trigger evaluation
• Trade generation
• Turnover control
• Transaction-cost-aware rebalancing

This module DOES NOT

• Select securities
• Score securities
• Optimize portfolios
• Execute trades

It determines WHEN and HOW portfolios should be rebalanced.

==============================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1e-12


# ============================================================
# REBALANCE TYPES
# ============================================================

class RebalanceType(str, Enum):
    """
    Supported rebalance modes.
    """

    SCHEDULED = "scheduled"

    DRIFT = "drift"

    RISK = "risk"

    TURNOVER = "turnover"

    HYBRID = "hybrid"


# ============================================================
# REBALANCE FREQUENCY
# ============================================================

class RebalanceFrequency(str, Enum):

    DAILY = "daily"

    WEEKLY = "weekly"

    MONTHLY = "monthly"

    QUARTERLY = "quarterly"

    SEMI_ANNUAL = "semi_annual"

    ANNUAL = "annual"


# ============================================================
# CONFIG
# ============================================================

@dataclass(slots=True)
class RebalanceConfig:
    """
    Global rebalance configuration.
    """

    rebalance_type: RebalanceType = (
        RebalanceType.HYBRID
    )

    frequency: RebalanceFrequency = (
        RebalanceFrequency.MONTHLY
    )

    drift_threshold: float = 0.05

    max_turnover: float = 0.30

    min_trade_weight: float = 0.001

    transaction_cost_bps: float = 10.0

    enforce_turnover_limit: bool = True

    enforce_cost_budget: bool = True

    cost_budget: float = 0.01

    risk_trigger_volatility: float = 0.35


# ============================================================
# VALIDATION
# ============================================================

def validate_rebalance_config(
    config: RebalanceConfig,
) -> None:

    if config.drift_threshold < 0:

        raise ValueError(
            "drift_threshold must be >= 0"
        )

    if config.max_turnover < 0:

        raise ValueError(
            "max_turnover must be >= 0"
        )

    if config.min_trade_weight < 0:

        raise ValueError(
            "min_trade_weight must be >= 0"
        )

    if config.transaction_cost_bps < 0:

        raise ValueError(
            "transaction_cost_bps must be >= 0"
        )


# ============================================================
# PORTFOLIO VALIDATION
# ============================================================

def validate_portfolio_frame(
    portfolio: pd.DataFrame,
) -> None:
    """
    Minimal validation required
    by the rebalance engine.
    """

    required = [
        "Ticker",
        "Position_Weight",
    ]

    missing = [
        c
        for c in required
        if c not in portfolio.columns
    ]

    if missing:

        raise ValueError(
            "Portfolio missing columns:\n"
            + "\n".join(missing)
        )

    weights = portfolio[
        "Position_Weight"
    ]

    if weights.isna().any():

        raise ValueError(
            "NaN weights detected."
        )

    if np.isinf(weights).any():

        raise ValueError(
            "Infinite weights detected."
        )


# ============================================================
# REBALANCE BASE CLASS
# ============================================================

class BaseRebalanceEngine(ABC):
    """
    Abstract rebalance engine.
    """

    def __init__(
        self,
        config: RebalanceConfig,
    ) -> None:

        validate_rebalance_config(
            config
        )

        self.config = config

    @abstractmethod
    def should_rebalance(
        self,
        current_date: datetime,
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame,
        previous_rebalance_date: Optional[
            datetime
        ] = None,
    ) -> bool:
        """
        Determine whether
        rebalance should occur.
        """

        raise NotImplementedError


# ============================================================
# HELPERS
# ============================================================

def normalize_weights(
    weights: pd.Series,
) -> pd.Series:

    total = weights.sum()

    if total <= EPS:

        return pd.Series(
            0.0,
            index=weights.index,
        )

    return weights / total


def align_portfolios(
    current_portfolio: pd.DataFrame,
    target_portfolio: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Align portfolios onto
    common ticker universe.
    """

    current = (
        current_portfolio
        .set_index("Ticker")
        ["Position_Weight"]
    )

    target = (
        target_portfolio
        .set_index("Ticker")
        ["Position_Weight"]
    )

    universe = (
        current.index.union(
            target.index
        )
    )

    current = (
        current.reindex(universe)
        .fillna(0.0)
    )

    target = (
        target.reindex(universe)
        .fillna(0.0)
    )

    return current, target

# ============================================================
# PART 2: TRADE INSTRUCTION
# ============================================================

@dataclass(slots=True)
class TradeInstruction:
    """
    Single rebalance trade.

    Represents a change from current
    weight to target weight.
    """

    ticker: str

    current_weight: float

    target_weight: float

    trade_weight: float

    direction: str

    estimated_cost: float = 0.0


# ============================================================
# REBALANCE DIAGNOSTICS
# ============================================================

@dataclass(slots=True)
class RebalanceDiagnostics:
    """
    Diagnostic information
    produced by a rebalance run.
    """

    turnover: float = 0.0

    gross_trade_weight: float = 0.0

    net_trade_weight: float = 0.0

    estimated_cost: float = 0.0

    drift_score: float = 0.0

    positions_added: int = 0

    positions_removed: int = 0

    positions_modified: int = 0

    trade_count: int = 0

    metadata: dict = field(
        default_factory=dict
    )


# ============================================================
# REBALANCE DECISION
# ============================================================

@dataclass(slots=True)
class RebalanceDecision:
    """
    Output of a rebalance trigger.

    Used before trade generation.
    """

    should_rebalance: bool

    trigger_type: str

    reason: str

    score: float = 0.0


# ============================================================
# REBALANCE RESULT
# ============================================================

@dataclass(slots=True)
class RebalanceResult:
    """
    Final output from the
    rebalance engine.
    """

    decision: RebalanceDecision

    target_portfolio: pd.DataFrame

    trades: list[TradeInstruction]

    diagnostics: RebalanceDiagnostics


# ============================================================
# RESULT HELPERS
# ============================================================

def empty_rebalance_result(
    portfolio: pd.DataFrame,
) -> RebalanceResult:
    """
    Convenience helper used
    when no rebalance occurs.
    """

    return RebalanceResult(
        decision=RebalanceDecision(
            should_rebalance=False,
            trigger_type="none",
            reason="No rebalance required",
            score=0.0,
        ),
        target_portfolio=portfolio.copy(),
        trades=[],
        diagnostics=RebalanceDiagnostics(),
    )


# ============================================================
# TRADE HELPERS
# ============================================================

def trade_direction(
    trade_weight: float,
) -> str:
    """
    Determine trade direction.
    """

    if trade_weight > EPS:

        return "BUY"

    if trade_weight < -EPS:

        return "SELL"

    return "HOLD"


def estimate_trade_cost(
    trade_weight: float,
    transaction_cost_bps: float,
) -> float:
    """
    Simple cost estimate.

    Cost = |trade_weight| * bps
    """

    return (
        abs(trade_weight)
        * transaction_cost_bps
        / 10000.0
    )

# ============================================================
# PART 3: SCHEDULED REBALANCE ENGINE
# ============================================================

class ScheduledRebalanceEngine:
    """
    Calendar-based rebalance engine.

    Supports:

        DAILY
        WEEKLY
        MONTHLY
        QUARTERLY
        SEMI_ANNUAL
        ANNUAL
    """

    def __init__(
        self,
        config: RebalanceConfig,
    ) -> None:

        self.config = config

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def should_rebalance(
        self,
        current_date: pd.Timestamp,
        previous_rebalance_date: pd.Timestamp | None,
    ) -> RebalanceDecision:
        """
        Evaluate calendar schedule.
        """

        if previous_rebalance_date is None:

            return RebalanceDecision(
                should_rebalance=True,
                trigger_type="initial",
                reason="Initial portfolio build",
                score=1.0,
            )

        frequency = (
            self.config.rebalance_frequency
        )

        if (
            frequency
            == RebalanceFrequency.DAILY
        ):

            return self._daily(
                current_date,
                previous_rebalance_date,
            )

        if (
            frequency
            == RebalanceFrequency.WEEKLY
        ):

            return self._weekly(
                current_date,
                previous_rebalance_date,
            )

        if (
            frequency
            == RebalanceFrequency.MONTHLY
        ):

            return self._monthly(
                current_date,
                previous_rebalance_date,
            )

        if (
            frequency
            == RebalanceFrequency.QUARTERLY
        ):

            return self._quarterly(
                current_date,
                previous_rebalance_date,
            )

        if (
            frequency
            == RebalanceFrequency.SEMI_ANNUAL
        ):

            return self._semi_annual(
                current_date,
                previous_rebalance_date,
            )

        if (
            frequency
            == RebalanceFrequency.ANNUAL
        ):

            return self._annual(
                current_date,
                previous_rebalance_date,
            )

        return RebalanceDecision(
            should_rebalance=False,
            trigger_type="schedule",
            reason="Unknown schedule",
            score=0.0,
        )

    # --------------------------------------------------------
    # DAILY
    # --------------------------------------------------------

    def _daily(
        self,
        current_date: pd.Timestamp,
        previous_date: pd.Timestamp,
    ) -> RebalanceDecision:

        if current_date > previous_date:

            return RebalanceDecision(
                should_rebalance=True,
                trigger_type="daily",
                reason="Daily schedule reached",
                score=1.0,
            )

        return RebalanceDecision(
            should_rebalance=False,
            trigger_type="daily",
            reason="Already rebalanced today",
            score=0.0,
        )

    # --------------------------------------------------------
    # WEEKLY
    # --------------------------------------------------------

    def _weekly(
        self,
        current_date: pd.Timestamp,
        previous_date: pd.Timestamp,
    ) -> RebalanceDecision:

        days = (
            current_date
            - previous_date
        ).days

        rebalance = days >= 7

        return RebalanceDecision(
            should_rebalance=rebalance,
            trigger_type="weekly",
            reason=(
                "Weekly schedule reached"
                if rebalance
                else "Waiting for weekly cycle"
            ),
            score=float(days) / 7.0,
        )

    # --------------------------------------------------------
    # MONTHLY
    # --------------------------------------------------------

    def _monthly(
        self,
        current_date: pd.Timestamp,
        previous_date: pd.Timestamp,
    ) -> RebalanceDecision:

        rebalance = (

            current_date.year
            != previous_date.year

            or

            current_date.month
            != previous_date.month
        )

        return RebalanceDecision(
            should_rebalance=rebalance,
            trigger_type="monthly",
            reason=(
                "Monthly schedule reached"
                if rebalance
                else "Same month"
            ),
            score=float(rebalance),
        )

    # --------------------------------------------------------
    # QUARTERLY
    # --------------------------------------------------------

    def _quarterly(
        self,
        current_date: pd.Timestamp,
        previous_date: pd.Timestamp,
    ) -> RebalanceDecision:

        current_q = (
            current_date.month - 1
        ) // 3

        previous_q = (
            previous_date.month - 1
        ) // 3

        rebalance = (

            current_date.year
            != previous_date.year

            or

            current_q
            != previous_q
        )

        return RebalanceDecision(
            should_rebalance=rebalance,
            trigger_type="quarterly",
            reason=(
                "Quarterly schedule reached"
                if rebalance
                else "Same quarter"
            ),
            score=float(rebalance),
        )

    # --------------------------------------------------------
    # SEMI ANNUAL
    # --------------------------------------------------------

    def _semi_annual(
        self,
        current_date: pd.Timestamp,
        previous_date: pd.Timestamp,
    ) -> RebalanceDecision:

        current_half = (
            1 if current_date.month <= 6
            else 2
        )

        previous_half = (
            1 if previous_date.month <= 6
            else 2
        )

        rebalance = (

            current_date.year
            != previous_date.year

            or

            current_half
            != previous_half
        )

        return RebalanceDecision(
            should_rebalance=rebalance,
            trigger_type="semi_annual",
            reason=(
                "Semi-annual schedule reached"
                if rebalance
                else "Same half-year"
            ),
            score=float(rebalance),
        )

    # --------------------------------------------------------
    # ANNUAL
    # --------------------------------------------------------

    def _annual(
        self,
        current_date: pd.Timestamp,
        previous_date: pd.Timestamp,
    ) -> RebalanceDecision:

        rebalance = (
            current_date.year
            != previous_date.year
        )

        return RebalanceDecision(
            should_rebalance=rebalance,
            trigger_type="annual",
            reason=(
                "Annual schedule reached"
                if rebalance
                else "Same year"
            ),
            score=float(rebalance),
        )

# ============================================================
# PART 4: DRIFT-BASED REBALANCE ENGINE
# ============================================================

class DriftRebalanceEngine:
    """
    Institutional drift-based rebalancing.

    Rebalances only when actual portfolio
    weights drift sufficiently from target
    weights.

    Benefits

        Lower turnover
        Lower costs
        Better tax efficiency
        Institutional standard
    """

    def __init__(
        self,
        config: RebalanceConfig,
    ) -> None:

        self.config = config

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def should_rebalance(
        self,
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame,
    ) -> RebalanceDecision:

        drift_score = self.compute_drift_score(
            current_portfolio,
            target_portfolio,
        )

        threshold = (
            self.config.drift_threshold
        )

        rebalance = (
            drift_score >= threshold
        )

        return RebalanceDecision(
            should_rebalance=rebalance,
            trigger_type="drift",
            reason=(
                f"Drift {drift_score:.4f} "
                f">= threshold {threshold:.4f}"
                if rebalance
                else
                f"Drift {drift_score:.4f} "
                f"< threshold {threshold:.4f}"
            ),
            score=float(drift_score),
        )

    # --------------------------------------------------------
    # DRIFT SCORE
    # --------------------------------------------------------

    def compute_drift_score(
        self,
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame,
    ) -> float:
        """
        Institutional drift metric.

        Drift =

            sum(
                |current_weight
                 -
                 target_weight|
            )

        Result

            0.00 = identical

            0.20 = 20% portfolio drift
        """

        merged = self._align_portfolios(
            current_portfolio,
            target_portfolio,
        )

        drift = (

            merged["CurrentWeight"]
            -
            merged["TargetWeight"]

        ).abs().sum()

        return float(drift)

    # --------------------------------------------------------
    # MAX POSITION DRIFT
    # --------------------------------------------------------

    def max_position_drift(
        self,
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame,
    ) -> float:

        merged = self._align_portfolios(
            current_portfolio,
            target_portfolio,
        )

        drift = (

            merged["CurrentWeight"]
            -
            merged["TargetWeight"]

        ).abs()

        return float(
            drift.max()
        )

    # --------------------------------------------------------
    # RMS DRIFT
    # --------------------------------------------------------

    def rms_drift(
        self,
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame,
    ) -> float:
        """
        Root mean squared drift.

        More sensitive to
        large deviations.
        """

        merged = self._align_portfolios(
            current_portfolio,
            target_portfolio,
        )

        drift = (

            merged["CurrentWeight"]
            -
            merged["TargetWeight"]

        )

        return float(
            np.sqrt(
                np.mean(
                    drift ** 2
                )
            )
        )

    # --------------------------------------------------------
    # DRIFT REPORT
    # --------------------------------------------------------

    def drift_report(
        self,
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame,
    ) -> pd.DataFrame:

        merged = self._align_portfolios(
            current_portfolio,
            target_portfolio,
        )

        merged["Drift"] = (

            merged["CurrentWeight"]
            -
            merged["TargetWeight"]

        )

        merged["AbsDrift"] = (
            merged["Drift"]
            .abs()
        )

        return (

            merged

            .sort_values(
                "AbsDrift",
                ascending=False,
            )

            .reset_index(
                drop=True
            )

        )

    # --------------------------------------------------------
    # ALIGN PORTFOLIOS
    # --------------------------------------------------------

    @staticmethod
    def _align_portfolios(
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align current and target portfolios.

        Missing securities
        receive zero weight.
        """

        current = (

            current_portfolio[
                [
                    "Ticker",
                    "Position_Weight",
                ]
            ]

            .rename(
                columns={
                    "Position_Weight":
                    "CurrentWeight"
                }
            )

        )

        target = (

            target_portfolio[
                [
                    "Ticker",
                    "Position_Weight",
                ]
            ]

            .rename(
                columns={
                    "Position_Weight":
                    "TargetWeight"
                }
            )

        )

        merged = current.merge(
            target,
            on="Ticker",
            how="outer",
        )

        merged[
            "CurrentWeight"
        ] = (

            merged[
                "CurrentWeight"
            ]

            .fillna(0.0)

        )

        merged[
            "TargetWeight"
        ] = (

            merged[
                "TargetWeight"
            ]

            .fillna(0.0)

        )

        return merged
    

# ============================================================
# PART 5: TURNOVER-AWARE REBALANCE ENGINE
# ============================================================

class TurnoverAwareRebalanceEngine:
    """
    Institutional turnover-aware rebalancing.

    Purpose
    -------
    Prevent excessive trading even when
    drift signals rebalance.

    Typical workflow

        Schedule Trigger
                +
        Drift Trigger
                +
        Turnover Budget
                =
        Final Decision

    This is very common among
    institutional asset managers.
    """

    def __init__(
        self,
        config: RebalanceConfig,
    ) -> None:

        self.config = config

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def should_rebalance(
        self,
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame,
        *,
        trigger_decision: RebalanceDecision,
    ) -> RebalanceDecision:

        turnover = self.estimate_turnover(
            current_portfolio,
            target_portfolio,
        )

        max_turnover = (
            self.config.max_turnover
        )

        # -------------------------------------
        # Trigger says NO
        # -------------------------------------

        if not trigger_decision.should_rebalance:

            return RebalanceDecision(
                should_rebalance=False,
                trigger_type="turnover",
                reason=(
                    "No upstream rebalance trigger"
                ),
                score=float(turnover),
            )

        # -------------------------------------
        # Trigger says YES
        # Turnover acceptable
        # -------------------------------------

        if turnover <= max_turnover:

            return RebalanceDecision(
                should_rebalance=True,
                trigger_type="turnover",
                reason=(
                    f"Turnover {turnover:.4f} "
                    f"<= limit {max_turnover:.4f}"
                ),
                score=float(turnover),
            )

        # -------------------------------------
        # Turnover too high
        # -------------------------------------

        return RebalanceDecision(
            should_rebalance=False,
            trigger_type="turnover",
            reason=(
                f"Turnover {turnover:.4f} "
                f"> limit {max_turnover:.4f}"
            ),
            score=float(turnover),
        )

    # --------------------------------------------------------
    # TURNOVER ESTIMATION
    # --------------------------------------------------------

    def estimate_turnover(
        self,
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame,
    ) -> float:
        """
        Portfolio turnover estimate.

        Formula

            0.5 *
            sum(
                |new_weight
                 -
                 old_weight|
            )

        Standard institutional definition.
        """

        merged = self._align_portfolios(
            current_portfolio,
            target_portfolio,
        )

        turnover = 0.5 * (

            (
                merged["TargetWeight"]
                -
                merged["CurrentWeight"]
            )

            .abs()

            .sum()

        )

        return float(turnover)

    # --------------------------------------------------------
    # TURNOVER REPORT
    # --------------------------------------------------------

    def turnover_report(
        self,
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame,
    ) -> pd.DataFrame:

        merged = self._align_portfolios(
            current_portfolio,
            target_portfolio,
        )

        merged["Trade"] = (

            merged["TargetWeight"]
            -
            merged["CurrentWeight"]

        )

        merged["AbsTrade"] = (
            merged["Trade"]
            .abs()
        )

        return (

            merged

            .sort_values(
                "AbsTrade",
                ascending=False,
            )

            .reset_index(
                drop=True
            )

        )

    # --------------------------------------------------------
    # BUY TURNOVER
    # --------------------------------------------------------

    def buy_turnover(
        self,
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame,
    ) -> float:

        merged = self._align_portfolios(
            current_portfolio,
            target_portfolio,
        )

        trades = (

            merged["TargetWeight"]
            -
            merged["CurrentWeight"]

        )

        return float(
            trades.clip(lower=0)
            .sum()
        )

    # --------------------------------------------------------
    # SELL TURNOVER
    # --------------------------------------------------------

    def sell_turnover(
        self,
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame,
    ) -> float:

        merged = self._align_portfolios(
            current_portfolio,
            target_portfolio,
        )

        trades = (

            merged["TargetWeight"]
            -
            merged["CurrentWeight"]

        )

        return float(
            np.abs(
                trades.clip(upper=0)
            ).sum()
        )

    # --------------------------------------------------------
    # TRANSACTION COST ESTIMATE
    # --------------------------------------------------------

    def estimate_transaction_cost(
        self,
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame,
        *,
        cost_per_turnover: float | None = None,
    ) -> float:
        """
        Estimate portfolio trading cost.

        Example

            10 bps turnover cost

            cost_per_turnover=0.001
        """

        turnover = self.estimate_turnover(
            current_portfolio,
            target_portfolio,
        )

        cost_rate = (

            cost_per_turnover

            if cost_per_turnover
            is not None

            else

            self.config.transaction_cost
        )

        return float(
            turnover * cost_rate
        )

    # --------------------------------------------------------
    # ALIGN PORTFOLIOS
    # --------------------------------------------------------

    @staticmethod
    def _align_portfolios(
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame,
    ) -> pd.DataFrame:

        current = (

            current_portfolio[
                [
                    "Ticker",
                    "Position_Weight",
                ]
            ]

            .rename(
                columns={
                    "Position_Weight":
                    "CurrentWeight"
                }
            )

        )

        target = (

            target_portfolio[
                [
                    "Ticker",
                    "Position_Weight",
                ]
            ]

            .rename(
                columns={
                    "Position_Weight":
                    "TargetWeight"
                }
            )

        )

        merged = current.merge(
            target,
            on="Ticker",
            how="outer",
        )

        merged[
            "CurrentWeight"
        ] = (
            merged[
                "CurrentWeight"
            ].fillna(0.0)
        )

        merged[
            "TargetWeight"
        ] = (
            merged[
                "TargetWeight"
            ].fillna(0.0)
        )

        return merged
    

# ============================================================
# PART 6: EVENT-DRIVEN REBALANCE ENGINE
# ============================================================

class EventDrivenRebalanceEngine:
    """
    Institutional event-driven rebalance engine.

    Triggers portfolio review when
    significant market or portfolio
    events occur.

    Examples

        Earnings surprise
        Regime shift
        Risk breach
        Liquidity deterioration
        Factor breakdown
        Volatility spike
        Drawdown breach
    """

    def __init__(
        self,
        config: RebalanceConfig,
    ) -> None:

        self.config = config

    # --------------------------------------------------------
    # MASTER API
    # --------------------------------------------------------

    def evaluate_events(
        self,
        *,
        earnings_surprise: float | None = None,
        regime_changed: bool = False,
        volatility_spike: bool = False,
        drawdown_breach: bool = False,
        liquidity_event: bool = False,
        factor_breakdown: bool = False,
        risk_limit_breach: bool = False,
    ) -> RebalanceDecision:
        """
        Evaluate all event triggers.
        """

        triggers = []

        score = 0.0

        # -------------------------------------
        # Earnings
        # -------------------------------------

        if (
            earnings_surprise
            is not None
        ):

            if abs(
                earnings_surprise
            ) >= (
                self.config.earnings_surprise_threshold
            ):

                triggers.append(
                    f"Earnings surprise "
                    f"{earnings_surprise:.2%}"
                )

                score += 1.0

        # -------------------------------------
        # Regime
        # -------------------------------------

        if regime_changed:

            triggers.append(
                "Market regime change"
            )

            score += 2.0

        # -------------------------------------
        # Volatility
        # -------------------------------------

        if volatility_spike:

            triggers.append(
                "Volatility spike"
            )

            score += 1.5

        # -------------------------------------
        # Drawdown
        # -------------------------------------

        if drawdown_breach:

            triggers.append(
                "Drawdown breach"
            )

            score += 3.0

        # -------------------------------------
        # Liquidity
        # -------------------------------------

        if liquidity_event:

            triggers.append(
                "Liquidity event"
            )

            score += 2.0

        # -------------------------------------
        # Factor breakdown
        # -------------------------------------

        if factor_breakdown:

            triggers.append(
                "Factor breakdown"
            )

            score += 2.0

        # -------------------------------------
        # Risk breach
        # -------------------------------------

        if risk_limit_breach:

            triggers.append(
                "Risk limit breach"
            )

            score += 4.0

        rebalance = (
            len(triggers) > 0
        )

        return RebalanceDecision(
            should_rebalance=rebalance,
            trigger_type="event",
            reason=(
                "; ".join(triggers)
                if triggers
                else "No event trigger"
            ),
            score=float(score),
        )

    # --------------------------------------------------------
    # EARNINGS SURPRISE
    # --------------------------------------------------------

    def earnings_trigger(
        self,
        surprise: float,
    ) -> bool:

        return (

            abs(surprise)

            >=

            self.config
            .earnings_surprise_threshold
        )

    # --------------------------------------------------------
    # REGIME CHANGE
    # --------------------------------------------------------

    @staticmethod
    def regime_change_trigger(
        previous_regime: str,
        current_regime: str,
    ) -> bool:

        return (
            previous_regime
            != current_regime
        )

    # --------------------------------------------------------
    # VOLATILITY SPIKE
    # --------------------------------------------------------

    def volatility_spike_trigger(
        self,
        current_volatility: float,
        baseline_volatility: float,
    ) -> bool:

        if baseline_volatility <= 0:

            return False

        ratio = (
            current_volatility
            / baseline_volatility
        )

        return (

            ratio

            >=

            self.config
            .volatility_spike_multiple
        )

    # --------------------------------------------------------
    # DRAWDOWN BREACH
    # --------------------------------------------------------

    def drawdown_trigger(
        self,
        current_drawdown: float,
    ) -> bool:

        return (

            current_drawdown

            <=

            -abs(
                self.config
                .drawdown_threshold
            )
        )

    # --------------------------------------------------------
    # FACTOR BREAKDOWN
    # --------------------------------------------------------

    def factor_breakdown_trigger(
        self,
        factor_ic: float,
    ) -> bool:
        """
        Information coefficient collapse.
        """

        return (

            factor_ic

            <=

            self.config
            .factor_breakdown_threshold
        )

    # --------------------------------------------------------
    # LIQUIDITY EVENT
    # --------------------------------------------------------

    def liquidity_trigger(
        self,
        current_adv: float,
        baseline_adv: float,
    ) -> bool:

        if baseline_adv <= 0:

            return False

        ratio = (
            current_adv
            / baseline_adv
        )

        return (

            ratio

            <=

            self.config
            .liquidity_shock_threshold
        )

    # --------------------------------------------------------
    # RISK BREACH
    # --------------------------------------------------------

    @staticmethod
    def risk_breach_trigger(
        breached: bool,
    ) -> bool:

        return bool(
            breached
        )

    # --------------------------------------------------------
    # EVENT SCORECARD
    # --------------------------------------------------------

    def event_scorecard(
        self,
        *,
        earnings_surprise: float | None = None,
        regime_changed: bool = False,
        volatility_spike: bool = False,
        drawdown_breach: bool = False,
        liquidity_event: bool = False,
        factor_breakdown: bool = False,
        risk_limit_breach: bool = False,
    ) -> pd.DataFrame:

        rows = [

            {
                "Event":
                "Earnings Surprise",

                "Triggered":
                (
                    earnings_surprise
                    is not None
                    and
                    self.earnings_trigger(
                        earnings_surprise
                    )
                ),
            },

            {
                "Event":
                "Regime Change",

                "Triggered":
                regime_changed,
            },

            {
                "Event":
                "Volatility Spike",

                "Triggered":
                volatility_spike,
            },

            {
                "Event":
                "Drawdown Breach",

                "Triggered":
                drawdown_breach,
            },

            {
                "Event":
                "Liquidity Event",

                "Triggered":
                liquidity_event,
            },

            {
                "Event":
                "Factor Breakdown",

                "Triggered":
                factor_breakdown,
            },

            {
                "Event":
                "Risk Limit Breach",

                "Triggered":
                risk_limit_breach,
            },

        ]

        return pd.DataFrame(
            rows
        )
    

# ============================================================
# PART 7: HYBRID REBALANCE COORDINATOR
# ============================================================

@dataclass(slots=True)
class HybridRebalanceResult:
    """
    Final rebalance decision.

    Combines all rebalance engines.
    """

    should_rebalance: bool

    final_score: float

    winning_trigger: str

    reasons: list[str]

    scheduled_decision: RebalanceDecision | None = None

    drift_decision: RebalanceDecision | None = None

    turnover_decision: RebalanceDecision | None = None

    event_decision: RebalanceDecision | None = None


# ============================================================
# HYBRID COORDINATOR
# ============================================================

class HybridRebalanceCoordinator:
    """
    Institutional rebalance coordinator.

    Integrates

        Scheduled Engine
        Drift Engine
        Turnover Engine
        Event Engine

    Final decision logic can be
    configured without changing
    the underlying engines.
    """

    def __init__(
        self,
        config: RebalanceConfig,
    ) -> None:

        self.config = config

        self.minimum_trigger_score = (
            getattr(
                config,
                "minimum_trigger_score",
                1.0,
            )
        )

    # --------------------------------------------------------
    # MASTER DECISION
    # --------------------------------------------------------

    def decide(
        self,
        *,
        scheduled_decision: RebalanceDecision | None = None,
        drift_decision: RebalanceDecision | None = None,
        turnover_decision: RebalanceDecision | None = None,
        event_decision: RebalanceDecision | None = None,
    ) -> HybridRebalanceResult:

        reasons: list[str] = []

        total_score = 0.0

        trigger_scores = {}

        # -------------------------------------
        # Scheduled
        # -------------------------------------

        if scheduled_decision:

            trigger_scores["schedule"] = (
                scheduled_decision.score
            )

            if (
                scheduled_decision
                .should_rebalance
            ):

                reasons.append(
                    scheduled_decision.reason
                )

                total_score += (
                    scheduled_decision.score
                )

        # -------------------------------------
        # Drift
        # -------------------------------------

        if drift_decision:

            trigger_scores["drift"] = (
                drift_decision.score
            )

            if (
                drift_decision
                .should_rebalance
            ):

                reasons.append(
                    drift_decision.reason
                )

                total_score += (
                    drift_decision.score
                )

        # -------------------------------------
        # Event
        # -------------------------------------

        if event_decision:

            trigger_scores["event"] = (
                event_decision.score
            )

            if (
                event_decision
                .should_rebalance
            ):

                reasons.append(
                    event_decision.reason
                )

                total_score += (
                    event_decision.score
                )

        # -------------------------------------
        # Turnover
        # -------------------------------------

        turnover_allowed = True

        if turnover_decision:

            turnover_allowed = (

                turnover_decision
                .should_rebalance

            )

            trigger_scores["turnover"] = (
                turnover_decision.score
            )

            reasons.append(
                turnover_decision.reason
            )

        # -------------------------------------
        # Final Logic
        # -------------------------------------

        rebalance = (

            total_score
            >=
            self.minimum_trigger_score

            and

            turnover_allowed

        )

        winning_trigger = self._winning_trigger(
            trigger_scores
        )

        return HybridRebalanceResult(
            should_rebalance=rebalance,
            final_score=float(
                total_score
            ),
            winning_trigger=winning_trigger,
            reasons=reasons,
            scheduled_decision=scheduled_decision,
            drift_decision=drift_decision,
            turnover_decision=turnover_decision,
            event_decision=event_decision,
        )

    # --------------------------------------------------------
    # WINNING TRIGGER
    # --------------------------------------------------------

    @staticmethod
    def _winning_trigger(
        trigger_scores: dict[str, float],
    ) -> str:

        if not trigger_scores:

            return "none"

        return max(
            trigger_scores,
            key=trigger_scores.get,
        )

    # --------------------------------------------------------
    # EXPLAIN DECISION
    # --------------------------------------------------------

    @staticmethod
    def explain(
        result: HybridRebalanceResult,
    ) -> str:

        decision = (

            "REBALANCE"

            if result.should_rebalance

            else

            "HOLD"

        )

        return (
            f"{decision} | "
            f"Score={result.final_score:.4f} | "
            f"Winner={result.winning_trigger} | "
            f"Reasons="
            f"{'; '.join(result.reasons)}"
        )

    # --------------------------------------------------------
    # DIAGNOSTICS REPORT
    # --------------------------------------------------------

    @staticmethod
    def diagnostics_report(
        result: HybridRebalanceResult,
    ) -> pd.DataFrame:

        rows = []

        engines = [

            (
                "Scheduled",
                result.scheduled_decision,
            ),

            (
                "Drift",
                result.drift_decision,
            ),

            (
                "Turnover",
                result.turnover_decision,
            ),

            (
                "Event",
                result.event_decision,
            ),

        ]

        for (
            name,
            decision,
        ) in engines:

            if decision is None:

                continue

            rows.append({

                "Engine":
                name,

                "Triggered":
                decision.should_rebalance,

                "Score":
                decision.score,

                "Reason":
                decision.reason,

            })

        return pd.DataFrame(
            rows
        )

    # --------------------------------------------------------
    # SIMPLE BOOLEAN API
    # --------------------------------------------------------

    def should_rebalance(
        self,
        **kwargs,
    ) -> bool:

        result = self.decide(
            **kwargs
        )

        return result.should_rebalance
    

# ============================================================
# REBALANCE ANALYTICS & REPORTING LAYER
# ============================================================

@dataclass(slots=True)
class RebalanceAnalytics:
    """
    Institutional rebalance analytics.

    Used by

        Portfolio Managers
        Risk Teams
        Investment Committees
        Audit & Compliance
    """

    total_decisions: int

    rebalance_count: int

    hold_count: int

    rebalance_ratio: float

    average_trigger_score: float

    average_turnover: float

    estimated_cost: float


# ============================================================
# PART 8: REBALANCE REPORTING ENGINE
# ============================================================

class RebalanceReportingEngine:
    """
    Production reporting layer.

    Generates

        Trigger statistics
        Turnover analytics
        Cost analytics
        Governance reports
    """

    def __init__(
        self,
        config: RebalanceConfig,
    ) -> None:

        self.config = config

    # --------------------------------------------------------
    # SUMMARY ANALYTICS
    # --------------------------------------------------------

    def summary(
        self,
        decisions: list[HybridRebalanceResult],
        *,
        turnover_series: pd.Series | None = None,
        cost_series: pd.Series | None = None,
    ) -> RebalanceAnalytics:

        if not decisions:

            return RebalanceAnalytics(
                total_decisions=0,
                rebalance_count=0,
                hold_count=0,
                rebalance_ratio=0.0,
                average_trigger_score=0.0,
                average_turnover=0.0,
                estimated_cost=0.0,
            )

        rebalance_count = sum(
            d.should_rebalance
            for d in decisions
        )

        total = len(decisions)

        hold_count = (
            total - rebalance_count
        )

        avg_score = float(
            np.mean(
                [
                    d.final_score
                    for d in decisions
                ]
            )
        )

        avg_turnover = (

            float(
                turnover_series.mean()
            )

            if turnover_series
            is not None

            and len(turnover_series) > 0

            else 0.0

        )

        avg_cost = (

            float(
                cost_series.mean()
            )

            if cost_series
            is not None

            and len(cost_series) > 0

            else 0.0

        )

        return RebalanceAnalytics(
            total_decisions=total,
            rebalance_count=rebalance_count,
            hold_count=hold_count,
            rebalance_ratio=float(
                rebalance_count
                / total
            ),
            average_trigger_score=avg_score,
            average_turnover=avg_turnover,
            estimated_cost=avg_cost,
        )

    # --------------------------------------------------------
    # TRIGGER ATTRIBUTION
    # --------------------------------------------------------

    def trigger_attribution(
        self,
        decisions: list[
            HybridRebalanceResult
        ],
    ) -> pd.DataFrame:

        if not decisions:

            return pd.DataFrame()

        trigger_counts = {}

        for d in decisions:

            trigger = (
                d.winning_trigger
            )

            trigger_counts[
                trigger
            ] = (

                trigger_counts.get(
                    trigger,
                    0,
                )

                + 1

            )

        report = pd.DataFrame({

            "Trigger":
            list(
                trigger_counts.keys()
            ),

            "Count":
            list(
                trigger_counts.values()
            ),

        })

        report["Pct"] = (

            report["Count"]

            /

            report["Count"].sum()

        )

        return report.sort_values(
            "Count",
            ascending=False,
        )

    # --------------------------------------------------------
    # REBALANCE FREQUENCY
    # --------------------------------------------------------

    def rebalance_frequency(
        self,
        decisions: list[
            HybridRebalanceResult
        ],
    ) -> dict:

        total = len(decisions)

        if total == 0:

            return {}

        rebalance_count = sum(
            d.should_rebalance
            for d in decisions
        )

        return {

            "TotalPeriods":
            total,

            "Rebalances":
            rebalance_count,

            "Holds":
            total
            - rebalance_count,

            "Frequency":
            rebalance_count
            / total,

        }

    # --------------------------------------------------------
    # TURNOVER ANALYTICS
    # --------------------------------------------------------

    @staticmethod
    def turnover_report(
        turnover_series: pd.Series,
    ) -> pd.DataFrame:

        if turnover_series.empty:

            return pd.DataFrame()

        return pd.DataFrame({

            "Metric": [

                "Mean",
                "Median",
                "Max",
                "Min",
                "Std",

            ],

            "Value": [

                turnover_series.mean(),

                turnover_series.median(),

                turnover_series.max(),

                turnover_series.min(),

                turnover_series.std(),

            ],

        })

    # --------------------------------------------------------
    # COST ANALYTICS
    # --------------------------------------------------------

    @staticmethod
    def transaction_cost_report(
        cost_series: pd.Series,
    ) -> pd.DataFrame:

        if cost_series.empty:

            return pd.DataFrame()

        return pd.DataFrame({

            "Metric": [

                "Mean",

                "Median",

                "Max",

                "Min",

                "Total",

            ],

            "Value": [

                cost_series.mean(),

                cost_series.median(),

                cost_series.max(),

                cost_series.min(),

                cost_series.sum(),

            ],

        })

    # --------------------------------------------------------
    # GOVERNANCE REPORT
    # --------------------------------------------------------

    def governance_report(
        self,
        decisions: list[
            HybridRebalanceResult
        ],
    ) -> pd.DataFrame:

        rows = []

        for idx, d in enumerate(
            decisions
        ):

            rows.append({

                "DecisionID":
                idx,

                "Rebalance":
                d.should_rebalance,

                "Score":
                d.final_score,

                "WinningTrigger":
                d.winning_trigger,

                "Reason":
                "; ".join(
                    d.reasons
                ),

            })

        return pd.DataFrame(
            rows
        )

    # --------------------------------------------------------
    # MASTER REPORT PACKAGE
    # --------------------------------------------------------

    def build_report_package(
        self,
        decisions: list[
            HybridRebalanceResult
        ],
        *,
        turnover_series:
            pd.Series | None = None,
        cost_series:
            pd.Series | None = None,
    ) -> dict:

        return {

            "Summary":
            self.summary(
                decisions,
                turnover_series=
                turnover_series,
                cost_series=
                cost_series,
            ),

            "TriggerAttribution":
            self.trigger_attribution(
                decisions
            ),

            "Frequency":
            self.rebalance_frequency(
                decisions
            ),

            "Turnover":
            (
                self.turnover_report(
                    turnover_series
                )
                if turnover_series
                is not None
                else pd.DataFrame()
            ),

            "Costs":
            (
                self.transaction_cost_report(
                    cost_series
                )
                if cost_series
                is not None
                else pd.DataFrame()
            ),

            "Governance":
            self.governance_report(
                decisions
            ),

        }