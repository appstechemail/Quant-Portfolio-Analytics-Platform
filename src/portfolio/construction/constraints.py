# ============================================================
# PART 1: FRAMEWORK & VALIDATION
# ============================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .portfolio_builder import TargetPortfolio


# ============================================================
# NUMERICAL CONSTANTS
# ============================================================

EPSILON: float = 1e-12


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_columns(
    df: pd.DataFrame,
    required: list[str],
) -> None:
    """
    Validate required dataframe columns.
    """

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )


def validate_portfolio(
    portfolio: Portfolio,
) -> None:
    """
    Validate Portfolio object.
    """

    if portfolio is None:

        raise ValueError(
            "Portfolio cannot be None."
        )

    if getattr(
        portfolio,
        "data",
        None,
    ) is None:

        raise ValueError(
            "Portfolio.data missing."
        )

    if portfolio.data.empty:

        raise ValueError(
            "Portfolio is empty."
        )

    required = [
        "Ticker",
        "Position_Weight",
    ]

    validate_columns(
        portfolio.data,
        required,
    )

    weights = (
        portfolio.data[
            "Position_Weight"
        ]
        .astype(float)
    )

    if weights.isna().any():

        raise ValueError(
            "NaN weights detected."
        )

    if np.isinf(
        weights.values
    ).any():

        raise ValueError(
            "Infinite weights detected."
        )


# ============================================================
# ALIGNED WEIGHT HELPER
# ============================================================

def aligned_weights(
    current: Portfolio,
    previous: Portfolio | None,
) -> tuple[pd.Series, pd.Series]:
    """
    Align two portfolios to the same universe.

    Missing positions receive zero weight.
    """

    current_weights = (
        current.weights
        .astype(float)
        .copy()
    )

    if previous is None:

        previous_weights = pd.Series(
            0.0,
            index=current_weights.index,
            dtype=float,
        )

        return (
            current_weights,
            previous_weights,
        )

    previous_weights = (
        previous.weights
        .astype(float)
        .copy()
    )

    universe = (
        current_weights.index
        .union(
            previous_weights.index
        )
    )

    current_weights = (
        current_weights
        .reindex(universe)
        .fillna(0.0)
    )

    previous_weights = (
        previous_weights
        .reindex(universe)
        .fillna(0.0)
    )

    return (
        current_weights,
        previous_weights,
    )


# ============================================================
# CONSTRAINT RESULT
# ============================================================

@dataclass(slots=True)
class ConstraintResult:
    """
    Standard result returned by every constraint.
    """

    name: str

    passed: bool

    metric: Any

    limit: Any

    violations: int = 0

    message: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # ------------------------------------------------------

    @staticmethod
    def success(
        name: str,
        metric: Any = None,
        limit: Any = None,
    ) -> "ConstraintResult":

        return ConstraintResult(
            name=name,
            passed=True,
            metric=metric,
            limit=limit,
            violations=0,
            message="PASS",
        )

    # ------------------------------------------------------

    @staticmethod
    def failure(
        name: str,
        metric: Any,
        limit: Any,
        message: str,
        violations: int = 1,
    ) -> "ConstraintResult":

        return ConstraintResult(
            name=name,
            passed=False,
            metric=metric,
            limit=limit,
            violations=violations,
            message=message,
        )


# ============================================================
# BASE CONSTRAINT
# ============================================================

class BaseConstraint(ABC):
    """
    Root constraint class.

    All constraints inherit from this.
    """

    name: str = "BaseConstraint"

    # ------------------------------------------------------

    @staticmethod
    def upper_violation(
        value: float,
        limit: float,
    ) -> float:

        return max(
            value - limit,
            0.0,
        )

    # ------------------------------------------------------

    @staticmethod
    def lower_violation(
        value: float,
        limit: float,
    ) -> float:

        return max(
            limit - value,
            0.0,
        )

    # ------------------------------------------------------

    @staticmethod
    def validate_columns(
        df: pd.DataFrame,
        required: list[str],
    ) -> None:

        validate_columns(
            df,
            required,
        )

    # ------------------------------------------------------

    @abstractmethod
    def evaluate(
        self,
        portfolio,
    ) -> ConstraintResult:
        """
        Evaluate constraint.
        """
        raise NotImplementedError


# ============================================================
# STATIC CONSTRAINT
# ============================================================

class StaticConstraint(
    BaseConstraint
):
    """
    Uses only current portfolio.
    """

    pass


# ============================================================
# DYNAMIC CONSTRAINT
# ============================================================

class DynamicConstraint(
    BaseConstraint
):
    """
    Uses current + previous portfolio.
    """

    @abstractmethod
    def evaluate(
        self,
        current: Portfolio,
        previous: Portfolio | None,
    ) -> ConstraintResult:
        raise NotImplementedError



# ============================================================
# PART 2: EXPOSURE CONSTRAINTS
# ============================================================

# ============================================================
# GROSS EXPOSURE
# ============================================================

class GrossExposureConstraint(
    StaticConstraint
):
    """
    Gross Exposure Limit

    Gross =
        Long Exposure + Short Exposure
    """

    name = "GrossExposure"

    def __init__(
        self,
        maximum: float,
    ) -> None:

        self.maximum = float(
            maximum
        )

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        gross = float(
            portfolio.gross_exposure
        )

        passed = (
            gross <= self.maximum
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=gross,
                limit=self.maximum,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=gross,
            limit=self.maximum,
            message=(
                f"Gross exposure "
                f"{gross:.4f} > "
                f"{self.maximum:.4f}"
            ),
        )


# ============================================================
# NET EXPOSURE
# ============================================================

class NetExposureConstraint(
    StaticConstraint
):
    """
    Net Exposure Band

    minimum <= net <= maximum
    """

    name = "NetExposure"

    def __init__(
        self,
        minimum: float,
        maximum: float,
    ) -> None:

        self.minimum = float(
            minimum
        )

        self.maximum = float(
            maximum
        )

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        net = float(
            portfolio.net_exposure
        )

        passed = (
            self.minimum
            <= net
            <= self.maximum
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=net,
                limit=(
                    self.minimum,
                    self.maximum,
                ),
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=net,
            limit=(
                self.minimum,
                self.maximum,
            ),
            message=(
                f"Net exposure "
                f"{net:.4f} outside "
                f"[{self.minimum:.4f}, "
                f"{self.maximum:.4f}]"
            ),
        )


# ============================================================
# LONG EXPOSURE
# ============================================================

class LongExposureConstraint(
    StaticConstraint
):
    """
    Maximum Long Exposure
    """

    name = "LongExposure"

    def __init__(
        self,
        maximum: float,
    ) -> None:

        self.maximum = float(
            maximum
        )

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        exposure = float(
            portfolio.long_exposure
        )

        passed = (
            exposure <= self.maximum
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=exposure,
                limit=self.maximum,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=exposure,
            limit=self.maximum,
            message=(
                f"Long exposure "
                f"{exposure:.4f} > "
                f"{self.maximum:.4f}"
            ),
        )


# ============================================================
# SHORT EXPOSURE
# ============================================================

class ShortExposureConstraint(
    StaticConstraint
):
    """
    Maximum Short Exposure

    Long-only portfolios:
        maximum = 0
    """

    name = "ShortExposure"

    def __init__(
        self,
        maximum: float,
    ) -> None:

        self.maximum = float(
            maximum
        )

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        exposure = float(
            portfolio.short_exposure
        )

        passed = (
            exposure <= self.maximum
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=exposure,
                limit=self.maximum,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=exposure,
            limit=self.maximum,
            message=(
                f"Short exposure "
                f"{exposure:.4f} > "
                f"{self.maximum:.4f}"
            ),
        )


# ============================================================
# CASH CONSTRAINT
# ============================================================

class CashConstraint(
    StaticConstraint
):
    """
    Minimum Cash Reserve
    """

    name = "Cash"

    def __init__(
        self,
        minimum_cash: float,
    ) -> None:

        self.minimum_cash = float(
            minimum_cash
        )

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        cash = float(
            max(
                1.0
                - portfolio.total_weight,
                0.0,
            )
        )

        passed = (
            cash >= self.minimum_cash
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=cash,
                limit=self.minimum_cash,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=cash,
            limit=self.minimum_cash,
            message=(
                f"Cash reserve "
                f"{cash:.4f} < "
                f"{self.minimum_cash:.4f}"
            ),
        )


# ============================================================
# LEVERAGE CONSTRAINT
# ============================================================

class LeverageConstraint(
    StaticConstraint
):
    """
    Maximum Leverage

    Currently:

        leverage = gross exposure

    Future:
        derivatives leverage
        margin leverage
    """

    name = "Leverage"

    def __init__(
        self,
        maximum: float,
    ) -> None:

        self.maximum = float(
            maximum
        )

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        leverage = float(
            portfolio.gross_exposure
        )

        passed = (
            leverage <= self.maximum
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=leverage,
                limit=self.maximum,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=leverage,
            limit=self.maximum,
            message=(
                f"Leverage "
                f"{leverage:.4f} > "
                f"{self.maximum:.4f}"
            ),
        )


# ============================================================
# FULLY INVESTED
# ============================================================

class FullyInvestedConstraint(
    StaticConstraint
):
    """
    Portfolio weight must sum
    to target exposure.
    """

    name = "FullyInvested"

    def __init__(
        self,
        target: float = 1.0,
        tolerance: float = 0.01,
    ) -> None:

        self.target = float(
            target
        )

        self.tolerance = float(
            tolerance
        )

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        total = float(
            portfolio.total_weight
        )

        deviation = abs(
            total
            - self.target
        )

        passed = (
            deviation
            <= self.tolerance
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=total,
                limit=self.target,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=total,
            limit=self.target,
            message=(
                f"Portfolio weight "
                f"{total:.4f} not within "
                f"{self.tolerance:.4f} of "
                f"{self.target:.4f}"
            ),
        )


# ============================================================
# PART 3: CONCENTRATION CONSTRAINTS
# ============================================================

# ============================================================
# POSITION LIMIT
# ============================================================

class PositionLimitConstraint(
    StaticConstraint
):
    """
    Maximum allowed weight
    for a single position.
    """

    name = "PositionLimit"

    def __init__(
        self,
        max_weight: float,
    ) -> None:

        self.max_weight = float(
            max_weight
        )

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        weights = (
            portfolio.weights
            .astype(float)
        )

        if weights.empty:

            return ConstraintResult.success(
                self.name
            )

        largest = float(
            weights.max()
        )

        passed = (
            largest
            <= self.max_weight
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=largest,
                limit=self.max_weight,
            )

        offenders = (
            weights[
                weights
                > self.max_weight
            ]
        )

        return ConstraintResult.failure(
            name=self.name,
            metric=largest,
            limit=self.max_weight,
            violations=len(
                offenders
            ),
            message=(
                f"{len(offenders)} "
                f"positions exceed "
                f"{self.max_weight:.4f}"
            ),
        )


# ============================================================
# MINIMUM POSITION SIZE
# ============================================================

class MinimumPositionConstraint(
    StaticConstraint
):
    """
    Prevent tiny positions.

    Useful for reducing
    turnover and costs.
    """

    name = "MinimumPosition"

    def __init__(
        self,
        minimum_weight: float,
    ) -> None:

        self.minimum_weight = float(
            minimum_weight
        )

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        weights = (
            portfolio.weights
            .astype(float)
        )

        positive = (
            weights[
                weights > 0
            ]
        )

        if positive.empty:

            return ConstraintResult.success(
                self.name
            )

        offenders = (
            positive[
                positive
                < self.minimum_weight
            ]
        )

        passed = (
            len(offenders)
            == 0
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=float(
                    positive.min()
                ),
                limit=self.minimum_weight,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=float(
                offenders.min()
            ),
            limit=self.minimum_weight,
            violations=len(
                offenders
            ),
            message=(
                f"{len(offenders)} "
                f"positions below "
                f"minimum weight"
            ),
        )


# ============================================================
# HOLDINGS COUNT
# ============================================================

class HoldingsCountConstraint(
    StaticConstraint
):
    """
    Enforce minimum and maximum
    number of holdings.
    """

    name = "HoldingsCount"

    def __init__(
        self,
        minimum: int,
        maximum: int,
    ) -> None:

        self.minimum = int(
            minimum
        )

        self.maximum = int(
            maximum
        )

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        count = int(
            len(
                portfolio.weights[
                    portfolio.weights > 0
                ]
            )
        )

        passed = (
            self.minimum
            <= count
            <= self.maximum
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=count,
                limit=(
                    self.minimum,
                    self.maximum,
                ),
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=count,
            limit=(
                self.minimum,
                self.maximum,
            ),
            message=(
                f"Holdings count "
                f"{count} outside "
                f"[{self.minimum}, "
                f"{self.maximum}]"
            ),
        )


# ============================================================
# SECTOR EXPOSURE
# ============================================================

class SectorExposureConstraint(
    StaticConstraint
):
    """
    Maximum exposure
    per sector.
    """

    name = "SectorExposure"

    def __init__(
        self,
        max_sector_weight: float,
        sector_column: str = (
            "Sector"
        ),
    ) -> None:

        self.max_sector_weight = (
            float(
                max_sector_weight
            )
        )

        self.sector_column = (
            sector_column
        )

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        if (
            self.sector_column
            not in portfolio.data.columns
        ):

            return ConstraintResult.success(
                self.name
            )

        sector_weights = (
            portfolio.data
            .groupby(
                self.sector_column
            )[
                "Position_Weight"
            ]
            .sum()
        )

        offenders = (
            sector_weights[
                sector_weights
                > self.max_sector_weight
            ]
        )

        passed = (
            len(offenders)
            == 0
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=float(
                    sector_weights.max()
                ),
                limit=self.max_sector_weight,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=float(
                sector_weights.max()
            ),
            limit=self.max_sector_weight,
            violations=len(
                offenders
            ),
            message=(
                f"{len(offenders)} "
                f"sectors exceed "
                f"sector limit"
            ),
        )


# ============================================================
# INDUSTRY EXPOSURE
# ============================================================

class IndustryExposureConstraint(
    StaticConstraint
):
    """
    Maximum exposure
    per industry.
    """

    name = "IndustryExposure"

    def __init__(
        self,
        max_industry_weight: float,
        industry_column: str = (
            "Industry"
        ),
    ) -> None:

        self.max_industry_weight = (
            float(
                max_industry_weight
            )
        )

        self.industry_column = (
            industry_column
        )

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        if (
            self.industry_column
            not in portfolio.data.columns
        ):

            return ConstraintResult.success(
                self.name
            )

        industry_weights = (
            portfolio.data
            .groupby(
                self.industry_column
            )[
                "Position_Weight"
            ]
            .sum()
        )

        offenders = (
            industry_weights[
                industry_weights
                > self.max_industry_weight
            ]
        )

        passed = (
            len(offenders)
            == 0
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=float(
                    industry_weights.max()
                ),
                limit=self.max_industry_weight,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=float(
                industry_weights.max()
            ),
            limit=self.max_industry_weight,
            violations=len(
                offenders
            ),
            message=(
                f"{len(offenders)} "
                f"industries exceed "
                f"industry limit"
            ),
        )


# ============================================================
# HERFINDAHL CONCENTRATION
# ============================================================

class HerfindahlConstraint(
    StaticConstraint
):
    """
    Concentration limit using HHI.

    HHI = sum(weight²)

    Lower is more diversified.
    """

    name = "Herfindahl"

    def __init__(
        self,
        maximum_hhi: float,
    ) -> None:

        self.maximum_hhi = float(
            maximum_hhi
        )

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        weights = (
            portfolio.weights
            .astype(float)
            .values
        )

        hhi = float(
            np.sum(
                weights ** 2
            )
        )

        passed = (
            hhi
            <= self.maximum_hhi
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=hhi,
                limit=self.maximum_hhi,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=hhi,
            limit=self.maximum_hhi,
            message=(
                f"HHI "
                f"{hhi:.6f} > "
                f"{self.maximum_hhi:.6f}"
            ),
        )


# ============================================================
# PART 4: LIQUIDITY & CAPACITY CONSTRAINTS
# ============================================================

# ============================================================
# MINIMUM ADV
# ============================================================

class MinimumADVConstraint(
    StaticConstraint
):
    """
    Minimum Average Daily Volume.

    Securities below ADV threshold
    are not investable.
    """

    name = "MinimumADV"

    def __init__(
        self,
        minimum_adv: float,
        adv_column: str = "ADV",
    ) -> None:

        self.minimum_adv = float(
            minimum_adv
        )

        self.adv_column = adv_column

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        if (
            self.adv_column
            not in portfolio.data.columns
        ):
            return ConstraintResult.success(
                self.name
            )

        adv = (
            portfolio.data[
                self.adv_column
            ]
            .fillna(0)
        )

        offenders = adv[
            adv < self.minimum_adv
        ]

        if offenders.empty:

            return ConstraintResult.success(
                self.name,
                metric=float(
                    adv.min()
                ),
                limit=self.minimum_adv,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=float(
                adv.min()
            ),
            limit=self.minimum_adv,
            violations=len(
                offenders
            ),
            message=(
                f"{len(offenders)} securities "
                f"below ADV threshold"
            ),
        )


# ============================================================
# LIQUIDITY CONSTRAINT
# ============================================================

class LiquidityConstraint(
    StaticConstraint
):
    """
    Maximum portfolio participation
    relative to liquidity.

    Position Weight / ADV
    """

    name = "Liquidity"

    def __init__(
        self,
        max_participation: float,
        adv_column: str = "ADV",
    ) -> None:

        self.max_participation = (
            float(
                max_participation
            )
        )

        self.adv_column = adv_column

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        if (
            self.adv_column
            not in portfolio.data.columns
        ):
            return ConstraintResult.success(
                self.name
            )

        df = portfolio.data.copy()

        adv = (
            df[
                self.adv_column
            ]
            .replace(
                0,
                np.nan,
            )
        )

        participation = (
            df[
                "Position_Weight"
            ].abs()
            /
            adv
        )

        participation = (
            participation.fillna(
                np.inf
            )
        )

        offenders = participation[
            participation
            > self.max_participation
        ]

        if offenders.empty:

            return ConstraintResult.success(
                self.name,
                metric=float(
                    participation.max()
                ),
                limit=self.max_participation,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=float(
                participation.max()
            ),
            limit=self.max_participation,
            violations=len(
                offenders
            ),
            message=(
                f"{len(offenders)} positions "
                f"exceed liquidity participation"
            ),
        )


# ============================================================
# CAPACITY CONSTRAINT
# ============================================================

class CapacityConstraint(
    StaticConstraint
):
    """
    Maximum deployable capital.

    Uses Dollar_Position.
    """

    name = "Capacity"

    def __init__(
        self,
        max_capacity: float,
        dollar_column: str = (
            "Dollar_Position"
        ),
    ) -> None:

        self.max_capacity = float(
            max_capacity
        )

        self.dollar_column = (
            dollar_column
        )

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        if (
            self.dollar_column
            not in portfolio.data.columns
        ):
            return ConstraintResult.success(
                self.name
            )

        deployed = float(
            portfolio.data[
                self.dollar_column
            ].sum()
        )

        passed = (
            deployed
            <= self.max_capacity
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=deployed,
                limit=self.max_capacity,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=deployed,
            limit=self.max_capacity,
            message=(
                f"Capital deployed "
                f"{deployed:,.0f} > "
                f"{self.max_capacity:,.0f}"
            ),
        )


# ============================================================
# ADV PARTICIPATION
# ============================================================

class ADVParticipationConstraint(
    StaticConstraint
):
    """
    Limit participation in daily volume.

    Dollar Position / ADV
    """

    name = "ADVParticipation"

    def __init__(
        self,
        max_participation: float,
        adv_column: str = "ADV",
        dollar_column: str = (
            "Dollar_Position"
        ),
    ) -> None:

        self.max_participation = (
            float(
                max_participation
            )
        )

        self.adv_column = adv_column

        self.dollar_column = (
            dollar_column
        )

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        required = [
            self.adv_column,
            self.dollar_column,
        ]

        missing = [
            c
            for c in required
            if c not in portfolio.data.columns
        ]

        if missing:

            return ConstraintResult.success(
                self.name
            )

        participation = (
            portfolio.data[
                self.dollar_column
            ]
            /
            portfolio.data[
                self.adv_column
            ].replace(
                0,
                np.nan,
            )
        )

        participation = (
            participation.fillna(
                np.inf
            )
        )

        offenders = participation[
            participation
            > self.max_participation
        ]

        if offenders.empty:

            return ConstraintResult.success(
                self.name,
                metric=float(
                    participation.max()
                ),
                limit=self.max_participation,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=float(
                participation.max()
            ),
            limit=self.max_participation,
            violations=len(
                offenders
            ),
            message=(
                f"{len(offenders)} securities "
                f"exceed ADV participation limit"
            ),
        )


# ============================================================
# MARKET IMPACT CONSTRAINT
# ============================================================

class MarketImpactConstraint(
    StaticConstraint
):
    """
    Institutional market impact model.

    Approximation:

        impact =
        Dollar_Position / ADV

    Future:
        Almgren-Chriss
        Square-root impact
        Venue models
    """

    name = "MarketImpact"

    def __init__(
        self,
        max_impact: float,
        adv_column: str = "ADV",
        dollar_column: str = (
            "Dollar_Position"
        ),
    ) -> None:

        self.max_impact = float(
            max_impact
        )

        self.adv_column = adv_column

        self.dollar_column = (
            dollar_column
        )

    def evaluate(
        self,
        portfolio: Portfolio,
    ) -> ConstraintResult:

        required = [
            self.adv_column,
            self.dollar_column,
        ]

        missing = [
            c
            for c in required
            if c not in portfolio.data.columns
        ]

        if missing:

            return ConstraintResult.success(
                self.name
            )

        impact = (
            portfolio.data[
                self.dollar_column
            ]
            /
            portfolio.data[
                self.adv_column
            ].replace(
                0,
                np.nan,
            )
        )

        impact = impact.fillna(
            np.inf
        )

        offenders = impact[
            impact > self.max_impact
        ]

        if offenders.empty:

            return ConstraintResult.success(
                self.name,
                metric=float(
                    impact.max()
                ),
                limit=self.max_impact,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=float(
                impact.max()
            ),
            limit=self.max_impact,
            violations=len(
                offenders
            ),
            message=(
                f"{len(offenders)} securities "
                f"exceed market impact limit"
            ),
        )




# ============================================================
# PART 5: PORTFOLIO STABILITY & TURNOVER CONSTRAINTS
# ============================================================

# ============================================================
# TURNOVER CONSTRAINT
# ============================================================

class TurnoverConstraint(
    DynamicConstraint
):
    """
    One-period portfolio turnover.

    Turnover =
        0.5 * sum(
            abs(
                w_t - w_t-1
            )
        )
    """

    name = "Turnover"

    def __init__(
        self,
        max_turnover: float,
    ) -> None:

        self.max_turnover = float(
            max_turnover
        )

    def evaluate(
        self,
        current: Portfolio,
        previous: Portfolio | None,
    ) -> ConstraintResult:

        curr, prev = (
            aligned_weights(
                current,
                previous,
            )
        )

        turnover = float(
            0.5
            * np.abs(
                curr - prev
            ).sum()
        )

        passed = (
            turnover
            <= self.max_turnover
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=turnover,
                limit=self.max_turnover,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=turnover,
            limit=self.max_turnover,
            message=(
                f"Turnover "
                f"{turnover:.4f} > "
                f"{self.max_turnover:.4f}"
            ),
        )


# ============================================================
# TRADE COUNT CONSTRAINT
# ============================================================

class TradeCountConstraint(
    DynamicConstraint
):
    """
    Limit number of trades.

    Trade occurs when
    weight changes materially.
    """

    name = "TradeCount"

    def __init__(
        self,
        max_trades: int,
        threshold: float = 1e-5,
    ) -> None:

        self.max_trades = int(
            max_trades
        )

        self.threshold = float(
            threshold
        )

    def evaluate(
        self,
        current: Portfolio,
        previous: Portfolio | None,
    ) -> ConstraintResult:

        curr, prev = (
            aligned_weights(
                current,
                previous,
            )
        )

        trades = int(
            (
                np.abs(
                    curr - prev
                )
                > self.threshold
            ).sum()
        )

        passed = (
            trades
            <= self.max_trades
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=trades,
                limit=self.max_trades,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=trades,
            limit=self.max_trades,
            message=(
                f"Trade count "
                f"{trades} > "
                f"{self.max_trades}"
            ),
        )


# ============================================================
# WEIGHT DRIFT CONSTRAINT
# ============================================================

class WeightDriftConstraint(
    DynamicConstraint
):
    """
    Restrict excessive
    portfolio drift.

    Drift =
        max(
            abs(
                w_t - w_t-1
            )
        )
    """

    name = "WeightDrift"

    def __init__(
        self,
        max_drift: float,
    ) -> None:

        self.max_drift = float(
            max_drift
        )

    def evaluate(
        self,
        current: Portfolio,
        previous: Portfolio | None,
    ) -> ConstraintResult:

        curr, prev = (
            aligned_weights(
                current,
                previous,
            )
        )

        drift = float(
            np.abs(
                curr - prev
            ).max()
        )

        passed = (
            drift
            <= self.max_drift
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=drift,
                limit=self.max_drift,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=drift,
            limit=self.max_drift,
            message=(
                f"Weight drift "
                f"{drift:.4f} > "
                f"{self.max_drift:.4f}"
            ),
        )


# ============================================================
# HOLDING OVERLAP CONSTRAINT
# ============================================================

class HoldingOverlapConstraint(
    DynamicConstraint
):
    """
    Maintain minimum overlap
    between consecutive portfolios.

    Overlap =
        |intersection|
        /
        |previous|
    """

    name = "HoldingOverlap"

    def __init__(
        self,
        minimum_overlap: float,
    ) -> None:

        self.minimum_overlap = float(
            minimum_overlap
        )

    def evaluate(
        self,
        current: Portfolio,
        previous: Portfolio | None,
    ) -> ConstraintResult:

        if previous is None:

            return ConstraintResult.success(
                self.name
            )

        current_names = set(
            current.weights[
                current.weights > 0
            ].index
        )

        previous_names = set(
            previous.weights[
                previous.weights > 0
            ].index
        )

        if len(
            previous_names
        ) == 0:

            overlap = 1.0

        else:

            overlap = (
                len(
                    current_names
                    &
                    previous_names
                )
                /
                len(
                    previous_names
                )
            )

        passed = (
            overlap
            >= self.minimum_overlap
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=overlap,
                limit=self.minimum_overlap,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=overlap,
            limit=self.minimum_overlap,
            message=(
                f"Overlap "
                f"{overlap:.4f} < "
                f"{self.minimum_overlap:.4f}"
            ),
        )


# ============================================================
# REBALANCE CONSTRAINT
# ============================================================

class RebalanceConstraint(
    DynamicConstraint
):
    """
    Trigger threshold.

    Prevents unnecessary
    rebalances when changes
    are too small.
    """

    name = "RebalanceTrigger"

    def __init__(
        self,
        min_rebalance_change: float,
    ) -> None:

        self.min_rebalance_change = (
            float(
                min_rebalance_change
            )
        )

    def evaluate(
        self,
        current: Portfolio,
        previous: Portfolio | None,
    ) -> ConstraintResult:

        if previous is None:

            return ConstraintResult.success(
                self.name
            )

        curr, prev = (
            aligned_weights(
                current,
                previous,
            )
        )

        total_change = float(
            np.abs(
                curr - prev
            ).sum()
        )

        passed = (
            total_change
            >= self.min_rebalance_change
        )

        if passed:

            return ConstraintResult.success(
                self.name,
                metric=total_change,
                limit=self.min_rebalance_change,
            )

        return ConstraintResult.failure(
            name=self.name,
            metric=total_change,
            limit=self.min_rebalance_change,
            message=(
                "Portfolio change "
                "below rebalance threshold"
            ),
        )



# ============================================================
# PART 6: CONSTRAINT ENGINE & REPORTING LAYER
# ============================================================

# ============================================================
# CONSTRAINT ENGINE
# ============================================================

class ConstraintEngine:
    """
    Institutional constraint engine.

    Responsibilities
    ----------------
    • Register constraints
    • Execute constraints
    • Aggregate violations
    • Produce diagnostics
    • Support optimizer integration
    """

    def __init__(
        self,
        constraints: Iterable[
            BaseConstraint
        ] | None = None,
    ) -> None:

        self.constraints = list(
            constraints or []
        )

    # --------------------------------------------------------
    # REGISTRATION
    # --------------------------------------------------------

    def add_constraint(
        self,
        constraint: BaseConstraint,
    ) -> None:

        self.constraints.append(
            constraint
        )

    # --------------------------------------------------------

    def remove_constraint(
        self,
        name: str,
    ) -> None:

        self.constraints = [

            c

            for c in self.constraints

            if c.name != name

        ]

    # --------------------------------------------------------

    def clear_constraints(
        self,
    ) -> None:

        self.constraints.clear()

    # --------------------------------------------------------

    def list_constraints(
        self,
    ) -> list[str]:

        return [

            c.name

            for c in self.constraints

        ]

    # ========================================================
    # EXECUTION
    # ========================================================

    def validate(
        self,
        portfolio: Portfolio,
        previous: Portfolio | None = None,
    ) -> list[ConstraintResult]:
        """
        Evaluate all constraints.
        """

        results: list[
            ConstraintResult
        ] = []

        for constraint in (
            self.constraints
        ):

            if isinstance(
                constraint,
                DynamicConstraint,
            ):

                result = (
                    constraint.evaluate(
                        current=portfolio,
                        previous=previous,
                    )
                )

            else:

                result = (
                    constraint.evaluate(
                        portfolio
                    )
                )

            results.append(
                result
            )

        return results

    # ========================================================
    # FEASIBILITY
    # ========================================================

    @staticmethod
    def is_feasible(
        results: Iterable[
            ConstraintResult
        ],
    ) -> bool:

        return all(
            r.passed
            for r in results
        )

    # --------------------------------------------------------

    @staticmethod
    def violations(
        results: Iterable[
            ConstraintResult
        ],
    ) -> list[
        ConstraintResult
    ]:

        return [

            r

            for r in results

            if not r.passed

        ]

    # --------------------------------------------------------

    @staticmethod
    def violation_count(
        results: Iterable[
            ConstraintResult
        ],
    ) -> int:

        return len(

            ConstraintEngine
            .violations(
                results
            )

        )

    # ========================================================
    # OPTIMIZER SUPPORT
    # ========================================================

    @staticmethod
    def total_violation(
        results: Iterable[
            ConstraintResult
        ],
    ) -> float:
        """
        Scalar violation score.

        Used by optimizers.
        """

        return float(

            sum(
                r.violation
                for r in results
            )

        )

    # --------------------------------------------------------

    @staticmethod
    def penalty_score(
        results: Iterable[
            ConstraintResult
        ],
    ) -> float:
        """
        Heavier penalty for failures.
        """

        score = 0.0

        for r in results:

            score += (
                r.violation
                ** 2
            )

        return float(score)

    # ========================================================
    # REPORTING
    # ========================================================

    @staticmethod
    def summary(
        results: Iterable[
            ConstraintResult
        ],
    ) -> pd.DataFrame:
        """
        Human-readable report.
        """

        rows = []

        for r in results:

            rows.append({

                "Constraint":
                    r.name,

                "Passed":
                    r.passed,

                "Current":
                    r.current,

                "Limit":
                    r.limit,

                "Violation":
                    r.violation,

            })

        report = (
            pd.DataFrame(
                rows
            )
        )

        if report.empty:

            return report

        return report.sort_values(
            [
                "Passed",
                "Violation",
            ],
            ascending=[
                True,
                False,
            ],
        )

    # --------------------------------------------------------

    @staticmethod
    def diagnostics(
        results: Iterable[
            ConstraintResult
        ],
    ) -> dict:
        """
        Machine-readable diagnostics.
        """

        results = list(
            results
        )

        failed = [

            r.name

            for r in results

            if not r.passed

        ]

        return {

            "passed":
                ConstraintEngine
                .is_feasible(
                    results
                ),

            "constraint_count":
                len(results),

            "failed_constraints":
                failed,

            "failed_count":
                len(failed),

            "total_violation":
                ConstraintEngine
                .total_violation(
                    results
                ),

            "penalty_score":
                ConstraintEngine
                .penalty_score(
                    results
                ),

        }

    # --------------------------------------------------------

    @staticmethod
    def print_report(
        results: Iterable[
            ConstraintResult
        ],
    ) -> None:

        report = (
            ConstraintEngine
            .summary(
                results
            )
        )

        if report.empty:

            print(
                "No constraints evaluated."
            )

            return

        print(
            "\n"
            + "=" * 70
        )

        print(
            "CONSTRAINT REPORT"
        )

        print(
            "=" * 70
        )

        print(
            report.to_string(
                index=False
            )
        )

        print(
            "=" * 70
        )