
"""
==============================================================
OBJECTIVES
Production Portfolio Construction Engine
==============================================================

File
----
objectives.py

Purpose
-------
Defines the objective-function framework used by the portfolio
optimizer.

This module is intentionally solver-independent.

Responsibilities
----------------
• Objective abstraction
• Objective registry
• Objective validation
• Shared utilities

This module DOES NOT know anything about

    PortfolioOptimizer
    Solver backend
    CVXPY
    SciPy

It only evaluates an objective given

    weights
    data

==============================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Type

import numpy as np
import pandas as pd

# ==============================================================
# OBJECTIVE TYPES
# ==============================================================


class ObjectiveType(str, Enum):
    """
    Supported optimization objectives.
    """

    EXPECTED_RETURN = "expected_return"

    COMPOSITE_SCORE = "composite_score"

    ALPHA_SCORE = "alpha_score"

    MINIMUM_VARIANCE = "minimum_variance"

    TRACKING_ERROR = "tracking_error"

    RISK_ADJUSTED = "risk_adjusted"

    UTILITY = "utility"

    EQUAL_WEIGHT = "equal_weight"

    DIVERSIFICATION = "diversification"

    CONCENTRATION = "concentration"


# ==============================================================
# OBJECTIVE RESULT
# ==============================================================


@dataclass(slots=True)
class ObjectiveResult:
    """
    Objective evaluation result.
    """

    objective_value: float

    objective_name: str


# ==============================================================
# BASE OBJECTIVE
# ==============================================================


class BaseObjective(ABC):
    """
    Base class for every optimization objective.

    Each objective returns ONE scalar.

    Higher value is assumed to be better.

    Solvers that minimize can simply negate
    the returned value.
    """

    objective_type: ObjectiveType

    def __call__(
        self,
        weights: np.ndarray,
        data: pd.DataFrame,
    ) -> float:

        weights = self.validate_weights(weights)

        self.validate_data(data)

        value = self.evaluate(
            weights=weights,
            data=data,
        )

        return float(value)

    # ----------------------------------------------------------

    @abstractmethod
    def evaluate(
        self,
        *,
        weights: np.ndarray,
        data: pd.DataFrame,
    ) -> float:
        """
        Evaluate objective.

        Returns
        -------
        float
        """

    # ----------------------------------------------------------

    @staticmethod
    def validate_weights(
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Validate optimization weights.
        """

        weights = np.asarray(
            weights,
            dtype=float,
        )

        if weights.ndim != 1:

            raise ValueError(
                "Weights must be one-dimensional."
            )

        if np.isnan(weights).any():

            raise ValueError(
                "Weights contain NaN."
            )

        if np.isinf(weights).any():

            raise ValueError(
                "Weights contain Inf."
            )

        return weights

    # ----------------------------------------------------------

    @staticmethod
    def validate_data(
        data: pd.DataFrame,
    ) -> None:
        """
        Validate optimization dataframe.
        """

        if data.empty:

            raise ValueError(
                "Objective received empty dataframe."
            )

    # ----------------------------------------------------------

    @staticmethod
    def normalize_weights(
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize weights to sum to one.
        """

        total = weights.sum()

        if total <= 0:

            return weights

        return weights / total

    # ----------------------------------------------------------

    @staticmethod
    def weighted_sum(
        values: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Stable weighted summation.
        """

        return float(
            np.dot(
                values,
                weights,
            )
        )


# ==============================================================
# OBJECTIVE REGISTRY
# ==============================================================


OBJECTIVE_REGISTRY: Dict[
    ObjectiveType,
    Type[BaseObjective],
] = {}


# ==============================================================
# REGISTRATION DECORATOR
# ==============================================================


def register_objective(
    objective_type: ObjectiveType,
):
    """
    Register an optimization objective.
    """

    def decorator(
        cls: Type[BaseObjective],
    ):

        if objective_type in OBJECTIVE_REGISTRY:

            raise ValueError(
                f"Objective '{objective_type}' "
                "already registered."
            )

        cls.objective_type = objective_type

        OBJECTIVE_REGISTRY[
            objective_type
        ] = cls

        return cls

    return decorator


# ==============================================================
# FACTORY
# ==============================================================


def build_objective(
    objective_type: ObjectiveType,
    **kwargs,
) -> BaseObjective:
    """
    Build objective from registry.
    """

    if objective_type not in OBJECTIVE_REGISTRY:

        raise ValueError(
            f"Unknown objective: {objective_type}"
        )

    cls = OBJECTIVE_REGISTRY[
        objective_type
    ]

    return cls(**kwargs)


# ==============================================================
# MODULE EXPORTS
# ==============================================================

__all__ = [
    "ObjectiveType",
    "ObjectiveResult",
    "BaseObjective",
    "OBJECTIVE_REGISTRY",
    "register_objective",
    "build_objective",
]


# ==============================================================
# RETURN OBJECTIVES
# ==============================================================


@register_objective(ObjectiveType.EXPECTED_RETURN,)
class ExpectedReturnObjective(
    BaseObjective,
):
    """
    Maximize expected return.

    Required Columns
    ----------------
    Expected_Return
    """

    COLUMN = "Expected_Return"

    def evaluate(
        self,
        *,
        weights: np.ndarray,
        data: pd.DataFrame,
    ) -> float:

        if self.COLUMN not in data.columns:
            raise KeyError(
                f"Missing required column '{self.COLUMN}'."
            )

        values = (
            pd.to_numeric(
                data[self.COLUMN],
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        weights = self.normalize_weights(weights)

        return self.weighted_sum(
            values,
            weights,
        )


# ==============================================================

@register_objective(ObjectiveType.COMPOSITE_SCORE,)
class CompositeScoreObjective(
    BaseObjective,
):
    """
    Maximize Composite Score.

    Required Columns
    ----------------
    Composite_Score
    """

    COLUMN = "Composite_Score"

    def evaluate(
        self,
        *,
        weights: np.ndarray,
        data: pd.DataFrame,
    ) -> float:

        if self.COLUMN not in data.columns:
            raise KeyError(
                f"Missing required column '{self.COLUMN}'."
            )

        values = (
            pd.to_numeric(
                data[self.COLUMN],
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        weights = self.normalize_weights(weights)

        return self.weighted_sum(
            values,
            weights,
        )


# ==============================================================

@register_objective(ObjectiveType.ALPHA_SCORE,)
class AlphaScoreObjective(
    BaseObjective,
):
    """
    Maximize Alpha Score.

    Required Columns
    ----------------
    Alpha_Score
    """

    COLUMN = "Alpha_Score"

    def evaluate(
        self,
        *,
        weights: np.ndarray,
        data: pd.DataFrame,
    ) -> float:

        if self.COLUMN not in data.columns:
            raise KeyError(
                f"Missing required column '{self.COLUMN}'."
            )

        values = (
            pd.to_numeric(
                data[self.COLUMN],
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        weights = self.normalize_weights(weights)

        return self.weighted_sum(
            values,
            weights,
        )


# ==============================================================
# OBJECTIVE UTILITIES
# ==============================================================


def available_objectives():
    """
    Return registered objective types.
    """

    return list(
        OBJECTIVE_REGISTRY.keys()
    )


def objective_exists(
    objective_type: ObjectiveType,
) -> bool:
    """
    Check whether an objective is registered.
    """

    return (
        objective_type
        in OBJECTIVE_REGISTRY
    )


# ==============================================================
# SELF TEST
# ==============================================================

if __name__ == "__main__":

    df = pd.DataFrame(
        {
            "Expected_Return": [0.10, 0.08, 0.15],
            "Composite_Score": [1.2, 0.8, 1.5],
            "Alpha_Score": [0.6, 0.4, 0.9],
        }
    )

    w = np.array(
        [0.30, 0.40, 0.30]
    )

    obj = ExpectedReturnObjective()

    print(
        obj(
            weights=w,
            data=df,
        )
    )

    print(
        available_objectives()
    )


# ==============================================================
# RISK OBJECTIVES
# ==============================================================

# --------------------------------------------------------------
# Minimum Variance
# --------------------------------------------------------------


@register_objective(
    ObjectiveType.MINIMUM_VARIANCE,
)
class MinimumVarianceObjective(
    BaseObjective,
):
    """
    Minimize portfolio variance.

    Since every objective in this framework is maximized,
    this objective returns NEGATIVE variance.

    Required
    --------
    Covariance matrix supplied during construction.
    """

    def __init__(
        self,
        covariance_matrix: np.ndarray,
    ) -> None:

        covariance_matrix = np.asarray(
            covariance_matrix,
            dtype=float,
        )

        if covariance_matrix.ndim != 2:

            raise ValueError(
                "Covariance matrix must be 2-dimensional."
            )

        self.covariance = covariance_matrix

    def evaluate(
        self,
        *,
        weights: np.ndarray,
        data: pd.DataFrame,
    ) -> float:

        weights = self.normalize_weights(weights)

        variance = (
            weights.T
            @ self.covariance
            @ weights
        )

        return -float(variance)


# --------------------------------------------------------------
# Tracking Error
# --------------------------------------------------------------


@register_objective(
    ObjectiveType.TRACKING_ERROR,
)
class TrackingErrorObjective(
    BaseObjective,
):
    """
    Minimize deviation from benchmark.

    Returns

        -Tracking Error²

    Required
    --------
    benchmark_weights
    covariance_matrix
    """

    def __init__(
        self,
        benchmark_weights: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> None:

        self.benchmark = np.asarray(
            benchmark_weights,
            dtype=float,
        )

        self.covariance = np.asarray(
            covariance_matrix,
            dtype=float,
        )

    def evaluate(
        self,
        *,
        weights: np.ndarray,
        data: pd.DataFrame,
    ) -> float:

        weights = self.normalize_weights(weights)

        diff = (
            weights
            - self.benchmark
        )

        tracking_error = (
            diff.T
            @ self.covariance
            @ diff
        )

        return -float(tracking_error)


# --------------------------------------------------------------
# Risk Adjusted
# --------------------------------------------------------------


@register_objective(
    ObjectiveType.RISK_ADJUSTED,
)
class RiskAdjustedObjective(
    BaseObjective,
):
    """
    Maximize

        Return
        -
        λ × Variance

    Required Columns
    ----------------
    Expected_Return
    """

    COLUMN = "Expected_Return"

    def __init__(
        self,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0,
    ) -> None:

        self.covariance = np.asarray(
            covariance_matrix,
            dtype=float,
        )

        self.risk_aversion = float(
            risk_aversion
        )

    def evaluate(
        self,
        *,
        weights: np.ndarray,
        data: pd.DataFrame,
    ) -> float:

        values = (
            pd.to_numeric(
                data[self.COLUMN],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy()
        )

        weights = self.normalize_weights(
            weights
        )

        expected_return = (
            weights
            @ values
        )

        variance = (
            weights.T
            @ self.covariance
            @ weights
        )

        utility = (
            expected_return
            -
            self.risk_aversion
            * variance
        )

        return float(utility)


# --------------------------------------------------------------
# Utility Objective
# --------------------------------------------------------------


@register_objective(
    ObjectiveType.UTILITY,
)
class UtilityObjective(
    BaseObjective,
):
    """
    Generic utility function.

    Maximizes

        Score
        -
        λ × Risk

    Required Columns
    ----------------
    Composite_Score
    """

    COLUMN = "Composite_Score"

    def __init__(
        self,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 2.0,
    ) -> None:

        self.covariance = np.asarray(
            covariance_matrix,
            dtype=float,
        )

        self.risk_aversion = float(
            risk_aversion
        )

    def evaluate(
        self,
        *,
        weights: np.ndarray,
        data: pd.DataFrame,
    ) -> float:

        scores = (
            pd.to_numeric(
                data[self.COLUMN],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy()
        )

        weights = self.normalize_weights(
            weights
        )

        reward = (
            weights
            @ scores
        )

        risk = (
            weights.T
            @ self.covariance
            @ weights
        )

        utility = (
            reward
            -
            self.risk_aversion
            * risk
        )

        return float(utility)


# ==============================================================
# RISK HELPERS
# ==============================================================


def portfolio_variance(
    weights: np.ndarray,
    covariance_matrix: np.ndarray,
) -> float:
    """
    Compute portfolio variance.
    """

    weights = np.asarray(
        weights,
        dtype=float,
    )

    covariance_matrix = np.asarray(
        covariance_matrix,
        dtype=float,
    )

    return float(
        weights.T
        @ covariance_matrix
        @ weights
    )


def portfolio_volatility(
    weights: np.ndarray,
    covariance_matrix: np.ndarray,
) -> float:
    """
    Portfolio standard deviation.
    """

    variance = portfolio_variance(
        weights,
        covariance_matrix,
    )

    return float(
        np.sqrt(
            max(
                variance,
                0.0,
            )
        )
    )


# ==============================================================
# PORTFOLIO STRUCTURE OBJECTIVES
# ==============================================================

# --------------------------------------------------------------
# Equal Weight Objective
# --------------------------------------------------------------


@register_objective(
    ObjectiveType.EQUAL_WEIGHT,
)
class EqualWeightObjective(
    BaseObjective,
):
    """
    Encourage equal-weight portfolios.

    Maximizes

        -Σ(w - w_equal)^2
    """

    def evaluate(
        self,
        *,
        weights: np.ndarray,
        data: pd.DataFrame,
    ) -> float:

        weights = self.normalize_weights(weights)

        n_assets = len(weights)

        if n_assets == 0:
            return 0.0

        equal_weights = np.full(
            n_assets,
            1.0 / n_assets,
            dtype=float,
        )

        deviation = np.sum(
            (weights - equal_weights) ** 2
        )

        return -float(deviation)


# --------------------------------------------------------------
# Diversification Objective
# --------------------------------------------------------------


@register_objective(
    ObjectiveType.DIVERSIFICATION,
)
class DiversificationObjective(
    BaseObjective,
):
    """
    Encourage diversification.

    Uses inverse Herfindahl index.

        H = Σ(w²)

    Maximizes

        1 / H
    """

    def evaluate(
        self,
        *,
        weights: np.ndarray,
        data: pd.DataFrame,
    ) -> float:

        weights = self.normalize_weights(weights)

        herfindahl = np.sum(
            np.square(weights)
        )

        if herfindahl <= 0:
            return 0.0

        return float(
            1.0 / herfindahl
        )


# --------------------------------------------------------------
# Concentration Penalty
# --------------------------------------------------------------


@register_objective(
    ObjectiveType.CONCENTRATION,
)
class ConcentrationPenaltyObjective(
    BaseObjective,
):
    """
    Penalize concentrated portfolios.

    Maximizes

        -Σ(w²)
    """

    def evaluate(
        self,
        *,
        weights: np.ndarray,
        data: pd.DataFrame,
    ) -> float:

        weights = self.normalize_weights(weights)

        concentration = np.sum(
            np.square(weights)
        )

        return -float(concentration)


# --------------------------------------------------------------
# Sparsity Penalty
# --------------------------------------------------------------


class SparsityPenaltyObjective(
    BaseObjective,
):
    """
    Encourage fewer active positions.

    Maximizes

        -L1 norm

    Larger lambda
        -> fewer positions
    """

    def __init__(
        self,
        penalty: float = 0.01,
    ) -> None:

        self.penalty = float(
            penalty
        )

    def evaluate(
        self,
        *,
        weights: np.ndarray,
        data: pd.DataFrame,
    ) -> float:

        weights = self.normalize_weights(
            weights
        )

        l1_norm = np.sum(
            np.abs(weights)
        )

        return float(
            -self.penalty * l1_norm
        )


# ==============================================================
# STRUCTURE HELPERS
# ==============================================================


def herfindahl_index(
    weights: np.ndarray,
) -> float:
    """
    Portfolio concentration.

    Returns

        Σ(w²)
    """

    weights = np.asarray(
        weights,
        dtype=float,
    )

    return float(
        np.sum(
            np.square(weights)
        )
    )


def effective_number_of_positions(
    weights: np.ndarray,
) -> float:
    """
    Effective number of holdings.

        ENP = 1 / H
    """

    h = herfindahl_index(
        weights
    )

    if h <= 0:
        return 0.0

    return float(
        1.0 / h
    )


def largest_position(
    weights: np.ndarray,
) -> float:
    """
    Largest portfolio weight.
    """

    weights = np.asarray(
        weights,
        dtype=float,
    )

    if weights.size == 0:
        return 0.0

    return float(
        np.max(weights)
    )


def active_positions(
    weights: np.ndarray,
    threshold: float = 1e-6,
) -> int:
    """
    Number of active positions.
    """

    weights = np.asarray(
        weights,
        dtype=float,
    )

    return int(
        np.sum(
            weights > threshold
        )
    )


# ==============================================================
# OBJECTIVE ENGINE
# ==============================================================

from dataclasses import dataclass, field
from typing import Iterable


# ==============================================================
# OBJECTIVE COMPONENT
# ==============================================================


@dataclass(slots=True)
class ObjectiveComponent:
    """
    One component of a composite objective.

    Example

        0.60 × Expected Return

        -0.25 × Variance

        +0.15 × Diversification
    """

    objective: BaseObjective

    weight: float = 1.0

    enabled: bool = True

    name: str | None = None


# ==============================================================
# COMPOSITE OBJECTIVE
# ==============================================================


class CompositeObjective(
    BaseObjective,
):
    """
    Production-grade objective engine.

    Computes

        Σ weight_i × objective_i

    All objectives remain completely independent.
    """

    def __init__(
        self,
        objectives: Iterable[
            ObjectiveComponent
        ],
    ) -> None:

        self.components = list(objectives)

        if len(self.components) == 0:

            raise ValueError(
                "Composite objective requires "
                "at least one objective."
            )

    # ----------------------------------------------------------

    def evaluate(
        self,
        *,
        weights: np.ndarray,
        data: pd.DataFrame,
    ) -> float:

        total = 0.0

        for component in self.components:

            if not component.enabled:
                continue

            score = component.objective(
                weights,
                data,
            )

            total += (
                component.weight
                * score
            )

        return float(total)


# ==============================================================
# OBJECTIVE REPORT
# ==============================================================


@dataclass(slots=True)
class ObjectiveReport:
    """
    Individual objective contributions.
    """

    total: float

    contributions: dict[str, float]


# ==============================================================
# REPORTING ENGINE
# ==============================================================


class ObjectiveEvaluator:
    """
    Evaluate every objective independently.

    Useful for diagnostics and optimizer debugging.
    """

    def __init__(
        self,
        objective: CompositeObjective,
    ) -> None:

        self.objective = objective

    # ----------------------------------------------------------

    def evaluate(
        self,
        *,
        weights: np.ndarray,
        data: pd.DataFrame,
    ) -> ObjectiveReport:

        contributions = {}

        total = 0.0

        for component in self.objective.components:

            if not component.enabled:
                continue

            value = component.objective(
                weights,
                data,
            )

            weighted = (
                component.weight
                * value
            )

            name = (
                component.name
                or component.objective.__class__.__name__
            )

            contributions[name] = weighted

            total += weighted

        return ObjectiveReport(
            total=float(total),
            contributions=contributions,
        )


# ==============================================================
# DEFAULT FACTORIES
# ==============================================================


def build_return_objective():

    return CompositeObjective(

        [

            ObjectiveComponent(

                objective=ExpectedReturnObjective(),

                weight=1.0,

            )

        ]

    )


def build_score_objective():

    return CompositeObjective(

        [

            ObjectiveComponent(

                objective=CompositeScoreObjective(),

                weight=1.0,

            )

        ]

    )


def build_balanced_objective(

    covariance_matrix: np.ndarray,

    risk_aversion: float = 1.0,

):

    """
    Typical production objective.

    Max Return

    Min Variance

    Encourage Diversification
    """

    return CompositeObjective(

        [

            ObjectiveComponent(

                objective=ExpectedReturnObjective(),

                weight=0.60,

                name="Return",

            ),

            ObjectiveComponent(

                objective=MinimumVarianceObjective(

                    covariance_matrix,

                ),

                weight=0.25,

                name="Variance",

            ),

            ObjectiveComponent(

                objective=DiversificationObjective(),

                weight=0.15,

                name="Diversification",

            ),

        ]

    )


# ==============================================================
# MODULE EXPORTS
# ==============================================================

__all__.extend(

    [
        "ObjectiveComponent",
        "CompositeObjective",
        "ObjectiveEvaluator",
        "ObjectiveReport",
        "build_return_objective",
        "build_score_objective",
        "build_balanced_objective",
    ]

)



