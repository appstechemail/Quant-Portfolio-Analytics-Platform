"""
==============================================================
PORTFOLIO OPTIMIZER
Institutional Grade Quant Platform
==============================================================

Purpose
-------
Convert portfolio candidates into optimized allocations.

Supported Optimizers
--------------------
• Mean Variance
• Minimum Variance
• Maximum Sharpe
• Risk Parity
• Black Litterman
• Custom Objective

Design Goals
------------
• Solver agnostic
• Constraint aware
• Risk model aware
• Transaction-cost aware
• Rebalance aware
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import logging
import numpy as np
import pandas as pd

from .portfolio_builder import TargetPortfolio
from .constraints import ConstraintEngine
from .risk_model import RiskModelResult, BaseRiskModel

logger = logging.getLogger(__name__)

EPSILON: float = 1e-12


# ============================================================
# OPTIMIZATION METHODS
# ============================================================

class OptimizationMethod(str, Enum):

    MEAN_VARIANCE = "mean_variance"

    MIN_VARIANCE = "min_variance"

    MAX_SHARPE = "max_sharpe"

    RISK_PARITY = "risk_parity"

    BLACK_LITTERMAN = "black_litterman"

    CUSTOM = "custom"


# ============================================================
# OPTIMIZER CONFIG
# ============================================================

@dataclass(slots=True)
class OptimizerConfig:

    method: OptimizationMethod = (
        OptimizationMethod.MEAN_VARIANCE
    )

    max_iterations: int = 500

    tolerance: float = 1e-8

    risk_aversion: float = 1.0

    turnover_penalty: float = 0.0

    transaction_cost_penalty: float = 0.0

    min_weight: float = 0.0

    max_weight: float = 0.10

    long_only: bool = True


# ============================================================
# OPTIMIZATION INPUTS
# ============================================================

@dataclass(slots=True)
class OptimizationInputs:

    expected_returns: pd.Series

    covariance_matrix: pd.DataFrame

    current_weights: pd.Series | None = None

    benchmark_weights: pd.Series | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# OPTIMIZATION RESULT
# ============================================================

@dataclass(slots=True)
class OptimizationResult:

    portfolio: Portfolio

    success: bool

    objective_value: float

    iterations: int

    message: str

    weights: pd.Series

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# BASE SOLVER
# ============================================================

class BaseSolverBackend(ABC):

    @abstractmethod
    def solve(
        self,
        inputs: OptimizationInputs,
        config: OptimizerConfig,
    ) -> tuple[np.ndarray, float]:
        """
        Returns

        optimized_weights,
        objective_value
        """
        raise NotImplementedError


# ============================================================
# BASE OPTIMIZER
# ============================================================

class BaseOptimizer(ABC):

    def __init__(
        self,
        constraint_engine: ConstraintEngine,
        config: OptimizerConfig,
    ) -> None:

        self.constraint_engine = (
            constraint_engine
        )

        self.config = config

    @abstractmethod
    def optimize(
        self,
        portfolio: Portfolio,
        optimization_inputs: OptimizationInputs,
        risk_result: RiskModelResult | None = None,
    ) -> OptimizationResult:

        raise NotImplementedError


# ============================================================
# VALIDATION
# ============================================================

def validate_optimization_inputs(
    inputs: OptimizationInputs,
) -> None:

    if inputs.expected_returns.empty:

        raise ValueError(
            "Expected returns empty."
        )

    if inputs.covariance_matrix.empty:

        raise ValueError(
            "Covariance matrix empty."
        )

    if len(
        inputs.expected_returns
    ) != len(
        inputs.covariance_matrix
    ):

        raise ValueError(
            "Return vector and covariance matrix mismatch."
        )

    if np.isnan(
        inputs.expected_returns
    ).any():

        raise ValueError(
            "NaN expected returns detected."
        )

    if np.isnan(
        inputs.covariance_matrix.values
    ).any():

        raise ValueError(
            "NaN covariance matrix detected."
        )
    
# ============================================================
# PART 2: OPTIMIZATION UNIVERSE PREPARATION
# ============================================================

def build_optimization_universe(
    portfolio: Portfolio,
    inputs: OptimizationInputs,
) -> pd.Index:
    """
    Construct common optimization universe.

    Universe must exist in

    • portfolio
    • expected returns
    • covariance matrix
    """

    portfolio_universe = pd.Index(
        portfolio.weights.index
    )

    returns_universe = pd.Index(
        inputs.expected_returns.index
    )

    covariance_universe = pd.Index(
        inputs.covariance_matrix.index
    )

    universe = (
        portfolio_universe
        .intersection(
            returns_universe
        )
        .intersection(
            covariance_universe
        )
    )

    if len(universe) == 0:

        raise ValueError(
            "Optimization universe empty."
        )

    return universe


# ============================================================
# EXPECTED RETURN VECTOR
# ============================================================

def prepare_expected_returns(
    inputs: OptimizationInputs,
    universe: pd.Index,
) -> pd.Series:
    """
    Align expected returns
    to optimization universe.
    """

    expected_returns = (
        inputs.expected_returns
        .reindex(universe)
        .fillna(0.0)
    )

    return expected_returns


# ============================================================
# COVARIANCE MATRIX
# ============================================================

def prepare_covariance_matrix(
    inputs: OptimizationInputs,
    universe: pd.Index,
) -> pd.DataFrame:
    """
    Align covariance matrix
    to optimization universe.
    """

    covariance = (
        inputs.covariance_matrix
        .reindex(
            index=universe,
            columns=universe,
        )
        .fillna(0.0)
    )

    return covariance


# ============================================================
# CURRENT WEIGHTS
# ============================================================

def prepare_current_weights(
    portfolio: Portfolio,
    universe: pd.Index,
) -> pd.Series:
    """
    Align portfolio weights.
    """

    weights = (
        portfolio.weights
        .reindex(universe)
        .fillna(0.0)
        .astype(float)
    )

    return weights


# ============================================================
# BENCHMARK WEIGHTS
# ============================================================

def prepare_benchmark_weights(
    inputs: OptimizationInputs,
    universe: pd.Index,
) -> pd.Series:
    """
    Benchmark alignment.

    If benchmark absent,
    return zero vector.
    """

    if (
        inputs.benchmark_weights
        is None
    ):

        return pd.Series(
            0.0,
            index=universe,
        )

    benchmark = (
        inputs.benchmark_weights
        .reindex(universe)
        .fillna(0.0)
    )

    return benchmark


# ============================================================
# OPTIMIZATION DATASET
# ============================================================

@dataclass(slots=True)
class OptimizationDataset:
    """
    Fully aligned optimization inputs.
    """

    universe: pd.Index

    expected_returns: pd.Series

    covariance_matrix: pd.DataFrame

    current_weights: pd.Series

    benchmark_weights: pd.Series


# ============================================================
# BUILD DATASET
# ============================================================

def build_optimization_dataset(
    portfolio: Portfolio,
    inputs: OptimizationInputs,
) -> OptimizationDataset:
    """
    Create fully aligned
    optimization dataset.
    """

    universe = (
        build_optimization_universe(
            portfolio,
            inputs,
        )
    )

    expected_returns = (
        prepare_expected_returns(
            inputs,
            universe,
        )
    )

    covariance_matrix = (
        prepare_covariance_matrix(
            inputs,
            universe,
        )
    )

    current_weights = (
        prepare_current_weights(
            portfolio,
            universe,
        )
    )

    benchmark_weights = (
        prepare_benchmark_weights(
            inputs,
            universe,
        )
    )

    return OptimizationDataset(
        universe=universe,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        current_weights=current_weights,
        benchmark_weights=benchmark_weights,
    )


# ============================================================
# PORTFOLIO RECONSTRUCTION
# ============================================================

def portfolio_from_weights(
    portfolio: Portfolio,
    weights: pd.Series,
) -> Portfolio:
    """
    Rebuild Portfolio object
    using optimized weights.
    """

    new_portfolio = (
        portfolio.copy()
    )

    new_portfolio.data.loc[
        weights.index,
        "Position_Weight",
    ] = weights.values

    return new_portfolio


# ============================================================
# FEASIBILITY CHECK
# ============================================================

def validate_dataset(
    dataset: OptimizationDataset,
) -> None:
    """
    Validate aligned inputs.
    """

    n = len(
        dataset.universe
    )

    if n == 0:

        raise ValueError(
            "Dataset universe empty."
        )

    if len(
        dataset.expected_returns
    ) != n:

        raise ValueError(
            "Expected returns mismatch."
        )

    if (
        dataset.covariance_matrix.shape
        != (n, n)
    ):

        raise ValueError(
            "Covariance shape mismatch."
        )

    if np.isnan(
        dataset.expected_returns
    ).any():

        raise ValueError(
            "NaN expected returns."
        )

    if np.isnan(
        dataset.covariance_matrix.values
    ).any():

        raise ValueError(
            "NaN covariance matrix."
        )
    
# ============================================================
# PART 3: OBJECTIVE FUNCTION FRAMEWORK
# ============================================================

class BaseObjectiveFunction(ABC):
    """
    Abstract optimization objective.

    Every optimizer objective returns
    a scalar score.

    Optimizer minimizes score.
    """

    @abstractmethod
    def evaluate(
        self,
        weights: np.ndarray,
        dataset: OptimizationDataset,
    ) -> float:
        raise NotImplementedError


# ============================================================
# PORTFOLIO STATISTICS
# ============================================================

def portfolio_expected_return(
    weights: np.ndarray,
    expected_returns: pd.Series,
) -> float:
    """
    Portfolio expected return.
    """

    return float(
        np.dot(
            weights,
            expected_returns.values,
        )
    )


def portfolio_variance(
    weights: np.ndarray,
    covariance_matrix: pd.DataFrame,
) -> float:
    """
    Portfolio variance.
    """

    covariance = (
        covariance_matrix.values
    )

    return float(
        weights.T
        @ covariance
        @ weights
    )


def portfolio_volatility(
    weights: np.ndarray,
    covariance_matrix: pd.DataFrame,
) -> float:
    """
    Portfolio volatility.
    """

    return float(
        np.sqrt(
            max(
                portfolio_variance(
                    weights,
                    covariance_matrix,
                ),
                0.0,
            )
        )
    )


def tracking_error(
    weights: np.ndarray,
    benchmark_weights: np.ndarray,
    covariance_matrix: pd.DataFrame,
) -> float:
    """
    Tracking error volatility.
    """

    active = (
        weights
        - benchmark_weights
    )

    covariance = (
        covariance_matrix.values
    )

    return float(
        np.sqrt(
            max(
                active.T
                @ covariance
                @ active,
                0.0,
            )
        )
    )


# ============================================================
# MEAN VARIANCE OBJECTIVE
# ============================================================

class MeanVarianceObjective(
    BaseObjectiveFunction
):
    """
    Maximize

        Return - λ * Variance

    Optimizer minimizes,
    so we return negative utility.
    """

    def __init__(
        self,
        risk_aversion: float = 5.0,
    ):
        self.risk_aversion = (
            risk_aversion
        )

    def evaluate(
        self,
        weights: np.ndarray,
        dataset: OptimizationDataset,
    ) -> float:

        expected_return = (
            portfolio_expected_return(
                weights,
                dataset.expected_returns,
            )
        )

        variance = (
            portfolio_variance(
                weights,
                dataset.covariance_matrix,
            )
        )

        utility = (
            expected_return
            -
            self.risk_aversion
            * variance
        )

        return -utility


# ============================================================
# MINIMUM VARIANCE OBJECTIVE
# ============================================================

class MinimumVarianceObjective(
    BaseObjectiveFunction
):
    """
    Pure variance minimization.
    """

    def evaluate(
        self,
        weights: np.ndarray,
        dataset: OptimizationDataset,
    ) -> float:

        return portfolio_variance(
            weights,
            dataset.covariance_matrix,
        )


# ============================================================
# MAXIMUM SHARPE OBJECTIVE
# ============================================================

class MaximumSharpeObjective(
    BaseObjectiveFunction
):
    """
    Maximize Sharpe Ratio.

    Minimizer returns negative Sharpe.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
    ):
        self.risk_free_rate = (
            risk_free_rate
        )

    def evaluate(
        self,
        weights: np.ndarray,
        dataset: OptimizationDataset,
    ) -> float:

        expected_return = (
            portfolio_expected_return(
                weights,
                dataset.expected_returns,
            )
        )

        volatility = (
            portfolio_volatility(
                weights,
                dataset.covariance_matrix,
            )
        )

        if volatility <= EPSILON:

            return 1e12

        sharpe = (
            expected_return
            -
            self.risk_free_rate
        ) / volatility

        return -float(sharpe)


# ============================================================
# TRACKING ERROR OBJECTIVE
# ============================================================

class TrackingErrorObjective(
    BaseObjectiveFunction
):
    """
    Minimize tracking error
    versus benchmark.
    """

    def evaluate(
        self,
        weights: np.ndarray,
        dataset: OptimizationDataset,
    ) -> float:

        return tracking_error(
            weights,
            dataset.benchmark_weights.values,
            dataset.covariance_matrix,
        )


# ============================================================
# TURNOVER PENALTY
# ============================================================

class TurnoverPenalty:
    """
    Penalize trading activity.
    """

    def __init__(
        self,
        penalty: float = 0.10,
    ):
        self.penalty = penalty

    def evaluate(
        self,
        weights: np.ndarray,
        current_weights: np.ndarray,
    ) -> float:

        turnover = np.sum(
            np.abs(
                weights
                - current_weights
            )
        )

        return float(
            self.penalty
            * turnover
        )


# ============================================================
# TRANSACTION COST PENALTY
# ============================================================

class TransactionCostPenalty:
    """
    Penalize estimated costs.
    """

    def __init__(
        self,
        cost_rate: float = 0.001,
    ):
        self.cost_rate = (
            cost_rate
        )

    def evaluate(
        self,
        weights: np.ndarray,
        current_weights: np.ndarray,
    ) -> float:

        turnover = np.sum(
            np.abs(
                weights
                - current_weights
            )
        )

        cost = (
            turnover
            * self.cost_rate
        )

        return float(cost)


# ============================================================
# MULTI OBJECTIVE FUNCTION
# ============================================================

class MultiObjectiveFunction(
    BaseObjectiveFunction
):
    """
    Institutional objective.

    Objective
    ---------

    Core Objective
    + Turnover Penalty
    + Transaction Cost Penalty
    """

    def __init__(
        self,
        objective: BaseObjectiveFunction,
        turnover_penalty:
        TurnoverPenalty | None = None,
        transaction_penalty:
        TransactionCostPenalty | None = None,
    ):

        self.objective = (
            objective
        )

        self.turnover_penalty = (
            turnover_penalty
        )

        self.transaction_penalty = (
            transaction_penalty
        )

    def evaluate(
        self,
        weights: np.ndarray,
        dataset: OptimizationDataset,
    ) -> float:

        score = (
            self.objective.evaluate(
                weights,
                dataset,
            )
        )

        if (
            self.turnover_penalty
            is not None
        ):

            score += (
                self.turnover_penalty
                .evaluate(
                    weights,
                    dataset.current_weights.values,
                )
            )

        if (
            self.transaction_penalty
            is not None
        ):

            score += (
                self.transaction_penalty
                .evaluate(
                    weights,
                    dataset.current_weights.values,
                )
            )

        return float(score)


# ============================================================
# OBJECTIVE FACTORY
# ============================================================

class ObjectiveFactory:
    """
    Creates institutional objectives.
    """

    @staticmethod
    def mean_variance(
        risk_aversion: float = 5.0,
    ) -> BaseObjectiveFunction:

        return MeanVarianceObjective(
            risk_aversion=
            risk_aversion
        )

    @staticmethod
    def minimum_variance(
    ) -> BaseObjectiveFunction:

        return (
            MinimumVarianceObjective()
        )

    @staticmethod
    def maximum_sharpe(
        risk_free_rate: float = 0.0,
    ) -> BaseObjectiveFunction:

        return MaximumSharpeObjective(
            risk_free_rate=
            risk_free_rate
        )

    @staticmethod
    def tracking_error(
    ) -> BaseObjectiveFunction:

        return (
            TrackingErrorObjective()
        )
    
# ============================================================
# PART 4: CONSTRAINT INTEGRATION
# ============================================================

@dataclass(slots=True)
class OptimizationConstraints:
    """
    Optimization-ready constraints.
    """

    lower_bounds: np.ndarray

    upper_bounds: np.ndarray

    equality_constraints: list

    inequality_constraints: list


# ============================================================
# BOUNDS ENGINE
# ============================================================

class BoundsBuilder:
    """
    Construct optimizer bounds.

    Current implementation:
        Long-only

    Future:
        Long-short
        Leverage
        Asset-class bounds
    """

    @staticmethod
    def build(
        dataset: OptimizationDataset,
        *,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:

        n = len(
            dataset.universe
        )

        lower = np.full(
            n,
            min_weight,
            dtype=float,
        )

        upper = np.full(
            n,
            max_weight,
            dtype=float,
        )

        return (
            lower,
            upper,
        )


# ============================================================
# EQUALITY CONSTRAINTS
# ============================================================

class EqualityConstraintBuilder:
    """
    Sum(weights) = target
    """

    @staticmethod
    def fully_invested(
        target: float = 1.0,
    ):

        def constraint(
            weights: np.ndarray,
        ) -> float:

            return (
                np.sum(weights)
                - target
            )

        return constraint


# ============================================================
# INEQUALITY CONSTRAINTS
# ============================================================

class InequalityConstraintBuilder:
    """
    Optimization constraints.

    Convention

        g(x) <= 0
    """

    @staticmethod
    def max_turnover(
        current_weights: np.ndarray,
        max_turnover: float,
    ):

        def constraint(
            weights: np.ndarray,
        ) -> float:

            turnover = np.sum(
                np.abs(
                    weights
                    - current_weights
                )
            )

            return (
                turnover
                - max_turnover
            )

        return constraint

    # --------------------------------------------------------

    @staticmethod
    def max_tracking_error(
        benchmark_weights: np.ndarray,
        covariance_matrix: pd.DataFrame,
        max_te: float,
    ):

        def constraint(
            weights: np.ndarray,
        ) -> float:

            te = tracking_error(
                weights,
                benchmark_weights,
                covariance_matrix,
            )

            return (
                te
                - max_te
            )

        return constraint


# ============================================================
# FEASIBILITY ENGINE
# ============================================================

class FeasibilityEngine:
    """
    Converts portfolio constraints
    into optimizer constraints.

    Uses ConstraintEngine
    from constraints.py.
    """

    def __init__(
        self,
        constraint_engine:
        ConstraintEngine,
    ):
        self.constraint_engine = (
            constraint_engine
        )

    # --------------------------------------------------------

    def validate(
        self,
        portfolio: Portfolio,
    ) -> list[ConstraintResult]:

        return (
            self.constraint_engine
            .validate(
                portfolio
            )
        )

    # --------------------------------------------------------

    def feasible(
        self,
        portfolio: Portfolio,
    ) -> bool:

        results = (
            self.validate(
                portfolio
            )
        )

        return (
            self.constraint_engine
            .is_feasible(
                results
            )
        )

    # --------------------------------------------------------

    def violations(
        self,
        portfolio: Portfolio,
    ) -> list[ConstraintResult]:

        results = (
            self.validate(
                portfolio
            )
        )

        return (
            self.constraint_engine
            .violations(
                results
            )
        )


# ============================================================
# CONSTRAINT REPAIR ENGINE
# ============================================================

class ConstraintRepairEngine:
    """
    Simple production-safe repair.

    Future:
        Projection methods
        Convex repair
        SQP repair
    """

    def __init__(
        self,
        feasibility_engine:
        FeasibilityEngine,
    ):

        self.feasibility_engine = (
            feasibility_engine
        )

    # --------------------------------------------------------

    @staticmethod
    def normalize(
        weights: np.ndarray,
    ) -> np.ndarray:

        weights = np.clip(
            weights,
            0.0,
            None,
        )

        total = np.sum(
            weights
        )

        if total <= EPSILON:

            raise ValueError(
                "Zero weights."
            )

        return (
            weights
            / total
        )

    # --------------------------------------------------------

    def repair(
        self,
        portfolio: Portfolio,
        *,
        max_iterations: int = 20,
    ) -> Portfolio:

        repaired = (
            portfolio.copy()
        )

        for _ in range(
            max_iterations
        ):

            weights = (
                repaired.weights
                .values
            )

            weights = (
                self.normalize(
                    weights
                )
            )

            repaired.data[
                "Position_Weight"
            ] = weights

            if (
                self.feasibility_engine
                .feasible(
                    repaired
                )
            ):

                return repaired

        return repaired


# ============================================================
# CONSTRAINT DATASET BUILDER
# ============================================================

def build_constraints(
    dataset: OptimizationDataset,
    *,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> OptimizationConstraints:
    """
    Build optimization constraints.
    """

    (
        lower_bounds,
        upper_bounds,
    ) = BoundsBuilder.build(
        dataset,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    equality_constraints = [

        EqualityConstraintBuilder
        .fully_invested(
            target=1.0
        )

    ]

    inequality_constraints = []

    return OptimizationConstraints(
        lower_bounds=
        lower_bounds,

        upper_bounds=
        upper_bounds,

        equality_constraints=
        equality_constraints,

        inequality_constraints=
        inequality_constraints,
    )


# ============================================================
# FEASIBILITY DIAGNOSTICS
# ============================================================

@dataclass(slots=True)
class FeasibilityDiagnostics:
    """
    Optimization diagnostics.
    """

    feasible: bool

    violation_count: int

    total_violation: float

    failed_constraints: list[str]


# ============================================================
# BUILD FEASIBILITY REPORT
# ============================================================

def build_feasibility_report(
    portfolio: Portfolio,
    feasibility_engine:
    FeasibilityEngine,
) -> FeasibilityDiagnostics:

    results = (
        feasibility_engine
        .validate(
            portfolio
        )
    )

    violations = (
        feasibility_engine
        .violations(
            portfolio
        )
    )

    return FeasibilityDiagnostics(
        feasible=
        len(violations) == 0,

        violation_count=
        len(violations),

        total_violation=
        float(
            sum(
                v.violation
                for v in violations
            )
        ),

        failed_constraints=[
            v.name
            for v in violations
        ],
    )


# ============================================================
# PART 5: SOLVER RESULT
# ============================================================

@dataclass(slots=True)
class SolverResult:
    """
    Output from optimization backend.
    """

    weights: np.ndarray

    objective_value: float

    success: bool

    iterations: int

    message: str


# ============================================================
# BASE SOLVER BACKEND
# ============================================================

class BaseSolverBackend(ABC):
    """
    Abstract optimization backend.

    All numerical optimizers must
    implement this interface.
    """

    @abstractmethod
    def solve(
        self,
        *,
        initial_weights: np.ndarray,
        objective,
        constraints: OptimizationConstraints,
    ) -> SolverResult:

        raise NotImplementedError


# ============================================================
# HEURISTIC SOLVER
# ============================================================

class HeuristicSolverBackend(
    BaseSolverBackend,
):
    """
    Deterministic production-safe solver.

    Useful for:

        Backtesting
        Unit tests
        CI/CD
        Fallback mode

    No external dependencies.
    """

    def solve(
        self,
        *,
        initial_weights: np.ndarray,
        objective,
        constraints: OptimizationConstraints,
    ) -> SolverResult:

        weights = np.asarray(
            initial_weights,
            dtype=float,
        )

        weights = np.clip(
            weights,
            constraints.lower_bounds,
            constraints.upper_bounds,
        )

        total = weights.sum()

        if total > EPSILON:

            weights /= total

        obj = objective(
            weights
        )

        return SolverResult(
            weights=weights,
            objective_value=float(obj),
            success=True,
            iterations=1,
            message=(
                "Heuristic optimization complete."
            ),
        )


# ============================================================
# SCIPY SOLVER
# ============================================================

class ScipySolverBackend(
    BaseSolverBackend,
):
    """
    SciPy-based optimizer.

    Uses SLSQP.

    Automatically falls back
    if scipy unavailable.
    """

    def solve(
        self,
        *,
        initial_weights: np.ndarray,
        objective,
        constraints: OptimizationConstraints,
    ) -> SolverResult:

        try:

            from scipy.optimize import (
                minimize,
            )

        except Exception:

            return SolverResult(
                weights=initial_weights,
                objective_value=np.nan,
                success=False,
                iterations=0,
                message=(
                    "SciPy unavailable."
                ),
            )

        scipy_constraints = []

        # ---------------------------------
        # Equality Constraints
        # ---------------------------------

        for eq in (
            constraints
            .equality_constraints
        ):

            scipy_constraints.append(
                {
                    "type": "eq",
                    "fun": eq,
                }
            )

        # ---------------------------------
        # Inequality Constraints
        # scipy wants g(x) >= 0
        # ---------------------------------

        for ineq in (
            constraints
            .inequality_constraints
        ):

            scipy_constraints.append(
                {
                    "type": "ineq",
                    "fun": (
                        lambda w,
                        f=ineq:
                        -f(w)
                    ),
                }
            )

        bounds = list(
            zip(
                constraints
                .lower_bounds,

                constraints
                .upper_bounds,
            )
        )

        result = minimize(
            objective,
            x0=initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=scipy_constraints,
        )

        return SolverResult(
            weights=result.x,
            objective_value=float(
                result.fun
            ),
            success=bool(
                result.success
            ),
            iterations=int(
                getattr(
                    result,
                    "nit",
                    0,
                )
            ),
            message=str(
                result.message
            ),
        )


# ============================================================
# CVXPY SOLVER
# ============================================================

class CVXPYSolverBackend(
    BaseSolverBackend,
):
    """
    Institutional-grade optimizer.

    Placeholder.

    Future support:

        Mean-Variance
        Risk Parity
        Black-Litterman
        Factor Risk
        Tracking Error
        Turnover Penalty
        ESG Constraints
    """

    def solve(
        self,
        *,
        initial_weights: np.ndarray,
        objective,
        constraints: OptimizationConstraints,
    ) -> SolverResult:

        return SolverResult(
            weights=initial_weights,
            objective_value=np.nan,
            success=False,
            iterations=0,
            message=(
                "CVXPY backend not implemented."
            ),
        )


# ============================================================
# SOLVER TYPE
# ============================================================

class SolverType(
    Enum
):
    """
    Available solver backends.
    """

    HEURISTIC = "heuristic"

    SCIPY = "scipy"

    CVXPY = "cvxpy"


# ============================================================
# SOLVER FACTORY
# ============================================================

class SolverFactory:
    """
    Creates optimization backends.
    """

    @staticmethod
    def create(
        solver_type:
        SolverType,
    ) -> BaseSolverBackend:

        if (
            solver_type
            ==
            SolverType.HEURISTIC
        ):

            return (
                HeuristicSolverBackend()
            )

        if (
            solver_type
            ==
            SolverType.SCIPY
        ):

            return (
                ScipySolverBackend()
            )

        if (
            solver_type
            ==
            SolverType.CVXPY
        ):

            return (
                CVXPYSolverBackend()
            )

        raise ValueError(
            f"Unknown solver: "
            f"{solver_type}"
        )


# ============================================================
# SOLVER DIAGNOSTICS
# ============================================================

@dataclass(slots=True)
class SolverDiagnostics:
    """
    Solver monitoring metrics.
    """

    backend: str

    success: bool

    objective_value: float

    iterations: int

    message: str


# ============================================================
# BUILD SOLVER REPORT
# ============================================================

def build_solver_report(
    result: SolverResult,
    backend_name: str,
) -> SolverDiagnostics:

    return SolverDiagnostics(
        backend=backend_name,
        success=result.success,
        objective_value=
        result.objective_value,
        iterations=
        result.iterations,
        message=result.message,
    )


# ============================================================
# BACKEND CAPABILITY CHECK
# ============================================================

def available_solvers() -> list[str]:
    """
    Detect installed solvers.
    """

    solvers = [
        SolverType.HEURISTIC.value
    ]

    try:

        import scipy

        solvers.append(
            SolverType.SCIPY.value
        )

    except Exception:
        pass

    try:

        import cvxpy

        solvers.append(
            SolverType.CVXPY.value
        )

    except Exception:
        pass

    return solvers


# ============================================================
# PART 6: OPTIMIZATION RESULT
"""
This is where everything gets connected:

RiskModel
ConstraintEngine
SolverBackend
OptimizationDataset
OptimizationResult
FeasibilityEngine

into the final production optimizer class.
"""
# ============================================================

@dataclass(slots=True)
class OptimizationResult:
    """
    Final optimizer output.
    """

    portfolio: Portfolio

    success: bool

    objective_value: float

    iterations: int

    message: str

    feasibility: FeasibilityDiagnostics | None

    solver: SolverDiagnostics | None


# ============================================================
# INSTITUTIONAL PORTFOLIO OPTIMIZER
# ============================================================

class PortfolioOptimizer:
    """
    Institutional-grade portfolio optimizer.

    Responsibilities
    ----------------
    • Build optimization dataset
    • Compute risk estimates
    • Build objective
    • Build constraints
    • Call solver
    • Repair infeasible portfolios
    • Generate diagnostics
    """

    def __init__(
        self,
        *,
        risk_model: BaseRiskModel,
        constraint_engine: ConstraintEngine,
        solver_backend: BaseSolverBackend | None = None,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        repair_constraints: bool = True,
    ) -> None:

        self.risk_model = (
            risk_model
        )

        self.constraint_engine = (
            constraint_engine
        )

        self.solver_backend = (
            solver_backend
            if solver_backend is not None
            else HeuristicSolverBackend()
        )

        self.min_weight = (
            min_weight
        )

        self.max_weight = (
            max_weight
        )

        self.repair_constraints = (
            repair_constraints
        )

    # --------------------------------------------------------
    # INITIAL WEIGHTS
    # --------------------------------------------------------

    @staticmethod
    def _initial_weights(
        portfolio: Portfolio,
    ) -> np.ndarray:
        """
        Starting point.

        Future:
            Risk parity seed
            Previous portfolio seed
            Factor-neutral seed
        """

        return (
            portfolio.weights
            .astype(float)
            .values
        )

    # --------------------------------------------------------
    # BUILD DATASET
    # --------------------------------------------------------

    def _build_dataset(
        self,
        portfolio: Portfolio,
    ) -> OptimizationDataset:

        return build_optimization_dataset(
            portfolio
        )

    # --------------------------------------------------------
    # BUILD RISK OBJECT
    # --------------------------------------------------------

    def _build_risk_result(
        self,
        portfolio: Portfolio,
    ):

        return (
            self.risk_model
            .evaluate(
                portfolio
            )
        )

    # --------------------------------------------------------
    # BUILD OBJECTIVE
    # --------------------------------------------------------

    def _build_objective(
        self,
        dataset: OptimizationDataset,
        risk_result,
    ):

        covariance_matrix = (
            risk_result
            .covariance_matrix
        )

        expected_returns = (
            dataset.expected_returns
        )

        return MeanVarianceObjective(
            expected_returns=
            expected_returns,

            covariance_matrix=
            covariance_matrix,

            risk_aversion=1.0,
        )

    # --------------------------------------------------------
    # BUILD CONSTRAINTS
    # --------------------------------------------------------

    def _build_constraints(
        self,
        dataset: OptimizationDataset,
    ) -> OptimizationConstraints:

        return build_constraints(
            dataset,
            min_weight=
            self.min_weight,

            max_weight=
            self.max_weight,
        )

    # --------------------------------------------------------
    # SOLVE
    # --------------------------------------------------------

    def _solve(
        self,
        *,
        portfolio: Portfolio,
        objective,
        constraints:
        OptimizationConstraints,
    ) -> SolverResult:

        return (
            self.solver_backend
            .solve(
                initial_weights=
                self._initial_weights(
                    portfolio
                ),

                objective=
                objective,

                constraints=
                constraints,
            )
        )

    # --------------------------------------------------------
    # APPLY WEIGHTS
    # --------------------------------------------------------

    @staticmethod
    def _apply_weights(
        portfolio: Portfolio,
        weights: np.ndarray,
    ) -> Portfolio:

        optimized = (
            portfolio.copy()
        )

        optimized.data[
            "Position_Weight"
        ] = weights

        return optimized

    # --------------------------------------------------------
    # REPAIR
    # --------------------------------------------------------

    def _repair(
        self,
        portfolio: Portfolio,
    ) -> Portfolio:

        feasibility_engine = (
            FeasibilityEngine(
                self.constraint_engine
            )
        )

        repair_engine = (
            ConstraintRepairEngine(
                feasibility_engine
            )
        )

        return repair_engine.repair(
            portfolio
        )

    # --------------------------------------------------------
    # MASTER OPTIMIZATION
    # --------------------------------------------------------

    def optimize(
        self,
        portfolio: Portfolio,
    ) -> OptimizationResult:
        """
        Main production optimization.
        """

        if portfolio.empty:

            raise ValueError(
                "Portfolio is empty."
            )

        # ---------------------------------
        # Dataset
        # ---------------------------------

        dataset = (
            self._build_dataset(
                portfolio
            )
        )

        # ---------------------------------
        # Risk
        # ---------------------------------

        risk_result = (
            self._build_risk_result(
                portfolio
            )
        )

        # ---------------------------------
        # Objective
        # ---------------------------------

        objective_engine = (
            self._build_objective(
                dataset,
                risk_result,
            )
        )

        objective = (
            objective_engine
            .objective
        )

        # ---------------------------------
        # Constraints
        # ---------------------------------

        constraints = (
            self._build_constraints(
                dataset
            )
        )

        # ---------------------------------
        # Solver
        # ---------------------------------

        solver_result = (
            self._solve(
                portfolio=
                portfolio,

                objective=
                objective,

                constraints=
                constraints,
            )
        )

        # ---------------------------------
        # Portfolio
        # ---------------------------------

        optimized = (
            self._apply_weights(
                portfolio,
                solver_result.weights,
            )
        )

        # ---------------------------------
        # Repair
        # ---------------------------------

        if (
            self.repair_constraints
        ):

            optimized = (
                self._repair(
                    optimized
                )
            )

        # ---------------------------------
        # Feasibility
        # ---------------------------------

        feasibility_engine = (
            FeasibilityEngine(
                self.constraint_engine
            )
        )

        feasibility = (
            build_feasibility_report(
                optimized,
                feasibility_engine,
            )
        )

        # ---------------------------------
        # Solver Report
        # ---------------------------------

        solver_report = (
            build_solver_report(
                solver_result,
                type(
                    self.solver_backend
                ).__name__,
            )
        )

        return OptimizationResult(
            portfolio=
            optimized,

            success=
            solver_result.success,

            objective_value=
            solver_result.objective_value,

            iterations=
            solver_result.iterations,

            message=
            solver_result.message,

            feasibility=
            feasibility,

            solver=
            solver_report,
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def optimize_portfolio(
    portfolio: Portfolio,
    *,
    risk_model: BaseRiskModel,
    constraint_engine:
    ConstraintEngine,
    solver_backend:
    BaseSolverBackend | None = None,
) -> OptimizationResult:
    """
    Production optimization API.
    """

    optimizer = (
        PortfolioOptimizer(
            risk_model=
            risk_model,

            constraint_engine=
            constraint_engine,

            solver_backend=
            solver_backend,
        )
    )

    return optimizer.optimize(
        portfolio
    )


# ============================================================
# DEFAULT PRODUCTION OPTIMIZER
# ============================================================

def create_default_optimizer(
    *,
    risk_model:
    BaseRiskModel,
    constraint_engine:
    ConstraintEngine,
) -> PortfolioOptimizer:
    """
    Institutional default.
    """

    return PortfolioOptimizer(
        risk_model=
        risk_model,

        constraint_engine=
        constraint_engine,

        solver_backend=
        ScipySolverBackend(),
    )


# ============================================================
# PART 7: RISK PARITY OPTIMIZER
# ============================================================

class RiskParityOptimizer:
    """
    Institutional Risk Parity Optimizer.

    Objective
    ---------
    Equalize risk contribution
    across positions.

    Risk Contribution_i

        w_i * (Σw)_i

    Total Risk

        sqrt(w'Σw)

    Supports

        • Long-only
        • Production-safe fallback
        • Covariance-based allocation

    Future

        • Hierarchical Risk Parity
        • Cluster Risk Parity
        • Regime-Aware Risk Parity
    """

    def __init__(
        self,
        *,
        max_iterations: int = 500,
        tolerance: float = 1e-8,
    ) -> None:

        self.max_iterations = (
            max_iterations
        )

        self.tolerance = (
            tolerance
        )

    # --------------------------------------------------------
    # PORTFOLIO VOLATILITY
    # --------------------------------------------------------

    @staticmethod
    def portfolio_volatility(
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> float:

        return float(
            np.sqrt(
                weights.T
                @ covariance_matrix
                @ weights
            )
        )

    # --------------------------------------------------------
    # MARGINAL RISK
    # --------------------------------------------------------

    @staticmethod
    def marginal_risk(
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> np.ndarray:

        portfolio_vol = (
            RiskParityOptimizer
            .portfolio_volatility(
                weights,
                covariance_matrix,
            )
        )

        if portfolio_vol <= EPSILON:

            return np.zeros_like(
                weights
            )

        return (
            covariance_matrix
            @ weights
        ) / portfolio_vol

    # --------------------------------------------------------
    # RISK CONTRIBUTION
    # --------------------------------------------------------

    @staticmethod
    def risk_contribution(
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> np.ndarray:

        marginal = (
            RiskParityOptimizer
            .marginal_risk(
                weights,
                covariance_matrix,
            )
        )

        return (
            weights
            * marginal
        )

    # --------------------------------------------------------
    # TARGET RISK
    # --------------------------------------------------------

    @staticmethod
    def target_risk_contribution(
        n_assets: int,
    ) -> np.ndarray:

        return np.full(
            n_assets,
            1.0 / n_assets,
            dtype=float,
        )

    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    def objective(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> float:

        rc = (
            self.risk_contribution(
                weights,
                covariance_matrix,
            )
        )

        total_rc = (
            rc.sum()
        )

        if total_rc <= EPSILON:

            return np.inf

        rc_pct = (
            rc
            / total_rc
        )

        target = (
            self.target_risk_contribution(
                len(weights)
            )
        )

        return float(
            np.sum(
                (
                    rc_pct
                    - target
                ) ** 2
            )
        )

    # --------------------------------------------------------
    # SOLVE
    # --------------------------------------------------------

    def solve(
        self,
        covariance_matrix: pd.DataFrame,
        *,
        initial_weights:
        np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Risk parity allocation.

        Returns
        -------
        weights
        """

        cov = (
            covariance_matrix
            .values
        )

        n_assets = len(cov)

        if (
            initial_weights
            is None
        ):

            weights = np.full(
                n_assets,
                1.0 / n_assets,
                dtype=float,
            )

        else:

            weights = np.asarray(
                initial_weights,
                dtype=float,
            )

            weights = (
                weights
                / weights.sum()
            )

        # ---------------------------------
        # Try SciPy
        # ---------------------------------

        try:

            from scipy.optimize import (
                minimize,
            )

            constraints = [

                {
                    "type": "eq",
                    "fun":
                    lambda w:
                    np.sum(w)
                    - 1.0,
                }

            ]

            bounds = [

                (
                    0.0,
                    1.0,
                )

                for _ in range(
                    n_assets
                )

            ]

            result = minimize(
                lambda w:
                self.objective(
                    w,
                    cov,
                ),

                weights,

                method="SLSQP",

                bounds=bounds,

                constraints=
                constraints,

                options={
                    "maxiter":
                    self.max_iterations,
                    "ftol":
                    self.tolerance,
                },
            )

            if (
                result.success
            ):

                w = (
                    result.x
                )

                return (
                    w
                    / w.sum()
                )

        except Exception:

            pass

        # ---------------------------------
        # Fallback
        # Inverse Volatility
        # ---------------------------------

        diag = np.diag(
            cov
        )

        diag = np.where(
            diag <= EPSILON,
            EPSILON,
            diag,
        )

        vol = np.sqrt(
            diag
        )

        inv_vol = (
            1.0 / vol
        )

        return (
            inv_vol
            / inv_vol.sum()
        )


# ============================================================
# RISK PARITY RESULT
# ============================================================

@dataclass(slots=True)
class RiskParityResult:
    """
    Diagnostics.
    """

    weights: np.ndarray

    portfolio_volatility: float

    risk_contributions: np.ndarray

    contribution_pct: np.ndarray


# ============================================================
# BUILD REPORT
# ============================================================

def build_risk_parity_report(
    weights: np.ndarray,
    covariance_matrix:
    pd.DataFrame,
) -> RiskParityResult:

    optimizer = (
        RiskParityOptimizer()
    )

    rc = (
        optimizer
        .risk_contribution(
            weights,
            covariance_matrix
            .values,
        )
    )

    total_rc = (
        rc.sum()
    )

    if total_rc <= EPSILON:

        rc_pct = (
            np.zeros_like(
                rc
            )
        )

    else:

        rc_pct = (
            rc
            / total_rc
        )

    return RiskParityResult(
        weights=weights,

        portfolio_volatility=
        optimizer
        .portfolio_volatility(
            weights,
            covariance_matrix
            .values,
        ),

        risk_contributions=
        rc,

        contribution_pct=
        rc_pct,
    )


# ============================================================
# CONVENIENCE API
# ============================================================

def optimize_risk_parity(
    covariance_matrix:
    pd.DataFrame,
    *,
    initial_weights:
    np.ndarray | None = None,
) -> np.ndarray:
    """
    Production API.
    """

    optimizer = (
        RiskParityOptimizer()
    )

    return optimizer.solve(
        covariance_matrix,
        initial_weights=
        initial_weights,
    )



# ============================================================
# PART 8: BLACK-LITTERMAN VIEW
# ============================================================

@dataclass(slots=True)
class BLView:
    """
    Single investor view.

    Examples
    --------

    Absolute:

        Asset A expected
        return = 12%

    Relative:

        Asset A expected
        return > Asset B
        by 3%
    """

    name: str

    assets: list[str]

    weights: np.ndarray

    expected_return: float

    confidence: float = 0.50


# ============================================================
# BLACK-LITTERMAN RESULT
# ============================================================

@dataclass(slots=True)
class BlackLittermanResult:
    """
    Posterior output.
    """

    prior_returns: pd.Series

    posterior_returns: pd.Series

    posterior_covariance: pd.DataFrame

    tau: float

    n_views: int


# ============================================================
# BLACK-LITTERMAN ENGINE
# ============================================================

class BlackLittermanEngine:
    """
    Institutional Black-Litterman model.

    References
    ----------

    Black & Litterman (1992)

    Future upgrades

        • Bayesian views
        • Regime-aware priors
        • Dynamic confidence
        • Macro-factor views
    """

    def __init__(
        self,
        *,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
    ) -> None:

        self.tau = tau

        self.risk_aversion = (
            risk_aversion
        )

    # --------------------------------------------------------
    # IMPLIED EQUILIBRIUM RETURNS
    # --------------------------------------------------------

    def equilibrium_returns(
        self,
        covariance_matrix:
        pd.DataFrame,

        market_weights:
        pd.Series,
    ) -> pd.Series:
        """
        Pi = λ Σ w
        """

        cov = (
            covariance_matrix
            .values
        )

        w = (
            market_weights
            .values
        )

        implied = (
            self.risk_aversion
            * cov
            @ w
        )

        return pd.Series(
            implied,
            index=
            covariance_matrix.index,
        )

    # --------------------------------------------------------
    # BUILD PICK MATRIX
    # --------------------------------------------------------

    @staticmethod
    def build_pick_matrix(
        assets:
        list[str],

        views:
        list[BLView],
    ) -> np.ndarray:
        """
        P matrix.
        """

        n_assets = (
            len(assets)
        )

        n_views = (
            len(views)
        )

        P = np.zeros(
            (
                n_views,
                n_assets,
            ),
            dtype=float,
        )

        lookup = {

            asset: idx

            for idx, asset
            in enumerate(
                assets
            )

        }

        for row, view in enumerate(
            views
        ):

            for (
                asset,
                weight,
            ) in zip(
                view.assets,
                view.weights,
            ):

                if (
                    asset
                    in lookup
                ):

                    P[
                        row,
                        lookup[
                            asset
                        ],
                    ] = weight

        return P

    # --------------------------------------------------------
    # VIEW VECTOR
    # --------------------------------------------------------

    @staticmethod
    def build_view_vector(
        views:
        list[BLView],
    ) -> np.ndarray:

        return np.asarray(
            [
                v.expected_return
                for v in views
            ],
            dtype=float,
        )

    # --------------------------------------------------------
    # OMEGA MATRIX
    # --------------------------------------------------------

    @staticmethod
    def build_omega(
        views:
        list[BLView],
    ) -> np.ndarray:
        """
        View uncertainty matrix.

        Higher confidence
        -> smaller uncertainty.
        """

        omega = []

        for view in views:

            conf = max(
                min(
                    view.confidence,
                    0.999,
                ),
                0.001,
            )

            uncertainty = (
                1.0
                - conf
            )

            omega.append(
                uncertainty
            )

        return np.diag(
            omega
        )

    # --------------------------------------------------------
    # POSTERIOR
    # --------------------------------------------------------

    def posterior_returns(
        self,
        *,
        covariance_matrix:
        pd.DataFrame,

        prior_returns:
        pd.Series,

        views:
        list[BLView],
    ) -> BlackLittermanResult:
        """
        Black-Litterman update.
        """

        if (
            len(views)
            == 0
        ):

            return (
                BlackLittermanResult(
                    prior_returns=
                    prior_returns,

                    posterior_returns=
                    prior_returns,

                    posterior_covariance=
                    covariance_matrix,

                    tau=
                    self.tau,

                    n_views=0,
                )
            )

        assets = list(
            covariance_matrix.index
        )

        Sigma = (
            covariance_matrix
            .values
        )

        Pi = (
            prior_returns
            .values
        )

        P = (
            self.build_pick_matrix(
                assets,
                views,
            )
        )

        Q = (
            self.build_view_vector(
                views
            )
        )

        Omega = (
            self.build_omega(
                views
            )
        )

        tauSigma = (
            self.tau
            * Sigma
        )

        # ---------------------------------
        # BL Posterior Mean
        # ---------------------------------

        middle = np.linalg.inv(

            P
            @ tauSigma
            @ P.T

            + Omega

        )

        posterior_mean = (

            Pi

            +

            tauSigma
            @ P.T
            @ middle
            @ (
                Q
                - P @ Pi
            )

        )

        # ---------------------------------
        # BL Posterior Covariance
        # ---------------------------------

        posterior_cov = (

            Sigma

            +

            tauSigma

            -

            tauSigma
            @ P.T
            @ middle
            @ P
            @ tauSigma

        )

        return (
            BlackLittermanResult(
                prior_returns=
                prior_returns,

                posterior_returns=
                pd.Series(
                    posterior_mean,
                    index=assets,
                ),

                posterior_covariance=
                pd.DataFrame(
                    posterior_cov,
                    index=assets,
                    columns=assets,
                ),

                tau=
                self.tau,

                n_views=
                len(views),
            )
        )


# ============================================================
# VIEW FACTORY
# ============================================================

class BLViewFactory:
    """
    Convenience builders.
    """

    @staticmethod
    def absolute_view(
        *,
        asset: str,
        expected_return:
        float,
        confidence:
        float = 0.50,
    ) -> BLView:

        return BLView(
            name=
            f"ABS_{asset}",

            assets=[asset],

            weights=
            np.array(
                [1.0]
            ),

            expected_return=
            expected_return,

            confidence=
            confidence,
        )

    @staticmethod
    def relative_view(
        *,
        asset_long: str,
        asset_short: str,
        spread: float,
        confidence:
        float = 0.50,
    ) -> BLView:

        return BLView(
            name=
            f"REL_{asset_long}_{asset_short}",

            assets=[
                asset_long,
                asset_short,
            ],

            weights=
            np.array(
                [
                    1.0,
                    -1.0,
                ]
            ),

            expected_return=
            spread,

            confidence=
            confidence,
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def run_black_litterman(
    *,
    covariance_matrix:
    pd.DataFrame,

    market_weights:
    pd.Series,

    views:
    list[BLView],

    tau: float = 0.05,
    risk_aversion:
    float = 2.5,
) -> BlackLittermanResult:
    """
    Production API.
    """

    engine = (
        BlackLittermanEngine(
            tau=tau,
            risk_aversion=
            risk_aversion,
        )
    )

    prior = (
        engine
        .equilibrium_returns(
            covariance_matrix,
            market_weights,
        )
    )

    return (
        engine
        .posterior_returns(
            covariance_matrix=
            covariance_matrix,

            prior_returns=
            prior,

            views=views,
        )
    )



# ============================================================
# PART 9: MULTI-OBJECTIVE CONFIGURATION
# ============================================================

@dataclass(slots=True)
class MultiObjectiveConfig:
    """
    Controls relative importance
    of competing objectives.

    All weights are normalized
    automatically.
    """

    return_weight: float = 1.0

    risk_weight: float = 1.0

    turnover_weight: float = 0.0

    transaction_cost_weight: float = 0.0

    diversification_weight: float = 0.0

    tracking_error_weight: float = 0.0


# ============================================================
# MULTI-OBJECTIVE RESULT
# ============================================================

@dataclass(slots=True)
class MultiObjectiveResult:
    """
    Objective diagnostics.
    """

    total_score: float

    expected_return_score: float

    risk_score: float

    turnover_score: float

    transaction_cost_score: float

    diversification_score: float

    tracking_error_score: float


# ============================================================
# MULTI-OBJECTIVE ENGINE
# ============================================================

class MultiObjectiveEngine:
    """
    Institutional multi-objective optimizer.

    Supports simultaneous optimization of:

        Return
        Risk
        Turnover
        Costs
        Diversification
        Tracking Error

    All objectives converted
    into a common scalar score.
    """

    def __init__(
        self,
        config:
        MultiObjectiveConfig,
    ) -> None:

        self.config = config

        self._normalize_weights()

    # --------------------------------------------------------
    # NORMALIZE OBJECTIVE WEIGHTS
    # --------------------------------------------------------

    def _normalize_weights(
        self,
    ) -> None:

        total = (

            self.config.return_weight

            + self.config.risk_weight

            + self.config.turnover_weight

            + self.config.transaction_cost_weight

            + self.config.diversification_weight

            + self.config.tracking_error_weight

        )

        if total <= EPSILON:

            total = 1.0

        self.config.return_weight /= total

        self.config.risk_weight /= total

        self.config.turnover_weight /= total

        self.config.transaction_cost_weight /= total

        self.config.diversification_weight /= total

        self.config.tracking_error_weight /= total

    # --------------------------------------------------------
    # EXPECTED RETURN
    # --------------------------------------------------------

    @staticmethod
    def expected_return(
        weights: np.ndarray,
        expected_returns:
        np.ndarray,
    ) -> float:

        return float(
            weights
            @ expected_returns
        )

    # --------------------------------------------------------
    # PORTFOLIO RISK
    # --------------------------------------------------------

    @staticmethod
    def portfolio_risk(
        weights: np.ndarray,
        covariance_matrix:
        np.ndarray,
    ) -> float:

        return float(
            np.sqrt(
                weights.T
                @ covariance_matrix
                @ weights
            )
        )

    # --------------------------------------------------------
    # TURNOVER
    # --------------------------------------------------------

    @staticmethod
    def turnover(
        weights: np.ndarray,
        previous_weights:
        np.ndarray | None,
    ) -> float:

        if (
            previous_weights
            is None
        ):

            return 0.0

        return float(
            np.sum(
                np.abs(
                    weights
                    - previous_weights
                )
            )
        )

    # --------------------------------------------------------
    # TRANSACTION COST
    # --------------------------------------------------------

    @staticmethod
    def transaction_cost(
        weights: np.ndarray,
        previous_weights:
        np.ndarray | None,
        cost_rate:
        float = 0.001,
    ) -> float:

        if (
            previous_weights
            is None
        ):

            return 0.0

        turnover = (
            np.sum(
                np.abs(
                    weights
                    - previous_weights
                )
            )
        )

        return float(
            turnover
            * cost_rate
        )

    # --------------------------------------------------------
    # DIVERSIFICATION
    # --------------------------------------------------------

    @staticmethod
    def diversification_penalty(
        weights: np.ndarray,
    ) -> float:
        """
        Herfindahl index.

        Lower is better.
        """

        return float(
            np.sum(
                weights ** 2
            )
        )

    # --------------------------------------------------------
    # TRACKING ERROR
    # --------------------------------------------------------

    @staticmethod
    def tracking_error(
        weights: np.ndarray,
        benchmark_weights:
        np.ndarray | None,
        covariance_matrix:
        np.ndarray,
    ) -> float:

        if (
            benchmark_weights
            is None
        ):

            return 0.0

        active = (
            weights
            - benchmark_weights
        )

        return float(
            np.sqrt(
                active.T
                @ covariance_matrix
                @ active
            )
        )

    # --------------------------------------------------------
    # MASTER OBJECTIVE
    # --------------------------------------------------------

    def objective(
        self,
        *,
        weights: np.ndarray,

        expected_returns:
        np.ndarray,

        covariance_matrix:
        np.ndarray,

        previous_weights:
        np.ndarray | None = None,

        benchmark_weights:
        np.ndarray | None = None,
    ) -> float:
        """
        Single scalar objective.

        Maximization form.

        Optimizers that minimize
        should use:

            -objective(...)
        """

        ret = (
            self.expected_return(
                weights,
                expected_returns,
            )
        )

        risk = (
            self.portfolio_risk(
                weights,
                covariance_matrix,
            )
        )

        turnover = (
            self.turnover(
                weights,
                previous_weights,
            )
        )

        cost = (
            self.transaction_cost(
                weights,
                previous_weights,
            )
        )

        diversification = (
            self.diversification_penalty(
                weights
            )
        )

        tracking_error = (
            self.tracking_error(
                weights,
                benchmark_weights,
                covariance_matrix,
            )
        )

        score = (

            self.config.return_weight
            * ret

            -

            self.config.risk_weight
            * risk

            -

            self.config.turnover_weight
            * turnover

            -

            self.config.transaction_cost_weight
            * cost

            -

            self.config.diversification_weight
            * diversification

            -

            self.config.tracking_error_weight
            * tracking_error

        )

        return float(
            score
        )

    # --------------------------------------------------------
    # FULL DIAGNOSTICS
    # --------------------------------------------------------

    def diagnostics(
        self,
        *,
        weights: np.ndarray,

        expected_returns:
        np.ndarray,

        covariance_matrix:
        np.ndarray,

        previous_weights:
        np.ndarray | None = None,

        benchmark_weights:
        np.ndarray | None = None,
    ) -> MultiObjectiveResult:

        ret = (
            self.expected_return(
                weights,
                expected_returns,
            )
        )

        risk = (
            self.portfolio_risk(
                weights,
                covariance_matrix,
            )
        )

        turnover = (
            self.turnover(
                weights,
                previous_weights,
            )
        )

        cost = (
            self.transaction_cost(
                weights,
                previous_weights,
            )
        )

        diversification = (
            self.diversification_penalty(
                weights
            )
        )

        te = (
            self.tracking_error(
                weights,
                benchmark_weights,
                covariance_matrix,
            )
        )

        total = self.objective(
            weights=weights,

            expected_returns=
            expected_returns,

            covariance_matrix=
            covariance_matrix,

            previous_weights=
            previous_weights,

            benchmark_weights=
            benchmark_weights,
        )

        return MultiObjectiveResult(
            total_score=total,

            expected_return_score=
            ret,

            risk_score=
            risk,

            turnover_score=
            turnover,

            transaction_cost_score=
            cost,

            diversification_score=
            diversification,

            tracking_error_score=
            te,
        )


# ============================================================
# MULTI-OBJECTIVE FACTORY
# ============================================================

def create_multi_objective_engine(
    *,
    return_weight: float = 1.0,

    risk_weight: float = 1.0,

    turnover_weight: float = 0.0,

    transaction_cost_weight:
    float = 0.0,

    diversification_weight:
    float = 0.0,

    tracking_error_weight:
    float = 0.0,
) -> MultiObjectiveEngine:

    return MultiObjectiveEngine(
        MultiObjectiveConfig(
            return_weight=
            return_weight,

            risk_weight=
            risk_weight,

            turnover_weight=
            turnover_weight,

            transaction_cost_weight=
            transaction_cost_weight,

            diversification_weight=
            diversification_weight,

            tracking_error_weight=
            tracking_error_weight,
        )
    )


# ============================================================
# PART 10: OPTIMIZATION DIAGNOSTICS
# ============================================================

@dataclass(slots=True)
class OptimizationDiagnostics:
    """
    Institutional optimizer diagnostics.

    Designed for:

        • Research
        • Audit trails
        • Production monitoring
        • Regulatory review
    """

    optimizer_name: str

    success: bool

    objective_value: float

    iterations: int

    solve_time_seconds: float

    message: str

    portfolio_return: float

    portfolio_volatility: float

    portfolio_sharpe: float

    portfolio_turnover: float

    effective_number_of_positions: float

    concentration_hhi: float

    tracking_error: float

    active_share: float

    max_position_weight: float

    min_position_weight: float

    gross_exposure: float

    net_exposure: float

    constraint_pass_rate: float

    failed_constraints: list[str]


# ============================================================
# DIAGNOSTIC CALCULATOR
# ============================================================

class OptimizationDiagnosticEngine:
    """
    Generates institutional optimizer reports.
    """

    def __init__(
        self,
        risk_model: BaseRiskModel,
        constraint_engine: ConstraintEngine | None = None,
    ) -> None:

        self.risk_model = risk_model

        self.constraint_engine = constraint_engine

    # --------------------------------------------------------
    # EFFECTIVE N
    # --------------------------------------------------------

    @staticmethod
    def effective_number_positions(
        weights: np.ndarray,
    ) -> float:

        hhi = np.sum(
            weights ** 2
        )

        if hhi <= EPSILON:

            return 0.0

        return float(
            1.0 / hhi
        )

    # --------------------------------------------------------
    # HHI
    # --------------------------------------------------------

    @staticmethod
    def concentration_hhi(
        weights: np.ndarray,
    ) -> float:

        return float(
            np.sum(
                weights ** 2
            )
        )

    # --------------------------------------------------------
    # ACTIVE SHARE
    # --------------------------------------------------------

    @staticmethod
    def active_share(
        weights: np.ndarray,
        benchmark_weights:
        np.ndarray | None,
    ) -> float:

        if benchmark_weights is None:

            return 0.0

        return float(
            0.5
            * np.sum(
                np.abs(
                    weights
                    - benchmark_weights
                )
            )
        )

    # --------------------------------------------------------
    # SHARPE
    # --------------------------------------------------------

    @staticmethod
    def sharpe_ratio(
        expected_return: float,
        volatility: float,
        risk_free_rate:
        float = 0.0,
    ) -> float:

        if volatility <= EPSILON:

            return 0.0

        return float(
            (
                expected_return
                - risk_free_rate
            )
            / volatility
        )

    # --------------------------------------------------------
    # TURNOVER
    # --------------------------------------------------------

    @staticmethod
    def turnover(
        weights: np.ndarray,
        previous_weights:
        np.ndarray | None,
    ) -> float:

        if previous_weights is None:

            return 0.0

        return float(
            np.sum(
                np.abs(
                    weights
                    - previous_weights
                )
            )
        )

    # --------------------------------------------------------
    # TRACKING ERROR
    # --------------------------------------------------------

    @staticmethod
    def tracking_error(
        weights: np.ndarray,
        benchmark_weights:
        np.ndarray | None,
        covariance_matrix:
        np.ndarray,
    ) -> float:

        if benchmark_weights is None:

            return 0.0

        active = (
            weights
            - benchmark_weights
        )

        return float(
            np.sqrt(
                active.T
                @ covariance_matrix
                @ active
            )
        )

    # --------------------------------------------------------
    # CONSTRAINT REPORT
    # --------------------------------------------------------

    def constraint_summary(
        self,
        portfolio: Portfolio,
    ) -> tuple[float, list[str]]:

        if (
            self.constraint_engine
            is None
        ):
            return 1.0, []

        results = (
            self.constraint_engine.validate(
                portfolio
            )
        )

        passed = sum(
            r.passed
            for r in results
        )

        total = max(
            len(results),
            1,
        )

        failed = [

            r.name

            for r in results

            if not r.passed

        ]

        return (

            passed / total,

            failed,

        )

    # --------------------------------------------------------
    # MASTER REPORT
    # --------------------------------------------------------

    def build_report(
        self,
        *,
        optimizer_name: str,

        portfolio: Portfolio,

        objective_value: float,

        success: bool,

        iterations: int,

        solve_time_seconds: float,

        message: str,

        expected_returns:
        np.ndarray,

        covariance_matrix:
        np.ndarray,

        benchmark_weights:
        np.ndarray | None = None,

        previous_weights:
        np.ndarray | None = None,
    ) -> OptimizationDiagnostics:

        weights = (
            portfolio.weights
            .astype(float)
            .values
        )

        portfolio_return = float(
            weights
            @ expected_returns
        )

        portfolio_volatility = float(
            np.sqrt(
                weights.T
                @ covariance_matrix
                @ weights
            )
        )

        sharpe = (
            self.sharpe_ratio(
                portfolio_return,
                portfolio_volatility,
            )
        )

        turnover = (
            self.turnover(
                weights,
                previous_weights,
            )
        )

        effective_n = (
            self.effective_number_positions(
                weights
            )
        )

        hhi = (
            self.concentration_hhi(
                weights
            )
        )

        tracking_error = (
            self.tracking_error(
                weights,
                benchmark_weights,
                covariance_matrix,
            )
        )

        active_share = (
            self.active_share(
                weights,
                benchmark_weights,
            )
        )

        pass_rate, failed = (
            self.constraint_summary(
                portfolio
            )
        )

        return OptimizationDiagnostics(

            optimizer_name=
            optimizer_name,

            success=
            success,

            objective_value=
            objective_value,

            iterations=
            iterations,

            solve_time_seconds=
            solve_time_seconds,

            message=
            message,

            portfolio_return=
            portfolio_return,

            portfolio_volatility=
            portfolio_volatility,

            portfolio_sharpe=
            sharpe,

            portfolio_turnover=
            turnover,

            effective_number_of_positions=
            effective_n,

            concentration_hhi=
            hhi,

            tracking_error=
            tracking_error,

            active_share=
            active_share,

            max_position_weight=
            float(weights.max()),

            min_position_weight=
            float(weights.min()),

            gross_exposure=
            float(
                np.sum(
                    np.abs(weights)
                )
            ),

            net_exposure=
            float(
                np.sum(weights)
            ),

            constraint_pass_rate=
            pass_rate,

            failed_constraints=
            failed,
        )


# ============================================================
# REPORT EXPORT
# ============================================================

def diagnostics_to_frame(
    diagnostics:
    OptimizationDiagnostics,
) -> pd.DataFrame:
    """
    Convert diagnostics
    into report dataframe.
    """

    data = {

        "Optimizer":
            diagnostics.optimizer_name,

        "Success":
            diagnostics.success,

        "Objective":
            diagnostics.objective_value,

        "Iterations":
            diagnostics.iterations,

        "SolveTime":
            diagnostics.solve_time_seconds,

        "ExpectedReturn":
            diagnostics.portfolio_return,

        "Volatility":
            diagnostics.portfolio_volatility,

        "Sharpe":
            diagnostics.portfolio_sharpe,

        "Turnover":
            diagnostics.portfolio_turnover,

        "EffectiveN":
            diagnostics.effective_number_of_positions,

        "HHI":
            diagnostics.concentration_hhi,

        "TrackingError":
            diagnostics.tracking_error,

        "ActiveShare":
            diagnostics.active_share,

        "GrossExposure":
            diagnostics.gross_exposure,

        "NetExposure":
            diagnostics.net_exposure,

        "ConstraintPassRate":
            diagnostics.constraint_pass_rate,

        "FailedConstraints":
            ", ".join(
                diagnostics.failed_constraints
            ),
    }

    return pd.DataFrame(
        [data]
    )


# ============================================================
# PRETTY PRINT REPORT
# ============================================================

def print_diagnostics(
    diagnostics:
    OptimizationDiagnostics,
) -> None:

    print("\n" + "=" * 80)
    print("OPTIMIZATION DIAGNOSTICS")
    print("=" * 80)

    print(
        f"Optimizer           : {diagnostics.optimizer_name}"
    )

    print(
        f"Success             : {diagnostics.success}"
    )

    print(
        f"Objective           : {diagnostics.objective_value:.6f}"
    )

    print(
        f"Expected Return     : {diagnostics.portfolio_return:.6f}"
    )

    print(
        f"Volatility          : {diagnostics.portfolio_volatility:.6f}"
    )

    print(
        f"Sharpe Ratio        : {diagnostics.portfolio_sharpe:.4f}"
    )

    print(
        f"Turnover            : {diagnostics.portfolio_turnover:.6f}"
    )

    print(
        f"Effective N         : {diagnostics.effective_number_of_positions:.2f}"
    )

    print(
        f"HHI                 : {diagnostics.concentration_hhi:.6f}"
    )

    print(
        f"Tracking Error      : {diagnostics.tracking_error:.6f}"
    )

    print(
        f"Active Share        : {diagnostics.active_share:.6f}"
    )

    print(
        f"Constraint PassRate : {diagnostics.constraint_pass_rate:.2%}"
    )

    if diagnostics.failed_constraints:

        print(
            "\nFailed Constraints:"
        )

        for c in diagnostics.failed_constraints:

            print(
                f"  - {c}"
            )

    print("=" * 80)


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

MeanVarianceOptimizer = PortfolioOptimizer

BlackLittermanOptimizer = BlackLittermanEngine


# ============================================================
# PART 11: OPTIMIZER FACTORY
# ============================================================

class OptimizerFactory:
    """
    Institutional optimizer factory.

    Creates production optimizer
    instances from standardized
    configuration inputs.
    """

    _REGISTRY: dict[str, type] = {
        "mean_variance": MeanVarianceOptimizer,

        "risk_parity": RiskParityOptimizer,

        "black_litterman": BlackLittermanOptimizer,
    }

    # --------------------------------------------------------

    @classmethod
    def available_optimizers(
        cls,
    ) -> list[str]:

        return sorted(
            cls._REGISTRY.keys()
        )

    # --------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        optimizer_type: str,

        risk_model:
        BaseRiskModel,

        constraint_engine:
        ConstraintEngine,

        **kwargs,
    ):
        """
        Create optimizer instance.

        Example
        -------
        factory.create(
            optimizer_type="mean_variance",
            ...
        )
        """

        optimizer_type = (
            optimizer_type
            .lower()
            .strip()
        )

        if (
            optimizer_type
            not in cls._REGISTRY
        ):
            raise ValueError(
                f"Unknown optimizer: "
                f"{optimizer_type}"
            )

        optimizer_cls = (
            cls._REGISTRY[
                optimizer_type
            ]
        )

        return optimizer_cls(
            risk_model=risk_model,
            constraint_engine=
            constraint_engine,
            **kwargs,
        )


# ============================================================
# STANDARD OPTIMIZER CONFIG
# ============================================================

@dataclass(slots=True)
class OptimizerConfig:
    """
    Master optimizer settings.

    Shared across all optimizers.
    """

    optimizer_type:str = "mean_variance"
    risk_aversion:float = 5.0
    max_iterations:int = 100
    tolerance:float = 1e-8
    allow_short:bool = False
    long_only:bool = True
    use_risk_model:bool = True
    use_constraints:bool = True
    use_transaction_costs:bool = False
    random_seed:int = 42


# ============================================================
# DEFAULT OBJECTIVE ENGINE
# ============================================================

def create_default_multi_objective(
) -> MultiObjectiveEngine:
    """
    Institutional default.

    Return and risk only.
    """

    return MultiObjectiveEngine(
        MultiObjectiveConfig(
            return_weight=1.0,
            risk_weight=1.0,
            turnover_weight=0.0,
            transaction_cost_weight=0.0,
            diversification_weight=0.0,
            tracking_error_weight=0.0,
        )
    )


# ============================================================
# DEFAULT RISK PARITY
# ============================================================

def create_risk_parity_optimizer(
    *,
    risk_model:
    BaseRiskModel,

    constraint_engine:
    ConstraintEngine,
):
    """
    Convenience wrapper.
    """

    return RiskParityOptimizer(
        risk_model=risk_model,
        constraint_engine=
        constraint_engine,
    )


# ============================================================
# DEFAULT MEAN VARIANCE
# ============================================================

def create_mean_variance_optimizer(
    *,
    risk_model:
    BaseRiskModel,

    constraint_engine:
    ConstraintEngine,

    risk_aversion:
    float = 5.0,
):
    """
    Convenience wrapper.
    """

    return MeanVarianceOptimizer(
        risk_model=risk_model,
        constraint_engine=
        constraint_engine,
        risk_aversion=
        risk_aversion,
    )


# ============================================================
# DEFAULT BLACK-LITTERMAN
# ============================================================

def create_black_litterman_optimizer(
    *,
    risk_model:
    BaseRiskModel,

    constraint_engine:
    ConstraintEngine,
):
    """
    Convenience wrapper.
    """

    return BlackLittermanOptimizer(
        risk_model=risk_model,
        constraint_engine=
        constraint_engine,
    )


# ============================================================
# MASTER CONVENIENCE API
# ============================================================

def optimize_portfolio(
    *,
    portfolio: Portfolio,

    optimizer_type:
    str,

    risk_model:
    BaseRiskModel,

    constraint_engine:
    ConstraintEngine,

    expected_returns:
    np.ndarray | None = None,

    covariance_matrix:
    np.ndarray | None = None,

    benchmark_weights:
    np.ndarray | None = None,

    previous_weights:
    np.ndarray | None = None,

    **optimizer_kwargs,
) -> OptimizationResult:
    """
    Institutional one-line API.

    Example
    -------
    result = optimize_portfolio(
        portfolio=portfolio,
        optimizer_type="risk_parity",
        risk_model=risk_model,
        constraint_engine=constraint_engine,
    )
    """

    optimizer = (
        OptimizerFactory.create(
            optimizer_type=
            optimizer_type,

            risk_model=
            risk_model,

            constraint_engine=
            constraint_engine,

            **optimizer_kwargs,
        )
    )

    return optimizer.optimize(
        portfolio
    )


# ============================================================
# OPTIMIZER REGISTRATION
# ============================================================

def register_optimizer(
    name: str,
    optimizer_class,
) -> None:
    """
    Register custom optimizer.

    Allows external teams to plug
    in proprietary optimizers.
    """

    OptimizerFactory._REGISTRY[
        name.lower()
    ] = optimizer_class


# ============================================================
# REGISTRY REPORT
# ============================================================

def optimizer_registry_report(
) -> pd.DataFrame:
    """
    Show all registered optimizers.
    """

    rows = []

    for (
        name,
        cls,
    ) in OptimizerFactory._REGISTRY.items():

        rows.append({

            "Optimizer":
                name,

            "Class":
                cls.__name__,
        })

    return pd.DataFrame(
        rows
    )



# ============================================================
# INSTITUTIONAL OPTIMIZER ENGINE
# ============================================================

class InstitutionalOptimizerEngine:
    """
    Institutional wrapper around the optimizer stack.
    """

    def mean_variance(
        self,
        inputs,
    ):

        return create_mean_variance_optimizer().optimize(
            inputs
        )

    # ----------------------------------------------------

    def risk_parity(
        self,
        inputs,
    ):

        return optimize_risk_parity(
            inputs
        )

    # ----------------------------------------------------

    def black_litterman(
        self,
        inputs,
    ):

        return run_black_litterman(
            inputs
        )

    # ----------------------------------------------------

    def multi_objective(
        self,
        inputs,
    ):

        engine = (
            create_default_multi_objective()
        )

        return engine.optimize(
            inputs
        )

    # ----------------------------------------------------

    def optimize(
        self,
        inputs,
    ):
        """
        Default institutional optimizer.
        """

        optimizer = (
            create_default_optimizer()
        )

        return optimizer.optimize(
            inputs
        )

# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [

    # Core
    "OptimizationResult",
    "OptimizationDiagnostics",

    # Config
    "OptimizerConfig",
    "MultiObjectiveConfig",

    # Engines
    "MultiObjectiveEngine",
    "OptimizationDiagnosticEngine",

    # Optimizers
    "MeanVarianceOptimizer",
    "RiskParityOptimizer",
    "BlackLittermanOptimizer",

    # Factory
    "OptimizerFactory",

    # Convenience
    "optimize_portfolio",

    "create_mean_variance_optimizer",

    "create_risk_parity_optimizer",

    "create_black_litterman_optimizer",

    "create_default_multi_objective",

    # Registry
    "register_optimizer",

    "optimizer_registry_report",
]


