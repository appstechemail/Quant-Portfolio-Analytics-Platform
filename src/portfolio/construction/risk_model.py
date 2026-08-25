"""
risk_model.py

Final institutional structure:

Framework & Validation
Risk Result Objects & Core Analytics
Volatility Models
Covariance & Correlation Models
Factor Models & Risk Decomposition
Stress Testing & Scenario Analysis
VaR & Expected Shortfall (CVaR)
Monte Carlo Risk Engine
Forecasting Models
Institutional Master Reporting Layer


==============================================================
RISK MODEL ENGINE
==============================================================

Institutional-grade portfolio risk framework.

Responsibilities
----------------
• Volatility estimation
• Covariance estimation
• Correlation analysis
• Factor risk
• Concentration risk
• Portfolio risk decomposition
• Stress testing
• Risk attribution

This file is intentionally solver-independent and can be used
by:

    optimizer.py
    rebalance.py
    diagnostics.py
    pipeline.py

==============================================================
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from dataclasses import dataclass
from dataclasses import field

from typing import Any

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# NUMERICAL CONSTANTS
# ============================================================

EPS = 1e-12

TRADING_DAYS = 252


# ============================================================
# RISK MODEL CONFIG
# ============================================================

@dataclass(slots=True)
class RiskModelConfig:
    """
    Global risk model configuration.
    """

    lookback_window: int = 252

    minimum_observations: int = 60

    annualization_factor: int = TRADING_DAYS

    shrinkage_intensity: float = 0.10

    use_ewma: bool = True

    ewma_lambda: float = 0.94

    use_factor_model: bool = False

    factor_columns: list[str] = field(
        default_factory=list
    )

    stress_confidence: float = 0.99


# ============================================================
# RISK RESULT OBJECTS
# ============================================================

@dataclass(slots=True)
class RiskResult:
    """
    Generic output returned by risk models.
    """

    success: bool

    message: str

    risk_value: float

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# PORTFOLIO RISK
# ============================================================

@dataclass(slots=True)
class PortfolioRisk:
    """
    Portfolio-level risk snapshot.
    """

    volatility: float = 0.0

    variance: float = 0.0

    tracking_error: float = 0.0

    value_at_risk: float = 0.0

    expected_shortfall: float = 0.0

    concentration_hhi: float = 0.0

    effective_positions: float = 0.0


# ============================================================
# FACTOR RISK
# ============================================================

@dataclass(slots=True)
class FactorRisk:
    """
    Factor model output.
    """

    factor_name: str

    exposure: float

    factor_volatility: float

    contribution: float


# ============================================================
# BASE RISK MODEL
# ============================================================

class BaseRiskModel(ABC):
    """
    Abstract risk model.

    Every volatility, covariance,
    factor or portfolio model
    inherits from this class.
    """

    def __init__(
        self,
        config: RiskModelConfig,
    ) -> None:

        self.config = config

    @abstractmethod
    def fit(
        self,
        returns: pd.DataFrame,
    ) -> None:
        """
        Fit model.
        """

        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
    ) -> RiskResult:
        """
        Produce risk estimate.
        """

        raise NotImplementedError


# ============================================================
# VALIDATION
# ============================================================

def validate_returns_dataframe(
    returns: pd.DataFrame,
    config: RiskModelConfig,
) -> None:
    """
    Validate returns matrix.
    """

    if returns is None:

        raise ValueError(
            "Returns dataframe is None."
        )

    if returns.empty:

        raise ValueError(
            "Returns dataframe is empty."
        )

    if len(returns) < config.minimum_observations:

        raise ValueError(
            f"Need at least "
            f"{config.minimum_observations} "
            f"observations."
        )

    if returns.isna().all().all():

        raise ValueError(
            "All returns are NaN."
        )


# ============================================================
# RETURN MATRIX CLEANING
# ============================================================

def clean_returns_matrix(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Basic institutional cleaning.
    """

    out = returns.copy()

    out = out.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    out = out.dropna(
        axis=1,
        how="all",
    )

    out = out.fillna(
        method="ffill",
    )

    out = out.fillna(
        method="bfill",
    )

    out = out.dropna(
        axis=0,
        how="all",
    )

    return out


# ============================================================
# COVARIANCE VALIDATION
# ============================================================

def validate_covariance_matrix(
    cov: pd.DataFrame,
) -> None:
    """
    Validate covariance matrix.
    """

    if cov.empty:

        raise ValueError(
            "Covariance matrix empty."
        )

    if cov.shape[0] != cov.shape[1]:

        raise ValueError(
            "Covariance matrix must be square."
        )

    if np.isnan(
        cov.values
    ).any():

        raise ValueError(
            "NaN values detected."
        )

    eigvals = np.linalg.eigvals(
        cov.values
    )

    if np.min(
        eigvals
    ) < -1e-8:

        logger.warning(
            "Covariance matrix is not PSD."
        )


# ============================================================
# POSITIVE SEMI-DEFINITE REPAIR
# ============================================================

def nearest_psd_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Repair covariance matrix.
    """

    eigvals, eigvecs = np.linalg.eigh(
        matrix
    )

    eigvals = np.maximum(
        eigvals,
        0.0,
    )

    repaired = (
        eigvecs
        @ np.diag(eigvals)
        @ eigvecs.T
    )

    return repaired


# ============================================================
# ANNUALIZATION HELPERS
# ============================================================

def annualize_volatility(
    volatility: float,
    annualization_factor: int = TRADING_DAYS,
) -> float:

    return (
        volatility
        * np.sqrt(
            annualization_factor
        )
    )


def annualize_variance(
    variance: float,
    annualization_factor: int = TRADING_DAYS,
) -> float:

    return (
        variance
        * annualization_factor
    )


# ============================================================
# CONCENTRATION HELPERS
# ============================================================

def concentration_hhi(
    weights: np.ndarray,
) -> float:
    """
    Herfindahl index.
    """

    weights = np.asarray(
        weights,
        dtype=float,
    )

    return float(
        np.sum(
            weights ** 2
        )
    )


def effective_number_of_positions(
    weights: np.ndarray,
) -> float:
    """
    Effective diversification.
    """

    hhi = concentration_hhi(
        weights
    )

    if hhi <= EPS:

        return 0.0

    return float(
        1.0 / hhi
    )


# ============================================================
# PART 2: RISK RESULT OBJECTS & CORE ANALYTICS
# ============================================================

# ============================================================
# RISK CONTRIBUTION
# ============================================================

@dataclass(slots=True)
class RiskContribution:
    """
    Marginal/component risk contribution.
    """

    asset: str

    weight: float

    marginal_risk: float

    component_risk: float

    percentage_contribution: float


# ============================================================
# RISK BUDGET
# ============================================================

@dataclass(slots=True)
class RiskBudget:
    """
    Risk budgeting output.
    """

    total_risk: float

    contributions: list[RiskContribution]


# ============================================================
# TRACKING ERROR RESULT
# ============================================================

@dataclass(slots=True)
class TrackingErrorResult:
    """
    Benchmark-relative risk.
    """

    tracking_error: float

    active_return: float

    information_ratio: float


# ============================================================
# VAR RESULT
# ============================================================

@dataclass(slots=True)
class VaRResult:
    """
    Value-at-Risk output.
    """

    confidence_level: float

    var_value: float

    horizon_days: int


# ============================================================
# EXPECTED SHORTFALL RESULT
# ============================================================

@dataclass(slots=True)
class ExpectedShortfallResult:
    """
    Conditional VaR output.
    """

    confidence_level: float

    expected_shortfall: float

    horizon_days: int


# ============================================================
# PORTFOLIO RISK REPORT
# ============================================================

@dataclass(slots=True)
class PortfolioRiskReport:
    """
    Institutional risk report.
    """

    volatility: float

    variance: float

    value_at_risk: float

    expected_shortfall: float

    concentration_hhi: float

    effective_positions: float

    tracking_error: float = 0.0

    information_ratio: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# PORTFOLIO VARIANCE
# ============================================================

def portfolio_variance(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """
    Portfolio variance.

    w'Σw
    """

    w = np.asarray(
        weights,
        dtype=float,
    )

    cov = np.asarray(
        covariance,
        dtype=float,
    )

    return float(
        w.T @ cov @ w
    )


# ============================================================
# PORTFOLIO VOLATILITY
# ============================================================

def portfolio_volatility(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """
    Portfolio volatility.
    """

    variance = portfolio_variance(
        weights,
        covariance,
    )

    return float(
        np.sqrt(
            max(
                variance,
                0.0,
            )
        )
    )


# ============================================================
# MARGINAL RISK CONTRIBUTION
# ============================================================

def marginal_risk_contribution(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """
    dσ/dw

    Marginal contribution
    to total risk.
    """

    w = np.asarray(
        weights,
        dtype=float,
    )

    cov = np.asarray(
        covariance,
        dtype=float,
    )

    port_vol = portfolio_volatility(
        w,
        cov,
    )

    if port_vol <= EPS:

        return np.zeros_like(
            w
        )

    return (
        cov @ w
    ) / port_vol


# ============================================================
# COMPONENT RISK CONTRIBUTION
# ============================================================

def component_risk_contribution(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """
    Component contribution
    to portfolio risk.
    """

    mrc = (
        marginal_risk_contribution(
            weights,
            covariance,
        )
    )

    return (
        np.asarray(
            weights,
            dtype=float,
        )
        * mrc
    )


# ============================================================
# PERCENTAGE RISK CONTRIBUTION
# ============================================================

def percentage_risk_contribution(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """
    Percent contribution
    to total portfolio risk.
    """

    crc = (
        component_risk_contribution(
            weights,
            covariance,
        )
    )

    total = crc.sum()

    if abs(total) <= EPS:

        return np.zeros_like(
            crc
        )

    return crc / total


# ============================================================
# BUILD RISK BUDGET
# ============================================================

def build_risk_budget(
    asset_names: list[str],
    weights: np.ndarray,
    covariance: np.ndarray,
) -> RiskBudget:
    """
    Build institutional risk budget.
    """

    crc = (
        component_risk_contribution(
            weights,
            covariance,
        )
    )

    prc = (
        percentage_risk_contribution(
            weights,
            covariance,
        )
    )

    contributions = []

    for (
        asset,
        weight,
        comp,
        pct,
    ) in zip(
        asset_names,
        weights,
        crc,
        prc,
    ):

        contributions.append(
            RiskContribution(
                asset=asset,
                weight=float(weight),
                marginal_risk=float(
                    comp / max(
                        weight,
                        EPS,
                    )
                ),
                component_risk=float(
                    comp
                ),
                percentage_contribution=float(
                    pct
                ),
            )
        )

    return RiskBudget(
        total_risk=float(
            crc.sum()
        ),
        contributions=contributions,
    )


# ============================================================
# TRACKING ERROR
# ============================================================

def tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualize: bool = True,
) -> float:
    """
    Tracking error.
    """

    active = (
        portfolio_returns
        - benchmark_returns
    )

    te = float(
        active.std()
    )

    if annualize:

        te = annualize_volatility(
            te
        )

    return te


# ============================================================
# INFORMATION RATIO
# ============================================================

def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Information ratio.
    """

    active = (
        portfolio_returns
        - benchmark_returns
    )

    te = tracking_error(
        portfolio_returns,
        benchmark_returns,
        annualize=False,
    )

    if te <= EPS:

        return 0.0

    return float(
        active.mean()
        / te
    )


# ============================================================
# HISTORICAL VAR
# ============================================================

def historical_var(
    returns: pd.Series,
    confidence: float = 0.99,
) -> float:
    """
    Historical Value-at-Risk.
    """

    alpha = (
        1.0 - confidence
    )

    return float(
        np.quantile(
            returns,
            alpha,
        )
    )


# ============================================================
# HISTORICAL EXPECTED SHORTFALL
# ============================================================

def historical_expected_shortfall(
    returns: pd.Series,
    confidence: float = 0.99,
) -> float:
    """
    Historical Expected Shortfall.
    """

    var = historical_var(
        returns,
        confidence,
    )

    tail = returns[
        returns <= var
    ]

    if len(tail) == 0:

        return var

    return float(
        tail.mean()
    )


# ============================================================
# BUILD PORTFOLIO RISK REPORT
# ============================================================

def build_portfolio_risk_report(
    weights: np.ndarray,
    covariance: np.ndarray,
    returns: pd.Series | None = None,
    confidence: float = 0.99,
) -> PortfolioRiskReport:
    """
    Master report object used
    throughout optimizer,
    diagnostics,
    rebalance,
    stress testing.
    """

    variance = portfolio_variance(
        weights,
        covariance,
    )

    volatility = portfolio_volatility(
        weights,
        covariance,
    )

    hhi = concentration_hhi(
        weights
    )

    eff_n = (
        effective_number_of_positions(
            weights
        )
    )

    var = 0.0
    es = 0.0

    if returns is not None:

        var = historical_var(
            returns,
            confidence,
        )

        es = (
            historical_expected_shortfall(
                returns,
                confidence,
            )
        )

    return PortfolioRiskReport(
        volatility=float(
            volatility
        ),
        variance=float(
            variance
        ),
        value_at_risk=float(
            var
        ),
        expected_shortfall=float(
            es
        ),
        concentration_hhi=float(
            hhi
        ),
        effective_positions=float(
            eff_n
        ),
    )

# ============================================================
# PART 3: VOLATILITY MODELS
# ============================================================

# ============================================================
# BASE VOLATILITY MODEL
# ============================================================

class BaseVolatilityModel(
    BaseRiskModel
):
    """
    Abstract volatility model.
    """

    @abstractmethod
    def fit(
        self,
        returns: pd.Series,
    ) -> None:

        raise NotImplementedError

    @abstractmethod
    def forecast_volatility(
        self,
    ) -> float:

        raise NotImplementedError


# ============================================================
# ROLLING VOLATILITY
# ============================================================

class RollingVolatilityModel(
    BaseVolatilityModel
):
    """
    Standard rolling volatility model.
    """

    def __init__(
        self,
        config: RiskModelConfig,
        window: int | None = None,
    ) -> None:

        super().__init__(config)

        self.window = (
            window
            or config.lookback_window
        )

        self._returns = None

    def fit(
        self,
        returns: pd.Series,
    ) -> None:

        self._returns = (
            returns
            .dropna()
            .astype(float)
        )

    def forecast_volatility(
        self,
    ) -> float:

        if (
            self._returns is None
            or len(self._returns)
            == 0
        ):

            return 0.0

        sample = (
            self._returns
            .tail(self.window)
        )

        return float(
            sample.std()
        )

    def predict(
        self,
    ) -> RiskResult:

        vol = (
            self.forecast_volatility()
        )

        return RiskResult(
            success=True,
            message="Rolling volatility",
            risk_value=vol,
        )


# ============================================================
# EWMA VOLATILITY
# ============================================================

class EWMAVolatilityModel(
    BaseVolatilityModel
):
    """
    RiskMetrics EWMA model.

    sigma²(t)
      =
    λ sigma²(t-1)
      +
    (1-λ) r²(t)
    """

    def __init__(
        self,
        config: RiskModelConfig,
    ) -> None:

        super().__init__(config)

        self.lambda_ = (
            config.ewma_lambda
        )

        self._returns = None

    def fit(
        self,
        returns: pd.Series,
    ) -> None:

        self._returns = (
            returns
            .dropna()
            .astype(float)
        )

    def forecast_volatility(
        self,
    ) -> float:

        if (
            self._returns is None
            or len(self._returns)
            == 0
        ):

            return 0.0

        r = (
            self._returns.values
        )

        variance = (
            np.var(r)
        )

        for ret in r:

            variance = (

                self.lambda_
                * variance

                +

                (1.0 - self.lambda_)
                * ret**2

            )

        return float(
            np.sqrt(
                max(
                    variance,
                    0.0,
                )
            )
        )

    def predict(
        self,
    ) -> RiskResult:

        vol = (
            self.forecast_volatility()
        )

        return RiskResult(
            success=True,
            message="EWMA volatility",
            risk_value=vol,
        )


# ============================================================
# REALIZED VOLATILITY
# ============================================================

class RealizedVolatilityModel(
    BaseVolatilityModel
):
    """
    Realized volatility.
    """

    def __init__(
        self,
        config: RiskModelConfig,
        window: int = 21,
    ) -> None:

        super().__init__(config)

        self.window = window

        self._returns = None

    def fit(
        self,
        returns: pd.Series,
    ) -> None:

        self._returns = (
            returns
            .dropna()
            .astype(float)
        )

    def forecast_volatility(
        self,
    ) -> float:

        if (
            self._returns is None
            or len(self._returns)
            == 0
        ):

            return 0.0

        r = (
            self._returns
            .tail(self.window)
        )

        realized_var = (
            np.sum(
                r**2
            )
            /
            len(r)
        )

        return float(
            np.sqrt(
                realized_var
            )
        )

    def predict(
        self,
    ) -> RiskResult:

        vol = (
            self.forecast_volatility()
        )

        return RiskResult(
            success=True,
            message="Realized volatility",
            risk_value=vol,
        )


# ============================================================
# PARKINSON VOLATILITY
# ============================================================

class ParkinsonVolatilityModel(
    BaseVolatilityModel
):
    """
    High-Low estimator.

    Requires:

        High
        Low
    """

    def __init__(
        self,
        config: RiskModelConfig,
    ) -> None:

        super().__init__(config)

        self.high = None
        self.low = None

    def fit(
        self,
        returns: pd.Series,
    ) -> None:

        raise NotImplementedError(
            "Use fit_ohlc()"
        )

    def fit_ohlc(
        self,
        high: pd.Series,
        low: pd.Series,
    ) -> None:

        self.high = high
        self.low = low

    def forecast_volatility(
        self,
    ) -> float:

        if (
            self.high is None
            or self.low is None
        ):

            return 0.0

        hl = np.log(
            self.high
            /
            self.low
        )

        variance = (

            (
                hl**2
            ).mean()

            /

            (
                4.0
                *
                np.log(2)
            )

        )

        return float(
            np.sqrt(
                variance
            )
        )

    def predict(
        self,
    ) -> RiskResult:

        vol = (
            self.forecast_volatility()
        )

        return RiskResult(
            success=True,
            message="Parkinson volatility",
            risk_value=vol,
        )


# ============================================================
# GARMAN-KLASS VOLATILITY
# ============================================================

class GarmanKlassVolatilityModel(
    BaseVolatilityModel
):
    """
    OHLC estimator.

    Requires

        Open
        High
        Low
        Close
    """

    def __init__(
        self,
        config: RiskModelConfig,
    ) -> None:

        super().__init__(config)

        self.open = None
        self.high = None
        self.low = None
        self.close = None

    def fit(
        self,
        returns: pd.Series,
    ) -> None:

        raise NotImplementedError(
            "Use fit_ohlc()"
        )

    def fit_ohlc(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> None:

        self.open = open_
        self.high = high
        self.low = low
        self.close = close

    def forecast_volatility(
        self,
    ) -> float:

        if any(
            x is None
            for x in [
                self.open,
                self.high,
                self.low,
                self.close,
            ]
        ):

            return 0.0

        log_hl = np.log(
            self.high
            /
            self.low
        )

        log_co = np.log(
            self.close
            /
            self.open
        )

        variance = (

            0.5
            * (log_hl**2)

            -

            (
                2*np.log(2)-1
            )
            * (log_co**2)

        ).mean()

        return float(
            np.sqrt(
                max(
                    variance,
                    0.0,
                )
            )
        )

    def predict(
        self,
    ) -> RiskResult:

        vol = (
            self.forecast_volatility()
        )

        return RiskResult(
            success=True,
            message="Garman-Klass volatility",
            risk_value=vol,
        )


# ============================================================
# GARCH PLACEHOLDER
# ============================================================

class GARCHVolatilityModel(
    BaseVolatilityModel
):
    """
    Future production GARCH model.

    Placeholder only.
    """

    def __init__(
        self,
        config: RiskModelConfig,
    ) -> None:

        super().__init__(config)

        self._returns = None

    def fit(
        self,
        returns: pd.Series,
    ) -> None:

        self._returns = returns

    def forecast_volatility(
        self,
    ) -> float:

        raise NotImplementedError(
            "GARCH model added later."
        )

    def predict(
        self,
    ) -> RiskResult:

        return RiskResult(
            success=False,
            message="GARCH not implemented",
            risk_value=0.0,
        )


# ============================================================
# VOLATILITY MODEL FACTORY
# ============================================================

class VolatilityModelFactory:
    """
    Factory for volatility models.
    """

    @staticmethod
    def create(
        model_name: str,
        config: RiskModelConfig,
    ) -> BaseVolatilityModel:

        model_name = (
            model_name.lower()
        )

        if model_name == "rolling":

            return (
                RollingVolatilityModel(
                    config
                )
            )

        if model_name == "ewma":

            return (
                EWMAVolatilityModel(
                    config
                )
            )

        if model_name == "realized":

            return (
                RealizedVolatilityModel(
                    config
                )
            )

        if model_name == "parkinson":

            return (
                ParkinsonVolatilityModel(
                    config
                )
            )

        if model_name == "garman_klass":

            return (
                GarmanKlassVolatilityModel(
                    config
                )
            )

        if model_name == "garch":

            return (
                GARCHVolatilityModel(
                    config
                )
            )

        raise ValueError(
            f"Unknown volatility model: {model_name}"
        )


# ============================================================
# PART 4: COVARIANCE & CORRELATION MODELS
# ============================================================

# ============================================================
# SAMPLE COVARIANCE
# ============================================================

class SampleCovarianceModel:
    """
    Standard covariance estimator.

    Uses historical returns directly.

    Institutional baseline model.
    """

    def __init__(
        self,
        min_periods: int = 60,
    ) -> None:

        self.min_periods = min_periods

    # -------------------------------------------------------

    def fit(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:

        validate_returns_matrix(
            returns,
            self.min_periods,
        )

        return returns.cov()


# ============================================================
# EWMA COVARIANCE
# ============================================================

class EWMACovarianceModel:
    """
    Exponentially weighted covariance.

    RiskMetrics-style estimator.

    Default lambda = 0.94
    """

    def __init__(
        self,
        decay: float = 0.94,
        min_periods: int = 60,
    ) -> None:

        self.decay = decay
        self.min_periods = min_periods

    # -------------------------------------------------------

    def fit(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:

        validate_returns_matrix(
            returns,
            self.min_periods,
        )

        n_assets = returns.shape[1]

        cov_matrix = np.zeros(
            (
                n_assets,
                n_assets,
            )
        )

        centered = (
            returns
            - returns.mean()
        )

        for row in centered.values:

            row = row.reshape(
                -1,
                1,
            )

            cov_matrix = (
                self.decay * cov_matrix
                +
                (1.0 - self.decay)
                * (row @ row.T)
            )

        return pd.DataFrame(
            cov_matrix,
            index=returns.columns,
            columns=returns.columns,
        )


# ============================================================
# SHRINKAGE COVARIANCE
# ============================================================

class ShrinkageCovarianceModel:
    """
    Simple Ledoit-Wolf style shrinkage.

    Shrinks covariance matrix
    toward diagonal target.
    """

    def __init__(
        self,
        shrinkage: float = 0.20,
        min_periods: int = 60,
    ) -> None:

        self.shrinkage = shrinkage
        self.min_periods = min_periods

    # -------------------------------------------------------

    def fit(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:

        validate_returns_matrix(
            returns,
            self.min_periods,
        )

        sample_cov = (
            returns.cov()
        )

        diag_target = np.diag(
            np.diag(
                sample_cov.values
            )
        )

        shrunk = (

            (1.0 - self.shrinkage)
            * sample_cov.values

            +

            self.shrinkage
            * diag_target

        )

        return pd.DataFrame(
            shrunk,
            index=sample_cov.index,
            columns=sample_cov.columns,
        )


# ============================================================
# CORRELATION MATRIX
# ============================================================

def correlation_matrix(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pearson correlation matrix.
    """

    return returns.corr()


# ============================================================
# EWMA CORRELATION
# ============================================================

def ewma_correlation_matrix(
    returns: pd.DataFrame,
    decay: float = 0.94,
) -> pd.DataFrame:
    """
    EWMA correlation matrix.
    """

    cov = (
        EWMACovarianceModel(
            decay=decay,
        )
        .fit(returns)
    )

    vol = np.sqrt(
        np.diag(
            cov.values
        )
    )

    corr = (

        cov.values

        /

        np.outer(
            vol,
            vol,
        )

    )

    corr = np.clip(
        corr,
        -1.0,
        1.0,
    )

    return pd.DataFrame(
        corr,
        index=cov.index,
        columns=cov.columns,
    )


# ============================================================
# CORRELATION STABILITY
# ============================================================

def correlation_stability(
    returns: pd.DataFrame,
    window: int = 60,
) -> float:
    """
    Measures stability of correlations.

    Higher = more stable.

    Returns average correlation
    matrix similarity.
    """

    if len(returns) < 2 * window:

        return np.nan

    corr_1 = (
        returns.iloc[
            -2 * window:-window
        ]
        .corr()
    )

    corr_2 = (
        returns.iloc[
            -window:
        ]
        .corr()
    )

    diff = np.abs(
        corr_1.values
        -
        corr_2.values
    )

    return float(
        1.0
        -
        np.nanmean(diff)
    )


# ============================================================
# EFFECTIVE BETS
# ============================================================

def effective_bets(
    covariance_matrix: pd.DataFrame,
) -> float:
    """
    Effective number of independent bets.

    Based on eigenvalues.
    """

    eigvals = np.linalg.eigvalsh(
        covariance_matrix.values
    )

    eigvals = np.maximum(
        eigvals,
        0,
    )

    total = eigvals.sum()

    if total <= EPS:

        return 0.0

    probs = eigvals / total

    entropy = -np.sum(
        probs
        * np.log(
            probs + EPS
        )
    )

    return float(
        np.exp(entropy)
    )


# ============================================================
# AVERAGE CORRELATION
# ============================================================

def average_correlation(
    correlation_matrix: pd.DataFrame,
) -> float:
    """
    Average off-diagonal correlation.
    """

    corr = (
        correlation_matrix
        .values
        .copy()
    )

    np.fill_diagonal(
        corr,
        np.nan,
    )

    return float(
        np.nanmean(corr)
    )


# ============================================================
# MAX CORRELATION PAIR
# ============================================================

def max_correlation_pair(
    correlation_matrix: pd.DataFrame,
) -> tuple[str, str, float]:
    """
    Highest correlated pair.
    """

    corr = (
        correlation_matrix
        .copy()
    )

    np.fill_diagonal(
        corr.values,
        np.nan,
    )

    idx = np.nanargmax(
        corr.values
    )

    i, j = np.unravel_index(
        idx,
        corr.shape,
    )

    return (
        corr.index[i],
        corr.columns[j],
        float(
            corr.iloc[i, j]
        ),
    )


# ============================================================
# PART 5:               FACTOR MODELS & RISK DECOMPOSITION
# ============================================================

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# FACTOR MODEL RESULT
# ============================================================

@dataclass
class FactorModelResult:

    factor_loadings: pd.DataFrame

    factor_covariance: pd.DataFrame

    specific_risk: pd.Series

    explained_variance: float


# ============================================================
# MARKET BETA MODEL
# ============================================================

def estimate_market_beta(
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
) -> pd.Series:
    """
    CAPM beta estimation.

    Beta_i =
        Cov(R_i,R_m) / Var(R_m)
    """

    market_var = market_returns.var()

    if market_var <= EPS:

        return pd.Series(
            0.0,
            index=asset_returns.columns,
        )

    betas = {}

    for col in asset_returns.columns:

        cov = np.cov(
            asset_returns[col],
            market_returns,
        )[0, 1]

        betas[col] = (
            cov / market_var
        )

    return pd.Series(betas)


# ============================================================
# MULTI FACTOR REGRESSION
# ============================================================

def estimate_factor_loadings(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    OLS factor exposures.

    Asset Return =
        Alpha
        + Beta1*Factor1
        + Beta2*Factor2
        ...
    """

    X = factor_returns.values

    X = np.column_stack(
        [
            np.ones(len(X)),
            X,
        ]
    )

    loadings = {}

    for asset in asset_returns.columns:

        y = (
            asset_returns[asset]
            .values
        )

        beta = np.linalg.lstsq(
            X,
            y,
            rcond=None,
        )[0]

        loadings[asset] = beta[1:]

    return pd.DataFrame(
        loadings,
        index=factor_returns.columns,
    ).T


# ============================================================
# FACTOR COVARIANCE
# ============================================================

def estimate_factor_covariance(
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Factor covariance matrix.
    """

    return factor_returns.cov()


# ============================================================
# SPECIFIC RISK
# ============================================================

def estimate_specific_risk(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    loadings: pd.DataFrame,
) -> pd.Series:
    """
    Residual volatility.

    Asset Risk =
        Factor Risk
        + Specific Risk
    """

    specific = {}

    X = factor_returns.values

    for asset in asset_returns.columns:

        beta = (
            loadings
            .loc[asset]
            .values
        )

        predicted = (
            X @ beta
        )

        residual = (
            asset_returns[asset]
            -
            predicted
        )

        specific[asset] = (
            residual.std()
        )

    return pd.Series(
        specific
    )


# ============================================================
# EXPLAINED VARIANCE
# ============================================================

def explained_variance_ratio(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    loadings: pd.DataFrame,
) -> float:
    """
    Average R² across assets.
    """

    X = factor_returns.values

    r2_list = []

    for asset in asset_returns.columns:

        y = (
            asset_returns[asset]
            .values
        )

        beta = (
            loadings
            .loc[asset]
            .values
        )

        y_hat = (
            X @ beta
        )

        ss_res = np.sum(
            (y - y_hat) ** 2
        )

        ss_tot = np.sum(
            (
                y
                -
                y.mean()
            ) ** 2
        )

        r2 = (
            1
            -
            ss_res / ss_tot
            if ss_tot > EPS
            else 0
        )

        r2_list.append(r2)

    return float(
        np.mean(r2_list)
    )


# ============================================================
# BUILD FACTOR MODEL
# ============================================================

def build_factor_model(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> FactorModelResult:

    loadings = (
        estimate_factor_loadings(
            asset_returns,
            factor_returns,
        )
    )

    factor_cov = (
        estimate_factor_covariance(
            factor_returns,
        )
    )

    specific = (
        estimate_specific_risk(
            asset_returns,
            factor_returns,
            loadings,
        )
    )

    explained = (
        explained_variance_ratio(
            asset_returns,
            factor_returns,
            loadings,
        )
    )

    return FactorModelResult(
        factor_loadings=loadings,
        factor_covariance=factor_cov,
        specific_risk=specific,
        explained_variance=explained,
    )


# ============================================================
# PORTFOLIO FACTOR EXPOSURE
# ============================================================

def portfolio_factor_exposure(
    weights: pd.Series,
    factor_loadings: pd.DataFrame,
) -> pd.Series:
    """
    Portfolio factor exposures.
    """

    common_assets = (
        weights.index
        .intersection(
            factor_loadings.index
        )
    )

    w = (
        weights
        .loc[common_assets]
        .values
    )

    B = (
        factor_loadings
        .loc[common_assets]
        .values
    )

    exposure = (
        w @ B
    )

    return pd.Series(
        exposure,
        index=factor_loadings.columns,
    )


# ============================================================
# FACTOR RISK CONTRIBUTION
# ============================================================

def factor_risk_contribution(
    weights: pd.Series,
    factor_loadings: pd.DataFrame,
    factor_covariance: pd.DataFrame,
) -> pd.Series:
    """
    Risk contribution from each factor.
    """

    exposure = (
        portfolio_factor_exposure(
            weights,
            factor_loadings,
        )
    )

    factor_risk = (

        exposure.values

        *

        (
            factor_covariance.values
            @ exposure.values
        )

    )

    return pd.Series(
        factor_risk,
        index=exposure.index,
    )


# ============================================================
# PORTFOLIO VOLATILITY
# ============================================================

def portfolio_volatility(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:

    return float(
        np.sqrt(
            weights.T
            @ covariance
            @ weights
        )
    )


# ============================================================
# MARGINAL CONTRIBUTION TO RISK
# ============================================================

def marginal_contribution_to_risk(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """
    MCTR

    dSigma/dWeight
    """

    sigma = portfolio_volatility(
        weights,
        covariance,
    )

    if sigma <= EPS:

        return np.zeros_like(
            weights
        )

    return (
        covariance @ weights
    ) / sigma


# ============================================================
# COMPONENT CONTRIBUTION TO RISK
# ============================================================

def component_contribution_to_risk(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """
    CCTR

    Weight * MCTR
    """

    mctr = (
        marginal_contribution_to_risk(
            weights,
            covariance,
        )
    )

    return (
        weights * mctr
    )


# ============================================================
# PERCENT RISK CONTRIBUTION
# ============================================================

def percent_risk_contribution(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:

    cctr = (
        component_contribution_to_risk(
            weights,
            covariance,
        )
    )

    total = cctr.sum()

    if total <= EPS:

        return np.zeros_like(
            cctr
        )

    return (
        cctr / total
    )


# ============================================================
# RISK BUDGET ANALYSIS
# ============================================================

def risk_budget_analysis(
    weights: pd.Series,
    covariance_matrix: pd.DataFrame,
) -> pd.DataFrame:

    common_assets = (
        weights.index
        .intersection(
            covariance_matrix.index
        )
    )

    w = (
        weights
        .loc[common_assets]
        .values
    )

    cov = (
        covariance_matrix
        .loc[
            common_assets,
            common_assets,
        ]
        .values
    )

    mctr = (
        marginal_contribution_to_risk(
            w,
            cov,
        )
    )

    cctr = (
        component_contribution_to_risk(
            w,
            cov,
        )
    )

    pct = (
        percent_risk_contribution(
            w,
            cov,
        )
    )

    return pd.DataFrame(
        {
            "Weight": w,
            "MCTR": mctr,
            "CCTR": cctr,
            "RiskPct": pct,
        },
        index=common_assets,
    )


# ============================================================
# PART 6: STRESS TESTING & SCENARIO ANALYSIS
# ============================================================

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================
# STRESS RESULT
# ============================================================

@dataclass
class StressScenarioResult:

    scenario_name: str

    portfolio_return: float

    pnl: float

    worst_asset: str | None

    best_asset: str | None


# ============================================================
# HISTORICAL STRESS RESULT
# ============================================================

@dataclass
class HistoricalStressResult:

    worst_day_return: float

    worst_day_date: pd.Timestamp | None

    worst_week_return: float

    worst_month_return: float

    average_tail_return: float


# ============================================================
# PORTFOLIO RETURN
# ============================================================

def scenario_portfolio_return(
    weights: pd.Series,
    shocked_returns: pd.Series,
) -> float:
    """
    Portfolio return under scenario.
    """

    common = (
        weights.index
        .intersection(
            shocked_returns.index
        )
    )

    if len(common) == 0:

        return 0.0

    return float(
        np.dot(
            weights.loc[common],
            shocked_returns.loc[common],
        )
    )


# ============================================================
# SIMPLE SHOCK SCENARIO
# ============================================================

def apply_uniform_shock(
    asset_names: pd.Index,
    shock_pct: float,
) -> pd.Series:
    """
    Example:

    -10% market crash
    """

    return pd.Series(
        shock_pct,
        index=asset_names,
    )


# ============================================================
# CUSTOM SHOCK SCENARIO
# ============================================================

def apply_custom_shock(
    asset_names: pd.Index,
    shocks: dict,
) -> pd.Series:
    """
    Example:

    {
        "AAPL": -0.15,
        "MSFT": -0.08
    }
    """

    result = pd.Series(
        0.0,
        index=asset_names,
    )

    for asset, shock in shocks.items():

        if asset in result.index:

            result.loc[asset] = shock

    return result


# ============================================================
# FACTOR SHOCK SCENARIO
# ============================================================

def factor_shock_returns(
    factor_loadings: pd.DataFrame,
    factor_shocks: pd.Series,
) -> pd.Series:
    """
    Asset Shock

    = B * FactorShock
    """

    common_factors = (
        factor_loadings.columns
        .intersection(
            factor_shocks.index
        )
    )

    if len(common_factors) == 0:

        return pd.Series(
            0.0,
            index=factor_loadings.index,
        )

    B = factor_loadings[
        common_factors
    ]

    F = factor_shocks[
        common_factors
    ]

    shocked = B.values @ F.values

    return pd.Series(
        shocked,
        index=factor_loadings.index,
    )


# ============================================================
# CORRELATION BREAKDOWN
# ============================================================

def correlation_breakdown_covariance(
    covariance_matrix: pd.DataFrame,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Simulates crisis correlation regime.

    Correlations rise sharply.
    """

    vol = np.sqrt(
        np.diag(
            covariance_matrix
        )
    )

    corr = (
        covariance_matrix
        .values
        /
        np.outer(vol, vol)
    )

    corr = np.clip(
        corr * multiplier,
        -1.0,
        1.0,
    )

    stressed = (
        corr
        *
        np.outer(vol, vol)
    )

    return pd.DataFrame(
        stressed,
        index=covariance_matrix.index,
        columns=covariance_matrix.columns,
    )


# ============================================================
# PORTFOLIO VOL UNDER STRESS
# ============================================================

def stressed_volatility(
    weights: pd.Series,
    stressed_covariance: pd.DataFrame,
) -> float:

    common = (
        weights.index
        .intersection(
            stressed_covariance.index
        )
    )

    if len(common) == 0:

        return 0.0

    w = (
        weights
        .loc[common]
        .values
    )

    cov = (
        stressed_covariance
        .loc[
            common,
            common,
        ]
        .values
    )

    return float(
        np.sqrt(
            w.T @ cov @ w
        )
    )


# ============================================================
# HISTORICAL WORST DAY
# ============================================================

def historical_worst_day(
    portfolio_returns: pd.Series,
) -> tuple:

    idx = (
        portfolio_returns.idxmin()
    )

    value = (
        portfolio_returns.min()
    )

    return idx, float(value)


# ============================================================
# HISTORICAL WORST WINDOW
# ============================================================

def worst_rolling_window(
    portfolio_returns: pd.Series,
    window: int,
) -> float:

    if len(portfolio_returns) < window:

        return float(
            portfolio_returns.sum()
        )

    rolling = (

        portfolio_returns

        .rolling(window)

        .sum()

    )

    return float(
        rolling.min()
    )


# ============================================================
# HISTORICAL STRESS REPORT
# ============================================================

def historical_stress_test(
    portfolio_returns: pd.Series,
) -> HistoricalStressResult:

    worst_date, worst_day = (
        historical_worst_day(
            portfolio_returns
        )
    )

    worst_week = (
        worst_rolling_window(
            portfolio_returns,
            5,
        )
    )

    worst_month = (
        worst_rolling_window(
            portfolio_returns,
            21,
        )
    )

    tail = (
        portfolio_returns[
            portfolio_returns
            <
            portfolio_returns.quantile(
                0.05
            )
        ]
    )

    tail_avg = (
        tail.mean()
        if len(tail)
        else 0.0
    )

    return HistoricalStressResult(
        worst_day_return=worst_day,
        worst_day_date=worst_date,
        worst_week_return=worst_week,
        worst_month_return=worst_month,
        average_tail_return=float(
            tail_avg
        ),
    )


# ============================================================
# SCENARIO PNL
# ============================================================

def scenario_pnl(
    weights: pd.Series,
    scenario_returns: pd.Series,
    portfolio_value: float,
) -> float:

    r = scenario_portfolio_return(
        weights,
        scenario_returns,
    )

    return float(
        r * portfolio_value
    )


# ============================================================
# RUN STRESS SCENARIO
# ============================================================

def run_stress_scenario(
    weights: pd.Series,
    scenario_returns: pd.Series,
    portfolio_value: float,
    scenario_name: str,
) -> StressScenarioResult:

    pnl = scenario_pnl(
        weights,
        scenario_returns,
        portfolio_value,
    )

    portfolio_ret = (
        scenario_portfolio_return(
            weights,
            scenario_returns,
        )
    )

    worst_asset = (
        scenario_returns.idxmin()
        if len(scenario_returns)
        else None
    )

    best_asset = (
        scenario_returns.idxmax()
        if len(scenario_returns)
        else None
    )

    return StressScenarioResult(
        scenario_name=scenario_name,
        portfolio_return=float(
            portfolio_ret
        ),
        pnl=float(pnl),
        worst_asset=worst_asset,
        best_asset=best_asset,
    )


# ============================================================
# STRESS REPORT TABLE
# ============================================================

def stress_report_table(
    results: list[
        StressScenarioResult
    ],
) -> pd.DataFrame:

    rows = []

    for r in results:

        rows.append(
            {
                "Scenario":
                    r.scenario_name,

                "PortfolioReturn":
                    r.portfolio_return,

                "PnL":
                    r.pnl,

                "WorstAsset":
                    r.worst_asset,

                "BestAsset":
                    r.best_asset,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# INSTITUTIONAL STRESS DASHBOARD
# ============================================================

def institutional_stress_dashboard(
    portfolio_returns: pd.Series,
    stress_results: list[
        StressScenarioResult
    ],
) -> dict:

    return {

        "HistoricalStress":
            historical_stress_test(
                portfolio_returns
            ),

        "ScenarioReport":
            stress_report_table(
                stress_results
            ),
    }


# ============================================================
# PART 7: VAR & EXPECTED SHORTFALL (CVAR)
# ============================================================

from dataclasses import dataclass
from scipy.stats import norm


# ============================================================
# VAR RESULT
# ============================================================

@dataclass
class VaRResult:

    confidence_level: float

    var: float

    cvar: float

    methodology: str


# ============================================================
# HISTORICAL VAR
# ============================================================

def historical_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Historical VaR.

    Example:
        95% VaR
    """

    if len(returns) == 0:

        return 0.0

    alpha = (
        1.0
        - confidence_level
    )

    return float(
        np.quantile(
            returns,
            alpha,
        )
    )


# ============================================================
# HISTORICAL CVAR
# ============================================================

def historical_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Historical Expected Shortfall.
    """

    if len(returns) == 0:

        return 0.0

    var = historical_var(
        returns,
        confidence_level,
    )

    tail = returns[
        returns <= var
    ]

    if len(tail) == 0:

        return float(var)

    return float(
        tail.mean()
    )


# ============================================================
# PARAMETRIC VAR
# ============================================================

def parametric_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Gaussian VaR.
    """

    if len(returns) == 0:

        return 0.0

    mu = float(
        returns.mean()
    )

    sigma = float(
        returns.std()
    )

    z = norm.ppf(
        1.0 - confidence_level
    )

    return float(
        mu + z * sigma
    )


# ============================================================
# PARAMETRIC CVAR
# ============================================================

def parametric_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Gaussian Expected Shortfall.
    """

    if len(returns) == 0:

        return 0.0

    mu = float(
        returns.mean()
    )

    sigma = float(
        returns.std()
    )

    alpha = (
        1.0
        - confidence_level
    )

    z = norm.ppf(alpha)

    cvar = (

        mu

        - sigma

        * norm.pdf(z)

        / alpha

    )

    return float(cvar)


# ============================================================
# MONTE CARLO VAR
# ============================================================

def monte_carlo_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    simulations: int = 10000,
    random_state: int = 42,
) -> float:
    """
    Gaussian Monte Carlo VaR.
    """

    if len(returns) == 0:

        return 0.0

    rng = np.random.default_rng(
        random_state
    )

    mu = float(
        returns.mean()
    )

    sigma = float(
        returns.std()
    )

    sims = rng.normal(
        loc=mu,
        scale=sigma,
        size=simulations,
    )

    alpha = (
        1.0
        - confidence_level
    )

    return float(
        np.quantile(
            sims,
            alpha,
        )
    )


# ============================================================
# MONTE CARLO CVAR
# ============================================================

def monte_carlo_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95,
    simulations: int = 10000,
    random_state: int = 42,
) -> float:
    """
    Monte Carlo Expected Shortfall.
    """

    if len(returns) == 0:

        return 0.0

    rng = np.random.default_rng(
        random_state
    )

    mu = float(
        returns.mean()
    )

    sigma = float(
        returns.std()
    )

    sims = rng.normal(
        loc=mu,
        scale=sigma,
        size=simulations,
    )

    alpha = (
        1.0
        - confidence_level
    )

    var = np.quantile(
        sims,
        alpha,
    )

    tail = sims[
        sims <= var
    ]

    return float(
        np.mean(tail)
    )


# ============================================================
# ROLLING VAR
# ============================================================

def rolling_var(
    returns: pd.Series,
    window: int = 252,
    confidence_level: float = 0.95,
) -> pd.Series:
    """
    Rolling Historical VaR.
    """

    return (

        returns

        .rolling(window)

        .apply(
            lambda x:
            historical_var(
                pd.Series(x),
                confidence_level,
            ),
            raw=False,
        )

    )


# ============================================================
# VAR BREACHES
# ============================================================

def var_breaches(
    returns: pd.Series,
    rolling_var_series: pd.Series,
) -> pd.Series:
    """
    VaR violation indicator.
    """

    common = pd.concat(
        [
            returns,
            rolling_var_series,
        ],
        axis=1,
    ).dropna()

    return (
        common.iloc[:, 0]
        <
        common.iloc[:, 1]
    ).astype(int)


# ============================================================
# BREACH RATE
# ============================================================

def breach_rate(
    returns: pd.Series,
    rolling_var_series: pd.Series,
) -> float:

    breaches = var_breaches(
        returns,
        rolling_var_series,
    )

    if len(breaches) == 0:

        return 0.0

    return float(
        breaches.mean()
    )


# ============================================================
# KUPIEC TEST
# ============================================================

def kupiec_failure_rate(
    returns: pd.Series,
    rolling_var_series: pd.Series,
) -> dict:
    """
    Simple VaR coverage statistics.
    """

    breaches = var_breaches(
        returns,
        rolling_var_series,
    )

    n = len(breaches)

    if n == 0:

        return {}

    failures = int(
        breaches.sum()
    )

    return {

        "observations": n,

        "failures": failures,

        "failure_rate":
            failures / n,

    }


# ============================================================
# PORTFOLIO VAR REPORT
# ============================================================

def portfolio_var_report(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> dict:

    return {

        "HistoricalVaR":
            historical_var(
                returns,
                confidence_level,
            ),

        "HistoricalCVaR":
            historical_cvar(
                returns,
                confidence_level,
            ),

        "ParametricVaR":
            parametric_var(
                returns,
                confidence_level,
            ),

        "ParametricCVaR":
            parametric_cvar(
                returns,
                confidence_level,
            ),

        "MonteCarloVaR":
            monte_carlo_var(
                returns,
                confidence_level,
            ),

        "MonteCarloCVaR":
            monte_carlo_cvar(
                returns,
                confidence_level,
            ),
    }


# ============================================================
# INSTITUTIONAL VAR DASHBOARD
# ============================================================

def institutional_var_dashboard(
    returns: pd.Series,
    confidence_level: float = 0.95,
    window: int = 252,
) -> dict:

    rolling = rolling_var(
        returns,
        window,
        confidence_level,
    )

    return {

        "RiskMetrics":
            portfolio_var_report(
                returns,
                confidence_level,
            ),

        "RollingVaR":
            rolling,

        "Coverage":
            kupiec_failure_rate(
                returns,
                rolling,
            ),

        "BreachRate":
            breach_rate(
                returns,
                rolling,
            ),
    }


# ============================================================
# PART 8: MONTE CARLO RISK ENGINE
# ============================================================

from dataclasses import dataclass


# ============================================================
# MONTE CARLO RESULT
# ============================================================

@dataclass
class MonteCarloResult:

    simulations: int

    expected_return: float

    expected_volatility: float

    probability_loss: float

    probability_drawdown_10pct: float

    probability_drawdown_20pct: float

    worst_simulated_return: float

    best_simulated_return: float


# ============================================================
# CHOLESKY MATRIX
# ============================================================

def covariance_cholesky(
    covariance_matrix: pd.DataFrame,
) -> np.ndarray:
    """
    Stable Cholesky decomposition.
    """

    cov = covariance_matrix.values

    jitter = 1e-10

    return np.linalg.cholesky(
        cov
        + np.eye(len(cov)) * jitter
    )


# ============================================================
# MULTI-ASSET RETURN SIMULATION
# ============================================================

def simulate_asset_returns(
    mean_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    simulations: int = 10000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Simulate correlated asset returns.

    Output:
        [simulations x assets]
    """

    rng = np.random.default_rng(
        random_state
    )

    n_assets = len(
        mean_returns
    )

    Z = rng.normal(
        0,
        1,
        (
            simulations,
            n_assets,
        ),
    )

    L = covariance_cholesky(
        covariance_matrix
    )

    correlated = (
        Z @ L.T
    )

    correlated += (
        mean_returns.values
    )

    return correlated


# ============================================================
# PORTFOLIO RETURN SIMULATION
# ============================================================

def simulate_portfolio_returns(
    weights: pd.Series,
    mean_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    simulations: int = 10000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Simulate portfolio return distribution.
    """

    common = (
        weights.index
        .intersection(
            mean_returns.index
        )
        .intersection(
            covariance_matrix.index
        )
    )

    if len(common) == 0:

        return np.array([])

    w = (
        weights.loc[common]
        .values
    )

    mu = (
        mean_returns.loc[common]
    )

    cov = (
        covariance_matrix.loc[
            common,
            common,
        ]
    )

    asset_sims = (
        simulate_asset_returns(
            mu,
            cov,
            simulations,
            random_state,
        )
    )

    portfolio_sims = (
        asset_sims @ w
    )

    return portfolio_sims


# ============================================================
# TERMINAL VALUE SIMULATION
# ============================================================

def simulate_terminal_values(
    portfolio_returns: np.ndarray,
    initial_capital: float = 1.0,
) -> np.ndarray:
    """
    One-period terminal values.
    """

    return (

        initial_capital

        * (1.0 + portfolio_returns)

    )


# ============================================================
# PROBABILITY OF LOSS
# ============================================================

def probability_of_loss(
    simulated_returns: np.ndarray,
) -> float:

    if len(simulated_returns) == 0:

        return 0.0

    return float(
        np.mean(
            simulated_returns < 0
        )
    )


# ============================================================
# PROBABILITY OF DRAWDOWN
# ============================================================

def probability_drawdown(
    simulated_returns: np.ndarray,
    threshold: float,
) -> float:

    if len(simulated_returns) == 0:

        return 0.0

    return float(
        np.mean(
            simulated_returns
            <= -abs(threshold)
        )
    )


# ============================================================
# EXPECTED TAIL LOSS
# ============================================================

def expected_tail_loss(
    simulated_returns: np.ndarray,
    percentile: float = 0.05,
) -> float:

    if len(simulated_returns) == 0:

        return 0.0

    cutoff = np.quantile(
        simulated_returns,
        percentile,
    )

    tail = simulated_returns[
        simulated_returns
        <= cutoff
    ]

    if len(tail) == 0:

        return float(cutoff)

    return float(
        tail.mean()
    )


# ============================================================
# RISK OF RUIN
# ============================================================

def risk_of_ruin(
    terminal_values: np.ndarray,
    ruin_threshold: float = 0.5,
) -> float:
    """
    Probability capital falls below threshold.

    Example:
        50% capital loss
    """

    if len(terminal_values) == 0:

        return 0.0

    return float(
        np.mean(
            terminal_values
            <= ruin_threshold
        )
    )


# ============================================================
# MONTE CARLO SUMMARY
# ============================================================

def monte_carlo_summary(
    simulated_returns: np.ndarray,
) -> MonteCarloResult:

    if len(simulated_returns) == 0:

        return MonteCarloResult(
            simulations=0,
            expected_return=0.0,
            expected_volatility=0.0,
            probability_loss=0.0,
            probability_drawdown_10pct=0.0,
            probability_drawdown_20pct=0.0,
            worst_simulated_return=0.0,
            best_simulated_return=0.0,
        )

    return MonteCarloResult(

        simulations=len(
            simulated_returns
        ),

        expected_return=float(
            np.mean(
                simulated_returns
            )
        ),

        expected_volatility=float(
            np.std(
                simulated_returns
            )
        ),

        probability_loss=
        probability_of_loss(
            simulated_returns
        ),

        probability_drawdown_10pct=
        probability_drawdown(
            simulated_returns,
            0.10,
        ),

        probability_drawdown_20pct=
        probability_drawdown(
            simulated_returns,
            0.20,
        ),

        worst_simulated_return=float(
            np.min(
                simulated_returns
            )
        ),

        best_simulated_return=float(
            np.max(
                simulated_returns
            )
        ),
    )


# ============================================================
# INSTITUTIONAL MONTE CARLO ENGINE
# ============================================================

def run_monte_carlo_engine(
    weights: pd.Series,
    mean_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    simulations: int = 10000,
    initial_capital: float = 1.0,
    random_state: int = 42,
) -> dict:
    """
    Full institutional Monte Carlo workflow.
    """

    portfolio_returns = (
        simulate_portfolio_returns(
            weights,
            mean_returns,
            covariance_matrix,
            simulations,
            random_state,
        )
    )

    terminal_values = (
        simulate_terminal_values(
            portfolio_returns,
            initial_capital,
        )
    )

    summary = (
        monte_carlo_summary(
            portfolio_returns
        )
    )

    return {

        "Summary":
            summary,

        "ExpectedTailLoss":
            expected_tail_loss(
                portfolio_returns
            ),

        "RiskOfRuin":
            risk_of_ruin(
                terminal_values
            ),

        "SimulatedReturns":
            portfolio_returns,

        "TerminalValues":
            terminal_values,
    }


# ============================================================
# PART 9: FORECASTING MODELS
# ============================================================

from dataclasses import dataclass


# ============================================================
# FORECAST RESULT
# ============================================================

@dataclass
class RiskForecast:

    forecast_volatility: float

    forecast_variance: float

    forecast_covariance: pd.DataFrame | None

    forecast_correlation: pd.DataFrame | None

    methodology: str


# ============================================================
# EWMA VOLATILITY
# ============================================================

def ewma_variance(
    returns: pd.Series,
    decay: float = 0.94,
) -> float:
    """
    RiskMetrics EWMA variance.
    """

    returns = (
        returns.dropna()
    )

    if len(returns) == 0:

        return 0.0

    variance = (
        returns.iloc[0] ** 2
    )

    for r in returns.iloc[1:]:

        variance = (

            decay * variance

            +

            (1.0 - decay)

            * (r ** 2)

        )

    return float(
        variance
    )


# ============================================================
# EWMA VOL FORECAST
# ============================================================

def ewma_volatility_forecast(
    returns: pd.Series,
    decay: float = 0.94,
) -> float:

    return float(
        np.sqrt(
            ewma_variance(
                returns,
                decay,
            )
        )
    )


# ============================================================
# ROLLING VOL FORECAST
# ============================================================

def rolling_volatility_forecast(
    returns: pd.Series,
    window: int = 63,
) -> float:

    returns = (
        returns.dropna()
    )

    if len(returns) < 2:

        return 0.0

    sample = (
        returns.tail(window)
    )

    return float(
        sample.std()
    )


# ============================================================
# REGIME VOL FORECAST
# ============================================================

def regime_adjusted_volatility(
    returns: pd.Series,
    current_regime: str | None,
    regime_scalars: dict | None = None,
) -> float:
    """
    Example:

    BULL -> 0.8

    BEAR -> 1.3
    """

    base_vol = (
        ewma_volatility_forecast(
            returns
        )
    )

    if (
        regime_scalars is None
        or current_regime is None
    ):
        return base_vol

    scalar = regime_scalars.get(
        current_regime,
        1.0,
    )

    return float(
        base_vol * scalar
    )


# ============================================================
# EWMA COVARIANCE
# ============================================================

def ewma_covariance_matrix(
    returns_df: pd.DataFrame,
    decay: float = 0.94,
) -> pd.DataFrame:
    """
    Forward covariance forecast.
    """

    returns_df = (
        returns_df.dropna()
    )

    if len(returns_df) == 0:

        return pd.DataFrame()

    assets = (
        returns_df.columns
    )

    cov = np.zeros(
        (
            len(assets),
            len(assets),
        )
    )

    for _, row in returns_df.iterrows():

        r = row.values.reshape(
            -1,
            1,
        )

        cov = (

            decay * cov

            +

            (1.0 - decay)

            * (r @ r.T)

        )

    return pd.DataFrame(
        cov,
        index=assets,
        columns=assets,
    )


# ============================================================
# FORECAST CORRELATION
# ============================================================

def forecast_correlation_matrix(
    covariance_matrix: pd.DataFrame,
) -> pd.DataFrame:

    if covariance_matrix.empty:

        return covariance_matrix

    vol = np.sqrt(
        np.diag(
            covariance_matrix
        )
    )

    corr = (

        covariance_matrix.values

        /

        np.outer(
            vol,
            vol,
        )

    )

    corr = np.nan_to_num(
        corr
    )

    return pd.DataFrame(
        corr,
        index=covariance_matrix.index,
        columns=covariance_matrix.columns,
    )


# ============================================================
# FORWARD PORTFOLIO VOL
# ============================================================

def forecast_portfolio_volatility(
    weights: pd.Series,
    forecast_covariance: pd.DataFrame,
) -> float:

    common = (
        weights.index
        .intersection(
            forecast_covariance.index
        )
    )

    if len(common) == 0:

        return 0.0

    w = (
        weights
        .loc[common]
        .values
    )

    cov = (
        forecast_covariance
        .loc[
            common,
            common,
        ]
        .values
    )

    return float(
        np.sqrt(
            w.T @ cov @ w
        )
    )


# ============================================================
# FORECAST TRACKER
# ============================================================

def rolling_forecast_series(
    returns: pd.Series,
    window: int = 252,
    decay: float = 0.94,
) -> pd.Series:
    """
    Historical forecast record.
    """

    forecasts = []

    index = []

    for i in range(
        window,
        len(returns),
    ):

        sample = (
            returns.iloc[
                i - window:i
            ]
        )

        forecast = (
            ewma_volatility_forecast(
                sample,
                decay,
            )
        )

        forecasts.append(
            forecast
        )

        index.append(
            returns.index[i]
        )

    return pd.Series(
        forecasts,
        index=index,
    )


# ============================================================
# FORECAST ERROR
# ============================================================

def forecast_error_statistics(
    realized_vol: pd.Series,
    forecast_vol: pd.Series,
) -> dict:

    common = pd.concat(
        [
            realized_vol,
            forecast_vol,
        ],
        axis=1,
    ).dropna()

    if len(common) == 0:

        return {}

    error = (
        common.iloc[:, 0]
        -
        common.iloc[:, 1]
    )

    return {

        "MAE":
            float(
                np.mean(
                    np.abs(error)
                )
            ),

        "RMSE":
            float(
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                )
            ),

        "Bias":
            float(
                error.mean()
            ),
    }


# ============================================================
# RISK FORECAST ENGINE
# ============================================================

def run_risk_forecast_engine(
    returns_df: pd.DataFrame,
    portfolio_weights: pd.Series | None = None,
    decay: float = 0.94,
) -> RiskForecast:
    """
    Institutional risk forecast.
    """

    covariance = (
        ewma_covariance_matrix(
            returns_df,
            decay,
        )
    )

    correlation = (
        forecast_correlation_matrix(
            covariance
        )
    )

    if (
        portfolio_weights is not None
        and
        not covariance.empty
    ):

        portfolio_vol = (
            forecast_portfolio_volatility(
                portfolio_weights,
                covariance,
            )
        )

    else:

        portfolio_vol = 0.0

    return RiskForecast(

        forecast_volatility=
            float(
                portfolio_vol
            ),

        forecast_variance=
            float(
                portfolio_vol ** 2
            ),

        forecast_covariance=
            covariance,

        forecast_correlation=
            correlation,

        methodology=
            "EWMA",
    )


# ============================================================
# FORECAST DASHBOARD
# ============================================================

def institutional_forecast_dashboard(
    returns_df: pd.DataFrame,
    weights: pd.Series | None = None,
) -> dict:

    forecast = (
        run_risk_forecast_engine(
            returns_df,
            weights,
        )
    )

    return {

        "Methodology":
            forecast.methodology,

        "ForecastVolatility":
            forecast.forecast_volatility,

        "ForecastVariance":
            forecast.forecast_variance,

        "ForecastCovariance":
            forecast.forecast_covariance,

        "ForecastCorrelation":
            forecast.forecast_correlation,
    }


# ============================================================
# PART 10: INSTITUTIONAL MASTER REPORTING LAYER
# ============================================================

from dataclasses import dataclass, field


# ============================================================
# MASTER REPORT
# ============================================================

@dataclass
class InstitutionalRiskReport:

    report_date: pd.Timestamp

    portfolio_name: str

    summary: dict = field(
        default_factory=dict
    )

    volatility: dict = field(
        default_factory=dict
    )

    covariance: dict = field(
        default_factory=dict
    )

    factor_risk: dict = field(
        default_factory=dict
    )

    stress_testing: dict = field(
        default_factory=dict
    )

    var_report: dict = field(
        default_factory=dict
    )

    monte_carlo: dict = field(
        default_factory=dict
    )

    forecasting: dict = field(
        default_factory=dict
    )


# ============================================================
# PORTFOLIO SUMMARY
# ============================================================

def build_portfolio_risk_summary(
    portfolio_weights: pd.Series,
) -> dict:

    if len(portfolio_weights) == 0:

        return {}

    hhi = float(
        np.sum(
            portfolio_weights.values ** 2
        )
    )

    effective_n = (

        1.0 / hhi

        if hhi > 0

        else 0.0

    )

    return {

        "Positions":
            len(
                portfolio_weights
            ),

        "MaxWeight":
            float(
                portfolio_weights.max()
            ),

        "MinWeight":
            float(
                portfolio_weights.min()
            ),

        "GrossExposure":
            float(
                np.abs(
                    portfolio_weights
                ).sum()
            ),

        "NetExposure":
            float(
                portfolio_weights.sum()
            ),

        "EffectiveN":
            float(
                effective_n
            ),
    }


# ============================================================
# VOLATILITY REPORT
# ============================================================

def build_volatility_report(
    returns: pd.Series,
) -> dict:

    if len(returns) == 0:

        return {}

    return {

        "HistoricalVol":
            float(
                returns.std()
            ),

        "EWMAForecast":
            ewma_volatility_forecast(
                returns
            ),

        "RollingForecast":
            rolling_volatility_forecast(
                returns
            ),
    }


# ============================================================
# COVARIANCE REPORT
# ============================================================

def build_covariance_report(
    covariance_matrix: pd.DataFrame,
) -> dict:

    if covariance_matrix.empty:

        return {}

    return {

        "Assets":
            len(
                covariance_matrix
            ),

        "AverageVariance":
            float(
                np.mean(
                    np.diag(
                        covariance_matrix
                    )
                )
            ),

        "AverageCovariance":
            float(
                covariance_matrix
                .values
                .mean()
            ),
    }


# ============================================================
# FACTOR REPORT
# ============================================================

def build_factor_risk_report(
    factor_exposures: pd.Series | None,
) -> dict:

    if factor_exposures is None:

        return {}

    return {

        "FactorExposure":
            factor_exposures.to_dict()
    }


# ============================================================
# MASTER REPORT BUILDER
# ============================================================

def build_institutional_risk_report(
    *,
    report_date: pd.Timestamp,
    portfolio_name: str,
    portfolio_weights: pd.Series,
    portfolio_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    factor_exposures: pd.Series | None = None,
    stress_results: list | None = None,
    monte_carlo_results: dict | None = None,
) -> InstitutionalRiskReport:
    """
    Single institutional report object.
    """

    report = InstitutionalRiskReport(

        report_date=report_date,

        portfolio_name=portfolio_name,
    )

    # ---------------------------------
    # Summary
    # ---------------------------------

    report.summary = (
        build_portfolio_risk_summary(
            portfolio_weights
        )
    )

    # ---------------------------------
    # Volatility
    # ---------------------------------

    report.volatility = (
        build_volatility_report(
            portfolio_returns
        )
    )

    # ---------------------------------
    # Covariance
    # ---------------------------------

    report.covariance = (
        build_covariance_report(
            covariance_matrix
        )
    )

    # ---------------------------------
    # Factor Risk
    # ---------------------------------

    report.factor_risk = (
        build_factor_risk_report(
            factor_exposures
        )
    )

    # ---------------------------------
    # Stress
    # ---------------------------------

    if stress_results is not None:

        report.stress_testing = {

            "ScenarioCount":
                len(
                    stress_results
                ),

            "Scenarios":
                stress_report_table(
                    stress_results
                ),
        }

    # ---------------------------------
    # VaR
    # ---------------------------------

    report.var_report = (
        portfolio_var_report(
            portfolio_returns
        )
    )

    # ---------------------------------
    # Monte Carlo
    # ---------------------------------

    if monte_carlo_results is not None:

        report.monte_carlo = {

            "ExpectedTailLoss":
                monte_carlo_results.get(
                    "ExpectedTailLoss"
                ),

            "RiskOfRuin":
                monte_carlo_results.get(
                    "RiskOfRuin"
                ),

            "Summary":
                monte_carlo_results.get(
                    "Summary"
                ),
        }

    # ---------------------------------
    # Forecasting
    # ---------------------------------

    report.forecasting = (
        institutional_forecast_dashboard(
            covariance_matrix
            if isinstance(
                covariance_matrix,
                pd.DataFrame,
            )
            else pd.DataFrame(),
            portfolio_weights,
        )
    )

    return report


# ============================================================
# REPORT TO DICTIONARY
# ============================================================

def risk_report_to_dict(
    report: InstitutionalRiskReport,
) -> dict:

    return {

        "ReportDate":
            report.report_date,

        "Portfolio":
            report.portfolio_name,

        "Summary":
            report.summary,

        "Volatility":
            report.volatility,

        "Covariance":
            report.covariance,

        "FactorRisk":
            report.factor_risk,

        "StressTesting":
            report.stress_testing,

        "VaR":
            report.var_report,

        "MonteCarlo":
            report.monte_carlo,

        "Forecasting":
            report.forecasting,
    }


# ============================================================
# REPORT TO DATAFRAME
# ============================================================

def risk_report_table(
    report: InstitutionalRiskReport,
) -> pd.DataFrame:

    rows = []

    for section_name, section in {

        "Summary":
            report.summary,

        "Volatility":
            report.volatility,

        "Covariance":
            report.covariance,

        "VaR":
            report.var_report,

    }.items():

        if isinstance(
            section,
            dict,
        ):

            for key, value in section.items():

                rows.append({

                    "Section":
                        section_name,

                    "Metric":
                        key,

                    "Value":
                        value,
                })

    return pd.DataFrame(
        rows
    )


# ============================================================
# EXPORT REPORT
# ============================================================

def export_risk_report_csv(
    report: InstitutionalRiskReport,
    path: str,
) -> None:

    risk_report_table(
        report
    ).to_csv(
        path,
        index=False,
    )


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

def executive_risk_dashboard(
    report: InstitutionalRiskReport,
) -> dict:
    """
    High-level dashboard for CIO / PM.
    """

    return {

        "Portfolio":
            report.portfolio_name,

        "Positions":
            report.summary.get(
                "Positions"
            ),

        "GrossExposure":
            report.summary.get(
                "GrossExposure"
            ),

        "NetExposure":
            report.summary.get(
                "NetExposure"
            ),

        "HistoricalVol":
            report.volatility.get(
                "HistoricalVol"
            ),

        "EWMAVolForecast":
            report.volatility.get(
                "EWMAForecast"
            ),

        "HistoricalVaR":
            report.var_report.get(
                "HistoricalVaR"
            ),

        "HistoricalCVaR":
            report.var_report.get(
                "HistoricalCVaR"
            ),

        "RiskOfRuin":
            report.monte_carlo.get(
                "RiskOfRuin"
            )
            if report.monte_carlo
            else None,
    }


# ============================================================
# INSTITUTIONAL RISK ENGINE
# ============================================================

class InstitutionalRiskEngine:
    """
    Master institutional risk engine.

    Orchestrates:
        1. Volatility forecasting
        2. Factor risk
        3. VaR / CVaR
        4. Stress testing
        5. Monte Carlo simulation
        6. Institutional reporting
    """

    # --------------------------------------------------------
    # VOLATILITY FORECASTING
    # --------------------------------------------------------

    def forecast(
        self,
        returns: pd.DataFrame,
    ) -> RiskForecast:

        return run_risk_forecast_engine(
            returns
        )

    # --------------------------------------------------------
    # STRESS TESTING
    # --------------------------------------------------------

    def stress_test(
        self,
        portfolio_returns: pd.Series,
    ) -> HistoricalStressResult:

        return historical_stress_test(
            portfolio_returns
        )

    # --------------------------------------------------------
    # VAR / CVAR
    # --------------------------------------------------------

    def var(
        self,
        portfolio_returns: pd.Series,
    ) -> dict:

        return {

            "HistoricalVaR":
                historical_var(
                    portfolio_returns
                ),

            "HistoricalCVaR":
                historical_cvar(
                    portfolio_returns
                ),

            "ParametricVaR":
                parametric_var(
                    portfolio_returns
                ),

            "ParametricCVaR":
                parametric_cvar(
                    portfolio_returns
                ),
        }

    # --------------------------------------------------------
    # MONTE CARLO
    # --------------------------------------------------------

    def monte_carlo(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
    ) -> MonteCarloResult:

        return run_monte_carlo_engine(
            returns=returns,
            weights=weights,
        )

    # --------------------------------------------------------
    # FACTOR RISK
    # --------------------------------------------------------

    def factor_risk(
        self,
        returns: pd.DataFrame,
        factors: pd.DataFrame,
    ) -> FactorModelResult:

        return build_factor_model(
            returns=returns,
            factors=factors,
        )

    # --------------------------------------------------------
    # INSTITUTIONAL REPORT
    # --------------------------------------------------------

    def report(
        self,
        portfolio_name: str,
        portfolio,
        returns: pd.DataFrame,
        factors: pd.DataFrame | None = None,
    ) -> InstitutionalRiskReport:

        return build_institutional_risk_report(
            portfolio_name=portfolio_name,
            portfolio=portfolio,
            returns=returns,
            factors=factors,
        )
    

# ============================================================
# INSTITUTIONAL RISK ENGINE
# ============================================================

class InstitutionalRiskEngine:

    def forecast(self, returns):
        return run_risk_forecast_engine(returns)

    def stress_test(self, portfolio_returns):
        return historical_stress_test(portfolio_returns)

    def var(self, portfolio_returns):

        return {
            "HistoricalVaR":
                historical_var(portfolio_returns),

            "HistoricalCVaR":
                historical_cvar(portfolio_returns),
        }

    def monte_carlo(
        self,
        returns,
        weights,
    ):
        return run_monte_carlo_engine(
            returns=returns,
            weights=weights,
        )

    def factor_risk(
        self,
        returns,
        factors,
    ):
        return build_factor_model(
            returns=returns,
            factors=factors,
        )

    def report(
        self,
        portfolio_name,
        portfolio,
        returns,
        factors=None,
    ):
        return build_institutional_risk_report(
            portfolio_name=portfolio_name,
            portfolio=portfolio,
            returns=returns,
            factors=factors,
        )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

RiskModelResult = RiskResult
