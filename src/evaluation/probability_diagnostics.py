import numpy as np
import pandas as pd

from scipy.stats import skew
from scipy.stats import kurtosis


# ==========================================================
# PROBABILITY DIAGNOSTICS
# ==========================================================

def compute_probability_diagnostics(
    probabilities,
    confidence,
    dates=None
):
    """
    Compute probability and confidence diagnostics.

    Parameters
    ----------
    probabilities : array-like
        Ensemble probabilities.

    confidence : array-like
        Ensemble confidence values.

    dates : array-like, optional
        Dates used for cross-sectional diagnostics.

    Returns
    -------
    dict
    """

    eps = 1e-12

    p = np.asarray(
        probabilities,
        dtype=float
    )

    c = np.asarray(
        confidence,
        dtype=float
    )

    # --------------------------------------------
    # Numerical stability
    # --------------------------------------------

    p = np.clip(
        p,
        eps,
        1 - eps
    )

    diagnostics = {}

    # ======================================================
    # Probability Distribution
    # ======================================================

    diagnostics["Avg_Probability"] = float(np.mean(p))
    diagnostics["Std_Proba"] = float(np.std(p))
    diagnostics["Median_Proba"] = float(np.median(p))

    diagnostics["Min_Proba"] = float(np.min(p))
    diagnostics["Max_Proba"] = float(np.max(p))

    diagnostics["P05_Proba"] = float(np.percentile(p, 5))
    diagnostics["P10_Proba"] = float(np.percentile(p, 10))
    diagnostics["P25_Proba"] = float(np.percentile(p, 25))
    diagnostics["P75_Proba"] = float(np.percentile(p, 75))
    diagnostics["P90_Proba"] = float(np.percentile(p, 90))
    diagnostics["P95_Proba"] = float(np.percentile(p, 95))

    diagnostics["Probability_Range"] = (
        diagnostics["Max_Proba"]
        -
        diagnostics["Min_Proba"]
    )

    diagnostics["Probability_Spread"] = (
        diagnostics["P90_Proba"]
        -
        diagnostics["P10_Proba"]
    )

    diagnostics["Probability_IQR"] = (
        diagnostics["P75_Proba"]
        -
        diagnostics["P25_Proba"]
    )

    diagnostics["Probability_Spread_95_05"] = (
        diagnostics["P95_Proba"]
        -
        diagnostics["P05_Proba"]
    )

    # ======================================================
    # Confidence Buckets
    # ======================================================

    diagnostics["High_Confidence_Pct"] = float(
        np.mean(p > 0.75)
    )

    diagnostics["Low_Confidence_Pct"] = float(
        np.mean(p < 0.25)
    )

    diagnostics["Neutral_Probability_Pct"] = float(
        np.mean(
            (p >= 0.45)
            &
            (p <= 0.55)
        )
    )

    # ======================================================
    # Probability Entropy
    # ======================================================

    diagnostics["Probability_Entropy"] = float(
        -np.mean(
            p * np.log(p)
            +
            (1 - p) * np.log(1 - p)
        )
    )

    # ======================================================
    # Distribution Shape
    # ======================================================

    diagnostics["Probability_Skewness"] = float(
        skew(
            p,
            bias=False
        )
    )

    diagnostics["Probability_Kurtosis"] = float(
        kurtosis(
            p,
            fisher=True,
            bias=False
        )
    )

    # ======================================================
    # Confidence Diagnostics
    # ======================================================

    diagnostics["Confidence_Mean"] = float(
        np.mean(c)
    )

    diagnostics["Ensemble_Confidence_Std"] = float(
        np.std(c)
    )

    diagnostics["Confidence_CV"] = float(
        np.std(c)
        /
        (np.mean(c) + eps)
    )

    # ======================================================
    # Cross-sectional Dispersion
    # ======================================================

    if dates is not None:

        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(dates),
                "Probability": p
            }
        )

        diagnostics["CrossSectional_Dispersion"] = float(
            df.groupby("Date")["Probability"]
              .std()
              .mean()
        )

    else:

        diagnostics["CrossSectional_Dispersion"] = np.nan

    return diagnostics











import numpy as np
import pandas as pd

from scipy.stats import entropy
from scipy.stats import skew
from scipy.stats import kurtosis


def compute_probability_diagnostics(
    probabilities,
    confidence,
    dates=None
):
    """
    Institutional probability diagnostics.

    Parameters
    ----------
    probabilities : array-like
    confidence : array-like
    dates : pd.Series (optional)

    Returns
    -------
    dict
    """

    p = np.asarray(probabilities, dtype=float)
    c = np.asarray(confidence, dtype=float)

    eps = 1e-12

    p = np.clip(
        probabilities,
        1e-12,
        1 - eps
    )

    diagnostics = {}

    # --------------------------------------------------
    # Probability distribution
    # --------------------------------------------------

    diagnostics["Avg_Probability"] = float(np.mean(p))
    diagnostics["Std_Probability"] = float(np.std(p))
    diagnostics["Median_Probability"] = float(np.median(p))

    diagnostics["Min_Probability"] = float(np.min(p))
    diagnostics["Max_Probability"] = float(np.max(p))

    diagnostics["Probability_IQR"] = float(
        np.percentile(p, 75)
        -
        np.percentile(p, 25)
    )

    diagnostics["Probability_Spread_90_10"] = float(
        np.percentile(p, 90)
        -
        np.percentile(p, 10)
    )

    diagnostics["Probability_Spread_95_05"] = float(
        np.percentile(p, 95)
        -
        np.percentile(p, 5)
    )

    # --------------------------------------------------
    # Entropy
    # --------------------------------------------------

    diagnostics["Probability_Entropy"] = float(
        np.mean(
            -(p*np.log(p) + (1-p)*np.log(1-p))
        )
    )

    # --------------------------------------------------
    # Shape
    # --------------------------------------------------

    diagnostics["Probability_Skewness"] = float(
        skew(p, bias=False)
    )

    diagnostics["Probability_Kurtosis"] = float(
        kurtosis(
            p,
            fisher=True,
            bias=False
        )
    )

    # --------------------------------------------------
    # Confidence
    # --------------------------------------------------

    diagnostics["Confidence_Mean"] = float(np.mean(c))
    diagnostics["Confidence_Std"] = float(np.std(c))

    diagnostics["Confidence_CV"] = float(
        np.std(c)
        /
        (np.mean(c) + eps)
    )

    # --------------------------------------------------
    # Cross-sectional dispersion
    # --------------------------------------------------

    if dates is not None:

        df = pd.DataFrame(
            {
                "Date": dates.values,
                "Probability": p
            }
        )

        diagnostics["CrossSectional_Dispersion"] = float(
            df.groupby("Date")["Probability"]
              .std()
              .mean()
        )

    else:

        diagnostics["CrossSectional_Dispersion"] = np.nan

    return diagnostics