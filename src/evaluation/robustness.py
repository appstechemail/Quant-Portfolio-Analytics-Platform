import numpy as np
import pandas as pd


def evaluate_cost_sensitivity(
    backtest_df,
    cost_rates,
):
    """
    Recalculate strategy performance under
    alternative transaction-cost assumptions.

    backtest_df must contain:
        Gross_Return
        Turnover
    """

    if backtest_df is None or backtest_df.empty:
        raise ValueError(
            "Cost sensitivity requires non-empty backtest data."
        )

    required = [
        "Gross_Return",
        "Turnover",
    ]

    missing = [
        col
        for col in required
        if col not in backtest_df.columns
    ]

    if missing:
        raise ValueError(
            "Cost sensitivity missing columns: "
            f"{missing}"
        )

    rows = []

    gross = (
        pd.to_numeric(
            backtest_df["Gross_Return"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    turnover = (
        pd.to_numeric(
            backtest_df["Turnover"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    for rate in cost_rates:

        strategy_returns = (
            gross
            -
            turnover * float(rate)
        )

        std = strategy_returns.std()

        if std <= 1e-12 or pd.isna(std):
            sharpe = 0.0
        else:
            sharpe = (
                strategy_returns.mean()
                / std
            ) * np.sqrt(252)

        cumulative = (
            1.0
            + strategy_returns
        ).cumprod()

        final_return = float(
            cumulative.iloc[-1]
        )

        years = (
            len(strategy_returns)
            / 252.0
        )

        if final_return > 0 and years > 0:
            cagr = (
                final_return
                ** (1.0 / years)
            ) - 1.0
        else:
            cagr = -1.0

        running_max = cumulative.cummax()

        drawdown = (
            cumulative
            /
            running_max
            - 1.0
        )

        rows.append({
            "Cost_Rate": float(rate),
            "Sharpe": float(sharpe),
            "CAGR": float(cagr),
            "Max_Drawdown": float(
                drawdown.min()
            ),
            "Final_Return": final_return,
        })

    return pd.DataFrame(rows)


def evaluate_fold_consistency(
    walkforward_summary,
):
    """
    Evaluate whether performance is persistent
    across walk-forward folds.
    """

    if (
        walkforward_summary is None
        or walkforward_summary.empty
    ):
        raise ValueError(
            "No walk-forward summary available."
        )

    sharpe = (
        walkforward_summary["Sharpe"]
        .astype(float)
    )

    cagr = (
        walkforward_summary["CAGR"]
        .astype(float)
    )

    rank_ic = (
        walkforward_summary["Rank_IC"]
        .astype(float)
    )

    return {
        "fold_count": int(len(sharpe)),
        "mean_sharpe": float(
            sharpe.mean()
        ),
        "median_sharpe": float(
            sharpe.median()
        ),
        "std_sharpe": float(
            sharpe.std()
        ),
        "min_sharpe": float(
            sharpe.min()
        ),
        "q25_sharpe": float(
            sharpe.quantile(0.25)
        ),
        "q75_sharpe": float(
            sharpe.quantile(0.75)
        ),
        "positive_sharpe_pct": float(
            (sharpe > 0).mean()
        ),
        "positive_cagr_pct": float(
            (cagr > 0).mean()
        ),
        "positive_ic_pct": float(
            (rank_ic > 0).mean()
        ),
    }

def evaluate_worst_quartile(
    walkforward_summary,
):
    """
    Evaluate performance in the weakest
    25% of walk-forward folds.
    """

    n = max(
        1,
        int(
            np.ceil(
                len(walkforward_summary)
                * 0.25
            )
        ),
    )

    worst = (
        walkforward_summary
        .nsmallest(
            n,
            "Sharpe",
        )
    )

    return {
        "worst_quartile_folds": n,
        "worst_quartile_mean_sharpe": float(
            worst["Sharpe"].mean()
        ),
        "worst_quartile_median_sharpe": float(
            worst["Sharpe"].median()
        ),
        "worst_quartile_mean_cagr": float(
            worst["CAGR"].mean()
        ),
        "worst_quartile_mean_drawdown": float(
            worst["Max_Drawdown"].mean()
        ),
    }


def evaluate_regime_robustness(
    walkforward_summary,
):
    """
    Evaluate return behaviour across market regimes.
    """

    regimes = {
        "BEAR": "AvgRet_BEAR",
        "BEAR_VOLATILE": "AvgRet_BEAR_VOLATILE",
        "SIDEWAYS": "AvgRet_SIDEWAYS",
        "SIDEWAYS_VOLATILE": "AvgRet_SIDEWAYS_VOLATILE",
        "BULL": "AvgRet_BULL",
        "BULL_VOLATILE": "AvgRet_BULL_VOLATILE",
    }

    result = {}

    for regime, column in regimes.items():

        if column not in walkforward_summary.columns:
            result[regime] = {
                "mean_return": np.nan,
                "positive_pct": np.nan,
            }
            continue

        values = pd.to_numeric(
            walkforward_summary[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            result[regime] = {
                "mean_return": np.nan,
                "positive_pct": np.nan,
            }
            continue

        result[regime] = {
            "mean_return": float(values.mean()),
            "median_return": float(values.median()),
            "positive_pct": float(
                (values > 0).mean()
            ),
            "observations": int(len(values)),
        }

    return result


def bootstrap_sharpe(
    returns,
    n_bootstrap=2000,
    block_size=20,
    random_state=42,
):
    """
    Moving-block bootstrap for Sharpe ratio.
    """

    returns = pd.Series(
        returns,
        dtype=float,
    ).dropna()

    if len(returns) < block_size * 3:
        raise ValueError(
            "Insufficient observations for block bootstrap."
        )

    rng = np.random.default_rng(
        random_state
    )

    values = returns.to_numpy()

    n = len(values)

    sharpes = []

    for _ in range(
        n_bootstrap
    ):

        sample = []

        while len(sample) < n:

            start = rng.integers(
                0,
                n - block_size + 1,
            )

            block = values[
                start:
                start + block_size
            ]

            sample.extend(block)

        sample = np.asarray(
            sample[:n],
            dtype=float,
        )

        std = sample.std()

        if std <= 1e-12:
            continue

        sharpe = (
            sample.mean()
            / std
        ) * np.sqrt(252)

        sharpes.append(sharpe)

    sharpes = np.asarray(
        sharpes,
        dtype=float,
    )

    if len(sharpes) == 0:
        raise ValueError(
            "Bootstrap produced no valid Sharpe estimates."
        )

    return {
        "observed_sharpe": float(
            (
                values.mean()
                /
                values.std()
            )
            * np.sqrt(252)
        ),
        "bootstrap_median_sharpe": float(
            np.median(sharpes)
        ),
        "bootstrap_p05_sharpe": float(
            np.percentile(sharpes, 5)
        ),
        "bootstrap_p95_sharpe": float(
            np.percentile(sharpes, 95)
        ),
        "prob_sharpe_positive": float(
            (sharpes > 0).mean()
        ),
        "bootstrap_samples": int(
            len(sharpes)
        ),
    }

def build_quantitative_validation_scorecard(
    walkforward_summary,
    cost_results,
    bootstrap_results,
    fold_results,
    regime_results,
):
    """
    Produce an objective robustness scorecard.

    This is a research gate, not a production
    software health check.
    """

    checks = {}

    # --------------------------------------------------
    # Walk-forward
    # --------------------------------------------------

    checks["minimum_folds"] = (
        len(walkforward_summary) >= 10
    )

    checks["positive_sharpe_folds"] = (
        fold_results["positive_sharpe_pct"]
        >= 0.60
    )

    checks["median_sharpe_positive"] = (
        fold_results["median_sharpe"]
        > 0.0
    )

    checks["worst_quartile_positive"] = (
        fold_results[
            "worst_quartile_mean_sharpe"
        ]
        > 0.0
        if
        "worst_quartile_mean_sharpe"
        in fold_results
        else False
    )

    # --------------------------------------------------
    # Alpha
    # --------------------------------------------------

    checks["positive_ic_folds"] = (
        fold_results["positive_ic_pct"]
        >= 0.60
    )

    checks["mean_rank_ic_positive"] = (
        walkforward_summary["Rank_IC"].mean()
        > 0.0
    )

    # --------------------------------------------------
    # Bootstrap
    # --------------------------------------------------

    checks["bootstrap_sharpe_positive"] = (
        bootstrap_results[
            "bootstrap_p05_sharpe"
        ]
        > 0.0
    )

    checks["probability_sharpe_positive"] = (
        bootstrap_results[
            "prob_sharpe_positive"
        ]
        >= 0.95
    )

    # --------------------------------------------------
    # Cost robustness
    # --------------------------------------------------

    realistic_cost = cost_results[
        cost_results["Cost_Rate"] == 0.003
    ]

    if realistic_cost.empty:
        checks["cost_robust"] = False
    else:
        checks["cost_robust"] = bool(
            realistic_cost.iloc[0]["Sharpe"]
            > 0
        )

    # --------------------------------------------------
    # Regime robustness
    # --------------------------------------------------

    negative_regimes = 0
    available_regimes = 0

    for regime, values in regime_results.items():

        mean_return = values.get(
            "mean_return"
        )

        if pd.isna(mean_return):
            continue

        available_regimes += 1

        if mean_return < 0:
            negative_regimes += 1

    checks["not_regime_dependent"] = (
        negative_regimes
        <=
        max(
            2,
            available_regimes // 2
        )
    )

    # --------------------------------------------------
    # Overall
    # --------------------------------------------------

    passed = sum(
        bool(v)
        for v in checks.values()
    )

    total = len(checks)

    score = (
        passed / total
        if total > 0
        else 0.0
    )

    return {
        "checks": checks,
        "passed": passed,
        "total": total,
        "score": score,
        "robust": (
            score >= 0.80
            and all(
                checks.get(name, False)
                for name in [
                    "minimum_folds",
                    "median_sharpe_positive",
                    "mean_rank_ic_positive",
                ]
            )
        ),
    }


def evaluate_activity_quality(
    walkforward_summary,
    minimum_active_days_pct=0.50,
    minimum_holdings=0.25,
):
    """
    Evaluate whether walk-forward folds have
    sufficient trading activity to make performance
    statistics economically meaningful.

    This does NOT remove sparse folds.

    It classifies them separately so that:
        1. all folds remain visible
        2. activity-qualified folds can be analysed
        3. sparse-fold statistics are not mistaken
           for robust strategy behaviour
    """

    if (
        walkforward_summary is None
        or walkforward_summary.empty
    ):
        raise ValueError(
            "No walk-forward summary available."
        )

    required = [
        "Fold",
        "Test_Days",
        "Active_Days",
        "Avg_Holdings",
        "Sharpe",
        "CAGR",
        "Max_Drawdown",
    ]

    missing = [
        col
        for col in required
        if col not in walkforward_summary.columns
    ]

    if missing:
        raise ValueError(
            "Activity quality missing columns: "
            f"{missing}"
        )

    df = walkforward_summary.copy()

    df["Activity_Pct"] = (
        pd.to_numeric(
            df["Active_Days"],
            errors="coerce",
        )
        /
        pd.to_numeric(
            df["Test_Days"],
            errors="coerce",
        )
    )

    df["Activity_Qualified"] = (
        df["Activity_Pct"]
        >= float(minimum_active_days_pct)
    ) & (
        pd.to_numeric(
            df["Avg_Holdings"],
            errors="coerce",
        )
        >= float(minimum_holdings)
    )

    qualified = df.loc[
        df["Activity_Qualified"]
    ].copy()

    sparse = df.loc[
        ~df["Activity_Qualified"]
    ].copy()

    result = {
        "total_folds": int(len(df)),
        "qualified_folds": int(len(qualified)),
        "sparse_folds": int(len(sparse)),
        "qualified_pct": float(
            len(qualified) / len(df)
        ),
        "sparse_pct": float(
            len(sparse) / len(df)
        ),
        "qualified_mean_sharpe": (
            float(qualified["Sharpe"].mean())
            if not qualified.empty
            else np.nan
        ),
        "qualified_median_sharpe": (
            float(qualified["Sharpe"].median())
            if not qualified.empty
            else np.nan
        ),
        "qualified_positive_sharpe_pct": (
            float(
                (qualified["Sharpe"] > 0).mean()
            )
            if not qualified.empty
            else np.nan
        ),
        "qualified_mean_cagr": (
            float(qualified["CAGR"].mean())
            if not qualified.empty
            else np.nan
        ),
        "qualified_mean_drawdown": (
            float(
                qualified["Max_Drawdown"].mean()
            )
            if not qualified.empty
            else np.nan
        ),
        "sparse_mean_sharpe": (
            float(sparse["Sharpe"].mean())
            if not sparse.empty
            else np.nan
        ),
        "sparse_median_sharpe": (
            float(sparse["Sharpe"].median())
            if not sparse.empty
            else np.nan
        ),
        "activity_threshold_pct": float(
            minimum_active_days_pct
        ),
        "holdings_threshold": float(
            minimum_holdings
        ),
    }

    return result, df