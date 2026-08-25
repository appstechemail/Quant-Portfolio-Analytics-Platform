"""
==============================================================================
INSTITUTIONAL QUANT PLATFORM
==============================================================================

Module:
    alpha_retention_engine.py

Purpose
-------
Tracks alpha persistence across the complete alpha pipeline.

Institutional Metrics
---------------------

1. Signal Retention
2. Sharpe Contribution
3. Alpha Decay
4. Pipeline Efficiency
5. CAGR Contribution
6. Drawdown Contribution
7. Turnover Contribution
8. Win Rate Contribution

Pipeline
--------

RAW
 ↓
META
 ↓
REGIME
 ↓
VOLATILITY
 ↓
CROSS_SECTION
 ↓
PORTFOLIO

Outputs
-------

data/alpha/outputs/

    alpha_retention.csv
    pipeline_summary.json

==============================================================================

Author:
    Institutional Quant Platform

==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUTPUT_DIR = "data/alpha/outputs"


class AlphaStage(Enum):

    RAW = "RAW"
    META = "META"
    REGIME = "REGIME"
    VOLATILITY = "VOLATILITY"
    CROSS_SECTION = "CROSS_SECTION"
    PORTFOLIO = "PORTFOLIO"


@dataclass
class StageMetrics:

    stage: str

    rows: int
    signals: int

    unique_dates: int
    companies: int

    avg_position: float

    sharpe: float
    cagr: float
    max_dd: float

    volatility: float
    win_rate: float
    turnover: float

    mean_return: float
    median_return: float

    signal_retention: float = 1.0
    sharpe_contribution: float = 0.0


class AlphaRetentionEngine:

    """
    Institutional Alpha Retention Engine.
    """

    def __init__(self, annualization: int = 252):

        self.annualization = annualization

        self.stage_reports: list[StageMetrics] = []

    @staticmethod
    def _safe_series(
        df: pd.DataFrame,
        col: str,
        default=0.0
    ) -> pd.Series:

        if col in df.columns:
            return df[col].fillna(default)

        return pd.Series(default, index=df.index)

    def _compute_sharpe(
        self,
        returns: pd.Series
    ) -> float:

        if len(returns) < 2:
            return 0.0

        if returns.std() == 0:
            return 0.0

        return float(
            np.sqrt(self.annualization)
            * returns.mean()
            / returns.std()
        )

    def _compute_cagr(
        self,
        returns: pd.Series
    ) -> float:

        if len(returns) == 0:
            return 0.0

        cumulative = (1 + returns).cumprod()

        years = max(
            len(cumulative) / self.annualization,
            1 / self.annualization
        )

        return float(
            cumulative.iloc[-1] ** (1 / years) - 1
        )

    def _compute_max_dd(
        self,
        returns: pd.Series
    ) -> float:

        if len(returns) == 0:
            return 0.0

        equity = (1 + returns).cumprod()

        running_max = equity.cummax()

        drawdown = equity / running_max - 1

        return float(drawdown.min())

    def evaluate_stage(
        self,
        stage: AlphaStage | str,
        df: pd.DataFrame
    ) -> StageMetrics:

        stage_name = (
            stage.value
            if isinstance(stage, AlphaStage)
            else str(stage)
        )

        returns = self._safe_series(
            df,
            "Strategy_Return"
        )

        position = self._safe_series(
            df,
            "Position"
        )

        turnover = self._safe_series(
            df,
            "Position_Change"
        )

        if "Signal" in df.columns:

            signals = int(
                (
                    df["Signal"]
                    .astype(str)
                    .str.upper()
                    .isin(
                        [
                            "BUY",
                            "STRONG BUY"
                        ]
                    )
                ).sum()
            )

        else:

            signals = int(
                (position > 0).sum()
            )

        win_rate = (
            float((returns > 0).mean())
            if len(returns)
            else 0.0
        )

        metrics = StageMetrics(
            stage=stage_name,
            rows=len(df),
            signals=signals,
            unique_dates=(
                df["Date"].nunique()
                if "Date" in df.columns
                else 0
            ),

            companies=(
                df["Company"].nunique()
                if "Company" in df.columns
                else 0
            ),

            avg_position=float(position.mean()),
            sharpe=self._compute_sharpe(returns),
            cagr=self._compute_cagr(returns),
            max_dd=self._compute_max_dd(returns),
            volatility=float(returns.std() * np.sqrt(self.annualization)),
            win_rate=win_rate,
            turnover=float(turnover.mean()),
            mean_return=float(returns.mean()),
            median_return=float(returns.median()),
        )

        self.stage_reports.append(metrics)

        return metrics

    def compare_stages(
        self
    ) -> pd.DataFrame:

        if not self.stage_reports:
            return pd.DataFrame()

        base_signals = max(
            self.stage_reports[0].signals,
            1
        )

        max_sharpe = max(
            (
                x.sharpe
                for x in self.stage_reports
            ),
            default=1.0
        )

        rows = []

        for m in self.stage_reports:

            row = asdict(m)
            row["signal_retention"] = (
                m.signals / base_signals
            )
            row["sharpe_contribution"] = (
                0
                if max_sharpe == 0
                else m.sharpe / max_sharpe
            )
            rows.append(row)

        return pd.DataFrame(rows)

    def pipeline_summary(
        self
    ) -> dict:

        report = self.compare_stages()

        if report.empty:
            return {}

        best = report["sharpe"].idxmax()
        worst = report["sharpe"].idxmin()
        max_sharpe = report["sharpe"].max()
        final_sharpe = report.iloc[-1]["sharpe"]

        return {

            "best_stage":
                report.loc[best, "stage"],

            "worst_stage":
                report.loc[worst, "stage"],

            "max_pipeline_sharpe":
                float(max_sharpe),

            "final_sharpe":
                float(final_sharpe),

            "alpha_decay":
                float(
                    final_sharpe
                    - max_sharpe
                ),

            "pipeline_efficiency":
                float(
                    final_sharpe
                    / max_sharpe
                )
                if max_sharpe != 0
                else 0,
        }

    def export(
        self,
        output_dir: str = OUTPUT_DIR
    ) -> None:

        output = Path(output_dir)

        output.mkdir(
            parents=True,
            exist_ok=True
        )

        report = self.compare_stages()

        report.to_csv(
            output / "alpha_retention.csv",
            index=False
        )

        summary = self.pipeline_summary()

        with open(
            output / "pipeline_summary.json",
            "w"
        ) as f:

            json.dump(
                summary,
                f,
                indent=4
            )

        print(
            f"\n✓ Alpha Retention exported -> {output}"
        )

    def generate_report(
        self
    ) -> pd.DataFrame:

        report = self.compare_stages()

        print("\n" + "=" * 80)
        print("ALPHA RETENTION REPORT")
        print("=" * 80)

        print(
            report[
                [
                    "stage",
                    "signals",
                    "signal_retention",
                    "sharpe",
                    "cagr",
                    "max_dd",
                    "turnover",
                ]
            ]
        )

        print("\nPIPELINE SUMMARY")
        print(
            self.pipeline_summary()
        )

        return report

# Example:
#
# engine = AlphaRetentionEngine()
# engine.evaluate_stage(AlphaStage.RAW, raw_df)
# engine.evaluate_stage(AlphaStage.META, meta_df)
# engine.evaluate_stage(AlphaStage.REGIME, regime_df)
# engine.evaluate_stage(AlphaStage.VOLATILITY, vol_df)
# engine.evaluate_stage(AlphaStage.PORTFOLIO, portfolio_df)
# engine.generate_report()
# engine.export()
