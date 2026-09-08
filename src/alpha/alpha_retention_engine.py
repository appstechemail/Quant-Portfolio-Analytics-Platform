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
    signals: float

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

    signal_measurement: str = "UNAVAILABLE"


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

    def _stage_signal_definition(
        self,
        stage_name: str,
    ) -> str:
        """
        Return the intended signal representation for each
        alpha-pipeline stage.
        """

        definitions = {
            "RAW": "Prediction_Alpha",
            "META": "Prediction_Alpha",
            "REGIME": "Prediction_Alpha",
            "VOLATILITY": "Prediction_Alpha",
            "CROSS_SECTION": "Prediction_Alpha",
            "PORTFOLIO": "Position",
        }

        return definitions.get(
            str(stage_name).upper(),
            "AUTO",
        )

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

        # ========================================================
        # CANONICAL STAGE SIGNAL COUNT
        # ========================================================
        #
        # A stage signal must represent evidence that alpha survived
        # the stage. We therefore use an explicit priority:
        #
        # 1. Position
        #    Final portfolio/tradable stage.
        #
        # 2. Explicit Selected flag
        #    Candidate-selection stage.
        #
        # 3. Prediction probability
        #    Model-signal stage.
        #
        # 4. Prediction alpha
        #    Alpha-edge stage.
        #
        # 5. Signal label
        #    Legacy compatibility only.
        #
        # 6. Otherwise:
        #    signal count is undefined, not zero.
        #
        # IMPORTANT:
        # "No signal column" does NOT mean "zero alpha".
        # ========================================================

        signals = np.nan

        signal_measurement = "NONE"

        # --------------------------------------------------------
        # 1. Tradable position
        # --------------------------------------------------------

        if "Position" in df.columns:

            position_values = pd.to_numeric(
                df["Position"],
                errors="coerce",
            ).fillna(0.0)

            signals = int(
                (position_values.abs() > 1e-12).sum()
            )

            signal_measurement = "Position"

        # --------------------------------------------------------
        # 2. Explicit candidate selection
        # --------------------------------------------------------

        elif "Selected" in df.columns:

            selected = pd.to_numeric(
                df["Selected"],
                errors="coerce",
            ).fillna(0)

            signals = int(
                (selected == 1).sum()
            )

            signal_measurement = "Selected"

        # --------------------------------------------------------
        # 3. Prediction probability
        # --------------------------------------------------------

        elif "Prediction_Prob" in df.columns:

            probability = pd.to_numeric(
                df["Prediction_Prob"],
                errors="coerce",
            )

            signals = int(
                (probability > 0.50).sum()
            )

            signal_measurement = "Prediction_Prob"

        # --------------------------------------------------------
        # 4. Probability compatibility column
        # --------------------------------------------------------

        elif "Probability" in df.columns:

            probability = pd.to_numeric(
                df["Probability"],
                errors="coerce",
            )

            signals = int(
                (probability > 0.50).sum()
            )

            signal_measurement = "Probability"

        # --------------------------------------------------------
        # 5. Prediction alpha
        # --------------------------------------------------------

        elif "Prediction_Alpha" in df.columns:

            alpha = pd.to_numeric(
                df["Prediction_Alpha"],
                errors="coerce",
            )

            signals = int(
                (alpha > 0.0).sum()
            )

            signal_measurement = "Prediction_Alpha"

        # --------------------------------------------------------
        # 6. Legacy human-readable signal
        # --------------------------------------------------------

        elif "Signal" in df.columns:

            signal_values = (
                df["Signal"]
                .astype(str)
                .str.upper()
            )

            signals = int(
                signal_values.isin(
                    [
                        "BUY",
                        "STRONG BUY",
                    ]
                ).sum()
            )

            signal_measurement = "Signal"

        # --------------------------------------------------------
        # 7. No valid signal representation
        # --------------------------------------------------------

        else:

            signals = 0

            signal_measurement = "UNAVAILABLE"

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
            signal_measurement=signal_measurement,
        )

        self.stage_reports.append(metrics)

        return metrics

    def compare_stages(
        self,
    ) -> pd.DataFrame:
        """
        Compare alpha pipeline stages.

        Signal retention:
            - Uses the first stage with a valid positive signal count
            as the baseline.
            - Returns NaN when a stage's signal count is unavailable.
            - Never treats an unavailable signal count as zero.

        Sharpe contribution:
            - Normalized against the maximum valid stage Sharpe.
            - Returns NaN when the stage Sharpe is unavailable.
            - Returns 0 when the maximum Sharpe is zero.

        This prevents missing stage-level signal information from being
        incorrectly interpreted as zero alpha/signal retention.
        """

        if not self.stage_reports:
            return pd.DataFrame()

        # ============================================================
        # FIND BASELINE SIGNAL COUNT
        # ============================================================
        #
        # Use the first stage having a valid positive signal count.
        #
        # Do NOT use:
        #     max(..., 1)
        #
        # because an unavailable signal count must not be interpreted
        # as zero or one.
        # ============================================================

        base_signals = np.nan

        for metrics in self.stage_reports:
            if (
                pd.notna(metrics.signals)
                and metrics.signals > 0
            ):
                base_signals = float(
                    metrics.signals
                )
                break

        # ============================================================
        # FIND MAXIMUM VALID SHARPE
        # ============================================================

        valid_sharpes = [
            float(metrics.sharpe)
            for metrics in self.stage_reports
            if pd.notna(metrics.sharpe)
        ]

        max_sharpe = (
            max(valid_sharpes)
            if valid_sharpes
            else np.nan
        )

        # ============================================================
        # BUILD COMPARISON TABLE
        # ============================================================

        rows = []

        for metrics in self.stage_reports:

            row = asdict(metrics)

            # --------------------------------------------------------
            # SIGNAL RETENTION
            # --------------------------------------------------------
            #
            # If either the baseline or current-stage signal count is
            # unavailable, retention remains NaN.
            #
            # This is intentionally different from zero:
            #
            #   NaN = measurement unavailable
            #   0   = measurement available and zero
            # --------------------------------------------------------

            if (
                pd.isna(base_signals)
                or pd.isna(metrics.signals)
            ):
                row["signal_retention"] = np.nan

            else:
                row["signal_retention"] = (
                    float(metrics.signals)
                    / base_signals
                )

            # --------------------------------------------------------
            # SHARPE CONTRIBUTION
            # --------------------------------------------------------

            if pd.isna(metrics.sharpe):
                row["sharpe_contribution"] = np.nan

            elif pd.isna(max_sharpe):
                row["sharpe_contribution"] = np.nan

            elif max_sharpe == 0:
                row["sharpe_contribution"] = 0.0

            else:
                row["sharpe_contribution"] = (
                    float(metrics.sharpe)
                    / max_sharpe
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

        portfolio_rows = report[
            report["stage"].astype(str).str.upper()
            == "PORTFOLIO"
        ]

        final_sharpe = (
            float(
                portfolio_rows.iloc[-1]["sharpe"]
            )
            if not portfolio_rows.empty
            and pd.notna(
                portfolio_rows.iloc[-1]["sharpe"]
            )
            else np.nan
        )

        measured = report[
            report["sharpe"].notna()
        ]

        if measured.empty:
            best_stage = None
            worst_stage = None
            max_pipeline_sharpe = np.nan
        else:
            best_idx = measured["sharpe"].idxmax()
            worst_idx = measured["sharpe"].idxmin()

            best_stage = measured.loc[
                best_idx,
                "stage"
            ]

            worst_stage = measured.loc[
                worst_idx,
                "stage"
            ]

            max_pipeline_sharpe = float(
                measured["sharpe"].max()
            )

        return {

            "best_stage":
                best_stage,

            "worst_stage":
                worst_stage,

            "max_pipeline_sharpe":
                max_pipeline_sharpe,

            "final_portfolio_sharpe":
                final_sharpe,

            "alpha_decay":
                (
                    final_sharpe
                    - max_pipeline_sharpe
                )
                if (
                    pd.notna(final_sharpe)
                    and pd.notna(max_pipeline_sharpe)
                )
                else np.nan,

            "pipeline_efficiency":
                (
                    final_sharpe
                    / max_pipeline_sharpe
                )
                if (
                    pd.notna(final_sharpe)
                    and pd.notna(max_pipeline_sharpe)
                    and abs(max_pipeline_sharpe) > 1e-12
                )
                else np.nan,
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
