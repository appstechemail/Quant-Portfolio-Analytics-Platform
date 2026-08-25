"""
============================================================
RANKING ENGINE
============================================================

Converts Composite Scores into rankings.

Outputs
-------

Composite_Rank (1 = best)
Composite_Percentile (0–1)
Selected (Top K)
Long_Short (future-proof for long/short strategies)

Supports

• Cross-sectional ranking
• Daily ranking
• Long-only
• Long/Short
• Future portfolio construction

============================================================
"""

from __future__ import annotations

import pandas as pd


# ============================================================
# VALIDATION
# ============================================================

def _validate_input(
    df,
    score_column,
    group_cols,
):
    """
    Validate required columns.
    """

    if score_column not in df.columns:
        raise ValueError(
            f"{score_column} not found."
        )

    if group_cols is not None:
        for column in group_cols:
            if column not in df.columns:
                raise ValueError(
                    f"{column} not found."
                )


# ============================================================
# RANKING
# ============================================================

def rank_stocks(
    df,
    score_column="Composite_Score",
    group_cols=("Date",),
    top_k=20,
    bottom_k=20,
):
    """
    Rank stocks.

    Parameters
    ----------
    df

    score_column

    group_cols

    top_k

    bottom_k

    Returns
    -------
    DataFrame
    """

    _validate_input(
        df,
        score_column,
        group_cols,
    )

    out = df.copy()

    # -------------------------------------------------------
    # Cross-sectional Rank
    # -------------------------------------------------------

    if group_cols is None:

        out["Composite_Rank"] = (
            out[score_column]
            .rank(
                ascending=False,
                method="dense"
            )
            .astype(int)
        )

        out["Composite_Percentile"] = (
            out[score_column]
            .rank(
                pct=True
            )
        )

    else:

        out["Composite_Rank"] = (
            out
            .groupby(
                list(group_cols)
            )[score_column]
            .rank(
                ascending=False,
                method="dense"
            )
            .astype(int)
        )

        out["Composite_Percentile"] = (
            out
            .groupby(
                list(group_cols)
            )[score_column]
            .rank(
                pct=True
            )
        )

    # -------------------------------------------------------
    # Long Selection
    # -------------------------------------------------------

    out["Selected"] = (
        out["Composite_Rank"]
        <= top_k
    )

    # -------------------------------------------------------
    # Long / Short Labels
    # -------------------------------------------------------

    out["Long_Short"] = "Neutral"
    out.loc[
        out["Composite_Rank"]
        <= top_k,
        "Long_Short"
    ] = "Long"

    if bottom_k > 0:
        if group_cols is None:
            total = len(out)
            cutoff = total - bottom_k + 1
            out.loc[
                out["Composite_Rank"]
                >= cutoff,
                "Long_Short"
            ] = "Short"

        else:

            max_rank = (
                out
                .groupby(
                    list(group_cols)
                )["Composite_Rank"]
                .transform("max")
            )

            cutoff = max_rank - bottom_k + 1

            out.loc[
                out["Composite_Rank"]
                >= cutoff,
                "Long_Short"
            ] = "Short"


    return out


# ============================================================
# SUMMARY
# ============================================================

def ranking_summary(
    ranked_df,
):
    """
    Simple ranking summary.
    """

    print()
    print("=" * 60)
    print("RANKING SUMMARY")
    print("=" * 60)

    print(
        ranked_df["Long_Short"]
        .value_counts()
    )

    print("=" * 60)


# ============================================================
# TOP STOCKS
# ============================================================

def top_ranked(
    ranked_df,
    n=20,
):
    """
    Return top-ranked stocks.
    """

    return (
        ranked_df
        .sort_values(
            "Composite_Rank"
        )
        .head(n)
    )


# ============================================================
# BOTTOM STOCKS
# ============================================================

def bottom_ranked(
    ranked_df,
    n=20,
):
    """
    Return bottom-ranked stocks.
    """

    return (
        ranked_df
        .sort_values(
            "Composite_Rank",
            ascending=False,
        )
        .head(n)
    )




# ======================================================
# EXAMPLE USAGE
# ======================================================

# ranked_df = rank_stocks(
#     scored_df,
#     score_column="Composite_Score",
#     group_cols=["Date"],
#     top_k=30,
#     bottom_k=30,
# )

# ranking_summary(ranked_df)

# ======================================================