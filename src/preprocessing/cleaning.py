import numpy as np


# =========================
# 1. CLEAN DATA (ROBUST)
# =========================
def clean_data(df):

    df = df.copy()

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df = df.sort_values(
        ["Company", "Date"]
    )

    company_col = df["Company"]

    df = (
        df.groupby("Company")
          .ffill()
    )

    df["Company"] = company_col.values

    df = df.dropna()

    return df

