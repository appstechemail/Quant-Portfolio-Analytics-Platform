from pathlib import Path

import pandas as pd

# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_DATA_DIR = Path("data")

DEFAULT_IC_WEIGHT_FILE = (
    DEFAULT_DATA_DIR / "ic_weights.csv"
)

# ============================================================
# CSV LOADER
# ============================================================

def load_csv(
    path,
    required_columns=None,
):
    """
    Safely load a CSV file.

    Parameters
    ----------
    path : str | Path

    required_columns : list[str] | None

        Columns that must exist.

    Returns
    -------
    pd.DataFrame

        Empty DataFrame if file
        cannot be loaded.
    """

    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    try:

        df = pd.read_csv(path)

    except Exception:

        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if required_columns is not None:

        missing = (
            set(required_columns)
            - set(df.columns)
        )

        if missing:

            raise ValueError(
                f"{path.name} missing columns: "
                f"{sorted(missing)}"
            )

    return df


# ============================================================
# LOAD IC WEIGHTS
# ============================================================

def load_ic_weights(
    path=DEFAULT_IC_WEIGHT_FILE,
):
    """
    Load feature IC weights.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    dict

        {
            Feature : Weight
        }
    """

    df = load_csv(
        path,
        required_columns=[
            "Feature",
            "Weight",
        ],
    )

    if df.empty:
        return {}

    return (
        df.set_index("Feature")["Weight"]
        .to_dict()
    )


# ============================================================
# LOAD MASTER IC TABLE
# ============================================================

def load_master_ic_table(
    path=DEFAULT_DATA_DIR / "master_ic_table.csv",
):
    """
    Load the master IC table.

    Returns
    -------
    pd.DataFrame
    """

    return load_csv(path)


# ============================================================
# LOAD SELECTED FEATURES
# ============================================================

def load_selected_features(
    path=DEFAULT_DATA_DIR / "selected_ic_features.csv",
):
    """
    Load selected feature table.

    Returns
    -------
    pd.DataFrame
    """

    return load_csv(path)

