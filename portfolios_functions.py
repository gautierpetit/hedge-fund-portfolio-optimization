import pandas as pd
import numpy as np










### BELOW are better version of AUM functions to improve in mains

def test(weight):
    """
    Computes portfolio value based on Returns Dataframe
    """

    aum = pd.Series(
        data=[100] * (len(weight) + 1),
        index=Returns.iloc[rw - 1 :].index,
        name="AUM",
        dtype=float,
    )

    for i, w in enumerate(weight):

        returns_row = Returns.iloc[
            rw + i, Returns.columns.get_indexer(rw_returns[i].columns)
        ]

        aum.iloc[i + 1] = (1 + np.matmul(w.T, returns_row)) * aum.iloc[i]

    return aum

def test2(weight):
    """
    Computes portfolio value based on Returns Dataframe
    """

    aum = pd.Series(
        data=[100] * (len(weight) + 1),
        index=Returns.iloc[rw - 1 :].index,
        name="AUM",
        dtype=float,
    )

    for i, w in enumerate(weight):

        returns_row = Returns.iloc[
            rw + i, Returns.columns.get_indexer(rw_returns[i].columns)
        ]

        aum.iloc[i + 1] = (1 + w.T @ returns_row) * aum.iloc[i]

    return aum

def test_fast(weight):
    """
    Super-optimized portfolio value calculation
    """

    n_steps = len(weight)
    aum = np.empty(n_steps + 1, dtype=float)
    aum[0] = 100.0

    # Precompute column indexers to avoid recalculating every loop
    columns_idx = [Returns.columns.get_indexer(rw_returns[i].columns) for i in range(n_steps)]

    Returns_values = Returns.values  # Access raw NumPy array once

    for i, w in enumerate(weight):
        returns_row = Returns_values[rw + i, columns_idx[i]]
        aum[i + 1] = (1 + (w.T @ returns_row).iloc[0]) * aum[i]

    # Build pd.Series only once at the end
    aum_series = pd.Series(data=aum, index=Returns.iloc[rw - 1:].index, name="AUM")

    return aum_series

