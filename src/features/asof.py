"""Point-in-time ("as of") aggregation.

In plain words: The key anti-leak helper. For every review it adds up what the same user (or
product) did EARLIER, never including that review itself or anything later. "As-of" means "using
only what was known at that moment".

Every feature in this package is computed from rows that are **strictly
earlier** than the row it describes. That one rule is what keeps the label
and the future out of the inputs: a row's own rating, its own review text and
anything written on the same timestamp never contribute.
"""

import pandas as pd


def asof_prior_sums(keys: pd.Series, times: pd.Series, values: pd.DataFrame) -> pd.DataFrame:
    """For each row: column sums of ``values`` over rows with the same key and a strictly earlier time.

    Rows sharing a (key, time) pair see the same prior, so ties never leak
    into each other. Output is aligned to ``values.index``; a row with no
    earlier rows gets zeros.
    """
    k = keys.to_numpy()
    t = times.to_numpy()
    frame = values.astype("float64").copy()
    frame["_k"] = k
    frame["_t"] = t
    per_group = frame.groupby(["_k", "_t"], sort=True).sum()
    inclusive = per_group.groupby(level=0).cumsum()
    prior = inclusive - per_group  # drops the group's own (and later-tied) contribution
    out = prior.reindex(pd.MultiIndex.from_arrays([k, t]))
    out.index = values.index
    return out
