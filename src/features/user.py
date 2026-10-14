"""User-side features, built as-of each interaction.

In plain words: Builds the user-side numbers: how many reviews the user wrote before, their average
rating, how often they recommend, a vector summarising the text they wrote, and their skin/eye/hair
profile.

For the interaction at time t, the history features aggregate only the same
user's reviews with time < t. Demographics are the row's own self-reported
profile (available at serving time as profile info, and not derived from the
label). Review helpfulness / feedback counts are deliberately **not** used:
they are 2023 snapshots that keep accruing after a review is posted, so they
would leak the future into early rows.
"""

import numpy as np
import pandas as pd

from src.features.asof import asof_prior_sums
from src.features.text import has_text

DEMOGRAPHIC_COLUMNS = ["skin_tone", "skin_type", "eye_color", "hair_color"]


def text_columns(n: int) -> list[str]:
    """Names of the text-vector columns: u_text_00, u_text_01, ..."""
    return [f"u_text_{i:02d}" for i in range(n)]


def build_user_asof(df: pd.DataFrame, text_emb: np.ndarray) -> pd.DataFrame:
    """Raw (unscaled, un-imputed) user features, one row per interaction in ``df``.

    ``text_emb`` is the (len(df), d) embedding of each row's own review; it is
    only ever used through *earlier* rows of the same user.
    """
    d = text_emb.shape[1]
    tcols = text_columns(d)
    rec = df["is_recommended"]
    texted = has_text(df["review_text"])

    values = pd.DataFrame(
        {
            "n": 1.0,
            "rating": df["rating"].to_numpy(),
            "rec": rec.fillna(0).to_numpy(),
            "rec_n": rec.notna().astype(float).to_numpy(),
            "text_n": texted.astype(float),
        },
        index=df.index,
    )
    values[tcols] = text_emb  # zeros where the review has no text
    prior = asof_prior_sums(df["author_id"], df["submission_time"], values)

    n = prior["n"]
    out = pd.DataFrame(index=df.index)
    out["u_hist_count"] = n
    out["u_has_history"] = (n > 0).astype(float)
    out["u_rating_mean"] = (prior["rating"] / n).where(n > 0)
    out["u_rec_mean"] = (prior["rec"] / prior["rec_n"]).where(prior["rec_n"] > 0)
    text_n = prior["text_n"]
    text_mean = prior[tcols].div(text_n.where(text_n > 0), axis=0)  # NaN when no prior text
    out = pd.concat([out, text_mean], axis=1)
    for c in DEMOGRAPHIC_COLUMNS:
        out[c] = df[c]
    return out
