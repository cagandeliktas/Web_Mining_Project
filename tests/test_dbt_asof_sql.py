"""Run the new dbt models' SQL in DuckDB and compare them with the Python pipeline.

dbt itself only executes inside Snowflake, so this is the local proof that
int_interactions / split_cutoffs / fct_interactions_asof compute the same
interactions, split and as-of aggregates as src/data.py, src/split.py and
src/features/. The models use only portable SQL (window functions, QUALIFY);
the test renders the two Jinja constructs they use ({{ ref() }}, {{ var() }}).
"""

import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.data import clean_reviews
from src.features.asof import asof_prior_sums
from src.split import temporal_split
from tests.synthetic import make_interactions

DBT = Path(__file__).resolve().parent.parent / "dbt_project"
VARS = {"val_frac": "0.10", "test_frac": "0.10", "item_shrinkage": "10"}
MODEL_ORDER = [("intermediate", "int_interactions"), ("intermediate", "split_cutoffs"), ("marts", "fct_interactions_asof")]


def render(sql: str, overrides: dict | None = None) -> str:
    sql = re.sub(r"\{\{\s*ref\('([^']+)'\)\s*\}\}", r"\1", sql)
    return re.sub(r"\{\{\s*var\('([^']+)'\)\s*\}\}", lambda m: {**VARS, **(overrides or {})}[m.group(1)], sql)


def read(path: Path) -> str:
    return path.read_text()


def build_warehouse(raw: pd.DataFrame, fct_transform=None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("stg_reviews", raw)
    for folder, name in MODEL_ORDER:
        sql = read(DBT / "models" / folder / f"{name}.sql")
        if name == "fct_interactions_asof" and fct_transform:
            sql = fct_transform(sql)
        con.execute(f"create view {name} as {render(sql)}")
    return con


@pytest.fixture(scope="module")
def raw_reviews() -> pd.DataFrame:
    """Messy stg_reviews-like table: some pairs reviewed several times, feedback counts, ties."""
    base = make_interactions(n=900, n_users=60, n_items=15)
    rng = np.random.default_rng(5)
    extra = base.sample(frac=0.2, random_state=2).copy()
    extra["submission_time"] = extra["submission_time"] + pd.to_timedelta(rng.integers(-30, 40, len(extra)), unit="D")
    extra["rating"] = rng.integers(1, 6, len(extra))
    raw = pd.concat([base, extra], ignore_index=True)
    raw["total_feedback_count"] = rng.integers(0, 4, len(raw)).astype(float)
    return raw.sample(frac=1.0, random_state=9).reset_index(drop=True)


@pytest.fixture(scope="module")
def expected(raw_reviews):
    df = clean_reviews(raw_reviews)
    labels, _ = temporal_split(df, 0.10, 0.10)
    vals = pd.DataFrame({"n": 1.0, "r": df["rating"].astype(float), "rec": df["is_recommended"].fillna(0.0),
                         "rec_n": df["is_recommended"].notna().astype(float)}, index=df.index)
    u = asof_prior_sums(df["author_id"], df["submission_time"], vals)
    i = asof_prior_sums(df["product_id"], df["submission_time"], vals)
    out = df[["author_id", "product_id", "submission_time", "rating"]].copy()
    out["split"] = labels.astype(str).to_numpy()
    out["u_hist_count"], out["u_rating_sum"], out["u_rec_n"] = u["n"], u["r"], u["rec_n"]
    out["u_rating_mean"] = (u["r"] / u["n"]).where(u["n"] > 0)
    out["u_rec_mean"] = (u["rec"] / u["rec_n"]).where(u["rec_n"] > 0)
    out["i_hist_count"], out["i_rating_sum"] = i["n"], i["r"]
    mu = df.loc[labels == "train", "rating"].mean()
    out["i_rating_mean"] = (i["r"] + 10 * mu) / (i["n"] + 10)
    return out.set_index(["author_id", "product_id"]).sort_index(), mu


@pytest.fixture(scope="module")
def warehouse(raw_reviews):
    return build_warehouse(raw_reviews)


def test_interactions_split_and_asof_aggregates_match_the_python_pipeline(warehouse, expected):
    want, mu = expected
    got = warehouse.execute("select * from fct_interactions_asof").df().set_index(["author_id", "product_id"]).sort_index()
    assert len(got) == len(want) and got.index.equals(want.index)
    assert (got["split"] == want["split"]).all()
    assert (got["submission_time"] == want["submission_time"]).all()
    for col in ["u_hist_count", "u_rating_sum", "u_rec_n", "u_rating_mean", "u_rec_mean",
                "i_hist_count", "i_rating_sum", "i_rating_mean"]:
        np.testing.assert_allclose(got[col].astype(float), want[col].astype(float), atol=1e-9, equal_nan=True, err_msg=col)
    cut = warehouse.execute("select * from split_cutoffs").df().iloc[0]
    assert cut["train_mean_rating"] == pytest.approx(mu)


def test_latest_review_wins_not_the_average(warehouse, raw_reviews):
    multi = raw_reviews.groupby(["author_id", "product_id"]).size()
    pair = multi[multi > 1].index[0]
    rows = raw_reviews[(raw_reviews.author_id == pair[0]) & (raw_reviews.product_id == pair[1])].sort_values("submission_time")
    got = warehouse.execute("select rating from int_interactions where author_id=? and product_id=?", list(pair)).fetchall()
    assert got == [(rows.iloc[-1]["rating"],)]


@pytest.mark.parametrize("path", sorted((DBT / "tests").glob("*.sql")), ids=lambda p: p.name)
def test_dbt_singular_tests_pass_on_clean_data(warehouse, path):
    assert warehouse.execute(render(read(path))).df().empty


def test_the_leakage_guard_actually_catches_a_leaky_model(raw_reviews):
    """Swap the frame end '1 preceding' -> 'current row': the model then includes the row itself."""
    leaky = build_warehouse(raw_reviews, fct_transform=lambda s: s.replace("and 1 preceding", "and current row"))
    problems = leaky.execute(render(read(DBT / "tests" / "assert_asof_excludes_the_present.sql"))).df()
    assert len(problems) > 0
