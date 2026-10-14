# Web_Mining_Project

[![Tests](https://github.com/cagandeliktas/Web_Mining_Project/actions/workflows/tests.yml/badge.svg)](https://github.com/cagandeliktas/Web_Mining_Project/actions/workflows/tests.yml)

A content-based **two-tower** recommender in **PyTorch**, built to be evaluated honestly: a temporal
split, point-in-time ("as-of") features with no label leakage, and real baselines to beat.

```
y_hat = mu + b_user + b_item + <tower_user(x_user), tower_item(x_item)>
```

- **User tower** - the user's strictly-*earlier* reviews (count, mean rating, recommend rate, mean TF-IDF/SVD text embedding)
  plus self-reported skin/eye/hair profile.
- **Item tower** - static catalogue attributes (price, size, brand breadth, flags) plus the item's *as-of* review
  count and shrunk mean rating.
- **Bias head** - `mu` is the train mean; `b_user`/`b_item` are looked up by id, with slot 0 (id unseen in train)
  pinned to zero, so brand-new users and products are scored from content alone.

Everything is in `src/` (unit-tested in CI); `notebooks/two_tower_pytorch.ipynb` only *calls* it.

## Results

Temporal split by review date (train to 2022-02-01, val to 2022-08-15, test 2022-08-16 to 2023-03-21; 55% of test
rows are from users, and 44% from products, with no train review). Baseline shrinkage is tuned on val only.

| Test set (n = 108,910) | RMSE | MAE |
|---|---|---|
| **Two-tower (PyTorch)** | **1.026** | 0.739 |
| User + item bias, *as-of* (same history the NN sees) | 1.042 | **0.696** |
| User + item bias (train-only) | 1.078 | 0.809 |
| User mean | 1.080 | 0.807 |
| Item mean | 1.104 | 0.857 |
| Item-kNN with means | 1.108 | 0.857 |
| Global mean | 1.112 | 0.872 |

What to take from it:

- The two-tower wins on RMSE but only by ~1.5% over the as-of bias baseline, and **loses on MAE** (64% of ratings are
  5 stars, so a squared-error model predicts a bit low). Its edge is largest for users with no history
  (1.274 vs 1.296) and for 2-4 prior reviews (1.015 vs 1.030); with 5+ prior reviews it is a tie.
- At **full-catalog ranking** (rank the product a user actually reviewed and liked among ~2,200 candidates) it is
  only slightly above random and below simple baselines: HR@10 is roughly 0.009-0.013 across runs, against 0.005
  random, 0.019 popularity and 0.050 "highest mean rating". A rating regressor is not trained to rank; a
  ranking loss (e.g. in-batch softmax) is the natural next step.
- Run-to-run variation: training on Apple's `mps` backend is not bit-deterministic, so ranking numbers move a little
  between runs; the same-seed determinism tests run on CPU.
- The numbers in earlier versions of this README (RMSE 1.087 / MAE 0.730, on the `deprecated` branch) came from a
  random row split with label-leaking user features and are **not comparable**.

Full tables (by history bucket, val and test, ranking @5/10/20) are in [`results/results.md`](results/results.md);
`python -m scripts.evaluate` regenerates them into `runs/latest/results.md`.

### Why these numbers are trustworthy

- **Temporal split**, never a random row split, with `assert_no_leakage` on time ordering and on (user, product) pairs.
- **As-of features**: a row's features use only reviews with a strictly earlier timestamp, so its own rating, its own
  text and same-day reviews never enter. Tests overwrite a row's label/text, or everything at or after time T, and
  check the features do not move (`tests/test_features_leakage.py`).
- **Fit on train only**: TF-IDF, SVD, imputation values, scalers, category lists, id vocabularies and the global mean.
- **Dropped future-bearing columns**: `product_info`'s `rating`, `reviews`, `loves_count`, `out_of_stock` and review
  helpfulness/feedback counts are 2023 snapshots that keep accruing after a review is posted.
- **One interaction per (user, product)**: the latest review wins (the old pipeline averaged, blending later reviews
  into earlier timestamps).

## Quick start

First download the Kaggle dataset (see *Dataset* below) into `sephora_datasets/` (gitignored).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt          # torch, pandas, scikit-learn, pytest, ...
python -m scripts.prepare_data               # clean the raw CSVs + print the split report
python -m scripts.build_features             # as-of features, cached in data_cache/
python -m scripts.train                      # trains, early-stops on val, saves runs/latest/model.pt
python -m scripts.evaluate                   # baselines vs the model on val/test -> runs/latest/results.md
```

All hyper-parameters live in `configs/default.yaml`. For the notebook: `pip install -r requirements-notebook.txt`.

Dataset:

https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews/data  

•	product_info: stores product information  
•	reviews_0-250.csv: Contains reviews for products indexed from 0 to 250.  
•	reviews_250-500.csv: Contains reviews for products indexed from 250 to 500.  
•	reviews_500-750.csv: Contains reviews for products indexed from 500 to 750.  
•	reviews_750-1250.csv: Contains reviews for products indexed from 750 to 1250.  
•	reviews_1250-end.csv: Contains reviews for products indexed from 1250 to the last product in the dataset.  

## Development

```
configs/default.yaml        single source of truth for split, features, model, training
src/data.py                 load + clean reviews (mirrors dbt stg_reviews), one interaction per (user, product)
src/split.py                temporal split, leakage assertions, cold-start report
src/features/               as-of aggregation, user / item / text features
src/transforms.py           fit-on-train scaling, imputation, one-hot, id vocabularies
src/pipeline.py             FeatureSet + load_or_build_features (cached)
src/torch_data.py           in-memory torch Dataset
src/models/two_tower.py     the model
src/train.py                training loop (AdamW, bias shrinkage, id dropout, val early stopping)
src/baselines.py            mean / item / user / bias / as-of / item-kNN baselines
src/evaluate.py, ranking.py rating metrics by history bucket; full-catalog ranking
src/benchmark.py            run_benchmark: every method on val/test
src/ab_testing.py           paired tests + user-clustered bootstrap between two predictors
src/serving/                bundle export, request-time features, Recommender, FastAPI app
scripts/                    thin CLIs over the above
notebooks/                  calls src/ only (a test fails if a notebook defines functions)
```

Run the tests with `pytest -v`. CI (`.github/workflows/tests.yml`) installs the CPU build of torch and runs the same
suite on every push and pull request against `main`.

## Is the two-tower really better? (paired tests)

`python -m scripts.compare_methods` compares the model with every baseline on the *same* test rows. Reviews by one user
are correlated, so the evidence is a **user-clustered bootstrap** (whole users are resampled); row-level paired t and
Wilcoxon tests are also computed (`results/ab_test.json`) but overstate certainty. Positive delta = the two-tower is better.

Two-tower vs baselines, test split (108,910 reviews, 50,522 users). delta = baseline - two-tower, so positive means the two-tower is better. 95% CIs and p-values come from a user-clustered bootstrap (1000 resamples); p-values are Holm-adjusted across the 6 baselines. The smallest p a bootstrap this size can give is about 0.001 per comparison, so a value at that floor means no resample crossed zero.

|                 |   rmse (baseline) |   rmse (two-tower) |   delta rmse | 95% CI             |   p (cluster boot) |   delta mae | 95% CI (mae)       |   p (mae) |
|:----------------|------------------:|-------------------:|-------------:|:-------------------|-------------------:|------------:|:-------------------|----------:|
| user+item as-of |             1.042 |              1.026 |      0.01638 | [+0.0148, +0.0179] |           0.005994 |    -0.0438  | [-0.0457, -0.0420] |  0.005994 |
| user+item bias  |             1.078 |              1.026 |      0.05154 | [+0.0491, +0.0539] |           0.005994 |     0.07005 | [+0.0671, +0.0734] |  0.005994 |
| user mean       |             1.08  |              1.026 |      0.0535  | [+0.0503, +0.0567] |           0.005994 |     0.068   | [+0.0646, +0.0714] |  0.005994 |
| item mean       |             1.104 |              1.026 |      0.07795 | [+0.0751, +0.0810] |           0.005994 |     0.1178  | [+0.1136, +0.1225] |  0.005994 |
| item kNN (k=40) |             1.108 |              1.026 |      0.08172 | [+0.0789, +0.0846] |           0.005994 |     0.1178  | [+0.1135, +0.1224] |  0.005994 |
| global mean     |             1.112 |              1.026 |      0.08553 | [+0.0821, +0.0893] |           0.005994 |     0.1325  | [+0.1281, +0.1373] |  0.005994 |

Strongest baseline (`user+item as-of`) by user history:

|                               |   rmse (baseline) |   rmse (two-tower) |   delta rmse | 95% CI             |   p (cluster boot) |   delta mae | 95% CI (mae)       |   p (mae) |
|:------------------------------|------------------:|-------------------:|-------------:|:-------------------|-------------------:|------------:|:-------------------|----------:|
| user+item as-of - no history  |            1.296  |             1.274  |      0.02222 | [+0.0201, +0.0243] |           0.000999 |    -0.0155  | [-0.0175, -0.0135] |  0.000999 |
| user+item as-of - has history |            0.8578 |             0.8459 |      0.01182 | [+0.0097, +0.0141] |           0.000999 |    -0.06056 | [-0.0629, -0.0581] |  0.000999 |

Reading it: the two-tower's RMSE advantage is statistically solid against every baseline, including the as-of one
(delta RMSE +0.016, 95% CI [+0.015, +0.018]), but small, and against that same baseline it is *significantly worse* on MAE
(-0.044). Whether it is "better" depends on whether squared or absolute error is the goal.

## Serving (FastAPI)

The API reuses the training code instead of re-implementing it: a request's reviews are treated as the user's earlier
history, one "query" row is appended, and the *same* `build_user_asof` + train-fitted transform produce the user vector
(a test checks it equals the training features for the same user state). Item embeddings, item biases and metadata are
precomputed into a catalog snapshot, so a request is one small MLP pass plus a matrix-vector product (~30 ms).

```bash
pip install -r requirements-serving.txt
python -m scripts.train && python -m scripts.export_serving_bundle    # -> artifacts/serving/ (5 MB)
uvicorn src.serving.api:app --reload                                  # RECSYS_BUNDLE=<dir> to use another bundle
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | status, number of products, catalog snapshot date |
| `POST /recommendations` | top-K products by predicted rating for a user (`reviews`, optional `profile`, `top_k`, `exclude_product_ids`, `min_item_reviews`) |
| `POST /predict` | predicted rating of one `product_id` for that user (404 for an unknown product) |

```bash
curl -X POST localhost:8000/recommendations -H "Content-Type: application/json" -d '{
  "reviews": [{"rating": 5, "is_recommended": true, "review_text": "Loved it!", "skin_type": "oily", "product_id": "P420652"}],
  "profile": {"skin_tone": "medium"}, "top_k": 5, "min_item_reviews": 50}'
```

An empty `reviews` list is a cold-start user (scored from `profile` and content only). Unseen users get the cold bias
slot, as in evaluation. **Caveat:** `/recommendations` ranks by predicted *rating*, which the evaluation above shows is a
weak retrieval signal (it tends to surface niche, highly-rated items; `min_item_reviews` helps). `/predict` is the
model's stronger use. The bundle is a snapshot as of its export date.

## Data warehouse (Snowflake + dbt)

The raw Sephora CSVs are also loaded into a Snowflake database (`WEB_MINING`)
and modeled into a small star schema with dbt:

- **`WEB_MINING.RAW`** — the raw landing zone: `product_info.csv` loaded via
  Snowsight's upload wizard, and the review CSVs loaded with
  `scripts/load_reviews_to_snowflake.py` (too large for the browser
  uploader — one file alone is 282MB).
- **`dbt_project/`** — dbt models that clean and reshape the raw tables into:
  - `stg_products` / `stg_reviews` (staging, 1:1 cleaning over each source)
  - `dim_products`, `dim_users`, `fct_reviews` (the star schema — one row
    per product, one row per reviewer, one row per review, with dbt tests
    for uniqueness, not-null, and referential integrity between them)

- **`fct_interactions_asof`** (with `int_interactions` and `split_cutoffs`) - one row per interaction with the temporal
  split label and point-in-time user/item aggregates, each using only *strictly earlier* reviews, so it is safe as ML
  input. `dim_users` / `dim_products` aggregate over all time and must not be used as features. Singular dbt tests
  (`dbt_project/tests/`) fail on any train/val/test time overlap or an "as-of" count that includes the row itself.
  Because dbt only runs inside Snowflake, `tests/test_dbt_asof_sql.py` executes the same SQL in DuckDB and checks it
  against the Python pipeline (identical interactions, split and aggregates; also verified on the full 1.09M rows).
  Also run on Snowflake: `dbt build --select +fct_interactions_asof` finished with PASS=19 WARN=0 ERROR=0, and
  `dbt_project/analyses/verify_fct_interactions_asof.sql` returned exactly the Python figures (1,088,824 interactions,
  870,476 / 109,438 / 108,910 train / val / test, cut-offs 2022-02-02 and 2022-08-16, train mean 4.2905).

This dbt project runs as a **native dbt Project inside Snowsight** (no local
dbt install), connected read-only to this GitHub repo — the model SQL lives
here as real, version-controlled code, but execution happens entirely in
Snowflake's UI.

## Roadmap

- Serving: look up known users by id (their learned bias), refresh the catalog snapshot on a schedule, and rank with a retrieval-trained model.
- Train for ranking (e.g. in-batch softmax) instead of rating regression.
- Read the training features from `fct_interactions_asof` in Snowflake instead of recomputing them in Python (text embeddings stay in Python).

## Previous version

The earlier Keras/TensorFlow implementation (monolithic notebook, FastAPI service, A/B script, random-split results)
is preserved on the [`deprecated`](https://github.com/cagandeliktas/Web_Mining_Project/tree/deprecated) branch.
Its reported numbers came from a leaky random split and should not be compared with the results above.
