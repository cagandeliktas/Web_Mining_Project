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
scripts/                    thin CLIs over the above
notebooks/                  calls src/ only (a test fails if a notebook defines functions)
```

Run the tests with `pytest -v`. CI (`.github/workflows/tests.yml`) installs the CPU build of torch and runs the same
suite on every push and pull request against `main`.

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

This dbt project runs as a **native dbt Project inside Snowsight** (no local
dbt install), connected read-only to this GitHub repo — the model SQL lives
here as real, version-controlled code, but execution happens entirely in
Snowflake's UI.

## Roadmap

- Serve the PyTorch model (FastAPI) using the same fitted transforms and as-of feature code as training.
- Train for ranking (e.g. in-batch softmax) instead of rating regression; add paired significance tests between methods.
- Build the dbt layer's user/item aggregates *as of each review* (window functions over earlier rows) before using them as ML features: `dim_users` aggregates over all time, so using it directly would leak.

## Previous version

The earlier Keras/TensorFlow implementation (monolithic notebook, FastAPI service, A/B script, random-split results)
is preserved on the [`deprecated`](https://github.com/cagandeliktas/Web_Mining_Project/tree/deprecated) branch.
Its reported numbers came from a leaky random split and should not be compared with the results above.
