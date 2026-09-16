# Web_Mining_Project

[![Tests](https://github.com/cagandeliktas/Web_Mining_Project/actions/workflows/tests.yml/badge.svg)](https://github.com/cagandeliktas/Web_Mining_Project/actions/workflows/tests.yml)
[![API Tests](https://github.com/cagandeliktas/Web_Mining_Project/actions/workflows/api-tests.yml/badge.svg)](https://github.com/cagandeliktas/Web_Mining_Project/actions/workflows/api-tests.yml)

A content-based neural network recommendation model consists of two parallel subnetworks—one for users and one for products—each designed to learn latent representations from their respective feature sets. These embeddings capture user preferences and product characteristics in a shared vector space, enabling the model to predict interactions such as ratings or affinities.

*User Subnetwork:

The user subnetwork takes as input a rich set of features that reflect both user identity and behavior. These include unique identifiers such as author_id, behavioral statistics like average rating, number of reviews, and helpfulness score, as well as personal attributes such as skin tone. Additionally, semantic information from user reviews is represented using TF-IDF and compressed via Truncated SVD. These features are processed through the user subnetwork’s layers to produce a compact user embedding vector V_u, which represents the user’s preferences.

*Product Subnetwork:

The product subnetwork receives features that describe the product’s properties and overall appeal. These include numerical attributes such as price, average rating, and number of reviews, as well as binary indicators of availability and exclusivity—such as whether the product is new, limited edition, or sold exclusively by the platform. Categorical product descriptors like type, formulation, and category (e.g., makeup, skincare) provide contextual detail, while brand frequency and size-related attributes help capture product scale and popularity. These inputs are also passed through the product subnetwork’s layers to produce the product embedding vector V_m.

Scoring:

The final prediction is computed as the dot product of the user and product embedding vectors:

score = V_u * V_m (dot product)

This scalar score reflects the compatibility between a given user and product and can be used for tasks like ranking, recommendation, or rating prediction. The content-based neural network architecture is particularly effective in large-scale recommendation systems, where it allows for efficient candidate generation and retrieval by separately encoding users and items.

**Additionally, this project includes matrix factorization using SVD, as well as simple content-based and collaborative filtering approaches.

Dataset:

https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews/data  

•	product_info: stores product information  
•	reviews_0-250.csv: Contains reviews for products indexed from 0 to 250.  
•	reviews_250-500.csv: Contains reviews for products indexed from 250 to 500.  
•	reviews_500-750.csv: Contains reviews for products indexed from 500 to 750.  
•	reviews_750-1250.csv: Contains reviews for products indexed from 750 to 1250.  
•	reviews_1250-end.csv: Contains reviews for products indexed from 1250 to the last product in the dataset.  

## Development

The notebooks are exploratory (feature engineering, model training, hyperparameter
tuning). A few reusable pieces of logic from `NN_w_textFeatures.ipynb` have been
extracted into a small tested package under `src/`, so they can be verified in
isolation instead of only being checked by re-running the whole notebook:

- `src/preprocessing.py` — parsing the raw `size` field into millilitres.
- `src/metrics.py` — the precision/recall/F1/accuracy and precision@k/nDCG@k
  evaluation functions used to score the two-tower model.
- `src/similarity.py` — the nearest-neighbour lookup over item embeddings
  (parameterized to take `prod_ids` and the embedding matrix explicitly, so it
  doesn't depend on notebook globals).

Run the tests locally with:

```bash
pip install -r requirements-dev.txt
pytest -v
```

A GitHub Actions workflow (`.github/workflows/tests.yml`) runs the same test suite
on every push and pull request against `main`.

## Serving the model (FastAPI)

The trained two-tower model is served behind a small FastAPI app with a single
endpoint:

```bash
pip install -r requirements-api.txt
uvicorn app.main:app --reload
```

`POST /recommendations` takes a user's review history (optionally empty, for a
cold-start/anonymous user) and returns the top-K products the model predicts
they'd rate highest:

```bash
curl -X POST localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"reviews": [{"rating": 5, "is_recommended": true, "review_text": "Loved it!",
                      "skin_tone": "medium", "skin_type": "oily",
                      "eye_color": "brown", "hair_color": "black"}],
       "top_k": 5}'
```

`app/inference.py` ports the notebook's `build_user_vector` logic (now
`src/user_features.py`) and scores the user vector against every product in
`artifacts/item_matrix.pkl`, a precomputed catalog feature matrix built by
`scripts/build_item_matrix.py`.

**A note on faithfulness, not just runnability:** the notebook's own "let's
try recommending for a new user" cells (near the bottom of
`NN_w_textFeatures.ipynb`) turned out to feed the model **unscaled** item
features, even though the model was trained and evaluated on features scaled
with `StandardScaler` (`artifacts/scaler_item.pkl`). This API deliberately
does *not* reproduce that bug — it applies the same scaling the model was
actually trained on, which is the technically correct approach, even though
it means the API's recommendations differ from the notebook's cached example
output. See `scripts/build_item_matrix.py`'s docstring for the details; this
was caught by building a small script to reproduce the notebook's example
predictions from scratch and finding they only matched once the scaling step
was (incorrectly) skipped.

Preprocessing artifacts (`artifacts/*.pkl`) and `artifacts/product_info.csv`
are committed so the API is self-contained. The large raw review CSVs are
not committed (see `.gitignore`) - they're only needed to regenerate
`artifacts/item_matrix.pkl` via `scripts/build_item_matrix.py`, since that's
the one thing that requires knowing which products actually have reviews:

```bash
python scripts/build_item_matrix.py --data-dir /path/to/sephora_datasets
```

Because the model needs TensorFlow (a large, slow dependency), its tests
live in a separate suite (`tests/test_api_regression.py`, marked `api`) and
a separate, slower workflow (`.github/workflows/api-tests.yml`), so the fast
unit tests above stay fast on every push.

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

## A/B test: classical CF vs. the two-tower NN

`scripts/run_ab_comparison.py` compares `Basic_Recommender_System.ipynb`'s best
classical baseline (KNNWithMeans, item-based Pearson similarity — variant A)
against the two-tower neural network (variant B) with a proper paired
significance test, not just eyeballing two separate metrics.

**Methodology:** a leave-one-out evaluation. For every user with at least 2
ratings, one (product, rating) pair is held out and predicted from
everything else that user rated — variant A retrains on the same held-out
split, and variant B builds each user's vector from their remaining reviews
via `src/user_features.py`'s `build_user_vector` (the same function the
FastAPI service uses for a real user). Both variants are scored on the
*exact same* 1,000 held-out instances, which is what makes a paired test
(`src/ab_testing.py`) meaningful rather than comparing two unrelated
samples.

```bash
pip install -r requirements-ab.txt
python scripts/run_ab_comparison.py --data-dir /path/to/sephora_datasets
```

**Result** (`ab_test_results.json`, n=1,000): KNNWithMeans significantly
outperformed the two-tower NN — mean absolute error 0.644 vs. 0.837
(paired t-test p ≈ 1×10⁻¹², Wilcoxon signed-rank p ≈ 1×10⁻¹³).

This is a genuine, not-hoped-for result. The two-tower model's *own training
setup* had the same class of leak this evaluation was designed to avoid —
see the next section for the full fix and the retrained model this
comparison now actually runs against.

## Fixing the leak at its source: retraining the two-tower model

The A/B test above was built to be leak-free on its own terms, but that just
exposed a deeper problem: the leak wasn't only in how the *original* NN got
evaluated — it was baked into how it was *trained*. `user_features_ordered`
in the notebook built each user's feature vector (rating averages, TF-IDF/SVD
text embedding, sentiment, demographics) from **all** of that user's reviews,
*then* split rows into train/test — so a user's test-row profile included
that same test review's own rating and text. TF-IDF and SVD were also fit on
the full corpus, including test-set review text.

`NeuralNetworkApproach/NN_w_textFeatures.ipynb` now has a new
**"2. Correction: leak-free retraining"** section (appended after the
original, untouched cells) that splits review rows into train/test *first*,
fits TF-IDF/SVD/scalers on the train portion only, and builds every user's
profile from train-side reviews only — the same discipline `build_user_vector`
already uses at serving time, now applied at training time too. Running the
full Hyperband search this notebook section describes is realistically a
~60 minute job (not the many hours it looks like on paper — the model itself
is small; most of the cost is one-time feature-engineering setup), so the
actual production model was produced by running the equivalent background
script instead of tying up an interactive kernel:

```bash
pip install -r requirements-train.txt
python scripts/train_leakfree_model.py --data-dir /path/to/sephora_datasets --search full
```

**Result:** the original notebook reported a test RMSE of **0.582** (MAE
0.316) on its own (leaky) split. Retrained leak-free, with a full 90-trial
Hyperband search over the corrected data: test RMSE **1.087** (MAE 0.730) —
roughly **double** the error. That gap *is* the leak: closing it doesn't
just change an evaluation number, it reveals the model was never actually
that good at predicting a genuinely unseen review.

This is now the production model — `artifacts/*.pkl`, `artifacts/item_matrix.pkl`,
and `NeuralNetworkApproach/final_two_tower_model_last.keras` were all
regenerated from this retrain, the FastAPI service and the A/B test above
both run against it, and both were re-verified against it (`pytest -v -m api`,
`scripts/run_ab_comparison.py`) after the swap.
