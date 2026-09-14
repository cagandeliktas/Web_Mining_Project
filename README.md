# Web_Mining_Project

[![Tests](https://github.com/cagandeliktas/Web_Mining_Project/actions/workflows/tests.yml/badge.svg)](https://github.com/cagandeliktas/Web_Mining_Project/actions/workflows/tests.yml)

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
