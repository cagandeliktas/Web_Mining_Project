import numpy as np
import pandas as pd

from src.item_features import build_item_feature_table

RAW_COLUMNS = [
    "product_id",
    "product_name",
    "brand_id",
    "brand_name",
    "loves_count",
    "rating",
    "reviews",
    "size",
    "variation_type",
    "variation_value",
    "variation_desc",
    "ingredients",
    "price_usd",
    "value_price_usd",
    "sale_price_usd",
    "limited_edition",
    "new",
    "online_only",
    "out_of_stock",
    "sephora_exclusive",
    "highlights",
    "primary_category",
    "secondary_category",
    "tertiary_category",
    "child_count",
    "child_max_price",
    "child_min_price",
]


def _make_product_info(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[RAW_COLUMNS]


def test_one_hot_encodes_variation_type_and_primary_category():
    df = _make_product_info(
        [
            {
                "product_id": "P1",
                "brand_id": 1,
                "variation_type": "Color",
                "primary_category": "Makeup",
                "price_usd": 20.0,
                "size": "30 ml",
            },
            {
                "product_id": "P2",
                "brand_id": 2,
                "variation_type": "Size",
                "primary_category": "Skincare",
                "price_usd": 40.0,
                "size": "50 ml",
            },
        ]
    )
    out = build_item_feature_table(df)

    assert "var_type_Color" in out.columns
    assert "var_type_Size" in out.columns
    assert "pc_Makeup" in out.columns
    assert "pc_Skincare" in out.columns
    assert out.loc[out.product_id == "P1", "var_type_Color"].iloc[0] == 1
    assert out.loc[out.product_id == "P1", "pc_Makeup"].iloc[0] == 1
    assert out.loc[out.product_id == "P1", "var_type_Size"].iloc[0] == 0


def test_missing_variation_type_becomes_unknown_category():
    df = _make_product_info(
        [
            {"product_id": "P1", "brand_id": 1, "variation_type": np.nan, "primary_category": "Makeup"},
            {"product_id": "P2", "brand_id": 2, "variation_type": "Color", "primary_category": "Makeup"},
        ]
    )
    out = build_item_feature_table(df)

    assert "var_type_Unknown" in out.columns
    assert out.loc[out.product_id == "P1", "var_type_Unknown"].iloc[0] == 1


def test_size_is_parsed_and_original_columns_dropped():
    df = _make_product_info(
        [
            {"product_id": "P1", "brand_id": 1, "primary_category": "Makeup", "size": "1.7 oz"},
            {"product_id": "P2", "brand_id": 2, "primary_category": "Makeup", "size": np.nan},
        ]
    )
    out = build_item_feature_table(df)

    assert "size" not in out.columns
    assert "size_ml" not in out.columns
    assert out.loc[out.product_id == "P1", "size_missing"].iloc[0] == 0
    assert out.loc[out.product_id == "P2", "size_missing"].iloc[0] == 1
    # size_ml_log for the missing row is backfilled with the column median, not NaN
    assert not out["size_ml_log"].isna().any()


def test_dropped_columns_are_not_present():
    df = _make_product_info(
        [{"product_id": "P1", "brand_id": 1, "primary_category": "Makeup", "product_name": "Foo"}]
    )
    out = build_item_feature_table(df)

    for col in [
        "product_name",
        "brand_name",
        "value_price_usd",
        "sale_price_usd",
        "variation_value",
        "variation_desc",
        "tertiary_category",
        "secondary_category",
        "highlights",
        "ingredients",
        "brand_id",
        "brand_freq",
        "brand_freq_log",
    ]:
        assert col not in out.columns


def test_missing_numeric_fields_are_imputed_with_median():
    df = _make_product_info(
        [
            {"product_id": "P1", "brand_id": 1, "primary_category": "Makeup", "rating": np.nan, "reviews": np.nan},
            {"product_id": "P2", "brand_id": 2, "primary_category": "Makeup", "rating": 4.0, "reviews": 10},
            {"product_id": "P3", "brand_id": 3, "primary_category": "Makeup", "rating": 5.0, "reviews": 20},
        ]
    )
    out = build_item_feature_table(df)

    assert not out["rating"].isna().any()
    assert out.loc[out.product_id == "P1", "rating"].iloc[0] == 4.5  # median of 4.0, 5.0
    assert out.loc[out.product_id == "P1", "reviews"].iloc[0] == 0  # reviews NaN -> 0, not median


def test_brand_freq_scaled_is_between_zero_and_one():
    df = _make_product_info(
        [
            {"product_id": "P1", "brand_id": 1, "primary_category": "Makeup"},
            {"product_id": "P2", "brand_id": 1, "primary_category": "Makeup"},
            {"product_id": "P3", "brand_id": 2, "primary_category": "Makeup"},
        ]
    )
    out = build_item_feature_table(df)

    assert out["brand_freq_scaled"].between(0, 1).all()
    # brand_id 1 appears twice (more frequent) so should score higher than brand_id 2
    assert (
        out.loc[out.product_id == "P1", "brand_freq_scaled"].iloc[0]
        > out.loc[out.product_id == "P3", "brand_freq_scaled"].iloc[0]
    )
