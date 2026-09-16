with products as (
    select * from {{ ref('stg_products') }}
),

deduped as (
    select *
    from products
    qualify row_number() over (partition by product_id order by product_id) = 1
)

select
    product_id,
    product_name,
    brand_id,
    brand_name,
    primary_category,
    secondary_category,
    tertiary_category,
    variation_type,
    variation_value,
    size_raw,
    price_usd,
    value_price_usd,
    sale_price_usd,
    loves_count,
    avg_rating,
    review_count,
    is_limited_edition,
    is_new,
    is_online_only,
    is_out_of_stock,
    is_sephora_exclusive,
    child_count,
    child_max_price,
    child_min_price
from deduped
