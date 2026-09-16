with source as (
    select * from {{ source('raw', 'product_info_raw') }}
),

renamed as (
    select
        product_id,
        product_name,
        brand_id,
        brand_name,
        loves_count,
        rating          as avg_rating,
        reviews         as review_count,
        size            as size_raw,
        variation_type,
        variation_value,
        price_usd,
        value_price_usd,
        sale_price_usd,
        limited_edition = 1  as is_limited_edition,
        new             = 1  as is_new,
        online_only     = 1  as is_online_only,
        out_of_stock    = 1  as is_out_of_stock,
        sephora_exclusive = 1 as is_sephora_exclusive,
        primary_category,
        secondary_category,
        tertiary_category,
        child_count,
        child_max_price,
        child_min_price
    from source
)

select * from renamed
