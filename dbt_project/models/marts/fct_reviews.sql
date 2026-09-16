with reviews as (
    select * from {{ ref('stg_reviews') }}
)

select
    -- No natural review id exists in the raw data, so build a stable
    -- surrogate key from the fields that together identify one review.
    md5(concat_ws(
        '-', author_id, product_id, to_varchar(submission_time), review_text
    )) as review_id,
    author_id,
    product_id,
    rating,
    is_recommended,
    helpfulness,
    total_feedback_count,
    total_neg_feedback_count,
    total_pos_feedback_count,
    submission_time,
    review_text,
    review_title
from reviews
