with source as (
    select * from {{ source('raw', 'reviews_raw') }}
),

-- author_id is occasionally corrupted (letters mixed into what should be a
-- numeric id) - same data-quality rule already validated in the two-tower
-- notebook's df_reviews_final construction and scripts/build_item_matrix.py.
cleaned as (
    select
        author_id,
        product_id,
        rating,
        is_recommended,
        helpfulness,
        total_feedback_count,
        total_neg_feedback_count,
        total_pos_feedback_count,
        try_to_timestamp(submission_time) as submission_time,
        review_text,
        review_title,
        skin_tone,
        skin_type,
        eye_color,
        hair_color
    from source
    where not regexp_like(author_id, '.*[a-zA-Z].*')
)

select * from cleaned
