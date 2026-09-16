with source as (
    select * from {{ source('raw', 'reviews_raw') }}
),

-- author_id is occasionally corrupted (letters mixed into what should be a
-- numeric id) - same data-quality rule already validated in the two-tower
-- notebook's df_reviews_final construction and scripts/build_item_matrix.py.
--
-- 387 rows agree on (author_id, product_id, submission_time, review_text)
-- but differ in other fields - not exact duplicates (SELECT DISTINCT
-- doesn't touch them), most likely the same review scraped twice with
-- updated feedback counts. Keep one row per that key, preferring whichever
-- has the most feedback recorded (the more "complete" snapshot).
deduped as (
    select *
    from source
    qualify row_number() over (
        partition by author_id, product_id, submission_time, review_text
        order by total_feedback_count desc nulls last
    ) = 1
),

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
    from deduped
    where not regexp_like(author_id, '.*[a-zA-Z].*')
)

select * from cleaned
