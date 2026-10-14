-- One interaction per (author_id, product_id): the user's LATEST review of the
-- product wins. Mirrors clean_reviews() in src/data.py, so the warehouse and the
-- Python pipeline agree on what an interaction is. Averaging repeated reviews
-- would blend a later review into an earlier timestamp, which leaks under a
-- temporal split.
--
-- Written in portable SQL (plain window functions + QUALIFY) so
-- tests/test_dbt_asof_sql.py can execute it in DuckDB against the Python code.
select
    author_id,
    product_id,
    rating,
    is_recommended,
    submission_time,
    skin_tone,
    skin_type,
    eye_color,
    hair_color
from {{ ref('stg_reviews') }}
where submission_time is not null
  and rating between 1 and 5
qualify row_number() over (
    partition by author_id, product_id
    order by submission_time desc, total_feedback_count desc nulls last
) = 1
