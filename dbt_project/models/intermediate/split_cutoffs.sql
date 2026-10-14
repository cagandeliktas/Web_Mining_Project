-- The global temporal split, as one row: everything strictly before val_start is
-- train, [val_start, test_start) is val, the rest is test. Cut-offs snap to a
-- timestamp (never inside one), so same-day reviews always share a split.
-- Mirrors temporal_split() in src/split.py.
--
-- train_mean_rating is the global mean of TRAIN ratings only: the prior that
-- shrinks sparse item means toward the train mean.
--
-- Note: FLOOR(n * fraction) uses the fractions from dbt vars (val_frac, test_frac).
-- Snowflake evaluates decimal literals exactly while Python uses doubles, so on a
-- row count where n * 0.8 is an exact integer the two could differ by one row.
with ordered as (
    select
        submission_time,
        rating,
        row_number() over (order by submission_time, author_id, product_id) - 1 as rn,
        count(*) over () as n
    from {{ ref('int_interactions') }}
),

picks as (
    select
        min(case when rn = cast(floor(n * (1 - {{ var('val_frac') }} - {{ var('test_frac') }})) as bigint)
                 then submission_time end) as val_start,
        min(case when rn = cast(floor(n * (1 - {{ var('test_frac') }})) as bigint)
                 then submission_time end) as test_start
    from ordered
)

select
    p.val_start,
    p.test_start,
    (select avg(o.rating) from ordered as o where o.submission_time < p.val_start) as train_mean_rating
from picks as p
