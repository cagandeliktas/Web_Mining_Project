-- One row per interaction with point-in-time ("as of") user and item aggregates.
--
-- THE RULE: every aggregate uses only rows with a STRICTLY EARLIER submission_time
-- than the row it describes. The row's own rating, and anything sharing its
-- timestamp, never contribute. That is why dim_users / dim_products (which
-- aggregate over all time) must not be used as ML features: they would hand each
-- row the future, including its own label.
--
-- Technique: aggregate per (key, timestamp), then take a running sum over the
-- PRECEDING timestamps only (frame ends at "1 preceding"), then join back.
-- This matches asof_prior_sums() in src/features/asof.py.
--
-- Text embeddings are computed in Python (TF-IDF + SVD); everything numeric that
-- the user/item towers consume is available here.
with interactions as (
    select * from {{ ref('int_interactions') }}
),

cutoffs as (
    select * from {{ ref('split_cutoffs') }}
),

user_by_time as (
    select
        author_id,
        submission_time,
        count(*)            as n,
        sum(rating)         as rating_sum,
        sum(is_recommended) as rec_sum,
        count(is_recommended) as rec_n
    from interactions
    group by author_id, submission_time
),

user_prior as (
    select
        author_id,
        submission_time,
        coalesce(sum(n)          over (partition by author_id order by submission_time rows between unbounded preceding and 1 preceding), 0) as u_hist_count,
        coalesce(sum(rating_sum) over (partition by author_id order by submission_time rows between unbounded preceding and 1 preceding), 0) as u_rating_sum,
        coalesce(sum(rec_sum)    over (partition by author_id order by submission_time rows between unbounded preceding and 1 preceding), 0) as u_rec_sum,
        coalesce(sum(rec_n)      over (partition by author_id order by submission_time rows between unbounded preceding and 1 preceding), 0) as u_rec_n
    from user_by_time
),

item_by_time as (
    select
        product_id,
        submission_time,
        count(*)    as n,
        sum(rating) as rating_sum
    from interactions
    group by product_id, submission_time
),

item_prior as (
    select
        product_id,
        submission_time,
        coalesce(sum(n)          over (partition by product_id order by submission_time rows between unbounded preceding and 1 preceding), 0) as i_hist_count,
        coalesce(sum(rating_sum) over (partition by product_id order by submission_time rows between unbounded preceding and 1 preceding), 0) as i_rating_sum
    from item_by_time
)

select
    i.author_id,
    i.product_id,
    i.submission_time,
    i.rating,
    case
        when i.submission_time >= c.test_start then 'test'
        when i.submission_time >= c.val_start  then 'val'
        else 'train'
    end as split,

    u.u_hist_count,
    u.u_rating_sum,
    u.u_rec_n,
    1.0 * u.u_rating_sum / nullif(u.u_hist_count, 0) as u_rating_mean,
    1.0 * u.u_rec_sum    / nullif(u.u_rec_n, 0)      as u_rec_mean,

    p.i_hist_count,
    p.i_rating_sum,
    -- shrunk toward the TRAIN mean with item_shrinkage pseudo-ratings
    (p.i_rating_sum + {{ var('item_shrinkage') }} * c.train_mean_rating)
        / (p.i_hist_count + {{ var('item_shrinkage') }}) as i_rating_mean,

    i.skin_tone,
    i.skin_type,
    i.eye_color,
    i.hair_color
from interactions as i
cross join cutoffs as c
inner join user_prior as u on i.author_id = u.author_id and i.submission_time = u.submission_time
inner join item_prior as p on i.product_id = p.product_id and i.submission_time = p.submission_time
