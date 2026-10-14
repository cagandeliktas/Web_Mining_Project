-- Fails if a row's "prior" history could include itself or later rows: the prior
-- count of a user (or product) must be strictly smaller than their total count.
with user_totals as (
    select author_id, count(*) as total from {{ ref('int_interactions') }} group by author_id
),
item_totals as (
    select product_id, count(*) as total from {{ ref('int_interactions') }} group by product_id
)
select f.author_id, f.product_id, 'user prior >= total' as problem
from {{ ref('fct_interactions_asof') }} as f
inner join user_totals as u on f.author_id = u.author_id
where f.u_hist_count >= u.total
union all
select f.author_id, f.product_id, 'item prior >= total' as problem
from {{ ref('fct_interactions_asof') }} as f
inner join item_totals as t on f.product_id = t.product_id
where f.i_hist_count >= t.total
