-- Run this in a Snowsight worksheet AFTER `dbt build` to confirm the warehouse matches the Python
-- pipeline (these figures come from the full Sephora data; see tests/test_dbt_asof_sql.py):
--
--   interactions  1,088,824      users 503,195      products 2,351
--   train 870,476 | val 109,438 | test 108,910
--   val_start 2022-02-02 | test_start 2022-08-16 | train_mean_rating 4.2905
--   violations 0 (and the three singular dbt tests pass)
--
-- If a number differs, the raw tables in WEB_MINING.RAW differ from the CSVs used for training.
select
    count(*)                              as interactions,
    count(distinct f.author_id)           as users,
    count(distinct f.product_id)          as products,
    count_if(f.split = 'train')           as n_train,
    count_if(f.split = 'val')             as n_val,
    count_if(f.split = 'test')            as n_test,
    any_value(c.val_start)                as val_start,
    any_value(c.test_start)               as test_start,
    round(any_value(c.train_mean_rating), 4) as train_mean_rating,
    -- a row's prior history can never include itself: prior count < the user's total interactions
    count_if(f.u_hist_count >= t.total)   as violations
from {{ ref('fct_interactions_asof') }} as f
cross join {{ ref('split_cutoffs') }} as c
inner join (
    select author_id, count(*) as total from {{ ref('int_interactions') }} group by author_id
) as t on f.author_id = t.author_id
