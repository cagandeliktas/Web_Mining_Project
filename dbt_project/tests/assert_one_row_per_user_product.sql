-- Fails if a (user, product) pair appears more than once (it would straddle splits).
select author_id, product_id, count(*) as n
from {{ ref('fct_interactions_asof') }}
group by author_id, product_id
having count(*) > 1
