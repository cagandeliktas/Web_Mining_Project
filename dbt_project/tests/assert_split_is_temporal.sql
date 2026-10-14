-- Fails (returns rows) if train/val/test overlap in time.
with bounds as (
    select split, min(submission_time) as first_at, max(submission_time) as last_at
    from {{ ref('fct_interactions_asof') }}
    group by split
)
select 'train overlaps val' as problem
from bounds as t inner join bounds as v on t.split = 'train' and v.split = 'val'
where t.last_at >= v.first_at
union all
select 'val overlaps test' as problem
from bounds as v inner join bounds as e on v.split = 'val' and e.split = 'test'
where v.last_at >= e.first_at
