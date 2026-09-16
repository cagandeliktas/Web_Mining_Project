with reviews as (
    select * from {{ ref('stg_reviews') }}
),

-- Demographics are self-reported per review, and can vary or go missing
-- across a user's own reviews, so take the most recently submitted
-- non-null answer as their current profile.
latest_profile as (
    select
        author_id,
        skin_tone,
        skin_type,
        eye_color,
        hair_color
    from reviews
    qualify row_number() over (
        partition by author_id
        order by submission_time desc nulls last
    ) = 1
),

behavior as (
    select
        author_id,
        count(*)                                as review_count,
        avg(rating)                              as avg_rating_given,
        avg(iff(is_recommended, 1, 0))           as recommend_ratio,
        avg(helpfulness)                         as avg_helpfulness,
        min(submission_time)                     as first_review_at,
        max(submission_time)                     as last_review_at
    from reviews
    group by author_id
)

select
    b.author_id,
    p.skin_tone,
    p.skin_type,
    p.eye_color,
    p.hair_color,
    b.review_count,
    b.avg_rating_given,
    b.recommend_ratio,
    b.avg_helpfulness,
    b.first_review_at,
    b.last_review_at
from behavior as b
left join latest_profile as p on b.author_id = p.author_id
