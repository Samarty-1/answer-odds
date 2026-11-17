-- The one definition of the history features.
--
-- Training calls this function once per historical question; the API calls it
-- once per request. There is no second implementation to drift from the first.
--
-- Everything is "as of the start of the day the question is asked": each lookup
-- takes the latest stored row with day < p_at::date.

CREATE OR REPLACE FUNCTION feat.question_features(p_user integer, p_tags text[], p_at timestamp)
RETURNS TABLE (
    hour_of_day           integer,
    day_of_week           integer,
    n_tags                integer,
    n_tags_seen           integer,   -- tags that had been used before this day
    tag_asked_365_min     numeric,   -- questions in the previous 365 days, rarest tag
    tag_asked_365_max     numeric,   --   ... most common tag
    tag_rate_365_mean     numeric,   -- share answered within 7 days, previous 365 days, mean over tags
    tag_rate_365_min      numeric,
    tag_rate_365_max      numeric,
    site_asked_30         numeric,   -- questions on the whole site, previous 30 days
    site_rate_30          numeric,   -- share answered within 7 days, outcomes settled in the previous 30 days
    user_known            integer,   -- 0 when the asker's account is not in the data
    account_age_days      numeric,
    user_questions        numeric,   -- questions asked before this day
    user_settled          numeric,   -- ... of which the 7-day outcome is already known
    user_answered_rate    numeric,   -- share of those that were answered
    user_answers          numeric,   -- answers written before this day
    user_upvotes          numeric,   -- up-votes received before this day
    user_accepts          numeric    -- accepted answers received before this day
)
LANGUAGE sql STABLE
AS $$
WITH tag_stats AS (
    SELECT t.id,
           now_.asked_cum - coalesce(old_.asked_cum, 0)                       AS asked_365,
           now_.settled_cum - coalesce(old_.settled_cum, 0)                   AS settled_365,
           now_.settled_answered_cum - coalesce(old_.settled_answered_cum, 0) AS answered_365
    FROM unnest(p_tags) AS x(tag_name)
    JOIN tags t ON t.tag_name = x.tag_name
    JOIN LATERAL (
        SELECT d.asked_cum, d.settled_cum, d.settled_answered_cum
        FROM feat.tag_daily d
        WHERE d.tag_id = t.id AND d.day < p_at::date
        ORDER BY d.day DESC LIMIT 1
    ) now_ ON true
    LEFT JOIN LATERAL (
        SELECT d.asked_cum, d.settled_cum, d.settled_answered_cum
        FROM feat.tag_daily d
        WHERE d.tag_id = t.id AND d.day < p_at::date - 365
        ORDER BY d.day DESC LIMIT 1
    ) old_ ON true
), tag_agg AS (
    SELECT count(*)                                             AS n_seen,
           min(asked_365)                                       AS asked_min,
           max(asked_365)                                       AS asked_max,
           avg(answered_365::numeric / nullif(settled_365, 0))  AS rate_mean,
           min(answered_365::numeric / nullif(settled_365, 0))  AS rate_min,
           max(answered_365::numeric / nullif(settled_365, 0))  AS rate_max
    FROM tag_stats
), site AS (
    SELECT now_.asked_cum - coalesce(old_.asked_cum, 0) AS asked_30,
           (now_.settled_answered_cum - coalesce(old_.settled_answered_cum, 0))::numeric
               / nullif(now_.settled_cum - coalesce(old_.settled_cum, 0), 0) AS rate_30
    FROM (SELECT 1) one
    LEFT JOIN LATERAL (
        SELECT d.asked_cum, d.settled_cum, d.settled_answered_cum
        FROM feat.site_daily d WHERE d.day < p_at::date ORDER BY d.day DESC LIMIT 1
    ) now_ ON true
    LEFT JOIN LATERAL (
        SELECT d.asked_cum, d.settled_cum, d.settled_answered_cum
        FROM feat.site_daily d WHERE d.day < p_at::date - 30 ORDER BY d.day DESC LIMIT 1
    ) old_ ON true
), usr AS (
    SELECT u.id, u.creation_date, h.*
    FROM users u
    LEFT JOIN LATERAL (
        SELECT d.asked_cum, d.settled_cum, d.settled_answered_cum, d.answers_cum, d.upvotes_cum, d.accepts_cum
        FROM feat.user_daily d
        WHERE d.user_id = u.id AND d.day < p_at::date
        ORDER BY d.day DESC LIMIT 1
    ) h ON true
    WHERE u.id = p_user
)
SELECT extract(hour FROM p_at)::int,
       extract(isodow FROM p_at)::int,
       coalesce(cardinality(p_tags), 0),
       tag_agg.n_seen::int,
       tag_agg.asked_min, tag_agg.asked_max,
       tag_agg.rate_mean, tag_agg.rate_min, tag_agg.rate_max,
       coalesce(site.asked_30, 0), site.rate_30,
       (usr.id IS NOT NULL)::int,
       -- An account can be younger than its first post by a few seconds of clock skew; never negative.
       greatest(extract(epoch FROM p_at - usr.creation_date) / 86400.0, 0)::numeric,
       coalesce(usr.asked_cum, 0), coalesce(usr.settled_cum, 0),
       usr.settled_answered_cum::numeric / nullif(usr.settled_cum, 0),
       coalesce(usr.answers_cum, 0), coalesce(usr.upvotes_cum, 0), coalesce(usr.accepts_cum, 0)
FROM tag_agg
CROSS JOIN site
LEFT JOIN usr ON true
$$;
