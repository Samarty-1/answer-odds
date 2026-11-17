-- The feature store: running totals per day, for the site, each tag and each user.
--
-- Rule for every table here: an event filed under day E may be used only for
-- questions asked on a later day (lookups are "latest row with day < asked day").
--
--   a question asked on day S          is filed under S
--   its 7-day outcome                  is filed under S + 7
--       (the outcome is settled 7 x 24h after the question, which is always
--        before the end of day S + 7; filing it earlier would let a model see
--        answers that had not been written yet)
--   an answer, or a vote               is filed under the day it happened
--
-- Each table stores cumulative sums, so "the 365 days before day D" is one
-- subtraction between two lookups, and a lookup is a single index probe.

-- ---------------------------------------------------------------- site
CREATE TABLE feat.site_daily AS
WITH ev AS (
    SELECT asked_day AS day, 1 AS asked, 0 AS settled, 0 AS settled_answered FROM feat.questions
    UNION ALL
    SELECT asked_day + 7, 0, 1, answered_7d::int FROM feat.questions
), d AS (
    SELECT day, sum(asked) AS asked, sum(settled) AS settled, sum(settled_answered) AS settled_answered
    FROM ev GROUP BY day
)
SELECT day,
       sum(asked)            OVER w AS asked_cum,
       sum(settled)          OVER w AS settled_cum,
       sum(settled_answered) OVER w AS settled_answered_cum
FROM d
WINDOW w AS (ORDER BY day);

ALTER TABLE feat.site_daily ADD PRIMARY KEY (day);

-- ---------------------------------------------------------------- tags
CREATE TABLE feat.tag_daily AS
WITH qt AS (
    SELECT t.id AS tag_id, q.asked_day, q.answered_7d
    FROM feat.questions q
    CROSS JOIN LATERAL unnest(q.tags) AS x(tag_name)
    JOIN tags t ON t.tag_name = x.tag_name
), ev AS (
    SELECT tag_id, asked_day AS day, 1 AS asked, 0 AS settled, 0 AS settled_answered FROM qt
    UNION ALL
    SELECT tag_id, asked_day + 7, 0, 1, answered_7d::int FROM qt
), d AS (
    SELECT tag_id, day, sum(asked) AS asked, sum(settled) AS settled, sum(settled_answered) AS settled_answered
    FROM ev GROUP BY tag_id, day
)
SELECT tag_id, day,
       sum(asked)            OVER w AS asked_cum,
       sum(settled)          OVER w AS settled_cum,
       sum(settled_answered) OVER w AS settled_answered_cum
FROM d
WINDOW w AS (PARTITION BY tag_id ORDER BY day);

ALTER TABLE feat.tag_daily ADD PRIMARY KEY (tag_id, day);

-- ---------------------------------------------------------------- users
CREATE TABLE feat.user_daily AS
WITH ev AS (
    -- questions asked, and their outcomes 7 days later
    SELECT owner_user_id AS user_id, asked_day AS day,
           1 AS asked, 0 AS settled, 0 AS settled_answered, 0 AS answers, 0 AS upvotes, 0 AS accepts
    FROM feat.questions WHERE owner_user_id IS NOT NULL
    UNION ALL
    SELECT owner_user_id, asked_day + 7, 0, 1, answered_7d::int, 0, 0, 0
    FROM feat.questions WHERE owner_user_id IS NOT NULL
    UNION ALL
    -- answers the user wrote
    SELECT owner_user_id, creation_date::date, 0, 0, 0, 1, 0, 0
    FROM posts WHERE post_type_id = 2 AND owner_user_id IS NOT NULL
    UNION ALL
    -- up-votes (type 2) and accepts (type 1) received on the user's posts
    SELECT p.owner_user_id, v.creation_date::date, 0, 0, 0, 0,
           (v.vote_type_id = 2)::int, (v.vote_type_id = 1)::int
    FROM votes v JOIN posts p ON p.id = v.post_id
    WHERE v.vote_type_id IN (1, 2) AND p.owner_user_id IS NOT NULL
), d AS (
    SELECT user_id, day, sum(asked) AS asked, sum(settled) AS settled, sum(settled_answered) AS settled_answered,
           sum(answers) AS answers, sum(upvotes) AS upvotes, sum(accepts) AS accepts
    FROM ev GROUP BY user_id, day
)
SELECT user_id, day,
       sum(asked)            OVER w AS asked_cum,
       sum(settled)          OVER w AS settled_cum,
       sum(settled_answered) OVER w AS settled_answered_cum,
       sum(answers)          OVER w AS answers_cum,
       sum(upvotes)          OVER w AS upvotes_cum,
       sum(accepts)          OVER w AS accepts_cum
FROM d
WINDOW w AS (PARTITION BY user_id ORDER BY day);

ALTER TABLE feat.user_daily ADD PRIMARY KEY (user_id, day);

ANALYZE feat.site_daily;
ANALYZE feat.tag_daily;
ANALYZE feat.user_daily;
