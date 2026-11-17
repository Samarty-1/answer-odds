-- One row per question, with the outcome being predicted.
--
-- answered_7d: an answer that is still in the dump was posted within 7 days of
-- the question. Self-answers count. Answers later deleted are invisible here.

DROP SCHEMA IF EXISTS feat CASCADE;
CREATE SCHEMA feat;

CREATE TABLE feat.questions AS
SELECT q.id,
       q.owner_user_id,
       q.creation_date                        AS asked_at,
       q.creation_date::date                  AS asked_day,
       string_to_array(btrim(q.tags, '|'), '|') AS tags,
       fa.first_answer_at,
       (fa.first_answer_at <= q.creation_date + interval '7 days') IS TRUE AS answered_7d
FROM posts q
LEFT JOIN (
    SELECT parent_id, min(creation_date) AS first_answer_at
    FROM posts
    WHERE post_type_id = 2
    GROUP BY parent_id
) fa ON fa.parent_id = q.id
WHERE q.post_type_id = 1;

ALTER TABLE feat.questions ADD PRIMARY KEY (id);
CREATE INDEX questions_asked_at ON feat.questions (asked_at);
ANALYZE feat.questions;
