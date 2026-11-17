"""The three shortcuts the protocol says to measure (docs/PROTOCOL.md, "Leakage audit").

Each function returns extra columns, aligned to the given question ids, that a
careless pipeline would have used. None of them is available when a question
is posted.
"""
import pandas as pd

A_COLUMNS = ["leak_reputation", "leak_views", "leak_up_votes", "leak_down_votes"]
B_COLUMNS = ["leak_tag_rate_all", "leak_user_rate_all", "leak_user_questions_all"]
C_COLUMNS = ["leak_view_count", "leak_score", "leak_comment_count", "leak_favorite_count"]


def _fetch(conn, sql, ids, cols):
    df = pd.DataFrame(conn.execute(sql).fetchall(), columns=["id"] + cols).set_index("id")
    return df.reindex(ids).astype("float64").reset_index(drop=True)


def asker_snapshot(conn, ids):
    """A: the asker's profile as it stood when the dump was taken, years later."""
    sql = """SELECT q.id, u.reputation, u.views, u.up_votes, u.down_votes
             FROM feat.questions q JOIN users u ON u.id = q.owner_user_id"""
    return _fetch(conn, sql, ids, A_COLUMNS)


def future_aware_history(conn, ids):
    """B: the same kind of rates, computed over all questions ever, including this one."""
    sql = """
        WITH tag_rate AS (
            SELECT x.tag_name, avg(q.answered_7d::int) AS rate
            FROM feat.questions q CROSS JOIN LATERAL unnest(q.tags) AS x(tag_name)
            GROUP BY x.tag_name
        ), user_rate AS (
            SELECT owner_user_id, avg(answered_7d::int) AS rate, count(*) AS n
            FROM feat.questions WHERE owner_user_id IS NOT NULL GROUP BY owner_user_id
        )
        SELECT q.id,
               (SELECT avg(t.rate) FROM unnest(q.tags) AS x(tag_name) JOIN tag_rate t USING (tag_name)),
               u.rate, u.n
        FROM feat.questions q LEFT JOIN user_rate u ON u.owner_user_id = q.owner_user_id"""
    return _fetch(conn, sql, ids, B_COLUMNS)


def post_snapshot(conn, ids):
    """C: counters on the question that keep growing after it is posted."""
    sql = """SELECT p.id, p.view_count, p.score, p.comment_count, p.favorite_count
             FROM posts p WHERE p.post_type_id = 1"""
    return _fetch(conn, sql, ids, C_COLUMNS)


LEAKS = {
    "A asker snapshot": asker_snapshot,
    "B future-aware history": future_aware_history,
    "C post snapshot": post_snapshot,
}
