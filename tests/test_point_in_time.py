"""The feature function sees nothing from the day a question is asked or later."""
import random

import pytest

FEATS = "SELECT * FROM feat.question_features(%s, %s, %s)"


def _sample(conn, n=25, seed=1):
    rows = conn.execute("""SELECT id, owner_user_id, tags, asked_at FROM feat.questions
                           WHERE owner_user_id IS NOT NULL AND asked_at > '2018-01-01' ORDER BY id""").fetchall()
    return random.Random(seed).sample(rows, n)


def test_training_table_equals_the_function(conn):
    """feat.training is literally the function applied to each question."""
    for qid, user, tags, at in _sample(conn):
        direct = conn.execute(FEATS, (user, tags, at)).fetchone()
        cur = conn.execute("SELECT * FROM feat.training WHERE id = %s", (qid,))
        stored = dict(zip([d.name for d in cur.description], cur.fetchone()))
        names = [d.name for d in conn.execute(FEATS, (user, tags, at)).description]
        assert tuple(stored[n] for n in names) == direct


def test_features_do_not_change_when_the_future_is_deleted(conn):
    """Rebuild the store from data that ends the day before a question; its features must not move."""
    from odds.build import SQL_STEPS
    from odds.db import ROOT

    for _, user, tags, at in _sample(conn, n=3, seed=7):
        before = conn.execute(FEATS, (user, tags, at)).fetchone()
        conn.execute("BEGIN")
        try:
            cutoff = at.date()
            conn.execute("DELETE FROM votes WHERE creation_date >= %s", (cutoff,))
            conn.execute("DELETE FROM comments WHERE creation_date >= %s", (cutoff,))
            conn.execute("UPDATE posts SET accepted_answer_id = NULL WHERE accepted_answer_id IN "
                         "(SELECT id FROM posts WHERE creation_date >= %s)", (cutoff,))
            conn.execute("DELETE FROM posts WHERE creation_date >= %s", (cutoff,))
            # Outcomes of earlier questions that were not settled yet by the cutoff are unknowable.
            for name in SQL_STEPS:
                conn.execute((ROOT / "sql" / name).read_text(encoding="utf-8"))
            after = conn.execute(FEATS, (user, tags, at)).fetchone()
        finally:
            conn.execute("ROLLBACK")
        assert after == before


def test_own_outcome_enters_history_seven_days_later(conn):
    qid, user, tags, at = conn.execute("""
        SELECT id, owner_user_id, tags, asked_at FROM feat.questions
        WHERE owner_user_id IS NOT NULL ORDER BY asked_at DESC OFFSET 200 LIMIT 1""").fetchone()
    from datetime import timedelta

    cols = [d.name for d in conn.execute(FEATS, (user, tags, at)).description]
    get = lambda t: dict(zip(cols, conn.execute(FEATS, (user, tags, t)).fetchone()))
    next_day = get(at + timedelta(days=1))
    later = get(at + timedelta(days=8))
    now = get(at)
    assert next_day["user_questions"] >= now["user_questions"] + 1     # the question itself is counted next day
    assert next_day["user_settled"] == now["user_settled"] or next_day["user_settled"] >= now["user_settled"]
    assert later["user_settled"] >= now["user_questions"] + 1 - 0      # ...and its outcome a week later


def test_unknown_user_and_tags_are_flagged_not_invented(conn):
    cols = [d.name for d in conn.execute(FEATS, (-999, ["no-such-tag"], "2024-01-01")).description]
    f = dict(zip(cols, conn.execute(FEATS, (-999, ["no-such-tag"], "2024-01-01")).fetchone()))
    assert f["user_known"] == 0 and f["user_questions"] == 0
    assert f["n_tags_seen"] == 0 and f["tag_rate_365_mean"] is None


@pytest.mark.parametrize("hours", [0, 23])
def test_same_day_events_are_invisible(conn, hours):
    """A question asked at 00:00 and one at 23:59 on the same day get the same history."""
    _, user, tags, at = _sample(conn, n=1, seed=3)[0]
    from datetime import datetime, timedelta

    day = datetime.combine(at.date(), datetime.min.time())
    cols = [d.name for d in conn.execute(FEATS, (user, tags, day)).description]
    a = dict(zip(cols, conn.execute(FEATS, (user, tags, day + timedelta(hours=hours))).fetchone()))
    b = dict(zip(cols, conn.execute(FEATS, (user, tags, day)).fetchone()))
    skip = {"hour_of_day", "account_age_days"}
    assert {k: v for k, v in a.items() if k not in skip} == {k: v for k, v in b.items() if k not in skip}
