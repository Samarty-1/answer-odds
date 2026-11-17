"""Build the feature store and the training table inside Postgres.

    python -m odds.build

Steps: label every question, build the per-day running totals, define the
feature function, then call that function once per question to materialise
feat.training. The API calls the same function at request time.
"""
import sys
import time

from .db import ROOT, connect, run_sql_file, scalar

SQL_STEPS = ["010_labels.sql", "020_daily.sql", "030_features.sql"]

# Questions whose asker is missing are left out: the dump blanks the owner of a
# post when the account is deleted LATER, so "no owner" is knowledge from the
# future (and would also hide the asker's history from the model).
MATERIALISE = """
DROP TABLE IF EXISTS feat.training;
CREATE TABLE feat.training AS
SELECT q.id, q.owner_user_id, q.asked_at, q.tags, q.answered_7d, f.*
FROM feat.questions q
CROSS JOIN LATERAL feat.question_features(q.owner_user_id, q.tags, q.asked_at) f
WHERE q.owner_user_id IS NOT NULL;
ALTER TABLE feat.training ADD PRIMARY KEY (id);
ANALYZE feat.training;
"""


def build(verbose: bool = True) -> dict:
    with connect() as conn:
        for name in SQL_STEPS:
            t0 = time.time()
            run_sql_file(conn, ROOT / "sql" / name)
            if verbose:
                print(f"{name:20s} {time.time() - t0:6.1f}s", flush=True)
        t0 = time.time()
        conn.execute(MATERIALISE)
        if verbose:
            print(f"{'feat.training':20s} {time.time() - t0:6.1f}s", flush=True)
        counts = {t: scalar(conn, f"SELECT count(*) FROM feat.{t}")
                  for t in ["questions", "site_daily", "tag_daily", "user_daily", "training"]}
    if verbose:
        for t, n in counts.items():
            print(f"feat.{t:12s} {n:>10,} rows")
    return counts


if __name__ == "__main__":
    build()
    sys.exit(0)
