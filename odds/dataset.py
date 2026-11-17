"""Pull the modelling table out of Postgres and split it by time.

    python -m odds.dataset         # builds data/frame.parquet and data/docs.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .db import ROOT, connect
from .textfeat import TEXT_FEATURES, document, text_features

DATA = ROOT / "data"

HISTORY_FEATURES = [
    "hour_of_day", "day_of_week", "n_tags", "n_tags_seen",
    "tag_asked_365_min", "tag_asked_365_max", "tag_rate_365_mean", "tag_rate_365_min", "tag_rate_365_max",
    "site_asked_30", "site_rate_30",
    "account_age_days", "user_questions", "user_settled", "user_answered_rate",
    "user_answers", "user_upvotes", "user_accepts",
]
FEATURES = HISTORY_FEATURES + TEXT_FEATURES
TARGET = "answered_7d"

SCORED_FROM = "2011-01-01"
TRAIN_END = "2021-01-01"      # exclusive
VALID_END = "2022-01-01"      # exclusive; test is everything from here


def fetch(conn, with_docs: bool = True, chunk: int = 20000):
    """Return (frame, docs). Scored rows only: from SCORED_FROM, outcome settled before the dump ends."""
    cols = ", ".join(f"t.{c}" for c in HISTORY_FEATURES)
    sql = f"""
        SELECT t.id, t.asked_at, t.{TARGET}::int AS {TARGET}, {cols}, p.title, p.body
        FROM feat.training t
        JOIN posts p ON p.id = t.id
        WHERE t.asked_at >= %s
          AND t.asked_at < (SELECT max(creation_date) FROM posts) - interval '7 days'
        ORDER BY t.asked_at, t.id"""
    rows, docs = [], []
    with conn.cursor(name="frame") as cur:          # server-side cursor: bodies are large
        cur.itersize = chunk
        cur.execute(sql, (SCORED_FROM,))
        names = [d.name for d in cur.description]
        i_title, i_body = names.index("title"), names.index("body")
        keep = [i for i, n in enumerate(names) if n not in ("title", "body")]
        for r in cur:
            tf = text_features(r[i_title], r[i_body])
            rows.append([r[i] for i in keep] + [tf[k] for k in TEXT_FEATURES])
            if with_docs:
                docs.append(document(r[i_title], r[i_body]))
    frame = pd.DataFrame(rows, columns=[names[i] for i in keep] + TEXT_FEATURES)
    for c in FEATURES:
        frame[c] = pd.to_numeric(frame[c], errors="coerce").astype("float64")
    frame[TARGET] = frame[TARGET].astype("int8")
    frame["asked_at"] = pd.to_datetime(frame["asked_at"])
    return frame, (pd.DataFrame({"id": frame["id"], "doc": docs}) if with_docs else None)


def split(frame: pd.DataFrame):
    """Boolean masks (train, valid, test) by time asked."""
    t = frame["asked_at"]
    train = (t < TRAIN_END).to_numpy()
    valid = ((t >= TRAIN_END) & (t < VALID_END)).to_numpy()
    test = (t >= VALID_END).to_numpy()
    return train, valid, test


def load(with_docs: bool = False):
    frame = pd.read_parquet(DATA / "frame.parquet")
    docs = pd.read_parquet(DATA / "docs.parquet") if with_docs else None
    return frame, docs


def main():
    DATA.mkdir(exist_ok=True)
    with connect(autocommit=False) as conn:
        frame, docs = fetch(conn)
    frame.to_parquet(DATA / "frame.parquet", index=False)
    docs.to_parquet(DATA / "docs.parquet", index=False)
    train, valid, test = split(frame)
    print(f"{len(frame):,} scored questions: train {train.sum():,}, validation {valid.sum():,}, test {test.sum():,}")
    # Deliberately no statistics of the test period here.
    print(f"train base rate {frame.loc[train, TARGET].mean():.4f}, validation {frame.loc[valid, TARGET].mean():.4f}")
    print(f"missing values (train): "
          f"{ {c: round(float(np.mean(frame.loc[train, c].isna())), 3) for c in FEATURES if frame.loc[train, c].isna().any()} }")


if __name__ == "__main__":
    main()
