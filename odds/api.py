"""Serve the model: POST a draft question, get the chance it is answered within 7 days.

    uvicorn odds.api:app --port 8000

The history features come from the same SQL function that built the training
table (feat.question_features), and the text features from the same Python
function (odds.textfeat.text_features). There is no second implementation.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from functools import lru_cache

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .db import ROOT, connect
from .dataset import HISTORY_FEATURES
from .textfeat import text_features

MODELS = Path(os.environ.get("ODDS_MODELS", ROOT / "models"))


class Draft(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    body: str = Field(default="", max_length=60000, description="HTML, as the site stores it")
    tags: list[str] = Field(min_length=1, max_length=5)
    user_id: int
    asked_at: datetime | None = Field(default=None, description="defaults to the end of the dump")


@lru_cache(maxsize=1)
def model():
    meta = json.loads((MODELS / "meta.json").read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(MODELS / "m2.txt")), meta


def features(conn, d: Draft, at: datetime) -> dict:
    cur = conn.execute("SELECT * FROM feat.question_features(%s, %s, %s)", (d.user_id, d.tags, at))
    row = cur.fetchone()
    hist = dict(zip([c.name for c in cur.description], row))
    if not hist["user_known"]:
        raise HTTPException(404, f"user {d.user_id} is not in the data")
    out = {k: (None if hist[k] is None else float(hist[k])) for k in HISTORY_FEATURES}
    out.update(text_features(d.title, d.body))
    return out


app = FastAPI(title="answer-odds")


@app.get("/health")
def health():
    _, meta = model()
    return {"ok": True, "trained_on": meta["trained_on"], "test_auc": meta["test"]["auc"]}


@app.post("/predict")
def predict(d: Draft):
    booster, meta = model()
    with connect() as conn:
        at = d.asked_at or conn.execute("SELECT max(creation_date) FROM posts").fetchone()[0]
        f = features(conn, d, at)
    x = np.array([[np.nan if f[c] is None else f[c] for c in meta["features"]]], dtype=float)
    p = float(booster.predict(x)[0])
    warnings = []
    if (f["n_tags_seen"] or 0) < len(d.tags):
        warnings.append("some tags are unknown or have no history before this date")
    return {"p_answered_7d": round(p, 4), "as_of": str(at), "features": f, "warnings": warnings}
