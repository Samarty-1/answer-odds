"""The API returns the same features the training table holds for the same question."""
import json

import numpy as np
import pytest


@pytest.fixture()
def client(conn, tmp_path, monkeypatch):
    from odds import models as M
    from odds.dataset import FEATURES, TARGET, fetch

    from odds.db import connect
    with connect(autocommit=False) as c:          # the fetch uses a server-side cursor
        frame, _ = fetch(c, with_docs=False)
    booster = M.fit_m2({"num_leaves": 7, "min_child_samples": 20}, frame[FEATURES], frame[TARGET], rounds=20)
    booster.save_model(str(tmp_path / "m2.txt"))
    (tmp_path / "meta.json").write_text(json.dumps({"features": FEATURES, "trained_on": ["fixture"],
                                                    "test": {"auc": None}}))
    monkeypatch.setenv("ODDS_MODELS", str(tmp_path))
    import importlib
    from odds import api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    return TestClient(api.app), booster, frame


def test_api_matches_the_training_row(client, conn):
    tc, booster, frame = client
    from odds.dataset import FEATURES
    row = frame.iloc[len(frame) // 2]
    title, body, tags, user = conn.execute(
        "SELECT p.title, p.body, q.tags, q.owner_user_id FROM posts p JOIN feat.questions q ON q.id = p.id "
        "WHERE p.id = %s", (int(row["id"]),)).fetchone()
    r = tc.post("/predict", json={"title": title, "body": body, "tags": tags, "user_id": user,
                                  "asked_at": row["asked_at"].isoformat()})
    assert r.status_code == 200, r.text
    got = r.json()
    for c in FEATURES:
        want = row[c]
        have = got["features"][c]
        assert (have is None and np.isnan(want)) or have == pytest.approx(want), c
    expected = booster.predict(frame.loc[[row.name], FEATURES])[0]
    assert got["p_answered_7d"] == pytest.approx(expected, abs=1e-4)


def test_unknown_user_is_a_404_and_bad_input_a_422(client):
    tc, _, _ = client
    assert tc.post("/predict", json={"title": "x", "tags": ["bash"], "user_id": -999}).status_code == 404
    assert tc.post("/predict", json={"title": "", "tags": ["bash"], "user_id": 1}).status_code == 422
    assert tc.post("/predict", json={"title": "x", "tags": [], "user_id": 1}).status_code == 422
    assert tc.get("/health").json()["ok"] is True
