"""Stage 2: refit the chosen settings on train + validation and score the test period once.

    python -m odds.final          # writes results/test.json, results/RESULTS.md, models/

Refuses to run if results/test.json already exists (pass --again to override,
which is recorded in the output).
"""
import argparse
import json
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import models as M
from .dataset import FEATURES, TARGET, load, split
from .db import ROOT, connect
from .leaky import LEAKS

RESULTS = ROOT / "results"
MODELS = ROOT / "models"
N_BOOT = 1000


def bootstrap(y, preds: dict, pairs, seed=M.SEED):
    """95% intervals for each model's AUC/Brier and for AUC differences, resampling test questions."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(y)
    stats = {k: {"auc": [], "brier": []} for k in preds}
    diffs = {f"{a} - {b}": [] for a, b in pairs}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        aucs = {}
        for k, p in preds.items():
            pb = p[idx]
            aucs[k] = roc_auc_score(yb, pb)
            stats[k]["auc"].append(aucs[k])
            stats[k]["brier"].append(np.mean((pb - yb) ** 2))
        for a, b in pairs:
            diffs[f"{a} - {b}"].append(aucs[a] - aucs[b])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return ({k: {m: q(v) for m, v in s.items()} for k, s in stats.items()},
            {k: {"mean": float(np.mean(v)), "ci95": q(v)} for k, v in diffs.items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--again", action="store_true")
    args = ap.parse_args(argv)
    if (RESULTS / "test.json").exists() and not args.again:
        sys.exit("results/test.json exists: the test period has already been scored. Use --again to rerun.")

    chosen = json.loads((RESULTS / "validation.json").read_text(encoding="utf-8"))["models"]
    frame, docs = load(with_docs=True)
    train, valid, test = split(frame)
    fit = train | valid
    X_fit, y_fit = frame.loc[fit, FEATURES], frame.loc[fit, TARGET].to_numpy()
    X_te, y_te = frame.loc[test, FEATURES], frame.loc[test, TARGET].to_numpy()

    preds = {}
    base = float(y_fit.mean())
    preds["B0 constant"] = np.full(len(y_te), base)
    preds["B1 site rate"] = X_te["site_rate_30"].fillna(base).to_numpy()
    preds["B2 tag rate"] = X_te["tag_rate_365_mean"].fillna(X_te["site_rate_30"]).fillna(base).to_numpy()

    m1 = M.make_m1(**chosen["M1 logistic"]["chosen"]["config"]).fit(X_fit, y_fit)
    preds["M1 logistic"] = m1.predict_proba(X_te)[:, 1]

    m2_cfg = chosen["M2 LightGBM"]["chosen"]["config"]
    # Validation chose the number of trees on 10 years of data; refit on 11 with ~10% more.
    m2_rounds = int(round(chosen["M2 LightGBM"]["chosen"]["rounds"] * fit.sum() / train.sum()))
    m2 = M.fit_m2(m2_cfg, X_fit, y_fit, rounds=m2_rounds)
    preds["M2 LightGBM"] = m2.predict(X_te)

    m3 = M.make_m3(**chosen["M3 text only"]["chosen"]["config"]).fit(docs.loc[fit, "doc"], y_fit)
    preds["M3 text only"] = m3.predict_proba(docs.loc[test, "doc"])[:, 1]

    # Leakage audit: M2, same settings and rounds, one shortcut added at a time.
    leaks = {}
    with connect() as conn:
        ids = frame["id"].tolist()
        for name, fn in LEAKS.items():
            extra = fn(conn, ids)
            cols = FEATURES + list(extra.columns)
            X = pd.concat([frame[FEATURES].reset_index(drop=True), extra], axis=1)[cols]
            booster = M.fit_m2(m2_cfg, X.loc[fit], y_fit, rounds=m2_rounds)
            leaks[name] = M.metrics(y_te, booster.predict(X.loc[test]))

    results = {name: M.metrics(y_te, p) for name, p in preds.items()}
    intervals, diffs = bootstrap(y_te, {k: M.clip(v) for k, v in preds.items()},
                                 [("M2 LightGBM", "B2 tag rate"), ("M2 LightGBM", "M1 logistic"),
                                  ("M2 LightGBM", "M3 text only")])
    calibration = {k: M.calibration_table(y_te, preds[k]) for k in ("B1 site rate", "M2 LightGBM")}
    importance = sorted(zip(FEATURES, m2.feature_importance("gain").tolist()), key=lambda t: -t[1])

    d = diffs["M2 LightGBM - B2 tag rate"]
    claims = {
        "1 M2 is useful": {
            "rule": "test AUC >= 0.65 and beats B2 by >= 0.02 AUC with the 95% interval of the difference above 0",
            "m2_auc": results["M2 LightGBM"]["auc"], "difference": d["mean"], "difference_ci95": d["ci95"],
            "holds": bool(results["M2 LightGBM"]["auc"] >= 0.65 and d["mean"] >= 0.02 and d["ci95"][0] > 0),
        },
        "2 calibration holds": {
            "rule": "M2 test Brier score lower than B1 (rolling site rate)",
            "m2_brier": results["M2 LightGBM"]["brier"], "b1_brier": results["B1 site rate"]["brier"],
            "holds": bool(results["M2 LightGBM"]["brier"] < results["B1 site rate"]["brier"]),
        },
    }

    out = {
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rerun": bool(args.again),
        "rows": {"fit": int(fit.sum()), "test": int(test.sum())},
        "test_period": [str(frame.loc[test, "asked_at"].min()), str(frame.loc[test, "asked_at"].max())],
        "m2": {"config": m2_cfg, "rounds": m2_rounds},
        "models": results, "intervals": intervals, "differences": diffs,
        "claims": claims, "leakage": leaks, "calibration": calibration,
        "importance_gain": importance,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "test.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    MODELS.mkdir(exist_ok=True)
    m2.save_model(str(MODELS / "m2.txt"))
    with open(MODELS / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"features": FEATURES, "config": m2_cfg, "rounds": m2_rounds,
                   "trained_on": [str(frame.loc[fit, "asked_at"].min()), str(frame.loc[fit, "asked_at"].max())],
                   "test": results["M2 LightGBM"]}, f, indent=2)
    with open(MODELS / "m1.pkl", "wb") as f:
        pickle.dump(m1, f)

    for name, r in results.items():
        lo, hi = intervals[name]["auc"]
        print(f"{name:14s} auc {r['auc']:.4f} [{lo:.4f}, {hi:.4f}]  brier {r['brier']:.4f}  "
              f"mean p {r['mean_predicted']:.3f} vs observed {r['observed_rate']:.3f}")
    for name, r in leaks.items():
        print(f"leak {name:24s} auc {r['auc']:.4f}  brier {r['brier']:.4f}")
    for name, c in claims.items():
        print(f"claim {name}: {'HOLDS' if c['holds'] else 'FAILS'}")


if __name__ == "__main__":
    main()
