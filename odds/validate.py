"""Stage 1: fit on train, compare on validation, choose settings. Never reads the test rows.

    python -m odds.validate        # writes results/validation.json
"""
import json
import time

import numpy as np

from . import models as M
from .dataset import FEATURES, TARGET, load, split
from .db import ROOT

RESULTS = ROOT / "results"


def main():
    frame, docs = load(with_docs=True)
    train, valid, _ = split(frame)                  # the test mask is discarded on purpose
    X_tr, y_tr = frame.loc[train, FEATURES], frame.loc[train, TARGET].to_numpy()
    X_va, y_va = frame.loc[valid, FEATURES], frame.loc[valid, TARGET].to_numpy()
    d_tr, d_va = docs.loc[train, "doc"], docs.loc[valid, "doc"]
    out = {"rows": {"train": int(train.sum()), "validation": int(valid.sum())}, "models": {}}

    base = float(y_tr.mean())
    out["models"]["B0 constant"] = M.metrics(y_va, np.full(len(y_va), base))
    site = X_va["site_rate_30"].fillna(base).to_numpy()
    out["models"]["B1 site rate"] = M.metrics(y_va, site)
    out["models"]["B2 tag rate"] = M.metrics(y_va, X_va["tag_rate_365_mean"].fillna(X_va["site_rate_30"]).fillna(base))

    def run_grid(name, grid, fit_predict):
        tried = []
        for cfg in grid:
            t0 = time.time()
            p, extra = fit_predict(cfg)
            tried.append({"config": cfg, **extra, **M.metrics(y_va, p), "seconds": round(time.time() - t0, 1)})
            print(f"{name} {cfg} auc {tried[-1]['auc']:.4f} brier {tried[-1]['brier']:.4f}", flush=True)
        best = max(tried, key=lambda r: r["auc"])          # chosen on validation AUC, the primary metric
        out["models"][name] = {"chosen": best, "tried": tried}

    def m1(cfg):
        model = M.make_m1(**cfg).fit(X_tr, y_tr)
        return model.predict_proba(X_va)[:, 1], {}

    def m2(cfg):
        booster = M.fit_m2(cfg, X_tr, y_tr, X_va, y_va)
        return booster.predict(X_va, num_iteration=booster.best_iteration), {"rounds": int(booster.best_iteration)}

    def m3(cfg):
        model = M.make_m3(**cfg).fit(d_tr, y_tr)
        return model.predict_proba(d_va)[:, 1], {}

    run_grid("M1 logistic", M.M1_GRID, m1)
    run_grid("M2 LightGBM", M.M2_GRID, m2)
    run_grid("M3 text only", M.M3_GRID, m3)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nvalidation:")
    for name, r in out["models"].items():
        r = r.get("chosen", r)
        print(f"  {name:14s} auc {r['auc']:.4f}  brier {r['brier']:.4f}  log loss {r['log_loss']:.4f}")


if __name__ == "__main__":
    main()
