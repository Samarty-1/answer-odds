"""Model definitions and metrics shared by the validation run, the final run and the API."""
import lightgbm as lgb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SEED = 20240331

# At most 12 configurations per model (docs/PROTOCOL.md).
M1_GRID = [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]
M2_GRID = [
    {"num_leaves": nl, "min_child_samples": mcs, "reg_lambda": lam}
    for nl in (15, 63) for mcs in (50, 200) for lam in (0.0, 10.0)
]
M3_GRID = [{"alpha": a} for a in (1e-6, 1e-5, 1e-4)]

M2_FIXED = {"objective": "binary", "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 1, "verbose": -1, "seed": SEED, "num_threads": 6, "deterministic": True,
            "force_row_wise": True}
M2_MAX_ROUNDS = 3000
M2_PATIENCE = 100


def clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def metrics(y, p) -> dict:
    p = clip(p)
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p)), "mean_predicted": float(p.mean()), "observed_rate": float(np.mean(y))}


def calibration_table(y, p, bins: int = 10):
    """Predicted vs observed rate in equal-count bins, lowest predictions first."""
    y, p = np.asarray(y), clip(p)
    order = np.argsort(p, kind="stable")
    out = []
    for chunk in np.array_split(order, bins):
        out.append({"n": int(len(chunk)), "predicted": float(p[chunk].mean()), "observed": float(y[chunk].mean())})
    return out


def _signed_log1p(X):
    return np.sign(X) * np.log1p(np.abs(X))


def make_m1(C: float):
    """Logistic regression: impute (with missing flags), compress heavy tails, standardise."""
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        FunctionTransformer(_signed_log1p),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000),
    )


def fit_m2(params: dict, X, y, X_valid=None, y_valid=None, rounds: int | None = None):
    """LightGBM. With a validation set: early stopping. Without: exactly `rounds` trees."""
    full = {**M2_FIXED, **params}
    train = lgb.Dataset(X, label=y, free_raw_data=False)
    if X_valid is not None:
        valid = lgb.Dataset(X_valid, label=y_valid, reference=train)
        return lgb.train(full, train, num_boost_round=M2_MAX_ROUNDS, valid_sets=[valid],
                         callbacks=[lgb.early_stopping(M2_PATIENCE, verbose=False)])
    return lgb.train(full, train, num_boost_round=rounds)


def make_m3(alpha: float):
    """Text only: TF-IDF of title + body, logistic loss fitted by SGD."""
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=300_000, sublinear_tf=True, dtype=np.float32),
        SGDClassifier(loss="log_loss", alpha=alpha, max_iter=30, tol=1e-4, random_state=SEED),
    )
