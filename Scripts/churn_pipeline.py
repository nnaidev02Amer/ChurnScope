"""Leak-free, reproducible churn modelling pipeline (Phase 0).

Everything that learns from data (scaler, one-hot encoder, every model, hyper-parameter
search, decision threshold) is fitted on the TRAINING split only. The test split is used
exactly once, for the final selected model.

Run:  python Scripts/train_churn_model.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TEST_SIZE = 0.30  # same 70/30 split ratio as the original notebook
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data.csv"

TARGET = "Churn"
ID_COLUMN = "customerID"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------------------------------- data
def load_and_clean(path=DATA_PATH):
    """Return (X, y). Only deterministic, row-wise cleaning happens here (no statistics
    are learned from the data), so doing it before the split cannot leak anything.

    * customerID is dropped (an identifier, not a feature).
    * TotalCharges is blank (" ") for the 11 customers with tenure == 0, i.e. customers who
      have not been billed yet. As in the original notebook those rows are removed.
    * SeniorCitizen (0/1) is treated as a categorical flag.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=[ID_COLUMN])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df[df["tenure"] > 0].reset_index(drop=True)
    if df[FEATURES + [TARGET]].isnull().any().any():
        raise ValueError("Unexpected missing values after cleaning")
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    y = (df[TARGET] == "Yes").astype(int)
    return df[FEATURES], y


def split_data(X, y, seed=SEED):
    """Stratified train/test split. Must happen before anything is fitted."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)


# ----------------------------------------------------------------- preprocessing
def make_preprocessor():
    """Scale numeric columns, one-hot encode categoricals. Unseen categories at predict
    time are ignored instead of raising. Fitted inside each pipeline, so inside each CV fold."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def make_pipeline(model):
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


# ----------------------------------------------------------------------- models
def base_models(seed=SEED):
    """The original ten models with default-ish settings, every one seeded."""
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    gb = GradientBoostingClassifier(random_state=seed)
    ada = AdaBoostClassifier(random_state=seed)
    return {
        "Logistic Regression": lr,
        "SVC (linear)": SVC(kernel="linear", random_state=seed),
        "Kernel SVM (rbf)": SVC(kernel="rbf", random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Gaussian NB": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, criterion="entropy", random_state=seed, n_jobs=-1
        ),
        "AdaBoost": ada,
        "Gradient Boosting": gb,
        "Voting (GB+LR+Ada)": VotingClassifier(
            [("gbc", gb), ("lr", lr), ("abc", ada)], voting="soft"
        ),
    }


def tuning_grids():
    """Search spaces. Model parameters are chosen by CV on the training split only
    (this replaces the original notebook choosing KNN k and the number of forest trees
    by looking at test-set accuracy)."""
    return {
        "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]},
        "KNN": {"model__n_neighbors": [5, 11, 15, 21, 25, 31]},
        "Random Forest": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 8, 12],
            "model__min_samples_leaf": [1, 5],
        },
        "AdaBoost": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.05, 0.1, 0.3, 1.0],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [1, 2, 3],
            "model__learning_rate": [0.05, 0.1],
        },
    }


SCORING = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


def compare_models(models, X_train, y_train, seed=SEED):
    """Repeated stratified 5-fold CV on the training split. The whole pipeline is
    refitted in every fold, so scaler/encoder statistics never see the validation fold."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    rows = []
    for name, model in models.items():
        res = cross_validate(
            make_pipeline(model), X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1
        )
        row = {"Model": name}
        for key in SCORING:
            row[f"CV {key}"] = res[f"test_{key}"].mean()
        row["CV roc_auc std"] = res["test_roc_auc"].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("CV roc_auc", ascending=False).reset_index(drop=True)


def tune_models(X_train, y_train, seed=SEED):
    """GridSearchCV per model on the training split. Returns {name: fitted GridSearchCV}."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    models = base_models(seed)
    searches = {}
    for name, grid in tuning_grids().items():
        gs = GridSearchCV(
            make_pipeline(models[name]), grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True
        )
        gs.fit(X_train, y_train)
        searches[name] = gs
    return searches


def build_tuned_voting(searches, seed=SEED):
    """Soft-voting ensemble of the tuned Gradient Boosting, Logistic Regression and AdaBoost."""
    def tuned(name):  # unfitted copy carrying the CV-selected hyper-parameters
        return clone(searches[name].best_estimator_.named_steps["model"])

    return VotingClassifier(
        [
            ("gbc", tuned("Gradient Boosting")),
            ("lr", tuned("Logistic Regression")),
            ("abc", tuned("AdaBoost")),
        ],
        voting="soft",
    )


def cv_score(pipeline, X_train, y_train, seed=SEED):
    """Mean CV ROC-AUC on the same folds the grid searches used (training split only)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    res = cross_validate(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(res["test_score"].mean())


# --------------------------------------------------------------------- threshold
def best_f1_threshold(pipeline, X_train, y_train, seed=SEED):
    """Pick the probability threshold that maximises F1 on out-of-fold TRAINING predictions."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import cross_val_predict

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    grid = np.round(np.arange(0.20, 0.71, 0.01), 2)
    scores = [f1_score(y_train, proba >= t) for t in grid]
    return float(grid[int(np.argmax(scores))])


# -------------------------------------------------------------------- evaluation
def evaluate(pipeline, X_test, y_test, threshold=0.5):
    """Metrics of a fitted pipeline on held-out data. Counts are plain ints for JSON."""
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix, f1_score,
        precision_score, recall_score, roc_auc_score,
    )

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


MODEL_PATH = PROJECT_DIR / "models" / "churn_pipeline.joblib"
DEFAULT_THRESHOLD = 0.5  # only used for a bare pipeline that carries no saved threshold


def load_model(path=MODEL_PATH):
    """Load the saved bundle: {"pipeline", "threshold", "features", "sklearn_version"}."""
    import joblib

    return joblib.load(path)


def predict_churn(model, customers, threshold=None):
    """Churn probability and label for one dict or a DataFrame of raw customer rows
    (same columns as the cleaned data; SeniorCitizen as "Yes"/"No").

    `model` is the saved bundle from load_model() (preferred) or a fitted pipeline.
    The decision threshold defaults to the one saved with the model (chosen on training
    data only); a bare pipeline falls back to 0.5. Pass `threshold` to override."""
    if isinstance(model, dict):
        pipeline = model["pipeline"]
        if threshold is None:
            threshold = model["threshold"]
    else:
        pipeline = model
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
    frame = pd.DataFrame([customers]) if isinstance(customers, dict) else customers
    proba = pipeline.predict_proba(frame[FEATURES])[:, 1]
    return pd.DataFrame({"churn_probability": proba, "churn": proba >= threshold})
