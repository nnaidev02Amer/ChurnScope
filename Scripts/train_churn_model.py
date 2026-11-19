"""Train, select and evaluate the churn model end to end.

    python Scripts/train_churn_model.py

Order of operations (nothing touches the test split until step 5):
 1. load + deterministic cleaning
 2. stratified 70/30 split (seeded)
 3. compare the 10 baseline models by repeated stratified CV on the training split
 4. tune KNN / RandomForest / LogisticRegression / AdaBoost / GradientBoosting by CV on
    train only, build the tuned soft-voting ensemble, pick the best by CV ROC-AUC
 5. fit the winner on the full training split, choose an F1 threshold from out-of-fold
    training predictions, then evaluate ONCE on the untouched test split
All numbers written to reports/ come from this run.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parent))
import churn_pipeline as cp  # noqa: E402

REPORT_DIR = cp.PROJECT_DIR / "reports"
MODEL_DIR = cp.PROJECT_DIR / "models"


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    X, y = cp.load_and_clean()
    X_train, X_test, y_train, y_test = cp.split_data(X, y)
    print(f"rows={len(X)}  train={len(X_train)}  test={len(X_test)}  "
          f"churn rate train={y_train.mean():.3f} test={y_test.mean():.3f}")

    print("\n[3] Baseline comparison (repeated 5-fold CV, training split only)")
    comparison = cp.compare_models(cp.base_models(), X_train, y_train)
    print(comparison.round(4).to_string(index=False))

    print("\n[4] Hyper-parameter search (5-fold CV, training split only)")
    searches = cp.tune_models(X_train, y_train)
    candidates = {}
    for name, gs in searches.items():
        print(f"  {name:20s} best CV AUC={gs.best_score_:.4f}  {gs.best_params_}")
        candidates[name] = (gs.best_estimator_, float(gs.best_score_))
    voting = cp.make_pipeline(cp.build_tuned_voting(searches))
    candidates["Voting (tuned GB+LR+Ada)"] = (voting, cp.cv_score(voting, X_train, y_train))
    print(f"  {'Voting (tuned)':20s} CV AUC={candidates['Voting (tuned GB+LR+Ada)'][1]:.4f}")

    selected = max(candidates, key=lambda n: candidates[n][1])
    print(f"\nSelected by CV ROC-AUC (no test data used): {selected}")
    final = clone(candidates[selected][0])
    final.fit(X_train, y_train)

    threshold = cp.best_f1_threshold(final, X_train, y_train)
    print(f"F1-optimal threshold from out-of-fold training predictions: {threshold}")

    print("\n[5] Final evaluation on the untouched test split")
    at_default = cp.evaluate(final, X_test, y_test, 0.5)
    at_tuned = cp.evaluate(final, X_test, y_test, threshold)
    majority_acc = float(max(y_test.mean(), 1 - y_test.mean()))
    for label, m in (("threshold 0.50", at_default), (f"threshold {threshold}", at_tuned)):
        cm = m["confusion_matrix"]
        print(f"  {label}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} "
              f"f1={m['f1']:.4f} auc={m['roc_auc']:.4f}  TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print(f"  majority-class baseline accuracy: {majority_acc:.4f}")

    comparison.to_csv(REPORT_DIR / "model_comparison_cv.csv", index=False)
    metrics = {
        "seed": cp.SEED,
        "sklearn_version": sklearn.__version__,
        "n_rows_after_cleaning": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "majority_baseline_accuracy_test": majority_acc,
        "selected_model": selected,
        "selection_cv_roc_auc": {k: round(v[1], 4) for k, v in candidates.items()},
        "tuned_params_cv_selected": {n: {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
                                     for n, gs in searches.items()},
        "test_default_threshold": at_default,
        "test_tuned_threshold": at_tuned,
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = at_default["confusion_matrix"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, m, title in ((axes[0], at_default, "threshold 0.50"), (axes[1], at_tuned, f"threshold {threshold}")):
        c = m["confusion_matrix"]
        arr = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(arr, cmap="Blues")
        ax.set_xticks([0, 1], ["No churn", "Churn"]); ax.set_yticks([0, 1], ["No churn", "Churn"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{selected}\n{title}", fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, arr[i, j], ha="center", va="center",
                        color="white" if arr[i, j] > arr.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix_test.png", dpi=120)

    joblib.dump({"pipeline": final, "threshold": threshold, "features": cp.FEATURES,
                 "sklearn_version": sklearn.__version__}, MODEL_DIR / "churn_pipeline.joblib")
    print(f"\nSaved: {REPORT_DIR / 'metrics.json'}, model_comparison_cv.csv, "
          f"confusion_matrix_test.png, {MODEL_DIR / 'churn_pipeline.joblib'}")


if __name__ == "__main__":
    main()
