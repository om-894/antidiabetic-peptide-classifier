
"""
Phase 3.4: Stack the four base learners with a logistic-regression meta-learner
(Aim 1, ensemble against its best member, ESM-2/DoRA at test AUC 0.877).

Trained on out-of-fold probabilities rather than in-sample fits, so the four
columns are honest. stack_oof comes from cross_val_predict over the same
StratifiedKFold(5, shuffle=True, random_state=42) the base learners used.
stack_test comes from an LR fit on all OOF rows, applied to the base test probabilities.

  primary -> LR on the four raw OOF probabilities
  std -> standardise inside a Pipeline, refit per fold so nothing leaks
  logit -> log-odds transform first
  no-cnn -> drop the weakest base learner

INPUTS  base_tree_predictions.npz (rf, xgb), base_cnn_predictions.npz (cnn),
        esm2_dora_predictions.npz (esm), all row-aligned OOF plus test
OUTPUTS  stack_predictions.npz, stack_meta.joblib, stack_sensitivity.csv
ENV VARS  TREE_NPZ, CNN_NPZ override the inputs. STACK_TAG suffixes the outputs
REQUIREMENTS  pip install scikit-learn numpy joblib
"""

# Imports
import csv
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

SEED = 42
FOLDS = 5
BASE_NAMES = ["rf", "xgb", "cnn", "esm"] # column order of the meta-feature matrix

# inputs (override via env to stack the Optuna-tuned base learners). esm has no
# tuned variant, so its path stays fixed inside load_bases()
TREE_NPZ = os.environ.get("TREE_NPZ", "predictions/base_tree_predictions.npz")
CNN_NPZ  = os.environ.get("CNN_NPZ", "predictions/base_cnn_predictions.npz")

TAG = os.environ.get("STACK_TAG", "") # e.g. "_tuned" for separate outputs
OUT_NPZ = f"predictions/stack_predictions{TAG}.npz"
OUT_META = f"models/stack_meta{TAG}.joblib"
OUT_CSV = f"predictions/stack_sensitivity{TAG}.csv"


def load_bases():
    # three files for four learners, since rf and xgb share one. allow_pickle is
    # only needed for esm, whose npz carries an object array of sequences
    trees = np.load(TREE_NPZ)
    cnn = np.load(CNN_NPZ)
    esm = np.load("predictions/esm2_dora_predictions.npz", allow_pickle=True)

    # take the trees labels as the reference and check the other two match.
    # matching label vectors catch gross misalignment but don't prove row order,
    # since any label-preserving permutation would pass
    ytr = trees["y_train"].astype(int)
    yte = trees["y_test"].astype(int)
    for d in (cnn, esm):
        assert np.array_equal(d["y_train"].astype(int), ytr), "y_train misaligned"
        assert np.array_equal(d["y_test"].astype(int), yte), "y_test misaligned"

    # each learner's OOF (train) and test probabilities, keyed by name
    oof = {"rf": trees["rf_oof"], "xgb": trees["xgb_oof"],
           "cnn": cnn["cnn_oof"], "esm": esm["esm_oof"]}
    test = {"rf": trees["rf_test"], "xgb": trees["xgb_test"],
            "cnn": cnn["cnn_test"], "esm": esm["esm_test"]}
    return oof, test, ytr, yte


def logit(P, eps=1e-6):
    # log-odds transform, log(p/(1-p)). clip away from 0 and 1 first to avoid inf
    P = np.clip(P.astype(np.float64), eps, 1 - eps)
    return np.log(P / (1 - P))


def metrics(y, p, thr=0.5):
    """point estimates on one set of predictions. bootstrap CIs come later in phase 4.1."""
    # AUC scores the probabilities, everything else the thresholded predictions
    # sklearn has no specificity scorer, so unpack the confusion matrix for SP
    ypred = (p >= thr).astype(int)
    tn, fp, _, _ = confusion_matrix(y, ypred, labels=[0, 1]).ravel()
    return {"AUC": roc_auc_score(y, p), "ACC": accuracy_score(y, ypred),
            "MCC": matthews_corrcoef(y, ypred), "F1": f1_score(y, ypred),
            "SN": recall_score(y, ypred), "SP": tn / (tn + fp),
            "PREC": precision_score(y, ypred)}


def make_estimator(standardize):
    # logistic regression, optionally behind a StandardScaler. wrapping in a
    # Pipeline is what makes the scaler refit per CV fold rather than leak
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    if standardize:
        return Pipeline([("sc", StandardScaler()), ("lr", lr)])
    return lr


def run_variant(cols, transform, standardize, oof, test, ytr, yte):
    # meta-features from the chosen base learners, train from their OOF
    # probabilities and test from their test probabilities
    Xtr = np.column_stack([oof[c] for c in cols]).astype(np.float64)
    Xte = np.column_stack([test[c] for c in cols]).astype(np.float64)

    # logit transform the probabilities to log-odds before fitting the meta-learner
    if transform == "logit":
        Xtr, Xte = logit(Xtr), logit(Xte)

    est = make_estimator(standardize)

    # honest train-set estimate on the same 5-fold scheme as the base learners
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof_p = cross_val_predict(est, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]

    # cross_val_predict leaves est unfitted, so fit on all OOF rows for the test pass
    est.fit(Xtr, ytr)
    test_p = est.predict_proba(Xte)[:, 1]
    return est, oof_p, test_p


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    oof, test, ytr, yte = load_bases() # row-aligned base-learner predictions

    # the best single base learner on test is the benchmark the stack must beat (Aim 1)
    best_auc = max(roc_auc_score(yte, test[n]) for n in BASE_NAMES)

    # (label, base learners, transform, standardise). raw.all4 is canonical since
    # it applies neither transform. The other three are robustness checks.
    variants = [
        ("raw.all4", BASE_NAMES, "raw", False),
        ("std.all4", BASE_NAMES, "raw", True),
        ("logit.all4", BASE_NAMES, "logit", True),
        ("logit.no-cnn", ["rf", "xgb", "esm"], "logit", True),
    ]

    rows, primary = [], None
    for label, cols, transform, standardise in variants:
        est, oof_p, test_p = run_variant(cols, transform, standardise,
                                         oof, test, ytr, yte)
        mt = metrics(yte, test_p)
        rows.append((label, len(cols), roc_auc_score(ytr, oof_p), mt,
                     mt["AUC"] - best_auc)) # last item is delta vs the best single learner
        if label.startswith("raw"):
            primary = (oof_p, test_p, est) # canonical stack, this is what gets saved

    # the primary stack's OOF and test probabilities, the labels and the raw
    # meta-feature matrices for reference.
    oof_p, test_p, est = primary
    np.savez_compressed(
        OUT_NPZ,
        stack_oof=oof_p.astype(np.float32),
        stack_test=test_p.astype(np.float32),
        y_train=ytr, y_test=yte,
        base_names=np.array(BASE_NAMES, dtype=object),
        Xtr_meta=np.column_stack([oof[n] for n in BASE_NAMES]).astype(np.float32),
        Xte_meta=np.column_stack([test[n] for n in BASE_NAMES]).astype(np.float32),
    )
    joblib.dump(est, OUT_META) # the fitted primary meta-learner

    # one row per variant, the sensitivity table for the write-up
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "n_models", "oof_auc", "test_auc", "test_acc",
                    "test_mcc", "test_f1", "delta_vs_best_single"])
        for label, k, oof_auc, mt, d in rows: # k is the number of base learners
            w.writerow([label, k, f"{oof_auc:.4f}",
                        f"{mt['AUC']:.4f}", f"{mt['ACC']:.4f}",
                        f"{mt['MCC']:.4f}", f"{mt['F1']:.4f}", f"{d:+.4f}"])


if __name__ == "__main__":
    main()
