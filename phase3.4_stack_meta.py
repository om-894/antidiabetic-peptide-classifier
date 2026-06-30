
"""
Phase 3.4: stack the four base learners with a logistic-regression
meta-learner and test Aim 3 — whether ensembling beats the best single model
(ESM-2/DoRA, test AUC 0.877).

Inputs (each row-aligned OOF + test probabilities):
  rf, xgb <- base_tree_predictions.npz ; cnn <- base_cnn_predictions.npz ;
  esm     <- esm2_dora_predictions.npz

Primary (per the outline): logistic regression on the four raw OOF probabilities.
Sensitivity variants confirm the conclusion isn't config-specific:
  std    standardise the meta-features (comparable coefficients)
  logit  log-odds transform before the LR
  no-cnn drop the weak CNN

Method: train on the out-of-fold probabilities (honest, not in-sample).
stack_oof = cross_val_predict over the SAME StratifiedKFold(5, seed=42);
stack_test = LR fit on all OOF rows, applied to the base test probabilities.
Standardisation sits inside a Pipeline so it re-fits per fold (no leakage).

OUTPUTS  stack_predictions.npz, stack_meta.joblib, stack_sensitivity.csv
REQUIREMENTS  pip install scikit-learn numpy joblib
"""

# Imports
import csv
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED, FOLDS = 42, 5
BASE_NAMES = ["rf", "xgb", "cnn", "esm"]

# inputs (override via env to stack the Optuna-tuned base learners)
TREE_NPZ = os.environ.get("TREE_NPZ", "predictions/base_tree_predictions.npz")
CNN_NPZ  = os.environ.get("CNN_NPZ",  "predictions/base_cnn_predictions.npz")
TAG = os.environ.get("STACK_TAG", "")          # e.g. "_tuned" -> separate outputs
OUT_NPZ  = f"predictions/stack_predictions{TAG}.npz"     # -> predictions/
OUT_META = f"models/stack_meta{TAG}.joblib"              # -> models/
OUT_CSV  = f"predictions/stack_sensitivity{TAG}.csv"     # -> predictions/


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

def load_bases():
    # Load the three base-learner prediction files.
    trees = np.load(TREE_NPZ)
    cnn = np.load(CNN_NPZ)
    esm = np.load("predictions/esm2_dora_predictions.npz", allow_pickle=True)

    # Use the trees' labels as the reference, then assert the other two match.
    # Identical y_train/y_test proves every OOF/test column refers to the same rows
    # in the same order -> the precondition for stacking them column-wise.
    ytr = trees["y_train"].astype(int)
    yte = trees["y_test"].astype(int)
    
    # Assert the other two match the trees' labels, else stacking is misaligned.
    for d in (cnn, esm):
        assert np.array_equal(d["y_train"].astype(int), ytr), "y_train misaligned"
        assert np.array_equal(d["y_test"].astype(int), yte), "y_test misaligned"

    # Collect each learner's OOF (train) and test probabilities, keyed by name.
    oof = {"rf": trees["rf_oof"], "xgb": trees["xgb_oof"],
           "cnn": cnn["cnn_oof"], "esm": esm["esm_oof"]}
    test = {"rf": trees["rf_test"], "xgb": trees["xgb_test"],
            "cnn": cnn["cnn_test"], "esm": esm["esm_test"]}
    return oof, test, ytr, yte


def logit(P, eps=1e-6):
    # Log-odds transform: log(p/(1-p)). Clip away from 0 and 1 first to avoid inf.
    P = np.clip(P.astype(np.float64), eps, 1 - eps)
    return np.log(P / (1 - P))


def metrics(y, p, thr=0.5):
    # Standard classification metrics at a 0.5 threshold (AUC uses the probabilities).
    ypred = (p > thr).astype(int)
    

    return dict(AUC=roc_auc_score(y, p), ACC=accuracy_score(y, ypred),
                MCC=matthews_corrcoef(y, ypred), F1=f1_score(y, ypred),
                Prec=precision_score(y, ypred), Rec=recall_score(y, ypred))


def make_estimator(standardize):
    # The meta-learner: logistic regression, optionally with a StandardScaler in
    # front. Wrapping in a Pipeline means the scaler is re-fit per CV fold (no leakage).
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    if standardize:
        return Pipeline([("sc", StandardScaler()), ("lr", lr)])
    return lr


def run_variant(cols, transform, standardize, oof, test, ytr, yte):
    # Build the meta-feature matrices from the chosen base learners (columns):
    # train = their OOF probabilities, test = their test probabilities.
    Xtr = np.column_stack([oof[c] for c in cols]).astype(np.float64)
    Xte = np.column_stack([test[c] for c in cols]).astype(np.float64)
    
    # Optional log-odds transform (logit) before the LR. Standardisation is handled
    # inside the Pipeline, so it re-fits per fold (no leakage).
    if transform == "logit":
        Xtr, Xte = logit(Xtr), logit(Xte)        # optional log-odds transform
    est = make_estimator(standardize)
    # stack_oof: honest train-set estimate, same 5-fold scheme as the base learners.
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    
    # cross_val_predict gives OOF predictions for each row, honest estimate
    oof_p = cross_val_predict(est, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
    
    # stack_test: fit the meta-learner on all OOF rows, apply to the base test probs.
    est.fit(Xtr, ytr)
    test_p = est.predict_proba(Xte)[:, 1] # get test probabilities from the fitted meta-learner
    return est, oof_p, test_p


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    oof, test, ytr, yte = load_bases()           # row-aligned base-learner predictions

    # Best single base learner on the test set = the benchmark the stack must beat (Aim 3).
    base_auc = {n: roc_auc_score(yte, test[n]) for n in BASE_NAMES}
    best_name = max(base_auc, key=base_auc.get)
    best_auc = base_auc[best_name]

    # Stacking configs to try: (label, base learners, transform, standardise).
    # raw-all4 is the canonical one (per the outline); the rest are robustness checks.
    variants = [
        ("raw   . all4", BASE_NAMES, "raw", False), # primary / canonical
        ("std   . all4", BASE_NAMES, "raw", True), # comparable coefficients
        ("logit . all4", BASE_NAMES, "logit", True), # log-odds stacking
        ("logit . no-cnn", ["rf", "xgb", "esm"], "logit", True),  # drop the weak CNN
    ]

    rows = []
    primary = None
    coef_est = None

    # Run each variant, collect metrics and identify the primary and standardised fits.
    for label, cols, transform, standardise in variants:
        est, oof_p, test_p = run_variant(cols, transform, standardise,
                                         oof, test, ytr, yte)
        mt = metrics(yte, test_p)
        rows.append((label, len(cols), roc_auc_score(ytr, oof_p), mt,
                     mt["AUC"] - best_auc))      # last item = delta vs the best single learner
        if label.startswith("raw"):
            primary = (oof_p, test_p, est)       # canonical stack -> this is what gets saved
        if label.startswith("std"):
            coef_est = (cols, est)               # standardised fit -> for comparable weights

    # report the results in a clean table for the write-up
    print("base learners (test set):")
    print(f"  {'model':6s} {'AUC':>6s}")
    for n in BASE_NAMES:
        print(f"  {n:6s} {base_auc[n]:6.3f}" +
              ("   <- best single" if n == best_name else ""))
    print()

    print("stacking variants:")
    hdr = f"  {'variant':16s} {'k':>2s} {'OOF_AUC':>8s} {'AUC':>6s} " \
          f"{'ACC':>6s} {'MCC':>6s} {'F1':>6s} {'dAUC':>6s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2)) # separator
    for label, k, oof_auc, mt, d in rows:        # k = number of base learners in this stack
        
        # Print the results for each variant, highlighting the primary one.
        print(f"  {label:16s} {k:2d} {oof_auc:8.3f} {mt['AUC']:6.3f} "
              f"{mt['ACC']:6.3f} {mt['MCC']:6.3f} {mt['F1']:6.3f} {d:+6.3f}"
              + ("   (primary)" if label.startswith("raw") else ""))

    # Meta-learner weights from the standardised fit — the only version where the
    # coefficients are comparable across learners (raw-probability coeffs are not).
    cols, est = coef_est
    lr = est.named_steps["lr"]
    print(f"\nmeta-learner importance (standardised all4 — comparable weights):")
    for n, c in sorted(zip(cols, lr.coef_[0]), key=lambda kv: -kv[1]):
        print(f"  {n:6s} {c:+.3f}")


    # save canonical (primary raw all4) + sensitivity table
    # Save the primary stack's OOF + test probabilities, the
    # labels and the raw meta-feature matrices for reference.
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
    joblib.dump(est, OUT_META)                   # the fitted primary meta-learner

    # One row per variant. This gives a clean sensitivity table for the write-up.
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "n_models", "oof_auc", "test_auc", "test_acc",
                    "test_mcc", "test_f1", "delta_vs_best_single"])
        for label, k, oof_auc, mt, d in rows:
            w.writerow([label.replace(" ", ""), k, f"{oof_auc:.4f}",
                        f"{mt['AUC']:.4f}", f"{mt['ACC']:.4f}",
                        f"{mt['MCC']:.4f}", f"{mt['F1']:.4f}", f"{d:+.4f}"])

    print(f"\nsaved -> {OUT_NPZ}, {OUT_META}, {OUT_CSV}")


if __name__ == "__main__":
    main()
