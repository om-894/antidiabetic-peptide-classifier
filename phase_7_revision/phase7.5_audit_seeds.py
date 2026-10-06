"""
Revision audit: seed sensitivity, and whether frozen features remove it.

Two reviewer points at once.

Point 9 asks the text to state plainly that the 0.877 in Table 2 is the best of
five seeds rather than the seed-ensemble mean. This recomputes all five test
AUCs from the saved per-seed predictions so the two numbers in that row are
traceable to the same place.

Point 5 asks whether fixed feature extraction with a classical head removes the
seed sensitivity while keeping the ESM-2 representation. The frozen encoder is
deterministic, so anything fitted on top of it varies only through its own
fitting randomness. This fits logistic regression and XGBoost on the frozen
1,287-d vector under five seeds to put a number on that.

INPUTS  predictions/esm2_dora_seed4*.npz, fusion_vectors.npz,
        data/dataset_split.csv
OUTPUTS audit_seeds_summary.txt
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "results")
WORK = os.path.join(REPO, "phase_7_revision", "work")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]

XGB = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.5, reg_lambda=1.0, eval_metric="logloss")


def main():
    split = pd.read_csv(f"{REPO}/data/dataset_split.csv", keep_default_na=False)
    fus = np.load(f"{REPO}/fusion_vectors.npz", allow_pickle=True)
    tr = (split.Split == "train").to_numpy()
    y = split.Label.to_numpy()
    X = np.asarray(fus["X"])
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[~tr], y[~tr]

    lines = []

    # the fine-tune: a randomly initialised classification head moves the whole
    # optimisation, so the spread is structural rather than incidental
    aucs = []
    for s in SEEDS:
        d = np.load(f"{REPO}/predictions/esm2_dora_seed{s}_predictions.npz")
        aucs.append(roc_auc_score(d["y_test"], d["esm_test"]))
    aucs = np.array(aucs)
    lines.append("ESM-2/DoRA fine-tune, five seed refits:")
    lines += [f"  seed {s}: {a:.4f}" for s, a in zip(SEEDS, aucs)]
    lines.append(f"  mean {aucs.mean():.4f}, SD {aucs.std():.4f}, "
                 f"range {aucs.min():.4f} to {aucs.max():.4f}")
    lines.append(f"  the Table 2 point estimate {aucs.max():.4f} is the best of the "
                 f"five, {aucs.max() - aucs.mean():+.4f} above the mean")
    lines.append("")

    # frozen features with a classical head. the encoder never moves, so only
    # the head's own randomness is left
    for name, make in [
        ("logistic regression", lambda s: LogisticRegression(max_iter=5000, C=1.0,
                                                             random_state=s)),
        ("XGBoost", lambda s: XGBClassifier(**XGB, random_state=s)),
    ]:
        got = []
        for s in SEEDS:
            m = make(s).fit(Xtr, ytr)
            got.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
        got = np.array(got)
        lines.append(f"frozen ESM-2 + descriptors, {name}, five seeds:")
        lines.append("  " + "  ".join(f"{a:.4f}" for a in got))
        lines.append(f"  mean {got.mean():.4f}, SD {got.std():.4f}, "
                     f"range {got.max() - got.min():.4f}")
        lines.append("")

    lines.append(f"spread ratio, fine-tune range over XGBoost range: "
                 f"see the two range figures above")

    s = "\n".join(lines)
    print(s)
    open(f"{OUT}/phase7_5_seeds_summary.txt", "w").write(s + "\n")


if __name__ == "__main__":
    main()
