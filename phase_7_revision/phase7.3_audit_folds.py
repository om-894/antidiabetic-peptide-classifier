"""
Revision audit: does cluster-level leakage across the 5 CV folds inflate the
out-of-fold calibration the 0.90 threshold was set on?

The reviewer's second point. The published folds are StratifiedKFold over the
1,754 training rows, which splits on rows rather than on families, so a
validation row can have a near-identical relative among that same fold's
training rows. The threshold was swept on those out-of-fold predictions, so any
inflation passes straight into the operating point.

This measures the leakage directly, then re-runs the fold scheme grouped by
family and re-sweeps the threshold on the grouped out-of-fold predictions. The
readout is XGBoost on the fused vector, which is seed-stable to SD 0.002, so a
change is attributable to the fold scheme rather than to the fit.

INPUTS  data/dataset_split.csv, fusion_vectors.npz
OUTPUTS audit_folds_summary.txt, audit_folds_oof.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "results")
WORK = os.path.join(REPO, "phase_7_revision", "work")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

COVER = 0.50
SEED = 42
N_SPLITS = 5
MIN_KEEP = 30
MIN_PRECISION = 0.90

# the deployed base_xgb.joblib's own parameters, so the readout is the paper's
# learner rather than a fresh configuration
XGB = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.5, reg_lambda=1.0, eval_metric="logloss",
           random_state=SEED)


class Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def families(seqs, cover=COVER):
    """Single-linkage families under containment with a coverage floor."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    u = Union(len(seqs))
    for oi, i in enumerate(order):
        si = seqs[i]
        for j in order[oi + 1:]:
            sj = seqs[j]
            if len(si) < cover * len(sj):
                break
            if si in sj:
                u.join(i, j)
    return np.array([u.find(i) for i in range(len(seqs))])


def sweep(prob, y):
    """Lowest cut-off keeping >= MIN_KEEP rows at >= MIN_PRECISION, as in phase 5."""
    for t in np.round(np.arange(0.50, 1.00, 0.01), 2):
        sel = prob >= t
        if sel.sum() >= MIN_KEEP and y[sel].mean() >= MIN_PRECISION:
            return t, int(sel.sum()), float(y[sel].mean())
    return None, 0, float("nan")


def oof(X, y, folds):
    """Out-of-fold probabilities under a given fold iterator."""
    p = np.zeros(len(y))
    for tr, va in folds:
        m = XGBClassifier(**XGB).fit(X[tr], y[tr])
        p[va] = m.predict_proba(X[va])[:, 1]
    return p


def main():
    split = pd.read_csv(f"{REPO}/data/dataset_split.csv", keep_default_na=False)
    fus = np.load(f"{REPO}/fusion_vectors.npz", allow_pickle=True)

    # fusion_vectors.npz carries its own sequence order, so align on it rather
    # than assuming the csv and the npz were written in the same order
    assert list(fus["sequences"]) == split.Sequence.tolist(), "row order mismatch"
    tr = (split.Split == "train").to_numpy()
    X, y = fus["X"][tr], split.Label.to_numpy()[tr]
    seqs = split.Sequence[tr].tolist()
    print(f"{len(seqs)} training rows")

    fam = families(seqs)
    n_fam = len(set(fam))
    print(f"{n_fam} families among the training rows")

    # how much leakage does the published scheme carry?
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    std_folds = list(skf.split(X, y))
    grp_folds = list(sgkf.split(X, y, groups=fam))

    leak = []
    for k, (a, b) in enumerate(std_folds):
        tr_fams = set(fam[a])
        n = int(np.isin(fam[b], list(tr_fams)).sum())
        leak.append((k, len(b), n, 100 * n / len(b)))

    grp_leak = sum(int(np.isin(fam[b], list(set(fam[a]))).sum())
                   for a, b in grp_folds)

    p_std = oof(X, y, std_folds)
    p_grp = oof(X, y, grp_folds)
    auc_std, auc_grp = roc_auc_score(y, p_std), roc_auc_score(y, p_grp)
    t_std = sweep(p_std, y)
    t_grp = sweep(p_grp, y)

    pd.DataFrame({"Sequence": seqs, "label": y, "family": fam,
                  "oof_stratified": p_std, "oof_grouped": p_grp}
                 ).to_csv(f"{OUT}/phase7_3_folds_oof.csv", index=False)

    lines = [
        f"training rows                    {len(seqs)}",
        f"families (containment, >= {COVER:.0%})   {n_fam}",
        "",
        "leakage under the published StratifiedKFold scheme:",
        *[f"  fold {k}: {n:4d} of {nb:4d} validation rows have a family relative "
          f"in that fold's training rows ({pct:.1f}%)" for k, nb, n, pct in leak],
        f"  overall: {sum(l[2] for l in leak)} of {len(seqs)} "
        f"({100 * sum(l[2] for l in leak) / len(seqs):.1f}%)",
        f"  same count under StratifiedGroupKFold: {grp_leak} (zero by construction)",
        "",
        "XGBoost out-of-fold readout:",
        f"  stratified folds   AUC {auc_std:.4f}   swept threshold {t_std[0]} "
        f"(keeps {t_std[1]}, precision {t_std[2]:.3f})",
        f"  grouped folds      AUC {auc_grp:.4f}   swept threshold {t_grp[0]} "
        f"(keeps {t_grp[1]}, precision {t_grp[2]:.3f})",
        f"  AUC inflation from the fold scheme: {auc_std - auc_grp:+.4f}",
    ]
    s = "\n".join(lines)
    print("\n" + s)
    open(f"{OUT}/phase7_3_folds_summary.txt", "w").write(s + "\n")


if __name__ == "__main__":
    main()
