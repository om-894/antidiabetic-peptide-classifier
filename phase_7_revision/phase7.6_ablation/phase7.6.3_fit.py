"""
Revision analysis: fit and score the six negative-class arms.

Each arm keeps the 873 training positives and the published 178-row test set
and varies only its training negatives, so AUC, MCC and the two false-positive
rates are directly comparable across arms. The readout is XGBoost on the fused
1,287-d vector under the deployed model's own hyperparameters, chosen because
it is seed-stable to SD 0.002, so a difference between arms is the negative
class rather than the fit.

Charge separation is reported beside each arm: the AUC of net charge alone
predicting that a training row is a negative. It is the quantity the arms are
meant to move, and arm F is the test of whether it can be closed.

Arms C and D cannot reach 873 negatives, because the Basith pool holds nothing
below 9 residues and DBAASP runs out of sequences at the positives' lengths.
Each is therefore also fitted with the positives subsampled to the negative
count, so its false-positive rate is not read off a positive-heavy training
set.

INPUTS  ablation_arms.csv, fusion_vectors.npz, ablation_embeddings.npz,
        data/dataset_split.csv
OUTPUTS ablation_results.csv, ablation_results_summary.txt
"""

import os
import numpy as np
import pandas as pd
import peptides
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "results")
WORK = os.path.join(REPO, "phase_7_revision", "work")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

SEED = 42
THRESH = 0.5
N_REPEAT = 10 # training-row permutations per arm

XGB = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.5, reg_lambda=1.0, eval_metric="logloss",
           random_state=SEED)

# (row label, source arm, balance the positives, description)
ARMS = [
    ("A", "A", False, "dual-negative, as published"),
    ("B", "B", False, "Basith pool, length free"),
    ("C", "C", False, "Basith pool, length matched"),
    ("C-bal", "C", True, "Basith, length matched, balanced"),
    ("D", "D", False, "DBAASP only, length matched"),
    ("D-bal", "D", True, "DBAASP only, length matched, balanced"),
    ("E", "E", False, "proteome only, length matched"),
    ("F", "F", False, "length and charge matched"),
]


def descriptors(seq):
    """The seven descriptors of phase 2.2, in its order."""
    p = peptides.Peptide(seq)
    return [
        p.charge(pH=7.4),
        p.hydrophobicity(scale="KyteDoolittle"),
        p.isoelectric_point(),
        sum(seq.count(a) for a in "FWY") / len(seq),
        p.instability_index(),
        p.aliphatic_index(),
        p.boman(),
    ]


def main():
    arms = pd.read_csv(f"{WORK}/ablation_arms.csv", keep_default_na=False)
    fus = np.load(f"{REPO}/fusion_vectors.npz", allow_pickle=True)
    extra = np.load(f"{WORK}/ablation_embeddings.npz", allow_pickle=True)

    emb = {s: v for s, v in zip(fus["sequences"], np.asarray(fus["X"])[:, :1280])}
    emb.update({s: v for s, v in zip(extra["sequences"], extra["X"])})

    # reuse phase 2.2's saved raw descriptors where they exist so the originals
    # are bit-identical, and compute only the sequences the arms add
    desc = {s: d for s, d in zip(fus["sequences"], fus["descriptors_raw"])}
    todo = [s for s in set(arms.Sequence) if s not in desc]
    for s in todo:
        desc[s] = np.array(descriptors(s), dtype=np.float32)
    print(f"{len(emb)} embeddings, {len(desc)} descriptor rows "
          f"({len(todo)} newly computed)")

    missing = sorted(set(arms.Sequence) - set(emb))
    assert not missing, f"{len(missing)} sequences lack an embedding, e.g. {missing[:3]}"

    # arm A is the published model, so it reads its fused rows straight from
    # phase 2.2's saved file. rebuilding them from descriptors_raw changes the
    # seven scaled columns in the last float32 bit, which is enough to move this
    # learner by 0.003 and would stop the arm anchoring on the paper's own number
    saved_fused = {s: v for s, v in zip(fus["sequences"], np.asarray(fus["X"]))}

    split = pd.read_csv(f"{REPO}/data/dataset_split.csv", keep_default_na=False)
    negtype = dict(zip(split.Sequence, split.NegType))
    charge_of = {s: float(desc[s][0]) for s in desc} # column 0 is net charge

    rng = np.random.default_rng(SEED)
    rows, lines = [], []
    for label, src, balance, note in ARMS:
        a = arms[arms.arm == src]
        tr, te = a[a.Split == "train"], a[a.Split == "test"]

        if balance:
            n_neg = int((tr.Label == 0).sum())
            pos_idx = tr.index[tr.Label == 1].to_numpy()
            keep = rng.choice(pos_idx, size=n_neg, replace=False)
            tr = tr.loc[np.sort(np.concatenate([keep, tr.index[tr.Label == 0].to_numpy()]))]

        def build(frame, mu=None, sd=None):
            if src == "A":
                return np.vstack([saved_fused[s] for s in frame.Sequence]), mu, sd
            E = np.vstack([emb[s] for s in frame.Sequence])
            D = np.vstack([desc[s] for s in frame.Sequence]).astype(np.float32)
            if mu is None:
                mu, sd = D.mean(0), D.std(0)
                sd = np.where(sd == 0, 1.0, sd)
            return np.hstack([E, (D - mu) / sd]).astype(np.float32), mu, sd

        # scaling statistics come from this arm's training rows alone, as in
        # phase 2.2, so no test information reaches the transform
        Xtr, mu, sd = build(tr)
        Xte, _, _ = build(te, mu, sd)
        ytr, yte = tr.Label.to_numpy(), te.Label.to_numpy()

        # false positives by negative source, every row in each block a true non-ADP
        te_types = te.Sequence.map(lambda s: negtype.get(s, "")).to_numpy()
        hard = (yte == 0) & (te_types == "hard")
        soft = (yte == 0) & (te_types == "soft")

        # subsample draws rows in their given order, so permuting the training
        # rows moves this learner by about as much as the arms differ by. every
        # arm is therefore refitted under N_REPEAT permutations and reported as a
        # mean with a spread, which is what makes an arm comparison readable
        aucs, mccs, fh, fs = [], [], [], []
        for rep in range(N_REPEAT):
            perm = np.random.default_rng(rep).permutation(len(ytr))
            model = XGBClassifier(**XGB).fit(Xtr[perm], ytr[perm])
            p = model.predict_proba(Xte)[:, 1]
            pred = (p >= THRESH).astype(int)
            aucs.append(roc_auc_score(yte, p))
            mccs.append(matthews_corrcoef(yte, pred))
            fh.append(pred[hard].mean() * 100)
            fs.append(pred[soft].mean() * 100)

        auc, auc_sd = float(np.mean(aucs)), float(np.std(aucs))
        mcc = float(np.mean(mccs))
        fpr_hard, fpr_hard_sd = float(np.mean(fh)), float(np.std(fh))
        fpr_soft = float(np.mean(fs))

        # does net charge alone separate this arm's training classes?
        ch = tr.Sequence.map(charge_of).to_numpy()
        charge_sep = roc_auc_score(1 - ytr, ch)

        rows.append({
            "arm": label, "negatives": note,
            "n_train_pos": int((ytr == 1).sum()), "n_train_neg": int((ytr == 0).sum()),
            "mean_neg_length": round(float(np.mean([len(s) for s in
                                                   tr.Sequence[ytr == 0]])), 1),
            "mean_neg_charge": round(float(ch[ytr == 0].mean()), 2),
            "auc": round(auc, 4), "auc_sd": round(auc_sd, 4), "mcc": round(mcc, 4),
            "fpr_hard_pct": round(fpr_hard, 1), "fpr_hard_sd": round(fpr_hard_sd, 1),
            "fpr_soft_pct": round(fpr_soft, 1),
            "charge_separation_auc": round(charge_sep, 4),
        })
        lines.append(
            f"{label:6s} {note:34s} pos={int((ytr == 1).sum()):3d} "
            f"neg={int((ytr == 0).sum()):4d}  AUC {auc:.3f}+-{auc_sd:.3f}  "
            f"MCC {mcc:.3f}  FPR hard {fpr_hard:5.1f}+-{fpr_hard_sd:4.1f}%  "
            f"FPR soft {fpr_soft:5.1f}%  charge sep {charge_sep:.3f}")
        print(lines[-1])

    pd.DataFrame(rows).to_csv(f"{OUT}/phase7_6_ablation_results.csv", index=False)

    te_a = arms[(arms.arm == "A") & (arms.Split == "test")]
    t = te_a.Sequence.map(lambda s: negtype.get(s, ""))
    s = "\n".join([
        f"XGBoost on the fused vector, shared {len(te_a)}-row test set "
        f"({int((te_a.Label == 1).sum())} positives, "
        f"{int((t == 'soft').sum())} soft negatives, {int((t == 'hard').sum())} hard)",
        "", *lines])
    open(f"{OUT}/phase7_6_ablation_summary.txt", "w").write(s + "\n")
    print("\nsaved ablation_results.csv")


if __name__ == "__main__":
    main()
