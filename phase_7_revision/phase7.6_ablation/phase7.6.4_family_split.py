"""
Revision analysis: does the negative-class effect survive a family-aware split?

The published partition clusters at 40% identity only above 10 residues, so 870
sequences took a stratified random split and 21 positive families straddle the
result. If the negative-class effect were an artefact of those straddling
families, it would shrink once whole families are held out.

The shared test set is what makes an arm comparison mean anything, so it is
rebuilt once, family-aware, over the published 1,932-sequence dataset, and
every arm is then scored on it. Each arm trains on the family-aware training
positives plus its own negatives, with anything appearing in the new test set
removed. Giving each arm its own split instead would score each on its own
negative distribution, which flatters exactly the arms that are easiest to
separate.

The published and family-aware test sets differ, so only the gap between two
arms within one split is comparable, never one arm's AUC across splits.

INPUTS  ablation_arms.csv, fusion_vectors.npz, ablation_embeddings.npz,
        data/dataset_split.csv
OUTPUTS audit_family_split_summary.txt, audit_family_split_results.csv
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
N_REPEAT = 10
TEST_FRAC = 0.15
COVER = 0.50

XGB = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.5, reg_lambda=1.0, eval_metric="logloss",
           random_state=SEED)

ARMS = ["A", "B", "C", "D", "E", "F"]


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
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    u = Union(len(seqs))
    for oi, i in enumerate(order):
        si = seqs[i]
        for j in order[oi + 1:]:
            if len(si) < cover * len(seqs[j]):
                break
            if si in seqs[j]:
                u.join(i, j)
    return np.array([u.find(i) for i in range(len(seqs))])


def descriptors(seq):
    p = peptides.Peptide(seq)
    return [p.charge(pH=7.4), p.hydrophobicity(scale="KyteDoolittle"),
            p.isoelectric_point(), sum(seq.count(a) for a in "FWY") / len(seq),
            p.instability_index(), p.aliphatic_index(), p.boman()]


def main():
    arms = pd.read_csv(f"{WORK}/ablation_arms.csv", keep_default_na=False)
    fus = np.load(f"{REPO}/fusion_vectors.npz", allow_pickle=True)
    extra = np.load(f"{WORK}/ablation_embeddings.npz", allow_pickle=True)
    split = pd.read_csv(f"{REPO}/data/dataset_split.csv", keep_default_na=False)
    negtype = dict(zip(split.Sequence, split.NegType))

    emb = {s: v for s, v in zip(fus["sequences"], np.asarray(fus["X"])[:, :1280])}
    emb.update({s: v for s, v in zip(extra["sequences"], extra["X"])})
    desc = {s: d for s, d in zip(fus["sequences"], fus["descriptors_raw"])}
    for s in set(arms.Sequence) - set(desc):
        desc[s] = np.array(descriptors(s), dtype=np.float32)

    # one family-aware test set over the published dataset, reused by every arm.
    # hard negatives are held to their share of the test set explicitly, so the
    # block the deployment claim rests on does not shrink to a handful of rows
    seqs = split.Sequence.tolist()
    y = split.Label.to_numpy()
    block = np.where(y == 1, "pos", split.NegType.to_numpy())
    fam = families(seqs)
    want = {b: int(round(TEST_FRAC * (block == b).sum())) for b in ("pos", "soft", "hard")}
    got = {b: 0 for b in want}
    rng = np.random.default_rng(SEED)
    test_fams = set()
    for f in rng.permutation(np.unique(fam)):
        m = fam == f
        b = pd.Series(block[m]).mode()[0]
        if got[b] + int(m.sum()) <= want[b]:
            test_fams.add(f)
            got[b] += int(m.sum())
        if all(got[k] >= want[k] for k in got):
            break
    is_test = np.isin(fam, list(test_fams))
    test_seqs = set(np.array(seqs)[is_test])
    print(f"{len(np.unique(fam))} families, test {int(is_test.sum())} rows, "
          f"blocks {got}, targets {want}")

    # fixed shared test matrix
    te_list = [s for s in seqs if s in test_seqs]
    yte = np.array([y[seqs.index(s)] for s in te_list])
    te_block = np.array([block[seqs.index(s)] for s in te_list])
    hard = (yte == 0) & (te_block == "hard")
    soft = (yte == 0) & (te_block == "soft")

    tr_pos = [s for s, lab, t in zip(seqs, y, is_test) if lab == 1 and not t]
    print(f"{len(tr_pos)} training positives, test blocks: "
          f"{int((yte == 1).sum())} pos / {int(soft.sum())} soft / {int(hard.sum())} hard")

    rows, lines = [], []
    for arm in ARMS:
        a = arms[(arms.arm == arm) & (arms.Split == "train") & (arms.Label == 0)]
        negs = [s for s in a.Sequence.unique() if s not in test_seqs]
        tr_list = tr_pos + negs
        ytr = np.array([1] * len(tr_pos) + [0] * len(negs))

        D_tr = np.vstack([desc[s] for s in tr_list]).astype(np.float32)
        mu = D_tr.mean(0)
        sd = np.where(D_tr.std(0) == 0, 1.0, D_tr.std(0))

        def fuse(lst):
            E = np.vstack([emb[s] for s in lst])
            D = np.vstack([desc[s] for s in lst]).astype(np.float32)
            return np.hstack([E, (D - mu) / sd]).astype(np.float32)

        Xtr, Xte = fuse(tr_list), fuse(te_list)

        aucs, fh, fs, mccs = [], [], [], []
        for rep in range(N_REPEAT):
            perm = np.random.default_rng(rep).permutation(len(ytr))
            m = XGBClassifier(**XGB).fit(Xtr[perm], ytr[perm])
            p = m.predict_proba(Xte)[:, 1]
            pred = (p >= THRESH).astype(int)
            aucs.append(roc_auc_score(yte, p))
            mccs.append(matthews_corrcoef(yte, pred))
            fh.append(pred[hard].mean() * 100)
            fs.append(pred[soft].mean() * 100)

        rows.append({"arm": arm, "n_train_neg": len(negs),
                     "auc": round(float(np.mean(aucs)), 4),
                     "auc_sd": round(float(np.std(aucs)), 4),
                     "mcc": round(float(np.mean(mccs)), 4),
                     "fpr_hard_pct": round(float(np.mean(fh)), 1),
                     "fpr_soft_pct": round(float(np.mean(fs)), 1)})
        lines.append(f"arm {arm}: neg={len(negs):4d}  AUC {np.mean(aucs):.3f}"
                     f"+-{np.std(aucs):.3f}  FPR hard {np.mean(fh):5.1f}%  "
                     f"FPR soft {np.mean(fs):5.1f}%")
        print(lines[-1])

    fa = pd.DataFrame(rows).set_index("arm")
    fa.to_csv(f"{OUT}/phase7_6_family_split_results.csv")
    pub = pd.read_csv(f"{OUT}/phase7_6_ablation_results.csv").set_index("arm")

    def gaps(t, lo, hi):
        return (t.loc[lo, "auc"] - t.loc[hi, "auc"],
                t.loc[hi, "fpr_hard_pct"] - t.loc[lo, "fpr_hard_pct"])

    s = ["one shared family-aware test set, every arm scored on it",
         f"test {int(is_test.sum())} rows: {int((yte == 1).sum())} positives, "
         f"{int(soft.sum())} soft negatives, {int(hard.sum())} hard negatives",
         "", *lines, "",
         "negative-class effect, published split vs family-aware split:"]
    for lo, hi, name in [("A", "B", "A vs B, composition and length"),
                         ("F", "B", "F vs B, matched vs published")]:
        pa, pf = gaps(pub, lo, hi)
        fa_a, fa_f = gaps(fa, lo, hi)
        s.append(f"  {name}")
        s.append(f"    AUC gap   published {pa:+.3f}      family-aware {fa_a:+.3f}")
        s.append(f"    FPR gap   published {pf:+.1f} pts   family-aware {fa_f:+.1f} pts")
    s.append("")
    s.append("test sets differ between the two splits, so only within-split gaps "
             "are comparable.")
    out = "\n".join(s)
    print("\n" + out)
    open(f"{OUT}/phase7_6_family_split_summary.txt", "w").write(out + "\n")


if __name__ == "__main__":
    main()
