"""
Revision audit: how close is the held-out test set to the training set?

Answers the reviewer's first point. The 870 sequences below 11 residues took a
stratified random split because 40% identity is not meaningful over so few
residues, so the clustering guarantee does not cover them. This measures what
that costs three ways: exact matches, substring containment, and a graded
identity, then refits nothing and simply rescores the saved predictions with
the near-duplicate test rows dropped.

Identity is normalised over the LONGER sequence of each pair. CD-HIT's own
convention normalises over the shorter, which saturates at 1.0 for any
containment, so a dipeptide inside a 20-mer reads as 100% identity and the
metric is useless at the lengths that matter here.

INPUTS  data/dataset_split.csv, predictions/*.npz
OUTPUTS audit_overlap_pairs.csv, audit_overlap_summary.txt
"""

import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "results")
WORK = os.path.join(REPO, "phase_7_revision", "work")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

NEAR_DUP = 0.80
N_NULL = 200
SEED = 42


# --------------------------------------------------------------------------- #
# IDENTITY
# --------------------------------------------------------------------------- #

def lcs_len(a, b):
    """Length of the longest common subsequence, two rolling rows."""
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def comp_bound(ca, cb):
    """Upper bound on LCS from shared composition alone."""
    # a residue can be matched at most as often as it appears in both, so this
    # caps LCS without the quadratic fill
    return sum(min(n, cb.get(r, 0)) for r, n in ca.items())


def nearest_identity(s, cs, tr_seqs, tr_comp):
    """Highest identity over the longer against any training sequence, and its index."""
    # branch and bound on the composition cap: the quadratic fill only runs when
    # a pair could still beat the best seen, which skips almost every pair while
    # still returning the true maximum rather than a thresholded one
    best, best_j = 0.0, -1
    for j, t in enumerate(tr_seqs):
        longer = len(s) if len(s) > len(t) else len(t)
        if comp_bound(cs, tr_comp[j]) <= best * longer:
            continue
        v = lcs_len(s, t) / longer
        if v > best:
            best, best_j = v, j
    return best, best_j


# --------------------------------------------------------------------------- #
# SUBSTRING CONTAINMENT
# --------------------------------------------------------------------------- #

def substr_counter(tr_seqs):
    """Return a function counting test sequences in a substring relationship with train."""
    # one sentinel-joined string makes "test inside a training sequence" a single
    # C-level search, and a set of training sequences makes the other direction a
    # lookup over the test sequence's own substrings. the naive 178 x 1754 loop
    # is far too slow to repeat across the shuffled null
    big = "|".join(tr_seqs)
    tr_set = set(tr_seqs)
    lens = sorted({len(t) for t in tr_seqs})

    def hits(seqs):
        n = 0
        for s in seqs:
            if s in big:
                n += 1
                continue
            found = False
            for L in lens:
                if L > len(s):
                    break
                for i in range(len(s) - L + 1):
                    if s[i:i + L] in tr_set:
                        found = True
                        break
                if found:
                    break
            n += found
        return n

    return hits


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    df = pd.read_csv(f"{REPO}/data/dataset_split.csv", keep_default_na=False)
    train = df[df.Split == "train"].reset_index(drop=True)
    test = df[df.Split == "test"].reset_index(drop=True)
    tr_seqs = train.Sequence.tolist()
    te_seqs = test.Sequence.tolist()
    print(f"{len(tr_seqs)} train, {len(te_seqs)} test")

    tr_set = set(tr_seqs)
    tr_comp = [Counter(s) for s in tr_seqs]
    hits = substr_counter(tr_seqs)

    exact = [s for s in te_seqs if s in tr_set]
    n_substr = hits(te_seqs)

    # residue-shuffled null holding each test sequence's length and composition
    # fixed, so it isolates ordering from the two things the split controlled
    rng = np.random.default_rng(SEED)
    null = np.array([hits(["".join(rng.permutation(list(s))) for s in te_seqs])
                     for _ in range(N_NULL)])

    rows = []
    for i, s in enumerate(te_seqs):
        best, j = nearest_identity(s, Counter(s), tr_seqs, tr_comp)
        rows.append({
            "test_seq": s,
            "test_label": int(test.Label[i]),
            "test_class": test.Class[i],
            "test_len": len(s),
            "nearest_train": tr_seqs[j] if j >= 0 else "",
            "nearest_train_label": int(train.Label[j]) if j >= 0 else -1,
            "identity_over_longer": round(best, 4),
            "exact": s in tr_set,
            "substring": hits([s]) == 1,
        })
        if (i + 1) % 40 == 0:
            print(f"  scored {i + 1} of {len(te_seqs)}")

    pairs = pd.DataFrame(rows)
    pairs.to_csv(f"{OUT}/phase7_1_overlap_pairs.csv", index=False)
    near = pairs.identity_over_longer >= NEAR_DUP

    # rescore the saved predictions with near-duplicate test rows dropped. no
    # model is refitted, so this isolates the split's contribution to the AUC
    keep = (~near).to_numpy()
    y = test.Label.to_numpy()
    assert (np.load(f"{REPO}/predictions/esm2_dora_predictions.npz")["y_test"] == y).all(), \
        "test row order differs between dataset_split.csv and the saved predictions"

    lines = []
    esm = np.load(f"{REPO}/predictions/esm2_dora_predictions.npz")["esm_test"]
    tree = np.load(f"{REPO}/predictions/base_tree_predictions.npz")
    for name, prob in [("ESM-2/DoRA", esm), ("XGBoost", tree["xgb_test"]),
                       ("Random Forest", tree["rf_test"]),
                       ("consensus (ESM-2 + XGB)", (esm + tree["xgb_test"]) / 2)]:
        full = roc_auc_score(y, prob)
        clean = roc_auc_score(y[keep], prob[keep])
        lines.append(f"{name:24s} full {full:.4f}  near-dup removed {clean:.4f}  "
                     f"delta {full - clean:+.4f}")

    txt = [
        f"test rows                        {len(te_seqs)}",
        f"exact matches in train           {len(exact)}",
        f"substring relationship           {n_substr} ({100 * n_substr / len(te_seqs):.1f}%)",
        f"  shuffled null mean             {null.mean():.1f} "
        f"({100 * null.mean() / len(te_seqs):.1f}%), SD {null.std():.1f}",
        f"  null range                     {null.min()} to {null.max()}",
        f"  empirical P                    {(null >= n_substr).mean():.4f} "
        f"({N_NULL} shuffles, seed {SEED})",
        f"near duplicates >= {NEAR_DUP:.2f}        {int(near.sum())} "
        f"({100 * near.sum() / len(te_seqs):.1f}%)",
        f"  of which positives             {int((near & (pairs.test_label == 1)).sum())}",
        f"  median length of those         {pairs.test_len[near].median():.0f}",
        f"  below 11 residues              {int((near & (pairs.test_len < 11)).sum())}"
        f" of {int((pairs.test_len < 11).sum())} short test rows",
        f"  11 residues or more            {int((near & (pairs.test_len >= 11)).sum())}"
        f" of {int((pairs.test_len >= 11).sum())} long test rows",
        f"mean nearest identity            {pairs.identity_over_longer.mean():.3f}",
        f"  short rows (< 11)              "
        f"{pairs.identity_over_longer[pairs.test_len < 11].mean():.3f}",
        f"  long rows (>= 11)              "
        f"{pairs.identity_over_longer[pairs.test_len >= 11].mean():.3f}",
        "",
        "AUC with near-duplicate test rows removed (no refit):",
        *lines,
    ]
    out = "\n".join(txt)
    print("\n" + out)
    open(f"{OUT}/phase7_1_overlap_summary.txt", "w").write(out + "\n")


if __name__ == "__main__":
    main()
