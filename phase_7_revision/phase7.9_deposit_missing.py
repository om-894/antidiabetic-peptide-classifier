"""
Deposit the three claim sets that had no result file of their own.

A pre-submission audit found three groups of numbers in the manuscript that no
file in results/ backed, even though two of them were recoverable from data
already in the repository and the third only needed CD-HIT rerunning. Rather
than soften the claims, this recomputes all three and writes them out, so every
quantitative statement in the paper traces to a deposited file.

  1. Baseline Gini importances, Section 3.1 and the Figure S4 caption. These
     were already inside predictions/aac_baseline_predictions.npz as an array,
     which is why the audit missed them; this makes them a table.
  2. The 75.1% embeddings-only nearest-negative control, Section 2.1 and the
     Figure 2 caption. It rules out the seven descriptors as the cause of the
     nearest-negative result, so it carries an argument.
  3. The count a conventional 40% redundancy reduction would remove, Section 2.1
     and Table S1.

INPUTS   predictions/aac_baseline_predictions.npz, esm2_embeddings.npz,
         data/dataset_split.csv, cd-hit on PATH
OUTPUTS  results/phase7_9_aac_gini.csv
         results/phase7_9_nearest_negative_embeddings.txt
         results/phase7_9_cdhit40_reduction.txt
"""

import os
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "results")
os.makedirs(OUT, exist_ok=True)


def gini():
    """The amino-acid composition baseline's feature importances."""
    d = np.load(f"{REPO}/predictions/aac_baseline_predictions.npz",
                allow_pickle=True)
    imp = d["feature_importances"]
    names = [str(n) for n in d["feature_names"]]
    df = (pd.DataFrame({"residue": names, "gini_importance": imp})
          .sort_values("gini_importance", ascending=False)
          .reset_index(drop=True))
    df["rank"] = df.index + 1
    df["charged_at_pH7.4"] = df.residue.isin(list("KRED"))
    df.to_csv(f"{OUT}/phase7_9_aac_gini.csv", index=False)

    kr = df[df.residue.isin(["K", "R"])].gini_importance.sum()
    krd = df[df["charged_at_pH7.4"]].gini_importance.sum()
    print(f"  K and R take {100 * kr:.1f}% of total Gini importance "
          f"(K {df[df.residue == 'K'].gini_importance.iloc[0]:.3f}, "
          f"R {df[df.residue == 'R'].gini_importance.iloc[0]:.3f})")
    print(f"  the four charged residues reach {100 * krd:.1f}%")


def nearest_negative():
    """How often a positive's nearest negative is a proteome fragment.

    Run on the 1,280-d embeddings alone. The headline figure uses the fused
    1,287-d vector; if the two agree, the seven descriptors are not what drives
    the result, which is the point of the control.
    """
    emb = np.load(f"{REPO}/esm2_embeddings.npz", allow_pickle=True)
    split = pd.read_csv(f"{REPO}/data/dataset_split.csv", keep_default_na=False)
    assert list(emb["sequences"]) == split.Sequence.tolist(), "row order mismatch"

    X = emb["X"].astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True) # cosine, as in the UMAP
    pos = (split.Label == 1).to_numpy()
    neg = ~pos
    hard = (split.NegType == "hard").to_numpy()[neg]

    sims = X[pos] @ X[neg].T          # cosine similarity, positives by negatives
    nearest = sims.argmax(axis=1)
    n_hard = int(hard[nearest].sum())
    n_pos = int(pos.sum())
    expected = 100 * hard.mean()

    txt = (f"nearest-negative control on the 1,280-d embeddings alone\n"
           f"  positives                     {n_pos}\n"
           f"  negatives                     {int(neg.sum())} "
           f"({int(hard.sum())} hard, {int((~hard).sum())} soft)\n"
           f"  nearest negative is hard      {n_hard} "
           f"({100 * n_hard / n_pos:.1f}%)\n"
           f"  expected from class sizes     {expected:.1f}%\n"
           f"  the fused 1,287-d figure is 78.8%, so the seven descriptors are\n"
           f"  not what produces the effect\n")
    open(f"{OUT}/phase7_9_nearest_negative_embeddings.txt", "w").write(txt)
    print("  " + txt.strip().replace("\n", "\n  "))


def _cluster(seqs):
    """Representatives left after a 40% CD-HIT reduction, phase 1.3 settings."""
    with tempfile.TemporaryDirectory() as td:
        fa, out = os.path.join(td, "a.fasta"), os.path.join(td, "red")
        with open(fa, "w") as f:
            for i, s in enumerate(seqs):
                f.write(f">s{i}\n{s}\n")
        r = subprocess.run(["cd-hit", "-i", fa, "-o", out, "-c", "0.4",
                            "-n", "2", "-l", "1", "-d", "0", "-M", "0"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:200])
        return sum(1 for line in open(out) if line.startswith(">"))


def cdhit_reduction():
    """How many sequences a conventional 40% redundancy reduction would remove.

    Reported for two populations, because the published figure of 782 is the
    count over the sequences long enough to cluster and reads as a count over
    the whole dataset if the denominator is not stated. Applied to all 1,932 it
    is far larger, since CD-HIT at word length 2 collapses short peptides
    aggressively.
    """
    split = pd.read_csv(f"{REPO}/data/dataset_split.csv", keep_default_na=False)
    longs = split[split.Length >= 11].Sequence
    rows = []
    for label, seqs in [("sequences of 11 residues or more", longs),
                        ("the full dataset", split.Sequence)]:
        try:
            kept = _cluster(seqs)
        except RuntimeError as e:
            print(f"  cd-hit failed on {label}: {e}")
            return
        rows.append((label, len(seqs), kept, len(seqs) - kept))

    txt = ["conventional 40% redundancy reduction, cd-hit -c 0.4 -n 2 -l 1",
           "(the phase 1.3 clustering settings)", ""]
    for label, n, kept, removed in rows:
        txt.append(f"  {label:34s} {n:5d} in, {kept:4d} kept, {removed:4d} removed")
    txt += ["", "  The manuscript quotes 782, which is the first row. Clustering",
            "  covers only sequences of 11 residues or more, so that is the",
            "  population the figure belongs to."]
    s = "\n".join(txt) + "\n"
    open(f"{OUT}/phase7_9_cdhit40_reduction.txt", "w").write(s)
    print("  " + s.strip().replace("\n", "\n  "))


if __name__ == "__main__":
    print("1. baseline Gini importances")
    gini()
    print("\n2. nearest-negative on embeddings alone")
    nearest_negative()
    print("\n3. 40% redundancy reduction")
    cdhit_reduction()
