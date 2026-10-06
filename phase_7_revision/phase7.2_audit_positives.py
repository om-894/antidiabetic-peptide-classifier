"""
Revision audit: what is the internal redundancy of the 966-sequence positive set?

The reviewer's first priority. Nothing was deleted for redundancy when the
dataset was built, on the argument that near-duplicates inside the training set
are useful hard cases. That argument holds for training but not for counting:
966 sequences do not carry 966 sequences' worth of independent evidence if many
are overlapping fragments of one precursor.

Families are built by single linkage over pairs where one sequence contains the
other AND the shorter covers at least half the longer. The length condition is
load-bearing. Plain substring nesting is transitive through the two dipeptide
positives, which chains most of the set into one spurious family and does the
same to the independently drawn negatives, so a nesting-only definition
measures the metric rather than the data. Both variants are reported.

INPUTS  data/dataset_split.csv
OUTPUTS audit_positives_families.csv, audit_positives_summary.txt
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "results")
WORK = os.path.join(REPO, "phase_7_revision", "work")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

COVER = 0.50

# human preproinsulin P01308, used only to name the largest families. the four
# mature products are contiguous slices of it, so a family representative that
# is a substring of this string can be attributed without a database lookup
PREPROINSULIN = ("MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDL"
                 "QVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN")
REGIONS = [
    ("signal peptide", "MALWMRLLPLLALLALWGPDPAAA"),
    ("B chain", "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"),
    ("C-peptide", "RREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKR"),
    ("A chain", "GIVEQCCTSICSLYQLENYCN"),
]


class Union:
    """Union-find over sequence indices."""

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


def families(seqs, cover):
    """Single-linkage families under containment with a coverage floor."""
    # sorting by length lets the inner loop stop early: once the candidate is
    # longer than the outer sequence it cannot be contained in it
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    u = Union(len(seqs))
    for oi, i in enumerate(order):
        si = seqs[i]
        for j in order[oi + 1:]:
            sj = seqs[j]
            if len(si) < cover * len(sj):
                break # every later sequence is longer still
            if si in sj:
                u.join(i, j)
    return [u.find(i) for i in range(len(seqs))]


def region_of(seq):
    """Which mature insulin product a sequence falls in, or empty."""
    if seq not in PREPROINSULIN:
        return ""
    for name, region in REGIONS:
        if seq in region:
            return name
    return "spans products"


def summarise(df, label, cover, lines):
    seqs = df.Sequence.tolist()
    roots = families(seqs, cover)
    n_fam = len(set(roots))
    sizes = pd.Series(roots).value_counts()
    lines.append(f"{label:34s} {len(seqs)} sequences, {n_fam} families, "
                 f"largest {sizes.iloc[0]}, ESS {n_fam / len(seqs):.2f}")
    return roots, sizes


def main():
    df = pd.read_csv(f"{REPO}/data/dataset_split.csv", keep_default_na=False)
    pos = df[df.Label == 1].reset_index(drop=True)
    neg = df[df.Label == 0].reset_index(drop=True)
    lines = []

    lines.append(f"containment with the shorter covering >= {COVER:.0%} of the longer:")
    roots, sizes = summarise(pos, "  positives", COVER, lines)
    summarise(neg, "  negatives (control)", COVER, lines)

    lines.append("")
    lines.append("plain containment, no coverage floor, for comparison:")
    summarise(pos, "  positives", 0.0, lines)
    summarise(neg, "  negatives (control)", 0.0, lines)

    pos["family"] = roots
    pos["family_size"] = pos.family.map(sizes)
    pos["insulin_region"] = pos.Sequence.map(region_of)
    pos.to_csv(f"{OUT}/phase7_2_positives_families.csv", index=False)

    # a family split across train and test is a leakage channel the cluster-wise
    # partition was meant to close but could not reach below 11 residues
    byfam = pos.groupby("family").Split.nunique()
    straddle = byfam[byfam > 1].index
    n_straddle_seqs = int(pos.family.isin(straddle).sum())

    in_pre = pos.Sequence.map(lambda s: s in PREPROINSULIN)
    neg_in_pre = neg.Sequence.map(lambda s: s in PREPROINSULIN)

    lines += [
        "",
        f"families straddling the published split  {len(straddle)}, "
        f"carrying {n_straddle_seqs} positives ({100 * n_straddle_seqs / len(pos):.1f}%)",
        "",
        f"preproinsulin fragments among positives  {int(in_pre.sum())} of {len(pos)} "
        f"({100 * in_pre.mean():.1f}%)",
        f"preproinsulin fragments among negatives  {int(neg_in_pre.sum())} of {len(neg)}",
        "",
        "ten largest positive families:",
    ]
    for root, n in sizes.head(10).items():
        fam = pos[pos.family == root]
        rep = fam.loc[fam.Length.idxmax()]
        region = region_of(rep.Sequence)
        lines.append(f"  n={n:3d}  {rep.Length:2d} aa  "
                     f"{'insulin ' + region if region else 'unattributed'}"
                     f"  splits {'/'.join(sorted(fam.Split.unique()))}")
        lines.append(f"         {rep.Sequence}")

    s = "\n".join(lines)
    print(s)
    open(f"{OUT}/phase7_2_positives_summary.txt", "w").write(s + "\n")


if __name__ == "__main__":
    main()
