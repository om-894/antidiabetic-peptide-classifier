#!/usr/bin/env python3
"""
ablation_build_basith_negatives.py
==================================
Negative-class ablation, step 1: build Basith et al.'s conventional negative
pool (AntiDMPpred two-layer datasets) as a comparator for our dual-negative
class. All other components (the 966 positives, features, architecture,
hyperparameters) are held constant in the ablation; only the negatives change.

Basith's negatives are the union of the negative sequences across his four
files (Layer1_training, L1_Ind, Layer2_training, L2_Ind):
  Layer 1 negatives ~ antimicrobial / cell-penetrating / protein fragments
  Layer 2 negatives ~ short bioactive (ACE/DPP-IV-like) peptides
This matches the outline's description of Basith's pool (antimicrobial +
anticancer + antihypertensive, no hard negatives, no GO filtering).

FINDING reported here: how many of Basith's negatives are annotated as ADPs
(positives) in our consolidated Xie et al. set — i.e. label contradictions that
contaminate his decision boundary. These are removed before sampling (they
cannot serve as negatives against the same positives), but the rate is a
standalone Aim-1 result.

OUTPUTS
  data/basith_negatives_pool.csv   full cleaned pool (Sequence, Length, Sources)
  data/basith_negatives.csv        966 sampled, matching data/negatives.csv schema
                                   (Sequence, Label=0, Length, NegType="basith")
"""

import random
from collections import defaultdict

import pandas as pd

RAW = "data/basith_raw/"
FILES = ["Layer1_training.txt", "L1_Ind.txt", "Layer2_training.txt", "L2_Ind.txt"]
SEED = 42
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_fasta(path):
    pos, neg, cur = [], [], None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            cur = "pos" if "Positive" in line else "neg"
        else:
            (pos if cur == "pos" else neg).append(line)
    return pos, neg


def is_standard(s):
    return len(s) > 0 and set(s).issubset(STANDARD_AA)


def main():
    # collect unique negatives, tracking which file(s) each came from
    sources = defaultdict(set)
    for f in FILES:
        _, neg = parse_fasta(RAW + f)
        tag = "L1" if f.startswith(("Layer1", "L1")) else "L2"
        for s in neg:
            sources[s].add(tag)
    neg_all = set(sources)
    print(f"Basith negatives (unique across 4 files): {len(neg_all)}")

    non_std = {s for s in neg_all if not is_standard(s)}
    neg_all -= non_std
    print(f"  dropped {len(non_std)} non-standard-AA sequences -> {len(neg_all)}")

    # cross-reference against our consolidated dataset
    ds = pd.read_csv("data/dataset_split.csv")
    user_pos = set(ds[ds.Label == 1]["Sequence"])
    user_neg = set(ds[ds.Label == 0]["Sequence"])

    contradictions = neg_all & user_pos
    print(f"\n*** FINDING: {len(contradictions)} of Basith's negatives are ADPs "
          f"(positives) in our set ***")
    print(f"    = {100 * len(contradictions) / len(neg_all):.1f}% of his negative pool, "
          f"{100 * len(contradictions) / len(user_pos):.1f}% relative to our 966 positives")
    overlap_neg = neg_all & user_neg
    print(f"    (overlap with OUR negatives: {len(overlap_neg)} — essentially none)")

    clean = neg_all - user_pos - user_neg
    print(f"\nclean Basith pool (contradictions + our-neg overlap removed): {len(clean)}")

    # sample 966 to match the positives 1:1 (Basith did NOT length-match — faithful)
    n_target = (ds.Label == 1).sum()
    rng = random.Random(SEED)
    clean_sorted = sorted(clean)
    rng.shuffle(clean_sorted)
    sample = clean_sorted[:n_target]
    print(f"sampled {len(sample)} negatives (target {n_target}, 1:1 with positives)")

    # save the full cleaned pool (for reference / sensitivity)
    pool = pd.DataFrame({"Sequence": clean_sorted,
                         "Length": [len(s) for s in clean_sorted],
                         "Sources": ["|".join(sorted(sources[s])) for s in clean_sorted]})
    pool.to_csv("data/basith_negatives_pool.csv", index=False)

    # save the 966 sample in negatives.csv schema
    neg = pd.DataFrame({"Sequence": sample, "Label": 0,
                        "Length": [len(s) for s in sample], "NegType": "basith"})
    neg.to_csv("data/basith_negatives.csv", index=False)

    # length comparison vs our negatives and positives
    pos_len = ds[ds.Label == 1]["Length"]
    ourneg_len = ds[ds.Label == 0]["Length"]
    bl = neg["Length"]
    print(f"\nlength (mean): positives {pos_len.mean():.1f} | "
          f"our negatives {ourneg_len.mean():.1f} | Basith negatives {bl.mean():.1f}")
    print(f"length (median): positives {pos_len.median():.0f} | "
          f"our negatives {ourneg_len.median():.0f} | Basith negatives {bl.median():.0f}")
    print("\nsaved -> data/basith_negatives_pool.csv, data/basith_negatives.csv")


if __name__ == "__main__":
    main()
