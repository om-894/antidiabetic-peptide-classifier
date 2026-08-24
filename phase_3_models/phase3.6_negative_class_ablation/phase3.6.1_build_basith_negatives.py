
"""
Phase 3.6.1: Build ADP-Fuse (Basith et al., 2023) conventional non-ADP negatives
as a comparator for the dual-negative class. Only the negatives change, with
positives, features, architecture and hyperparameters held constant.

ADP-Fuse is two-layer, Layer 1 separating ADP from non-ADP and Layer 2 separating
type-1 from type-2 diabetes. Only Layer 1's negatives are true non-ADPs (random
peptides plus antimicrobial, anticancer and antihypertensive, reduced with CD-HIT
0.6). Layer 2's "negatives" are T2D ADPs, so they are excluded. The pool is the
negatives from Layer1_training and L1_Ind.

The 966 are sampled without length matching, deliberately, since published sets
are not matched and matching here would flatter the comparator.

INPUTS  basith_raw/Layer1_training.txt, basith_raw/L1_Ind.txt, dataset_split.csv
OUTPUTS  basith_negatives_pool.csv (full cleaned pool, in this folder)
         basith_negatives.csv (966 sampled, negatives.csv schema)
"""

# Imports
import os
import random
from collections import defaultdict

import pandas as pd


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__)) # this ablation folder
RAW = os.path.join(HERE, "basith_raw") # inputs live beside this script
DATASET_SPLIT = "data/dataset_split.csv" # repo root, as in every other phase script
POOL_OUT = os.path.join(HERE, "basith_negatives_pool.csv") # outputs stay in this folder
NEG_OUT = os.path.join(HERE, "basith_negatives.csv")

FILES = ["Layer1_training.txt", "L1_Ind.txt"] # Layer 1 only, the true non-ADPs
SEED = 42 # reproducible sampling
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY") # the 20 standard amino acids


def parse_fasta(path):
    # read a FASTA-style file and return only the negative sequences. every header
    # in these files is either >Positive_N or >Negative_N, so cur tracks which
    # block we are inside and the lines below are kept only when it says neg
    neg, cur = [], None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                cur = "pos" if "Positive" in line else "neg"
            elif cur == "neg":
                neg.append(line)
    return neg


def is_standard(s):
    # keep only sequences made of the 20 standard AA, dropping anything with
    # non-standard residues like X/B/Z/U
    return len(s) > 0 and set(s).issubset(STANDARD_AA)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    # collect the Layer-1 negatives across both files. sources records which file
    # each sequence came from. its keys are the unique set
    sources = defaultdict(set)
    for f in FILES:
        for s in parse_fasta(os.path.join(RAW, f)):
            sources[s].add(f.replace(".txt", ""))
    neg_all = {s for s in sources if is_standard(s)}

    # integrity guard, since a sequence can't be both a Basith negative and one of
    # the dual-negative dataset's own sequences. taking Layer 1 alone leaves no
    # overlap with the positives
    ds = pd.read_csv(DATASET_SPLIT)
    dual_pos = set(ds[ds.Label == 1]["Sequence"])
    dual_neg = set(ds[ds.Label == 0]["Sequence"])

    # sorted first so the shuffle below is a deterministic permutation. set order
    # is not stable across runs, since Python randomises string hashing
    clean = sorted(neg_all - dual_pos - dual_neg)

    # sample 966 negatives, 1:1 with the positives. no length matching, which is
    # faithful to Basith. the resulting length gap is the point of section 3.2
    n_target = (ds.Label == 1).sum()
    rng = random.Random(SEED)
    rng.shuffle(clean)
    sample = clean[:n_target]

    # the full cleaned pool for reference, then the 966. the sample uses the same
    # schema as data/negatives.csv so it drops straight into the split step
    # note the pool is written post-shuffle, so its row order is not alphabetical
    pd.DataFrame({"Sequence": clean,
                  "Length": [len(s) for s in clean],
                  "Sources": ["|".join(sorted(sources[s])) for s in clean]}
                 ).to_csv(POOL_OUT, index=False)
    pd.DataFrame({"Sequence": sample, "Label": 0,
                  "Length": [len(s) for s in sample], "NegType": "basith"}
                 ).to_csv(NEG_OUT, index=False)


if __name__ == "__main__":
    main()
