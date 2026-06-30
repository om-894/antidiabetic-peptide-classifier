
"""
Negative-class ablation, step 1: build Basith et al.'s (ADP-Fuse, 2023) conventional
non-ADP negatives as a comparator for the dual-negative class. Only the negatives
change; positives/features/architecture/hyperparameters stay constant.

ADP-Fuse is a two-layer model: Layer 1 = ADP vs non-ADP, Layer 2 = type-1 vs type-2
diabetes. Only Layer 1's negatives are true non-ADPs (random peptides + antimicrobial/
anticancer/antihypertensive, reduced with CD-HIT 0.6). Layer 2's "negatives" are T2D
ADPs, so they are excluded. Pool = negatives from Layer1_training + L1_Ind.

OUTPUTS  basith_negatives_pool.csv  (full cleaned pool, in this folder)
         basith_negatives.csv       (966 sampled, negatives.csv schema)
"""

# Imports
import random
from collections import defaultdict
import pandas as pd


import os
HERE = os.path.dirname(os.path.abspath(__file__))          # this ablation folder
RAW  = os.path.join(HERE, "basith_raw")                    # inputs live beside this script
DATASET_SPLIT = os.path.join(HERE, "..", "data", "dataset_split.csv")  # main pipeline split
POOL_OUT = os.path.join(HERE, "basith_negatives_pool.csv") # outputs stay in this folder
NEG_OUT  = os.path.join(HERE, "basith_negatives.csv")
FILES = ["Layer1_training.txt", "L1_Ind.txt"]    # Layer 1 only = the true non-ADPs
SEED = 42                                         # reproducible sampling
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")         # the 20 standard amino acids


def parse_fasta(path):
    # Read a FASTA-style file and return only the negative sequences. Each ">"
    # header says whether the lines below it are a Positive or a Negative; we
    # track that with `cur` and keep the sequence only when cur == "neg".
    neg, cur = [], None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            cur = "pos" if "Positive" in line else "neg"
        elif cur == "neg":
            neg.append(line)
    return neg


def is_standard(s):
    # Keep only sequences made of the 20 standard AA (drops anything
    # with non-standard residues like X/B/Z/U).
    return len(s) > 0 and set(s).issubset(STANDARD_AA)


def main():
    # Collect the Layer-1 negatives across both files. `sources` records which
    # file(s) each sequence came from (provenance); the keys form the unique set.
    sources = defaultdict(set)
    for f in FILES:
        for s in parse_fasta(os.path.join(RAW, f)):
            sources[s].add(f.replace(".txt", ""))
    neg_all = {s for s in sources if is_standard(s)}     # unique, standard-AA negatives
    print(f"Basith Layer-1 non-ADP negatives (unique, standard AA): {len(neg_all)}")

    # Integrity guard: a sequence can't be both a negative and one of my
    # positives/negatives (that would be a label contradiction). Remove any
    # overlaps. (With Layer-1-only, the positive overlap is now 0.)
    ds = pd.read_csv(DATASET_SPLIT)
    user_pos = set(ds[ds.Label == 1]["Sequence"])
    user_neg = set(ds[ds.Label == 0]["Sequence"])
    print(f"  removed {len(neg_all & user_pos)} matching current positives, "
          f"{len(neg_all & user_neg)} matching our negatives")
    clean = sorted(neg_all - user_pos - user_neg)        # sorted -> deterministic order

    # Sample 966 negatives (1:1 with the positives). Shuffle with a fixed seed so
    # the draw is random but reproducible. no length-matching here — that is
    # faithful to Basith (the resulting length mismatch is a discussion point).
    n_target = (ds.Label == 1).sum()
    rng = random.Random(SEED)
    rng.shuffle(clean)
    sample = clean[:n_target]
    print(f"clean pool {len(clean)} -> sampled {len(sample)} (1:1 with positives)")

    # Save the full cleaned pool (for reference) and the 966 samples. The sample
    # uses the same schema as data/negatives.csv (Label=0, NegType tag) so it can
    # drop straight into the split step.
    pd.DataFrame({"Sequence": clean,
                  "Length": [len(s) for s in clean],
                  "Sources": ["|".join(sorted(sources[s])) for s in clean]}
                 ).to_csv(POOL_OUT, index=False)
    neg = pd.DataFrame({"Sequence": sample, "Label": 0,
                        "Length": [len(s) for s in sample], "NegType": "basith"})
    neg.to_csv(NEG_OUT, index=False)

    # Length sanity print: shows Basith's negatives are longer than the positives
    # (the length confound), since Basith did not length-match.
    pos_len = ds[ds.Label == 1]["Length"]
    print(f"\nlength (mean): positives {pos_len.mean():.1f} | "
          f"Basith negatives {neg.Length.mean():.1f}")
    print("saved -> " + POOL_OUT + ", " + NEG_OUT)


if __name__ == "__main__":
    main()
