
"""
Phase 3.6.2: Assemble the Basith-negative training set for the ablation.

Only the negatives change, with the training positives, the test set, the
features and the hyperparameters held constant, so the comparison isolates
the negative class.

  TRAIN -> the same 873 training positives, 873 Basith negatives (1:1).
  TEST -> the unchanged 178-row test set (93 positive, 45 hard, 40 soft).

A Basith negative resembling a test sequence would leak, so cd-hit-2d drops
any above 40% identity to the test set. Sequences under MIN_CDHIT_LEN bypass
it, since 40% identity means nothing at that length, so they get an exact-match
check instead.

INPUTS  basith_negatives.csv (966, from phase 3.6.1), dataset_split.csv
OUTPUTS  dataset_split_basith.csv (dataset_split.csv schema, in this folder)
REQUIREMENTS  cd-hit v4.8.1 on PATH; pip install pandas
"""

# Imports
import os
import random
import subprocess
import tempfile
import pandas as pd


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

# these mirror phase 1.3, so the ablation's leakage guard applies the same threshold
# over the same length range as the main split
IDENTITY = 0.40
WORD_SIZE = 2 # CD-HIT -n; must be 2 when -c is in [0.4, 0.5)
MIN_CDHIT_LEN = 11 # sequences shorter than this bypass CD-HIT
SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__)) # this ablation folder
BASITH_NEG = os.path.join(HERE, "basith_negatives.csv")
DATASET_SPLIT = os.path.join(HERE, "..", "data", "dataset_split.csv") # main pipeline split
OUT = os.path.join(HERE, "dataset_split_basith.csv") # stays beside the ablation


def write_fasta(seqs, path):
    # header ids are list positions (>0, >1, ...) so cd-hit-2d's output maps back
    # onto positions in seqs
    with open(path, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{i}\n{s}\n")


def cdhit2d_kept(ref_seqs, query_seqs, tmp):
    # cd-hit-2d writes out the query sequences that are not similar to anything in
    # the reference. ref is the test set and query is the Basith pool, so the
    # survivors are the negatives that are safe to train on
    fref, fqry = os.path.join(tmp, "r.fa"), os.path.join(tmp, "q.fa")
    fout = os.path.join(tmp, "q.out")
    write_fasta(ref_seqs, fref)
    write_fasta(query_seqs, fqry)

    # same flags as phase 1.3. -l 1 keeps short seqs, -d 0 writes full ids,
    # -M 0 lifts the memory cap, -T 0 uses every thread
    subprocess.run(["cd-hit-2d", "-i", fref, "-i2", fqry, "-o", fout,
                    "-c", str(IDENTITY), "-n", str(WORD_SIZE), "-l", "1",
                    "-d", "0", "-M", "0", "-T", "0"],
                   check=True, capture_output=True, text=True)

    # -o holds only the surviving queries, so their header ids are the kept positions
    kept = set()
    with open(fout) as fh:
        for line in fh:
            if line.startswith(">"):
                kept.add(int(line[1:].split()[0]))
    return kept


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    rng = random.Random(SEED)
    ds = pd.read_csv(DATASET_SPLIT)
    basith = pd.read_csv(BASITH_NEG)

    # the training positives and the whole test set carry over untouched, so the
    # training negatives are the only thing separating this dataset from the main one
    train_pos = ds[(ds.Split == "train") & (ds.Label == 1)].copy()
    test_rows = ds[ds.Split == "test"].copy()
    n_need = len(train_pos) # 1:1, as in the main split

    test_seqs = test_rows["Sequence"].tolist()
    cand = basith["Sequence"].tolist()
    rng.shuffle(cand) # seeded, so which 873 reach the slice below is fixed

    # long candidates are checked against the test set with cd-hit-2d. short ones
    # bypass it for the reason phase 1.3 gives, leaving them an exact-match check
    long_q = [s for s in cand if len(s) >= MIN_CDHIT_LEN]
    short_q = [s for s in cand if len(s) < MIN_CDHIT_LEN]
    with tempfile.TemporaryDirectory() as tmp: # CD-HIT scratch, auto-deleted
        long_ref = [s for s in test_seqs if len(s) >= MIN_CDHIT_LEN]
        kept_idx = cdhit2d_kept(long_ref, long_q, tmp)

    # sorting the kept positions restores the shuffled draw order, so the slice
    # further down stays a random sample. long candidates precede short ones, which
    # costs nothing here since only 3 of the 966 fall under MIN_CDHIT_LEN
    test_set = set(test_seqs)
    clean = [long_q[i] for i in sorted(kept_idx)]
    clean += [s for s in short_q if s not in test_set]

    # stop rather than pad from elsewhere. a train set below 1:1 would no longer be
    # comparable with the main run
    if len(clean) < n_need:
        raise SystemExit(f"only {len(clean)} clean Basith negs, need {n_need}")
    chosen = clean[:n_need]

    # nothing downstream reads Class or NegType, phase 3.3 takes Sequence, Label and
    # Split only. they are here so the file is readable beside dataset_split.csv
    neg_train = pd.DataFrame({
        "Sequence": chosen, "Label": 0, "Length": [len(s) for s in chosen],
        "Class": "negative", "NegType": "basith", "Split": "train"})

    cols = ["Sequence", "Label", "Length", "Class", "NegType", "Split"]
    out = pd.concat([train_pos[cols], neg_train, test_rows[cols]], ignore_index=True)
    out.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
