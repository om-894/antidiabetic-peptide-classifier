
"""
Negative-class ablation, step 2: assemble the Basith-negative training set, swapping
ONLY the negatives so everything else is held constant for a fair comparison.

  TRAIN = the SAME 873 training positives + 873 Basith negatives (1:1)
  TEST  = our UNCHANGED 178-row test set (so the Basith-trained model is scored on
          the identical test set as the dual-negative model)

Headline result: the Basith model's false-positive rate on the 45 hard test negatives
it never saw in training. Leakage guard: drop any Basith train negative >40% identical
to a test sequence (via cd-hit-2d); sequences <11 aa bypass CD-HIT.

OUTPUT  data/dataset_split_basith.csv  (same schema as data/dataset_split.csv)
"""

import os
import random
import subprocess
import tempfile

import pandas as pd

# CD-HIT settings + reproducible sampling. IDENTITY / WORD_SIZE / MIN_CDHIT_LEN
# mirror the main leakage-safe split, so this guard uses the same 40% threshold.
IDENTITY, WORD_SIZE, MIN_CDHIT_LEN, SEED = 0.40, 2, 11, 42

# Paths anchored to this folder, so the script runs from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
BASITH_NEG    = os.path.join(HERE, "basith_negatives.csv")             # from step 1 (this folder)
DATASET_SPLIT = os.path.join(HERE, "..", "data", "dataset_split.csv")  # main pipeline split
OUT           = os.path.join(HERE, "dataset_split_basith.csv")         # ablation dataset (this folder)


def write_fasta(seqs, path):
    # Write sequences to FASTA, using each one's list index as the header id
    # (>0, >1, ...) so CD-HIT's output can be mapped back to positions.
    with open(path, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{i}\n{s}\n")


def cdhit2d_kept(ref_seqs, query_seqs, tmp):
    """Indices of query_seqs NOT >IDENTITY to any ref sequence (long seqs only)."""
    # cd-hit-2d compares two sets and outputs the query sequences that are NOT
    # similar to anything in the reference. Here ref = test sequences, query =
    # Basith negatives -> the kept queries are the ones safe to use (no leakage).
    fref, fqry = os.path.join(tmp, "r.fa"), os.path.join(tmp, "q.fa")
    fout = os.path.join(tmp, "q.out")
    write_fasta(ref_seqs, fref)
    write_fasta(query_seqs, fqry)
    subprocess.run(["cd-hit-2d", "-i", fref, "-i2", fqry, "-o", fout,
                    "-c", str(IDENTITY), "-n", str(WORD_SIZE), "-l", "1",
                    "-d", "0", "-M", "0", "-T", "0"],
                   check=True, capture_output=True, text=True)
    # The output FASTA lists the kept queries; pull their integer ids (indices).
    kept = set()
    with open(fout) as fh:
        for line in fh:
            if line.startswith(">"):
                kept.add(int(line[1:].split()[0]))
    return kept


def main():
    rng = random.Random(SEED)
    ds = pd.read_csv(DATASET_SPLIT)          # main split (our positives + our negatives)
    basith = pd.read_csv(BASITH_NEG)         # the 966 Basith negatives from step 1

    # Keep training positives and whole test set unchanged; only the
    # training negatives get swapped for Basith's.
    train_pos = ds[(ds.Split == "train") & (ds.Label == 1)].copy()
    test_rows = ds[ds.Split == "test"].copy()                    # unchanged (93 pos + 85 neg)
    n_need = len(train_pos)                                      # Basith negs needed (1:1 with pos)
    print(f"train positives {n_need} | test rows {len(test_rows)} (unchanged)")

    test_seqs = test_rows["Sequence"].tolist()
    cand = basith["Sequence"].tolist()
    rng.shuffle(cand)                        # shuffle so the draw is random but seeded

    # Leakage guard: no Basith training negative may be >40% identical to a
    # test sequence. Long peptides go through cd-hit-2d; short ones (where a
    # 40% threshold is meaningless) are only exact-deduplicated vs the test set.
    long_q = [s for s in cand if len(s) >= MIN_CDHIT_LEN]
    short_q = [s for s in cand if len(s) < MIN_CDHIT_LEN]
    with tempfile.TemporaryDirectory() as tmp:                   # CD-HIT scratch, auto-deleted
        long_ref = [s for s in test_seqs if len(s) >= MIN_CDHIT_LEN]
        kept_idx = cdhit2d_kept(long_ref, long_q, tmp)           # long negs not close to test
    test_set = set(test_seqs)
    clean = [long_q[i] for i in sorted(kept_idx)]                # safe long negatives
    clean += [s for s in short_q if s not in test_set]          # safe short negatives (exact dedup)
    dropped = len(cand) - len(clean)
    print(f"leakage guard vs test: dropped {dropped} Basith negs >40% identical "
          f"to a test sequence -> {len(clean)} usable")

    if len(clean) < n_need:                  # not enough clean negatives to match the positives
        raise SystemExit(f"only {len(clean)} clean Basith negs, need {n_need}")
    chosen = clean[:n_need]                   # take the first 873 (1:1 with positives)

    # Assemble the new dataset: train positives + Basith train negatives +
    # unchanged test set, in the same schema as dataset_split.csv.
    neg_train = pd.DataFrame({
        "Sequence": chosen, "Label": 0, "Length": [len(s) for s in chosen],
        "Class": "negative", "NegType": "basith", "Split": "train"})
    pos_train = train_pos[["Sequence", "Label", "Length", "Class", "NegType", "Split"]]
    test_out = test_rows[["Sequence", "Label", "Length", "Class", "NegType", "Split"]]

    out = pd.concat([pos_train, neg_train, test_out], ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(out)} rows)")
    print(pd.crosstab(out["Split"], out["Label"]).to_string())  # train/test x pos/neg counts
    print(f"\nBasith train-neg length mean {neg_train.Length.mean():.1f} "
          f"vs train-pos {pos_train.Length.mean():.1f}")


if __name__ == "__main__":
    main()
