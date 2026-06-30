#!/usr/bin/env python3
"""
ablation_build_dataset.py
=========================
Negative-class ablation, step 2: assemble the Basith-negative training set that
mirrors our dual-negative setup but swaps ONLY the negatives.

Design (holds everything else constant for a fair cross-evaluation):
  TRAIN = the SAME 873 training positives  +  873 Basith negatives (1:1)
  TEST  = our UNCHANGED 178-row test set (93 pos + 85 dual negatives, 45 hard)

So a Basith-trained model is evaluated on the identical test set our
dual-negative model used (esm2_dora_predictions.npz). The key result is its
false-positive rate on the 45 hard negatives it never saw in training.

Leakage guard (same philosophy as phase1_leakage_safe_split.py): no Basith
training negative may be >40% identical to any TEST sequence. Sequences < 11 aa
bypass CD-HIT (40% identity is not meaningful there) and are kept if not exact
duplicates of a test sequence.

OUTPUT  data/dataset_split_basith.csv   (same schema as data/dataset_split.csv)
"""

import os
import random
import subprocess
import tempfile

import pandas as pd

IDENTITY, WORD_SIZE, MIN_CDHIT_LEN, SEED = 0.40, 2, 11, 42


def write_fasta(seqs, path):
    with open(path, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{i}\n{s}\n")


def cdhit2d_kept(ref_seqs, query_seqs, tmp):
    """Indices of query_seqs NOT >IDENTITY to any ref sequence (long seqs only)."""
    fref, fqry = os.path.join(tmp, "r.fa"), os.path.join(tmp, "q.fa")
    fout = os.path.join(tmp, "q.out")
    write_fasta(ref_seqs, fref)
    write_fasta(query_seqs, fqry)
    subprocess.run(["cd-hit-2d", "-i", fref, "-i2", fqry, "-o", fout,
                    "-c", str(IDENTITY), "-n", str(WORD_SIZE), "-l", "1",
                    "-d", "0", "-M", "0", "-T", "0"],
                   check=True, capture_output=True, text=True)
    kept = set()
    with open(fout) as fh:
        for line in fh:
            if line.startswith(">"):
                kept.add(int(line[1:].split()[0]))
    return kept


def main():
    rng = random.Random(SEED)
    ds = pd.read_csv("data/dataset_split.csv")
    basith = pd.read_csv("data/basith_negatives.csv")

    train_pos = ds[(ds.Split == "train") & (ds.Label == 1)].copy()
    test_rows = ds[ds.Split == "test"].copy()                     # unchanged
    n_need = len(train_pos)
    print(f"train positives {n_need} | test rows {len(test_rows)} (unchanged)")

    test_seqs = test_rows["Sequence"].tolist()
    cand = basith["Sequence"].tolist()
    rng.shuffle(cand)

    # leakage guard vs the TEST set
    long_q = [s for s in cand if len(s) >= MIN_CDHIT_LEN]
    short_q = [s for s in cand if len(s) < MIN_CDHIT_LEN]
    with tempfile.TemporaryDirectory() as tmp:
        long_ref = [s for s in test_seqs if len(s) >= MIN_CDHIT_LEN]
        kept_idx = cdhit2d_kept(long_ref, long_q, tmp)
    test_set = set(test_seqs)
    clean = [long_q[i] for i in sorted(kept_idx)]
    clean += [s for s in short_q if s not in test_set]          # short: exact-dedup only
    dropped = len(cand) - len(clean)
    print(f"leakage guard vs test: dropped {dropped} Basith negs >40% identical "
          f"to a test sequence -> {len(clean)} usable")

    if len(clean) < n_need:
        raise SystemExit(f"only {len(clean)} clean Basith negs, need {n_need}")
    chosen = clean[:n_need]

    neg_train = pd.DataFrame({
        "Sequence": chosen, "Label": 0, "Length": [len(s) for s in chosen],
        "Class": "negative", "NegType": "basith", "Split": "train"})
    pos_train = train_pos[["Sequence", "Label", "Length", "Class", "NegType", "Split"]]
    test_out = test_rows[["Sequence", "Label", "Length", "Class", "NegType", "Split"]]

    out = pd.concat([pos_train, neg_train, test_out], ignore_index=True)
    out.to_csv("data/dataset_split_basith.csv", index=False)
    print(f"\nwrote data/dataset_split_basith.csv  ({len(out)} rows)")
    print(pd.crosstab(out["Split"], out["Label"]).to_string())
    print(f"\nBasith train-neg length mean {neg_train.Length.mean():.1f} "
          f"vs train-pos {pos_train.Length.mean():.1f}  "
          f"(length confound: Basith negs are longer)")


if __name__ == "__main__":
    main()
