
"""
Phase 1.3: Leakage-safe train/test split (group-aware, keeps all data).

CD-HIT clusters the full positive+negative set at 40% identity; whole clusters
go to train OR test, so no test sequence is >40% identical to any train one.
Nothing is deleted for redundancy — near-duplicate pairs fall in the same split,
and within train they're useful hard cases.

Only deletion: exact contradictions (a sequence appearing as both positive and
negative) — an integrity guard that removes nothing here.

Short sequences (< MIN_CDHIT_LEN) bypass CD-HIT (40% identity isn't meaningful
there) and are split by stratified random assignment.

REQUIREMENTS  cd-hit on PATH; pip install pandas
"""
# used CD-HIT 4.8.1 to put in methods.

# Imports
import os
import random
import shutil
import subprocess
import sys
import tempfile
import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
POSITIVES_CSV = "data/positives_ADP.csv"
NEGATIVES_CSV = "data/negatives.csv"
OUTPUT_CSV    = "data/dataset_split.csv"

IDENTITY      = 0.40       # identity threshold: >40% similar = "too close" (leakage)
WORD_SIZE     = 2          # CD-HIT -n; must be 2 when -c is in [0.4, 0.5)
MIN_CDHIT_LEN = 11         # sequences shorter than this, bypass CD-HIT
TEST_FRACTION = 0.15       # aim for 15% of sequences in the test set
SEED          = 42         # reproducible shuffling/splitting

CD_HIT, CD_HIT_2D = "cd-hit", "cd-hit-2d"

# Check that CD-HIT and CD-HIT-2D are installed and on PATH
def check_tools():
    for t in (CD_HIT, CD_HIT_2D):
        if shutil.which(t) is None:
            sys.exit(f"'{t}' not found on PATH. Install CD-HIT "
                     f"(brew install cd-hit  /  conda install -c bioconda cd-hit).")


def write_fasta(df, path):
    # Write sequences to FASTA, using each row's DataFrame index as the >header
    # ID — so CD-HIT's output can be mapped straight back to the original rows.
    with open(path, "w") as fh:
        for idx, seq in zip(df.index, df["Sequence"]):
            fh.write(f">{idx}\n{seq}\n")


def fasta_ids(path):
    # Return the set of integer IDs (row indices) of the sequences in a FASTA.
    out = set()
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                out.add(int(line[1:].split()[0]))
    return out


def parse_clstr(path):
    # Parse a CD-HIT .clstr file into a list of clusters, each a list of the row
    # indices it contains. ">Cluster" starts a new cluster; member lines look
    # like "0  12aa, >37... *", so pull out the ">37" id.
    clusters, cur = [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith(">Cluster"):
                cur = []; clusters.append(cur)
            else:
                cur.append(int(line.split(">")[1].split("...")[0]))
    return clusters


def cd_hit_clusters(df, tmp):
    # Cluster every sequence in df at IDENTITY; return the clusters (index lists).
    if len(df) == 0:
        return []
    fin, fout = os.path.join(tmp, "c.fasta"), os.path.join(tmp, "c.out")
    write_fasta(df, fin)

    # Run CD-HIT on df and return its clusters (as lists of row indices).
    # -c identity | -n word size | -l 1 keep short seqs | -d 0 full ids in .clstr
    # -M 0 no memory cap | -T 0 use all threads
    subprocess.run([CD_HIT, "-i", fin, "-o", fout, "-c", str(IDENTITY),
                    "-n", str(WORD_SIZE), "-l", "1", "-d", "0", "-M", "0", "-T", "0"],
                   check=True, capture_output=True, text=True)
    return parse_clstr(fout + ".clstr")


def cd_hit_2d_kept(df_ref, df_query, tmp):
    """Index set of df_query NOT >IDENTITY to any sequence in df_ref."""
    # cd-hit-2d compares two sets and keeps the query sequences that are NOT
    # similar to anything in the reference. Used to check test-vs-train leakage
    # (ref = train, query = test): the returned ids are the "safe" test rows.
    if len(df_ref) == 0 or len(df_query) == 0:
        return set(df_query.index)
    
    fref, fqry = os.path.join(tmp, "r.fasta"), os.path.join(tmp, "q.fasta") # write FASTA files for the reference and query sets
    fout = os.path.join(tmp, "q.out")
    write_fasta(df_ref, fref); write_fasta(df_query, fqry) # Run CD-HIT-2D to compare the two sets and keep only the query sequences 
                                                           # that are not similar to any reference sequence
    
    # same flags as cd_hit_clusters, but cd-hit-2d with two inputs:
    # -i (reference set) and -i2 (query set)
    subprocess.run([CD_HIT_2D, "-i", fref, "-i2", fqry, "-o", fout,
                    "-c", str(IDENTITY), "-n", str(WORD_SIZE), "-l", "1",
                    "-d", "0", "-M", "0", "-T", "0"],
                   check=True, capture_output=True, text=True)
    return fasta_ids(fout)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    check_tools() # stop early if CD-HIT isn't installed
    rng = random.Random(SEED) # for a reproducable split

    # Load both classes, tagging each with Class (and NegType for negatives).
    pos = pd.read_csv(POSITIVES_CSV); pos["Class"] = "positive"; pos["NegType"] = pd.NA
    neg = pd.read_csv(NEGATIVES_CSV); neg["Class"] = "negative"
    if "NegType" not in neg.columns:
        neg["NegType"] = pd.NA
    print(f"start: {len(pos)} positives, {len(neg)} negatives\n")

    # Integrity guard: a sequence can't be both a positive and a negative, so
    # drop any negative that exactly matches a positive (none expected here).
    posset = set(pos["Sequence"])
    before = len(neg)
    neg = neg[~neg["Sequence"].isin(posset)].reset_index(drop=True)
    print(f"integrity: removed {before - len(neg)} negatives identical to a "
          f"positive (exact contradictions)\n")

    # Combine into one frame; the row index now serves as each sequence's ID.
    data = pd.concat([pos, neg], ignore_index=True)

    with tempfile.TemporaryDirectory() as tmp: # CD-HIT scratch files, auto-deleted
        print("group-aware split (no test sequence >40% identical to train)")
        data["Split"] = "train" # every sequence starts in train
        long = data[data.Length >= MIN_CDHIT_LEN]  # CD-HIT handles these
        short = data[data.Length < MIN_CDHIT_LEN]  # too short for 40% identity

        # Cluster the long sequences (positives + negatives together) and assign
        # whole clusters to test until TEST_FRACTION is reached. Keeping a cluster
        # intact is what stops near-duplicates leaking across the train/test line.
        clusters = cd_hit_clusters(long, tmp)
        rng.shuffle(clusters)
        target = round(len(long) * TEST_FRACTION)
        test_ids, n = [], 0
        for c in clusters:
            if n >= target:
                break
            test_ids += c; n += len(c)
        data.loc[test_ids, "Split"] = "test"

        # Short sequences: 40% identity is meaningless here, so just do a
        # stratified random split per class instead.
        for cls in ("positive", "negative"):
            ids = list(short.index[short.Class == cls])
            rng.shuffle(ids)
            data.loc[ids[:round(len(ids) * TEST_FRACTION)], "Split"] = "test"

        # CD-HIT clustering is heuristic, so a few test sequences may still be
        # >40% identical to a train one. Iteratively move any leaked sequences back
        # to train (cd-hit-2d finds them) until none remain (max 15 passes).
        for _ in range(15):
            tr = data[(data.Split == "train") & (data.Length >= MIN_CDHIT_LEN)]
            te = data[(data.Split == "test") & (data.Length >= MIN_CDHIT_LEN)]
            if len(te) == 0:
                break
            leak = set(te.index) - cd_hit_2d_kept(tr, te, tmp) # test rows too close to train
            if not leak:
                break
            data.loc[sorted(leak), "Split"] = "train"

        # Verify leakage is actually zero before writing out.
        tr = data[(data.Split == "train") & (data.Length >= MIN_CDHIT_LEN)]
        te = data[(data.Split == "test") & (data.Length >= MIN_CDHIT_LEN)]
        residual = len(te) - len(cd_hit_2d_kept(tr, te, tmp)) if len(te) else 0
        n_train = (data.Split == "train").sum()
        n_test  = (data.Split == "test").sum()

        # Final summary of the split and leakage check.
        print(f"  split: {n_train} train / {n_test} test")
        print(f"  leakage check: {residual} test sequences too close to train (want 0)")

    # Write out the split to CSV, keeping only the relevant columns.
    out = data[["Sequence", "Label", "Length", "Class", "NegType", "Split"]]
    out.to_csv(OUTPUT_CSV, index=False)

    # Final summary of the split and leakage check.
    n_pos = (out.Class == "positive").sum()
    n_neg = (out.Class == "negative").sum()
    pos_mean = data[data.Class == "positive"]["Length"].mean()
    neg_mean = data[data.Class == "negative"]["Length"].mean()
    print(f"Saved {len(out)} sequences ({n_pos} positives, {n_neg} negatives) to {OUTPUT_CSV}")
    print(f"  mean length: positives {pos_mean:.1f}, negatives {neg_mean:.1f}")
    print(pd.crosstab(out.Split, out.Class)) # .crosstab shows the counts of positives and negatives in train/test splits


if __name__ == "__main__":
    main()
