"""
Phase 1.3: Leakage-safe train/test split (group-aware, keeps all data).

CD-HIT clusters positives and negatives together at 40% identity. Whole clusters
go to train or test so near-duplicates can't straddle the split. Clustering is
greedy and misses some pairs. cd-hit-2d re-checks test against train afterwards
then moves anything leaked back to train until it reports none. That only moves
test -> train and never refills, so the realised test fraction lands under
TEST_FRACTION (15% target -> 9.2%, n = 178).

Nothing is deleted for redundancy. Near-duplicates fall in the same split and
inside train they're useful hard cases. Conventional 40% redundancy reduction
would delete 782/1932 sequences (~40% of the data) since many ADPs are
overlapping insulin-derived fragments. The only deletion is a sequence appearing
as both positive and negative, an integrity guard that removes nothing here.

Short sequences (< MIN_CDHIT_LEN) bypass CD-HIT since 40% identity isn't
meaningful there. They get a stratified random split instead.

INPUTS  positives_ADP.csv (966 positives), negatives.csv (966 negatives)
OUTPUTS  dataset_split.csv (Sequence, Label, Length, Class, NegType, Split)
REQUIREMENTS  cd-hit v4.8.1 on PATH; pip install pandas
"""

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
OUTPUT_CSV = "data/dataset_split.csv"

IDENTITY = 0.40 # identity threshold: >40% similar = "too close" (leakage)
WORD_SIZE = 2 # CD-HIT -n; must be 2 when -c is in [0.4, 0.5)
MIN_CDHIT_LEN = 11 # sequences shorter than this, bypass CD-HIT
TEST_FRACTION = 0.15 # aim for 15% of sequences in the test set
SEED = 42 # reproducible shuffling/splitting

CD_HIT, CD_HIT_2D = "cd-hit", "cd-hit-2d"

# Check that CD-HIT and CD-HIT-2D are installed and on PATH
def check_tools():
    for t in (CD_HIT, CD_HIT_2D):
        if shutil.which(t) is None:
            sys.exit(f"'{t}' not found on PATH. Install CD-HIT "
                     f"(brew install cd-hit  /  conda install -c bioconda cd-hit).")


def write_fasta(df, path):
    # Write sequences to FASTA, using each row's DataFrame index as the >header
    # ID - so CD-HIT's output can be mapped straight back to the original rows.
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
    # cd-hit-2d compares two sets and keeps the query sequences that are not
    # similar to anything in the reference. Used to check test-vs-train leakage
    # (ref = train, query = test): the returned ids are the "safe" test rows.
    if len(df_ref) == 0 or len(df_query) == 0:
        return set(df_query.index)
    
    fref, fqry = os.path.join(tmp, "r.fasta"), os.path.join(tmp, "q.fasta") # write FASTA files for the reference and query sets
    fout = os.path.join(tmp, "q.out")
    write_fasta(df_ref, fref); write_fasta(df_query, fqry) # Run CD-HIT-2D to compare the two sets and keep only the query sequences 
                                                           # that are not similar to any reference sequence
    
    # same flags as cd_hit_clusters but cd-hit-2d takes two inputs, -i (reference)
    # and -i2 (query)
    subprocess.run([CD_HIT_2D, "-i", fref, "-i2", fqry, "-o", fout,
                    "-c", str(IDENTITY), "-n", str(WORD_SIZE), "-l", "1",
                    "-d", "0", "-M", "0", "-T", "0"],
                   check=True, capture_output=True, text=True)
    return fasta_ids(fout) # -o holds only the query seqs that survived, so these are the safe rows


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    check_tools() # stop early if CD-HIT isn't installed
    rng = random.Random(SEED) # for a reproducible split

    # Load both classes, tagging each with Class (and NegType for negatives).
    pos = pd.read_csv(POSITIVES_CSV); pos["Class"] = "positive"; pos["NegType"] = pd.NA
    neg = pd.read_csv(NEGATIVES_CSV); neg["Class"] = "negative"
    if "NegType" not in neg.columns:
        neg["NegType"] = pd.NA

    # Integrity guard, since a sequence can't be both a positive and a negative.
    # Drops any negative that exactly matches a positive (none expected here).
    posset = set(pos["Sequence"])
    neg = neg[~neg["Sequence"].isin(posset)].reset_index(drop=True)

    # Combine into one frame. ignore_index makes the row index run 0..n-1, which is
    # what write_fasta puts in the headers, so CD-HIT ids stay valid .loc labels.
    data = pd.concat([pos, neg], ignore_index=True)

    with tempfile.TemporaryDirectory() as tmp: # CD-HIT scratch files, auto-deleted
        data["Split"] = "train" # every sequence starts in train
        long = data[data.Length >= MIN_CDHIT_LEN]  # CD-HIT handles these
        short = data[data.Length < MIN_CDHIT_LEN]  # too short for 40% identity

        # Cluster the long sequences (positives + negatives together) and assign
        # whole clusters to test until TEST_FRACTION is reached. Keeping a cluster
        # intact is what stops near-duplicates leaking across the train/test line.
        # Clusters are never split, so the last one added overshoots the target.
        clusters = cd_hit_clusters(long, tmp)
        rng.shuffle(clusters)
        target = round(len(long) * TEST_FRACTION)
        test_ids, n = [], 0
        for c in clusters:
            if n >= target:
                break
            test_ids += c; n += len(c)
        data.loc[test_ids, "Split"] = "test" # ids came from the FASTA headers, so they're valid .loc labels

        # Short sequences get a stratified random split per class instead, since
        # 40% identity is meaningless here.
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

        # Verify leakage is actually zero before writing out. The loop above exits
        # on either a clean pass or 15 attempts, so this catches the second case.
        tr = data[(data.Split == "train") & (data.Length >= MIN_CDHIT_LEN)]
        te = data[(data.Split == "test") & (data.Length >= MIN_CDHIT_LEN)]
        residual = len(te) - len(cd_hit_2d_kept(tr, te, tmp)) if len(te) else 0
        assert residual == 0, f"{residual} test sequences still >40% identical to train"

    # Write out the split to CSV, keeping only the relevant columns.
    out = data[["Sequence", "Label", "Length", "Class", "NegType", "Split"]]
    out.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
