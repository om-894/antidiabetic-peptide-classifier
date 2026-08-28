
"""
Phase 6.1: Export the hard negatives for BertADP's inference script.

BertADP is the published state of the art and draws its negatives from the same
curated peptide databases the ablation control does, so it should fail on proteome
fragments the same way. Every peptide written here is a true non-ADP, so any
Prediction = 1 that comes back is a false positive.

Three sets go out. The 45 test hard negatives match the three-way comparison, all
328 give the same rate at a larger n, and the test soft negatives are the control,
since antimicrobial peptides are the class BertADP's own negatives come from.

INPUTS  dataset_split.csv (1932 rows with Label, NegType and Split)
OUTPUTS  bertadp_hardneg_test.csv (45 test hard negatives)
         bertadp_hardneg_all.csv (all 328 hard negatives)
         bertadp_softneg_test.csv (40 test soft negatives, the control)
         bertadp_overlap_check.csv (overlap against Xie's released dataset)
REQUIREMENTS  pip install pandas. Xie's allData.csv for the overlap control
"""

# Imports
import os
import pandas as pd


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
SPLIT_CSV = "data/dataset_split.csv"
OUT_DIR = "benchmark"
OUT_HARD_TEST = "benchmark/bertadp_hardneg_test.csv"
OUT_HARD_ALL = "benchmark/bertadp_hardneg_all.csv"
OUT_SOFT_TEST = "benchmark/bertadp_softneg_test.csv"
OUT_OVERLAP = "benchmark/bertadp_overlap_check.csv"

# Xie's released dataset, cloned beside their inference script. the control is skipped
# rather than fatal, since the benchmark itself runs without it
XIE_CSV = os.path.expanduser("~/BertADP/data/allData.csv")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(SPLIT_CSV)

    # hard negatives are the Swiss-Prot fragments, the sequence space a real screen returns
    hard = df[(df.Label == 0) & (df.NegType == "hard")]
    test = hard[hard.Split == "test"][["Sequence", "Label"]] # the 45 held out from every model
    test.to_csv(OUT_HARD_TEST, index=False)
    hard[["Sequence", "Label"]].to_csv(OUT_HARD_ALL, index=False)

    # the soft negatives are the comparison. BertADP trained on this kind of peptide, so a
    # low error rate here beside a high one on the hard negatives locates the failure
    soft = df[(df.Label == 0) & (df.NegType == "soft") & (df.Split == "test")][["Sequence", "Label"]]
    soft.to_csv(OUT_SOFT_TEST, index=False)

    # the benchmark assumes BertADP has not seen these peptides, so the overlap against its
    # released dataset is measured rather than assumed
    if os.path.exists(XIE_CSV):
        xie = set(pd.read_csv(XIE_CSV, keep_default_na=False).Sequence.astype(str))
        rows = [{"set": name, "n": len(set(sub.Sequence.astype(str))),
                 "n_in_bertadp_training": len(set(sub.Sequence.astype(str)) & xie)}
                for name, sub in [("hard_test", test), ("hard_all", hard), ("soft_test", soft)]]
        pd.DataFrame(rows).to_csv(OUT_OVERLAP, index=False)
    else:
        print(f"[warn] {XIE_CSV} not found, overlap control skipped")

    print(f"{len(test)} test hard, {len(hard)} all hard, {len(soft)} test soft -> {OUT_DIR}/")


if __name__ == "__main__":
    main()