
"""
Phase 6.1 (benchmark input): export hard negatives for BertADP's inference script.

Experiment 2 of the external benchmark: run the SOTA BertADP on hard negatives and
measure its false-positive rate. Trained on weak negatives, BertADP should over-predict
ADP on hard negatives - the section 4 Basith-neg failure (93% FPR) vs the dual-negative i built (47%).

INPUT   data/dataset_split.csv
OUTPUT  benchmark/bertadp_hardneg_test.csv   45 test hard negatives (matched to §4)
        benchmark/bertadp_hardneg_all.csv    all 328 hard negatives (larger-n FPR)

REQUIREMENTS  pip install pandas
"""

import os
import pandas as pd

OUT_DIR = "benchmark"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv("data/dataset_split.csv")
    hard = df[(df.Label == 0) & (df.NegType == "hard")] # Swiss-Prot fragment negatives

    test = hard[hard.Split == "test"][["Sequence", "Label"]] # matched to the S4 45-row FPR
    test.to_csv(f"{OUT_DIR}/bertadp_hardneg_test.csv", index=False)
    hard[["Sequence", "Label"]].to_csv(f"{OUT_DIR}/bertadp_hardneg_all.csv", index=False)

    # soft negatives (antimicrobial peptides) - BertADP should handle these, unlike the hard ones
    soft = df[(df.Label == 0) & (df.NegType == "soft") & (df.Split == "test")][["Sequence", "Label"]]
    soft.to_csv(f"{OUT_DIR}/bertadp_softneg_test.csv", index=False)

    # validity control: none of the benchmarked peptides may appear in BertADP's training data
    XIE = os.path.expanduser("~/BertADP/data/allData.csv")
    if os.path.exists(XIE):
        xie = set(pd.read_csv(XIE, keep_default_na=False).Sequence.astype(str))
        rows = [{"set": name, "n": len(set(sub.Sequence.astype(str))),
                 "n_in_bertadp_training": len(set(sub.Sequence.astype(str)) & xie)}
                for name, sub in [("hard_test", test), ("hard_all", hard), ("soft_test", soft)]]
        pd.DataFrame(rows).to_csv(f"{OUT_DIR}/bertadp_overlap_check.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print(f"[warn] {XIE} not found, overlap control skipped")

    # report the counts and where they were written
    print(f"test soft negatives: {len(soft)} -> {OUT_DIR}/bertadp_softneg_test.csv")
    print(f"test hard negatives: {len(test)} -> {OUT_DIR}/bertadp_hardneg_test.csv")
    print(f"all  hard negatives: {len(hard)} -> {OUT_DIR}/bertadp_hardneg_all.csv")
    print("these are all non-ADPs (Label 0); any BertADP Prediction=1 is a false positive")


if __name__ == "__main__":
    main()