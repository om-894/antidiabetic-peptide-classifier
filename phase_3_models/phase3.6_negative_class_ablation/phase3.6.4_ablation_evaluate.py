
"""
Phase 3.6.4: Cross-evaluate both ESM-2 runs on the identical 178-row test set.

The two models differ only in their training negatives, so any gap between them
is the negative class. The headline is the false-positive rate on the 45 hard
test negatives, which the Basith model never saw in training.

  hard -> Swiss-Prot fragments, the sequence space a real screen returns.
  soft -> DBAASP antimicrobial peptides, the generic-bioactivity control.

The Basith pool is not length-matched to the positives, so its errors could be
a length shortcut rather than a negative-class effect. The length control
scores length as a predictor of which negatives each model wrongly accepts,
over test negatives only so the true label cannot leak into the AUC.

INPUTS  dataset_split.csv, dataset_split_basith.csv, both prediction npz files
OUTPUTS  phase3_6_ablation.csv (per-model metrics)
         phase3_6_train_lengths.csv (mean training length per class)
         phase3_6_length_control.csv (length as a predictor of false positives)
"""


# Imports
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__)) # this ablation folder
DATASET_SPLIT = os.path.join(HERE, "..", "data", "dataset_split.csv") # main pipeline split
BASITH_SPLIT = os.path.join(HERE, "dataset_split_basith.csv")
DUAL_NPZ = os.path.join(HERE, "..", "predictions", "esm2_dora_predictions.npz") # main run
BASITH_NPZ = os.path.join(HERE, "..", "predictions", "esm2_dora_basith_predictions.npz") # ablation run
OUT = os.path.join(HERE, "..", "results") # shared results folder, one level up

THRESHOLD = 0.5 # the operating point every count below is taken at


def test_probs(path, expected):
    """Test-row probabilities from one npz, in `expected` sequence order."""
    # the npz stores train rows then test rows, so the tail is the test set. every
    # number below assumes both models cover the same rows in the same order and
    # nothing downstream would show a mismatch, so it is asserted here
    npz = np.load(path, allow_pickle=True)
    p = npz["esm_test"]
    seqs = [str(s) for s in npz["sequences"][-len(p):]]
    assert seqs == expected, f"{os.path.basename(path)} test rows misaligned"
    return p


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    # keep_default_na=False because a two-residue peptide spelled NA would
    # otherwise parse as a missing value. the shortest sequence here is 2
    ds = pd.read_csv(DATASET_SPLIT, keep_default_na=False)
    test = ds[ds.Split == "test"].reset_index(drop=True)

    # NegType comes from the main split rather than the Basith one, since the
    # test rows are identical in both and only the main split labels them
    seqs = test["Sequence"].tolist()
    models = [("dual-neg", test_probs(DUAL_NPZ, seqs, "dual-neg")),
              ("basith", test_probs(BASITH_NPZ, seqs, "basith"))]

    y = test["Label"].to_numpy()
    pos = (test.Label == 1).to_numpy()
    hard = (test.NegType == "hard").to_numpy() # Swiss-Prot fragments
    soft = (test.NegType == "soft").to_numpy() # DBAASP antimicrobial peptides

    # every peptide behind FPR_hard and FPR_soft is a true non-ADP, so those two
    # columns are error rates and not scores. FPR_hard is the headline
    metrics = []
    for name, p in models:
        yhat = p > THRESHOLD
        metrics.append({"model": name,
                        "AUC": roc_auc_score(y, p),
                        "ACC": accuracy_score(y, yhat),
                        "MCC": matthews_corrcoef(y, yhat),
                        "recall": float(yhat[pos].mean()),
                        "FPR_hard": float(yhat[hard].mean()),
                        "FPR_soft": float(yhat[soft].mean()),
                        "n_pos": int(pos.sum()), "n_hard": int(hard.sum()),
                        "n_soft": int(soft.sum())})

    # the Basith pool is not length-matched, which is faithful to the published
    # set but leaves a length gap between the two training classes. section 3.2
    # quotes these means
    bsplit = pd.read_csv(BASITH_SPLIT, keep_default_na=False)
    train_len = lambda df, label: df[(df.Split == "train") & (df.Label == label)].Length
    train_lengths = [
        {"set": "positives (both runs)", "n": len(train_len(ds, 1)),
         "mean_len": train_len(ds, 1).mean()},
        {"set": "dual negatives", "n": len(train_len(ds, 0)),
         "mean_len": train_len(ds, 0).mean()},
        {"set": "Basith negatives", "n": len(train_len(bsplit, 0)),
         "mean_len": train_len(bsplit, 0).mean()}]

    # a wrongly accepted negative scores 1, so an AUC against negated length says
    # the model's mistakes are its short peptides. the first row is the control,
    # over all test rows, showing length carries no signal about the true label
    lengths = test["Length"].to_numpy()
    neg = y == 0
    length_rows = [{"model": "control: length vs true label", "subset": "all test rows",
                    "length_auc": roc_auc_score(y, -lengths), "n": len(y)}]
    for name, p in models:
        for subset, mask in [("all negatives", neg), ("hard", neg & hard), ("soft", neg & soft)]:
            fp = (p[mask] > THRESHOLD).astype(int)
            if len(set(fp)) < 2: # roc_auc_score needs both classes present
                continue
            length_rows.append({"model": name, "subset": subset,
                                "length_auc": roc_auc_score(fp, -lengths[mask]),
                                "mean_len_FP": lengths[mask][fp == 1].mean(),
                                "mean_len_TN": lengths[mask][fp == 0].mean(),
                                "n_FP": int(fp.sum()), "n": int(mask.sum())})

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(metrics).to_csv(os.path.join(OUT, "phase3_6_ablation.csv"), index=False)
    pd.DataFrame(train_lengths).to_csv(os.path.join(OUT, "phase3_6_train_lengths.csv"), index=False)
    pd.DataFrame(length_rows).to_csv(os.path.join(OUT, "phase3_6_length_control.csv"), index=False)


if __name__ == "__main__":
    main()