
"""
Phase 6.4: Is BertADP reading net charge?

Xie's negatives are antimicrobial peptides, which are strongly cationic, so a rule
separating cationic from neutral would carry their training data and fail on near-neutral
Swiss-Prot fragments. That is the shape of the error already seen, a far higher rate on
hard negatives than on soft ones, so this measures whether charge alone reproduces the
model's decisions and whether length could explain the same pattern instead.

Every peptide scored here is a true non-ADP, so any positive call is an error.

INPUTS  bertadp_hardneg_test_pred.csv, bertadp_softneg_test_pred.csv (phase 6.2)
        bertadp_hardneg_all_pred.csv (all 328, for the length control)
OUTPUTS  charge_shortcut.csv (false-positive rate by negative type)
         charge_vs_length_control.csv (charge and length as predictors, by length band)
         fpr_by_length.csv (the rate at every individual length)
REQUIREMENTS  pip install numpy pandas scikit-learn peptides
"""

# Imports
import os

import pandas as pd
import peptides
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
HARD = "benchmark/bertadp_hardneg_test_pred.csv"
SOFT = "benchmark/bertadp_softneg_test_pred.csv"
HARD_ALL = "benchmark/bertadp_hardneg_all_pred.csv" # all 328, not just the 45 in the test split

OUT_DIR = "benchmark"
OUT_SHORTCUT = "benchmark/charge_shortcut.csv"
OUT_CONTROL = "benchmark/charge_vs_length_control.csv"
OUT_BY_LENGTH = "benchmark/fpr_by_length.csv"

# the first band is the whole set and the rest isolate single lengths, where length is
# constant by construction and cannot be what the model is reading
BANDS = [("all", 2, 41), ("3-9aa", 3, 9), ("10aa", 10, 10), ("20aa", 20, 20)]


def charges(seqs):
    """Net charge at pH 7.4 per sequence, the same call phase 2.2 makes for its descriptor."""
    return [peptides.Peptide(str(s)).charge(pH=7.4) for s in seqs]


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # a two-residue NA is Asn-Ala rather than a missing value, so it is read as a string
    hard = pd.read_csv(HARD, keep_default_na=False).assign(type="hard")
    soft = pd.read_csv(SOFT, keep_default_na=False).assign(type="soft")
    d = pd.concat([hard, soft], ignore_index=True)

    # the rate on each negative class. BertADP trained on peptides like the soft ones, so
    # the gap between the two rates is where the failure sits
    pd.DataFrame([{"negatives": t, "n": len(g),
                   "false_pos": int((g.Prediction == 1).sum()),
                   "FPR": round((g.Prediction == 1).mean(), 3)}
                  for t, g in d.groupby("type")]).to_csv(OUT_SHORTCUT, index=False)

    # charge against length over all 328, since the test split alone is too small to band
    allh = pd.read_csv(HARD_ALL, keep_default_na=False)
    allh["charge"] = charges(allh.Sequence)
    allh["length"] = allh.Sequence.str.len()

    # a wrongly accepted negative scores 1, so an AUC against negated charge says the
    # model's mistakes are its anionic peptides. The same against negated length says
    # they are its short ones
    ctrl = []
    for band, lo, hi in BANDS:
        sub = allh[allh.length.between(lo, hi)]
        w = sub.Prediction == 1
        if w.nunique() < 2: # roc_auc_score needs both classes present
            continue
        ctrl.append({"band": band, "n": len(sub), "false_pos": int(w.sum()),
                     "FPR": w.mean(),
                     "charge_auc": roc_auc_score(w, -sub.charge),

                     # length cannot discriminate where every peptide in the band is one length
                     "length_auc": roc_auc_score(w, -sub.length) if sub.length.nunique() > 1 else None,
                     "mean_charge_FP": sub.loc[w, "charge"].mean(),
                     "mean_charge_TN": sub.loc[~w, "charge"].mean()})
    pd.DataFrame(ctrl).to_csv(OUT_CONTROL, index=False)

    # the rate at every individual length, which is what shows no length escapes
    per_len = allh.assign(fp=allh.Prediction == 1).groupby("length").agg(
        n=("fp", "size"), false_pos=("fp", "sum"))
    per_len["FPR"] = per_len.false_pos / per_len.n
    per_len.to_csv(OUT_BY_LENGTH)

    print(f"{len(d)} test negatives, {len(allh)} hard negatives -> {OUT_DIR}/")


if __name__ == "__main__":
    main()