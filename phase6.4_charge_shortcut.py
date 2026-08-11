
"""
Phase 6.4 (benchmark): is BertADP just reading net charge?

Pools the test hard + soft negatives BertADP scored (Phase 6.2) and tests whether its
decisions track net charge. Xie's negatives are antimicrobial peptides, so a
"antimicrobial = non-ADP" shortcut separates their training data but fails on near-neutral
Swiss-Prot fragments hence the much worse FPR on hard than soft negatives.

INPUT   benchmark/bertadp_hardneg_test_pred.csv
        benchmark/bertadp_softneg_test_pred.csv
OUTPUT  benchmark/charge_shortcut.csv + benchmark/charge_vs_length_control.csv

REQUIREMENTS  pip install numpy pandas scikit-learn peptides
"""

# imports
import numpy as np
import pandas as pd
import peptides
from sklearn.metrics import roc_auc_score

# files to read in defined as gloabls
HARD = "benchmark/bertadp_hardneg_test_pred.csv"
SOFT = "benchmark/bertadp_softneg_test_pred.csv"
HARD_ALL = "benchmark/bertadp_hardneg_all_pred.csv" # all 328, for the length control


def main():

    # read in the BertADP predictions on the hard and soft negatives, add a column for the type
    hard = pd.read_csv(HARD); hard["type"] = "hard"
    soft = pd.read_csv(SOFT); soft["type"] = "soft"
    d = pd.concat([hard, soft], ignore_index=True)
    d["charge"] = d.Sequence.map(lambda s: peptides.Peptide(s).charge(pH=7.4))

    # FPR per negative type - every row is a true non-ADP, so Prediction==1 is an error
    tab = pd.DataFrame([{"negatives": t, "n": len(g),
                         "false_pos": int((g.Prediction == 1).sum()),
                         "FPR": round((g.Prediction == 1).mean(), 3)}
                        for t, g in d.groupby("type")])
    tab.to_csv("benchmark/charge_shortcut.csv", index=False)
    print(tab.to_string(index=False))

    # can one hand-computed feature reproduce BertADP's decision
    auc = roc_auc_score((d.Prediction == 0).astype(int), d.charge)
    r = np.corrcoef(d.charge, d.Positive_Probability)[0, 1]
    print(f"\nmean charge | called ADP (wrong) : {d[d.Prediction==1].charge.mean():+.2f}")
    print(f"mean charge | called non-ADP (right) : {d[d.Prediction==0].charge.mean():+.2f}")
    print(f"AUC of net charge alone predicting BertADP's rejection: {auc:.3f}")
    print(f"correlation net charge vs BertADP P(ADP): r = {r:+.3f}")

    # charge or just length? Xie's negatives are 18-35 aa, mine are length-matched
    allh = pd.read_csv(HARD_ALL, keep_default_na=False)
    allh["charge"] = allh.Sequence.map(lambda s: peptides.Peptide(s).charge(pH=7.4))
    allh["length"] = allh.Sequence.str.len()

    ctrl = []
    for lab, lo, hi in [("all", 2, 41), ("2-9aa", 2, 9), ("10-14aa", 10, 14), ("15-41aa", 15, 41)]:
        sub = allh[allh.length.between(lo, hi)]
        w = sub.Prediction == 1 # every row is a true non-ADP
        if w.nunique() < 2:
            continue
        ctrl.append({"band": lab, "n": len(sub), "false_pos": int(w.sum()),
                     "FPR": w.mean(),
                     "charge_auc": roc_auc_score(w, -sub.charge),
                     "length_auc": roc_auc_score(w, -sub.length),
                     "mean_charge_FP": sub.loc[w, "charge"].mean(),
                     "mean_charge_TN": sub.loc[~w, "charge"].mean()})
    ctrl = pd.DataFrame(ctrl)
    ctrl.to_csv("benchmark/charge_vs_length_control.csv", index=False)
    print("\ncharge vs length, all 328 hard negatives")
    print(ctrl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    # the 15-41aa band overlaps Xie's own negative lengths, so a high FPR there rules out length


if __name__ == "__main__":
    main()