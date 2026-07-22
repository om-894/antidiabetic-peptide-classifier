
"""
Phase 6.4 (benchmark): is BertADP just reading net charge?

Pools the test hard + soft negatives BertADP scored (Phase 6.2) and tests whether its
decisions track net charge. Xie's negatives are antimicrobial peptides, so a
"antimicrobial = non-ADP" shortcut separates their training data but fails on near-neutral
Swiss-Prot fragments hence the much worse FPR on hard than soft negatives.

INPUT   benchmark/bertadp_hardneg_test_pred.csv
        benchmark/bertadp_softneg_test_pred.csv
OUTPUT  benchmark/charge_shortcut.csv + benchmark/charge_shortcut.png

REQUIREMENTS  pip install numpy pandas scikit-learn matplotlib peptides
"""

# imports
import numpy as np
import pandas as pd
import peptides
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# files to read in defined as gloabls
HARD = "benchmark/bertadp_hardneg_test_pred.csv"
SOFT = "benchmark/bertadp_softneg_test_pred.csv"


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

    # plot figure for charge vs BertADP's probability, split by negative type
    plt.figure(figsize=(5, 3.5))
    for t, m, c in [("hard", "o", "firebrick"), ("soft", "^", "steelblue")]:
        g = d[d.type == t]
        plt.scatter(g.charge, g.Positive_Probability, marker=m, c=c, alpha=0.75,
                    label=f"{t} negatives")
    plt.axhline(0.5, color="k", ls="--", lw=0.8) # BertADP's decision boundary
    plt.xlabel("net charge (pH 7.4)"); plt.ylabel("BertADP P(ADP)")
    plt.title(f"BertADP tracks net charge (AUC {auc:.2f})")
    plt.legend(); plt.tight_layout()
    plt.savefig("benchmark/charge_shortcut.png", dpi=150)
    print("saved -> benchmark/charge_shortcut.csv + charge_shortcut.png")


if __name__ == "__main__":
    main()