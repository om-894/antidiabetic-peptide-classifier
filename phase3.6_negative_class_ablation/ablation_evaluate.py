#!/usr/bin/env python3
"""
ablation_evaluate.py
====================
Negative-class ablation, final step: cross-evaluate the dual-negative ESM-2 and
the Basith-negative ESM-2 on the SAME 178-row test set, to isolate what the
dual-negative class buys. Run after copying esm2_dora_basith_predictions.npz
back from Viking.

The headline: each model's FALSE-POSITIVE RATE on the 45 hard negatives (random
Swiss-Prot fragments). The Basith model never saw hard negatives in training, so
if it lights up on them while the dual-negative model rejects them, that is the
quantified contribution of the negative-class construction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

ds = pd.read_csv("data/dataset_split.csv")
test = ds[ds.Split == "test"].reset_index(drop=True)

dual = np.load("esm2_dora_predictions.npz", allow_pickle=True)
bas = np.load("esm2_dora_basith_predictions.npz", allow_pickle=True)

# alignment: both test-prediction vectors must correspond to the same 178 rows
n = len(dual["esm_test"])
dual_seqs = [str(s) for s in dual["sequences"][-n:]]
bas_seqs = [str(s) for s in bas["sequences"][-len(bas["esm_test"]):]]
assert dual_seqs == test["Sequence"].tolist(), "dual test rows misaligned"
assert bas_seqs == test["Sequence"].tolist(), "Basith test rows misaligned"
print("alignment OK — both models scored on the identical 178-row test set\n")

y = test["Label"].to_numpy()
hard = (test.NegType == "hard").to_numpy()
soft = (test.NegType == "soft").to_numpy()
pos = (test.Label == 1).to_numpy()
print(f"test set: {pos.sum()} positives | {hard.sum()} hard neg | {soft.sum()} soft neg\n")

print(f"{'model':12s} {'AUC':>6s} {'ACC':>6s} {'MCC':>6s} "
      f"{'recall':>7s} {'FPR_hard':>9s} {'FPR_soft':>9s}")
print("-" * 62)
for name, p in [("dual-neg", dual["esm_test"]), ("basith", bas["esm_test"])]:
    yhat = p > 0.5
    print(f"{name:12s} {roc_auc_score(y, p):6.3f} {accuracy_score(y, yhat):6.3f} "
          f"{matthews_corrcoef(y, yhat):6.3f} {yhat[pos].mean():7.3f} "
          f"{yhat[hard].mean():9.3f} {yhat[soft].mean():9.3f}")

fh_dual = (dual["esm_test"][hard] > 0.5).mean()
fh_bas = (bas["esm_test"][hard] > 0.5).mean()
print(f"\nHEADLINE: false-positive rate on the {hard.sum()} HARD negatives — "
      f"dual-neg {fh_dual:.1%} vs Basith {fh_bas:.1%}")
print("(the Basith model never saw hard negatives in training; this gap is what "
      "the dual-negative class buys.)")
