
"""
Negative-class ablation, final step: cross-evaluate the dual-negative ESM-2 and the
Basith-negative ESM-2 on the same 178-row test set, to isolate what the dual-negative
class buys. Headline = each model's false-positive rate on the 45 hard negatives.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

# Paths anchored to this folder; the main split + predictions live in the repo, one level up.
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_SPLIT = os.path.join(HERE, "..", "data", "dataset_split.csv")
DUAL_NPZ   = os.path.join(HERE, "..", "predictions", "esm2_dora_predictions.npz")
BASITH_NPZ = os.path.join(HERE, "..", "predictions", "esm2_dora_basith_predictions.npz")
OUT = os.path.join(HERE, "..", "results")

# test set (from the main split) tells us the NegType of each test row.
ds = pd.read_csv(DATASET_SPLIT)
test = ds[ds.Split == "test"].reset_index(drop=True)

dual = np.load(DUAL_NPZ, allow_pickle=True)      # my dual-negative ESM-2 predictions
bas  = np.load(BASITH_NPZ, allow_pickle=True)    # Basith-negative ESM-2 predictions

# Alignment check: both models' test predictions must map to the same 178 test rows
# in the same order. The npz stores sequences as (train + test), so the last N are
# the test sequences.
n = len(dual["esm_test"])
dual_seqs = [str(s) for s in dual["sequences"][-n:]]
bas_seqs  = [str(s) for s in bas["sequences"][-len(bas["esm_test"]):]]
assert dual_seqs == test["Sequence"].tolist(), "dual test rows misaligned"
assert bas_seqs  == test["Sequence"].tolist(), "Basith test rows misaligned"
print("alignment OK - both models scored on the identical 178-row test set\n")

# Masks for the different kinds of test rows.
y    = test["Label"].to_numpy()
pos  = (test.Label == 1).to_numpy()              # positives (ADPs)
hard = (test.NegType == "hard").to_numpy()       # hard negatives (Swiss-Prot fragments)
soft = (test.NegType == "soft").to_numpy()       # soft negatives (bioactive peptides)
print(f"test set: {pos.sum()} positives | {hard.sum()} hard neg | {soft.sum()} soft neg\n")

# Per-model metrics on the common test set. FPR = fraction of that negative class
# wrongly predicted positive (threshold 0.5); recall = fraction of positives caught.
print(f"{'model':12s} {'AUC':>6s} {'ACC':>6s} {'MCC':>6s} "
      f"{'recall':>7s} {'FPR_hard':>9s} {'FPR_soft':>9s}")
print("-" * 62)
for name, p in [("dual-neg", dual["esm_test"]), ("basith", bas["esm_test"])]:
    yhat = p > 0.5
    print(f"{name:12s} {roc_auc_score(y, p):6.3f} {accuracy_score(y, yhat):6.3f} "
          f"{matthews_corrcoef(y, yhat):6.3f} {yhat[pos].mean():7.3f} "
          f"{yhat[hard].mean():9.3f} {yhat[soft].mean():9.3f}")

# The headline: how often each model wrongly calls a hard negative an ADP. The
# Basith model never saw hard negatives in training, so this gap is what its
# bioactive-only negatives cost it.
fh_dual = (dual["esm_test"][hard] > 0.5).mean()
fh_bas  = (bas["esm_test"][hard]  > 0.5).mean()
print(f"\nHEADLINE: false-positive rate on the {hard.sum()} HARD negatives - "
      f"dual-neg {fh_dual:.1%} vs Basith {fh_bas:.1%}")

# save ablation numbers to be used in results
os.makedirs(OUT, exist_ok=True)
rows = []
for name, p in [("dual-neg", dual["esm_test"]), ("basith", bas["esm_test"])]:
    yhat = p > 0.5
    rows.append({"model": name,
                 "AUC": roc_auc_score(y, p),
                 "ACC": accuracy_score(y, yhat),
                 "MCC": matthews_corrcoef(y, yhat),
                 "recall": float(yhat[pos].mean()),
                 "FPR_hard": float(yhat[hard].mean()),
                 "FPR_soft": float(yhat[soft].mean()),
                 "n_pos": int(pos.sum()), "n_hard": int(hard.sum()), "n_soft": int(soft.sum())})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "phase3_6_ablation.csv"), index=False)
print("saved -> results/phase3_6_ablation.csv")