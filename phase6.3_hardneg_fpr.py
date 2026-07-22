
"""
Phase 6.3 (benchmark): false-positive rate on hard negatives - BertADP vs my models.

On the identical 45 test hard negatives, compares the FPR of the SOTA BertADP against 
dual-negative and Basith-negative ESM-2 models, with bootstrap 95% CIs and a McNemar test
(BertADP vs dual-negative). External validation of the dual-negative contribution.

INPUT   benchmark/bertadp_hardneg_test_pred.csv         BertADP predictions (Phase 6.2)
        predictions/esm2_dora_predictions.npz           dual-negative ESM-2
        predictions/esm2_dora_basith_predictions.npz    Basith-negative ESM-2
        data/dataset_split.csv                          locates the 45 test hard negatives
OUTPUT  benchmark/hardneg_fpr.csv + benchmark/hardneg_fpr.png

REQUIREMENTS  pip install numpy pandas scipy matplotlib
"""

# imports
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# constants and outputs
BERT = "benchmark/bertadp_hardneg_test_pred.csv"
ESM = "predictions/esm2_dora_predictions.npz"
BAS = "predictions/esm2_dora_basith_predictions.npz"
SPLIT = "data/dataset_split.csv"
SEED, NBOOT = 42, 1000


def fpr_ci(preds, seed=SEED, n=NBOOT):
    """False-positive rate and a bootstrap 95% CI.

    `preds` is a 0/1 flag per peptide (1 = the model wrongly called this non-ADP an ADP),
    so the FPR is just the mean. For the CI we resample the peptides with replacement
    n times, recompute the FPR each time and take the middle 95% of those values -
    i.e. how much the rate would change on a different sample of this size."""
    preds = np.asarray(preds)
    rng = np.random.default_rng(seed) # make an rng so the bootstrap is reproducible
    vals = [preds[rng.integers(0, len(preds), len(preds))].mean() for _ in range(n)]
    lo, hi = np.percentile(vals, [2.5, 97.5]) # define the 95% CI.
    return preds.mean(), lo, hi # return the FPR and the CI


def npz_hardneg_preds(path, hardnegs):
    """Pull one of the models calls on the hard negatives, keyed by sequence.

    The npz stores train+test sequences together but only test probabilities, so the
    test sequences are the tail of `sequences` and line up with `esm_test`."""
    d = np.load(path, allow_pickle=True)
    y = d["y_test"]
    seqs = [str(s) for s in d["sequences"]][-len(y):] # the test sequences are the tail of `sequences` and line up with `y_test`
    call = dict(zip(seqs, (d["esm_test"].astype(float) > 0.5).astype(int))) # 0.5 threshold for ADP / non-ADP
    return {s: call[s] for s in hardnegs if s in call} # only keep the hard negatives that were actually scored by this model


def main():
    # the 45 hard negatives from the held-out test set - all true non-ADPs
    split = pd.read_csv(SPLIT)

    # the test set hard negatives are the only non-ADPs in the test set that were also in the training set
    hardnegs = set(split[(split.Split == "test") & (split.NegType == "hard")
                         & (split.Label == 0)].Sequence.astype(str))

    # each model's call on those peptides. every 1 is a false positive, since all are non-ADPs.
    bert = pd.read_csv(BERT); bert["Sequence"] = bert.Sequence.astype(str)
    bpred = dict(zip(bert.Sequence, (bert.Prediction == 1).astype(int)))
    dual = npz_hardneg_preds(ESM, hardnegs)
    bas  = npz_hardneg_preds(BAS, hardnegs)

    # only keep peptides all three models scored, so the comparison is like-for-like
    seqs = [s for s in hardnegs if s in bpred and s in dual and s in bas]

    # b represents BertADP, du means dual-negative ESM-2, ba means Basith-negative ESM-2
    b  = [bpred[s] for s in seqs]
    du = [dual[s]  for s in seqs]
    ba = [bas[s]   for s in seqs]
    print(f"matched hard negatives: {len(seqs)}")

    # FPR + CI for each model and made into a table
    rows = []
    for name, pr in [("BertADP (SOTA)", b), ("Basith-neg ESM-2", ba), ("dual-neg ESM-2 (ours)", du)]:
        f, lo, hi = fpr_ci(pr)
        rows.append({"model": name, "FPR": round(f, 3), "CI_low": round(lo, 3),
                     "CI_high": round(hi, 3), "false_pos": int(np.sum(pr)), "n": len(pr)})
    tab = pd.DataFrame(rows)
    tab.to_csv("benchmark/hardneg_fpr.csv", index=False)
    print(tab.to_string(index=False))

    # McNemar: is the BertADP vs dual-negative gap real, or could it be chance?
    # On a hard negative, getting it "right" means predicting non-ADP (0). The test only
    # looks at peptides where the two models disagree (b and c) and asks whether that
    # split is lopsided rather than the 50/50 you'd expect if they were equally good.
    b_ok, d_ok = np.array(b) == 0, np.array(du) == 0
    bc = int(np.sum(b_ok & ~d_ok)) # BertADP right, ours wrong
    cc = int(np.sum(~b_ok & d_ok)) # BertADP wrong, ours right
    p = scipy.stats.binomtest(min(bc, cc), bc + cc, 0.5).pvalue if (bc + cc) else 1.0
    print(f"\nMcNemar (BertADP vs dual-neg): discordant b={bc}, c={cc}, p={p:.4g}")

    # bar chart of the three FPRs with their CIs. others red, ours blue
    plt.figure(figsize=(5, 3))
    plt.bar(tab.model, tab.FPR, capsize=4,
            yerr=[tab.FPR - tab.CI_low, tab.CI_high - tab.FPR],   # asymmetric error bars from the CI
            color=["firebrick", "firebrick", "steelblue"])
    plt.ylabel("false-positive rate on hard negatives"); plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right"); plt.tight_layout()
    plt.savefig("benchmark/hardneg_fpr.png", dpi=150)
    print("saved -> benchmark/hardneg_fpr.csv + hardneg_fpr.png")


if __name__ == "__main__":
    main()