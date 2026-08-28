
"""
Phase 6.3: False-positive rate on hard negatives, BertADP against both ESM-2 models.

Every peptide scored here is a true non-ADP, so any positive call is an error. The three
models are compared on the identical 45 test hard negatives, which are the only ones held
out from all three, with bootstrap intervals on each rate and a McNemar test on the pair
that matters. BertADP is also scored on all 328 hard negatives, where nothing constrains
the n, since it never trained on any of them.

INPUTS  bertadp_hardneg_test_pred.csv (BertADP's calls, from phase 6.2)
        bertadp_hardneg_all_pred.csv (the same for all 328)
        esm2_dora_predictions.npz, esm2_dora_basith_predictions.npz
        dataset_split.csv (locates the 45 test hard negatives)
OUTPUTS  hardneg_fpr.csv (per-model rate with its interval)
         phase6_3_benchmark_tests.csv (the all-328 rate and the McNemar result)
REQUIREMENTS  pip install numpy pandas statsmodels
"""

# Imports
import os

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_exact


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
BERT = "benchmark/bertadp_hardneg_test_pred.csv"
BERT_ALL = "benchmark/bertadp_hardneg_all_pred.csv" # all 328, not just the 45 in the test split
ESM = "predictions/esm2_dora_predictions.npz"
BAS = "predictions/esm2_dora_basith_predictions.npz"
SPLIT = "data/dataset_split.csv"

OUT_FPR = "benchmark/hardneg_fpr.csv"
OUT_TESTS = "results/phase6_3_benchmark_tests.csv"

SEED = 42
NBOOT = 1000 # bootstrap resamples, the same figure phase 4.1 uses


# --------------------------------------------------------------------------- #
# BOOTSTRAP
# --------------------------------------------------------------------------- #
def fpr_ci(preds, seed=SEED, n=NBOOT):
    """False-positive rate and its 95% bootstrap interval.

    preds is a 0/1 flag per peptide, 1 meaning the model called this non-ADP an ADP, so
    the rate is the mean. Resampling the peptides with replacement and taking the middle
    95% of the recomputed rates answers how far the rate would move on a different sample
    of the same size. MLstatkit covers accuracy, F1 and AUC but not a false-positive rate,
    so the same resampling phase 4.1 uses for MCC is done here."""
    preds = np.asarray(preds)
    rng = np.random.default_rng(seed)
    vals = [preds[rng.integers(0, len(preds), len(preds))].mean() for _ in range(n)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return preds.mean(), lo, hi


# --------------------------------------------------------------------------- #
# MODEL CALLS
# --------------------------------------------------------------------------- #
def npz_hardneg_preds(path, hardnegs):
    """One model's threshold calls on the hard negatives, keyed by sequence.

    The npz holds every sequence but only the test probabilities. The test rows sit at
    the end of dataset_split.csv, so the last len(y_test) sequences line up with esm_test.
    Same slice as test_probs in phase 3.6.4, which asserts the order the two share."""
    d = np.load(path, allow_pickle=True)
    y = d["y_test"]
    seqs = [str(s) for s in d["sequences"]][-len(y):]
    call = dict(zip(seqs, (d["esm_test"].astype(float) > 0.5).astype(int)))
    return {s: call[s] for s in hardnegs if s in call}


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    os.makedirs("results", exist_ok=True)
    split = pd.read_csv(SPLIT, keep_default_na=False) # a two-residue NA is Asn-Ala, not missing

    # 283 of the 328 hard negatives sit in the dual-negative training set, so these 45 are
    # the only ones held out from every model and the only fair three-way comparison
    hardnegs = set(split[(split.Split == "test") & (split.NegType == "hard")
                         & (split.Label == 0)].Sequence.astype(str))

    bert = pd.read_csv(BERT, keep_default_na=False)
    bert["Sequence"] = bert.Sequence.astype(str)
    bpred = dict(zip(bert.Sequence, (bert.Prediction == 1).astype(int)))
    dual = npz_hardneg_preds(ESM, hardnegs)
    bas = npz_hardneg_preds(BAS, hardnegs)

    # only peptides all three scored, so the rates are over identical rows
    seqs = sorted(s for s in hardnegs if s in bpred and s in dual and s in bas)
    b = [bpred[s] for s in seqs] # BertADP
    du = [dual[s] for s in seqs] # dual-negative ESM-2
    ba = [bas[s] for s in seqs] # Basith-negative ESM-2

    rows = []
    for name, pr in [("BertADP", b), ("Basith-neg ESM-2", ba), ("dual-neg ESM-2", du)]:
        f, lo, hi = fpr_ci(pr)
        rows.append({"model": name, "FPR": round(f, 3), "CI_low": round(lo, 3),
                     "CI_high": round(hi, 3), "false_pos": int(np.sum(pr)), "n": len(pr)})
    pd.DataFrame(rows).to_csv(OUT_FPR, index=False)

    # the larger-n version, which only BertADP can be scored on
    ball = pd.read_csv(BERT_ALL, keep_default_na=False)
    wrong = ball.Prediction == 1
    fa, la, ha = fpr_ci(wrong.astype(int).tolist())
    mp = ball.Positive_Probability.mean()

    # McNemar asks whether the BertADP against dual-negative gap could be chance. On a hard
    # negative, being right means calling it non-ADP, so only the peptides where the two
    # disagree carry information and the test asks whether that split departs from an even
    # one. Same call as phase 4.1, with the concordant corners passed as zeros
    b_ok, d_ok = np.array(b) == 0, np.array(du) == 0
    bc = int(np.sum(b_ok & ~d_ok)) # BertADP right, dual-negative wrong
    cc = int(np.sum(~b_ok & d_ok)) # BertADP wrong, dual-negative right
    p = mcnemar_exact([[0, bc], [cc, 0]], exact=True).pvalue if (bc + cc) else 1.0

    pd.DataFrame([
        {"metric": "FPR_all_hard_negatives", "value": fa, "ci_low": la, "ci_high": ha,
         "n": len(ball), "false_pos": int(wrong.sum()), "mean_prob": mp},
        {"metric": "mcnemar_bertadp_vs_dual", "value": p, "n": len(seqs),
         "b_bert_right_ours_wrong": bc, "c_bert_wrong_ours_right": cc},
    ]).to_csv(OUT_TESTS, index=False)

    print(f"{len(seqs)} matched hard negatives -> {OUT_FPR}, {OUT_TESTS}")


if __name__ == "__main__":
    main()