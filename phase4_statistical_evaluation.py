
"""
Section 4: Statistical evaluation (internal model comparisons). 
Point estimates on a 178-sample test set are noisy, so this script quantifies the
uncertainty around each metric and the significance of the differences between
models, rather than reporting bare numbers.

It runs on already-saved test-set predictions (no models are re-run):
  predictions/esm2_dora_predictions.npz        dual-negative ESM-2  -> the final model
  predictions/stack_predictions.npz            the stacked ensemble
  predictions/esm2_dora_basith_predictions.npz Basith-negative ESM-2 (the ablation)
All three are probability predictions on the identical 178-row test set.

Two comparisons are tested:
  (i)  stack vs ESM-2       -> Aim: does ensembling beat the best single model?
  (ii) dual-neg vs Basith   -> the ablation: does the dual-negative class help?

Methods: bootstrap confidence intervals, DeLong's test and Cohen's d are taken from
the MLstatkit package; McNemar's test, the MCC bootstrap, calibration
(ECE/Brier) and the Holm-Bonferroni correction are implemented here directly.

Requires: pip install MLstatkit numpy scipy scikit-learn
"""

import numpy as np
import scipy.stats
from sklearn.metrics import matthews_corrcoef
from MLstatkit import Delong_test, Bootstrapping, AUC2OR # for DeLong, bootstrap CIs, Cohen's d

SEED, N_BOOT = 42, 1000 # fixed seed = reproducible; 1000 bootstrap resamples


# --------------------------------------------------------------------------- #
# MCC confidence interval by bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_mcc(y, p, n=N_BOOT, seed=SEED):
    """95% CI for Matthews correlation coefficient.

    MLstatkit's Bootstrapping supports accuracy/F1/AUC but not MCC, so bootstrap
    it here. Idea: repeatedly resample the test rows with replacement, recompute MCC
    each time; the 2.5th-97.5th percentiles of those values form the 95% CI -- i.e.
    'how much would MCC move if we drew a different sample of the same size?'.
    """
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y)) # a resampled test set (same size)
        if len(np.unique(y[idx])) < 2: # skip degenerate one-class samples
            continue
        vals.append(matthews_corrcoef(y[idx], p[idx] > 0.5)) # compute MCC on that resample and store it
    lo, hi = np.percentile(vals, [2.5, 97.5]) # define the 95% CI
    return matthews_corrcoef(y, p > 0.5), lo, hi  # point estimate on full data + CI


# --------------------------------------------------------------------------- #
# McNemar's test (paired comparison of two classifiers)
# --------------------------------------------------------------------------- #
def mcnemar(y, pred_a, pred_b):
    """McNemar's test: do two models differ on the same samples?

    It looks only at the samples where the two models disagree:
        b = model A right, model B wrong
        c = model A wrong, model B right
    Under the null hypothesis (the models are equally good) b and c should be roughly
    equal, so we test whether the split of the (b+c) discordant cases departs from
    50/50 with an exact binomial test. The odds ratio b/c summarises how lopsided the
    disagreements are (the effect size for this contingency table).
    """
    ac, bc = (pred_a == y), (pred_b == y) # which predictions are correct
    b = int(np.sum(ac & ~bc)) # A right, B wrong
    c = int(np.sum(~ac & bc)) # A wrong, B right
    n = b + c # total discordant pairs

    # exact two-sided binomial test that the b:c split is 50/50
    p = scipy.stats.binomtest(min(b, c), n, 0.5).pvalue if n else 1.0
    odds = (b / c) if c else np.inf # odds ratio (effect size)
    return p, b, c, odds


# --------------------------------------------------------------------------- #
# Calibration: are the predicted probabilities trustworthy as probabilities?
# --------------------------------------------------------------------------- #
def ece(y, p, bins=10):
    """Expected Calibration Error.

    Split predictions into 10 probability bins. In each bin, compare the mean
    predicted probability (what the model 'claims') with the empirical fraction of
    positives (what actually happens). ECE is the sample-weighted average gap. 0 = a
    perfectly calibrated model (e.g. of everything it scores 0.7 then 70% are positive).
    This matters because Phase 5 triage thresholds on probabilities, not just labels.
    """
    edges, e = np.linspace(0, 1, bins + 1), 0.0     # bin edges e.g. [0, .1, .2, ..., 1]; e = running total
    for i in range(bins):
        # Select the predictions whose probability falls in bin i, i.e. (edges[i], edges[i+1]].
        # The first bin is made inclusive of 0 so a prediction of exactly 0 isn't dropped.
        mask = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= 0) & (p <= edges[1])
        if mask.sum(): # skip empty bins (nothing to compare)
            # mask.mean() = fraction of all predictions in this bin  -> the bin's weight
            # p[mask].mean() = the model's average confidence in this bin
            # y[mask].mean() = the observed fraction of positives in this bin
            e += mask.mean() * abs(p[mask].mean() - y[mask].mean())   # weighted calibration gap
    return e                                        # total ECE (0 = perfectly calibrated)

def brier(y, p):
    """Brier score = mean squared error between predicted probability and outcome.
    Lower is better; 0 = perfect. A single number combining calibration and sharpness."""
    return float(np.mean((p - y) ** 2))


# --------------------------------------------------------------------------- #
# Holm-Bonferroni multiple-comparison correction
# --------------------------------------------------------------------------- #
def holm_bonferroni(pvals):
    """Adjust p-values for running several tests at once.

    Running k tests inflates the chance of at least one false positive. Holm-Bonferroni
    controls the family-wise error rate: sort the p-values, multiply the smallest by k,
    the next by k-1, ... and enforce monotonicity. Less conservative (more powerful)
    than plain Bonferroni. An adjusted p < 0.05 is significant after correction.
    """
    m = len(pvals) # number of tests in the family
    order = np.argsort(pvals) # indices that sort the p-values smallest -> largest
    adj = np.empty(m) # output: an adjusted p-value for each original position
    prev = 0.0 # running maximum, used to keep adjusted values non-decreasing

    # go from the smallest p-value (rank 0) up to the largest.
    for rank, i in enumerate(order): # rank = position in the sorted order; i = original index in pvals
        
        # The multiplier shrinks as you ascend: smallest p x m, next x (m-1), ... largest x 1.
        value = (m - rank) * pvals[i]
        
        # Monotonicity ("step-down"): an adjusted p may never be smaller than one from an
        # earlier (more significant) test, so we carry forward the running maximum.
        prev = max(prev, value)
        adj[i] = min(prev, 1.0) # cap at 1.0 (a probability), and store at the original index i
    return adj


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    # load the three sets of test-set probabilities
    esm = np.load("predictions/esm2_dora_predictions.npz", allow_pickle=True)
    stack = np.load("predictions/stack_predictions.npz", allow_pickle=True)
    bas = np.load("predictions/esm2_dora_basith_predictions.npz", allow_pickle=True)

    y = esm["y_test"].astype(int) # true labels (shared test set)
    p_esm = esm["esm_test"].astype(float) # final model: dual-negative ESM-2
    p_stk = stack["stack_test"].astype(float) # stacked ensemble
    p_bas = bas["esm_test"].astype(float) # Basith-negative ESM-2
    
    # all three must be scored on the same 178 rows, or the comparisons are invalid
    assert np.array_equal(stack["y_test"].astype(int), y)
    assert np.array_equal(bas["y_test"].astype(int), y)
    print(f"test set: {len(y)} rows ({y.sum()} positive)\n")

    # 1) Bootstrap 95% CIs (MLstatkit for acc/F1/AUC, manual for MCC)
    # Each metric is reported as: point estimate [lower bound, upper bound].
    print("Bootstrap 95% CIs (point [lo, hi]):")
    for name, p in [("ESM-2 (final)", p_esm), ("stack", p_stk)]:
        out = []
        for label, key in [("ACC", "accuracy"), ("F1", "f1"), ("AUC", "roc_auc")]:

            # MLstatkit's Bootstrapping returns the point estimate and the 95% CI for the requested metric
            s, lo, hi = Bootstrapping(y, p, key, n_bootstraps=N_BOOT, random_state=SEED) 
            out.append(f"{label} {s:.3f} [{lo:.3f}, {hi:.3f}]")
        s, lo, hi = bootstrap_mcc(y, p)
        out.append(f"MCC {s:.3f} [{lo:.3f}, {hi:.3f}]")
        print(f"  {name:14s} " + " | ".join(out)) # print all metrics for this model on one line

    # 2) Effect size: Cohen's d derived from each model's AUC
    # A p-value says whether a difference exists; an effect size says how large.
    # AUC2OR converts an AUC to Cohen's d (the standardised separation between the
    # positive and negative score distributions) under a binormal assumption.
    print("\nEffect size (Cohen's d from AUC):")
    for name, p in [("ESM-2 (final)", p_esm), ("stack", p_stk), ("basith", p_bas)]:
        auc = Bootstrapping(y, p, "roc_auc", n_bootstraps=1, random_state=SEED)[0]
        
        # AUC2OR returns Cohen's d and the odds ratio (OR) for a given AUC. 
        # The underscore is used to ignore the first return value, which is not needed here.
        _, d, _, OR = AUC2OR(auc, return_all=True)
        print(f"  {name:14s} AUC {auc:.3f} -> Cohen's d {d:.2f}, odds ratio {OR:.2f}")

    # 3) The two internal comparisons
    # DeLong's test  -> is the AUC difference significant (correlated ROC curves)?
    # McNemar's test -> do the label predictions differ on matched samples?
    comparisons = [
        ("stack vs ESM-2 (Aim 3)",        p_stk, p_esm), # does the ensemble help?
        ("dual-neg vs Basith (ablation)", p_esm, p_bas), # does the negative class help?
    ]
    print("\nPairwise comparisons:")
    raw_p, rows = [], []
    for label, pa, pb in comparisons:
        
        # DeLong returns z-stat, p-value, 95% CIs and AUCs for both models
        z, dp, ci_a, ci_b, auc_a, auc_b, _ = Delong_test(
            y, pa, pb, return_ci=True, return_auc=True, random_state=SEED)
        
        # McNemar returns p-value, counts of discordant pairs, and odds ratio
        mp, b, c, odds = mcnemar(y, (pa > 0.5).astype(int), (pb > 0.5).astype(int))
        raw_p += [dp, mp] # collect all p-values for correction
        rows.append((label, dp, auc_a, auc_b, ci_a, ci_b, mp, odds))
    
    # Apply Holm-Bonferroni correction to the raw p-values from all tests
    adj = holm_bonferroni(np.array(raw_p)) # correct across all 4 tests at once
    
    # Print the results of each comparison, including the adjusted p-values
    for i, (label, dp, auc_a, auc_b, ci_a, ci_b, mp, odds) in enumerate(rows):
        print(f"\n  {label}")
        print(f"    DeLong : AUC {auc_a:.3f} [{ci_a[0]:.3f}, {ci_a[1]:.3f}] vs "
              f"{auc_b:.3f} [{ci_b[0]:.3f}, {ci_b[1]:.3f}] | p={dp:.4f} (Holm {adj[2*i]:.4f})")
        print(f"    McNemar: p={mp:.4f} (Holm {adj[2*i+1]:.4f}) | odds ratio {odds:.2f}")

    # 4) Calibration of the final model
    print(f"\nCalibration (ESM-2 final): ECE {ece(y, p_esm):.3f} | Brier {brier(y, p_esm):.3f}")


if __name__ == "__main__":
    main()