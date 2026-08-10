
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
import os
import pandas as pd

SEED, N_BOOT = 42, 1000 # fixed seed = reproducible; 1000 bootstrap resamples

RESULTS_DIR = "results" # save numbers to results directory

# every model scored on the identical 178-row test set
MODELS = [
    ("AAC + RF (baseline)", "predictions/aac_baseline_predictions.npz", "aac_test"),
    ("Random Forest", "predictions/base_tree_predictions.npz", "rf_test"),
    ("XGBoost", "predictions/base_tree_predictions.npz", "xgb_test"),
    ("1D-CNN", "predictions/base_cnn_predictions.npz", "cnn_test"),
    ("ESM-2 / DoRA", "predictions/esm2_dora_predictions.npz", "esm_test"),
    ("Stacked ensemble", "predictions/stack_predictions.npz", "stack_test"),
    ("ESM-2 / DoRA (Basith negatives)", "predictions/esm2_dora_basith_predictions.npz", "esm_test"),
    ("XGBoost (tuned)", "predictions/base_tree_predictions_tuned.npz", "xgb_test"),
    ("Random Forest (tuned)", "predictions/base_tree_predictions_tuned.npz", "rf_test"),
    ("1D-CNN (tuned)", "predictions/base_cnn_predictions_tuned.npz", "cnn_test"),
]

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
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # load every model in MODELS. a missing npz is skipped
    scored, y = [], None
    for name, path, key in MODELS:
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} not found")
            continue
        d = np.load(path, allow_pickle=True)
        if y is None:
            y = d["y_test"].astype(int) # true labels (shared test set)

        # same 178 rows in the same order, or the paired tests are invalid
        assert np.array_equal(d["y_test"].astype(int), y), f"{name} scored on a different test set"
        scored.append((name, d[key].astype(float)))

    print(f"test set: {len(y)} rows ({y.sum()} positive)\n")

    # 1) Bootstrap 95% CIs (MLstatkit for acc/F1/AUC, manual for MCC)
    # Each metric is reported as a point estimate [lower bound, upper bound].
    print("Bootstrap 95% CIs (point [lo, hi]):")
    ci_rows = [] # one row per model, becomes table 3-1
    for name, p in scored:
        row, out = {"model": name}, []
        for label, key, kw in [("ACC", "accuracy", {}),
                               ("F1",  "f1",       {"average": "binary"}),
                               ("AUC", "roc_auc",  {})]:

            # average="binary" is required; MLstatkit defaults to macro, ~0.02 lower
            # and inconsistent with the F1 every notebook here reports
            s, lo, hi = Bootstrapping(y, p, key, n_bootstraps=N_BOOT, random_state=SEED, **kw)
            row |= {label: round(s, 4), f"{label}_lo": round(lo, 4), f"{label}_hi": round(hi, 4)}
            out.append(f"{label} {s:.3f} [{lo:.3f}, {hi:.3f}]")

        s, lo, hi = bootstrap_mcc(y, p) # MLstatkit has no MCC
        row |= {"MCC": round(s, 4), "MCC_lo": round(lo, 4), "MCC_hi": round(hi, 4)} # |= merges dicts into another one in its place
        out.append(f"MCC {s:.3f} [{lo:.3f}, {hi:.3f}]")

        row |= {"ECE": round(ece(y, p), 4), "Brier": round(brier(y, p), 4)}
        ci_rows.append(row)
        print(f"  {name:34s} " + " | ".join(out)) # all metrics for this model on one line

    # 2) Effect size: Cohen's d derived from each model's AUC
    # A p-value says whether a difference exists; an effect size says how large.
    # AUC2OR converts an AUC to Cohen's d (the standardised separation between the
    # positive and negative score distributions) under a binormal assumption.
    print("\nEffect size (Cohen's d from AUC):")
    for row in ci_rows:
        _, d, _, OR = AUC2OR(row["AUC"], return_all=True) # reuse the AUC from step 1
        row |= {"cohens_d": round(d, 3), "odds_ratio": round(OR, 2)}
        print(f"  {row['model']:34s} AUC {row['AUC']:.3f} -> Cohen's d {d:.2f}, odds ratio {OR:.2f}")

    # 3) The two internal comparisons
    # DeLong's test asks whether the AUC difference is significant (correlated ROC curves)
    # McNemar's test asks whether the label predictions differ on matched samples
    P = dict(scored)
    comparisons = [
        ("stack vs ESM-2 (Aim 3)", P["Stacked ensemble"], P["ESM-2 / DoRA"]), # does the ensemble help?
        ("dual-neg vs Basith (ablation)", P["ESM-2 / DoRA"], P["ESM-2 / DoRA (Basith negatives)"]), # does the negative class help?
    ]
    print("\nPairwise comparisons:")
    raw_p, rows = [], []
    for label, pa, pb in comparisons:

        # DeLong returns z-stat, p-value, 95% CIs and AUCs for both models
        _, dp, ci_a, ci_b, auc_a, auc_b, _ = Delong_test(
            y, pa, pb, return_ci=True, return_auc=True, random_state=SEED)

        # McNemar returns p-value, counts of discordant pairs and odds ratio
        mp, b, c, odds = mcnemar(y, (pa > 0.5).astype(int), (pb > 0.5).astype(int))
        raw_p += [dp, mp] # collect all p-values for correction
        rows.append((label, dp, auc_a, auc_b, ci_a, ci_b, mp, b, c, odds))

    # Apply Holm-Bonferroni correction to the raw p-values from all tests
    adj = holm_bonferroni(np.array(raw_p)) # correct across all 4 tests at once

    # Print the results of each comparison, including the adjusted p-values.
    # .4g not .4f, so 4e-05 doesn't print as 0.0000.
    test_rows = [] # one row per test, becomes the pairwise-tests CSV
    for i, (label, dp, auc_a, auc_b, ci_a, ci_b, mp, b, c, odds) in enumerate(rows):
        print(f"\n  {label}")
        print(f"    DeLong : AUC {auc_a:.3f} [{ci_a[0]:.3f}, {ci_a[1]:.3f}] vs "
              f"{auc_b:.3f} [{ci_b[0]:.3f}, {ci_b[1]:.3f}] | p={dp:.4g} (Holm {adj[2*i]:.4g})")
        print(f"    McNemar: p={mp:.4g} (Holm {adj[2*i+1]:.4g}) | {b} vs {c} discordant | OR {odds:.2f}")

        # keys differ per test; pandas fills the gaps
        test_rows += [
            {"comparison": label, "test": "DeLong", "p_raw": dp, "p_holm": adj[2*i],
             "auc_a": round(auc_a, 4), "auc_b": round(auc_b, 4),
             "ci_a_lo": round(ci_a[0], 4), "ci_a_hi": round(ci_a[1], 4),
             "ci_b_lo": round(ci_b[0], 4), "ci_b_hi": round(ci_b[1], 4)},
            {"comparison": label, "test": "McNemar", "p_raw": mp, "p_holm": adj[2*i+1],
             "b": b, "c": c, "odds_ratio": round(odds, 3)},
        ]

    # 4) Save. metrics_ci.csv is Table 3-1; pairwise_tests.csv is the Aim 3 and ablation stats.
    f1 = os.path.join(RESULTS_DIR, "phase4_1_metrics_ci.csv")
    f2 = os.path.join(RESULTS_DIR, "phase4_1_pairwise_tests.csv")
    pd.DataFrame(ci_rows).to_csv(f1, index=False)
    pd.DataFrame(test_rows).to_csv(f2, index=False)
    print(f"\nsaved -> {f1}\nsaved -> {f2}")


if __name__ == "__main__":
    main()