
"""
Phase 4.1: Bootstrap confidence intervals and paired significance tests.

A point estimate on 178 test rows is noisy, so each metric is reported with a
95% interval and each comparison with a p-value rather than a bare difference.
Nothing is retrained here. The script reads saved test-set probabilities, and
every model must be scored on the identical 178 rows, which is what makes the
paired tests valid.

  DeLong -> compares two AUCs, allowing for correlated ROC curves.
  McNemar -> compares two sets of threshold calls over the matched rows.

Two pairs are tested here, the stacked ensemble against its best single member
and the dual-negative model against ESM-2 / DoRA (Basith negatives). Phase 6.3
tests a third pair, BertADP against the dual-negative model, on negatives only.
That is five tests, all corrected together with Holm-Bonferroni, so this script
reads the phase 6.3 result rather than leaving it corrected on its own.

INPUTS  the npz files named in MODELS, each with y_test and a *_test key
        results/phase6_3_benchmark_tests.csv, for the third pair's McNemar
OUTPUTS  phase4_1_metrics_ci.csv (Table 7, also read by figures.ipynb)
         phase4_1_pairwise_tests.csv (the three comparisons)
REQUIREMENTS  pip install MLstatkit statsmodels scikit-learn pandas numpy
"""

# Imports
import os
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, matthews_corrcoef
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_exact
from statsmodels.stats.multitest import multipletests
from MLstatkit import AUC2OR, Bootstrapping, Delong_test


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

SEED = 42
N_BOOT = 1000 # bootstrap resamples, the figure section 2.5 reports
RESULTS_DIR = "results" # repo root, as in every other phase script

# the third pair is tested in phase 6.3 because only that script loads the
# BertADP predictions. its raw p is read back here so one correction covers all
# five tests rather than four here and one there
BENCH_TESTS = os.path.join(RESULTS_DIR, "phase6_3_benchmark_tests.csv")

# figures.ipynb matches rows of the metrics CSV on these names, so they are load
# bearing rather than labels. the three tuned variants never reach a table, they
# are here because section 4 quotes the 0.011 AUC gap between XGBoost and its
# tuned version and that gap has to come from somewhere
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
# MCC BOOTSTRAP
# --------------------------------------------------------------------------- #

def bootstrap_mcc(y, p, n=N_BOOT, seed=SEED):
    """Point estimate and 95% percentile interval for MCC."""
    # MLstatkit covers accuracy, F1 and AUC but not MCC, so the same resampling is
    # done here. drawing the test rows with replacement n times and taking the
    # 2.5th to 97.5th percentiles answers how far MCC would move on a different
    # sample of the same size. percentile rather than BCa, matching MLstatkit's ci.py
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: # never fires at 93 positives of 178, guards small inputs
            continue
        vals.append(matthews_corrcoef(y[idx], p[idx] > 0.5))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return matthews_corrcoef(y, p > 0.5), lo, hi


# --------------------------------------------------------------------------- #
# MCNEMAR
# --------------------------------------------------------------------------- #

def mcnemar(y, pred_a, pred_b):
    """Two-sided exact p, the two discordant counts and their odds ratio."""
    # only the rows where the two models disagree carry information. if the models
    # were equally good, b and c would split evenly, so the test asks whether the
    # observed split departs from 50/50. the exact test reads the off-diagonal
    # cells only, so the concordant corners are passed as zeros
    ac, bc = (pred_a == y), (pred_b == y)
    b = int(np.sum(ac & ~bc)) # A right, B wrong
    c = int(np.sum(~ac & bc)) # A wrong, B right

    p = mcnemar_exact([[0, b], [c, 0]], exact=True).pvalue
    return p, b, c, (b / c) if c else np.inf


# --------------------------------------------------------------------------- #
# CALIBRATION
# --------------------------------------------------------------------------- #

def ece(y, p, bins=10):
    # expected calibration error, the sample-weighted gap between what the model
    # claims and what happens. neither sklearn nor scipy implements it, unlike the
    # Brier score. phase 5 thresholds on the probability itself rather than on the
    # label, so a model can rank well and still be unusable there
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for i in range(bins):
        # bins are half open, (edge, edge]. the first is closed at 0 instead, so a
        # prediction of exactly 0 lands somewhere rather than being dropped
        mask = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= 0) & (p <= edges[1])
        if mask.sum():
            e += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return e


# --------------------------------------------------------------------------- #
# BENCHMARK TEST
# --------------------------------------------------------------------------- #

def benchmark_mcnemar(path=BENCH_TESTS):
    """The phase 6.3 McNemar as a raw p and its two discordant counts."""
    # a missing file means phase 6.3 has not been run, which would silently drop the
    # test from the family rather than fail, so it is an error instead
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found, run phase6.3_hardneg_fpr.py first")

    row = pd.read_csv(path).set_index("metric").loc["mcnemar_bertadp_vs_dual"]
    return float(row["value"]), int(row["b_bert_right_ours_wrong"]), int(row["c_bert_wrong_ours_right"])


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # a missing npz is skipped rather than fatal, so a partial run still reports
    # what it has. the assert is the load-bearing part, since DeLong and McNemar
    # are both paired and silently meaningless on mismatched rows
    scored, y = [], None
    for name, path, key in MODELS:
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} not found")
            continue
        d = np.load(path, allow_pickle=True)
        if y is None:
            y = d["y_test"].astype(int)
        assert np.array_equal(d["y_test"].astype(int), y), f"{name} scored on a different test set"
        scored.append((name, d[key].astype(float)))

    ci_rows = []
    for name, p in scored:
        row = {"model": name}
        for label, key, kw in [("ACC", "accuracy", {}),
                               ("F1", "f1", {"average": "binary"}),
                               ("AUC", "roc_auc", {})]:
            # average="binary" is not the MLstatkit default. macro comes out about
            # 0.02 lower here and would not match the F1 the notebooks report
            s, lo, hi = Bootstrapping(y, p, key, n_bootstraps=N_BOOT, random_state=SEED, **kw)
            row |= {label: round(s, 4), f"{label}_lo": round(lo, 4), f"{label}_hi": round(hi, 4)}

        s, lo, hi = bootstrap_mcc(y, p)
        row |= {"MCC": round(s, 4), "MCC_lo": round(lo, 4), "MCC_hi": round(hi, 4)}
        row |= {"ECE": round(ece(y, p), 4), "Brier": round(brier_score_loss(y, p), 4)}
        ci_rows.append(row)

    # a p-value says a difference exists, an effect size says how big. AUC2OR reads
    # the AUC as the separation between the two score distributions under a
    # binormal assumption and converts it to Cohen's d
    for row in ci_rows:
        _, d, _, OR = AUC2OR(row["AUC"], return_all=True)
        row |= {"cohens_d": round(d, 3), "odds_ratio": round(OR, 2)}

    # the two questions section 3.1 asks, one about architecture and one about the
    # negative class. both models in a pair see the same rows, so both tests pair
    P = dict(scored)
    comparisons = [
        ("stack vs ESM-2 (ensembling)", P["Stacked ensemble"], P["ESM-2 / DoRA"]),
        ("dual-neg vs Basith (ablation)", P["ESM-2 / DoRA"], P["ESM-2 / DoRA (Basith negatives)"]),
    ]

    raw_p, rows = [], []
    for label, pa, pb in comparisons:
        # the first return value is the DeLong z. it is kept rather than discarded so
        # the test statistic can be reported alongside the p-value
        z, dp, ci_a, ci_b, auc_a, auc_b, _ = Delong_test(
            y, pa, pb, return_ci=True, return_auc=True, random_state=SEED)
        mp, b, c, odds = mcnemar(y, (pa > 0.5).astype(int), (pb > 0.5).astype(int))
        raw_p += [dp, mp]
        rows.append((label, z, dp, auc_a, auc_b, ci_a, ci_b, mp, b, c, odds))

    # the benchmark pair joins the family here, so it is corrected with the other
    # four rather than reported uncorrected in section 3.2
    bench_p, bench_b, bench_c = benchmark_mcnemar()
    raw_p.append(bench_p)

    # corrected across all five at once, since they are one family of questions.
    # holm is step-down, so it stays more powerful than plain Bonferroni
    adj = multipletests(raw_p, method="holm")[1]

    # DeLong and McNemar write different columns, which pandas fills as blanks
    test_rows = []
    for i, (label, z, dp, auc_a, auc_b, ci_a, ci_b, mp, b, c, odds) in enumerate(rows):
        test_rows += [
            {"comparison": label, "test": "DeLong", "p_raw": dp, "p_holm": adj[2 * i],
             # MLstatkit signs z against auc_b - auc_a, so the magnitude is stored and
             # the direction is left to delta_auc
             "z": round(abs(z), 3), "delta_auc": round(auc_a - auc_b, 4),
             "auc_a": round(auc_a, 4), "auc_b": round(auc_b, 4),
             "ci_a_lo": round(ci_a[0], 4), "ci_a_hi": round(ci_a[1], 4),
             "ci_b_lo": round(ci_b[0], 4), "ci_b_hi": round(ci_b[1], 4)},
            {"comparison": label, "test": "McNemar", "p_raw": mp, "p_holm": adj[2 * i + 1],
             "b": b, "c": c, "odds_ratio": round(odds, 3)},
        ]

    # phase 6.3 owns the counts and the raw p, so only the corrected value is new
    test_rows.append(
        {"comparison": "BertADP vs dual-neg (benchmark)", "test": "McNemar",
         "p_raw": bench_p, "p_holm": adj[4], "b": bench_b, "c": bench_c,
         "odds_ratio": round(bench_b / bench_c, 3) if bench_c else np.inf})

    pd.DataFrame(ci_rows).to_csv(os.path.join(RESULTS_DIR, "phase4_1_metrics_ci.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(RESULTS_DIR, "phase4_1_pairwise_tests.csv"), index=False)

if __name__ == "__main__":
    main()