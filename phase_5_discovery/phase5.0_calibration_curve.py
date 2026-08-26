
"""
Phase 5.0: Calibrate the consensus score so the discovery threshold means something.

A top-N cutoff is arbitrary. A probability cutoff only carries meaning where the
score is calibrated, so this builds a reliability diagram for the consensus then
sweeps candidate cutoffs for the lowest one that still returns the target positive
rate. That sweep is what sets the discovery threshold in phase 5.3.

Calibration is measured on the 1,754 out-of-fold training rows rather than the
178-row test set, since the high-probability bins that decide the threshold need
enough samples to be worth trusting.

The screening candidates are food fragments with a far lower ADP prevalence than
this distribution, so the hit rate in the field sits below the figure here. The
control dockings in phase 5.5 are the independent check on that.

INPUTS  esm2_dora_predictions.npz (esm_oof, esm_test, y_train, y_test)
        base_tree_predictions.npz (xgb_oof, xgb_test, y_train)
OUTPUTS  consensus_calibration.csv (per-bin reliability table, plotted in figures.ipynb)
         phase5_0_calibration_summary.csv (every number section 3.3 quotes)
REQUIREMENTS  pip install numpy pandas scikit-learn
"""

# Imports
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

ESM_NPZ = "predictions/esm2_dora_predictions.npz"
TREE_NPZ = "predictions/base_tree_predictions.npz"
OUT_CSV = "screening/consensus_calibration.csv"
SUM_CSV = "results/phase5_0_calibration_summary.csv"

TARGET_PRECISION = 0.90 # the verified positive rate a discovery set has to reach
MIN_BIN_N = 30 # a cutoff leaving fewer survivors than this is not worth trusting
BINS = 10 # same binning phase 4.1 uses for ECE


def reliability_table(y, p, bins=BINS):
    """One row per probability bin, the model's mean claim against what happened."""
    # grouping by confidence is what makes miscalibration visible. a well-calibrated
    # model has mean_pred close to obs_pos in every bin, so of everything it scored
    # around 0.9, about 90% turn out to be positive
    edges, rows = np.linspace(0, 1, bins + 1), []
    for i in range(bins):
        # bins are half open, (edge, edge]. the first is closed at 0 instead, so a
        # prediction of exactly 0 lands somewhere rather than being dropped
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= 0) & (p <= edges[1])
        if m.sum():
            rows.append({"bin": f"({edges[i]:.1f}, {edges[i+1]:.1f}]",
                         "n": int(m.sum()),
                         "mean_pred": p[m].mean(),
                         "obs_pos": y[m].mean()})
    return pd.DataFrame(rows)


def operating_points(y, p, thresholds):
    """One row per candidate cutoff, how many survive it and what share are positive."""
    # where reliability_table looks inside a bin, this looks at everything at or above
    # a cutoff, which is the set a screen would actually keep. empirical_pos_rate is
    # therefore the precision you would get by drawing the discovery line there
    rows = []
    for t in thresholds:
        m = p >= t
        rows.append({"threshold": round(float(t), 2),
                     "n_above": int(m.sum()),
                     "empirical_pos_rate": y[m].mean() if m.sum() else float("nan")})
    return pd.DataFrame(rows)


def ece(y, p, bins=BINS):
    # duplicated from phase 4.1
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= 0) & (p <= edges[1])
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    # both npz hold out-of-fold predictions over the same train rows in the same
    # order, guaranteed by the shared StratifiedKFold in phase 3.3 and 3.1.
    esm, tree = np.load(ESM_NPZ, allow_pickle=True), np.load(TREE_NPZ, allow_pickle=True)
    y = esm["y_train"].astype(int)
    assert np.array_equal(y, tree["y_train"].astype(int)), "OOF label order mismatch"

    esm_oof, xgb_oof = esm["esm_oof"].astype(float), tree["xgb_oof"].astype(float)
    consensus = 0.5 * (esm_oof + xgb_oof) # unweighted, as ranked in phase 5.3

    # 0.01 steps rather than a large sweep, so the recommendation is not rounded
    # up to a grid point that throws away candidates
    grid = operating_points(y, consensus, np.arange(0.50, 0.991, 0.01))
    ok = grid[(grid.empirical_pos_rate >= TARGET_PRECISION) & (grid.n_above >= MIN_BIN_N)]
    thr = float(ok.threshold.min()) if len(ok) else float("nan")

    # summary table for section 3.3
    rows = [("n_oof", len(y)),
            ("n_oof_positive", int(y.sum())),
            ("ece_consensus_oof", round(ece(y, consensus), 4)),
            ("ece_esm2_oof", round(ece(y, esm_oof), 4)),
            ("ece_xgb_oof", round(ece(y, xgb_oof), 4)),
            ("recommended_threshold", thr)]

    # both cutoffs are reported. the sweep recommends 0.87, phase 5.3 cuts at the
    # rounder 0.90. Explained in section 2.5
    for t in (thr, 0.90):
        r = grid[grid.threshold == round(t, 2)].iloc[0]
        rows += [(f"n_above_{t:.2f}", int(r.n_above)),
                 (f"pos_rate_{t:.2f}", round(float(r.empirical_pos_rate), 4))]

    # the consensus on the held-out test set, quoted in section 3.3 beside the OOF
    # figures. computed here so all four calibration numbers live in one file
    yte = esm["y_test"].astype(int)
    esm_te = esm["esm_test"].astype(float)
    cons_te = 0.5 * (esm_te + tree["xgb_test"].astype(float))
    rows += [("test_auc_consensus", round(roc_auc_score(yte, cons_te), 4)),
             ("test_ece_consensus", round(ece(yte, cons_te), 4)),
             ("test_auc_esm2", round(roc_auc_score(yte, esm_te), 4)),
             ("test_ece_esm2", round(ece(yte, esm_te), 4))]

    # the same 0.90 cutoff applied to the test set, so the out-of-fold rate has a
    # held-out comparison. these are the peptides a threshold set on the training
    # folds would have admitted on data it never saw
    mte = cons_te >= 0.90
    rows += [("n_test_above_0.90", int(mte.sum())),
             ("n_test_above_0.90_positive", int(yte[mte].sum())),
             ("pos_rate_test_0.90", round(float(yte[mte].mean()), 4))]

    reliability_table(y, consensus).to_csv(OUT_CSV, index=False)
    pd.DataFrame(rows, columns=["quantity", "value"]).to_csv(SUM_CSV, index=False)


if __name__ == "__main__":
    main()