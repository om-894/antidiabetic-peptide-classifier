

"""
Phase 5.0: Calibration curve for the consensus score - justifies the discovery threshold.

A "top-N" cutoff is arbitrary; a probability cutoff is only meaningful if the model is
calibrated there. This builds a reliability diagram for the consensus (per bin: claimed
confidence vs true positive rate) so the discovery threshold (e.g. 0.90) can be set where
claimed is comparable to actual i.e. a score of 0.90 really does mean 90% chance of being an ADP.

Uses the out-of-fold train predictions (1754 rows), not the 178-row test set, since the
high-probability bins that set the threshold need enough samples. Score = consensus =
mean(ESM-2/DoRA OOF, XGBoost OOF), as ranked in Phase 5.3.

Caveat (write-up): calibration is measured in-distribution; screening candidates are 
OOD (out-of-distribution) food fragments with low ADP prevalence, so field precision 
sits below this figure - the control dockings (Phase 5.5) are the independent check.

INPUT   predictions/esm2_dora_predictions.npz   esm_oof, y_train
        predictions/base_tree_predictions.npz    xgb_oof, y_train
OUTPUT  screening/consensus_calibration.csv       per-bin reliability table
        results/phase5_0_calibration_summary.csv  ECE, row counts and operating points

REQUIREMENTS  pip install numpy pandas
"""

# Imports
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
ESM_NPZ = "predictions/esm2_dora_predictions.npz"
TREE_NPZ = "predictions/base_tree_predictions.npz"
OUT_CSV = "screening/consensus_calibration.csv"
OUT_PNG = "screening/consensus_calibration.png"
SUM_CSV = "results/phase5_0_calibration_summary.csv"

TARGET_PRECISION = 0.90 # supervisor's goal: discoveries with >=~90% verified positive rate
MIN_BIN_N = 30 # don't trust a threshold whose surviving set is tiny
BINS = 10 # reliability-diagram resolution


def reliability_table(y, p, bins=BINS):
    """One row per probability bin = one point on the calibration curve.

    The idea: group predictions by how confident the model was, then in each group
    compare the model's average claim (mean_pred) to what actually happened (obs_pos,
    the true positive fraction). A well-calibrated model has mean_pred ~ obs_pos in
    every bin - i.e. of everything it scored ~0.9, about 90% really are positive.
    """
    edges, rows = np.linspace(0, 1, bins + 1), [] # bin boundaries: [0, .1, .2, ..., 1]
    for i in range(bins):

        # pick the predictions whose probability falls in this bin, (edges[i], edges[i+1]].
        # the first bin is made inclusive of 0 so a prediction of exactly 0 isn't dropped.
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= 0) & (p <= edges[1])
        if m.sum(): # skip empty bins (nothing to compare)
            rows.append({
                "bin":       f"({edges[i]:.1f}, {edges[i+1]:.1f}]",
                "n":         int(m.sum()), # how many predictions in this bin (its weight/trust)
                "mean_pred": p[m].mean(), # the model's average claimed probability here
                "obs_pos":   y[m].mean(), # the observed fraction that are truly positive
            })
    return pd.DataFrame(rows) # compare mean_pred vs obs_pos -> the calibration curve


def operating_points(y, p, thresholds):
    """One row per candidate threshold -> this is what actually picks the cutoff.

    Where reliability_table looks within each bin, this looks at everything at or above
     a cutoff (the set to keep as 'discoveries'). For each cutoff it
    reports how many candidates survive and what fraction of them are truly positive
    i.e. the precision you'd get if you drew the discovery line there. The chosen
    threshold is the lowest cutoff whose survivors hit the target precision.
    """
    rows = []
    for t in thresholds:
        m = p >= t # everything the cutoff would keep
        rows.append({
            "threshold": round(float(t), 2),
            "n_above": int(m.sum()), # how many candidates survive this cutoff
            
            # fraction of survivors that are truly positive = precision at this cutoff.
            # nan if nothing survives (can't take a mean of an empty set).
            "empirical_pos_rate": (y[m].mean() if m.sum() else float("nan")),
        })
    return pd.DataFrame(rows)


def ece(y, p, bins=BINS):
    """Expected Calibration Error - one-number summary of the diagram (0 = perfect).
    Same binning as phase 4.1, applied to the consensus."""
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= 0) & (p <= edges[1])
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e



def main():
    # Load the aligned OOF probabilities (stacking contract guarantees identical row
    # order: StratifiedKFold(5, shuffle, random_state=42) over the train rows).
    esm, tree = np.load(ESM_NPZ, allow_pickle=True), np.load(TREE_NPZ, allow_pickle=True)
    y = esm["y_train"].astype(int)
    assert np.array_equal(y, tree["y_train"].astype(int)), "OOF label order mismatch" # sanity check
    consensus = 0.5 * (esm["esm_oof"].astype(float) + tree["xgb_oof"].astype(float))
    print(f"OOF rows: {len(y)} ({y.sum()} positive) | consensus ECE {ece(y, consensus):.3f}")

    # Reliability diagram (per-bin) + operating points (cumulative).
    rel = reliability_table(y, consensus)
    print("\nReliability (per bin) -- mean_pred should track obs_pos:")
    print(rel.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    ops = operating_points(y, consensus, np.arange(0.50, 0.991, 0.05))
    print("\nOperating points (candidates at/above each cutoff, and their true-positive rate):")
    print(ops.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Recommended threshold: the lowest cutoff whose surviving set is >=TARGET_PRECISION
    # positive and still well-populated (>=MIN_BIN_N). This keeps as many candidates as
    # possible while honouring the "verified ~90% probability" goal.
    grid = operating_points(y, consensus, np.arange(0.50, 0.991, 0.01))
    ok = grid[(grid.empirical_pos_rate >= TARGET_PRECISION) & (grid.n_above >= MIN_BIN_N)]
    thr = float(ok.threshold.min()) if len(ok) else float("nan")
    print(f"\nRecommended discovery threshold: consensus >= {thr:.2f}  "
          f"(>= {TARGET_PRECISION:.0%} verified positive rate on held-out data)")

    # persist the quoted numbers; previously print-only
    esm_oof, xgb_oof = esm["esm_oof"].astype(float), tree["xgb_oof"].astype(float)
    rows = [("n_oof", len(y)),
            ("n_oof_positive", int(y.sum())),
            ("ece_consensus_oof", round(ece(y, consensus), 4)),
            ("ece_esm2_oof", round(ece(y, esm_oof), 4)),
            ("ece_xgb_oof", round(ece(y, xgb_oof), 4)),
            ("recommended_threshold", thr)]
    for t in (thr, 0.90):
        r = grid[grid.threshold == round(t, 2)].iloc[0] # operating point at this cutoff
        rows += [(f"n_above_{t:.2f}", int(r.n_above)),
                 (f"pos_rate_{t:.2f}", round(float(r.empirical_pos_rate), 4))]

    rel.to_csv(OUT_CSV, index=False)
    pd.DataFrame(rows, columns=["quantity", "value"]).to_csv(SUM_CSV, index=False)
    print(f"saved {OUT_CSV} and {SUM_CSV}")


if __name__ == "__main__":
    main()