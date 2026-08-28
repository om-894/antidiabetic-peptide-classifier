
"""
Phase 6.5: Where the charge shortcut comes from.

The classifiers under-score cationic peptides. This measures that in three places.
First in the training data, where the antimicrobial negatives are strongly cationic and
the proteome fragments are not, so charge separates the positives from one half of the
negative class and barely from the other. Second in the DBAASP export before any
filtering, since a skew already present there is a property of the antimicrobial class
rather than of the draw taken from it. Third in the fine-tuned model's own behaviour,
where the held-out hard negatives it wrongly accepts are anionic and the ones it
correctly rejects are cationic, and in the screen output that follows from it.

INPUTS  dataset_split.csv, esm2_dora_predictions.npz
        peptides.csv (the raw DBAASP export, before any filtering)
        screening_ranked.csv (every candidate with its consensus score)
OUTPUTS  phase6_5_charge_source.csv (one row per quantity, with the n behind it)
REQUIREMENTS  pip install pandas numpy peptides scikit-learn scipy
"""

# Imports
import os

import numpy as np
import pandas as pd
import peptides
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
SPLIT = "data/dataset_split.csv"
ESM = "predictions/esm2_dora_predictions.npz"
SCREEN = "screening/screening_ranked.csv"
SOFT_CSV = "data/peptides.csv" # raw DBAASP export, before any filtering
OUT = "results/phase6_5_charge_source.csv"

CATIONIC = 0.5 # net charge above this counts as cationic

# the same 20 residues phase 1.2 filtered the soft pool on, so the export measured here
# and the 598 drawn from it are judged by one test
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def charges(seqs):
    """Net charge at pH 7.4 per sequence, the same call phase 2.2 makes for its descriptor."""
    return [peptides.Peptide(str(s)).charge(pH=7.4) for s in seqs]


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    os.makedirs("results", exist_ok=True)
    d = pd.read_csv(SPLIT, keep_default_na=False)
    d["charge"] = charges(d.Sequence)
    tr = d[d.Split == "train"]
    rows = []

    # composition of each training class. the antimicrobial negatives carry the charge,
    # the proteome fragments sit near the positives
    for name, g in [("positives", tr[tr.Label == 1]),
                    ("soft_antimicrobial", tr[(tr.Label == 0) & (tr.NegType == "soft")]),
                    ("hard_swissprot", tr[(tr.Label == 0) & (tr.NegType == "hard")])]:
        rows.append({"quantity": f"train_mean_charge_{name}",
                     "value": round(g.charge.mean(), 3), "n": len(g)})
        rows.append({"quantity": f"train_pct_cationic_{name}",
                     "value": round(100 * (g.charge > CATIONIC).mean(), 1), "n": len(g)})

    # the source pool before any filtering. if the export is as cationic as the 598 drawn
    # from it, the skew belongs to the antimicrobial class rather than to the draw, so no
    # larger or more careful sample would have helped
    src = pd.read_csv(SOFT_CSV, keep_default_na=False)
    seqs = src["SEQUENCE"].astype(str).str.strip()
    seqs = seqs[seqs != ""].drop_duplicates()
    std = seqs[[set(s).issubset(STANDARD_AA) for s in seqs]] # as in phase 1.2's is_standard
    for name, ss in [("dbaasp_export", seqs),
                     ("dbaasp_export_standard_aa", std)]:
        ch = np.array(charges(ss))
        rows.append({"quantity": f"source_mean_charge_{name}",
                     "value": round(float(ch.mean()), 3), "n": len(ch)})
        rows.append({"quantity": f"source_pct_cationic_{name}",
                     "value": round(100 * float((ch > CATIONIC).mean()), 1), "n": len(ch)})

    # can charge alone tell the training classes apart. label 1 = is a negative, so an AUC
    # above 0.5 means a higher charge marks a negative
    for name, sub in [("all_negatives", tr),
                      ("soft_only", tr[(tr.Label == 1) | (tr.NegType == "soft")]),
                      ("hard_only", tr[(tr.Label == 1) | (tr.NegType == "hard")])]:
        rows.append({"quantity": f"train_charge_auc_positives_vs_{name}",
                     "value": round(roc_auc_score((sub.Label == 0).astype(int), sub.charge), 4),
                     "n": len(sub)})

    # the model's own calls on the held-out hard negatives. every one is a true non-ADP,
    # so any probability above 0.5 is a false positive. the npz holds every sequence but
    # only the test probabilities, so its tail lines up with the test rows, as in phase 3.6.4
    z = np.load(ESM, allow_pickle=True)
    p = z["esm_test"].astype(float)
    test = d[d.Split == "test"].reset_index(drop=True)
    assert [str(s) for s in z["sequences"]][-len(p):] == test.Sequence.tolist(), "test rows misaligned"
    test["p"] = p

    hard = test[(test.Label == 0) & (test.NegType == "hard")]
    fp = hard.p > 0.5
    rows += [
        {"quantity": "test_hardneg_mean_charge_false_positive",
         "value": round(hard.charge[fp].mean(), 3), "n": int(fp.sum())},
        {"quantity": "test_hardneg_mean_charge_correct_reject",
         "value": round(hard.charge[~fp].mean(), 3), "n": int((~fp).sum())},
        {"quantity": "test_hardneg_charge_auc_predicting_false_positive",
         "value": round(roc_auc_score(fp.astype(int), hard.charge), 4), "n": len(hard)},
    ]

    # the same rule in the screen output. discoveries and non-discoveries come from one
    # pool, so a charge gap between them is the classifier's doing
    r = pd.read_csv(SCREEN, keep_default_na=False)
    r["charge"] = charges(r.sequence)
    disc = r[r.discovery]

    # the p is kept beside the coefficient, so the results section can cite both
    rho, rho_p = spearmanr(r.consensus, r.charge)
    rows += [
        {"quantity": "screen_mean_charge_all_candidates",
         "value": round(r.charge.mean(), 3), "n": len(r)},
        {"quantity": "screen_pct_cationic_all_candidates",
         "value": round(100 * (r.charge > CATIONIC).mean(), 1), "n": len(r)},
        {"quantity": "screen_mean_charge_discoveries",
         "value": round(disc.charge.mean(), 3), "n": len(disc)},
        {"quantity": "screen_pct_cationic_discoveries",
         "value": round(100 * (disc.charge > CATIONIC).mean(), 1), "n": len(disc)},
        {"quantity": "screen_spearman_consensus_vs_charge",
         "value": round(rho, 4), "n": len(r)},
        {"quantity": "screen_spearman_p_consensus_vs_charge",
         "value": float(f"{rho_p:.3g}"), "n": len(r)},
    ]

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{len(rows)} quantities -> {OUT}")


if __name__ == "__main__":
    main()