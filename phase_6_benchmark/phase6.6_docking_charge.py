
"""
Phase 6.6: Charge composition of the docked panel.

DPP-IV recognises one positive charge at one atom, the protonated N-terminus held by
Glu205 and Glu206, while HADDOCK's Coulomb term reads every formal charge in the
sequence. That argument needs two counts no earlier phase writes out. First, where each
docked peptide's charge sits, since side-chain Lys and Arg have no recognition site at
this target. Second, whether selecting the classifier's most confident rejections also
selected cationic peptides, which is what makes the docking control non-independent.

INPUTS  phase5_5_docking.csv (the eight docked peptides)
        dataset_split.csv (the 45 held-out hard negatives)
        esm2_dora_predictions.npz, base_tree_predictions.npz (for the consensus)
OUTPUTS  phase6_6_docking_charge.csv (per docked peptide)
         phase6_6_charge_summary.csv (the charge numbers to quote)
REQUIREMENTS  pip install pandas numpy peptides
"""

# Imports
import os

import numpy as np
import pandas as pd
import peptides


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
DOCKING = "results/phase5_5_docking.csv"
SPLIT = "data/dataset_split.csv"
ESM = "predictions/esm2_dora_predictions.npz"
TREE = "predictions/base_tree_predictions.npz"

OUT_PEP = "results/phase6_6_docking_charge.csv"
OUT_SUM = "results/phase6_6_charge_summary.csv"

N_CONTROLS = 3 # negative controls docked, taken as the lowest-consensus rejections


def charges(seqs):
    """Net charge at pH 7.4 per sequence, the same call phase 2.2 makes for its descriptor."""
    return [peptides.Peptide(str(s)).charge(pH=7.4) for s in seqs]


# --------------------------------------------------------------------------- #
# CONSENSUS
# --------------------------------------------------------------------------- #
def consensus_by_sequence():
    """Consensus score for every train and test row, keyed by sequence.

    The consensus is the mean of the ESM-2/DoRA and XGBoost probabilities, as in phase 5.3.
    Both npz files store out-of-fold probabilities for the train rows followed by test
    probabilities, in the row order of the sequences array."""
    esm, tree = np.load(ESM, allow_pickle=True), np.load(TREE, allow_pickle=True)
    assert np.array_equal(esm["y_train"], tree["y_train"]), "train rows misaligned"
    assert np.array_equal(esm["y_test"], tree["y_test"]), "test rows misaligned"
    p = 0.5 * (np.concatenate([esm["esm_oof"], esm["esm_test"]]).astype(float)
               + np.concatenate([tree["xgb_oof"], tree["xgb_test"]]).astype(float))
    seqs = [str(s) for s in esm["sequences"]]
    assert len(seqs) == len(p), "sequence and probability counts differ"
    return dict(zip(seqs, p))


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    os.makedirs("results", exist_ok=True)
    cons = consensus_by_sequence()

    # per docked peptide, the net charge and where that charge sits. only Lys and Arg
    # count as basic side chains, since His is largely neutral at pH 7.4
    d = pd.read_csv(DOCKING)
    d["net_charge_pH7.4"] = [round(c, 3) for c in charges(d.peptide)]
    d["n_lys_arg"] = [sum(s.count(a) for a in "KR") for s in d.peptide]
    d["n_his"] = [s.count("H") for s in d.peptide]
    d["n_asp_glu"] = [sum(s.count(a) for a in "DE") for s in d.peptide]

    # the leads and positive controls are screening candidates rather than split rows, so
    # they carry no consensus here and come out blank
    d["consensus"] = [round(cons[s], 4) if s in cons else np.nan for s in d.peptide]
    cols = ["role", "peptide", "net_charge_pH7.4", "n_lys_arg", "n_his", "n_asp_glu",
            "elec", "elec_pct_of_favourable", "consensus"]
    d[cols].to_csv(OUT_PEP, index=False)

    # did selecting on classifier confidence select cationic peptides? rank the 45 held-out
    # hard negatives by consensus, then set the lowest three, which are the docked controls,
    # against the rest of that set
    split = pd.read_csv(SPLIT, keep_default_na=False)
    hard = split[(split.Label == 0) & (split.NegType == "hard")
                 & (split.Split == "test")].copy()
    hard["consensus"] = [cons[s] for s in hard.Sequence]
    hard["charge"] = charges(hard.Sequence)
    hard = hard.sort_values("consensus").reset_index(drop=True)
    picked, rest = hard.head(N_CONTROLS), hard.iloc[N_CONTROLS:]

    # the docked controls must be those lowest-consensus rows, or the selection claim does
    # not describe what was actually docked
    docked = set(d.loc[d.role == "-ve control", "peptide"])
    assert set(picked.Sequence) == docked, "docked controls are not the lowest-consensus rows"

    lead = d[d.role == "Tier-1 lead"].iloc[0]
    genuine = d[d.role.isin(["Tier-1 lead", "+ve control"])]
    rows = [
        ("lead_net_charge", lead["net_charge_pH7.4"]),
        ("lead_n_lys_arg", int(lead.n_lys_arg)),
        ("genuine_n_lys_arg_total", int(genuine.n_lys_arg.sum())),
        ("control_n_lys_arg_min", int(d[d.role == "-ve control"].n_lys_arg.min())),
        ("control_n_lys_arg_max", int(d[d.role == "-ve control"].n_lys_arg.max())),
        ("n_test_hard_negatives", len(hard)),
        ("mean_charge_docked_controls", round(picked.charge.mean(), 3)),
        ("mean_charge_remaining_hard_negatives", round(rest.charge.mean(), 3)),
        ("mean_charge_all_test_hard_negatives", round(hard.charge.mean(), 3)),
        ("max_consensus_docked_controls", round(picked.consensus.max(), 4)),
    ]
    pd.DataFrame(rows, columns=["quantity", "value"]).to_csv(OUT_SUM, index=False)

    print(f"{len(d)} docked peptides, {len(hard)} test hard negatives -> {OUT_PEP}, {OUT_SUM}")


if __name__ == "__main__":
    main()