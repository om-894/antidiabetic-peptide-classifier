"""
Phase 6.6: charge composition of the docked panel, for Section 4.2.

Section 4.2 argues that DPP-IV recognises one positive charge at one atom, the
protonated N-terminus held by Glu205/Glu206, while HADDOCK's Coulomb term reads
every formal charge in the sequence. That argument needs two counts that no
earlier phase writes out. First, where each docked peptide's charge sits, since
side-chain Lys and Arg have no recognition site at this target. Second, whether
selecting the classifier's most confident rejections selected cationic peptides,
which is what makes the docking control non-independent.

INPUT   results/phase5_5_docking.csv        the eight docked peptides
        data/dataset_split.csv              the 45 held-out hard negatives
        predictions/esm2_dora_predictions.npz, predictions/base_tree_predictions.npz
        docking/receptor_DPP4_4A5S.pdb, docking/haddock_active_residues.txt
OUTPUT  results/phase6_6_docking_charge.csv  per docked peptide
        results/phase6_6_charge_summary.csv  the numbers quoted in Section 4.2

REQUIREMENTS  pip install pandas numpy peptides
"""

import os
import numpy as np
import pandas as pd
import peptides

DOCKING = "results/phase5_5_docking.csv"
SPLIT = "data/dataset_split.csv"
ESM = "predictions/esm2_dora_predictions.npz"
TREE = "predictions/base_tree_predictions.npz"
RECEPTOR = "docking/receptor_DPP4_4A5S.pdb"
ACTIVE = "docking/haddock_active_residues.txt"
OUT_PEP = "results/phase6_6_docking_charge.csv"
OUT_SUM = "results/phase6_6_charge_summary.csv"

N_CONTROLS = 3 # negative controls docked, taken as the lowest-consensus rejections
ACIDIC = {"ASP", "GLU"}
BASIC = {"ARG", "LYS"}


def consensus_by_sequence():
    """Consensus score for every train and test row, keyed by sequence.

    The consensus is the mean of the ESM-2/DoRA and XGBoost probabilities, as in
    phase5.3. Both npz files store out-of-fold probabilities for the train rows
    followed by test probabilities, in the row order of the sequences array.
    """
    esm, tree = np.load(ESM, allow_pickle=True), np.load(TREE, allow_pickle=True)
    assert np.array_equal(esm["y_train"], tree["y_train"]), "train rows misaligned"
    assert np.array_equal(esm["y_test"], tree["y_test"]), "test rows misaligned"
    p = 0.5 * (np.concatenate([esm["esm_oof"], esm["esm_test"]]).astype(float)
               + np.concatenate([tree["xgb_oof"], tree["xgb_test"]]).astype(float))
    seqs = [str(s) for s in esm["sequences"]]
    assert len(seqs) == len(p), "sequence and probability counts differ"
    return dict(zip(seqs, p))


def active_site_composition():
    """Acidic and basic residue counts over the restrained active site.

    The active-site list is the 22 chain-A residues within 5 A of the
    co-crystallised ligand, as written by phase5.5_docking_prep. Residue names
    come from the CA lines of the prepared receptor.
    """
    line = next(l for l in open(ACTIVE) if l.startswith("receptor"))
    sites = [int(x) for x in line.split(":", 1)[1].strip().split(",")]
    names = {}
    for ln in open(RECEPTOR):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            names[int(ln[22:26])] = ln[17:20].strip()
    resn = [names[s] for s in sites if s in names]
    return len(sites), sum(r in ACIDIC for r in resn), sum(r in BASIC for r in resn)


def main():
    os.makedirs("results", exist_ok=True)
    cons = consensus_by_sequence()

    # Per docked peptide: net charge and where that charge sits. Only Lys and Arg
    # are counted as basic side chains, since His is largely neutral at pH 7.4.
    d = pd.read_csv(DOCKING)
    d["net_charge_pH7.4"] = [round(peptides.Peptide(s).charge(pH=7.4), 3) for s in d.peptide]
    d["n_lys_arg"] = [sum(s.count(a) for a in "KR") for s in d.peptide]
    d["n_his"] = [s.count("H") for s in d.peptide]
    d["n_asp_glu"] = [sum(s.count(a) for a in "DE") for s in d.peptide]
    d["consensus"] = [round(cons[s], 4) if s in cons else np.nan for s in d.peptide]
    cols = ["role", "peptide", "net_charge_pH7.4", "n_lys_arg", "n_his", "n_asp_glu",
            "elec", "elec_pct_of_favourable", "consensus"]
    d[cols].to_csv(OUT_PEP, index=False)

    # Did selecting on classifier confidence select cationic peptides? Rank the 45
    # held-out hard negatives by consensus, then compare the lowest N, which are
    # the docked controls, against the rest of that set.
    split = pd.read_csv(SPLIT, keep_default_na=False)
    hard = split[(split.Label == 0) & (split.NegType == "hard")
                 & (split.Split == "test")].copy()
    hard["consensus"] = [cons[s] for s in hard.Sequence]
    hard["charge"] = [peptides.Peptide(str(s)).charge(pH=7.4) for s in hard.Sequence]
    hard = hard.sort_values("consensus").reset_index(drop=True)
    picked, rest = hard.head(N_CONTROLS), hard.tail(len(hard) - N_CONTROLS)

    # The docked controls must be the lowest-consensus rows, or the selection
    # claim in Section 4.2 does not describe what was actually docked.
    docked = set(d.loc[d.role == "-ve control", "peptide"])
    assert set(picked.Sequence) == docked, "docked controls are not the lowest-consensus rows"

    n_site, n_acid, n_base = active_site_composition()
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
        ("active_site_n_residues", n_site),
        ("active_site_n_acidic", n_acid),
        ("active_site_n_basic", n_base),
    ]
    pd.DataFrame(rows, columns=["quantity", "value"]).to_csv(OUT_SUM, index=False)

    print(d[cols].to_string(index=False))
    print()
    print(pd.DataFrame(rows, columns=["quantity", "value"]).to_string(index=False))
    print(f"\nsaved {OUT_PEP} and {OUT_SUM}")


if __name__ == "__main__":
    main()