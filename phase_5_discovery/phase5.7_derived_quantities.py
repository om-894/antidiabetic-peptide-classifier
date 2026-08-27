
"""
Phase 5.7: Collect the numbers Sections 3.3 and 3.4 quote into one sourced table.

Nothing is refitted, re-docked or re-scored. Every value is read from an earlier phase's
output.

  Site: acid, base and aromatic composition of the 22 restrained residues.
  Pose: how the contacts spread and the share the three phenylalanines carry.
  Tier 1: why FVAPFPEVF rather than GPFPSIL was the peptide docked.
  Seeds: how far the 412 discoveries rest on the one ESM-2 fit.

INPUTS  haddock_active_residues.txt (the 22 restrained residue numbers)
        receptor_DPP4_4A5S.pdb (residue identities for those numbers)
        phase5_5_pose_contacts.csv (per-residue contact counts within 5 A)
        phase5_5_pose_contacts_full.csv (atom-level contact list)
        tiered_discovery.csv (tier, consensus and safety calls)
        screening_ranked.csv (consensus and XGBoost probabilities)
OUTPUTS  phase5_7_derived_quantities.csv (quantity, value, source)
REQUIREMENTS  pip install pandas
"""

# Imports
import os

import pandas as pd


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

SITE_FILE = "docking/haddock_active_residues.txt"
RECEPTOR = "docking/receptor_DPP4_4A5S.pdb"
POSE_CONTACTS = "results/phase5_5_pose_contacts.csv"
POSE_CONTACTS_FULL = "results/phase5_5_pose_contacts_full.csv"
TIERED = "screening/tiered_discovery.csv"
RANKED = "screening/screening_ranked.csv"
OUT_CSV = "results/phase5_7_derived_quantities.csv"

# named only so the source column can cite where the toxicity cut came from
SAFETY_NOTEBOOK = "phase_5_discovery/phase5.4_safety_screening.ipynb"

TOX_CUT = 0.6 # identical to the ToxinPred2 gate in phase 5.4
CONTACT_CUT = 5.0 # identical to the n_contacts_within_5A column of phase 5.5
DISCOVERY_THRESHOLD = 0.90 # identical to phase5.3_score_and_rank.py

# define the three residue classes that Section 3.4 counts in the active site
ACIDIC = {"ASP", "GLU"}
BASIC = {"ARG", "LYS"}
AROMATIC = {"PHE", "TYR", "TRP"}
LEAD = "FVAPFPEVF"


# --------------------------------------------------------------------------- #
# ACTIVE SITE
# --------------------------------------------------------------------------- #

def site_composition():
    """Acid, base and aromatic counts over the 22 restrained receptor residues."""
    # the restraint file lists residue numbers only, so identities come from the receptor
    # PDB that was actually docked rather than from the RCSB entry
    line = next(l for l in open(SITE_FILE) if l.startswith("receptor"))
    site = [int(n) for n in line.split(":", 1)[1].split(",")]

    want = set(site) # test against 5,960 atom lines, so not the list
    name = {}
    for l in open(RECEPTOR):
        if l.startswith("ATOM"):
            n = int(l[22:26])
            if n in want:
                name.setdefault(n, l[17:20].strip())
    missing = [n for n in site if n not in name]
    assert not missing, f"site residues absent from the receptor PDB: {missing}"

    out = [("site_residues_total", len(site), SITE_FILE)]
    for label, kinds in [("acidic", ACIDIC), ("basic", BASIC), ("aromatic", AROMATIC)]:
        members = [f"{name[n]}{n}" for n in site if name[n] in kinds]
        out += [(f"site_{label}_n", len(members), RECEPTOR),
                (f"site_{label}", ", ".join(members), RECEPTOR)]
    return out


# --------------------------------------------------------------------------- #
# POSE CONTACTS
# --------------------------------------------------------------------------- #

def pose_contacts():
    """How the pose spreads its contacts and how many the phenylalanines carry."""
    pc = pd.read_csv(POSE_CONTACTS)
    glu = pc[pc.residue == "Glu7"].iloc[0]
    phe = pc[pc.residue.str.startswith("Phe")]
    total = int(pc.n_contacts_within_5A.sum())
    out = [("pose_contacts_total", total, POSE_CONTACTS),
           ("pose_contacts_phe", int(phe.n_contacts_within_5A.sum()), POSE_CONTACTS),
           ("pose_contacts_glu7", int(glu.n_contacts_within_5A), POSE_CONTACTS),
           ("glu7_nearest_site_residue", int(glu.nearest_site_residue), POSE_CONTACTS),
           ("glu7_nearest_dist_A", float(glu.nearest_dist), POSE_CONTACTS)]

    # coerce rather than filter, since a non-numeric distance becomes NaN and NaN fails
    # the cutoff test below on its own
    full = pd.read_csv(POSE_CONTACTS_FULL)
    full["distance"] = pd.to_numeric(full.distance, errors="coerce")

    # isin(pc.residue) drops the alpha-amino rows, which are reported distances rather
    # than contacts and lie beyond the cutoff in any case
    real = full[(full.distance <= CONTACT_CUT) & full.pep_residue.isin(pc.residue)]
    assert len(real) == total, "the two contact files disagree on the number of contacts"

    arom = real[real.pep_residue.str.startswith("Phe")
                & real.site_residue.str[:3].isin(AROMATIC)]
    partners = sorted(arom.site_residue.unique())
    return out + [("phe_contacts_to_aromatic", len(arom), POSE_CONTACTS_FULL),
                  ("aromatic_partners_distinct", len(partners), POSE_CONTACTS_FULL),
                  ("aromatic_partners", ", ".join(partners), POSE_CONTACTS_FULL)]


# --------------------------------------------------------------------------- #
# WHICH TIER-1 PEPTIDE WAS DOCKED
# --------------------------------------------------------------------------- #

def tier1_choice():
    """The top-ranked Tier-1 peptide against the one docked, with the tox score between."""
    t1 = (pd.read_csv(TIERED).query("tier == 1")
            .sort_values("consensus", ascending=False))
    describe = lambda r: f"{r.sequence} (consensus {r.consensus}, ToxinPred2 {r.tox_score})"
    return [("tier1_top_ranked", describe(t1.iloc[0]), TIERED),
            ("tier1_top_passing_toxinpred2",
             describe(t1[t1.tox_score < TOX_CUT].iloc[0]), TIERED),
            ("toxinpred2_cut", TOX_CUT, SAFETY_NOTEBOOK)]


# --------------------------------------------------------------------------- #
# DEPENDENCE ON THE SINGLE ESM-2 SEED
# --------------------------------------------------------------------------- #

def seed_dependence():
    """How far the 412 discoveries rest on the one ESM-2 fit rather than on both models.

    Section 3.1 reports ESM-2 test AUC moving by +/-0.026 across seeds 42 to 46 against
    +/-0.002 for XGBoost, so the consensus rests on one stable half and one that is not.
    Nothing is refit here. The margin is how far the ESM-2 probability could fall before
    the consensus drops below the phase 5.3 cutoff, with the tree at its seed-42 value."""
    rank = pd.read_csv(RANKED)
    disc = rank[rank.consensus >= DISCOVERY_THRESHOLD].copy()
    disc["esm_margin"] = 2 * (disc.consensus - DISCOVERY_THRESHOLD)
    tree_alone = disc.xgb_prob >= DISCOVERY_THRESHOLD
    return [("shortlist_n", len(disc), RANKED),
            ("shortlist_tree_alone", int(tree_alone.sum()), RANKED),
            ("shortlist_esm_dependent", int((~tree_alone).sum()), RANKED),
            ("shortlist_xgb_min", round(disc.xgb_prob.min(), 3), RANKED),
            ("esm_margin_median", round(disc.esm_margin.median(), 3), RANKED),
            ("esm_margin_max", round(disc.esm_margin.max(), 3), RANKED),
            ("lead_xgb_prob",
             float(disc.loc[disc.sequence == LEAD, "xgb_prob"].iloc[0]), RANKED)]


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    rows = site_composition() + pose_contacts() + tier1_choice() + seed_dependence()
    out = pd.DataFrame(rows, columns=["quantity", "value", "source"])
    out["source"] = out.source.map(os.path.basename) # cite the file, not the path

    os.makedirs("results", exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"{len(out)} quantities -> {OUT_CSV}")


if __name__ == "__main__":
    main()