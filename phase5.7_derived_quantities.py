"""
Phase 5.7: export the quantities quoted in Section 3.4 that no earlier phase writes out.

Nothing is refitted, re-docked or re-scored. Every value is read from a phase 5.4 or 5.5
output, or from the receptor and restraint files the docking run itself used, so this
script only makes explicit what those files already imply. It exists because five numbers
in Section 3.4 were otherwise derivable but not recorded, namely the acid/base/aromatic
composition of the active site, the share of pose contacts carried by the three
phenylalanines, and the reason FVAPFPEVF rather than GPFPSIL was the Tier-1 lead docked.

INPUT   docking/haddock_active_residues.txt       the 22 receptor residue numbers restrained
        docking/receptor_DPP4_4A5S.pdb            residue identities for those numbers
        results/phase5_5_pose_contacts.csv        per-residue contact counts within 5 A
        results/phase5_5_pose_contacts_full.csv   atom-level contact list
        screening/tiered_discovery.csv            tier, consensus and safety calls
        screening/screening_ranked.csv            consensus and XGBoost probabilities
OUTPUT  results/phase5_7_derived_quantities.csv   quantity, value, source
"""

import re
import pandas as pd

TOX_CUT = 0.6 # identical to phase5.4_safety_select.py
CONTACT_CUT = 5.0 # identical to the n_contacts_within_5A column of phase 5.5
ACIDIC, BASIC, AROMATIC = {"ASP", "GLU"}, {"ARG", "LYS"}, {"PHE", "TYR", "TRP"}
DISCOVERY_THRESHOLD = 0.90  # identical to phase5.3_score_and_rank.py
LEAD = "FVAPFPEVF"

rows = []
rec = lambda q, v, s: rows.append({"quantity": q, "value": v, "source": s})


def main():
    # ── active-site composition ───────────────────────────────────────────────
    # The restraint file lists residue numbers only, so identities come from the
    # receptor PDB that was actually docked rather than from the RCSB entry.
    line = next(l for l in open("docking/haddock_active_residues.txt")
                if l.startswith("receptor"))
    site = [int(n) for n in re.search(r":(.*)", line).group(1).split(",")]

    name = {}
    for l in open("docking/receptor_DPP4_4A5S.pdb"):
        if l.startswith("ATOM"):
            n = int(l[22:26])
            if n in site:
                name.setdefault(n, l[17:20].strip())
    missing = [n for n in site if n not in name]
    assert not missing, f"site residues absent from the receptor PDB: {missing}"

    listing = lambda kinds: ", ".join(f"{name[n]}{n}" for n in site if name[n] in kinds)
    rec("site_residues_total", len(site), "haddock_active_residues.txt")
    for label, kinds in [("acidic", ACIDIC), ("basic", BASIC), ("aromatic", AROMATIC)]:
        rec(f"site_{label}_n", sum(name[n] in kinds for n in site), "receptor_DPP4_4A5S.pdb")
        rec(f"site_{label}", listing(kinds), "receptor_DPP4_4A5S.pdb")

    # ── how the pose distributes its contacts ─────────────────────────────────
    pc  = pd.read_csv("results/phase5_5_pose_contacts.csv")
    phe = pc[pc.residue.str.startswith("Phe")]
    glu = pc[pc.residue == "Glu7"].iloc[0]
    rec("pose_contacts_total", int(pc.n_contacts_within_5A.sum()), "phase5_5_pose_contacts.csv")
    rec("pose_contacts_phe", int(phe.n_contacts_within_5A.sum()), "phase5_5_pose_contacts.csv")
    rec("pose_contacts_glu7", int(glu.n_contacts_within_5A), "phase5_5_pose_contacts.csv")
    rec("glu7_nearest_site_residue", int(glu.nearest_site_residue), "phase5_5_pose_contacts.csv")
    rec("glu7_nearest_dist_A", float(glu.nearest_dist), "phase5_5_pose_contacts.csv")

    full = pd.read_csv("results/phase5_5_pose_contacts_full.csv")
    full = full[pd.to_numeric(full.distance, errors="coerce").notna()].copy()
    full["distance"] = full.distance.astype(float)
    # isin(pc.residue) drops the alpha-amino rows, which are reported distances
    # rather than contacts and lie beyond the cutoff in any case
    real = full[(full.distance <= CONTACT_CUT) & full.pep_residue.isin(pc.residue)]
    assert len(real) == int(pc.n_contacts_within_5A.sum()), \
        "the two contact files disagree on the number of contacts"

    arom = real[real.pep_residue.str.startswith("Phe")
                & real.site_residue.str[:3].isin(AROMATIC)]
    rec("phe_contacts_to_aromatic", len(arom), "phase5_5_pose_contacts_full.csv")
    rec("aromatic_partners_distinct", arom.site_residue.nunique(), "phase5_5_pose_contacts_full.csv")
    rec("aromatic_partners", ", ".join(sorted(arom.site_residue.unique())),
        "phase5_5_pose_contacts_full.csv")

    # ── which Tier-1 peptide was docked, and why it was not the top-ranked one ─
    t1 = (pd.read_csv("screening/tiered_discovery.csv")
            .query("tier == 1")
            .sort_values("consensus", ascending=False))
    passing = t1[t1.tox_score < TOX_CUT]
    describe = lambda r: f"{r.sequence} (consensus {r.consensus}, ToxinPred2 {r.tox_score})"
    rec("tier1_top_ranked", describe(t1.iloc[0]), "tiered_discovery.csv")
    rec("tier1_top_passing_toxinpred2", describe(passing.iloc[0]), "tiered_discovery.csv")
    rec("toxinpred2_cut", TOX_CUT, "phase5.4_safety_select.py")

    # ── how far the shortlist depends on the single ESM-2 seed fit ────────────
    # Section 3.1 reports ESM-2 test AUC moving by +/-0.026 across seeds 42 to 46
    # against +/-0.002 for XGBoost, so the consensus rests on one stable half and
    # one that is not. Nothing is refit here. The margin is how far the ESM-2
    # probability could fall before the consensus drops below the phase 5.3
    # cutoff, with the tree held at its seed-42 value.
    rank = pd.read_csv("screening/screening_ranked.csv")
    disc = rank[rank.consensus >= DISCOVERY_THRESHOLD].copy()
    disc["esm_margin"] = 2 * (disc.consensus - DISCOVERY_THRESHOLD)
    tree_alone = disc.xgb_prob >= DISCOVERY_THRESHOLD
    rec("shortlist_n", len(disc), "screening_ranked.csv")
    rec("shortlist_tree_alone", int(tree_alone.sum()), "screening_ranked.csv")
    rec("shortlist_esm_dependent", int((~tree_alone).sum()), "screening_ranked.csv")
    rec("shortlist_xgb_min", round(disc.xgb_prob.min(), 3), "screening_ranked.csv")
    rec("esm_margin_median", round(disc.esm_margin.median(), 3), "screening_ranked.csv")
    rec("esm_margin_max", round(disc.esm_margin.max(), 3), "screening_ranked.csv")
    rec("lead_xgb_prob", float(disc.loc[disc.sequence == LEAD, "xgb_prob"].iloc[0]),
        "screening_ranked.csv")

    out = pd.DataFrame(rows)
    out.to_csv("results/phase5_7_derived_quantities.csv", index=False)
    print(out.to_string(index=False))
    print("\nwrote results/phase5_7_derived_quantities.csv")


if __name__ == "__main__":
    main()