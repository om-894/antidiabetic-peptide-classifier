
"""
Phase 5.5 (tiered table): split the discoveries into two tiers for docking.

  Tier 1 - short (<=10 aa), high-consensus peptides: the ideal DPP-IV pocket size.
    Safety predictors over-flag short peptides, so their tox/allergen scores are
    soft-flagged as a length artefact, not filtered out.
  Tier 2 - longer peptides passing both hard safety filters (non-toxic + non-allergen).

Docking then contrasts a short Tier-1 lead against longer Tier-2 leads.

INPUT   screening/screening_discovery.csv     412 calibrated discoveries
        screening/toxinpred2_discovery.csv    ToxinPred2 tox scores
        screening/algpred2_safe.csv           AlgPred allergen calls (non-toxic subset)
OUTPUT  screening/tiered_discovery.csv        paper table: peptide, scores, tier, flag

REQUIREMENTS  pip install pandas
"""

import pandas as pd

# file imports and exports
DISCOVERY = "screening/screening_discovery.csv"
TOXPRED = "screening/toxinpred2_discovery.csv"
ALGPRED = "screening/algpred2_safe.csv"
OUT = "screening/tiered_discovery.csv"

SHORT_LEN = 10 # Tier 1 = short peptides at/below this length (ideal DPP-IV pocket size)
TOX_CUT = 0.6 # ToxinPred2 non-toxic threshold


def main():

    # read in files and merge into a single table
    disc = pd.read_csv(DISCOVERY, keep_default_na=False)
    tox  = pd.read_csv(TOXPRED, keep_default_na=False).rename(columns={"ML_Score": "tox_score"})
    m = disc.merge(tox[["Sequence", "tox_score"]], left_on="sequence", right_on="Sequence",
                   how="left").drop(columns="Sequence")

    # read in AlgPred allergen predictions and merge
    alg = pd.read_csv(ALGPRED, keep_default_na=False)
    alg.columns = [c.strip() for c in alg.columns]                  # "ML Score" has a space
    alg = alg.rename(columns={"Subject": "peptide_id", "ML Score": "alg_score", "Prediction": "alg_pred"})
    m = m.merge(alg[["peptide_id", "alg_score", "alg_pred"]], on="peptide_id", how="left")

    # assign tiers
    m["tier"], m["flag"] = "", ""
    t1 = m.length <= SHORT_LEN # Tier 1: short, high-consensus
    m.loc[t1, ["tier", "flag"]] = ["1", "soft-flag: short-length safety artefact"]
    t2 = (~t1) & (m.tox_score < TOX_CUT) & (m.alg_pred == "Non-Allergen") # Tier 2: longer, hard-filtered
    m.loc[t2, ["tier", "flag"]] = ["2", "passed hard toxicity + allergen filters"]

    # save the tiered table
    tiered = m[m.tier != ""].sort_values(["tier", "consensus"], ascending=[True, False]).reset_index(drop=True)
    cols = ["tier", "peptide_id", "sequence", "length", "sources", "consensus", "tox_score", "alg_score", "flag"]
    tiered[cols].to_csv(OUT, index=False)

    # print summary stats
    print(f"Tier 1 (short, high-consensus): {(tiered.tier=='1').sum()}")
    print(f"Tier 2 (longer, hard-filtered): {(tiered.tier=='2').sum()}")
    for t in ["1", "2"]:
        print(f"\nTier {t} top 5:")
        print(tiered[tiered.tier == t].head(5)[
            ["peptide_id", "sequence", "length", "consensus", "tox_score", "alg_score"]]
            .to_string(index=False)) # to_string() avoids truncation of long sequences


if __name__ == "__main__":
    main()