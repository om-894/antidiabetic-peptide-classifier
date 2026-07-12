
# How to produce this script's allergenicity input (run first):
# https://webs.iiitd.edu.in/raghava/algpred2/batch_action.php
# upload screening/safe_shortlist.fasta to AlgPred 2.0 (AAC based RF, threshold 0.3)
# saved the results as screening/algpred2_safe.csv  (columns: Subject, ML Score, Prediction)

"""
Phase 5.4 (final): add the allergenicity gate -> final safe leads.

Second safety gate after toxicity: a lead is kept only if non-toxic and non-allergen.
Scores are borderline on these short food peptides (and milk/egg/soy are themselves
major allergens), so the continuous alg_score is carried through and the gate is soft.

INPUT   screening/screening_safe_shortlist.csv   (non-toxic leads, from 5.4-select)
        screening/algpred2_safe.csv              (AlgPred 2.0: Subject, ML Score, Prediction)
OUTPUT  screening/screening_final_leads.csv       (non-toxic and non-allergen, by consensus)

REQUIREMENTS  pip install pandas
"""

import pandas as pd

SAFE = "screening/screening_safe_shortlist.csv"
ALG  = "screening/algpred2_safe.csv"
OUT  = "screening/screening_final_leads.csv"

# read in csvs, join on peptide_id, gate on non-allergen, rank by consensus, save final leads
def main():
    safe = pd.read_csv(SAFE, keep_default_na=False)
    alg  = pd.read_csv(ALG, keep_default_na=False)
    alg.columns = [c.strip() for c in alg.columns] # "ML Score" has a space
    alg = alg.rename(columns={"Subject": "peptide_id", "ML Score": "alg_score", "Prediction": "alg_pred"})

    # join the AlgPred results onto the safe leads by peptide_id, then gate and rank
    m = safe.merge(alg[["peptide_id", "alg_score", "alg_pred"]], on="peptide_id", how="left")
    assert m["alg_pred"].notna().all(), "some safe leads missing an AlgPred call"

    m["non_allergen"] = m["alg_pred"] == "Non-Allergen" # soft gate, scores are borderline
    final = m[m["non_allergen"]].sort_values("consensus", ascending=False).reset_index(drop=True)

    # save the final leads to csv, with the continuous scores for traceability
    cols = ["peptide_id", "sequence", "length", "sources", "consensus", "tox_score", "alg_score"]
    final[cols].to_csv(OUT, index=False)

    # also print the final leads to console, for quick inspection
    print(f"non-toxic leads: {len(safe)} | also non-allergen: {len(final)}")
    print(f"saved -> {OUT}")
    print("\ntop 10 final leads (non-toxic and non-allergen):")
    print(final.head(10)[cols].to_string(index=False))


if __name__ == "__main__":
    main()