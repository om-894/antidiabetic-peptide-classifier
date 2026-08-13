
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
        screening/screening_candidates.csv
        data/dataset_split.csv 
OUTPUT  screening/tiered_discovery.csv        paper table: peptide, scores, tier, flag
        results/phase5_5_screening_funnel.csv stage-by-stage counts

REQUIREMENTS  pip install pandas
"""

import pandas as pd
from scipy.stats import fisher_exact

# file imports and exports
DISCOVERY = "screening/screening_discovery.csv"
TOXPRED = "screening/toxinpred2_discovery.csv"
ALGPRED = "screening/algpred2_safe.csv"
OUT = "screening/tiered_discovery.csv"
CANDIDATES = "screening/screening_candidates.csv"
SPLIT = "data/dataset_split.csv"
OUT_FUNNEL = "results/phase5_5_screening_funnel.csv"
OUT_LEN = "results/phase5_5_length_artefact.csv"

THRESHOLD = 0.90 # mirrors DISCOVERY_THRESHOLD in phase5.3, used here only as a label

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

    # length-artefact control. flag rate by length class for each tool.
    # the left merges leave NaN for unscored rows, so select on isin rather than != ""
    rows, short = [], m.length <= SHORT_LEN
    for tool, scored, flag in [("toxinpred2", m.tox_score.notna(), m.tox_score >= TOX_CUT),
                               ("algpred2", m.alg_pred.isin(["Allergen", "Non-Allergen"]),
                                m.alg_pred == "Allergen")]:
        tab = [[int((scored & short & flag).sum()),  int((scored & short & ~flag).sum())],
               [int((scored & ~short & flag).sum()), int((scored & ~short & ~flag).sum())]]
        p = fisher_exact(tab)[1]
        for i, sel in enumerate([short, ~short]):
            n = int((scored & sel).sum())
            rows.append({"tool": tool, "length_class": ["<=10", ">10"][i], "n_scored": n,
                         "n_flagged": tab[i][0], "pct_flagged": round(100 * tab[i][0] / n, 1),
                         "fisher_p": float(f"{p:.3g}")})
    pd.DataFrame(rows).to_csv(OUT_LEN, index=False)

    # screening funnel stats for figure 5
    cand = pd.read_csv(CANDIDATES, keep_default_na=False)
    split = pd.read_csv(SPLIT, keep_default_na=False)
    n_prot = len({p for v in cand.sources for p in str(v).split(";")})
    funnel = [
        ("source_proteins", n_prot, "UniProt entries contributing candidates"),
        ("novel_candidates", len(cand), "unique digestion fragments after novelty filter"),
        ("overlap_with_split", len(set(cand.sequence) & set(split.Sequence)),
         "exact matches against train plus test"),
        ("discoveries", len(disc), f"consensus >= {THRESHOLD:.2f}"),
        ("non_toxic", int((m.tox_score < TOX_CUT).sum()), f"ToxinPred2 score < {TOX_CUT}"),
        ("non_toxic_non_allergen",
         int(((m.tox_score < TOX_CUT) & (m.alg_pred == "Non-Allergen")).sum()),
         "AlgPred 2.0 Non-Allergen"),
        ("tier1", int(t1.sum()), f"discoveries of {SHORT_LEN} aa or fewer"),
        ("tier2", int(t2.sum()), "longer discoveries passing both filters"),
    ]
    pd.DataFrame(funnel, columns=["stage", "n", "detail"]).to_csv(OUT_FUNNEL, index=False)

    # save the tiered table
    tiered = m[m.tier != ""].sort_values(["tier", "consensus"], ascending=[True, False]).reset_index(drop=True)
    cols = ["tier", "peptide_id", "sequence", "length", "sources", "consensus", "tox_score", "alg_score", "flag"]
    tiered[cols].to_csv(OUT, index=False)


if __name__ == "__main__":
    main()