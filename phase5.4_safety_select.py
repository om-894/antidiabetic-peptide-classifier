
# How to produce this script's toxicity input (run these first):
#   pip install toxinpred2                       # one-time; standalone pip tool, no BLAST needed
#   # export all 412 discoveries to FASTA (set TOP_N = None in phase5.4_safety_export.py first):
#   python phase5.4_safety_export.py
#   # predict toxicity — Model 1 (AAC-RF), -d 2 reports all peptides (not just toxins):
#   toxinpred2 -i screening/discovery_top412.fasta -o screening/toxinpred2_discovery.csv -m 1 -d 2
#   # then run this file:
#   python phase5.4_safety_select.py

"""
Phase 5.4 (select): gate the discovery set on toxicity, then rank by ADP activity.

Funnel, not a blended score: ToxinPred2 decides in/out (non-toxic), the consensus
ADP score decides the order. Toxicity scores on these short food peptides are
borderline, so the gate is coarse and the continuous ML_Score is carried through.

INPUT   screening/screening_discovery.csv      (412 calibrated discoveries, from 5.3)
        screening/toxinpred2_discovery.csv      (ToxinPred2 -m 1 output: ID, Sequence, ML_Score, Prediction)
OUTPUT  screening/screening_safe_shortlist.csv  (non-toxic discoveries, ranked by consensus)

REQUIREMENTS  pip install pandas
"""

import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
DISCOVERY = "screening/screening_discovery.csv"
TOXPRED   = "screening/toxinpred2_discovery.csv"
OUT       = "screening/screening_safe_shortlist.csv"

TOX_THRESHOLD = 0.6   # ToxinPred2 ML_Score below this = non-toxic (their default cut)


def main():
    # keep_default_na=False -> the peptide "NA" (Asn-Ala) stays a string, not NaN.
    disc = pd.read_csv(DISCOVERY, keep_default_na=False)
    tox  = pd.read_csv(TOXPRED, keep_default_na=False)[["Sequence", "ML_Score"]]

    # join toxicity onto the discovery set by sequence, then gate + rank.
    m = disc.merge(tox, left_on="sequence", right_on="Sequence", how="left").drop(columns="Sequence")
    assert m["ML_Score"].notna().all(), "some discoveries missing a ToxinPred2 score (run it on all 412)"

    m["tox_score"] = m["ML_Score"]                       # P(toxin); lower is safer
    m["non_toxic"] = m["ML_Score"] < TOX_THRESHOLD       # the gate
    safe = m[m["non_toxic"]].sort_values("consensus", ascending=False).reset_index(drop=True)

    cols = ["peptide_id", "sequence", "length", "sources", "consensus", "tox_score"]
    safe[cols].to_csv(OUT, index=False)

    print(f"discoveries: {len(m)} | non-toxic (ML_Score<{TOX_THRESHOLD}): {len(safe)}")
    print(f"saved -> {OUT}")
    print("\ntop 10 safe leads (high ADP, non-toxic):")
    print(safe.head(10)[cols].to_string(index=False))


if __name__ == "__main__":
    main()