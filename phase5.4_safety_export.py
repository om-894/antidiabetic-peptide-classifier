
# How to produce this script's toxicity input (run these first). see the githbub (https://github.com/raghavagps/toxinpred2)
#   pip install toxinpred2                       # one-time; standalone pip tool, no BLAST needed
#   # export all 412 discoveries to FASTA (set TOP_N = None in phase5.4_safety_export.py first):
#   python phase5.4_safety_export.py
#   # predict toxicity — Model 1 (AAC-RF), -d 2 reports all peptides (not just toxins):
#   toxinpred2 -i screening/discovery_top412.fasta -o screening/toxinpred2_discovery.csv -m 1 -d 2
#   # then run this file:
#   python phase5.4_safety_select.py



"""
Phase 5.4 (export): write the top-ranked discovery set as FASTA for safety screening.

Docking only takes roughly 3 candidates, so rather than screen all 412 calibrated discoveries
we take the top TOP_N by consensus, screen those for toxicity (ToxinPred2)
and allergenicity (AlgPred 2.0) and the top surviving candidates go to docking.
This just produces the FASTA file to upload to those two web servers.

INPUT   screening/screening_discovery.csv   (from 5.3, calibrated set, ranked by consensus)
OUTPUT  screening/discovery_top{N}.fasta     (upload this to ToxinPred2 + AlgPred 2.0)

REQUIREMENTS  pip install pandas
"""

# Imports
import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
DISCOVERY = "screening/screening_discovery.csv"
TOP_N = None # how many top candidates to screen (None = all)


def main():
    # keep_default_na=False -> the peptide "NA" stays a string, not NaN.
    df = pd.read_csv(DISCOVERY, keep_default_na=False)
    top = df if TOP_N is None else df.head(TOP_N) # already sorted by consensus in 5.3

    out = f"screening/discovery_top{len(top)}.fasta"
    with open(out, "w") as fh:
        for _, r in top.iterrows():
            
            # header carries the id and the consensus score, so results stay traceable
            fh.write(f">{r['peptide_id']}_c{r['consensus']}\n{r['sequence']}\n")

    print(f"wrote {len(top)} sequences -> {out}")
    print("upload this file to both:")
    print("ToxinPred2: https://webs.iiitd.edu.in/raghava/toxinpred2/")
    print("AlgPred 2.0: https://webs.iiitd.edu.in/raghava/algpred2/")


if __name__ == "__main__":
    main()