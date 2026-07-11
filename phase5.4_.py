

"""
Phase 5.4 (export): write the top-ranked shortlist as FASTA for safety screening.

Docking only takes roughly 3 candidates, so rather than screen all 1188 high-confidence
hits we take the top TOP_N by consensus, screen those for toxicity (ToxinPred2)
and allergenicity (AlgPred 2.0) and the top surviving candidates go to docking.
This just produces the FASTA file to upload to those two web servers.

INPUT   screening/screening_shortlist.csv   (from 5.3, already ranked by consensus)
OUTPUT  screening/shortlist_top{N}.fasta     (upload this to ToxinPred2 + AlgPred 2.0)

REQUIREMENTS  pip install pandas
"""

# Imports
import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
SHORTLIST = "screening/screening_shortlist.csv"
TOP_N = 50 # how many top candidates to screen (None = all)


def main():
    # keep_default_na=False -> the peptide "NA" stays a string, not NaN.
    df = pd.read_csv(SHORTLIST, keep_default_na=False)
    top = df if TOP_N is None else df.head(TOP_N) # already sorted by consensus in 5.3

    out = f"screening/shortlist_top{len(top)}.fasta"
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