
"""
Phase 5.5 (docking prep): assemble the DPP-IV docking inputs.

Builds the peptide set (Tier-1 + Tier-2 leads + positive/negative controls) and a
cleaned DPP-IV receptor (PDB 4A5S, chain A) for HPEPDOCK. Each peptide is docked
separately; candidates are judged against the controls.

OUTPUT  docking/docking_manifest.csv     peptide_id, sequence, role, length
        docking/peptides.fasta           all peptides to submit
        docking/DPP4_4A5S_chainA.pdb     cleaned receptor

REQUIREMENTS  pip install pandas
"""

# imports
import os
import urllib.request
import pandas as pd


# output directory, receptor PDB ID and chain and the peptides to dock
OUT_DIR = "docking"
PDB_ID  = "4A5S"
CHAIN   = "A"

# role = tier1 / tier2 / pos_control / neg_control
PEPTIDES = [
    # tier 1 and tier 2 leads from the screening
    ("LEAD_T1_FVAPFPEVF", "FVAPFPEVF", "tier1"),
    ("LEAD_T2_CAND3348", "ALPMHIRLSFNPTQLEEQCHI", "tier2"),
    ("LEAD_T2_CAND3554", "TNDTPMIGTLAGANSLLNALPEEVIQHTFNLK", "tier2"),
    
    ("POS_DiprotinA", "IPI", "pos_control"), # textbook DPP-IV inhibitor and well documented positive control
    ("POS_IPAVF", "IPAVF", "pos_control"), # known inhibitor the screen recovered (internal positive) from my set
    
    # model-confident non-ADPs from my screening
    ("NEG_hard_1", "RRRLNKHEE", "neg_control"),
    ("NEG_hard_2", "SSKDRLRR", "neg_control"),
    ("NEG_hard_3", "GRADR", "neg_control"),
]


def clean_receptor(pdb_id, chain, out_path):
    """Download a PDB and keep only chain `chain` protein atoms (drop waters,
    sugars, ligand, ions, other chains) to give a clean monomer receptor."""
    raw = urllib.request.urlopen(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=30).read().decode()
    
    # keep only ATOM lines for the requested chain and the first alternate conformation (A or blank)
    kept = [l for l in raw.splitlines()
            if l.startswith("ATOM") and l[21] == chain and l[16] in (" ", "A")]
    
    # add TER and END lines to the PDB
    with open(out_path, "w") as fh:
        fh.write("\n".join(kept) + "\nTER\nEND\n")
    return len(kept)


def main():
    # make the docking output directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # write the docking manifest csv
    df = pd.DataFrame(PEPTIDES, columns=["peptide_id", "sequence", "role"])
    df["length"] = df.sequence.str.len()
    df.to_csv(f"{OUT_DIR}/docking_manifest.csv", index=False)
    
    # write the peptides to a FASTA file
    with open(f"{OUT_DIR}/peptides.fasta", "w") as fh:
        for _, r in df.iterrows():
            fh.write(f">{r.peptide_id}\n{r.sequence}\n")

    # download and clean the DPP-IV receptor PDB
    rec = f"{OUT_DIR}/DPP4_{PDB_ID}_chain{CHAIN}.pdb" # output receptor path
    n = clean_receptor(PDB_ID, CHAIN, rec) # clean receptors up
    print(f"{len(df)} peptides -> {OUT_DIR}/docking_manifest.csv + peptides.fasta")
    print(f"receptor {PDB_ID} chain {CHAIN}: {n} atom lines -> {rec}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()