
"""
Phase 5.5 (docking prep): assemble HADDOCK inputs for DPP-IV docking.

For each peptide, builds a 3-conformation ensemble (extended / helix / PPII) that
HADDOCK docks flexibly, plus the cleaned DPP-IV receptor and the active-site residues
(from contacts with the co-crystal ligand N7F in 4A5S). Each peptide is docked
separately; candidates are judged against the controls.

OUTPUT  docking/receptor_DPP4_4A5S.pdb        cleaned receptor (chain A)
        docking/peptides/<id>.pdb             3-model peptide ensemble (chain B)
        docking/docking_manifest.csv          peptide_id, sequence, role, length
        docking/haddock_active_residues.txt   receptor + peptide active residues for the AIRs

REQUIREMENTS  pip install PeptideBuilder biopython pandas
"""

# imports
import io
import os
import urllib.request
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import PeptideBuilder as PB
import Bio.PDB


# configurations and outputs
OUT_DIR = "docking"
PDB_ID, CHAIN, LIGAND = "4A5S", "A", "N7F"
SITE_CUTOFF = 5.0 # A: receptor residues within this of the ligand = active site
PEP_CHAIN = "B"
CONF = {"ext": (-139, 135), "helix": (-57, -47), "ppii": (-75, 145)} # backbone phi/psi per conformation

# role = tier1 / tier2 / pos_control / neg_control
PEPTIDES = [
    ("LEAD_T1_FVAPFPEVF", "FVAPFPEVF", "tier1"), # top short lead (9 aa)
    ("LEAD_T2_CAND3348", "ALPMHIRLSFNPTQLEEQCHI", "tier2"), # Tier-2 lead (21 aa)
    ("LEAD_T2_CAND3554", "TNDTPMIGTLAGANSLLNALPEEVIQHTFNLK", "tier2"), # longer Tier-2 lead (32 aa) for short-vs-long
    ("POS_DiprotinA", "IPI", "pos_control"), # textbook DPP-IV inhibitor
    ("POS_IPAVF", "IPAVF", "pos_control"), # known inhibitor the screen recovered
    ("NEG_hard_1", "RRRLNKHEE", "neg_control"), # model-confident non-ADP (cons ~0.004) for noise floor
    ("NEG_hard_2", "SSKDRLRR", "neg_control"), # model-confident non-ADP
    ("NEG_hard_3", "GRADR", "neg_control"), # model-confident non-ADP
]


def fetch_pdb(pdb_id):
    # Pull the raw PDB straight from RCSB and hand back a list of lines.
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    return urllib.request.urlopen(url, timeout=30).read().decode().splitlines()


def clean_receptor(raw, chain, out_path):
    """Keep only chain `chain` protein atoms (drop waters, sugars, ligand, ions, other chains).
    Also collapse alternate conformations: keep the first form of each atom and blank the altLoc
    column - otherwise HADDOCK rejects the file for having 'multiple forms' of a residue."""
    kept, seen = [], set()
    for l in raw:
        if not (l.startswith("ATOM") and l[21] == chain):
            continue
        key = (l[12:16], l[22:27]) # atom name + residue (resSeq + insertion code)
        if key in seen: # an alternate conformer of an atom we already kept -> skip
            continue
        seen.add(key)
        kept.append(l[:16] + " " + l[17:]) # blank col 17 (altLoc) so no stale conformer flag remains
    with open(out_path, "w") as fh:
        fh.write("\n".join(kept) + "\nTER\nEND\n")


def active_site(raw, chain, ligand, cutoff):
    """Receptor residues within `cutoff` A of the co-crystal ligand -> the binding site."""
    # Grab the ligand's atom coordinates (cols 31-54 = x, y, z). these mark the pocket
    lig = np.array([(float(l[30:38]), float(l[38:46]), float(l[46:54]))
                    for l in raw
                    if l.startswith("HETATM") and l[17:20].strip() == ligand and l[21] == chain])

    # Any protein residue with an atom closer than `cutoff` to any ligand atom is in the site.
    site = set()
    for l in raw:
        if l.startswith("ATOM") and l[21] == chain:
            xyz = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            if lig.size and np.sqrt(((lig - xyz) ** 2).sum(1)).min() < cutoff:
                site.add(int(l[22:26])) # col 23-26 = residue number
    return sorted(site)


def build_ensemble(seq, chain, out_path):
    """3-conformation peptide ensemble (extended/helix/PPII) as a multi-model PDB.

    HADDOCK docks the peptide flexibly from a few starting shapes, so hand it one
    extended, one helical and one polyproline-II conformation rather than guessing one."""
    models = []
    for i, (phi, psi) in enumerate(CONF.values(), 1):
        # build the backbone at fixed phi/psi -> n residues have n-1 angles between them
        st = PB.make_structure(seq, [phi] * (len(seq) - 1), [psi] * (len(seq) - 1))

        # PeptideBuilder writes chain A; rewrite col 22 to peptide chain (B) so it
        # doesn't clash with the receptor's chain A.
        buf = io.StringIO(); w = Bio.PDB.PDBIO(); w.set_structure(st); w.save(buf)
        body = [l[:21] + chain + l[22:] if l.startswith(("ATOM", "TER")) and len(l) >= 22 else l
                for l in buf.getvalue().splitlines() if l.startswith(("ATOM", "TER"))]

        # wrap each conformation as MODEL..ENDMDL so HADDOCK reads them as one ensemble
        models.append(f"MODEL     {i}\n" + "\n".join(body) + "\nENDMDL")
    with open(out_path, "w") as fh:
        fh.write("\n".join(models) + "\nEND\n")


def main():

    # prep the docking inputs: cleaned receptor, active site, peptide ensembles, manifest
    os.makedirs(f"{OUT_DIR}/peptides", exist_ok=True)
    raw = fetch_pdb(PDB_ID)

    clean_receptor(raw, CHAIN, f"{OUT_DIR}/receptor_DPP4_{PDB_ID}.pdb")
    site = active_site(raw, CHAIN, LIGAND, SITE_CUTOFF)

    # write the peptide ensembles and the docking manifest
    df = pd.DataFrame(PEPTIDES, columns=["peptide_id", "sequence", "role"])
    df["length"] = df.sequence.str.len()
    df.to_csv(f"{OUT_DIR}/docking_manifest.csv", index=False)
    for _, r in df.iterrows():
        build_ensemble(r.sequence, PEP_CHAIN, f"{OUT_DIR}/peptides/{r.peptide_id}.pdb")

    # write the HADDOCK active residues (ambiguous interaction restraints) file
    with open(f"{OUT_DIR}/haddock_active_residues.txt", "w") as fh:
        fh.write("# HADDOCK active residues (ambiguous interaction restraints)\n")
        fh.write(f"receptor (chain {CHAIN}) active: {','.join(map(str, site))}\n")
        fh.write(f"peptide  (chain {PEP_CHAIN}) active: all residues (1..N per peptide)\n")

    print(f"receptor -> {OUT_DIR}/receptor_DPP4_{PDB_ID}.pdb")
    print(f"active site ({len(site)} residues): {site}")
    print(f"{len(df)} peptide ensembles -> {OUT_DIR}/peptides/")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()