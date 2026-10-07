"""
Did the peptide backbone move during HADDOCK's semi-flexible refinement?

Section 2.6 states that the three fixed conformers seed the search rather than
hold the peptide while it is scored. The run.cns parameter file would record the
flexibility settings, but the summary archives the server returns do not contain
it and the runs are months old. This measures the thing the settings would only
imply: how far each refined backbone ended up from the conformer it started at.

Each peptide was submitted as a three-model ensemble built at fixed phi/psi,
extended (-139, 135), alpha-helical (-57, -47) and polyproline-II (-75, 145).
For every output model, the backbone dihedrals of chain B are compared against
all three inputs and the closest is taken, so the figure reported is the
smallest change consistent with the data. A pose locked to its starting shape
would score near zero.

INPUTS   docking/peptides/*.pdb, docking/haddock_runs/*_summary.tgz
OUTPUTS  results/phase7_8_backbone_flexibility.csv and .txt
"""

import os
import glob
import math
import tarfile
import tempfile
import numpy as np
from Bio.PDB import PDBParser, PPBuilder

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "results")

# archive stem to the input ensemble that was submitted for it
PAIRS = {
    "734078-DPP4_FVAPFPEVF": ("LEAD_T1_FVAPFPEVF.pdb", "FVAPFPEVF (Tier-1 lead)"),
    "734086-DPP4_CAND3348": ("LEAD_T2_CAND3348.pdb", "CAND3348 (Tier-2 lead)"),
    "734087-DPP4_CAND3554": ("LEAD_T2_CAND3554.pdb", "CAND3554 (Tier-2 lead)"),
    "734089-DPP4_DiprotinA": ("POS_DiprotinA.pdb", "IPI (Diprotin A, +ve control)"),
    "734090-DPP4_IPAVF": ("POS_IPAVF.pdb", "IPAVF (+ve control)"),
    "734091-DPP4_NEGhard1": ("NEG_hard_1.pdb", "hard negative 1"),
    "734093-DPP4_NEGhard2": ("NEG_hard_2.pdb", "hard negative 2"),
    "734094-DPP4_NEGhard3": ("NEG_hard_3.pdb", "hard negative 3"),
}

parser = PDBParser(QUIET=True)
ppb = PPBuilder()


def dihedrals(path, model_id=None, chain="B"):
    """phi and psi per residue of one chain, in degrees, None at the termini."""
    s = parser.get_structure("x", path)
    models = list(s) if model_id is None else [s[model_id]]
    out = []
    for m in models:
        if chain not in m:
            continue
        angles = []
        for pp in ppb.build_peptides(m[chain]):
            for phi, psi in pp.get_phi_psi_list():
                angles.append((None if phi is None else math.degrees(phi),
                               None if psi is None else math.degrees(psi)))
        out.append(angles)
    return out


def circ_rmsd(a, b):
    """RMS difference over paired angles, wrapped to the shorter way round."""
    # dihedrals are circular, so a naive subtraction makes 179 and -179 look
    # 358 degrees apart instead of 2
    diffs = []
    for (p1, s1), (p2, s2) in zip(a, b):
        for x, y in ((p1, p2), (s1, s2)):
            if x is None or y is None:
                continue
            d = (x - y + 180.0) % 360.0 - 180.0
            diffs.append(d)
    if not diffs:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(diffs))))


def main():
    rows, lines = [], []
    for stem, (inp, label) in PAIRS.items():
        arc = f"{REPO}/docking/haddock_runs/{stem}_summary.tgz"
        ref = f"{REPO}/docking/peptides/{inp}"
        if not (os.path.exists(arc) and os.path.exists(ref)):
            lines.append(f"{label}: missing archive or input, skipped")
            continue

        starts = dihedrals(ref)                      # the three submitted conformers
        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(arc) as t:
                t.extractall(td, filter="data")
            best = []
            for pdb in sorted(glob.glob(f"{td}/*.pdb")):
                got = dihedrals(pdb, model_id=0)
                if not got:
                    continue
                pose = got[0]
                # nearest of the three starting shapes, so the number reported is
                # the least movement the data can support
                d = min(circ_rmsd(pose, s) for s in starts
                        if len(s) == len(pose))
                best.append(d)

        if not best:
            lines.append(f"{label}: no chain B found in the outputs, skipped")
            continue
        best = np.array(best)
        rows.append({"peptide": label, "n_models": len(best),
                     "min_deg": round(float(best.min()), 1),
                     "median_deg": round(float(np.median(best)), 1),
                     "max_deg": round(float(best.max()), 1)})
        lines.append(f"{label:32s} n={len(best):3d}  closest {best.min():6.1f}"
                     f"  median {np.median(best):6.1f}  furthest {best.max():6.1f}")
        print(lines[-1])

    os.makedirs(OUT, exist_ok=True)
    import csv
    with open(f"{OUT}/phase7_8_backbone_flexibility.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    allmin = min(r["min_deg"] for r in rows)
    txt = "\n".join([
        "Backbone movement during semi-flexible refinement.",
        "RMS change in phi/psi from the nearest of the three submitted conformers,",
        "in degrees, over every model in each run's summary archive.",
        "", *lines, "",
        f"smallest change anywhere in the panel: {allmin:.1f} degrees",
        "A pose held at its starting geometry would read near zero.",
    ])
    open(f"{OUT}/phase7_8_backbone_flexibility.txt", "w").write(txt + "\n")
    print(f"\nsmallest change anywhere in the panel: {allmin:.1f} degrees")


if __name__ == "__main__":
    main()
