
"""
phase5.5_docking_parse.py — collapse the HADDOCK docking runs into one results table.

Why   HADDOCK docked each peptide into DPP-IV and handed back a folder of model files per
      peptide, but no single table to compare them. This reads those models back and for
      each peptide, pulls out its best binding score -> discovery leads, known binders
      (+ve controls) and known non-binders (-ve controls) line up side by side. That
      comparison is what closes the docking section.

What  re-derive every model's HADDOCK score from the energies in its PDB header, average
      per pose-cluster, and keep the best (most negative = tightest predicted binder).

INPUT         docking/haddock_runs/<id>-<name>_summary.tgz  -> one run per peptide, holds its cluster PDBs
OUTPUT        docking/haddock_scores.csv
"""

import csv
import os
import re
import glob
import tarfile
from statistics import mean, pstdev # population sd -> matches haddock's own report

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

# find the downloaded files in my downloads folder
ARCHIVE_DIR = os.path.expanduser(os.environ.get("HADDOCK_DIR", "docking/haddock_runs"))
OUT_CSV = os.environ.get("OUT_CSV", "docking/haddock_scores.csv")

# haddock never saves the final score, so rebuild it per model - a weighted sum of four
# energies, using its water-refinement weights (elec drops to 0.2 once models are water-refined):
#   vdw    -> van der waals - shape fit, how well the peptide packs the pocket
#   elec   -> electrostatics - charge attraction/repulsion
#   desolv -> empirical desolvation - favours burying hydrophobic (water-hating) surface
#   air    -> restraint violation - how far the pose drifts off the active site we pinned it
#             to (a docking penalty, not a real binding energy)
# more negative overall = better predicted binder
W_VDW, W_ELEC, W_DESOLV, W_AIR = 1.0, 0.2, 1.0, 0.1

# job name to what role the peptide plays here. the sequence itself is read from the pdb,
# so this only records the experiment: a discovery lead, a known binder, or a known non-binder.
ROLES = {
    "DPP4_FVAPFPEVF": "Tier-1 lead", "DPP4_CAND3554": "Tier-2 lead",
    "DPP4_CAND3348": "Tier-2 lead", "DPP4_DiprotinA": "+ve control",
    "DPP4_IPAVF": "+ve control", "DPP4_NEGhard1":  "-ve control",
    "DPP4_NEGhard2": "-ve control", "DPP4_NEGhard3": "-ve control",
}

# three-letter to one-letter amino-acid codes, for spelling the peptide out of the pdb
AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
       "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
       "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

# every model is named "clusterN_M.pdb": N = which cluster, M = its 1-4 rank inside that cluster
CL_RE = re.compile(r"cluster(\d+)_(\d+)\.pdb$")


# --------------------------------------------------------------------------- #
# READ ONE MODEL
# --------------------------------------------------------------------------- #
def parse_rep(text):
    """one model pdb -> (haddock score, its energy breakdown, peptide sequence).

    haddock writes the energies as plain-text REMARK lines at the top of each pdb, so
    scan the text for them and add up the score (the score isn't stored directly).
    """
    vdw = elec = air = desolv = bsa = None # define as none to detect a half-written model
    chain_ca, resn = {}, {} # per-chain Cα counts + residue names

    for ln in text.splitlines():
        # energies come as one fixed-order, comma-separated line - strip the REMARK energies:
        # label, split on commas and take the three we need by position:
        #   0=total 1=bonds 2=angles 3=improper 4=dihe  5=vdw  6=elec  7=air  8+=other restraints (unused)
        if ln.startswith("REMARK energies:"):
            v = ln.split(":", 1)[1].split(",")
            vdw, elec, air = float(v[5]), float(v[6]), float(v[7])

        # the desolvation energy and buried surface area are each on their own line
        elif ln.startswith("REMARK Desolvation energy:"):
            desolv = float(ln.split(":", 1)[1])

        # buried surface area is also on its own line, but don't use it in the score
        elif ln.startswith("REMARK buried surface area:"):
            bsa = float(ln.split(":", 1)[1])

        # every residue has one "CA" backbone atom. Counting them per chain gives chain
        # length and their residue names, read in order, spell the sequence
        elif ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            ch = ln[21] # chain id sits at column 22 of a pdb line
            chain_ca[ch] = chain_ca.get(ch, 0) + 1 # +1 because the CA line is always present for every residue

            # residue name is in columns 18-20, residue number in 23-26. store them per chain
            resn.setdefault(ch, {})[int(ln[22:26])] = ln[17:20].strip()

    # a cut-off download can leave a half-written model,
    # so if any of the energies are missing or no CA atoms were found, skip it
    if None in (vdw, elec, air, desolv, bsa) or not chain_ca:
        return None

    # haddock's score is a weighted sum of the four energies, with the weights defined above
    score = W_VDW*vdw + W_ELEC*elec + W_DESOLV*desolv + W_AIR*air
    lig = min(chain_ca, key=chain_ca.get) # two chains: DPP-IV (big) + peptide (small)

    # spell the peptide out of its residue names, in N- to C-terminus order.
    # if a residue is missing, use "X" to mark it.
    seq = "".join(AA3.get(resn[lig][n], "X") for n in sorted(resn[lig])) # N- to C-terminus
    return score, {"vdw":vdw, "elec":elec, "desolv":desolv, "air":air, "bsa":bsa}, seq


# --------------------------------------------------------------------------- #
# SUMMARISE ONE RUN
# --------------------------------------------------------------------------- #
def parse_job(tgz):
    """Read one peptide's docking archive and return the best cluster summary.

    Returns None if the archive contains no usable cluster files.
    """
    clusters = {} # cluster number -> list of (score, energy_breakdown)
    seq = None # peptide sequence, recovered from the first valid model

    # Read the .tgz directly instead of unpacking it first
    with tarfile.open(tgz) as tar:
        for m in tar.getmembers():

            # Only process files that look like HADDOCK cluster PDBs
            hit = CL_RE.search(os.path.basename(m.name))
            if not hit:
                continue

            # Parse one model PDB from the archive member
            rep = parse_rep(tar.extractfile(m).read().decode())
            if rep is None:
                # A missing REMARK/ATOM section usually means the file is incomplete
                continue

            score, comp, s = rep
            cl = int(hit.group(1))

            # Store this model under its cluster number
            clusters.setdefault(cl, []).append((score, comp))

            # Save the peptide sequence the first time we see it
            seq = seq or s

    if not clusters:
        return None

    # Collapse each cluster's models into one summary row
    # score = mean HADDOCK score across models in the cluster
    # sd = spread of those scores
    # n = how many models were found in that cluster
    # other fields = mean of each energy term across models
    stats = {}
    for cl, reps in clusters.items():
        scores = [score for score, _ in reps]
        stats[cl] = {
            "score": mean(scores),
            "sd": pstdev(scores),
            "n": len(scores),
            **{k: mean(comp[k] for _, comp in reps) for k in reps[0][1]},
        }

    # Pick the cluster with the lowest score
    # More negative = better predicted binder
    best = min(stats, key=lambda c: stats[c]["score"])

    # Compare the best cluster against the rest
    means = [stats[c]["score"] for c in stats]
    z = ((stats[best]["score"] - mean(means)) / pstdev(means)) if len(means) > 1 else None # calculate z-score only if more than one cluster exists

    return {
        "peptide": seq,
        "n_clusters": len(stats),
        "best_cl": best,
        "z": z,
        **stats[best],
    }


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    rows = []

    # Find every downloaded HADDOCK summary archive.
    for tgz in sorted(glob.glob(os.path.join(ARCHIVE_DIR, "*_summary.tgz"))):
        base = os.path.basename(tgz)

        # Extract the job ID and job name from the filename.
        # Example: 12345-DPP4_IPAVF_summary.tgz
        jid = base.split("-", 1)[0]
        name = base.split("-", 1)[1].rsplit("_summary", 1)[0]

        # Parse the archive into one summary row.
        r = parse_job(tgz)
        if r is None:
            continue

        # Add job metadata to the parsed docking result.
        rows.append({"job_id": jid, "job": name, "role": ROLES.get(name, "?"), **r})

    # Sort from best binder to worst.
    rows.sort(key=lambda r: r["score"])

    # Write the final results table.
    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    cols = ["role", "job_id", "job", "peptide", "score", "sd", "n", "best_cl", "n_clusters",
            "z", "vdw", "elec", "desolv", "air", "bsa"]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()