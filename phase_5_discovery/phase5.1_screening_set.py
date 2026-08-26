
"""
Phase 5.1: Build the screening pool of novel peptides for the discovery stage.

Anti-diabetic peptides occur inside food proteins and are released when gut
enzymes cut those proteins during digestion, so rather than inventing sequences
this reconstructs that process in silico. Twelve dietary proteins are fetched
from UniProt, digested under five enzyme regimes and their fragments kept as
candidates.

Digestion rather than a database dump or de-novo generation, because it is the
established route for food-derived DPP-IV work. Every fragment is a real
digestion product whose size already sits in the training range. BIOPEP-UWM's
enzyme action tool does the same thing.

Novelty is enforced as in the phase 1.3 split. A fragment identical to a train
or test sequence is dropped. cd-hit-2d then removes anything 40% or more
identical to one, so a novel candidate cannot be a peptide the model has met.

INPUTS  dataset_split.csv (known sequences and the training length range)
        UniProt REST, one FASTA per accession in SOURCE_PROTEINS
OUTPUTS  screening_candidates.csv (peptide_id, sequence, length, sources,
         enzymes)
         screening_candidates.fasta (input for phase 5.2 and the safety servers)
         screening_provenance.json (accessions and settings for the run)
REQUIREMENTS  pip install pandas. cd-hit on PATH enables the identity filter
"""

# Imports
import json
import os
import subprocess
import tempfile
import urllib.request
from collections import defaultdict
import pandas as pd


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

SPLIT_CSV = "data/dataset_split.csv" # the train and test sequences to avoid
OUT_DIR = "screening"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# major constituents of milk, egg and soy. lactoferrin, serum albumin and lysozyme
# enter on hydrolysate-level evidence alone (Nongonierma and FitzGerald 2019,
# Figure 1, DPP-IV IC50 below 1.0 mg/mL) rather than as sources of any identified
# inhibitory sequence. P25974 is the beta-conglycinin beta subunit and P04776 is
# glycinin G1 only, both of which are multi-subunit families. the sequences are
# full precursors, so they still carry their signal peptides
SOURCE_PROTEINS = {
    "bovine_beta_casein": "P02666",
    "bovine_alpha_s1_casein": "P02662",
    "bovine_alpha_s2_casein": "P02663",
    "bovine_kappa_casein": "P02668",
    "bovine_beta_lactoglobulin": "P02754",
    "bovine_alpha_lactalbumin": "P00711",
    "bovine_serum_albumin": "P02769",
    "bovine_lactoferrin": "P24627",
    "chicken_ovalbumin": "P01012",
    "chicken_lysozyme_c": "P00698",
    "soybean_glycinin_g1": "P04776",
    "soybean_beta_conglycinin": "P25974",
}

MISSED_CLEAVAGES = 1 # allow one uncut site
MIN_CDHIT_LEN = 11 # below this, novelty falls back to exact match, as in phase 1.3
IDENTITY_CUTOFF = 0.40

# each rule cuts just after a trigger residue, unless the next residue is proline,
# which blocks it. trypsin and chymotrypsin follow textbook specificities. pepsin
# is broad and only approximated here
ENZYME_RULES = {
    "pepsin": {"p1": set("FLWYAE"), "block_p_prime": True}, # stomach, acidic, low specificity
    "trypsin": {"p1": set("KR"), "block_p_prime": True}, # gut, cuts after Lys or Arg
    "chymotrypsin": {"p1": set("FYWL"), "block_p_prime": True}, # gut, bulky and aromatic
}


# --------------------------------------------------------------------------- #
# UNIPROT FETCH
# --------------------------------------------------------------------------- #
def fetch_uniprot(accession):
    """One protein's amino-acid sequence, uppercased, headers stripped."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    with urllib.request.urlopen(url, timeout=30) as r:
        fasta = r.read().decode()
    return "".join(l for l in fasta.splitlines() if not l.startswith(">")).strip().upper()


# --------------------------------------------------------------------------- #
# DIGESTION
# --------------------------------------------------------------------------- #
# one protein through the pipeline, using trypsin, which cuts after K or R
#
#   fetch     "...MKPADRTGK..." arrives from UniProt
#   find cuts after every K or R, unless proline follows
#   digest    split at those sites -> "ADR", "TGK", ...
#   filter    keep standard-AA fragments inside the training length range
#   novelty   drop anything already in train plus test, or close to it
#   store     each survivor becomes one row of the candidates csv
#
# so a handful of large food proteins becomes a few thousand short novel peptides
def cleavage_positions(seq, enzymes):
    """Sorted cut positions, where i means a cut between residue i-1 and i."""
    # several enzymes at once cut wherever any one of them recognises a site,
    # which is what the tryp+chymo regime below needs
    cuts = set()
    for i in range(len(seq) - 1):
        for e in enzymes:
            rule = ENZYME_RULES[e]
            if seq[i] in rule["p1"] and not (rule["block_p_prime"] and seq[i + 1] == "P"):
                cuts.add(i + 1)
                break # one enzyme is enough
    return sorted(cuts)


def digest(seq, enzymes, missed):
    """Set of fragments, including those spanning up to `missed` uncut sites."""
    # spanning uncut sites is what mimics incomplete digestion. without it every
    # fragment would be a full cut-to-cut segment and the longer partials would
    # never appear
    bounds = [0] + cleavage_positions(seq, enzymes) + [len(seq)]
    frags = set()
    for a in range(len(bounds) - 1):
        for b in range(a + 1, min(a + 2 + missed, len(bounds))):
            frags.add(seq[bounds[a]:bounds[b]])
    return frags


def digest_all(seq):
    """Fragments of one protein mapped to the regimes that produced each."""
    # four single-stage regimes plus a two-stage cascade, which is the five the
    # report counts. the cascade is the physiological one, pepsin in the stomach
    # first and the gut enzymes acting on what it leaves
    out = defaultdict(set)
    regimes = {
        "pepsin": {"pepsin"},
        "trypsin": {"trypsin"},
        "chymotrypsin": {"chymotrypsin"},
        "tryp+chymo": {"trypsin", "chymotrypsin"},
    }
    for name, enz in regimes.items():
        for f in digest(seq, enz, MISSED_CLEAVAGES):
            out[f].add(name)

    # the cascade is a two-stage digestion, so it needs to be handled separately
    for pep_frag in digest(seq, {"pepsin"}, MISSED_CLEAVAGES):
        for f in digest(pep_frag, {"trypsin", "chymotrypsin"}, MISSED_CLEAVAGES):
            out[f].add("gi_cascade")
    return out


# --------------------------------------------------------------------------- #
# NOVELTY FILTER
# --------------------------------------------------------------------------- #
def cdhit_novel(cands, known):
    """Candidates of at least MIN_CDHIT_LEN that are under IDENTITY_CUTOFF to every
    known sequence. Falls back to all of them if cd-hit is missing."""
    # near-copies matter as much as exact ones here. a fragment 90% identical to a
    # training peptide would score well for the wrong reason, so cd-hit-2d removes
    # it the same way phase 1.3 removes leakage across the split
    long_cands = [c for c in cands if len(c) >= MIN_CDHIT_LEN]
    if not long_cands:
        return set()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "known.fasta") # -i, the reference set
            q = os.path.join(tmp, "cand.fasta") # -i2, the queries
            out = os.path.join(tmp, "novel") # queries with no close reference match
            _write_fasta(db, {f"k{i}": s for i, s in enumerate(known)})
            _write_fasta(q, {f"c{i}": s for i, s in enumerate(long_cands)})

            # -n 2 is required at a 0.40 threshold. -l discards sequences at or below
            # its value, which costs nothing since long_cands is already filtered
            subprocess.run(
                ["cd-hit-2d", "-i", db, "-i2", q, "-o", out,
                 "-c", str(IDENTITY_CUTOFF), "-n", "2",
                 "-l", str(MIN_CDHIT_LEN - 1),
                 "-M", "0", "-T", "0", "-d", "0"],
                check=True, capture_output=True)
            return set(_read_fasta(out).values())
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        # the run still completes, but on exact-match novelty only
        print(f"[warn] cd-hit-2d unavailable or failed ({e}), keeping all long "
              f"candidates on exact-match novelty only")
        return set(long_cands)


def _write_fasta(path, id2seq):
    with open(path, "w") as fh:
        for name, seq in id2seq.items():
            fh.write(f">{name}\n{seq}\n")


def _read_fasta(path):
    id2seq, name = {}, None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]; id2seq[name] = ""
            elif name is not None:
                id2seq[name] += line
    return id2seq


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # the split supplies both halves of the filter below, the sequences to exclude
    # and the length range the model was trained to score
    split = pd.read_csv(SPLIT_CSV)
    known = set(split["Sequence"].astype(str))
    lo, hi = split["Sequence"].str.len().min(), split["Sequence"].str.len().max()

    frag_sources = defaultdict(set) # fragment -> the food proteins it came from
    frag_enzymes = defaultdict(set) # fragment -> the regimes that produced it
    for name, acc in SOURCE_PROTEINS.items():
        try:
            seq = fetch_uniprot(acc)
        except Exception as e:
            # one unreachable accession should not lose the other eleven, but it
            # does change the candidate pool, so it has to be visible
            print(f"[warn] could not fetch {name} ({acc}), {e}")
            continue
        for frag, enz in digest_all(seq).items():
            frag_sources[frag].add(name)
            frag_enzymes[frag].update(enz)

    # keeping to the training length range stops the model being asked to score
    # sizes it never saw
    kept = [f for f in frag_sources if lo <= len(f) <= hi and set(f) <= STANDARD_AA]
    kept = [f for f in kept if f not in known]

    # short peptides skip cd-hit for the reason phase 1.3 gives, so they carry the
    # exact match check alone
    short = [f for f in kept if len(f) < MIN_CDHIT_LEN]
    final = sorted(set(short) | cdhit_novel(kept, known), key=lambda s: (len(s), s))

    # a stable id per candidate, so one peptide can be followed through 5.2 to 5.5
    rows = [{"peptide_id": f"CAND{i:04d}",
             "sequence": f,
             "length": len(f),
             "sources": ";".join(sorted(frag_sources[f])),
             "enzymes": ";".join(sorted(frag_enzymes[f]))} for i, f in enumerate(final)]

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "screening_candidates.csv"), index=False)
    _write_fasta(os.path.join(OUT_DIR, "screening_candidates.fasta"),
                 {r["peptide_id"]: r["sequence"] for r in rows})

    # the accessions and cutoffs that produced this pool, so a rerun can be checked
    # against it rather than assumed to match
    with open(os.path.join(OUT_DIR, "screening_provenance.json"), "w") as fh:
        json.dump({"source_proteins": SOURCE_PROTEINS,
                   "missed_cleavages": MISSED_CLEAVAGES,
                   "length_range": [int(lo), int(hi)],
                   "identity_cutoff": IDENTITY_CUTOFF,
                   "n_candidates": len(final)}, fh, indent=2)


if __name__ == "__main__":
    main()