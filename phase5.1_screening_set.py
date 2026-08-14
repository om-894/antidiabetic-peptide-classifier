
"""
Phase 5.1: Build the screening pool - the set of novel peptides the trained
pipeline is applied to for discovery (Aim 5).

Anti-diabetic peptides occur naturally inside food proteins and are released when
gut enzymes cut those proteins up during digestion. So rather than inventing
sequences, this reconstructs that process in-silico: download dietary proteins,
digest them with the main gut proteases and collect the fragments as candidates.

Why digestion (not a database dump or de-novo generation): it is the established
route for food-derived DPP-IV-inhibitory peptide discovery - BIOPEP-UWM's "enzyme
action" tool does the same thing - it is fully reproducible and every fragment is
a real digestion product already length-matched to the training peptides.

Novelty is enforced the same way as the Phase 1.3 split: any fragment identical
to, or >=40% identical to (CD-HIT), a train/test sequence is dropped, so a
"novel" candidate cannot just be a peptide the model has already seen.

OUTPUTS  screening/screening_candidates.csv    peptide_id, sequence, length, sources, enzymes
         screening/screening_candidates.fasta  input for Phase 5.2 + the safety-screening servers
         screening/screening_provenance.json   run settings, for reproducibility

REQUIREMENTS  pip install pandas   (UniProt fetch uses the standard library)
              cd-hit on PATH is optional (enables the 40%-identity novelty filter)
"""

"""
Example — one protein through the pipeline (trypsin, which cuts after K/R):

  1. fetch      "...MKPADRTGK..."   full protein sequence from UniProt
  2. find cuts   after each K or R (unless the next residue is P) -> cut sites
  3. digest      split the sequence at those sites                -> "ADR", "TGK", ...
  4. filter      keep standard-AA fragments 2-41 aa long          (the training range)
  5. novelty     drop any fragment already in / >=40% identical to train+test
  6. store       the survivors -> screening_candidates.csv, one row each

So a handful of large food proteins becomes a few hundred novel short peptides,
ready to be scored by the classifier in Phase 5.2.
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

SPLIT_CSV = "data/dataset_split.csv" # train/test set
OUT_DIR = "screening" # where the candidate list is written to
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY") # the 20 standard amino acids

# Dietary proteins to digest, as {name: UniProt accession}. Major constituents of
# milk, egg and soy. Lactoferrin, serum albumin and lysozyme enter on hydrolysate-level
# evidence only (Nongonierma and FitzGerald 2019, Figure 1, DPP-IV IC50 < 1.0 mg/mL),
# not as sources of any identified inhibitory sequence. P25974 is the beta-conglycinin
# beta subunit and P04776 is glycinin G1 only, both being multi-subunit families.
# Sequences are full precursors and include signal peptides.
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

MISSED_CLEAVAGES = 1 # real digestion is incomplete -> allow up to 1 uncut site (keeps some longer partial fragments)
MIN_CDHIT_LEN = 11 # peptides below this skip CD-HIT (as in Phase 1.3) -> exact-match novelty only
IDENTITY_CUTOFF = 0.40 # a fragment >=40% identical to a known sequence counts as "already seen" and is dropped

# Where each enzyme cuts. Each rule cuts just after a "trigger" residue (p1),
# unless the next residue is proline (P), which blocks the cut. Trypsin and
# chymotrypsin follow textbook specificities; pepsin's is broad and approximated.
ENZYME_RULES = {
    "pepsin": {"p1": set("FLWYAE"), "block_p_prime": True}, # stomach, acidic, low specificity
    "trypsin": {"p1": set("KR"), "block_p_prime": True}, # gut, cuts after Lys/Arg
    "chymotrypsin": {"p1": set("FYWL"), "block_p_prime": True}, # gut, cuts after bulky/aromatic residues
}


# --------------------------------------------------------------------------- #
# UNIPROT FETCH
# --------------------------------------------------------------------------- #

def fetch_uniprot(accession):
    """Fetch one protein's amino-acid sequence from UniProt (FASTA via REST)."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    with urllib.request.urlopen(url, timeout=30) as r:
        fasta = r.read().decode()

    # FASTA = one '>' header line then sequence lines; drop the header, glue the
    # sequence lines together, standardise to uppercase.
    seq = "".join(line for line in fasta.splitlines() if not line.startswith(">"))
    return seq.strip().upper()


# --------------------------------------------------------------------------- #
# DIGESTION
# --------------------------------------------------------------------------- #

def cleavage_positions(seq, enzymes):
    """Positions where the enzyme(s) cut. Position i = cut between residue i-1 and i.
    With several enzymes at once, cut wherever any of them recognises a site."""
    cuts = set()
    for i in range(len(seq) - 1): # each residue and the one after it
        for e in enzymes:
            rule = ENZYME_RULES[e]
            # cut after residue i if it's a trigger residue and not followed by proline
            if seq[i] in rule["p1"] and not (rule["block_p_prime"] and seq[i + 1] == "P"):
                cuts.add(i + 1) # cut just after residue i
                break # one enzyme is enough
    return sorted(cuts)


def digest(seq, enzymes, missed):
    """Cut seq at the enzyme sites -> set of fragments. `missed` lets a fragment
    span a few uncut sites (mimics incomplete digestion -> some longer products)."""
    bounds = [0] + cleavage_positions(seq, enzymes) + [len(seq)] # segment boundaries incl. both ends
    frags = set()
    for a in range(len(bounds) - 1): # fragment start
        
        # end may extend across up to `missed` extra boundaries (the uncut sites)
        for b in range(a + 1, min(a + 2 + missed, len(bounds))):
            frags.add(seq[bounds[a]:bounds[b]])
    return frags


def digest_all(seq):
    """Digest one protein several ways -> {fragment: {regimes that produced it}}.
    Each enzyme alone, trypsin+chymotrypsin together, and a two-stage gut cascade
    (pepsin in the stomach first, then the gut enzymes act on those pieces)."""
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

    # gastrointestinal cascade: gut enzymes chew on the stomach (pepsin) fragments
    for pep_frag in digest(seq, {"pepsin"}, MISSED_CLEAVAGES):
        for f in digest(pep_frag, {"trypsin", "chymotrypsin"}, MISSED_CLEAVAGES):
            out[f].add("gi_cascade")
    return out


# --------------------------------------------------------------------------- #
# NOVELTY FILTER  (>=40% identity, via CD-HIT-2D — mirrors Phase 1.3)
# --------------------------------------------------------------------------- #

def cdhit_novel(cands, known):
    """Candidates (>= MIN_CDHIT_LEN aa) that are <40% identical to every known
    sequence, via CD-HIT-2D. Like Phase 1.3, this removes near-copies as well as
    exact copies, so a "novel" candidate isn't just a lightly-edited training
    peptide. Falls back to exact-match only if cd-hit isn't installed."""
    long_cands = [c for c in cands if len(c) >= MIN_CDHIT_LEN] # short peptides handled separately
    if not long_cands:
        return set()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            db  = os.path.join(tmp, "known.fasta") # -i   reference set = train + test
            q   = os.path.join(tmp, "cand.fasta") # -i2  query set = my candidates
            out = os.path.join(tmp, "novel") # survivors = query NOT similar to reference
            _write_fasta(db, {f"k{i}": s for i, s in enumerate(known)})
            _write_fasta(q,  {f"c{i}": s for i, s in enumerate(long_cands)})

            # cd-hit-2d keeps only db2 (-i2) sequences dissimilar to db1 (-i).
            subprocess.run(
                ["cd-hit-2d", "-i", db, "-i2", q, "-o", out,
                 "-c", str(IDENTITY_CUTOFF), "-n", "2", # n=2 required at a 40% identity threshold
                 "-l", str(MIN_CDHIT_LEN - 1), # ignore sequences <=10 aa (matches Phase 1.3)
                 "-M", "0", "-T", "0", "-d", "0"],
                check=True, capture_output=True)
            return set(_read_fasta(out).values()) # the fragments CD-HIT judged novel
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"  [warn] cd-hit-2d unavailable/failed ({e}); "
              f"keeping all long candidates on exact-match novelty only.")
        return set(long_cands)


def _write_fasta(path, id2seq):
    # FASTA = '>name' line then the sequence line, one record after another.
    with open(path, "w") as fh:
        for name, seq in id2seq.items():
            fh.write(f">{name}\n{seq}\n")

# Read a FASTA file into {name: sequence}. Ignores the header line
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

    # Load the split. Needed for two things: the length range of real ADPs (so I
    # keep only sensibly-sized candidates), and the sequences already used (so I
    # can exclude them as "not novel").
    split = pd.read_csv(SPLIT_CSV)
    known = set(split["Sequence"].astype(str))
    lo, hi = split["Sequence"].str.len().min(), split["Sequence"].str.len().max()
    print(f"training length range: {lo}-{hi} aa | {len(known)} known sequences")

    # Fetch + digest every source protein.
    frag_sources = defaultdict(set) # fragment -> which food proteins it came from
    frag_enzymes = defaultdict(set) # fragment -> which enzyme regimes produced it
    for name, acc in SOURCE_PROTEINS.items():
        try:
            seq = fetch_uniprot(acc)
        except Exception as e:
            print(f"  [warn] could not fetch {name} ({acc}): {e}"); continue
        for frag, enz in digest_all(seq).items():
            frag_sources[frag].add(name)
            frag_enzymes[frag].update(enz)
        print(f"  {name:26s} {acc}  protein_len={len(seq):4d}")

    # Keep standard-AA fragments within the training length range (so the model
    # isn't asked to score sizes it never saw).
    kept = [f for f in frag_sources if lo <= len(f) <= hi and set(f) <= STANDARD_AA]
    print(f"fragments after length + amino-acid filter: {len(kept)}")

    # Novelty step 1: drop exact copies of anything in train/test.
    kept = [f for f in kept if f not in known]
    print(f"after exact-match novelty filter: {len(kept)}")

    # Novelty step 2: drop near-copies (>=40% identical). Short peptides (<11 aa)
    # skip this and rely on the exact-match check, exactly as Phase 1.3 does.
    short = [f for f in kept if len(f) < MIN_CDHIT_LEN]
    novel_long = cdhit_novel(kept, known)
    final = sorted(set(short) | novel_long, key=lambda s: (len(s), s))
    print(f"final novel candidates: {len(final)} ({len(short)} short + {len(novel_long)} long)")

    # Write the shortlist: readable CSV (with provenance) + FASTA for the next
    # stages. Each candidate gets a stable ID so it can be tracked through 5.2-5.5.
    rows = [{
        "peptide_id": f"CAND{i:04d}",
        "sequence":   f,
        "length":     len(f),
        "sources":    ";".join(sorted(frag_sources[f])), # food protein(s) it derives from
        "enzymes":    ";".join(sorted(frag_enzymes[f])), # digestion regime(s) that produced it
    } for i, f in enumerate(final)]


    # Save the outputs to the output directory
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "screening_candidates.csv"), index=False)
    _write_fasta(os.path.join(OUT_DIR, "screening_candidates.fasta"),
                 {r["peptide_id"]: r["sequence"] for r in rows})
    with open(os.path.join(OUT_DIR, "screening_provenance.json"), "w") as fh:
        json.dump({
            "source_proteins":  SOURCE_PROTEINS,
            "missed_cleavages": MISSED_CLEAVAGES,
            "length_range":     [int(lo), int(hi)],
            "identity_cutoff":  IDENTITY_CUTOFF,
            "n_candidates":     len(final),
        }, fh, indent=2)
    print(f"saved -> {OUT_DIR}/screening_candidates.{{csv,fasta}}")


if __name__ == "__main__":
    main()