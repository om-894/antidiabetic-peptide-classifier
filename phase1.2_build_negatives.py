
"""
Phase 1.2: Build the dual-negative class (1:1 with the 966 ADP positives).

Length-matched exactly: one negative per positive at the same length, so the
negative length distribution mirrors the positives (no length signal to exploit).
At each length, soft negatives are used first; any shortfall is filled with hard
negatives (excisable at any length, so short lengths still fill).

  Soft  -> DBAASP antimicrobial peptides, standard amino acids only.
  Hard  -> random Swiss-Prot fragments (UniProt REST API), excluding GO (Gene Ontology) terms for
           glucose homeostasis / insulin signalling / hormone activity.

INPUTS   positives_ADP.csv (966 positives), peptides.csv (DBAASP export)
RUN      pip install requests pandas numpy && python build_negatives.py
"""

# Imports
import io
import random
import sys
import time
from collections import defaultdict, Counter
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# DEFINE GLOBAL VARIABLES
# --------------------------------------------------------------------------- #
POSITIVES_CSV = "data/positives_ADP.csv"
SOFT_CSV      = "data/peptides.csv"          # DBAASP export
OUTPUT_CSV    = "data/negatives.csv"
SEED          = 42

SOFT_CAP_FRACTION = None # use all available DBAASP soft negatives. hard negatives fill the rest.

EXCLUDE_GO          = ["0042593", "0008286", "0005179"] # GO ids for glucose homeostasis, insulin signaling, hormone activity

# Hard negative parameters
HARD_POOL_SIZE      = 4000
HARD_PROTEIN_LENMIN = 60
HARD_PROTEIN_LENMAX = 2000

# keep only sequences of the 20 standard amino acids 
# this drops X/B/Z/U and DBAASP's lowercase D-amino-acid (modified) peptides.
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY") # 
def is_standard(seq: str) -> bool:
    return len(seq) > 0 and set(seq).issubset(STANDARD_AA)


# --------------------------------------------------------------------------- #
# SOFT POOL  (DBAASP)
# --------------------------------------------------------------------------- #
def load_soft_by_length(csv_path: str, exclude: set) -> dict:
    """Read DBAASP CSV -> {length: [unique standard sequences]} excluding any
    sequence in `exclude` (the positives)."""
    df = pd.read_csv(csv_path)
    
    # Locate the sequence column without assuming its exact spelling/case
    # (DBAASP exports vary): match "SEQUENCE" ignoring case and surrounding spaces.
    col = next((c for c in df.columns if c.strip().upper() == "SEQUENCE"), None)
    if col is None:
        # No usable column = stop and print what columns were found, so the fix is obvious.
        sys.exit(f"No sequence column found in {csv_path}. Columns: {list(df.columns)}")
    
    # Coerce to string and trim stray whitespace around each sequence.
    seqs = df[col].astype(str).str.strip()
    
    # seen = deduplicates sequences
    # by_len = groups sequences by length, so the main loop can grab a soft
    #          negative of an exact length on demand (negatives are length-matched).
    seen, by_len = set(), defaultdict(list)
    
    for s in seqs:
        # Skip duplicates and skip any sequence that is actually a positive (ADP)
        # so a soft negative can never be identical to a positive.
        if s in seen or s in exclude:
            continue
        
        # Keep only standard 20-aa sequences.
        if is_standard(s): # drops lowercase D-aa, X/B/Z/U, symbols
            seen.add(s)
            by_len[len(s)].append(s) # bucket by length, e.g. {9: [...], 12: [...]}
    return by_len


# --------------------------------------------------------------------------- #
# HARD POOL  (Swiss-Prot via UniProt)
# --------------------------------------------------------------------------- #
def fetch_swissprot_pool(n_proteins: int) -> list:

    # Build the GO exclusion clause: "go:0042593 OR go:0008286 OR go:0005179".
    not_go = " OR ".join(f"go:{g}" for g in EXCLUDE_GO)
    
    # UniProt search query:
    #   reviewed:true     -> Swiss-Prot only (manually curated, high quality)
    #   fragment:false    -> whole protein entries, not partial sequences
    #   length:[MIN TO MAX] -> sane-sized proteins to excise fragments from
    #   NOT (go terms)    -> drop glucose-homeostasis / insulin / hormone proteins,
    #                        so a random fragment can't secretly be ADP-like
    query = (f"reviewed:true AND fragment:false "
             f"AND length:[{HARD_PROTEIN_LENMIN} TO {HARD_PROTEIN_LENMAX}] "
             f"NOT ({not_go})")
    print(f"[hard] querying UniProt: {query}") # print the query for debugging purposes
    
    proteins, url, first = [], "https://rest.uniprot.org/uniprotkb/search", True
    params = {"query": query, "format": "fasta", "size": 500} # # 500 results per page
    sess = requests.Session() # reuse one connection
    
    # First call sends the query params, later calls follow UniProt's "next"
    # link, which already has the params embedded in -> pass params only once.
    while url and len(proteins) < n_proteins:
        r = sess.get(url, params=params if first else None, timeout=60); first = False
        r.raise_for_status() # stop on any HTTP error
        
        # Parse the FASTA page. A protein spans several lines, so accumulate the
        # sequence lines in `cur` and cut it whenever the next ">" header starts.
        cur = []
        for line in io.StringIO(r.text):
            line = line.strip()
            if line.startswith(">"):
                if cur: proteins.append("".join(cur).upper()); cur = []
            elif line:
                cur.append(line)
        if cur: proteins.append("".join(cur).upper()) # cut the last protein
        print(f"[hard]   pulled {len(proteins)} proteins so far")
        url = r.links.get("next", {}).get("url"); time.sleep(0.2) # dont have too many requests in a short time
    return [p for p in proteins[:n_proteins] if is_standard(p)]


def excise_at_length(proteins: list, L: int, n: int, taken: set, rng) -> list:
    """Return up to n unique standard fragments of EXACT length L."""

    # Only proteins longer than L can yield a length-L window.
    usable = [p for p in proteins if len(p) > L]
    
    # `cap` bounds the attempts so that it wont loop forever if no more fragmnents are available.
    out, tries, cap = [], 0, n * 200 + 500
    while len(out) < n and tries < cap:
        tries += 1 # start counting attempts
        prot = rng.choice(usable) # random protein
        start = rng.randint(0, len(prot) - L) # random window start
        frag = prot[start:start + L] # slice an exact-length fragment

        # Keep it only if it's standard-AA and globally unique (`taken` spans all
        # lengths, so no fragment is ever reused as another negative).
        if is_standard(frag) and frag not in taken:
            taken.add(frag); out.append(frag)
    return out


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    rng = random.Random(SEED); np.random.seed(SEED)

    pos = pd.read_csv(POSITIVES_CSV)
    pos_seqs = set(pos["Sequence"].astype(str))
    pos_len_counts = Counter(pos["Length"])
    n_total = len(pos)

    soft_by_len = load_soft_by_length(SOFT_CSV, pos_seqs)
    for L in soft_by_len:                       # shuffle for reproducible draws
        rng.shuffle(soft_by_len[L])
    print(f"[soft] DBAASP usable: {sum(len(v) for v in soft_by_len.values())} "
          f"unique standard sequences\n")

    # Optional global soft cap
    soft_budget = {L: len(v) for L, v in soft_by_len.items()}
    if SOFT_CAP_FRACTION is not None:
        cap = int(round(n_total * SOFT_CAP_FRACTION))
        running = 0
        for L in sorted(soft_budget):
            take = min(soft_budget[L], pos_len_counts.get(L, 0))
            if running + take > cap:
                take = max(0, cap - running)
            soft_budget[L] = take; running += take

    # Exact length matching: one negative per positive, same length
    proteins = fetch_swissprot_pool(HARD_POOL_SIZE)
    print(f"[hard] pool: {len(proteins)} standard proteins\n")

    taken, rows = set(pos_seqs), []
    soft_n = hard_n = 0
    for L, need in sorted(pos_len_counts.items()):
        avail = min(need, soft_budget.get(L, 0), len(soft_by_len.get(L, [])))
        chosen_soft = soft_by_len.get(L, [])[:avail]
        for s in chosen_soft:
            taken.add(s)
        rows += [{"Sequence": s, "Label": 0, "Length": L, "NegType": "soft"} for s in chosen_soft]
        soft_n += len(chosen_soft)

        need_hard = need - len(chosen_soft)
        hard_frags = excise_at_length(proteins, L, need_hard, taken, rng)
        rows += [{"Sequence": s, "Label": 0, "Length": L, "NegType": "hard"} for s in hard_frags]
        hard_n += len(hard_frags)
        if len(hard_frags) < need_hard:
            print(f"[warn] length {L}: wanted {need_hard} hard, got {len(hard_frags)}")

    neg = pd.DataFrame(rows).drop_duplicates("Sequence").reset_index(drop=True)
    neg.to_csv(OUTPUT_CSV, index=False)

    print(f"\nwrote {len(neg)} negatives -> {OUTPUT_CSV}")
    print(f"  soft (DBAASP): {soft_n}  ({100*soft_n/len(neg):.0f}%)")
    print(f"  hard (UniProt): {hard_n}  ({100*hard_n/len(neg):.0f}%)")
    print("\nlength match (should mirror positives exactly):")
    print("  positives mean {:.2f} / negatives mean {:.2f}".format(
        pos["Length"].mean(), neg["Length"].mean()))
    print("\nNOTE: >40% identity removal vs positives (CD-HIT) is the next, "
          "separate Phase-1 step.")


if __name__ == "__main__":
    main()
