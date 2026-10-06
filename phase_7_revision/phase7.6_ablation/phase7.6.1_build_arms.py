"""
Revision analysis: build six negative-class arms for the ablation.

The reviewer's second priority asks to disentangle negative-class composition
from length and charge. The published comparison varies composition and length
together, because the Basith pool is not length-matched, so it cannot say which
of the two carries the effect. These six arms separate them.

  A  dual-negative, as published            composition controlled, length matched
  B  Basith pool, length free               the published-negative baseline
  C  Basith pool, length matched            removes B's length shortcut
  D  DBAASP only, length matched            the cationic half alone
  E  proteome only, length matched          the near-neutral half alone
  F  length AND charge matched              closes the residual shortcut

Every arm shares the published 178-row test set and the 873 training positives,
so only the training negatives vary and the AUCs are directly comparable.

Arms C and D cannot reach 873 pairs. The Basith pool holds nothing below 9
residues while 45% of the positives are shorter, and DBAASP runs out of
non-cationic sequences at the lengths the positives occupy. Those caps are
findings rather than defects, so each arm reports the size it reached.

INPUTS  data/dataset_split.csv, data/peptides.csv (DBAASP export),
        phase_3_models/phase3.6_negative_class_ablation/basith_negatives_pool.csv
        plus one GO-filtered Swiss-Prot fetch, cached to swissprot_pool.txt
OUTPUTS ablation_arms.csv (arm, Sequence, Label, Split, NegType)
        ablation_arms_summary.txt
"""

import io
import os
import random
import time
import numpy as np
import pandas as pd
import peptides
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "results")
WORK = os.path.join(REPO, "phase_7_revision", "work")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

POOL_CACHE = f"{WORK}/swissprot_pool.txt"
SEED = 42
N_PROTEINS = 4000
CHARGE_TOL = 0.5 # a matched negative sits within this of its positive's charge
STANDARD = set("ACDEFGHIKLMNPQRSTVWY")
EXCLUDE_GO = ["0042593", "0008286", "0005179"]


def is_standard(s):
    return bool(s) and set(s) <= STANDARD


def charge(s):
    """Net charge at pH 7.4, the same call phase 2.2 uses for the descriptor."""
    return peptides.Peptide(s).charge(pH=7.4)


# --------------------------------------------------------------------------- #
# POOLS
# --------------------------------------------------------------------------- #

def swissprot_pool():
    """GO-filtered reviewed Swiss-Prot proteins, cached after the first fetch."""
    if os.path.exists(POOL_CACHE):
        prots = [l.strip() for l in open(POOL_CACHE) if l.strip()]
        print(f"pool from cache: {len(prots)} proteins")
        return prots

    not_go = " OR ".join(f"go:{g}" for g in EXCLUDE_GO)
    query = (f"reviewed:true AND fragment:false AND length:[60 TO 2000] "
             f"NOT ({not_go})")
    prots, url, first = [], "https://rest.uniprot.org/uniprotkb/search", True
    params = {"query": query, "format": "fasta", "size": 500}
    sess = requests.Session()
    while url and len(prots) < N_PROTEINS:
        r = sess.get(url, params=params if first else None, timeout=60)
        first = False
        r.raise_for_status()
        cur = []
        for line in io.StringIO(r.text):
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    prots.append("".join(cur).upper())
                    cur = []
            elif line:
                cur.append(line)
        if cur:
            prots.append("".join(cur).upper())
        print(f"  fetched {len(prots)}")
        url = r.links.get("next", {}).get("url")
        time.sleep(0.2)
    prots = [p for p in prots[:N_PROTEINS] if is_standard(p)]
    open(POOL_CACHE, "w").write("\n".join(prots) + "\n")
    print(f"pool fetched: {len(prots)} proteins")
    return prots


def dbaasp_by_length(exclude):
    """DBAASP monomers by length, standard residues only, positives removed."""
    df = pd.read_csv(f"{REPO}/data/peptides.csv", keep_default_na=False)
    col = next(c for c in df.columns if c.strip().upper() == "SEQUENCE")
    seen, by_len = set(), {}
    for s in df[col].astype(str):
        s = s.strip().upper()
        if not is_standard(s) or s in seen or s in exclude:
            continue
        seen.add(s)
        by_len.setdefault(len(s), []).append(s)
    print(f"DBAASP pool: {len(seen)} unique standard sequences")
    return by_len


def basith_pool():
    p = f"{REPO}/phase_3_models/phase3.6_negative_class_ablation/basith_negatives_pool.csv"
    df = pd.read_csv(p, keep_default_na=False)
    col = next(c for c in df.columns if "seq" in c.lower())
    seqs = [s.strip().upper() for s in df[col].astype(str) if is_standard(s.strip().upper())]
    print(f"Basith pool: {len(seqs)} sequences")
    return seqs


# --------------------------------------------------------------------------- #
# DRAWS
# --------------------------------------------------------------------------- #

class Fragments:
    """Excises unique proteome fragments of an exact length on demand."""

    def __init__(self, prots, rng, taken):
        self.by_len = {}
        self.prots, self.rng, self.taken = prots, rng, taken

    def draw(self, L, want_charge=None, tol=CHARGE_TOL, cap=4000):
        usable = self.by_len.get(L)
        if usable is None:
            usable = [p for p in self.prots if len(p) > L]
            self.by_len[L] = usable
        if not usable:
            return None
        for _ in range(cap):
            prot = self.rng.choice(usable)
            start = self.rng.randint(0, len(prot) - L)
            frag = prot[start:start + L]
            if not is_standard(frag) or frag in self.taken:
                continue
            if want_charge is not None and abs(charge(frag) - want_charge) > tol:
                continue
            self.taken.add(frag)
            return frag
        return None


def length_matched(pos_lens, by_len, taken):
    """One negative per positive at that positive's exact length, where available."""
    # drawn first at each length, matching how the published soft half was built
    out, cursor = [], {}
    for L in pos_lens:
        pool = by_len.get(L, [])
        i = cursor.get(L, 0)
        while i < len(pool) and pool[i] in taken:
            i += 1
        cursor[L] = i + 1
        if i < len(pool):
            taken.add(pool[i])
            out.append(pool[i])
    return out


def main():
    rng = random.Random(SEED)
    np.random.seed(SEED)

    split = pd.read_csv(f"{REPO}/data/dataset_split.csv", keep_default_na=False)
    pos = split[split.Label == 1]
    tr_pos = pos[pos.Split == "train"].reset_index(drop=True)
    pos_all = set(pos.Sequence)
    pos_lens = tr_pos.Length.tolist()
    print(f"{len(tr_pos)} training positives")

    # the published test rows are reused unchanged by every arm
    test_rows = split[split.Split == "test"].copy()

    prots = swissprot_pool()
    soft_by_len = dbaasp_by_length(pos_all)
    basith = [s for s in basith_pool() if s not in pos_all]
    basith_by_len = {}
    for s in basith:
        basith_by_len.setdefault(len(s), []).append(s)

    # nothing drawn for any arm may equal a positive or a published test negative
    reserved = set(pos_all) | set(test_rows.Sequence)
    arms, notes = {}, []

    # A, the published training negatives
    arms["A"] = [(s, t) for s, t in zip(
        split[(split.Split == "train") & (split.Label == 0)].Sequence,
        split[(split.Split == "train") & (split.Label == 0)].NegType)]

    # B, Basith with no length matching
    pool = [s for s in basith if s not in reserved]
    rng.shuffle(pool)
    arms["B"] = [(s, "basith") for s in pool[:len(tr_pos)]]

    # C, Basith length matched
    arms["C"] = [(s, "basith") for s in
                 length_matched(pos_lens, basith_by_len, set(reserved))]

    # D, DBAASP only, length matched
    arms["D"] = [(s, "soft") for s in
                 length_matched(pos_lens, soft_by_len, set(reserved))]

    # E, proteome only, length matched
    taken_e = set(reserved)
    frag_e = Fragments(prots, rng, taken_e)
    arms["E"] = [(f, "hard") for f in
                 (frag_e.draw(L) for L in pos_lens) if f]

    # F, length and charge matched. DBAASP is tried first so the arm keeps as
    # much of the bioactivity control as a 91% cationic database can supply at a
    # near-neutral positive's charge, and the proteome fills the rest
    taken_f = set(reserved)
    frag_f = Fragments(prots, rng, taken_f)
    rows_f = []
    for L, c in zip(pos_lens, tr_pos.Sequence.map(charge)):
        hit = None
        for s in soft_by_len.get(L, []):
            if s not in taken_f and abs(charge(s) - c) <= CHARGE_TOL:
                taken_f.add(s)
                hit = (s, "soft")
                break
        if hit is None:
            f = frag_f.draw(L, want_charge=c)
            if f:
                hit = (f, "hard")
        if hit:
            rows_f.append(hit)
    arms["F"] = rows_f

    # assemble
    out = []
    for arm, rows in arms.items():
        for s, t in rows:
            out.append({"arm": arm, "Sequence": s, "Label": 0,
                        "Split": "train", "NegType": t})
        for _, r in tr_pos.iterrows():
            out.append({"arm": arm, "Sequence": r.Sequence, "Label": 1,
                        "Split": "train", "NegType": ""})
        for _, r in test_rows.iterrows():
            out.append({"arm": arm, "Sequence": r.Sequence, "Label": int(r.Label),
                        "Split": "test", "NegType": r.NegType})
    df = pd.DataFrame(out)
    df.to_csv(f"{WORK}/ablation_arms.csv", index=False)

    for arm, rows in arms.items():
        negs = [s for s, _ in rows]
        types = pd.Series([t for _, t in rows]).value_counts().to_dict()
        soft_share = types.get("soft", 0) / max(len(rows), 1)
        notes.append(
            f"arm {arm}: {len(rows):4d} train negatives, "
            f"mean length {np.mean([len(s) for s in negs]):5.1f}, "
            f"mean charge {np.mean([charge(s) for s in negs]):+5.2f}, "
            f"soft share {soft_share:.3f}, {types}")

    s = "\n".join([f"training positives: {len(tr_pos)}, "
                   f"mean length {tr_pos.Length.mean():.1f}, "
                   f"mean charge {tr_pos.Sequence.map(charge).mean():+.2f}", ""] + notes)
    print("\n" + s)
    open(f"{OUT}/phase7_6_arms_summary.txt", "w").write(s + "\n")


if __name__ == "__main__":
    main()
