"""
Revision audit: how close are the 412 discoveries to the training positives?

The reviewer's first point asks for the nearest training distance of the top
Tier-1 and Tier-2 leads, and for the "discovery" claim to be set against
sequence extension and motif matching. The novelty filter removed exact matches
against train and test, then filtered only candidates of 11 residues or more at
40% identity with CD-HIT-2D, so short candidates passed on exact matching
alone. CD-HIT-2D's -s2 default also lets a short reference miss a longer query,
so a short training positive cannot suppress a longer candidate.

Distance is measured against the 873 TRAINING POSITIVES, the sequences whose
label the model actually learned, rather than against train plus test.
Identity is normalised over the longer sequence of each pair, as in
audit_overlap.py.

INPUTS  data/dataset_split.csv, screening/screening_discovery.csv,
        screening/tiered_discovery.csv
OUTPUTS audit_discovery_distance.csv, audit_discovery_leads.csv,
        audit_discovery_summary.txt
"""

import os
import numpy as np
import pandas as pd
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "results")
WORK = os.path.join(REPO, "phase_7_revision", "work")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

NEAR_DUP = 0.80
N_LEADS = 10


def lcs_len(a, b):
    """Length of the longest common subsequence, two rolling rows."""
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def comp_bound(ca, cb):
    """Upper bound on LCS from shared composition alone."""
    return sum(min(n, cb.get(r, 0)) for r, n in ca.items())


def nearest(s, pos_seqs, pos_comp):
    """Highest identity over the longer against any training positive, with its index."""
    cs = Counter(s)
    best, best_j = 0.0, -1
    for j, t in enumerate(pos_seqs):
        longer = len(s) if len(s) > len(t) else len(t)
        if comp_bound(cs, pos_comp[j]) <= best * longer:
            continue
        v = lcs_len(s, t) / longer
        if v > best:
            best, best_j = v, j
    return best, best_j


def longest_contained(s, pos_set, pos_lens):
    """The longest training positive wholly contained in s, or empty."""
    # containment is what the novelty filter missed, so it is reported separately
    # from graded identity: an extension of a known active is not a discovery
    for L in sorted(pos_lens, reverse=True):
        if L > len(s):
            continue
        for i in range(len(s) - L + 1):
            sub = s[i:i + L]
            if sub in pos_set:
                return sub
    return ""


def main():
    df = pd.read_csv(f"{REPO}/data/dataset_split.csv", keep_default_na=False)
    pos = df[(df.Split == "train") & (df.Label == 1)].Sequence.tolist()
    pos_set, pos_lens = set(pos), {len(p) for p in pos}
    pos_comp = [Counter(p) for p in pos]
    print(f"{len(pos)} training positives")

    disc = pd.read_csv(f"{REPO}/screening/screening_discovery.csv", keep_default_na=False)
    tier = pd.read_csv(f"{REPO}/screening/tiered_discovery.csv", keep_default_na=False)
    tier_of = dict(zip(tier.peptide_id, tier.tier))
    print(f"{len(disc)} discoveries, {len(tier)} tiered")

    rows = []
    for i, r in disc.iterrows():
        s = r.sequence
        best, j = nearest(s, pos, pos_comp)
        cont = longest_contained(s, pos_set, pos_lens)
        rows.append({
            "peptide_id": r.peptide_id,
            "sequence": s,
            "length": len(s),
            "tier": tier_of.get(r.peptide_id, 0),
            "consensus": r.consensus,
            "nearest_train_positive": pos[j] if j >= 0 else "",
            "identity_over_longer": round(best, 4),
            "contains_train_positive": cont,
            "contained_len": len(cont),
        })
        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1} of {len(disc)}")

    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/phase7_4_discovery_distance.csv", index=False)

    short, long = out.length < 11, out.length >= 11
    near = out.identity_over_longer >= NEAR_DUP
    has = out.contained_len > 0

    # the table the reviewer asked for: top leads of each tier by consensus
    leads = pd.concat([
        out[out.tier == 1].nlargest(N_LEADS, "consensus"),
        out[out.tier == 2].nlargest(N_LEADS, "consensus"),
    ])[["tier", "peptide_id", "sequence", "length", "consensus",
        "nearest_train_positive", "identity_over_longer", "contains_train_positive"]]
    leads.to_csv(f"{OUT}/phase7_4_discovery_leads.csv", index=False)

    txt = [
        f"discoveries                      {len(out)}",
        f"  below 11 residues              {int(short.sum())} (novelty by exact match only)",
        f"  11 residues or more            {int(long.sum())} (also CD-HIT-2D at 40%)",
        "",
        f"wholly contain a training positive   {int(has.sum())} "
        f"({100 * has.mean():.1f}%)",
        f"  among the short                {int((has & short).sum())} of {int(short.sum())}",
        f"  among the long                 {int((has & long).sum())} of {int(long.sum())}",
        f"  longest contained positive     {out.contained_len.max()} residues",
        "",
        f"near duplicates >= {NEAR_DUP:.2f}        {int(near.sum())} "
        f"({100 * near.mean():.1f}%)",
        f"  among the short                {int((near & short).sum())} of {int(short.sum())}",
        f"  among the long                 {int((near & long).sum())} of {int(long.sum())}",
        "",
        f"mean nearest identity            {out.identity_over_longer.mean():.3f}",
        f"  short                          {out.identity_over_longer[short].mean():.3f}",
        f"  long                           {out.identity_over_longer[long].mean():.3f}",
        "",
        "Tier-1 leads by consensus:",
        leads[leads.tier == 1].to_string(index=False),
        "",
        "Tier-2 leads by consensus:",
        leads[leads.tier == 2].to_string(index=False),
    ]
    s = "\n".join(txt)
    print("\n" + s)
    open(f"{OUT}/phase7_4_discovery_summary.txt", "w").write(s + "\n")


if __name__ == "__main__":
    main()
