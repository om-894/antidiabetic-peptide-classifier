"""
Phase 7.6: selection stability of the shortlist across the five ESM-2 seeds.

Answers the third revision priority. The published shortlist is the seed-42
fit's output, and that seed is the best of five at test AUC 0.877 against a
0.842 mean, so the question is how much of the shortlist is a property of the
screen rather than of that fit.

Only the ESM-2 half of the consensus moves with the seed. XGBoost is refitted
from the frozen embedding, which the adapter does not touch, so its probability
is held at the deployed value and the consensus is recomputed per seed as the
mean of the two. That isolates the fine-tune's contribution.

The headline is the selection frequency: how many of the five seeds put each
candidate above the 0.90 operating threshold. Two shortlists that do not depend
on a single fit are reported beside it, the intersection across seeds and the
set selected by the mean ensemble.

INPUTS   screening/screening_esm2_seeds.npz (from the phase 5.8 Viking job)
         screening/screening_ranked.csv (fixed XGBoost probabilities)
         screening/tiered_discovery.csv
OUTPUTS  results/phase7_6_seed_stability.csv, phase7_6_seed_stability.txt
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "results")
WORK = os.path.join(REPO, "phase_7_revision", "work")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

THRESH = 0.90
CASE = ["FVAPFPEVF", "GPFPSIL", "LPGF", "IPAVF"]


def main():
    npz = f"{REPO}/screening/screening_esm2_seeds.npz"
    if not os.path.exists(npz):
        raise SystemExit(f"missing {npz}: run the phase 5.8 job on Viking first")

    d = np.load(npz, allow_pickle=True)
    seeds = [int(s) for s in d["seeds"]]
    esm = d["esm_prob"]                       # [n_seeds, n_candidates]
    ids = [str(p) for p in d["peptide_id"]]

    ranked = pd.read_csv(f"{REPO}/screening/screening_ranked.csv", keep_default_na=False)
    ranked = ranked.set_index("peptide_id").loc[ids].reset_index()
    assert len(ranked) == esm.shape[1], "candidate count differs from the npz"

    xgb = ranked.xgb_prob.to_numpy()
    cons = (esm + xgb) / 2.0                  # XGBoost is seed-independent
    sel = cons >= THRESH                      # [n_seeds, n_candidates]
    freq = sel.sum(0)

    # the published shortlist, for reference
    published = ranked.discovery.astype(str).str.lower().eq("true").to_numpy()

    out = pd.DataFrame({
        "peptide_id": ranked.peptide_id,
        "sequence": ranked.sequence,
        "length": ranked.length,
        "xgb_prob": xgb,
        "published_discovery": published,
        "n_seeds_selected": freq,
        "consensus_mean": cons.mean(0).round(4),
        "consensus_sd": cons.std(0).round(4),
    })
    for i, s in enumerate(seeds):
        out[f"consensus_seed{s}"] = cons[i].round(4)
    out = out.sort_values(["n_seeds_selected", "consensus_mean"], ascending=False)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(f"{OUT}/phase7_6_seed_stability.csv", index=False)

    # the two fit-independent shortlists
    inter = freq == len(seeds)
    ensemble = cons.mean(0) >= THRESH

    lines = [
        f"candidates {esm.shape[1]}, seeds {seeds}, threshold {THRESH}",
        "",
        "shortlist size per seed:",
        *[f"  seed {s}: {int(sel[i].sum())}" for i, s in enumerate(seeds)],
        f"  published (seed 42 consensus): {int(published.sum())}",
        "",
        "selection frequency across the five seeds:",
        *[f"  selected by {k} of {len(seeds)}: {int((freq == k).sum())}"
          for k in range(len(seeds), 0, -1)],
        f"  never selected: {int((freq == 0).sum())}",
        "",
        f"intersection across all five seeds: {int(inter.sum())} candidates",
        f"mean-ensemble shortlist: {int(ensemble.sum())} candidates",
        "",
        "the published 412 against the five-seed view:",
        f"  selected by all five: {int((published & inter).sum())} "
        f"({100 * (published & inter).sum() / max(published.sum(), 1):.1f}%)",
        f"  selected by three or more: {int((published & (freq >= 3)).sum())}",
        f"  selected by seed 42 alone: {int((published & (freq == 1)).sum())}",
        f"  median selection frequency: {np.median(freq[published]):.0f} of {len(seeds)}",
        "",
        "tiered leads and known actives:",
    ]

    tier = pd.read_csv(f"{REPO}/screening/tiered_discovery.csv", keep_default_na=False)
    tmap = dict(zip(tier.peptide_id, tier.tier))
    look = out[out.sequence.isin(CASE) | out.peptide_id.isin(tier.peptide_id)]
    for t in (1, 2):
        sub = look[look.peptide_id.map(tmap).eq(t)]
        if len(sub):
            lines.append(f"  Tier {t} (n = {len(sub)}): "
                         f"all five {int((sub.n_seeds_selected == 5).sum())}, "
                         f"three or more {int((sub.n_seeds_selected >= 3).sum())}, "
                         f"median {sub.n_seeds_selected.median():.0f} of {len(seeds)}")
    for seq in CASE:
        r = out[out.sequence == seq]
        if len(r):
            r = r.iloc[0]
            per = "  ".join(f"{r[f'consensus_seed{s}']:.3f}" for s in seeds)
            lines.append(f"  {seq:10s} selected by {int(r.n_seeds_selected)} of "
                         f"{len(seeds)}  per-seed consensus: {per}")

    txt = "\n".join(lines)
    print(txt)
    open(f"{OUT}/phase7_6_seed_stability.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
