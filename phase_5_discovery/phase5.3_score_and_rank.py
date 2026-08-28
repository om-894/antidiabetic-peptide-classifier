
"""
Phase 5.3: Fuse the features, score with XGBoost and rank the pool by consensus.

The tree half of the consensus, kept apart from the torch stage in 5.2 because
the two libraries bundle their own libomp and segfault together on macOS. Three
steps run here.

  fuse -> rebuild the phase 2.2 vector, seven descriptors z-scored with the
          saved train statistics, concatenated onto the 5.2 embeddings.
  score -> the trained XGBoost base learner turns that into P(ADP).
  rank -> average with the ESM-2 probability and sort.

Two models rather than one, because ESM-2 alone flags roughly half the pool and
is the more miscalibrated of the two. The mean of two independent views is the
better-calibrated screen, which phase 5.0 measures. Discoveries are then taken at
a calibrated cutoff rather than an arbitrary top-N.

Novelty is scored here as well, as the highest difflib ratio between a candidate
and any training positive. Candidates under 11 aa bypassed the cd-hit filter in
phase 5.1, so this quantifies how close the pool really sits to the training set.

The pool is also broken down by source protein, which is the table reporting what
each precursor contributed and how far its yield tracks its length.

INPUTS  screening_esm2.npz (X_emb, esm_prob, peptide_id, sequence)
        screening_candidates.csv (length, sources, enzymes)
        fusion_vectors.npz (desc_mean, desc_std, the saved train z-score)
        base_xgb.joblib (the 1,287-d base learner)
OUTPUTS  screening_ranked.csv (every candidate, both scores, ranked)
         screening_discovery.csv (the calibrated set, feeds phase 5.4)
         screening_shortlist.csv (the older both-models-agree set)
         phase5_3_novelty.csv, phase5_3_novelty_summary.csv, phase5_3_score_funnel.csv
         phase5_3_per_protein.csv (the per-source breakdown)
REQUIREMENTS  pip install xgboost scikit-learn joblib peptides pandas numpy scipy
"""

# Imports
import os
from difflib import SequenceMatcher
import joblib
import numpy as np
import pandas as pd
import peptides
from scipy.stats import spearmanr


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

ESM2_NPZ = "screening/screening_esm2.npz"
CANDIDATES = "screening/screening_candidates.csv"
FUSION_NPZ = "fusion_vectors.npz" # only for the saved train z-score statistics
XGB_MODEL = "models/base_xgb.joblib"
SPLIT = "data/dataset_split.csv" # training positives, for the novelty check

OUT_RANKED = "screening/screening_ranked.csv"
OUT_DISCOVERY = "screening/screening_discovery.csv"
OUT_SHORTLIST = "screening/screening_shortlist.csv"
OUT_NOVELTY = "results/phase5_3_novelty.csv"
OUT_NOVELTY_SUM = "results/phase5_3_novelty_summary.csv"
OUT_SCORE_FUNNEL = "results/phase5_3_score_funnel.csv"
OUT_PER_PROTEIN = "results/phase5_3_per_protein.csv"

# phase 5.0's sweep permits 0.87. 0.90 is the conservative choice, sitting in the
# top calibration bin where 450 out-of-fold peptides came in at 90.9% positive
DISCOVERY_THRESHOLD = 0.90

# the older screen, where both models had to call a candidate positive on their
# own. kept for comparison, superseded by the calibrated cutoff above
THRESHOLD = 0.5

# same seven descriptors in the same order as phase 2.2, which is what the fused
# vector's last seven columns mean to XGBoost
DESCRIPTOR_NAMES = [
    "net_charge_pH7.4", "gravy_kd", "isoelectric_point", "aromaticity",
    "instability_index", "aliphatic_index", "boman_index",
]

# the twelve keys phase 5.1 writes into the sources column, mapped to the group and
# name the report prints and held in the row order the report tabulates them in
SOURCE_LABELS = {
    "bovine_alpha_s1_casein": ("Bovine milk, casein", "αs1-casein"),
    "bovine_alpha_s2_casein": ("Bovine milk, casein", "αs2-casein"),
    "bovine_beta_casein": ("Bovine milk, casein", "β-casein"),
    "bovine_kappa_casein": ("Bovine milk, casein", "κ-casein"),
    "bovine_beta_lactoglobulin": ("Bovine milk, whey", "β-lactoglobulin"),
    "bovine_alpha_lactalbumin": ("Bovine milk, whey", "α-lactalbumin"),
    "bovine_lactoferrin": ("Bovine milk, whey", "lactoferrin"),
    "bovine_serum_albumin": ("Bovine milk, whey", "serum albumin"),
    "chicken_ovalbumin": ("Hen egg white", "ovalbumin"),
    "chicken_lysozyme_c": ("Hen egg white", "lysozyme C"),
    "soybean_glycinin_g1": ("Soybean", "glycinin G1"),
    "soybean_beta_conglycinin": ("Soybean", "β-conglycinin, β subunit 1"),
}


def descriptors(seq):
    """The seven descriptors for one peptide, in DESCRIPTOR_NAMES order."""
    # aromaticity is computed here rather than taken from peptides, since phase 2.2
    # used the Lobry FWY fraction and the two definitions do not agree
    p = peptides.Peptide(seq)
    aromaticity = sum(seq.count(a) for a in "FWY") / len(seq)
    return [
        p.charge(pH=7.4),
        p.hydrophobicity(scale="KyteDoolittle"),
        p.isoelectric_point(),
        aromaticity,
        p.instability_index(),
        p.aliphatic_index(),
        p.boman(),
    ]


def nearest_positive(cands, pos):
    """Closest training positive to each candidate, as sequence and difflib ratio."""
    # 3,596 candidates against 873 positives is over three million comparisons, so
    # the two cheap bounds matter. real_quick_ratio and quick_ratio are upper bounds
    # on ratio, so a pair that cannot beat the running best is skipped before the
    # expensive match. set_seq2 caches the positive's index across the inner loop
    best_r, best_s, sm = np.zeros(len(cands)), [""] * len(cands), SequenceMatcher()
    for p in pos:
        sm.set_seq2(p)
        for i, s in enumerate(cands):
            sm.set_seq1(s)
            if sm.real_quick_ratio() <= best_r[i] or sm.quick_ratio() <= best_r[i]:
                continue
            r = sm.ratio()
            if r > best_r[i]:
                best_r[i], best_s[i] = r, p
    return best_s, best_r


def per_protein(out):
    """One row per source protein, with its candidate and discovery counts."""
    # a fragment can be released from more than one precursor
    ex = out.assign(source=out.sources.str.split(";")).explode("source")
    disc = ex[ex.discovery]
    n_cand, n_disc = ex.groupby("source").size(), disc.groupby("source").size()

    # out arrives sorted by consensus then peptide_id and groupby preserves that order
    # within a group, so the first row of each group is that protein's top scorer
    top = disc.groupby("source", sort=False).first()

    rows = []
    for src, (group, protein) in SOURCE_LABELS.items():
        c, d = int(n_cand.get(src, 0)), int(n_disc.get(src, 0))
        rows.append({"group": group, "protein": protein, "source": src,
                     "candidates": c, "discoveries": d,
                     "rate_pct": round(100 * d / c, 1) if c else None,
                     "top_discovery": top.sequence.get(src, ""),
                     "top_score": round(top.consensus[src], 3) if d else None})

    # the Total row counts each fragment once, so its rate is 412 of 3,596 rather than
    # the 414 of 4,052 a reader would get by summing the column above it
    n, k = len(out), int(out.discovery.sum())
    rows.append({"group": "Total", "protein": "", "source": "", "candidates": n,
                 "discoveries": k, "rate_pct": round(100 * k / n, 1),
                 "top_discovery": "", "top_score": None})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    os.makedirs("results", exist_ok=True)

    z = np.load(ESM2_NPZ, allow_pickle=True)
    X_emb = z["X_emb"].astype(np.float32)
    esm_prob = z["esm_prob"].astype(float)
    pid = [str(x) for x in z["peptide_id"]]
    seqs = [str(s) for s in z["sequence"]]

    # the metadata is reindexed onto the npz order rather than assumed to share it,
    # and the assert catches the case where the two files differ
    cand = pd.read_csv(CANDIDATES, keep_default_na=False).set_index("peptide_id")
    meta = cand.loc[pid]
    assert (meta["sequence"].to_numpy() == np.array(seqs)).all(), "sequence order mismatch"

    # the z-score comes from the saved training statistics rather than being refitted
    # on the candidates, which would leak the screening distribution into the scaling
    D = np.array([descriptors(s) for s in seqs], dtype=np.float32)
    assert np.isfinite(D).all(), "non-finite descriptor"
    fz = np.load(FUSION_NPZ, allow_pickle=True)
    D_z = (D - fz["desc_mean"]) / fz["desc_std"]
    X_fused = np.hstack([X_emb, D_z]).astype(np.float32)
    assert X_fused.shape[1] == 1287, f"fused width {X_fused.shape[1]}, expected 1287"

    xgb = joblib.load(XGB_MODEL)
    xgb_prob = xgb.predict_proba(X_fused)[:, 1]
    consensus = 0.5 * (esm_prob + xgb_prob) # unweighted, as phase 5.0 calibrated it

    # peptide_id breaks ties, so the order does not depend on the pandas sort being
    # stable. over a third of the pool shares a consensus value with another row
    out = pd.DataFrame({
        "peptide_id": pid,
        "sequence": seqs,
        "length": meta["length"].to_numpy(),
        "sources": meta["sources"].to_numpy(),
        "enzymes": meta["enzymes"].to_numpy(),
        "esm_prob": np.round(esm_prob, 4),
        "xgb_prob": np.round(xgb_prob, 4),
        "consensus": np.round(consensus, 4),
        "discovery": consensus >= DISCOVERY_THRESHOLD,
        "high_confidence": (esm_prob >= THRESHOLD) & (xgb_prob >= THRESHOLD),
    }).sort_values(["consensus", "peptide_id"], ascending=[False, True]).reset_index(drop=True)

    out.to_csv(OUT_RANKED, index=False)
    discovery = out[out["discovery"]].reset_index(drop=True)
    discovery.to_csv(OUT_DISCOVERY, index=False)
    shortlist = out[out["high_confidence"]].reset_index(drop=True)
    shortlist.to_csv(OUT_SHORTLIST, index=False)
    per_protein(out).to_csv(OUT_PER_PROTEIN, index=False)

    # novelty is measured over the whole ranking rather than the discoveries alone,
    # so the score similarity correlation below has the full range to work with
    split = pd.read_csv(SPLIT, keep_default_na=False)
    pos = split[(split.Label == 1) & (split.Split == "train")].Sequence.tolist()

    nov = out[["peptide_id", "sequence", "length", "consensus", "discovery"]].copy()
    nov.insert(0, "rank", nov.index + 1)
    nov["nn_sequence"], nov["nn_similarity"] = nearest_positive(nov.sequence.tolist(), pos)
    nov.to_csv(OUT_NOVELTY, index=False)

    # a strong correlation would mean the model is rewarding resemblance to its
    # training set rather than the property, so this is a check on the screen itself
    d = nov[nov.discovery]
    rho_a, p_a = spearmanr(nov.consensus, nov.nn_similarity)
    rho_d, p_d = spearmanr(d.consensus, d.nn_similarity)
    stats = [
        ("n_train_positives", len(pos)),
        ("n_candidates", len(nov)),
        ("n_discoveries", len(d)),
        ("median_nn_all", round(nov.nn_similarity.median(), 4)),
        ("median_nn_discoveries", round(d.nn_similarity.median(), 4)),
        ("pct_below_0.70_discoveries", round(100 * (d.nn_similarity < 0.70).mean(), 1)),
        ("pct_at_or_above_0.90_discoveries", round(100 * (d.nn_similarity >= 0.90).mean(), 1)),
        ("median_nn_short_discoveries", round(d.nn_similarity[d.length < 11].median(), 4)),
        ("median_nn_long_discoveries", round(d.nn_similarity[d.length >= 11].median(), 4)),
        ("n_short_discoveries", int((d.length < 11).sum())),
        ("n_long_discoveries", int((d.length >= 11).sum())),
        ("spearman_rho_all", round(rho_a, 4)),
        ("spearman_p_all", float(f"{p_a:.3g}")),
        ("spearman_rho_discoveries", round(rho_d, 4)),
        ("spearman_p_discoveries", float(f"{p_d:.3g}")),
    ]
    pd.DataFrame(stats, columns=["quantity", "value"]).to_csv(OUT_NOVELTY_SUM, index=False)

    # the last three make the per-protein table's totals checkable, since a fragment
    # from two precursors is counted against each and the columns overshoot the pool
    ex = out.assign(source=out.sources.str.split(";")).explode("source")

    # how the pool narrows at each gate, which is the source of Figure 5's funnel
    funnel = [("n_candidates", len(seqs)),
              ("esm_above_0.5", int((esm_prob >= 0.5).sum())),
              ("xgb_above_0.5", int((xgb_prob >= 0.5).sum())),
              ("both_above_0.5", len(shortlist)),
              ("discoveries", len(discovery)),
              ("discovery_threshold", DISCOVERY_THRESHOLD),
              ("n_multi_source_fragments", int((out.sources.str.count(";") > 0).sum())),
              ("per_protein_candidate_sum", len(ex)),
              ("per_protein_discovery_sum", int(ex.discovery.sum()))]
    pd.DataFrame(funnel, columns=["quantity", "value"]).to_csv(OUT_SCORE_FUNNEL, index=False)


if __name__ == "__main__":
    main()