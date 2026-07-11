

# Install:
# pip install xgboost scikit-learn joblib peptides pandas numpy

"""
Phase 5.3: Fuse features, score with XGBoost and rank the screening pool by consensus.

The tree half of the consensus (separate from the torch stage in 5.2 to avoid the
macOS torch+xgboost libomp clash):
  1. rebuild the fused vector as in Phase 2.2 - 7 descriptors, z-scored with the
     saved train stats (desc_mean/desc_std), concatenated onto the 5.2 embeddings.
  2. score with the trained XGBoost base learner -> P(ADP).
  3. average with the ESM-2/DoRA probability -> consensus, then rank.

Consensus, not one model: ESM-2 alone flags roughly half the pool and is mildly
miscalibrated, so ranking on the mean of two independent views and keeping only
candidates both call positive is a more precise screen. The ranking is what's used,
not the absolute probability.

INPUT   screening/screening_esm2.npz        X_emb, esm_prob, peptide_id, sequence
        screening/screening_candidates.csv  metadata: length, sources, enzymes
        fusion_vectors.npz                   desc_mean/desc_std (the train z-score)
        models/base_xgb.joblib              XGBoost base learner, 1287-d input
OUTPUT  screening/screening_ranked.csv       every candidate, ranked, both scores
        screening/screening_shortlist.csv    high-confidence subset (feeds 5.4)

REQUIREMENTS  pip install xgboost scikit-learn joblib peptides pandas numpy
"""

# Imports
import joblib
import numpy as np
import pandas as pd
import peptides

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
ESM2_NPZ   = "screening/screening_esm2.npz"
CANDIDATES = "screening/screening_candidates.csv"
FUSION_NPZ = "fusion_vectors.npz" # for the saved train z-score stats
XGB_MODEL  = "models/base_xgb.joblib" # base learner (matches the multi-seed results)

OUT_RANKED    = "screening/screening_ranked.csv"
OUT_SHORTLIST = "screening/screening_shortlist.csv"

# Agreement threshold: a candidate is "high-confidence" only if both models put it
# above this. 0.5 = each model's own positive call; raise it for a stricter screen.
THRESHOLD = 0.5

# Same 7 descriptors, same order as Phase 2.2 (must match how XGBoost was trained).
DESCRIPTOR_NAMES = [
    "net_charge_pH7.4", "gravy_kd", "isoelectric_point", "aromaticity",
    "instability_index", "aliphatic_index", "boman_index",
]


def descriptors(seq):
    """The 7 global physicochemical descriptors, in DESCRIPTOR_NAMES order
    (identical to Phase 2.2 so the fused vector matches training)."""
    p = peptides.Peptide(seq)
    aromaticity = sum(seq.count(a) for a in "FWY") / len(seq)   # Lobry FWY fraction
    return [
        p.charge(pH=7.4),
        p.hydrophobicity(scale="KyteDoolittle"),
        p.isoelectric_point(),
        aromaticity,
        p.instability_index(),
        p.aliphatic_index(),
        p.boman(),
    ]


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    # ESM-2 outputs from 5.2 (embeddings + DoRA probabilities), all aligned.
    z = np.load(ESM2_NPZ, allow_pickle=True)
    X_emb = z["X_emb"].astype(np.float32) # [N, 1280]
    esm_prob = z["esm_prob"].astype(float) # [N]
    pid = [str(x) for x in z["peptide_id"]]
    seqs = [str(s) for s in z["sequence"]]
    print(f"{len(seqs)} candidates loaded from 5.2")

    # Pull the candidate metadata (length/sources/enzymes) and reorder it to match
    # the npz row order. keep_default_na=False so the peptide "NA" stays a string.
    cand = pd.read_csv(CANDIDATES, keep_default_na=False).set_index("peptide_id")
    
    meta = cand.loc[pid] # reindex to npz order
    assert (meta["sequence"].to_numpy() == np.array(seqs)).all(), "sequence/order mismatch"

    # Rebuild the fused vector exactly as Phase 2.2: descriptors, then z-score with
    # the saved train stats (never re-fit on the candidates -> no leakage), then
    # concatenate onto the embeddings.
    D = np.array([descriptors(s) for s in seqs], dtype=np.float32) # [N, 7]
    assert np.isfinite(D).all(), "non-finite descriptor"
    fz = np.load(FUSION_NPZ, allow_pickle=True)
    D_z = (D - fz["desc_mean"]) / fz["desc_std"] # apply train z-score
    X_fused = np.hstack([X_emb, D_z]).astype(np.float32) # [N, 1287]
    assert X_fused.shape[1] == 1287

    # XGBoost probability, then the consensus (mean of the two independent views).
    xgb = joblib.load(XGB_MODEL)
    xgb_prob = xgb.predict_proba(X_fused)[:, 1] # P(ADP)
    consensus = 0.5 * (esm_prob + xgb_prob)

    # Assemble the ranked table.
    out = pd.DataFrame({
        "peptide_id": pid,
        "sequence": seqs,
        "length": meta["length"].to_numpy(),
        "sources": meta["sources"].to_numpy(),
        "enzymes": meta["enzymes"].to_numpy(),
        "esm_prob": np.round(esm_prob, 4),
        "xgb_prob": np.round(xgb_prob, 4),
        "consensus": np.round(consensus, 4),
        
        # high-confidence = both models independently call it positive
        "high_confidence": (esm_prob >= THRESHOLD) & (xgb_prob >= THRESHOLD),
    }).sort_values("consensus", ascending=False).reset_index(drop=True)

    out.to_csv(OUT_RANKED, index=False)
    shortlist = out[out["high_confidence"]].reset_index(drop=True)
    shortlist.to_csv(OUT_SHORTLIST, index=False)

    # Show how many candidates each model calls positive and how many both agree on.
    print(f"ESM-2 P>0.5 : {(esm_prob >= 0.5).sum()}/{len(seqs)}")
    print(f"XGB   P>0.5 : {(xgb_prob >= 0.5).sum()}/{len(seqs)}")
    print(f"BOTH agree  : {len(shortlist)}/{len(seqs)}  (high-confidence shortlist)")
    print(f"\nsaved -> {OUT_RANKED}  and  {OUT_SHORTLIST}")
    print("\ntop 10 by consensus:")
    print(shortlist.head(10)[["peptide_id", "sequence", "length",
                              "esm_prob", "xgb_prob", "consensus"]].to_string(index=False))


if __name__ == "__main__":
    main()