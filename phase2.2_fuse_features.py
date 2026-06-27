#!/usr/bin/env python3
"""
Phase 2.2: Fuse features. Compute 7 physicochemical descriptors per sequence,
z-score them (fit on TRAIN only), and concatenate onto the frozen ESM-2 embeddings:

  embeddings (1280) + z-scored descriptors (7) = 1287-d vector

Why z-score: the 7 descriptors live on very different scales (e.g. charge ~ -5..+10,
instability index ~ 0..100) and are far larger than the embedding values (~ -1..1).
Left raw, they'd dominate the model by magnitude alone, so z-scoring (mean 0, std 1)
puts them on a comparable footing with each other and the embeddings. The embeddings
are already well-scaled, so only the descriptors are normalised.

Z-score stats are fit on train rows alone -> no test leakage, and are saved so the
same transform can be reapplied to novel candidate peptides later.

INPUT   esm2_embeddings.npz        OUTPUT  fusion_vectors.npz
REQUIREMENTS  pip install peptides scikit-learn numpy
"""

# Imports
import numpy as np
import peptides
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

EMB_IN  = "esm2_embeddings.npz"
OUTPUT  = "fusion_vectors.npz"

DESCRIPTOR_NAMES = [
    "net_charge_pH7.4", "gravy_kd", "isoelectric_point", "aromaticity",
    "instability_index", "aliphatic_index", "boman_index",
]


def descriptors(seq):
    """Seven global physicochemical descriptors, in DESCRIPTOR_NAMES order."""
    p = peptides.Peptide(seq) # peptides library does the chemistry
    aromaticity = sum(seq.count(a) for a in "FWY") / len(seq)   # Lobry FWY fraction
    
    return [
        p.charge(pH=7.4),                       # net charge at physiological pH
        p.hydrophobicity(scale="KyteDoolittle"),# GRAVY (mean hydrophobicity)
        p.isoelectric_point(),                  # pI (pH where net charge = 0)
        aromaticity,                            # fraction of aromatic residues (F/W/Y)
        p.instability_index(),                  # Guruprasad stability estimate
        p.aliphatic_index(),                    # relative volume of aliphatic side chains
        p.boman(),                              # protein-binding 
    ]

# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    # Load the frozen embeddings + their aligned seqs/labels/splits.
    d = np.load(EMB_IN, allow_pickle=True)
    
    X_emb = d["X"].astype(np.float32)           # [N, 1280]
    seqs = [str(s) for s in d["sequences"]]
    label, split = d["label"], d["split"]
    
    # embeddings and seqs must line up
    assert X_emb.shape[0] == len(seqs)

    # Compute the 7 descriptors for every sequence -> [N, 7].
    D = np.array([descriptors(s) for s in seqs], dtype=np.float32)   # [N, 7]
    assert np.isfinite(D).all(), "non-finite descriptor value"
    print(f"descriptors: {D.shape}  ({', '.join(DESCRIPTOR_NAMES)})")

    # Z-score the descriptors, fitting mean/std on TRAIN rows ONLY (so the test
    # set never influences normalisation), then apply to all rows.
    train = split == "train"
    scaler = StandardScaler().fit(D[train])
    D_z = scaler.transform(D).astype(np.float32)

    # Fuse: embeddings (unchanged) + z-scored descriptors, side by side.
    X_fused = np.hstack([X_emb, D_z]).astype(np.float32)            # [N, 1287]
    print(f"fused vector: {X_fused.shape}  "
          f"(dims 0-1279 = ESM-2, 1280-1286 = descriptors)")

    # Sanity checks: right shape, no NaNs and z-scoring actually worked
    # (train means ~0, train stds ~1).
    assert X_fused.shape == (len(seqs), 1287)
    assert not np.isnan(X_fused).any()
    tr_mean = D_z[train].mean(0)
    tr_std = D_z[train].std(0)
    print(f"train descriptor means after z-score: {np.round(tr_mean, 3)}  (~0 expected)")
    print(f"train descriptor stds  after z-score: {np.round(tr_std, 3)}  (~1 expected)")

    # Save the fused vectors, plus the raw descriptors (for interpretability/SHAP),
    # the descriptor names, and the z-score stats (mean/std) so they can be
    # re-applied to novel candidate peptides later.
    np.savez_compressed(
        OUTPUT,
        X=X_fused,
        descriptors_raw=D, # un-normalised descriptors, for interpretability/SHAP
        descriptor_names=np.array(DESCRIPTOR_NAMES, dtype=object),
        desc_mean=scaler.mean_.astype(np.float32), # saved to re-apply the same
        desc_std=scaler.scale_.astype(np.float32), # z-score to novel peptides later
        label=label, split=split,
        sequences=np.array(seqs, dtype=object),
    )


if __name__ == "__main__":
    main()
