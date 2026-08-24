
"""
Phase 2.2: Fuse features. Compute 7 physicochemical descriptors per sequence,
z-score them on TRAIN rows only then concatenate onto the frozen ESM-2 embeddings.

  embeddings (1280) + z-scored descriptors (7) = 1287-d vector
  dims 0-1279 are ESM-2, dims 1280-1286 are the descriptors in this order
  net charge (pH 7.4), GRAVY, isoelectric point, aromaticity (FWY fraction),
  instability index, aliphatic index, Boman index

The descriptors live on very different scales (charge ~ -5..+10, instability
index ~ 0..100) and are far larger than the embedding values (~ -1..1). Left raw
they'd dominate by magnitude alone, both over the embeddings and over each other.
Only the descriptors are normalised, since the embeddings are already well-scaled.

Fitting the mean and std on train rows alone keeps the test set out of the
normalisation. Both are saved so the same transform reaches novel candidates later.

INPUTS  esm2_embeddings.npz (X, sequences, label, split)
OUTPUTS  fusion_vectors.npz (X [N, 1287], descriptors_raw, descriptor_names,
              desc_mean, desc_std, sequences, label, split)
REQUIREMENTS  pip install peptides scikit-learn numpy
"""

# Imports
import numpy as np
import peptides
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

EMB_IN = "esm2_embeddings.npz"
OUTPUT = "fusion_vectors.npz"

# Define the 7 physicochemical descriptors, in this order. The names are saved to
# the output file so they can be re-used for interpretability/SHAP later.
DESCRIPTOR_NAMES = [
    "net_charge_pH7.4", "gravy_kd", "isoelectric_point", "aromaticity",
    "instability_index", "aliphatic_index", "boman_index",
]


def descriptors(seq):
    """Seven global physicochemical descriptors, in DESCRIPTOR_NAMES order."""
    p = peptides.Peptide(seq) # peptides library does the chemistry
    aromaticity = sum(seq.count(a) for a in "FWY") / len(seq)   # Lobry FWY fraction
    
    return [
        p.charge(pH=7.4), # net charge at physiological pH
        p.hydrophobicity(scale="KyteDoolittle"), # GRAVY (mean hydrophobicity)
        p.isoelectric_point(), # pI (pH where net charge = 0)
        aromaticity, # fraction of aromatic residues (F/W/Y)
        p.instability_index(), # Guruprasad stability estimate
        p.aliphatic_index(), # relative volume of aliphatic side chains
        p.boman(), # protein-binding 
    ]

# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    # Load the frozen embeddings + their aligned seqs/labels/splits.
    d = np.load(EMB_IN, allow_pickle=True)

    X_emb = d["X"].astype(np.float32) # [N, 1280]
    seqs = [str(s) for s in d["sequences"]]
    label, split = d["label"], d["split"]
    assert X_emb.shape[0] == len(seqs), "embeddings and sequences don't line up"

    # Compute the 7 descriptors for every sequence -> [N, 7].
    D = np.array([descriptors(s) for s in seqs], dtype=np.float32)
    assert np.isfinite(D).all(), "non-finite descriptor value"

    # Z-score the descriptors, fitting mean/std on train rows only (so the test
    # set never influences normalisation), then apply to all rows.
    train = split == "train"
    scaler = StandardScaler().fit(D[train])
    D_z = scaler.transform(D).astype(np.float32)

    # Fuse the unchanged embeddings with the z-scored descriptors, side by side.
    X_fused = np.hstack([X_emb, D_z]).astype(np.float32) # [N, 1287]

    # Sanity checks. Shape, finite values, plus confirmation that z-scoring worked
    # by testing the train rows land on mean 0 and std 1.
    tr_mean = D_z[train].mean(0)
    tr_std = D_z[train].std(0)
    assert X_fused.shape == (len(seqs), 1287), f"unexpected shape {X_fused.shape}"
    assert np.isfinite(X_fused).all(), "non-finite value in fused vector"
    assert np.allclose(tr_mean, 0, atol=1e-4), f"train means not near 0, got {tr_mean}"
    assert np.allclose(tr_std, 1, atol=1e-4), f"train stds not near 1, got {tr_std}"

    # Save the fused vectors, plus the raw descriptors (for interpretability/SHAP),
    # the descriptor names and the z-score stats (mean/std) so they can be
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
