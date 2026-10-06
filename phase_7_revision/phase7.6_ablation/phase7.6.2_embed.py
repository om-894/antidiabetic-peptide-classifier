"""
Revision analysis: frozen ESM-2 embeddings for the sequences the ablation adds.

The six arms draw negatives the published dataset never contained, so
esm2_embeddings.npz covers only 1,932 of the 4,686 sequences they use. This
embeds the remainder with the same frozen encoder and the same masked mean
pool as phase 2.1, including its padding=True with no truncation, so the new
vectors sit in the same space as the originals.

INPUTS  ablation_arms.csv, fusion_vectors.npz
OUTPUTS ablation_embeddings.npz (X [N, 1280], sequences)
        need_embed.txt, the sequence list, kept for inspection
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "results")
WORK = os.path.join(REPO, "phase_7_revision", "work")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

MODEL_ID = "facebook/esm2_t33_650M_UR50D"
BATCH_SIZE = 16
DEVICE = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")


def masked_mean_pool(hidden, attn):
    """Mean over residue positions only, identical to phase 2.1."""
    mask = attn.clone()
    mask[:, 0] = 0
    lengths = attn.sum(1)
    mask[torch.arange(mask.size(0)), lengths - 1] = 0
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(1)
    counts = mask.sum(1).clamp(min=1)
    return summed / counts


def needed():
    """Arm sequences that phase 2.2 never embedded, in sorted order."""
    # derived here rather than passed in, so the ablation runs end to end from
    # the arms file alone and the list cannot drift from it
    have = set(np.load(f"{REPO}/fusion_vectors.npz",
                       allow_pickle=True)["sequences"].tolist())
    arms = pd.read_csv(f"{WORK}/ablation_arms.csv", keep_default_na=False)
    seqs = sorted(set(arms.Sequence) - have)
    open(f"{WORK}/need_embed.txt", "w").write("\n".join(seqs) + "\n")
    return seqs


def main():
    seqs = needed()
    print(f"{len(seqs)} sequences on {DEVICE}")

    # longest first, so each batch pads to a similar length and the slowest
    # batches run while nothing else is competing for the device
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval().to(DEVICE)

    vecs = np.zeros((len(seqs), 1280), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            enc = tok([seqs[j] for j in idx], return_tensors="pt", padding=True).to(DEVICE)
            out = model(**enc).last_hidden_state
            vecs[idx] = masked_mean_pool(out, enc["attention_mask"]).float().cpu().numpy()
            if (i // BATCH_SIZE) % 20 == 0:
                print(f"  embedded {min(i + BATCH_SIZE, len(order))} of {len(order)}")

    assert np.isfinite(vecs).all(), "non-finite values in embeddings"
    np.savez_compressed(f"{WORK}/ablation_embeddings.npz", X=vecs,
                        sequences=np.array(seqs, dtype=object))
    print(f"saved ablation_embeddings.npz {vecs.shape}")


if __name__ == "__main__":
    main()
