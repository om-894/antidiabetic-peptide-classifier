#!/usr/bin/env python3


# Install these:
# -m pip install torch transformers
# pip install torch transformers pandas numpy
# Put esm2_embeddings.py next to dataset_split.csv.
# python esm2_embeddings.py


"""
Phase 2.1: Extract a fixed 1280-d ESM-2 embedding per sequence (for later fusion
with physicochemical descriptors).

Why: ESM-2 is pre-trained on ~65M proteins, so each embedding is a rich learned
representation that captures sequence context far better than handcrafted features
-- and transfer learning like this is what makes it work on a small dataset.
Here the model is frozen (a feature extractor for the trees/CNN); it gets
fine-tuned separately later (DoRA, Phase 3).

Per sequence: tokenise, forward pass through facebook/esm2_t33_650M_UR50D
(inference only), then mean-pool the final-layer hidden states over the real
residues only (excluding <cls>, <eos>, padding) -> one 1280-d vector.

Saved in dataset_split.csv row order, with sequences stored alongside so
alignment to the labels can be checked.

REQUIREMENTS  pip install torch transformers pandas numpy
              (first run downloads the model, ~2.5 GB)
"""

"""
Example — one sequence through the pipeline ("VAGTWY", a 6-residue ADP):

  1. raw string     "VAGTWY"
  2. tokenise       [<cls>] V A G T W Y [<eos>]          (6 residues + 2 special tokens)
  3. forward pass   one 1280-d vector per token       -> shape [8, 1280]
  4. mean-pool      average the 6 RESIDUE vectors only -> shape [1280]
                    (<cls> and <eos> excluded)
  5. store          X[i] = the 1280-d vector, alongside sequences[i]="VAGTWY",
                    label[i]=1, split[i]="train"

So a variable-length string becomes one fixed 1280-number vector -- ESM-2's
learned encoding of the peptide, which the downstream classifiers consume.
"""

# Imports
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
DATA       = "data/dataset_split.csv"
OUTPUT     = "esm2_embeddings.npz"
MODEL_ID   = "facebook/esm2_t33_650M_UR50D"   # ESM-2, 650M params
BATCH_SIZE = 16                               # sequences per forward pass
# "auto" picks up my Apple mac's MPS GPU if available, else CPU. If MPS throws an
# unsupported-op error, set this to "cpu".
DEVICE     = "auto"


def pick_device():
    # Use the fastest available backend: Apple MPS > CUDA > CPU.
    if DEVICE != "auto":
        return DEVICE
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def masked_mean_pool(hidden, attn):
    """Mean over residue positions only.
    hidden: [B, T, H] final-layer states.  attn: [B, T] (1 for <cls>..<eos>, 0 for pad).
    Excludes <cls> (first token), <eos> (last real token), and padding."""

    # B = Batch — how many sequences processed at once
    # T = Tokens — sequence length in tokens (<cls> + residues + <eos> + padding)
        # Varies per batch, since padding=True pads every sequence up to the longest one in that batch.
    # H = Hidden — the embedding size per token = 1280 for ESM-2 650M
    
    # Build a mask that is 1 only on real amino-acid residues.
    mask = attn.clone() # .clone() so we don't modify the original attention mask
    
    # drop <cls> (always token 0)
    mask[:, 0] = 0                                
    lengths = attn.sum(1) # real-token count per seq (<cls>..<eos>)
    mask[torch.arange(mask.size(0)), lengths - 1] = 0  # drop <eos> (the last real token)

    # [B, T, 1] so it broadcasts over H
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(1) # sum the residue vectors -> [B, H]
    counts = mask.sum(1).clamp(min=1) # clamp avoids diviion by 0
    return summed / counts # mean -> [B, H]


def main():
    df = pd.read_csv(DATA)
    seqs = df["Sequence"].astype(str).tolist()
    device = pick_device()
    print(f"{len(seqs)} sequences | device: {device}")

    # Load tokenizer + model once. .eval() disables dropout (inference mode).
    print(f"loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval().to(device)

    vecs = []
    with torch.no_grad(): # no gradients = faster, less memory
        for i in range(0, len(seqs), BATCH_SIZE):
            batch = seqs[i:i + BATCH_SIZE]

            # padding=True pads the batch to equal length; attention_mask marks
            # which tokens are real (1) vs padding (0).
            enc = tok(batch, return_tensors="pt", padding=True).to(device)
            out = model(**enc).last_hidden_state # [B, T, 1280] per-token vectors
            pooled = masked_mean_pool(out, enc["attention_mask"]) # [B, 1280] per-sequence
            
            # back to CPU/numpy to store
            vecs.append(pooled.float().cpu().numpy())
            if (i // BATCH_SIZE) % 10 == 0:
                print(f"  {min(i + BATCH_SIZE, len(seqs))}/{len(seqs)}")

    X = np.vstack(vecs).astype(np.float32) # stack all batches [N, 1280]

    # sanity checks - right shape and no NaNs before saving.
    assert X.shape == (len(seqs), 1280), f"unexpected shape {X.shape}"
    assert not np.isnan(X).any(), "NaNs in embeddings"
    print(f"embeddings: {X.shape}, dtype {X.dtype}, finite={np.isfinite(X).all()}")

    # Save embeddings + sequences/labels/splits together, in dataset_split.csv
    # row order, so downstream steps stay aligned to the labels.
    np.savez_compressed(
        OUTPUT,
        X=X,
        sequences=np.array(seqs, dtype=object),
        label=df["Label"].to_numpy(),
        split=df["Split"].to_numpy(),
    )
    print(f"saved -> {OUTPUT}  (load with np.load('{OUTPUT}', allow_pickle=True))")


if __name__ == "__main__":
    main()
