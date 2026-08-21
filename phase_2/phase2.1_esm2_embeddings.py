
"""
Phase 2.1: Extract a fixed 1280-d ESM-2 embedding per sequence, for later fusion
with physicochemical descriptors.

ESM-2 is pre-trained on ~65M protein sequences, so transfer learning gives a
richer, context-aware representation than handcrafted features on a dataset this
small. esm2_t33_650M_UR50D is frozen here as a feature extractor for the
trees/CNN. It gets fine-tuned separately in Phase 3 (DoRA).

Worked example, "VAGTWY" (a 6-residue ADP from the positive set)

  1. raw string     "VAGTWY"
  2. tokenise       [<cls>] V A G T W Y [<eos>]           (6 residues + 2 special tokens)
  3. forward pass   final-layer vector per token -> shape [8, 1280]
  4. mean-pool      average the 6 RESIDUE vectors only -> shape [1280]
                    (<cls>, <eos> and padding excluded)
  5. store          X[i] = the 1280-d vector, alongside sequences[i]="VAGTWY",
                    label[i]=1, split[i]="train"

INPUTS  dataset_split.csv (1932 sequences with Label and Split)
OUTPUTS  esm2_embeddings.npz (X [N, 1280], sequences, label, split), saved in
              dataset_split.csv row order so alignment can be checked downstream
REQUIREMENTS  pip install torch transformers pandas numpy
              (first run downloads the model, ~2.5 GB)
"""

# Imports
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
DATA = "data/dataset_split.csv"
OUTPUT = "esm2_embeddings.npz"
MODEL_ID = "facebook/esm2_t33_650M_UR50D" # ESM-2, 650M params
BATCH_SIZE = 16 # sequences per forward pass


# "auto" resolves to MPS on Apple, else CUDA on a GPU node, else CPU.
DEVICE = "auto"

# Define function to choose
def pick_device():
    if DEVICE != "auto":
        return DEVICE
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def masked_mean_pool(hidden, attn):
    """Mean over residue positions only, so <cls>, <eos> and padding are excluded.
    hidden is [B, T, H] final-layer states, attn is [B, T] with 1 on <cls>..<eos>."""

    # B = Batch, how many sequences go through in one forward pass
    # T = Tokens, sequence length in tokens (<cls> + residues + <eos> + padding).
    #     Varies per batch, since padding=True pads up to the longest seq in that batch.
    # H = Hidden, the embedding size per token = 1280 for ESM-2 650M

    # Build a mask that is 1 only on real amino-acid residues.
    mask = attn.clone() # .clone() so we don't modify the original attention mask
    mask[:, 0] = 0 # drop <cls>, always token 0

    # <eos> sits at a different column in every row, since each sequence has its own
    # length, so index the rows and their <eos> columns together to zero one per row.
    lengths = attn.sum(1) # real-token count per seq (<cls>..<eos>)
    mask[torch.arange(mask.size(0)), lengths - 1] = 0

    mask = mask.unsqueeze(-1).to(hidden.dtype) # [B, T, 1] so it broadcasts over H
    summed = (hidden * mask).sum(1) # sum the residue vectors -> [B, H]
    counts = mask.sum(1).clamp(min=1) # defensive, a sequence with no residues would divide by 0
    return summed / counts # mean -> [B, H]


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    df = pd.read_csv(DATA)
    seqs = df["Sequence"].astype(str).tolist()
    device = pick_device()

    # Load tokenizer + model once. .eval() disables dropout (inference mode).
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
            vecs.append(pooled.float().cpu().numpy()) # back to CPU/numpy to store

    X = np.vstack(vecs).astype(np.float32) # stack all batches [N, 1280]

    # Sanity checks before saving. isfinite covers NaN and inf, so one test does both.
    assert X.shape == (len(seqs), 1280), f"unexpected shape {X.shape}"
    assert np.isfinite(X).all(), "non-finite values in embeddings"

    # Save embeddings + sequences/labels/splits together, in dataset_split.csv
    # row order, so downstream steps stay aligned to the labels. sequences is an
    # object array, so loading needs np.load(..., allow_pickle=True).
    np.savez_compressed(
        OUTPUT,
        X=X,
        sequences=np.array(seqs, dtype=object),
        label=df["Label"].to_numpy(),
        split=df["Split"].to_numpy(),
    )


if __name__ == "__main__":
    main()
