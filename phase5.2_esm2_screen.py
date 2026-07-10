
#!/usr/bin/env python3

# Install these:
# pip install torch transformers peft pandas numpy
# python phase5.2_esm2_screen.py

"""
Phase 5.2: Score the screening pool with the ESM-2 side of the consensus.

Two things are produced here, both requiring a forward pass through ESM-2, so
they live in one torch-only script (kept apart from the XGBoost stage in 5.3 to
avoid the macOS torch+xgboost libomp clash):

  1. frozen ESM-2 embeddings  -- the same 1280-d mean-pooled vectors as Phase 2.1,
     needed by XGBoost in 5.3 (it was trained on the fused vector).
  2. ESM-2/DoRA probabilities -- P(ADP) from the fine-tuned adapter (Phase 3.3),
     the ESM-2 half of the consensus ranking.

Featurisation is identical to training (same model, same mean-pool) so the
candidates are represented exactly as the classifiers expect.

INPUT   screening/screening_candidates.csv
OUTPUT  screening/screening_esm2.npz  (X_emb, esm_prob, peptide_id, sequence)

REQUIREMENTS  pip install torch transformers peft pandas numpy
              (the ESM-2 base model must be available/cached; DoRA adapter in models/)
"""

# Imports
import os
os.environ["HF_HUB_OFFLINE"] = "1"        # use the locally cached ESM-2
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # set these before transformers is imported

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from peft import PeftModel

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
CANDIDATES = "screening/screening_candidates.csv"
OUTPUT     = "screening/screening_esm2.npz"
MODEL_ID   = "facebook/esm2_t33_650M_UR50D" # ESM-2 650M (same as Phase 2/3)
ADAPTER    = "models/esm2_dora_adapter" # DoRA adapter saved in Phase 3.3
BATCH_SIZE = 16
MAX_LEN    = 64 # matches Phase 3.3 tokenisation
DEVICE = os.environ.get("DEVICE", "auto")   # I was running on mac but too slow, so now can run on viking.


def pick_device():
    # Fastest available backend: Apple MPS > CUDA > CPU.
    if DEVICE != "auto":
        return DEVICE
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def masked_mean_pool(hidden, attn):
    """Mean over residue positions only (identical to Phase 2.1).
    Excludes <cls> (token 0), <eos> (last real token), and padding, so the
    embedding summarises the real amino acids alone."""
    mask = attn.clone()
    mask[:, 0] = 0 # drop <cls>
    lengths = attn.sum(1)
    mask[torch.arange(mask.size(0)), lengths - 1] = 0 # drop <eos>
    mask = mask.unsqueeze(-1).to(hidden.dtype) # [B, T, 1] -> broadcast over H
    summed = (hidden * mask).sum(1)
    counts = mask.sum(1).clamp(min=1) # clamp avoids divide-by-zero
    return summed / counts


# --------------------------------------------------------------------------- #
# EMBEDDINGS  (frozen ESM-2, for XGBoost's fused vector)
# --------------------------------------------------------------------------- #
def embed(seqs, tok, device):
    """Frozen 1280-d mean-pooled ESM-2 embedding per sequence -> [N, 1280]."""
    model = AutoModel.from_pretrained(MODEL_ID).eval().to(device) # plain encoder, no adapter
    vecs = []
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH_SIZE):
            batch = seqs[i:i + BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LEN).to(device)
            out = model(**enc).last_hidden_state
            vecs.append(masked_mean_pool(out, enc["attention_mask"]).float().cpu().numpy())
            if (i // BATCH_SIZE) % 20 == 0:
                print(f"  embed {min(i + BATCH_SIZE, len(seqs))}/{len(seqs)}")
    del model
    return np.vstack(vecs).astype(np.float32)


# --------------------------------------------------------------------------- #
# DoRA SCORING  (fine-tuned adapter, the ESM-2 half of the consensus)
# --------------------------------------------------------------------------- #
def dora_proba(seqs, tok, device):
    """P(ADP) from the DoRA-fine-tuned classifier -> [N]."""
    # Load the base classifier, then attach the saved DoRA adapter (Phase 3.3).
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    model = PeftModel.from_pretrained(base, ADAPTER).eval().to(device)
    probs = []

    # Forward pass in batches, softmax to get P(ADP) from column 1 of the logits.
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH_SIZE):
            batch = seqs[i:i + BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LEN).to(device)
            logits = model(**enc).logits # **enc makes the model accept the tokenised batch
            probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())  # column 1 = P(ADP)
            if (i // BATCH_SIZE) % 20 == 0: # // means integer division, so this prints every 20 batches
                print(f"  score {min(i + BATCH_SIZE, len(seqs))}/{len(seqs)}")
    del model
    return np.concatenate(probs).astype(np.float32)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    # keep_default_na=False -> the peptide "NA" (Asn-Ala) stays a string, not NaN.
    df = pd.read_csv(CANDIDATES, keep_default_na=False)
    seqs = df["sequence"].astype(str).tolist()
    device = pick_device()
    print(f"{len(seqs)} candidates | device: {device}")

    # Load the ESM-2 tokenizer (same as Phase 2.1/3.3).
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # Embeddings first, then DoRA scoring (models loaded one at a time to keep
    # memory modest on my Mac).
    X_emb = embed(seqs, tok, device)
    esm_prob = dora_proba(seqs, tok, device)

    # sanity checks before saving
    assert X_emb.shape == (len(seqs), 1280), f"unexpected embedding shape {X_emb.shape}"
    assert esm_prob.shape == (len(seqs),) and np.isfinite(esm_prob).all()
    print(f"embeddings {X_emb.shape} | esm_prob range "
          f"[{esm_prob.min():.3f}, {esm_prob.max():.3f}] | "
          f"P>0.5: {(esm_prob > 0.5).sum()}/{len(seqs)}")

    # Save aligned to the candidate order (peptide_id/sequence carried along so
    # 5.3 can join back without ambiguity).
    np.savez_compressed(
        OUTPUT,
        X_emb=X_emb,
        esm_prob=esm_prob,
        peptide_id=df["peptide_id"].to_numpy(dtype=object),
        sequence=np.array(seqs, dtype=object),
    )
    print(f"saved -> {OUTPUT}")


if __name__ == "__main__":
    main()