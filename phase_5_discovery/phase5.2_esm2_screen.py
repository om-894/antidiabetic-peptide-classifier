
"""
Phase 5.2: Score the screening pool with the ESM-2 half of the consensus.

Two things come out of one script because both need a forward pass through
ESM-2. Keeping them together also keeps torch away from xgboost, which phase 5.3
uses, since the two bundle their own libomp and segfault when imported side by
side on macOS.

  embeddings -> frozen 1280-d mean-pooled vectors, as in phase 2.1, which
                phase 5.3 needs to rebuild the fused vector for XGBoost.
  scores -> P(ADP) from the DoRA adapter fine-tuned in phase 3.3.

Featurisation matches training exactly, same base model and same mean pool, so
the candidates reach each classifier represented the way it was trained.

INPUTS  screening_candidates.csv (peptide_id, sequence)
OUTPUTS  screening_esm2.npz (X_emb, esm_prob, peptide_id, sequence)
REQUIREMENTS  pip install torch transformers peft pandas numpy
              the ESM-2 base must be cached locally and the DoRA adapter present
"""

# Imports
import os


# set these before the first from_pretrained call. transformers checks them when
# resolving a model/tokenizer. Viking's GPU nodes have no internet, so offline
# mode prevents a hub lookup from blocking the job.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from peft import PeftModel


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

CANDIDATES = "screening/screening_candidates.csv"
OUTPUT = "screening/screening_esm2.npz"
MODEL_ID = "facebook/esm2_t33_650M_UR50D" # the 650M base used in phases 2 and 3
ADAPTER = "models/esm2_dora_adapter" # the adapter saved by phase 3.3
BATCH_SIZE = 16 # 32 exceeded A40 memory at 64 tokens, as in phase 3.3
MAX_LEN = 64 # phase 3.3's tokenisation, long enough that nothing truncates
DEVICE = os.environ.get("DEVICE", "auto")


def pick_device():
    # mps first, since my machine is a Mac and Viking has no mps to offer,
    # so the order costs nothing there and picks the accelerator here
    if DEVICE != "auto":
        return DEVICE
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def masked_mean_pool(hidden, attn):
    """Mean over residue positions only, identical to phase 2.1."""
    # <cls> and <eos> carry no residue, so averaging them in would shift every
    # vector by the same amount and dilute short peptides most
    mask = attn.clone()
    mask[:, 0] = 0
    lengths = attn.sum(1)
    mask[torch.arange(mask.size(0)), lengths - 1] = 0
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(1)
    counts = mask.sum(1).clamp(min=1) # clamp guards a hypothetical empty sequence
    return summed / counts


# --------------------------------------------------------------------------- #
# EMBEDDINGS
# --------------------------------------------------------------------------- #

def embed(seqs, tok, device):
    """Frozen mean-pooled ESM-2 embedding per sequence, shaped [N, 1280]."""
    # the plain encoder, with no adapter attached, so these match the phase 2.1
    # vectors XGBoost was trained on rather than the fine-tuned representation
    model = AutoModel.from_pretrained(MODEL_ID).eval().to(device)
    vecs = []
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH_SIZE):
            enc = tok(seqs[i:i + BATCH_SIZE], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LEN).to(device)
            out = model(**enc).last_hidden_state
            vecs.append(masked_mean_pool(out, enc["attention_mask"]).float().cpu().numpy())

            # the SLURM out-file is the only record of a Viking run. this one is
            # long, so progress goes to stdout every twenty batches
            if (i // BATCH_SIZE) % 20 == 0:
                print(f"embedded {min(i + BATCH_SIZE, len(seqs))} of {len(seqs)}")

    # del alone only drops the reference. without emptying the cache the weights
    # stay in torch's allocator while the second 650M model loads below
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(vecs).astype(np.float32)


# --------------------------------------------------------------------------- #
# DoRA SCORING
# --------------------------------------------------------------------------- #

def dora_proba(seqs, tok, device):
    """P(ADP) from the DoRA-fine-tuned classifier, shaped [N]."""
    # the adapter is a set of weight updates rather than a whole model, so the
    # base classifier loads first and PeftModel layers phase 3.3's training on top
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    model = PeftModel.from_pretrained(base, ADAPTER).eval().to(device)
    probs = []
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH_SIZE):
            enc = tok(seqs[i:i + BATCH_SIZE], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LEN).to(device)
            logits = model(**enc).logits

            # softmax in float32 rather than bf16, matching phase 3.3. column one
            # is the ADP class
            probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
            if (i // BATCH_SIZE) % 20 == 0:
                print(f"scored {min(i + BATCH_SIZE, len(seqs))} of {len(seqs)}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(probs).astype(np.float32)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    # keep_default_na=False, or the dipeptide NA would parse as a missing value
    df = pd.read_csv(CANDIDATES, keep_default_na=False)
    seqs = df["sequence"].astype(str).tolist()
    device = pick_device()
    print(f"{len(seqs)} candidates on {device}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # one model in memory at a time
    X_emb = embed(seqs, tok, device)
    esm_prob = dora_proba(seqs, tok, device)

    # a wrong shape here would surface as a silently misaligned join in phase 5.3
    assert X_emb.shape == (len(seqs), 1280), f"unexpected embedding shape {X_emb.shape}"
    assert esm_prob.shape == (len(seqs),), f"unexpected probability shape {esm_prob.shape}"
    assert np.isfinite(esm_prob).all(), "non-finite probability from the adapter"
    print(f"{(esm_prob > 0.5).sum()} of {len(seqs)} candidates score above 0.5")

    # peptide_id and sequence travel with the arrays, so phase 5.3 joins on an id
    # rather than trusting row order across two files
    np.savez_compressed(OUTPUT, X_emb=X_emb, esm_prob=esm_prob,
                        peptide_id=df["peptide_id"].to_numpy(dtype=object),
                        sequence=np.array(seqs, dtype=object))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()