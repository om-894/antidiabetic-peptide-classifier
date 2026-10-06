"""
Phase 5.8: Re-score the screening pool under every ESM-2 fine-tuning seed.

The published shortlist rests on the seed-42 adapter, the highest of the five
refits at test AUC 0.877 against a 0.842 mean. This scores the same 3,596
candidates with all five adapters so the shortlist can be reported as a
selection frequency rather than as one fit's output.

Only the adapter probabilities are recomputed. The frozen 1,280-d embeddings
phase 5.3 feeds to XGBoost come from the base encoder with no adapter attached,
so they do not move with the seed and phase 5.2's screening_esm2.npz still
holds them. That halves the work per seed against re-running phase 5.2 five
times.

INPUTS  screening/screening_candidates.csv (peptide_id, sequence)
        models/esm2_seed{42..46}_adapter/, written by the phase 3.3 seed refits
OUTPUTS screening/screening_esm2_seeds.npz (esm_prob [5, N], seeds, peptide_id,
        sequence)
REQUIREMENTS  pip install torch transformers peft pandas numpy
              the ESM-2 base must be cached locally and all five adapters present
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "results")
WORK = os.path.join(REPO, "phase_7_revision", "work")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

CANDIDATES = "screening/screening_candidates.csv"
OUTPUT = "screening/screening_esm2_seeds.npz"
MODEL_ID = "facebook/esm2_t33_650M_UR50D" # the 650M base used in phases 2 and 3
BATCH_SIZE = 16 # 32 exceeded A40 memory at 64 tokens, as in phase 3.3
MAX_LEN = 64 # phase 3.3's tokenisation, long enough that nothing truncates
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]
DEVICE = os.environ.get("DEVICE", "auto")

# the seed refits wrote models/esm2_seed{SEED}_adapter. seed 42 repeats the main
# phase 3.3 fit exactly, so its adapter falls back to the deployed one when the
# per-seed copy was never kept
ADAPTER_FMT = "models/esm2_seed{seed}_adapter"
ADAPTER_42_FALLBACK = "models/esm2_dora_adapter"


def pick_device():
    if DEVICE != "auto":
        return DEVICE
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def adapter_path(seed):
    """Directory holding the DoRA adapter for one seed."""
    path = ADAPTER_FMT.format(seed=seed)
    if not os.path.isdir(path) and seed == 42 and os.path.isdir(ADAPTER_42_FALLBACK):
        return ADAPTER_42_FALLBACK
    return path


def resolve_adapters():
    """Map each seed to its adapter directory, failing on any that is absent."""
    # five hours of GPU time can be wasted by discovering a missing adapter on
    # the last seed, so every path is checked before the first model loads
    paths = {seed: adapter_path(seed) for seed in SEEDS}
    missing = [f"seed {s}: {p}" for s, p in paths.items() if not os.path.isdir(p)]
    if missing:
        raise SystemExit(
            "missing adapters, run the phase 3.3 seed refits for these first:\n  "
            + "\n  ".join(missing)
        )
    return paths


# --------------------------------------------------------------------------- #
# DoRA SCORING
# --------------------------------------------------------------------------- #

def dora_proba(seqs, tok, device, adapter):
    """P(ADP) from one DoRA-fine-tuned classifier, shaped [N]."""
    # the adapter is a set of weight updates rather than a whole model, so the
    # base classifier loads first and PeftModel layers the fine-tune on top
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    model = PeftModel.from_pretrained(base, adapter).eval().to(device)
    probs = []
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH_SIZE):
            enc = tok(seqs[i:i + BATCH_SIZE], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LEN).to(device)
            logits = model(**enc).logits

            # softmax in float32 rather than bf16, matching phase 3.3. column one
            # is the ADP class
            probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
            if (i // BATCH_SIZE) % 40 == 0:
                print(f"  scored {min(i + BATCH_SIZE, len(seqs))} of {len(seqs)}")

    # del alone only drops the reference. without emptying the cache each seed's
    # weights stay in torch's allocator while the next 650M model loads
    del model, base
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(probs).astype(np.float32)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    adapters = resolve_adapters()

    # keep_default_na=False, or the dipeptide NA would parse as a missing value
    df = pd.read_csv(CANDIDATES, keep_default_na=False)
    seqs = df["sequence"].astype(str).tolist()
    device = pick_device()
    print(f"{len(seqs)} candidates on {device}, seeds {SEEDS}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # one model in memory at a time, same reason as phase 5.2
    rows = []
    for seed in SEEDS:
        print(f"seed {seed} from {adapters[seed]}")
        prob = dora_proba(seqs, tok, device, adapters[seed])
        assert prob.shape == (len(seqs),), f"unexpected shape {prob.shape} at seed {seed}"
        assert np.isfinite(prob).all(), f"non-finite probability at seed {seed}"
        print(f"  {(prob > 0.5).sum()} of {len(seqs)} above 0.5")
        rows.append(prob)

    esm_prob = np.vstack(rows)
    assert esm_prob.shape == (len(SEEDS), len(seqs)), f"unexpected matrix {esm_prob.shape}"

    # peptide_id and sequence travel with the arrays, so the local stability
    # script joins on an id rather than trusting row order across two files
    np.savez_compressed(OUTPUT, esm_prob=esm_prob, seeds=np.array(SEEDS),
                        peptide_id=df["peptide_id"].to_numpy(dtype=object),
                        sequence=np.array(seqs, dtype=object))
    print(f"saved {OUTPUT} with shape {esm_prob.shape}")


if __name__ == "__main__":
    main()
