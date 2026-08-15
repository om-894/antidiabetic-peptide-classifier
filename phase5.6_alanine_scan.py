
"""
Phase 5.6 (alanine scan): residue importance for the case-study peptide.

Mutates each position to alanine, re-scores with the fine-tuned ESM-2/DoRA model, and
reports the drop in ADP log-odds from wild-type (delta = logodds_wt - logodds_mut).
Log-odds, not probability, because a strong lead saturates near P=1 - log-odds is
unbounded and exposes which residues actually drive the call (large delta = matters).
In-silico version of wet-lab alanine mutagenesis.

INPUT   models/esm2_dora_adapter/          fine-tuned adapter (Phase 3.3)
OUTPUT  results/phase5_6_alanine_scan.csv  peptide, position, wt, P_mut, delta_logodds

REQUIREMENTS  pip install torch transformers peft pandas numpy
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1" # use the cached ESM-2
os.environ["TRANSFORMERS_OFFLINE"] = "1" # (set before transformers is imported)

# imports
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
MODEL_ID = "facebook/esm2_t33_650M_UR50D"
ADAPTER  = "models/esm2_dora_adapter"
DEVICE   = os.environ.get("DEVICE", "cpu")
MAX_LEN  = 64

# case-study peptides to scan
PEPTIDES = {
    "FVAPFPEVF": "FVAPFPEVF", # top Tier-1 lead / case study
}


def load_model():
    # ESM-2 with a 2-class head, then load DoRA fine-tune on top.
    # The adapter also stores the trained classifier head, so this is the Phase-3 model.
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    return PeftModel.from_pretrained(base, ADAPTER).eval().to(DEVICE)


def logodds(seqs, tok, model):
    """ADP log-odds (logit_ADP - logit_nonADP) per sequence, plus P(ADP) for reference.
    Log-odds is the importance signal - it isn't capped at 1 like the probability, so it
    still moves for a lead the model is already sure about."""
    with torch.no_grad():
        enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)
        lg = model(**enc).logits.float() # two raw scores per sequence: [non-ADP, ADP]
    p = torch.softmax(lg, 1)[:, 1].cpu().numpy() # P(ADP), just for the printout
    return (lg[:, 1] - lg[:, 0]).cpu().numpy(), p # ADP minus non-ADP = the log-odds


def scan(seq, tok, model):
    """Wild-type + every single-alanine mutant -> how much each residue matters."""
    # build the wild-type, then one mutant per position with that residue swapped for alanine
    variants = [seq] + [seq[:i] + "A" + seq[i + 1:] for i in range(len(seq))]
    lo, p = logodds(variants, tok, model)
    lo_wt, p_wt = lo[0], p[0] # first entry is the wild-type

    # per residue: how far the ADP log-odds falls when it's mutated to alanine.
    # a big drop means the model leans heavily on that residue for the ADP call.
    rows = [{"position": i + 1, "wt": seq[i],
             "P_mut": round(float(p[1 + i]), 4),
             "wt_P_adp": round(float(p_wt), 4),
             "wt_logodds": round(float(lo_wt), 3),
             "delta_logodds": round(float(lo_wt - lo[1 + i]), 3),
             "already_ala": seq[i] == "A",} for i in range(len(seq))] # A->A is a no-op, flag it
    return lo_wt, p_wt, pd.DataFrame(rows)


def main():
    print(f"device: {DEVICE}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)  # tokenizer for the base ESM-2 model
    model = load_model()
    os.makedirs("results", exist_ok=True)

    # scan each case-study peptide and save its table
    for name, seq in PEPTIDES.items():
        lo_wt, p_wt, df = scan(seq, tok, model)
        df.insert(0, "peptide", name)
        df.to_csv("results/phase5_6_alanine_scan.csv", index=False)
        print(f"\n{name}: wild-type P(ADP)={p_wt:.3f}, log-odds={lo_wt:.2f}")
        print(df.sort_values("delta_logodds", ascending=False).to_string(index=False))
        print("saved -> results/phase5_6_alanine_scan.csv")


if __name__ == "__main__":
    main()