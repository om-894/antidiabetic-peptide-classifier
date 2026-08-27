
"""
Phase 5.6: Alanine scan of the case-study peptide (FVAPFPEVF, the top Tier-1 lead).

Mutates each position to alanine, re-scores with the fine-tuned ESM-2/DoRA model and
reports the drop in ADP log-odds from wild-type (delta = logodds_wt - logodds_mut).
Log-odds rather than probability, because a strong lead saturates near P=1.

INPUTS  esm2_dora_adapter/ (fine-tuned ESM-2/DoRA adapter from phase 3.3)
OUTPUTS  phase5_6_alanine_scan.csv (one row per position: wt, P_mut, delta_logodds)
REQUIREMENTS  pip install torch transformers peft pandas
"""

# Imports
import os

# set these before the first from_pretrained call, since transformers reads them when it
# resolves a model or tokenizer. the adapter and its ESM-2 base are already cached, so
# offline mode keeps this off the hub entirely
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

MODEL_ID = "facebook/esm2_t33_650M_UR50D"
ADAPTER = "models/esm2_dora_adapter"
OUT_CSV = "results/phase5_6_alanine_scan.csv"

DEVICE = os.environ.get("DEVICE", "cpu")
MAX_LEN = 64 # the phase 3.3 cap, so nothing is tokenised differently here

PEPTIDE = "FVAPFPEVF" # the top Tier-1 lead, the section 3.5 case study


def load_model():
    """ESM-2 with a two-class head, carrying the phase 3.3 DoRA fine-tune."""
    # the adapter stores the trained classifier head as well, so this is the phase 3 model
    # rather than the base one with a fresh head on top
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    return PeftModel.from_pretrained(base, ADAPTER).eval().to(DEVICE) # load my adapter and put it in eval mode


def logodds(seqs, tok, model):
    """ADP log-odds (logit_ADP - logit_nonADP) per sequence, plus P(ADP) for reference.
    Log-odds is the importance signal. It isn't capped at 1 like the probability, so it
    still moves for a lead the model is already sure about."""
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True,
              max_length=MAX_LEN).to(DEVICE)
    with torch.inference_mode():
        lg = model(**enc).logits.float() # two raw scores per sequence: [non-ADP, ADP]
        lo = lg[:, 1] - lg[:, 0] # ADP minus non-ADP = the log-odds
        # with two classes the softmax probability is the sigmoid of that same difference,
        # so P(ADP) comes off the log-odds rather than a second pass over the logits
        p = torch.sigmoid(lo)
    return lo.cpu().numpy(), p.cpu().numpy()


def scan(seq, tok, model):
    """Wild-type plus every single-alanine mutant, as one row per position."""
    # wild-type first, then one mutant per position with that residue swapped for alanine
    variants = [seq] + [seq[:i] + "A" + seq[i + 1:] for i in range(len(seq))]
    lo, p = logodds(variants, tok, model)
    lo_wt, p_wt = lo[0], p[0] # first entry is the wild-type

    # per residue: how far the ADP log-odds falls when it's mutated to alanine.
    # a big drop means the model leans heavily on that residue for the ADP call
    return pd.DataFrame([{"peptide": seq, "position": i + 1, "wt": seq[i],
                          "P_mut": round(float(p[1 + i]), 4),
                          "wt_P_adp": round(float(p_wt), 4),
                          "wt_logodds": round(float(lo_wt), 3),
                          "delta_logodds": round(float(lo_wt - lo[1 + i]), 3),
                          "already_ala": seq[i] == "A"} # A to A is a no-op, flag it
                         for i in range(len(seq))])


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID) # tokenizer for the base ESM-2
    df = scan(PEPTIDE, tok, load_model())

    os.makedirs("results", exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    top = df.loc[df.delta_logodds.idxmax()]
    print(f"{PEPTIDE}: wild-type log-odds {df.wt_logodds[0]:.2f}, largest drop "
          f"{top.wt}{top.position} at {top.delta_logodds:.2f} -> {OUT_CSV}")


if __name__ == "__main__":
    main()