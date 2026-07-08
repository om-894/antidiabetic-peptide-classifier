
"""
Phase 3.3: ESM-2/DoRA fine-tune — base learner 4 of 4. Fine-tunes ESM-2 (650M) on
raw peptide sequences using DoRA (Weight-Decomposed Low-Rank Adaptation),
replicating BertADP with ESM-2 in place of ProtBert. Runs on a Viking GPU node.

Emits probabilities in the same stacking convention as the other base learners:
  esm_oof   out-of-fold probabilities for the 1754 train rows
  esm_test  probabilities for the 178 test rows
Row alignment with rf/xgb/cnn is guaranteed by reusing their OOF scheme:
StratifiedKFold(5, shuffle=True, random_state=42) over dataset_split.csv train
rows, plus a final all-train fit to predict the test rows.

OUTPUTS  esm2_dora_predictions.npz  (esm_oof, esm_test, y_train, y_test, sequences)
         esm2_dora_adapter/         DoRA adapter weights (reused in Phase 4)

REQUIREMENTS  pip install torch transformers peft accelerate pandas numpy scikit-learn
              Model must be pre-cached on a login node (GPU nodes have no internet;
              set TRANSFORMERS_OFFLINE=1).

ENV VARS  SMOKE_TEST=1   run 1 fold x 1 epoch on 64 rows to validate the path
"""

# Imports
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from peft import LoraConfig, TaskType, get_peft_model

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

# Input/output paths come from env vars (with defaults) so the ablation can point
# the same script at a different dataset/outputs without editing the code.
DATA = os.environ.get("DATA", "data/dataset_split.csv")
MODEL_ID = "facebook/esm2_t33_650M_UR50D"       # ESM-2, 650M params
OUT_NPZ = os.environ.get("OUT_NPZ", "predictions/esm2_dora_predictions.npz")
OUT_ADAPTER = os.environ.get("OUT_ADAPTER", "models/esm2_dora_adapter")

# Ablation mode: skip the 5-fold OOF (only the test prediction is needed), so the
# Basith-negative run is a single fine-tune. SKIP_OOF=1 enables it.
SKIP_OOF = os.environ.get("SKIP_OOF", "0") == "1"

# reproducibility and OOF scheme (must match the trees/CNN for stacking)
SEED = int(os.environ.get("SEED", "42"))
FOLDS = 5

# tokenisation / training
MAX_LEN = 64           # max tokens; dataset max is 41 residues (+<cls>/<eos>), so 64 is safe
BATCH_SIZE = 16
MAX_EPOCHS = 20
PATIENCE = 4            # early stopping: stop after 4 epochs of no val-loss improvement
LR = 2e-4         # adapter learning rate (higher than typical full fine-tuning)
WEIGHT_DECAY = 0.01       # L2 regularisation
WARMUP_FRAC = 0.10       # ramp LR up over the first 10% of steps (stabilises early training)
INTERNAL_VAL_FRAC = 0.12  # carved from each fold's train portion for early stopping

# DoRA / LoRA adapter config
DORA_R = 16         # rank of the low-rank update (adapter capacity)
DORA_ALPHA = 32         # adapter scaling (effective strength ~ alpha / r)
DORA_DROPOUT = 0.05

# q/k/v = query, key, value — the three learned linear projections inside a transformer's self-attention mechanism.
TARGET_MODULES = ["query", "key", "value"]   # attach adapters to the attention q/k/v projections

# SMOKE_TEST: quick tiny pass (64 rows, 1 fold, 1 epoch) to confirm the code runs end-to-end before the real ~6-fine-tune GPU job
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = device.type == "cuda" and torch.cuda.is_bf16_supported()   # mixed precision on GPU

def set_seed(seed):
    # Seed every RNG (Python, NumPy, PyTorch CPU + CUDA) for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# MODEL
# --------------------------------------------------------------------------- #
def build_model():
    """Fresh ESM-2 classifier wrapped with a DoRA adapter (head fully trained)."""
    # Load ESM-2 with a new 2-class classification head (ADP vs non-ADP) on top.
    # The head is randomly initialised. the backbone carries the pretrained weights.
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2)
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,      # sequence-classification task
        use_dora=True,                   # use DoRA, not plain LoRA
        r=DORA_R, lora_alpha=DORA_ALPHA, lora_dropout=DORA_DROPOUT,
        target_modules=TARGET_MODULES,   # adapters on the attention q/k/v projections
        modules_to_save=["classifier"],  # train the new head fully (it must learn from scratch)
    )
    # get_peft_model freezes the 650M backbone and leaves ONLY the DoRA adapters +
    # the classifier head trainable -> parameter-efficient fine-tuning.
    return get_peft_model(base, cfg).to(device)


# --------------------------------------------------------------------------- #
# DATA HELPERS
# --------------------------------------------------------------------------- #
def make_batches(seqs, labels, batch_size, shuffle, rng=None):
    """Yield (list_of_seqs, label_tensor_or_None) batches over index order."""
    idx = list(range(len(seqs)))
    if shuffle:
        (rng or random).shuffle(idx)            # shuffle indicies (rng -> reproducible); seqs<->labels stay aligned
    for i in range(0, len(idx), batch_size):    # walk through in chunks of batch_size
        chunk = idx[i:i + batch_size]
        bseqs = [seqs[j] for j in chunk]        # the raw sequence strings for this batch
        # labels=None when predicting (no targets needed); else a long tensor for the loss.
        blab = (torch.tensor([labels[j] for j in chunk], dtype=torch.long)
                if labels is not None else None)
        yield bseqs, blab                       # generator hands back one batch at a time

def encode(tok, bseqs):
    # Tokenise a batch of raw sequences into model-ready tensors on the device
    # Pad to the longest in the batch, truncate at MAX_LEN, return PyTorch tensors.
    return tok(bseqs, padding=True, truncation=True, max_length=MAX_LEN,
               return_tensors="pt").to(device)


# --------------------------------------------------------------------------- #
# TRAIN / PREDICT
# --------------------------------------------------------------------------- #
def train_model(tok, tr_seqs, tr_lab, va_seqs, va_lab, fold_seed, max_epochs):
    set_seed(fold_seed)                          # reproducible per fold
    model = build_model()                        # fresh frozen-backbone + DoRA model
    
    # Optimise only the trainable params (DoRA adapters + classifier head); the
    # frozen 650M backbone has no gradients, so it is excluded.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Linear LR schedule with warmup: ramp up over the first WARMUP_FRAC of steps,
    # then decay -> stabilises early adapter training.
    steps_per_epoch = max(1, (len(tr_seqs) + BATCH_SIZE - 1) // BATCH_SIZE)
    total_steps = steps_per_epoch * max_epochs
    sched = get_linear_schedule_with_warmup(
        opt, int(total_steps * WARMUP_FRAC), total_steps)
    loss_fn = nn.CrossEntropyLoss()              # 2-class logits (ADP vs non-ADP)
    rng = random.Random(fold_seed)               # per-fold batch shuffling

    best_val, best_state, wait = np.inf, None, 0   # early-stopping logging
    for epoch in range(max_epochs):
        model.train()
        for bseqs, blab in make_batches(tr_seqs, tr_lab, BATCH_SIZE, True, rng):
            opt.zero_grad()
            enc = encode(tok, bseqs)
            
            # bf16 autocast = mixed precision (faster, less GPU memory on the A40).
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16, enabled=USE_BF16):
                
                # model(**enc).logits because AutoModelForSequenceClassification returns a dict with logits, loss, etc.
                logits = model(**enc).logits
                loss = loss_fn(logits.float(), blab.to(device))   # logits->fp32 for a stable loss
            loss.backward()
            opt.step()
            sched.step()

        # internal validation loss -> drives early stopping
        model.eval()
        vloss, nb = 0.0, 0
        with torch.no_grad():
            for bseqs, blab in make_batches(va_seqs, va_lab, BATCH_SIZE, False):
                enc = encode(tok, bseqs)
                with torch.autocast(device_type=device.type,
                                    dtype=torch.bfloat16, enabled=USE_BF16):
                    logits = model(**enc).logits
                vloss += loss_fn(logits.float(), blab.to(device)).item()
                nb += 1
        vloss /= max(1, nb)                       # mean validation loss this epoch
        if vloss < best_val - 1e-4:               # improved -> snapshot best weights
            best_val = vloss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1                             # no improvement this epoch
            if wait >= PATIENCE:                  # stalled for patience epochs = stop early
                break
    if best_state is not None:
        model.load_state_dict(best_state)         # restore the best epoch, not the last
    return model


def predict_proba(model, tok, seqs):
    model.eval()
    out = []
    with torch.no_grad():
        for bseqs, _ in make_batches(seqs, None, BATCH_SIZE, False):   # labels=None: prediction
            enc = encode(tok, bseqs)
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16, enabled=USE_BF16):
                logits = model(**enc).logits
            
            # softmax over the 2 logits, take column 1 = P(ADP).
            out.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()) # back to CPU/numpy for storage
    return np.concatenate(out).astype(np.float32)


def free(model):
    # Release the model and clear the CUDA cache so GPU memory doesn't accumulate
    # across the 6 sequential fine-tunes (5 OOF folds + 1 full) and cause an out of memory error.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    set_seed(SEED)
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    print(f"device: {device} | bf16: {USE_BF16} | smoke_test: {SMOKE_TEST}")

    # Load the split and pull out raw sequences and integer labels for each side.
    df = pd.read_csv(DATA)
    tr = df[df.Split == "train"].reset_index(drop=True)   # reset_index -> clean 0..N positions
    te = df[df.Split == "test"].reset_index(drop=True)
    tr_seqs = tr["Sequence"].astype(str).tolist()
    te_seqs = te["Sequence"].astype(str).tolist()
    ytr = tr["Label"].to_numpy(dtype=int)
    yte = te["Label"].to_numpy(dtype=int)
    print(f"train {len(tr_seqs)} | test {len(te_seqs)}")


    # SMOKE_TEST: quick tiny pass (64 rows, 1 fold, 1 epoch) to confirm the code runs end-to-end before the real ~6-fine-tune GPU job
    if SMOKE_TEST:
        # Tiny balanced subset, 1 fold, 1 epoch — validates the whole path cheaply
        # before committing to the real (6 fine-tune) run.
        keep = np.r_[np.where(ytr == 1)[0][:32], np.where(ytr == 0)[0][:32]] # 32 ADP + 32 non-ADP
        tr_seqs = [tr_seqs[i] for i in keep]; ytr = ytr[keep]
        tkeep = np.r_[np.where(yte == 1)[0][:16], np.where(yte == 0)[0][:16]] # 16 ADP + 16 non-ADP
        te_seqs = [te_seqs[i] for i in tkeep]; yte = yte[tkeep]
        n_folds, max_epochs = 1, 1 # smoke test: 1 fold x 1 epoch on 64 rows
    else:
        n_folds, max_epochs = FOLDS, MAX_EPOCHS

    print(f"loading tokenizer {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)


    # OOF predictions (identical folds to trees/CNN)
    # Each train row gets a probability from a model trained only on the other
    # folds -> honest out-of-sample input for the stack. Skipped in ablation mode.
    esm_oof = np.zeros(len(tr_seqs), dtype=np.float32)
    if SKIP_OOF:
        print("SKIP_OOF=1: skipping 5-fold OOF, single train->test fit only")
    else:
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        folds = list(skf.split(tr_seqs, ytr))[:n_folds]
        for k, (tri, vali) in enumerate(folds):
            # carve an internal val from this fold's train portion for early
            # stopping, so the held-out fold (validation) stays untouched -> no leakage.
            sub_tr, sub_va = train_test_split(
                tri, test_size=INTERNAL_VAL_FRAC, stratify=ytr[tri], # tri = train indicies for this fold
                random_state=SEED + k)
            
            # train a fresh model on this fold's train portion, with its own early-stopping val split
            model = train_model(
                tok,
                [tr_seqs[i] for i in sub_tr], ytr[sub_tr],
                [tr_seqs[i] for i in sub_va], ytr[sub_va],
                fold_seed=SEED + k, max_epochs=max_epochs)
            
            # score the held-out fold (vali) with this fold's model, store in esm_oof
            esm_oof[vali] = predict_proba(model, tok, [tr_seqs[i] for i in vali])  # score held-out fold
            free(model)                          # release GPU memory before the next fold
            print(f"fold {k + 1}/{n_folds} done | "
                  f"fold AUC {roc_auc_score(ytr[vali], esm_oof[vali]):.3f}")

        # sanity check: all train rows got a probability from some fold's model
        if not SMOKE_TEST:
            print(f"OOF train AUC {roc_auc_score(ytr, esm_oof):.3f}  (optimistic; "
                  f"matches trees/CNN scheme)")

    # final fit on ALL train -> test predictions + saved adapter #
    # A separate model trained on the full training set (with its own early-stopping
    # val split) scores the held-out test set and is saved as the deployment adapter.
    all_idx = np.arange(len(tr_seqs))
    full_tr, full_va = train_test_split(
        all_idx, test_size=INTERNAL_VAL_FRAC, stratify=ytr, random_state=SEED)
    
    # train a fresh model on the full training set, with its own early-stopping val split
    full = train_model(
        tok,
        [tr_seqs[i] for i in full_tr], ytr[full_tr],
        [tr_seqs[i] for i in full_va], ytr[full_va],
        fold_seed=SEED, max_epochs=max_epochs)
    esm_test = predict_proba(full, tok, te_seqs)
    
    # save the final adapter (trained on all data) to reuse on new peptides in Phase 4
    full.save_pretrained(OUT_ADAPTER)
    print(f"saved adapter -> {OUT_ADAPTER}/")
    free(full)

    print(f"TEST  AUC {roc_auc_score(yte, esm_test):.3f} | "
          f"ACC {accuracy_score(yte, esm_test > 0.5):.3f} | "
          f"MCC {matthews_corrcoef(yte, esm_test > 0.5):.3f}")

    # Save in the stacking convention: OOF (train) + test probabilities, labels,
    # and the sequences (so row alignment can be verified downstream).
    np.savez_compressed(
        OUT_NPZ,
        esm_oof=esm_oof, esm_test=esm_test,
        y_train=ytr.astype(np.int64), y_test=yte.astype(np.int64),
        sequences=np.array(tr_seqs + te_seqs, dtype=object),
    )
    print(f"saved -> {OUT_NPZ}")


if __name__ == "__main__":
    main()
