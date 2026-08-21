
"""
Phase 3.3: ESM-2/DoRA fine-tune, base learner 4 of 4. Fine-tunes ESM-2 (650M) on
raw peptide sequences using DoRA (Weight-Decomposed Low-Rank Adaptation), the same
approach BertADP (Xie et al., 2023) takes over ProtBert. Runs on a Viking GPU node.

Emits probabilities in the same stacking convention as the other base learners.
  esm_oof   out-of-fold probabilities for the 1754 train rows
  esm_test  probabilities for the 178 test rows

Row alignment with rf/xgb/cnn comes from reusing their OOF scheme, which is
StratifiedKFold(5, shuffle=True, random_state=42) over dataset_split.csv train
rows, plus a final all-train fit to predict the test rows.

INPUTS        dataset_split.csv (1932 rows with Sequence, Label, Split)
OUTPUTS       esm2_dora_predictions.npz (esm_oof, esm_test, y_train, y_test, sequences)
              esm2_dora_adapter/ DoRA adapter weights, reloaded in phase 5.2 and 5.6
REQUIREMENTS  pip install torch transformers peft accelerate pandas numpy scikit-learn
              Model must be pre-cached on a login node (GPU nodes have no internet;
              set TRANSFORMERS_OFFLINE=1).
ENV VARS      TEST_RUN=1 runs 1 fold x 1 epoch on 64 rows to validate the path
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

# paths come from env vars (with defaults) so the ablation can point the same
# script at a different dataset and outputs without editing the code.
DATA = os.environ.get("DATA", "data/dataset_split.csv")
OUT_NPZ = os.environ.get("OUT_NPZ", "predictions/esm2_dora_predictions.npz")
OUT_ADAPTER = os.environ.get("OUT_ADAPTER", "models/esm2_dora_adapter")

MODEL_ID = "facebook/esm2_t33_650M_UR50D" # ESM-2, 650M params

# ablation mode, which skips the 5-fold OOF since only the test prediction is
# needed, so the Basith-negative run is a single fine-tune
SKIP_OOF = os.environ.get("SKIP_OOF", "0") == "1"

# quick tiny pass (64 rows, 1 fold, 1 epoch) to confirm the code runs end to end
# before the real job, which is 5 fold fits plus a final all-train fit
TEST_RUN = os.environ.get("TEST_RUN", "0") == "1"

# reproducibility and OOF scheme (must match the trees/CNN for stacking)
SEED = int(os.environ.get("SEED", "42"))
FOLDS = 5

# tokenisation / training
MAX_LEN = 64 # max tokens; dataset max is 41 residues plus <cls>/<eos>, so 64 is safe
BATCH_SIZE = 16 # 32 hit OOM on the A40 at MAX_LEN 64
MAX_EPOCHS = 20
PATIENCE = 4 # stop after 4 epochs of no val-loss improvement
LR = 2e-4
WEIGHT_DECAY = 0.01 # L2 regularisation
WARMUP_FRAC = 0.10
INTERNAL_VAL_FRAC = 0.12 # the 12% in section 2.3

# DoRA adapter config
DORA_R = 16 # rank and alpha follow the DoRA paper's defaults
DORA_ALPHA = 32 # adapter scaling, alpha / r gives the effective strength
DORA_DROPOUT = 0.05
TARGET_MODULES = ["query", "key", "value"] # the attention query/key/value projections

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() # mixed precision on GPU


def set_seed(seed):
    # seed every RNG (Python, NumPy, PyTorch CPU + CUDA) for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# MODEL
# --------------------------------------------------------------------------- #
def build_model():
    """Fresh ESM-2 classifier wrapped with a DoRA adapter (head fully trained)."""
    # a new 2-class head (ADP vs non-ADP) on top of the pretrained backbone. the
    # head starts randomly initialised while the backbone carries the ESM-2 weights.
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)

    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        use_dora=True, # DoRA rather than plain LoRA
        r=DORA_R, lora_alpha=DORA_ALPHA, lora_dropout=DORA_DROPOUT,
        target_modules=TARGET_MODULES,
        modules_to_save=["classifier"], # train the new head fully, it has nothing pretrained to keep
    )

    # get_peft_model freezes the 650M backbone and leaves only the DoRA adapters
    # plus the classifier head trainable -> parameter-efficient fine-tuning.
    return get_peft_model(base, cfg).to(DEVICE)


# --------------------------------------------------------------------------- #
# DATA HELPERS
# --------------------------------------------------------------------------- #
def make_batches(seqs, labels, batch_size, shuffle, rng=None):
    """Yield (list_of_seqs, label_tensor_or_None) batches over index order."""
    # shuffle the indices rather than the data, so seqs and labels stay paired.
    # passing an rng makes the order reproducible, otherwise the module-level
    # random is used.
    idx = list(range(len(seqs)))
    if shuffle:
        (rng or random).shuffle(idx)

    for i in range(0, len(idx), batch_size):
        chunk = idx[i:i + batch_size]
        bseqs = [seqs[j] for j in chunk]
        # labels=None when predicting, since no targets are needed. otherwise a
        # long tensor for CrossEntropyLoss, left on CPU for the caller to move.
        blab = (torch.tensor([labels[j] for j in chunk], dtype=torch.long)
                if labels is not None else None)
        yield bseqs, blab


def encode(tok, bseqs):
    # tokenise a batch of raw sequences into model-ready tensors on DEVICE.
    # pads to the longest in the batch and truncates at MAX_LEN, which the
    # training data never reaches at 41 residues plus <cls>/<eos>.
    return tok(bseqs, padding=True, truncation=True, max_length=MAX_LEN,
               return_tensors="pt").to(DEVICE)


# --------------------------------------------------------------------------- #
# TRAIN / PREDICT
# --------------------------------------------------------------------------- #
def train_model(tok, tr_seqs, tr_lab, va_seqs, va_lab, fold_seed, max_epochs):
    set_seed(fold_seed) # reproducible per fold
    model = build_model() # fresh frozen-backbone + DoRA model

    # optimise only the trainable params (DoRA adapters plus classifier head).
    # the frozen 650M backbone has no gradients, so it is excluded.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    # linear LR schedule, ramping up over the first WARMUP_FRAC of steps then
    # decaying, which stabilises early adapter training. total_steps assumes the
    # full max_epochs, so early stopping leaves the decay unfinished.
    steps_per_epoch = max(1, (len(tr_seqs) + BATCH_SIZE - 1) // BATCH_SIZE)
    total_steps = steps_per_epoch * max_epochs
    sched = get_linear_schedule_with_warmup(
        opt, int(total_steps * WARMUP_FRAC), total_steps)
    loss_fn = nn.CrossEntropyLoss() # 2-class logits (ADP vs non-ADP)
    rng = random.Random(fold_seed) # per-fold batch shuffling

    best_val, best_state, wait = np.inf, None, 0 # early stopping trackers
    for epoch in range(max_epochs):
        model.train()
        for bseqs, blab in make_batches(tr_seqs, tr_lab, BATCH_SIZE, True, rng):
            opt.zero_grad()
            enc = encode(tok, bseqs)
            # bf16 autocast = mixed precision, faster and lighter on the A40
            with torch.autocast(device_type=DEVICE.type,
                                dtype=torch.bfloat16, enabled=USE_BF16):
                # .logits because the model returns an object, not a bare tensor
                logits = model(**enc).logits
                loss = loss_fn(logits.float(), blab.to(DEVICE)) # fp32 for a stable loss
            loss.backward()
            opt.step()
            sched.step()

        # validation loss for this epoch, which is what early stopping watches.
        # eval() matters here, since it switches off the 0.05 adapter dropout and
        # makes the number deterministic rather than noisy
        model.eval()
        vloss, nb = 0.0, 0
        with torch.no_grad(): # no gradients needed just to score
            # shuffle=False, since a random order gains nothing outside training
            for bseqs, blab in make_batches(va_seqs, va_lab, BATCH_SIZE, False):
                enc = encode(tok, bseqs)
                # same autocast as training, so the two losses stay comparable
                with torch.autocast(device_type=DEVICE.type,
                                    dtype=torch.bfloat16, enabled=USE_BF16):
                    logits = model(**enc).logits
                vloss += loss_fn(logits.float(), blab.to(DEVICE)).item()
                nb += 1
        # mean over batches rather than over samples, so a short final batch counts
        # the same as a full one. the batching is fixed, so the bias is identical
        # every epoch and doesn't distort the comparison early stopping makes.
        vloss /= max(1, nb)

        if vloss < best_val - 1e-4: # improved, so snapshot the best weights
            best_val = vloss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1 # no improvement this epoch
            if wait >= PATIENCE: # stalled for patience epochs, stop early
                break

    if best_state is not None:
        model.load_state_dict(best_state) # restore the best epoch, not the last
    return model


def predict_proba(model, tok, seqs):
    """P(ADP) for each sequence, as a float32 array in the same order as seqs."""
    # eval() switches off the adapter dropout, so the same sequence always scores
    # the same. shuffle=False keeps batches in input order, which is what lets the
    # caller assign straight into oof[vali] without tracking indices.
    model.eval()
    out = []
    with torch.no_grad():
        for bseqs, _ in make_batches(seqs, None, BATCH_SIZE, False): # labels=None, prediction only
            enc = encode(tok, bseqs)
            with torch.autocast(device_type=DEVICE.type,
                                dtype=torch.bfloat16, enabled=USE_BF16):
                logits = model(**enc).logits
            # float() before softmax, since bf16 has too few mantissa bits for
            # probabilities. column 1 is P(ADP) because positives carry Label=1.
            out.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def free():
    # clear the CUDA cache between the 6 sequential fine-tunes (5 OOF folds plus
    # 1 full) so memory doesn't accumulate into an OOM. the caller has to del its
    # own reference first, otherwise the weights are still live and nothing frees
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    set_seed(SEED)
    print(f"device {DEVICE} | bf16 {USE_BF16} | test_run {TEST_RUN}") # prints so the job log shows the config, which is useful for debugging

    # load the split and pull out raw sequences and integer labels for each side.
    df = pd.read_csv(DATA)
    tr = df[df.Split == "train"].reset_index(drop=True) # reset_index -> clean 0..N positions
    te = df[df.Split == "test"].reset_index(drop=True)
    tr_seqs = tr["Sequence"].astype(str).tolist() # convert to str in case the CSV has any NaN or other non-string values
    te_seqs = te["Sequence"].astype(str).tolist()
    ytr = tr["Label"].to_numpy(dtype=int)
    yte = te["Label"].to_numpy(dtype=int)
    print(f"train {len(tr_seqs)} | test {len(te_seqs)}")

    if TEST_RUN:
        # tiny balanced subset so the whole path runs cheaply before committing the
        # GPU to six real fine-tunes. the balance is required rather than tidy,
        # since stratified splitting and roc_auc_score both fail on a single-class
        # set. np.where(...)[0] gives each class's row positions, np.r_ joins them
        keep = np.r_[np.where(ytr == 1)[0][:32], np.where(ytr == 0)[0][:32]] # 32 ADP + 32 non-ADP
        tr_seqs = [tr_seqs[i] for i in keep]; ytr = ytr[keep]
        tkeep = np.r_[np.where(yte == 1)[0][:16], np.where(yte == 0)[0][:16]] # 16 ADP + 16 non-ADP
        te_seqs = [te_seqs[i] for i in tkeep]; yte = yte[tkeep]
        n_folds, max_epochs = 1, 1
    else:
        n_folds, max_epochs = FOLDS, MAX_EPOCHS

    print(f"loading tokenizer {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # OOF predictions on the same folds as the trees and CNN. each train row gets
    # a probability from a model trained only on the other folds, which is what
    # makes it honest input for the stack. skipped in ablation mode, where the
    # saved esm_oof stays all-zero and should not be read
    esm_oof = np.zeros(len(tr_seqs), dtype=np.float32)
    if SKIP_OOF:
        print("SKIP_OOF=1, skipping the 5-fold OOF, single train to test fit only")
    else:
        # always build all 5 folds so the split matches the other learners, then
        # take only the first when TEST_RUN cuts n_folds to 1.
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        folds = list(skf.split(tr_seqs, ytr))[:n_folds]
        for k, (tri, vali) in enumerate(folds):
            # carve an internal val from this fold's train portion for early
            # stopping, so the held-out fold stays untouched and no leakage occurs
            sub_tr, sub_va = train_test_split(
                tri, test_size=INTERNAL_VAL_FRAC, stratify=ytr[tri], # tri = train indices for this fold
                random_state=SEED + k)

            # train the model on this fold's train portion, with its own early-stopping
            # val split. the seed is shifted by the fold number so each fold's model
            # is different, even though the data is the same. the caller must free()
            # the model reference after this returns, or the GPU memory won't clear
            model = train_model(
                tok,
                [tr_seqs[i] for i in sub_tr], ytr[sub_tr],
                [tr_seqs[i] for i in sub_va], ytr[sub_va],
                fold_seed=SEED + k, max_epochs=max_epochs)

            esm_oof[vali] = predict_proba(model, tok, [tr_seqs[i] for i in vali])
            del model # drop the reference before clearing the cache, or nothing frees
            free() # clear the GPU memory so the next fold can fit
            print(f"fold {k + 1}/{n_folds} done | "
                  f"fold AUC {roc_auc_score(ytr[vali], esm_oof[vali]):.3f}")

        if not TEST_RUN:
            assert (esm_oof > 0).all(), "some train rows never got an OOF prediction"
            print(f"OOF train AUC {roc_auc_score(ytr, esm_oof):.3f} (optimistic, "
                  f"matches the trees/CNN scheme)")

    # a separate model trained on the full training set, with its own early-stopping
    # val split, scores the held-out test set and is saved as the deployment adapter.
    all_idx = np.arange(len(tr_seqs))
    full_tr, full_va = train_test_split(
        all_idx, test_size=INTERNAL_VAL_FRAC, stratify=ytr, random_state=SEED)

    full = train_model(
        tok,
        [tr_seqs[i] for i in full_tr], ytr[full_tr],
        [tr_seqs[i] for i in full_va], ytr[full_va],
        fold_seed=SEED, max_epochs=max_epochs)
    esm_test = predict_proba(full, tok, te_seqs)

    full.save_pretrained(OUT_ADAPTER) # reloaded in phase 5.2 and 5.6
    print(f"saved adapter -> {OUT_ADAPTER}/")
    del full
    free() # clear the GPU memory so the job can exit cleanly without OOM

    print(f"TEST  AUC {roc_auc_score(yte, esm_test):.3f} | "
          f"ACC {accuracy_score(yte, esm_test > 0.5):.3f} | "
          f"MCC {matthews_corrcoef(yte, esm_test > 0.5):.3f}")

    # stacking convention, OOF for the train rows then test probabilities, with
    # labels and sequences alongside. sequences runs train first then test, so it
    # lines up with esm_oof followed by esm_test.
    np.savez_compressed(
        OUT_NPZ,
        esm_oof=esm_oof, esm_test=esm_test,
        y_train=ytr.astype(np.int64), y_test=yte.astype(np.int64),
        sequences=np.array(tr_seqs + te_seqs, dtype=object),
    )
    print(f"saved -> {OUT_NPZ}")