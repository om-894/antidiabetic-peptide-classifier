
"""
Phase 3 tuning: Optuna (Bayesian/TPE) with nested cross-validation for the three
local base learners (XGBoost, RF, 1D-CNN) on the fused vector. ESM-2 is not tuned
(full nested tuning of a 650M model is expensive and low-ROI; it keeps its DoRA config).

Nested CV: outer folds give an unbiased OOF; an inner CV inside each outer-training
split picks the hyperparameters without seeing that fold, so the OOF isn't optimistic
about the tuning. The outer folds are the same StratifiedKFold(5, seed=42) as the other
learners, so the tuned OOF/test stay row-aligned for the meta-learner.

Per model: outer 5-fold -> tuned *_oof; final tune on all train -> tuned *_test + model.
Outputs (new files; untuned baselines kept for comparison):
  base_tree_predictions_tuned.npz, base_cnn_predictions_tuned.npz,
  base_{xgb,rf}_tuned.joblib, base_cnn_tuned.pt, optuna_best_params.json

ENV VARS  MODELS=xgb,rf,cnn (which to tune) | TREE_TRIALS=40 | CNN_TRIALS=20 |
          SMOKE=1 (2 trials, 1 fold — fast path check)
"""

# Imports
import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)


# define constants and env vars
SEED, OUTER_FOLDS, INNER_FOLDS = 42, 5, 3
SMOKE = os.environ.get("SMOKE", "0") == "1"
MODELS = os.environ.get("MODELS", "xgb,rf,cnn").split(",")
TREE_TRIALS = 2 if SMOKE else int(os.environ.get("TREE_TRIALS", "40"))
CNN_TRIALS = 2 if SMOKE else int(os.environ.get("CNN_TRIALS", "20"))
N_OUTER = 1 if SMOKE else OUTER_FOLDS

_T = {}                                  # lazy torch handle cache


# Lazy torch loader: imports torch only the first time this is called (i.e. only
# when the CNN is actually being tuned). _T is a module-level cache dict, empty
# at first. This is what avoids the macOS torch+xgboost libomp segfault — in a
# trees-only process this is never called, so torch is never imported next to xgboost.
def torch_ctx():
    if not _T:                                   # cache empty -> first call: do the imports once
        import torch
        import torch.nn as nn
        _T["torch"], _T["nn"] = torch, nn        # stash the modules in the cache
        # pick the fastest available backend
        _T["device"] = ("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")
    return _T


def load_fused():
    # Load the fused vectors + labels + split, return the train/test slices.
    d = np.load("fusion_vectors.npz", allow_pickle=True)
    X, y, split = d["X"].astype(np.float32), d["label"].astype(int), d["split"]
    tr, te = split == "train", split == "test"   # boolean masks for the two splits
    return X[tr], y[tr], X[te], y[te]            # -> Xtr, ytr, Xte, yte


# --------------------------------------------------------------------------- #
# TREES (XGBoost / Random Forest)
# --------------------------------------------------------------------------- #

# Each *_space defines the search space: trial.suggest_* asks Optuna to propose a
# value in the given range for this trial. log=True samples on a log scale (for
# params spanning orders of magnitude, e.g. learning rate / regularisation).
def xgb_space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),      # number of boosting trees
        max_depth=trial.suggest_int("max_depth", 3, 8),                          # tree depth (complexity)
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True), # step size per tree
        subsample=trial.suggest_float("subsample", 0.6, 1.0),                    # rows sampled per tree
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),      # features sampled per tree
        min_child_weight=trial.suggest_int("min_child_weight", 1, 8),            # regularisation
        gamma=trial.suggest_float("gamma", 0.0, 5.0),                            # min loss gain to make a split
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))      # L2 regularisation


def make_xgb(p):
    from xgboost import XGBClassifier 
    return XGBClassifier(**p, eval_metric="logloss", n_jobs=-1, random_state=SEED)


def rf_space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),      # number of trees
        max_depth=trial.suggest_categorical("max_depth", [None, 10, 20, 30]),    # None = grow fully
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),  # features per split
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),            # regularisation
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10))


def make_rf(p):
    return RandomForestClassifier(**p, n_jobs=-1, random_state=SEED)


def tune_tree(make, space, X, y, n_trials):
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED)   # inner CV scores each trial

    def objective(trial):
        # Build a model from the trial's suggested params; score = mean inner-CV AUC.
        model = make(space(trial))
        return cross_val_score(model, X, y, cv=inner, scoring="roc_auc",
                               n_jobs=1).mean()

    # TPE = Bayesian search (learns promising regions); seeded for reproducibility.
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)   # run n_trials of search
    return study.best_params                       # the best hyperparameters found


def nested_tree(name, make, space, Xtr, ytr, Xte, n_trials):
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(ytr), dtype=np.float32)
    # --- outer loop -> tuned out-of-fold predictions ---
    for k, (train_ind, test_ind) in enumerate(outer.split(Xtr, ytr)):
        if k >= N_OUTER:                           # N_OUTER < 5 only in smoke test
            break
        bp = tune_tree(make, space, Xtr[train_ind], ytr[train_ind], n_trials)  # tune on this fold's train (inner CV)
        m = make(bp).fit(Xtr[train_ind], ytr[train_ind])                       # fit with the chosen params
        oof[test_ind] = m.predict_proba(Xtr[test_ind])[:, 1]                 # predict the held-out outer fold
        print(f"  [{name}] outer fold {k + 1}/{N_OUTER} tuned")
    # --- final: tune on all train, fit, predict the test set ---
    best = tune_tree(make, space, Xtr, ytr, n_trials)
    final = make(best).fit(Xtr, ytr)
    test = final.predict_proba(Xte)[:, 1].astype(np.float32)
    return oof, test, best, final                  # OOF + test preds, best params, fitted model


# --------------------------------------------------------------------------- #
# 1D-CNN (tunable version of the notebook's FusionCNN)
# --------------------------------------------------------------------------- #
def build_cnn(n_feat, ch1=32, ch2=64, k1=7, p=0.4):
    """Tunable FusionCNN (same structure as phase3.2). Built so
    torch is only imported when the CNN is actually being tuned."""
    nn = torch_ctx()["nn"]                        # lazy torch (see torch_ctx)

    # Same architecture as the notebook CNN, but ch1/ch2/k1/p are now arguments
    # so Optuna can tune the channel sizes, kernel size and dropout.
    class FusionCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, ch1, k1, padding=k1 // 2), nn.BatchNorm1d(ch1), nn.ReLU(),
                nn.MaxPool1d(2), nn.Dropout(p),
                nn.Conv1d(ch1, ch2, 5, padding=2), nn.BatchNorm1d(ch2), nn.ReLU(),
                nn.MaxPool1d(2), nn.Dropout(p),
                nn.AdaptiveAvgPool1d(1))
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(ch2, 32), nn.ReLU(),
                                      nn.Dropout(p), nn.Linear(32, 1))

        def forward(self, x):
            return self.head(self.conv(x)).squeeze(1)

    return FusionCNN()


def cnn_space(trial):
    # CNN search space: architecture (channels, kernel, dropout) + optimisation (lr, wd, batch).
    return dict(
        ch1=trial.suggest_categorical("ch1", [16, 32, 64]),       # 1st conv channels
        ch2=trial.suggest_categorical("ch2", [32, 64, 128]),      # 2nd conv channels
        k1=trial.suggest_categorical("k1", [5, 7, 9]),            # 1st conv kernel size
        p=trial.suggest_float("p", 0.2, 0.6),                     # dropout rate
        lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),       # learning rate
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),  # L2
        batch=trial.suggest_categorical("batch", [32, 64, 128]))  # batch size


def fit_cnn(Xtr_raw, ytr, params, seed=SEED, max_epochs=200, patience=15):
    """Scale on the training data given, then train FusionCNN with early stopping
    on an internal 15% validation split. Returns (model, scaler)."""
    t = torch_ctx()
    torch, nn, device = t["torch"], t["nn"], t["device"]
    torch.manual_seed(seed)
    # CNN is scale-sensitive: fit a scaler on this training data and apply it.
    scaler = StandardScaler().fit(Xtr_raw)
    Xs = scaler.transform(Xtr_raw).astype(np.float32)
    # internal 15% val split for early stopping
    xtr, xval, ytr_, yval = train_test_split(Xs, ytr, test_size=0.15,
                                             stratify=ytr, random_state=seed)
    to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)
    xtr_t, xval_t = to(xtr).unsqueeze(1), to(xval).unsqueeze(1) # add channel dim
    ytr_t, yval_t = to(ytr_), to(yval)

    # build the model from the tuned hyperparameters
    model = build_cnn(Xtr_raw.shape[1], params["ch1"], params["ch2"],
                      params["k1"], params["p"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"],
                           weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    bs = params["batch"]

    best, best_state, wait = np.inf, None, 0 # early-stopping logging
    for _ in range(max_epochs):
        model.train()
        for i in range(0, len(xtr_t), bs): # mini-batches of the tuned size
            opt.zero_grad()
            loss_fn(model(xtr_t[i:i + bs]), ytr_t[i:i + bs]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(xval_t), yval_t).item() # val loss drives early stopping
        if vl < best - 1e-4:
            best, best_state, wait = vl, {k: v.cpu().clone()
                                          for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state) # restore the best epoch
    return model, scaler # return the scaler also, needed to predict


def cnn_predict(model, scaler, Xraw):
    # Scale with the model's own fitted scaler, then forward -> sigmoid -> probabilities.
    t = torch_ctx()
    torch, device = t["torch"], t["device"]
    model.eval()
    with torch.no_grad():

        # convert the raw input to a torch tensor on the correct device, scaled and with a channel dimension
        x = torch.tensor(scaler.transform(Xraw).astype(np.float32),
                         device=device).unsqueeze(1)
        
        # return the predicted probabilities as a numpy array of float32
        return torch.sigmoid(model(x)).cpu().numpy().astype(np.float32)


def tune_cnn(Xtr_raw, ytr, n_trials, max_epochs):
    """Inner single stratified holdout AUC (keeps nested-CNN tractable)."""
    # CNN fits are slow, so the inner loop uses one 20% holdout (not k-fold CV)
    # to score each trial -> far fewer fits than inner k-fold would need.
    xin, xval, yin, yval = train_test_split(Xtr_raw, ytr, test_size=0.2,
                                            stratify=ytr, random_state=SEED)

    def objective(trial):
        # fit on the inner-train with this trial's params, score AUC on the holdout
        m, sc = fit_cnn(xin, yin, cnn_space(trial), seed=SEED,
                        max_epochs=max_epochs, patience=10)
        return roc_auc_score(yval, cnn_predict(m, sc, xval))

    # create_study with TPE (Bayesian) sampler, seeded for reproducibility
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def nested_cnn(Xtr, ytr, Xte, n_trials):
    # Shorter epochs during tuning (cheap search); full epochs for the real fits.
    tune_epochs = 60 if not SMOKE else 5
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(ytr), dtype=np.float32)
    # outer loop -> tuned OOF
    for k, (train_ind, test_ind) in enumerate(outer.split(Xtr, ytr)):
        if k >= N_OUTER:
            break
        bp = tune_cnn(Xtr[train_ind], ytr[train_ind], n_trials, tune_epochs) # tune on this fold's train
        m, sc = fit_cnn(Xtr[train_ind], ytr[train_ind], bp, seed=SEED + k,
                        max_epochs=10 if SMOKE else 200) # fit best params (full epochs)
        oof[test_ind] = cnn_predict(m, sc, Xtr[test_ind]) # predict the held-out fold
        print(f"  [cnn] outer fold {k + 1}/{N_OUTER} tuned")
    # final: tune on all train, fit, predict test
    best = tune_cnn(Xtr, ytr, n_trials, tune_epochs)
    final, sc = fit_cnn(Xtr, ytr, best, seed=SEED, max_epochs=10 if SMOKE else 200)
    test = cnn_predict(final, sc, Xte)
    return oof, test, best, (final, sc) # final model returned with its scaler


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    print(f"models {MODELS} | tree_trials {TREE_TRIALS} | "
          f"cnn_trials {CNN_TRIALS} | smoke {SMOKE}")
    os.makedirs("models", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)
    Xtr, ytr, Xte, yte = load_fused()            # the fused vectors + labels
    print(f"fused: train {Xtr.shape}, test {Xte.shape}\n")

    # models (env var) decides which learners this process tunes. Trees and CNN
    # are run as separate processes (xgb,rf then cnn) to dodge the libomp segfault,
    # so any given run fills only some of these dicts.
    best_params, tree_preds, cnn_preds = {}, {}, {}

    if "xgb" in MODELS:
        print("tuning XGBoost ...")
        oof, test, bp, model = nested_tree("xgb", make_xgb, xgb_space,
                                           Xtr, ytr, Xte, TREE_TRIALS)
        tree_preds.update(xgb_oof=oof, xgb_test=test)   # OOF + test preds for the stack
        best_params["xgb"] = bp                          # chosen hyperparameters
        joblib.dump(model, "models/base_xgb_tuned.joblib")   # the tuned model
        print(f"  XGB tuned: OOF {roc_auc_score(ytr, oof):.3f} "
              f"(fold0 only if smoke) | TEST {roc_auc_score(yte, test):.3f}\n")

    if "rf" in MODELS:
        print("tuning Random Forest ...")
        oof, test, bp, model = nested_tree("rf", make_rf, rf_space,
                                           Xtr, ytr, Xte, TREE_TRIALS)
        tree_preds.update(rf_oof=oof, rf_test=test)
        best_params["rf"] = bp
        joblib.dump(model, "models/base_rf_tuned.joblib")
        print(f"  RF tuned: OOF {roc_auc_score(ytr, oof):.3f} | "
              f"TEST {roc_auc_score(yte, test):.3f}\n")

    if "cnn" in MODELS:
        print("tuning 1D-CNN ...")
        oof, test, bp, (model, sc) = nested_cnn(Xtr, ytr, Xte, CNN_TRIALS)
        cnn_preds.update(cnn_oof=oof, cnn_test=test)
        best_params["cnn"] = bp
        torch_ctx()["torch"].save(model.state_dict(), "models/base_cnn_tuned.pt")  # lazy torch to save
        print(f"  CNN tuned: OOF {roc_auc_score(ytr, oof):.3f} | "
              f"TEST {roc_auc_score(yte, test):.3f}\n")

    # Save predictions only for the models this process actually ran.
    if tree_preds:
        np.savez_compressed("predictions/base_tree_predictions_tuned.npz",
                            y_train=ytr, y_test=yte, **tree_preds)
        print("saved predictions/base_tree_predictions_tuned.npz")
    if cnn_preds:
        np.savez_compressed("predictions/base_cnn_predictions_tuned.npz",
                            y_train=ytr, y_test=yte, **cnn_preds)
        print("saved predictions/base_cnn_predictions_tuned.npz")

    # The best-params JSON is shared by both processes, so MERGE (load existing,
    # update, write back) instead of overwriting — otherwise the cnn run would
    # wipe the trees' entries (and vice versa).
    merged = {}
    if os.path.exists("optuna_best_params.json"):
        with open("optuna_best_params.json") as fh:
            merged = json.load(fh)
    merged.update(best_params)
    with open("optuna_best_params.json", "w") as fh:
        json.dump(merged, fh, indent=2, default=str)
    print("saved optuna_best_params.json")

    # Quick tuned-vs-untuned test-AUC comparison (skipped if the untuned files
    # aren't present — hence the try/except).
    print("\ntuned vs untuned (test AUC):")
    try:
        ut = np.load("predictions/base_tree_predictions.npz")
        uc = np.load("predictions/base_cnn_predictions.npz")
        base = {"xgb": ut["xgb_test"], "rf": ut["rf_test"], "cnn": uc["cnn_test"]}
        for m in MODELS:
            if m in ("xgb", "rf") and tree_preds:
                t = roc_auc_score(yte, tree_preds[f"{m}_test"])
            elif m == "cnn" and cnn_preds:
                t = roc_auc_score(yte, cnn_preds["cnn_test"])
            else:
                continue
            print(f"  {m:4s} {roc_auc_score(yte, base[m]):.3f} -> {t:.3f} "
                  f"({t - roc_auc_score(yte, base[m]):+.3f})")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
