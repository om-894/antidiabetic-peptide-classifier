
"""
Phase 3.5: Optuna (TPE) tuning with nested cross-validation for the three local
base learners (XGBoost, RF, 1D-CNN) on the fused vector. ESM-2 keeps its DoRA
config, since nested tuning of a 650M model is prohibitively expensive.

Outer folds give an unbiased OOF while an inner CV inside each outer-training
split picks the hyperparameters without seeing that fold. The outer folds are the
same StratifiedKFold(5, shuffle=True, random_state=42) as the other learners, so
the tuned predictions stay row-aligned for the meta-learner.

Per model, the outer 5-fold gives tuned *_oof, then a final tune on all train
gives tuned *_test and the saved model. Untuned baselines are kept for comparison.

INPUTS  fusion_vectors.npz (X, label, split)
OUTPUTS  base_tree_predictions_tuned.npz, base_cnn_predictions_tuned.npz,
         base_xgb_tuned.joblib, base_rf_tuned.joblib, base_cnn_tuned.pt,
         optuna_best_params.json
ENV VARS  MODELS=xgb,rf,cnn selects what to tune. TREE_TRIALS and CNN_TRIALS
          set the budget. TEST_RUN=1 runs 2 trials on 1 fold as a path check
REQUIREMENTS  pip install optuna scikit-learn xgboost torch numpy joblib
"""

# Imports
import json
import os

# must be set before torch or xgboost import, or their bundled OpenMP runtimes
# abort on the duplicate library. leave above the third-party imports
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

# optuna logs one info line per trial, which at the default budgets is about 600
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3

TEST_RUN = os.environ.get("TEST_RUN", "0") == "1"
MODELS = os.environ.get("MODELS", "xgb,rf,cnn").split(",")
TREE_TRIALS = 2 if TEST_RUN else int(os.environ.get("TREE_TRIALS", "40"))
CNN_TRIALS = 2 if TEST_RUN else int(os.environ.get("CNN_TRIALS", "20"))
N_OUTER = 1 if TEST_RUN else OUTER_FOLDS


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
_T = {} # lazy torch handle cache

def torch_ctx():
    # torch is imported here rather than at the top so a trees-only run never
    # loads it next to xgboost. that pairing segfaults on macOS, since both ship
    # their own libomp and the two collide. returns a dict of torch, nn and device.
    if not _T: # empty only on the first call, populated from then on
        import torch
        import torch.nn as nn
        _T["torch"], _T["nn"] = torch, nn
        _T["device"] = ("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")
    return _T


def load_fused():
    # allow_pickle is needed since the npz also holds object arrays of sequences
    # and descriptor names, even though neither is read here
    d = np.load("fusion_vectors.npz", allow_pickle=True)
    X, y, split = d["X"].astype(np.float32), d["label"].astype(int), d["split"]

    # split holds "train" and "test" strings, so these give boolean masks that
    # pick whole rows out of X and y
    tr, te = split == "train", split == "test"
    return X[tr], y[tr], X[te], y[te]


# --------------------------------------------------------------------------- #
# TREES (XGBoost / Random Forest)
# --------------------------------------------------------------------------- #

# each *_space defines a search space. trial.suggest_* asks Optuna to propose a
# value in the given range for this trial. log=True samples on a log scale, for
# params spanning orders of magnitude such as learning rate
def xgb_space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0), # rows sampled per tree
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0), # features per tree
        min_child_weight=trial.suggest_int("min_child_weight", 1, 8),
        gamma=trial.suggest_float("gamma", 0.0, 5.0), # min loss gain to make a split
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)) # L2


def make_xgb(p):
    # imported here for the same reason torch is, so a run without xgb never loads
    # it next to torch. see torch_ctx
    from xgboost import XGBClassifier
    return XGBClassifier(**p, eval_metric="logloss", n_jobs=-1, random_state=SEED)


def rf_space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
        max_depth=trial.suggest_categorical("max_depth", [None, 10, 20, 30]), # None grows fully
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10))


def make_rf(p):
    # top-level import, since sklearn isn't part of the torch/xgboost conflict
    return RandomForestClassifier(**p, n_jobs=-1, random_state=SEED)


def tune_tree(make_model, suggest_params, X, y, n_trials):
    # every trial is scored on the same inner split, so trials are comparable
    inner_cv = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED)

    # optuna calls this once per trial and maximises whatever it returns
    def objective(trial):
        params = suggest_params(trial) # one candidate config from the search space
        model = make_model(params) # unfitted, cross_val_score does the fitting
        # n_jobs=1 since the model itself already takes every core
        return cross_val_score(model, X, y, cv=inner_cv, scoring="roc_auc",
                               n_jobs=1).mean()

    # TPE is Bayesian search, learning which regions of the space look promising
    # seeded so the search path is reproducible
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def nested_tree(name, make, space, Xtr, ytr, Xte, n_trials):
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(ytr), dtype=np.float32)

    # outer loop gives the tuned out-of-fold predictions. under TEST_RUN only the
    # first fold runs, so the rest of oof stays zero and should not be read.
    for k, (train_ind, test_ind) in enumerate(outer.split(Xtr, ytr)):
        if k >= N_OUTER:
            break
        bp = tune_tree(make, space, Xtr[train_ind], ytr[train_ind], n_trials) # inner CV on this fold's train
        m = make(bp).fit(Xtr[train_ind], ytr[train_ind])
        oof[test_ind] = m.predict_proba(Xtr[test_ind])[:, 1] # score the held-out outer fold
        print(f"  [{name}] outer fold {k + 1}/{N_OUTER} tuned")

    # then tune once more on all of train, fit, then predict the test set
    best = tune_tree(make, space, Xtr, ytr, n_trials)
    final = make(best).fit(Xtr, ytr)
    test = final.predict_proba(Xte)[:, 1].astype(np.float32)
    return oof, test, best, final


# --------------------------------------------------------------------------- #
# 1D-CNN (tunable version of the notebook's FusionCNN)
# --------------------------------------------------------------------------- #

def build_cnn(ch1=32, ch2=64, k1=7, p=0.4):
    """Tunable FusionCNN, same structure as phase3.2. The class is defined inside
    the function so nn.Module is only needed once torch has actually been imported."""
    nn = torch_ctx()["nn"] # lazy torch, see torch_ctx

    # same architecture as the notebook CNN, but ch1, ch2, k1 and p are arguments
    # now so Optuna can tune them. the second conv keeps a fixed kernel of 5 to
    # hold the search space down
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
    # architecture (channels, kernel, dropout) plus optimisation (lr, wd, batch)
    return dict(
        ch1=trial.suggest_categorical("ch1", [16, 32, 64]), # 1st conv channels
        ch2=trial.suggest_categorical("ch2", [32, 64, 128]), # 2nd conv channels
        k1=trial.suggest_categorical("k1", [5, 7, 9]), # 1st conv kernel size
        p=trial.suggest_float("p", 0.2, 0.6), # dropout rate
        lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True), # L2
        batch=trial.suggest_categorical("batch", [32, 64, 128]))


def fit_cnn(Xtr_raw, ytr, params, seed=SEED, max_epochs=200, patience=15):
    """Scale on the training data given, then train FusionCNN with early stopping
    on an internal 15% validation split. Returns (model, scaler)."""
    t = torch_ctx()
    torch, nn, device = t["torch"], t["nn"], t["device"]
    torch.manual_seed(seed)

    # the CNN is scale-sensitive, so fit a scaler on this training data and keep
    # it, since predicting later needs the same transform
    scaler = StandardScaler().fit(Xtr_raw)
    Xs = scaler.transform(Xtr_raw).astype(np.float32)

    # internal 15% val split for early stopping
    xtr, xval, ytr_, yval = train_test_split(Xs, ytr, test_size=0.15,
                                             stratify=ytr, random_state=seed)

    def to(a): # numpy -> tensor on device
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)

    xtr_t, xval_t = to(xtr).unsqueeze(1), to(xval).unsqueeze(1) # add channel dim
    ytr_t, yval_t = to(ytr_), to(yval)

    model = build_cnn(params["ch1"], params["ch2"],
                      params["k1"], params["p"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"],
                           weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    bs = params["batch"]

    best, best_state, wait = np.inf, None, 0 # early stopping trackers
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
            best, wait = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state) # restore the best epoch
    return model, scaler # the scaler comes back too, prediction needs it


def cnn_predict(model, scaler, Xraw):
    # scale with the model's own fitted scaler, then forward -> sigmoid -> probabilities
    t = torch_ctx()
    torch, device = t["torch"], t["device"]
    model.eval()
    with torch.no_grad():
        x = torch.tensor(scaler.transform(Xraw).astype(np.float32),
                         device=device).unsqueeze(1) # add channel dim
        return torch.sigmoid(model(x)).cpu().numpy().astype(np.float32)


def tune_cnn(Xtr_raw, ytr, n_trials, max_epochs):
    """Inner single stratified holdout AUC (keeps nested CNN tractable)."""
    # CNN fits are slow, so each trial is scored on one 20% holdout rather than
    # inner k-fold, which would cost three times the fits. fit_cnn then carves a
    # further 15% out of the remaining 80% for its own early stopping
    xin, xval, yin, yval = train_test_split(Xtr_raw, ytr, test_size=0.2,
                                            stratify=ytr, random_state=SEED)

    def objective(trial):
        # fit on the inner-train with this trial's params, score AUC on the holdout
        m, sc = fit_cnn(xin, yin, cnn_space(trial), seed=SEED,
                        max_epochs=max_epochs, patience=10)
        return roc_auc_score(yval, cnn_predict(m, sc, xval))

    # TPE is Bayesian search, learning which regions of the space look promising
    # seeded so the search path is reproducible.
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def nested_cnn(Xtr, ytr, Xte, n_trials):
    # shorter epochs while searching, full epochs for the real fits
    tune_epochs = 5 if TEST_RUN else 60
    fit_epochs = 10 if TEST_RUN else 200
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(ytr), dtype=np.float32)

    # outer loop gives the tuned out-of-fold predictions. under TEST_RUN only the
    # first fold runs, so the rest of oof stays zero and should not be read
    for k, (train_ind, test_ind) in enumerate(outer.split(Xtr, ytr)):
        if k >= N_OUTER:
            break
        bp = tune_cnn(Xtr[train_ind], ytr[train_ind], n_trials, tune_epochs)
        m, sc = fit_cnn(Xtr[train_ind], ytr[train_ind], bp, seed=SEED + k,
                        max_epochs=fit_epochs)
        oof[test_ind] = cnn_predict(m, sc, Xtr[test_ind]) # score the held-out fold
        print(f"  [cnn] outer fold {k + 1}/{N_OUTER} tuned")

    # then tune once more on all of train, fit, then predict the test set
    best = tune_cnn(Xtr, ytr, n_trials, tune_epochs)
    final, sc = fit_cnn(Xtr, ytr, best, seed=SEED, max_epochs=fit_epochs)
    test = cnn_predict(final, sc, Xte)
    return oof, test, best, (final, sc) # the final model comes back with its scaler


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    print(f"tuning {', '.join(MODELS)} with {TREE_TRIALS} tree trials "
          f"and {CNN_TRIALS} cnn trials" + (" (test run)" if TEST_RUN else ""))
    Xtr, ytr, Xte, yte = load_fused()
    print(f"loaded {len(ytr)} train and {len(yte)} test rows, {Xtr.shape[1]} features each")

    # MODELS decides which learners this process tunes. trees and cnn run as
    # separate processes (xgb,rf then cnn) to dodge the libomp segfault, so any
    # one run fills only some of these dicts. under TEST_RUN the OOF AUCs below
    # cover fold 0 only, since the rest of each oof array stays zero.
    best_params, tree_preds, cnn_preds = {}, {}, {}

    if "xgb" in MODELS:
        print("tuning XGBoost")
        oof, test, bp, model = nested_tree("xgb", make_xgb, xgb_space,
                                           Xtr, ytr, Xte, TREE_TRIALS)
        tree_preds.update(xgb_oof=oof, xgb_test=test)
        best_params["xgb"] = bp
        joblib.dump(model, "models/base_xgb_tuned.joblib")
        print(f"XGBoost done, OOF AUC {roc_auc_score(ytr, oof):.3f} "
              f"and test AUC {roc_auc_score(yte, test):.3f}")

    if "rf" in MODELS:
        print("tuning Random Forest")
        oof, test, bp, model = nested_tree("rf", make_rf, rf_space,
                                           Xtr, ytr, Xte, TREE_TRIALS)
        tree_preds.update(rf_oof=oof, rf_test=test)
        best_params["rf"] = bp
        joblib.dump(model, "models/base_rf_tuned.joblib")
        print(f"Random Forest done, OOF AUC {roc_auc_score(ytr, oof):.3f} "
              f"and test AUC {roc_auc_score(yte, test):.3f}")

    if "cnn" in MODELS:
        print("tuning 1D-CNN")
        oof, test, bp, (model, sc) = nested_cnn(Xtr, ytr, Xte, CNN_TRIALS)
        cnn_preds.update(cnn_oof=oof, cnn_test=test)
        best_params["cnn"] = bp
        torch_ctx()["torch"].save(model.state_dict(), "models/base_cnn_tuned.pt")
        joblib.dump(sc, "models/base_cnn_tuned_scaler.joblib") # weights alone can't predict
        print(f"1D-CNN done, OOF AUC {roc_auc_score(ytr, oof):.3f} "
              f"and test AUC {roc_auc_score(yte, test):.3f}")

    # only the models this process actually ran
    if tree_preds:
        np.savez_compressed("predictions/base_tree_predictions_tuned.npz",
                            y_train=ytr, y_test=yte, **tree_preds)
        print("saved predictions/base_tree_predictions_tuned.npz")
    if cnn_preds:
        np.savez_compressed("predictions/base_cnn_predictions_tuned.npz",
                            y_train=ytr, y_test=yte, **cnn_preds)
        print("saved predictions/base_cnn_predictions_tuned.npz")

    # the best-params JSON is shared by both processes, so load, update and write
    # back rather than overwriting. a plain dump from the cnn run would wipe the
    # trees entries, or the other way round.
    merged = {}
    if os.path.exists("optuna_best_params.json"):
        with open("optuna_best_params.json") as fh:
            merged = json.load(fh)
    merged.update(best_params)
    with open("optuna_best_params.json", "w") as fh:
        json.dump(merged, fh, indent=2, default=str)
    print("saved optuna_best_params.json")

    # quick tuned against untuned comparison, skipped if the untuned files are
    # not present, which is what the try/except is for.
    print("test AUC before and after tuning")
    try:
        ut = np.load("predictions/base_tree_predictions.npz")
        uc = np.load("predictions/base_cnn_predictions.npz")
        base = {"xgb": ut["xgb_test"], "rf": ut["rf_test"], "cnn": uc["cnn_test"]}
        for m in MODELS:
            if m in ("xgb", "rf") and tree_preds:
                tuned = tree_preds[f"{m}_test"]
            elif m == "cnn" and cnn_preds:
                tuned = cnn_preds["cnn_test"]
            else:
                continue
            b, t = roc_auc_score(yte, base[m]), roc_auc_score(yte, tuned)
            print(f"{m:4s} {b:.3f} -> {t:.3f} ({t - b:+.3f})")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
