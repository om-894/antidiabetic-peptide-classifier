
"""
Phase 4.2: Run-to-run stability, the test AUC spread across five random seeds.

Each learner is refitted on the full training set once per seed and scored on
the same 178-row test set, so the standard deviation of that AUC is roughly what
a rerun would move by. Using test AUC for all four keeps them comparable with
each other and with ESM-2, whose seeds were fine-tuned on Viking and are read
back from saved predictions rather than refitted here.

Trees and CNN have to run as separate processes. torch and xgboost each bundle
their own libomp and segfault when both are imported on macOS, so MODELS picks
which learners this process touches and their imports sit inside the functions.

  MODELS=xgb,rf python phase4.2_stability.py
  MODELS=cnn python phase4.2_stability.py

INPUTS  fusion_vectors.npz, esm2_dora_seed{42 to 46}_predictions.npz
OUTPUTS  phase4_2_seed_stability.csv (Table 7 seed column, Figure 6c)
ENV VARS  MODELS selects which learners run, default all four
REQUIREMENTS  pip install xgboost torch scikit-learn pandas numpy
"""

# Imports
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

SEEDS = [42, 43, 44, 45, 46] # the seeds Table 7 and Figure 6c report
RESULTS_DIR = "results" # repo root, as in every other phase script

# figures.ipynb maps these short names onto the display labels, so they are load
# bearing rather than labels
MODELS = os.environ.get("MODELS", "xgb,rf,cnn,esm2").split(",")


def load_fused():
    # the fused vector and the phase 1.3 split, so every learner here sees the
    # same rows the base learners saw
    d = np.load("fusion_vectors.npz", allow_pickle=True)
    X, y, split = d["X"].astype(np.float32), d["label"].astype(int), d["split"]
    tr, te = split == "train", split == "test"
    return X[tr], y[tr], X[te], y[te]


def summarise(name, aucs):
    """One CSV row, the mean and sample SD of test AUC over SEEDS."""
    # ddof=1 for the sample SD, since five seeds are a sample of the runs the
    # model could have produced rather than the whole population
    return {"model": name, "mean_auc": round(float(np.mean(aucs)), 4),
            "sd_auc": round(float(np.std(aucs, ddof=1)), 4),
            "n_seeds": len(SEEDS), "seeds": ",".join(map(str, SEEDS)),
            "per_seed_auc": ",".join(f"{a:.4f}" for a in aucs)}


# --------------------------------------------------------------------------- #
# TREES
# --------------------------------------------------------------------------- #
def xgb_aucs(Xtr, ytr, Xte, yte):
    # hyperparameters are phase 3.1's, held fixed so the seed is the only thing
    # varying. imported here rather than at module level
    from xgboost import XGBClassifier
    aucs = []
    for s in SEEDS:
        m = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                          eval_metric="logloss", n_jobs=-1, random_state=s).fit(Xtr, ytr)
        aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    return aucs


def rf_aucs(Xtr, ytr, Xte, yte):
    from sklearn.ensemble import RandomForestClassifier
    aucs = []
    for s in SEEDS:
        m = RandomForestClassifier(n_estimators=400, max_features="sqrt",
                                   min_samples_leaf=2, n_jobs=-1, random_state=s).fit(Xtr, ytr)
        aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    return aucs


# --------------------------------------------------------------------------- #
# ESM-2
# --------------------------------------------------------------------------- #
def esm_aucs():
    # the only learner not refitted here. its five seeds each cost a Viking GPU
    # job, so this reads what phase3.3_esm2_dora_seed.sbatch already produced
    aucs = []
    for s in SEEDS:
        f = f"predictions/esm2_dora_seed{s}_predictions.npz"
        if not os.path.exists(f):
            print(f"[skip] esm2: {f} missing")
            return None
        d = np.load(f, allow_pickle=True)
        aucs.append(roc_auc_score(d["y_test"].astype(int), d["esm_test"].astype(float)))
    return aucs


# --------------------------------------------------------------------------- #
# CNN
# --------------------------------------------------------------------------- #
def cnn_aucs(Xtr_raw, ytr, Xte_raw, yte):
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    class FusionCNN(nn.Module): # identical to phase 3.2's
        def __init__(self, p=0.4):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
                nn.MaxPool1d(2), nn.Dropout(p),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                nn.MaxPool1d(2), nn.Dropout(p),
                nn.AdaptiveAvgPool1d(1)) # pools to length 1, so the head never sees 1287
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(),
                                      nn.Dropout(p), nn.Linear(32, 1))
        def forward(self, x): return self.head(self.conv(x)).squeeze(1)

    def fit_predict(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # the CNN restandardises all 1,287 columns, where phase 2.2 z-scored only
        # the seven descriptors. fitted on train alone so the test rows stay unseen
        scaler = StandardScaler().fit(Xtr_raw)
        Xs = scaler.transform(Xtr_raw).astype(np.float32)

        # a 15% internal split for early stopping, carved from train so the test
        # set plays no part in deciding when to stop
        xtr, xval, ytr_, yval = train_test_split(Xs, ytr, test_size=0.15,
                                                 stratify=ytr, random_state=seed)
        to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)
        xtr_t, xval_t, ytr_t, yval_t = to(xtr).unsqueeze(1), to(xval).unsqueeze(1), to(ytr_), to(yval)
        model = FusionCNN().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()

        # best starts at inf, so the first epoch always sets best_state and the
        # load below can never see None
        best, best_state, wait = np.inf, None, 0
        for _ in range(200):
            model.train()
            for i in range(0, len(xtr_t), 64):
                opt.zero_grad()
                loss_fn(model(xtr_t[i:i + 64]), ytr_t[i:i + 64]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(xval_t), yval_t).item()
            if vl < best - 1e-4:
                best, wait = vl, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 15: # patience, matching phase 3.2
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            xte = torch.tensor(scaler.transform(Xte_raw).astype(np.float32), device=device).unsqueeze(1)
            return torch.sigmoid(model(xte)).cpu().numpy()

    return [roc_auc_score(yte, fit_predict(s)) for s in SEEDS]


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    Xtr, ytr, Xte, yte = load_fused()

    rows = []
    if "xgb" in MODELS:
        rows.append(summarise("xgb", xgb_aucs(Xtr, ytr, Xte, yte)))
    if "rf" in MODELS:
        rows.append(summarise("rf", rf_aucs(Xtr, ytr, Xte, yte)))
    if "cnn" in MODELS:
        rows.append(summarise("cnn", cnn_aucs(Xtr, ytr, Xte, yte)))
    if "esm2" in MODELS:
        aucs = esm_aucs()
        if aucs:
            rows.append(summarise("esm2", aucs))


    if not rows: # nothing ran, so leave any existing CSV untouched
        return

    # the trees and the CNN run as separate processes, so their rows have to merge
    # into one file. rewriting a model's row rather than appending keeps a rerun
    # from leaving two rows for the same learner, which figures.ipynb would plot twice
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "phase4_2_seed_stability.csv")
    out = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        out = pd.concat([old[~old.model.isin(out.model)], out], ignore_index=True)
    out.to_csv(path, index=False)


if __name__ == "__main__":
    main()