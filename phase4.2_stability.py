

"""
Section 4: run-to-run stability, measured as test-set AUC across random seeds.

For each learner and each seed, the model is trained on the full training set and
evaluated on the 178-row test set; the SD of that test AUC across seeds is the
run-to-run variance. The same metric (test AUC) is used for every learner so the
numbers are directly comparable (and comparable to ESM-2's 5 single-fit Viking seeds).

Run trees and CNN as separate processes (torch + xgboost both bundle libomp and
segfault together on macOS), selected with the MODELS env var:
    MODELS=xgb,rf python phase4_stability.py   # trees (no torch imported)
    MODELS=cnn    python phase4_stability.py   # CNN (no xgboost imported)
"""

import os
import numpy as np
from sklearn.metrics import roc_auc_score

SEEDS = [42, 43, 44, 45, 46] # 5 random seeds
MODELS = os.environ.get("MODELS", "xgb,rf,cnn").split(",")

# Load the fused vectors and labels, split into train/test by the original split
def load_fused():
    d = np.load("fusion_vectors.npz", allow_pickle=True)
    X, y, split = d["X"].astype(np.float32), d["label"].astype(int), d["split"]
    tr, te = split == "train", split == "test"
    return X[tr], y[tr], X[te], y[te]

# Report the mean and SD of test AUC across seeds
def report(name, aucs):
    print(f"  {name:6s} test AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}  "
          f"(seeds {SEEDS[0]}-{SEEDS[-1]})")


# trees: refit on full train under each seed, score the test set
def xgb_stability(Xtr, ytr, Xte, yte):
    from xgboost import XGBClassifier
    aucs = []
    for s in SEEDS:
        m = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                          eval_metric="logloss", n_jobs=-1, random_state=s).fit(Xtr, ytr)
        aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    report("xgb", aucs)


# trees: refit on full train under each seed, score the test set
def rf_stability(Xtr, ytr, Xte, yte):
    from sklearn.ensemble import RandomForestClassifier
    aucs = []
    for s in SEEDS:
        m = RandomForestClassifier(n_estimators=400, max_features="sqrt",
                                   min_samples_leaf=2, n_jobs=-1, random_state=s).fit(Xtr, ytr)
        aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    report("rf", aucs)


# CNN: same idea, one full fit per seed
def cnn_stability(Xtr_raw, ytr, Xte_raw, yte):
    import torch, torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    class FusionCNN(nn.Module): # same architecture as the base CNN
        def __init__(self, n_feat, p=0.4):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
                nn.MaxPool1d(2), nn.Dropout(p),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                nn.MaxPool1d(2), nn.Dropout(p),
                nn.AdaptiveAvgPool1d(1))
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(),
                                      nn.Dropout(p), nn.Linear(32, 1))
        def forward(self, x): return self.head(self.conv(x)).squeeze(1)

    # Fit the CNN on the full training set under a given seed, return test-set predictions
    def fit_predict(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        scaler = StandardScaler().fit(Xtr_raw) # CNN needs standardised inputs
        Xs = scaler.transform(Xtr_raw).astype(np.float32)
        xtr, xval, ytr_, yval = train_test_split(Xs, ytr, test_size=0.15,
                                                 stratify=ytr, random_state=seed)
        to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)
        xtr_t, xval_t, ytr_t, yval_t = to(xtr).unsqueeze(1), to(xval).unsqueeze(1), to(ytr_), to(yval)
        model = FusionCNN(Xtr_raw.shape[1]).to(device)
        opt, loss_fn = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4), nn.BCEWithLogitsLoss()
        best, best_state, wait = np.inf, None, 0
        for _ in range(200): # early stopping on internal val loss
            model.train()
            for i in range(0, len(xtr_t), 64):
                opt.zero_grad()
                loss_fn(model(xtr_t[i:i+64]), ytr_t[i:i+64]).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(xval_t), yval_t).item()
            if vl < best - 1e-4:
                best, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= 15: break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            xte = torch.tensor(scaler.transform(Xte_raw).astype(np.float32), device=device).unsqueeze(1)
            return torch.sigmoid(model(xte)).cpu().numpy()

    report("cnn", [roc_auc_score(yte, fit_predict(s)) for s in SEEDS])

# Run each model under each seed, report mean and SD of test AUC
def main():
    Xtr, ytr, Xte, yte = load_fused()
    print(f"stability: test AUC across seeds {SEEDS}  (train {len(ytr)}, test {len(yte)})")
    if "xgb" in MODELS: xgb_stability(Xtr, ytr, Xte, yte)
    if "rf"  in MODELS: rf_stability(Xtr, ytr, Xte, yte)
    if "cnn" in MODELS: cnn_stability(Xtr, ytr, Xte, yte)


if __name__ == "__main__":
    main()