
# Install:
# pip install shap xgboost scikit-learn joblib pandas numpy matplotlib

"""
Phase 5.6: SHAP interpretability - which physicochemical features drive the tree
learners' ADP predictions.

TreeExplainer (exact for trees) on XGBoost and RF, both trained on the 1287-d fused
vector (1280 ESM-2 embedding dims + 7 physicochemical descriptors). The embedding dims
aren't individually interpretable, so the focus is the 7 named descriptors. Their
SHAP importance (mean |value|) and direction, plus how much the trees lean on the
descriptors vs the PLM embeddings overall. Explained on the held-out test set.

INPUT   fusion_vectors.npz     X, descriptor_names, split
        models/base_xgb.joblib, models/base_rf.joblib
OUTPUT  screening/shap_descriptor_importance.csv
        screening/shap_xgb_descriptors.png, screening/shap_rf_descriptors.png

REQUIREMENTS  pip install shap xgboost scikit-learn joblib pandas numpy matplotlib
"""

# Imports 
import warnings
warnings.filterwarnings("ignore") # silence warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # use 'agg' backend for matplotlib
import matplotlib.pyplot as plt
import shap

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
FUSION  = "fusion_vectors.npz"
MODELS  = {"xgb": "models/base_xgb.joblib", "rf": "models/base_rf.joblib"}
OUT_CSV = "screening/shap_descriptor_importance.csv"


def main():
    d = np.load(FUSION, allow_pickle=True)
    X = d["X"].astype(np.float32)
    desc_names = [str(s) for s in d["descriptor_names"]] # the 7 descriptors
    n_emb = X.shape[1] - len(desc_names) # 1280 embedding dims
    desc_idx = list(range(n_emb, X.shape[1])) # index 1280 to 1286

    Xte = X[d["split"] == "test"] # SHAP explainations on the held-out test set

    rows = []
    for tag, path in MODELS.items():
        model = joblib.load(path)
        sv = np.asarray(shap.TreeExplainer(model).shap_values(Xte))
        if sv.ndim == 3: # RF returns (n, features, classes)
            sv = sv[:, :, 1] # take class 1 = P(ADP)

        mean_abs = np.abs(sv).mean(0) # [1287] global importance per feature
        for j, name in zip(desc_idx, desc_names):
            rows.append({"model": tag, "feature": name, "mean_abs_shap": float(mean_abs[j])})

        # how much do the trees lean on the 7 descriptors vs the 1280 embeddings?
        desc_share = mean_abs[n_emb:].sum() / mean_abs.sum()
        print(f"[{tag}] descriptor share of total |SHAP|: {desc_share:.1%}")

        # plot the SHAP summary plot for the 7 descriptors, save to png
        shap.summary_plot(sv[:, desc_idx], Xte[:, desc_idx],
                          feature_names=desc_names, show=False)
        plt.title(f"SHAP - physicochemical descriptors ({tag})")
        plt.tight_layout()
        plt.savefig(f"screening/shap_{tag}_descriptors.png", dpi=150)
        plt.close()

    # save the descriptor importance to csv for later inspection
    imp = pd.DataFrame(rows)
    imp.to_csv(OUT_CSV, index=False)

    # print out a table of the descriptor importance ranked by xgb
    print("\ndescriptor importance (mean |SHAP|), ranked by xgb:")
    print(imp.pivot(index="feature", columns="model", values="mean_abs_shap")
             .sort_values("xgb", ascending=False).round(4).to_string())
    print(f"\nsaved -> {OUT_CSV} + screening/shap_*_descriptors.png")


if __name__ == "__main__":
    main()