
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
OUTPUT  results/phase5_6_shap_descriptors.csv

REQUIREMENTS  pip install shap xgboost scikit-learn joblib pandas numpy
"""

# Imports 
import warnings
warnings.filterwarnings("ignore") # silence warnings

import joblib
import numpy as np
import pandas as pd
import shap
import os

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
FUSION  = "fusion_vectors.npz"
MODELS  = {"xgb": "models/base_xgb.joblib", "rf": "models/base_rf.joblib"}
OUT_CSV = "results/phase5_6_shap_descriptors.csv"


def main():
    os.makedirs("results", exist_ok=True)
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
        if tag == "xgb":  # the learner in the consensus, and the one Figure 8b draws
            np.savez_compressed("results/phase5_6_shap_matrix.npz",
                                shap_desc=sv[:, desc_idx], value_desc=Xte[:, desc_idx],
                                names=np.array(desc_names, dtype=object))

        mean_abs = np.abs(sv).mean(0)  # [1287] global importance per feature

        # how much do the trees lean on the 7 descriptors vs the 1280 embeddings?
        desc_share = mean_abs[n_emb:].sum() / mean_abs.sum()

        for j, name in zip(desc_idx, desc_names):
            # sign of the relationship, as the beeswarm's colour axis showed it.
            # positive means a higher descriptor value pushes the prediction towards ADP
            direction = (float(np.corrcoef(Xte[:, j], sv[:, j])[0, 1])
                         if sv[:, j].std() > 0 else float("nan"))
            rows.append({"model": tag, "feature": name,
                         "mean_abs_shap": float(mean_abs[j]),
                         "value_shap_corr": round(direction, 3),
                         "descriptor_share_of_total": round(float(desc_share), 4)})
        print(f"[{tag}] descriptor share of total |SHAP|: {desc_share:.1%}")

    # save the descriptor importance to csv
    imp = pd.DataFrame(rows)
    imp.to_csv(OUT_CSV, index=False)

    # print out a table of the descriptor importance ranked by xgb
    print("\ndescriptor importance (mean |SHAP|), ranked by xgb:")
    print(imp.pivot(index="feature", columns="model", values="mean_abs_shap")
             .sort_values("xgb", ascending=False).round(4).to_string())
    print(f"\nsaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()