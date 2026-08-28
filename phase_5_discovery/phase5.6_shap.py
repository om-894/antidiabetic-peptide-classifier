
"""
Phase 5.6: SHAP attribution over the seven physicochemical descriptors.

TreeExplainer (exact for trees) on XGBoost and RF, both trained on the 1287-d fused
vector (1280 ESM-2 embedding dims + 7 physicochemical descriptors). The embedding dims
aren't individually interpretable, so the focus is the 7 named descriptors. Their SHAP
importance (mean |value|) and direction, plus how much the trees lean on the descriptors
against the embeddings overall. Explained on the held-out test set.

INPUTS  fusion_vectors.npz (X, descriptor_names, split), base_xgb.joblib, base_rf.joblib
OUTPUTS  phase5_6_shap_descriptors.csv (per model: mean_abs_shap, value_shap_corr)
         phase5_6_shap_matrix.npz (the xgb descriptor values behind figure 12b)
REQUIREMENTS  pip install shap xgboost scikit-learn joblib pandas numpy
"""

# Imports
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import pearsonr

# the shap and xgboost stack emits deprecation noise that has nothing to do with the
# attributions. It would bury the one line this script prints
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

FUSION = "fusion_vectors.npz"
MODELS = {"xgb": "models/base_xgb.joblib", "rf": "models/base_rf.joblib"}
OUT_CSV = "results/phase5_6_shap_descriptors.csv"
OUT_NPZ = "results/phase5_6_shap_matrix.npz" # the xgb values figure 12b draws


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    os.makedirs("results", exist_ok=True)
    d = np.load(FUSION, allow_pickle=True)
    X = d["X"].astype(np.float32)
    desc_names = [str(s) for s in d["descriptor_names"]] # the 7 descriptors
    n_emb = X.shape[1] - len(desc_names) # 1280 embedding dims
    desc_idx = list(range(n_emb, X.shape[1])) # index 1280 to 1286

    Xte = X[d["split"] == "test"] # explained on the held-out test set

    rows = []
    for tag, path in MODELS.items():
        model = joblib.load(path)
        sv = np.asarray(shap.TreeExplainer(model).shap_values(Xte))
        if sv.ndim == 3: # RF returns (n, features, classes)
            sv = sv[:, :, 1] # take class 1 = P(ADP)

        # only the xgb values are kept, since it is the learner in the consensus and
        # the one figure 12b draws. the seven descriptor columns are all the figure needs
        if tag == "xgb":
            np.savez_compressed(OUT_NPZ, shap_desc=sv[:, desc_idx],
                                value_desc=Xte[:, desc_idx],
                                names=np.array(desc_names, dtype=object))

        mean_abs = np.abs(sv).mean(0) # [1287] global importance per feature

        # how much do the trees lean on the 7 descriptors against the 1280 embeddings?
        desc_share = mean_abs[n_emb:].sum() / mean_abs.sum()

        for j, name in zip(desc_idx, desc_names):
            # sign of the relationship, as the beeswarm's colour axis showed it. positive
            # means a higher descriptor value pushes the prediction towards ADP. a feature
            # the model never split on has zero variance in its values, so it gets nan
            if sv[:, j].std() > 0:
                direction, direction_p = pearsonr(Xte[:, j], sv[:, j])
            else:
                direction, direction_p = float("nan"), float("nan")
            rows.append({"model": tag, "feature": name,
                        "mean_abs_shap": float(mean_abs[j]),
                        "value_shap_corr": round(float(direction), 3),
                        "value_shap_corr_p": float(f"{direction_p:.3g}"),
                        "descriptor_share_of_total": round(float(desc_share), 4)})

    imp = pd.DataFrame(rows)
    imp.to_csv(OUT_CSV, index=False)

    top = imp[imp.model == "xgb"].nlargest(1, "mean_abs_shap").iloc[0]
    print(f"xgb: descriptors carry {top.descriptor_share_of_total:.1%} of total |SHAP|, "
          f"top {top.feature} at {top.mean_abs_shap:.3f} -> {OUT_CSV}")


if __name__ == "__main__":
    main()