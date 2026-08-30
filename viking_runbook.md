# Viking HPC runbook

Four steps ran on the University of York Viking cluster (SLURM): the
ESM-2/DoRA fine-tune, its seed refits, the negative-class ablation and the
screening pass. Everything else runs locally. Username `*****`, account
`chem-data-2023`.

## What ran where

Phase 2.1 has no sbatch and was ran locally. Frozen embedding extraction stayed local. Four steps did not:

- **Phase 3.3, main fine-tune.** Six sequential fits: five OOF folds plus a final
  all-train fit, on a 650M backbone.
- **Phase 3.3, seed refits.** Five single fine-tunes.
- **Phase 3.6.3, ablation.** One fine-tune.
- **Phase 5.2, screening.** Two forward passes of the 650M model over 3,596
  candidates, once frozen and once with the adapter.

Every sbatch requests `--partition=gpu`. `gpuplus` (H100) appears only in a
comment on `phase3.3_esm2_dora.sbatch`.

## Prerequisites
- University VPN if off-campus.
- Log in: `ssh *****@viking.york.ac.uk`

## One-time environment setup
Run on a login node. Login nodes have internet, GPU compute nodes do not.
```bash
module purge
module load Python/3.11.3-GCCcore-12.3.0
python -m venv ~/esm2env
source ~/esm2env/bin/activate
pip install --upgrade pip
pip install torch transformers peft accelerate pandas numpy scikit-learn

# Pre-download ESM-2 into the cache the (offline) GPU job will read:
export HF_HOME=~/.cache/huggingface
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; \
AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D'); \
M.from_pretrained('facebook/esm2_t33_650M_UR50D', num_labels=2)"
```

The four jobs run two scripts, `phase3.3_esm2_dora.py` and
`phase5.2_esm2_screen.py`, differing only in the environment variables their
sbatch sets.

## 1. Main ESM-2 base learner
Produces `predictions/esm2_dora_predictions.npz` and `models/esm2_dora_adapter/`.

From **Mac**:
```bash
scp phase_3_models/phase3.3_esm2_dora.py phase_3_models/phase3.3_esm2_dora.sbatch *****@viking.york.ac.uk:~/adp/
scp data/dataset_split.csv *****@viking.york.ac.uk:~/adp/data/
```
On **Viking**:
```bash
cd ~/adp
sbatch phase3.3_esm2_dora.sbatch
squeue -u *****
```
Retrieve to **Mac**:
```bash
scp *****@viking.york.ac.uk:~/adp/predictions/esm2_dora_predictions.npz predictions/
scp -r *****@viking.york.ac.uk:~/adp/models/esm2_dora_adapter models/
```

## 2. Seed refits
Produces `predictions/esm2_dora_seed{42..46}_predictions.npz`, read by
`phase4.2_stability.py` under `MODELS=esm2`.

`SKIP_OOF=1` and the per-seed output paths are set in the sbatch; `SEED` comes
from the submitting shell. Uses the `dataset_split.csv` staged in step 1.

From **Mac**:
```bash
scp phase_3_models/phase3.3_esm2_dora_seed.sbatch *****@viking.york.ac.uk:~/adp/
```
On **Viking**, once per seed:
```bash
cd ~/adp
SEED=42 sbatch phase3.3_esm2_dora_seed.sbatch   # repeat for 43, 44, 45, 46
squeue -u *****
```
Retrieve to **Mac** (npz only):
```bash
scp '*****@viking.york.ac.uk:~/adp/predictions/esm2_dora_seed4[2-6]_predictions.npz' predictions/
```
Each job also writes `models/esm2_seed${SEED}_adapter/` on Viking. Not retrieved:
nothing reloads them and five copies of a 23 MB adapter is 117 MB.

Seed 42 repeats step 1's configuration and its `esm_test` matches it exactly:
```bash
python -c "import numpy as np; a=np.load('predictions/esm2_dora_predictions.npz')['esm_test']; \
b=np.load('predictions/esm2_dora_seed42_predictions.npz')['esm_test']; \
print('identical:', np.array_equal(a,b))"
```

## 3. Negative-class ablation
Produces `predictions/esm2_dora_basith_predictions.npz`. Same script, `SKIP_OOF=1`,
on the Basith-negative dataset.

From **Mac**:
```bash
scp phase_3_models/phase3.3_esm2_dora.py phase_3_models/phase3.6_negative_class_ablation/phase3.6.3_esm2_dora_basith.sbatch *****@viking.york.ac.uk:~/adp/
scp phase_3_models/phase3.6_negative_class_ablation/dataset_split_basith.csv *****@viking.york.ac.uk:~/adp/data/
```
On **Viking**:
```bash
cd ~/adp
sbatch phase3.6.3_esm2_dora_basith.sbatch
squeue -u *****
```
Retrieve to **Mac**:
```bash
scp *****@viking.york.ac.uk:~/adp/predictions/esm2_dora_basith_predictions.npz predictions/
```

## 4. Screening pass
Produces `screening/screening_esm2.npz`: the 1,280-d embeddings for XGBoost and
the DoRA probabilities, both used by phase 5.3. Nothing is trained.

Needs the phase 5.1 candidates and the step 1 adapter sent back up.

From **Mac**:
```bash
scp phase_5_discovery/phase5.2_esm2_screen.py phase_5_discovery/phase5.2_esm2_screen.sbatch *****@viking.york.ac.uk:~/adp/
scp screening/screening_candidates.csv *****@viking.york.ac.uk:~/adp/screening/
scp -r models/esm2_dora_adapter *****@viking.york.ac.uk:~/adp/models/
```
On **Viking** (30 min walltime requested):
```bash
cd ~/adp
sbatch phase5.2_esm2_screen.sbatch
squeue -u *****
```
Retrieve to **Mac**:
```bash
scp *****@viking.york.ac.uk:~/adp/screening/screening_esm2.npz screening/
```