# Viking HPC runbook

The two GPU-bound steps — the ESM-2/DoRA fine-tune (base learner 4) and the
negative-class ablation — were run on the University of York **Viking** cluster
(SLURM). Everything else in the pipeline runs locally. Username `*****`,
project account `chem-data-2023`.


## Prerequisites
- University VPN if off-campus.
- Log in: `ssh *****@viking.york.ac.uk`

## One-time environment setup
Run on a **login node** — login nodes have internet, GPU compute nodes do not.
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

## 1. Main ESM-2 base learner
Produces `predictions/esm2_dora_predictions.npz` and `models/esm2_dora_adapter/`.

From **Mac**:
```bash
scp phase_3_models/phase3.3_esm2_dora.py phase_3_models/phase3.3_esm2_dora.sbatch *****@viking.york.ac.uk:~/adp/
scp data/dataset_split.csv *****@viking.york.ac.uk:~/adp/data/
```
On **Viking** (5-fold OOF + final fit, ~15 min):
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

## 2. Negative-class ablation
Produces `predictions/esm2_dora_basith_predictions.npz`. Uses the same script with
`SKIP_OOF=1` (single fine-tune) on the Basith-negative dataset.

From **Mac**:
```bash
scp phase_3_models/phase3.3_esm2_dora.py phase_3_models/phase3.6_negative_class_ablation/phase3.6.3_esm2_dora_basith.sbatch *****@viking.york.ac.uk:~/adp/
scp phase_3_models/phase3.6_negative_class_ablation/dataset_split_basith.csv *****@viking.york.ac.uk:~/adp/data/
```
On **Viking** (single fit, ~3–4 min):
```bash
cd ~/adp
sbatch phase3.6.3_esm2_dora_basith.sbatch
squeue -u *****
```
Retrieve to **Mac**:
```bash
scp *****@viking.york.ac.uk:~/adp/predictions/esm2_dora_basith_predictions.npz predictions/
```

## Notes
- **GPU nodes have no internet** — the sbatch scripts set `TRANSFORMERS_OFFLINE=1`
  and `HF_HUB_OFFLINE=1` and load the pre-cached model.
- `--account=chem-data-2023` is **mandatory** or the job will not run.
- Monitor with `squeue -u *****`; the job emails `*****@york.ac.uk` on completion.
- Check the result: `cat esm2_*-<jobid>.log` (expect `State: COMPLETED` and a
  `saved -> predictions/...` line).