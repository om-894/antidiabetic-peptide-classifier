#!/usr/bin/env bash
#SBATCH --job-name=esm2_basith
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00  # one fine-tune, not six, so shorter than the main job
#SBATCH --account=chem-data-2023  # required, job won't run without it
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=om894@york.ac.uk
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err

# Negative-class ablation run, the control the dual-negative model is measured
# against in section 3.2. This is phase 3.3 unchanged, same hyperparameters and
# same seed, trained on a dataset whose negatives come from the Basith Layer 1
# pool rather than the dual-negative class, so the env vars below are the whole
# experiment.
#
# Expects the flat ~/adp/ layout from viking_runbook.md, not the repo tree, so
# dataset_split_basith.csv is read from ~/adp/data/ and not from the ablation
# folder that phase 3.6.2 writes it to.

set -e
module purge
module load Python/3.11.3-GCCcore-12.3.0
source ~/esm2env/bin/activate

# GPU nodes have no internet, so load the model from the cache populated on the
# login node and force offline so transformers never tries to reach the hub.
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Point phase 3.3 at the ablation dataset and at its own output names, so the
# main run's npz and adapter survive.
export DATA=data/dataset_split_basith.csv
export OUT_NPZ=predictions/esm2_dora_basith_predictions.npz
export OUT_ADAPTER=models/esm2_dora_basith_adapter

# The control feeds no stacking, only its own test predictions, so skip the OOF.
export SKIP_OOF=1

nvidia-smi
python phase3.3_esm2_dora.py