#!/usr/bin/env bash
#SBATCH --job-name=esm2_seed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00  # one fine-tune, not six, so shorter than the main job
#SBATCH --account=chem-data-2023  # REQUIRED, job won't run without it
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=om894@york.ac.uk
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err

# Seed-stability run behind the AUC spread in section 3.1. Refits the DoRA model
# on the full training set under one seed and scores the test set, skipping the
# 5-fold OOF since only the test AUC is needed. Submit once per seed 42 to 46.
#
#   SEED=43 sbatch phase3.3_esm2_dora_seed.sbatch
#
# SLURM exports the calling environment by default, which is how SEED reaches the
# job. Expects the flat ~/adp/ layout from viking_runbook.md, not the repo tree.

set -e
: "${SEED:?set SEED before submitting, otherwise every run overwrites one file}"

module purge
module load Python/3.11.3-GCCcore-12.3.0
source ~/esm2env/bin/activate

# GPU nodes have no internet, so load the model from the cache populated on the
# login node and force offline so transformers never tries to reach the hub.
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export DATA=data/dataset_split.csv
export SKIP_OOF=1
export OUT_NPZ=predictions/esm2_dora_seed${SEED}_predictions.npz
export OUT_ADAPTER=models/esm2_seed${SEED}_adapter

nvidia-smi
python phase3.3_esm2_dora.py