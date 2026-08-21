#!/usr/bin/env bash
#SBATCH --job-name=esm2_dora
#SBATCH --partition=gpu  # A40 (3/node); use gpuplus for H100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00  # six sequential fine-tunes, 5 OOF folds plus 1 full
#SBATCH --account=chem-data-2023  # REQUIRED, job won't run without it
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=om894@york.ac.uk
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err

# Main DoRA training run for base learner 4. Uses every default in the script, so
# it reads data/dataset_split.csv and writes predictions/esm2_dora_predictions.npz
# plus models/esm2_dora_adapter/. The ablation and seed launchers override those
# paths through env vars instead.
#
# Cheap path check first, 1 fold x 1 epoch on 64 rows.
#   TEST_RUN=1 sbatch phase3.3_esm2_dora.sbatch
#
# Expects the flat ~/adp/ layout from viking_runbook.md, not the repo tree.

set -e
module purge
module load Python/3.11.3-GCCcore-12.3.0
source ~/esm2env/bin/activate

# GPU nodes have no internet, so load the model from the cache populated on the
# login node and force offline so transformers never tries to reach the hub.
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

nvidia-smi
python phase3.3_esm2_dora.py
