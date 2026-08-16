#!/usr/bin/env bash
#SBATCH --job-name=esm2_dora
#SBATCH --partition=gpu            # A40 (3/node); use gpuplus for H100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --account=chem-data-2023   # REQUIRED — job won't run without it
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=om894@york.ac.uk
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err

set -e
module purge
module load Python/3.11.3-GCCcore-12.3.0
source ~/esm2env/bin/activate

# GPU nodes have no internet: load the model from the cache populated on the
# login node, and force offline so transformers never tries to reach the hub.
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

nvidia-smi
# Smoke test (1 fold x 1 epoch on 64 rows) before the real run:
#   SMOKE_TEST=1 sbatch phase3_esm2_dora.sbatch
python phase3.3_esm2_dora.py
