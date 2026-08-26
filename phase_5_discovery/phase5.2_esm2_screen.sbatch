#!/usr/bin/env bash
#SBATCH --job-name=esm2_screen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00  # two forward passes over 3,596 candidates, no training
#SBATCH --account=chem-data-2023  # REQUIRED, job won't run without it
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=om894@york.ac.uk
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err

# Screening pass for the ESM-2 half of the consensus. Embeds every candidate with
# the frozen encoder and scores it with the phase 3.3 adapter, so one run gives
# both what XGBoost needs in phase 5.3 and half the consensus itself. Nothing is
# trained here, which is why the walltime is short beside the fine-tuning jobs.
#
# Expects the flat ~/adp/ layout from viking_runbook.md, not the repo tree. This
# job stages more than the others, screening_candidates.csv into ~/adp/screening/
# and the whole esm2_dora_adapter directory into ~/adp/models/, with
# screening_esm2.npz retrieved afterwards.

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
python phase5.2_esm2_screen.py