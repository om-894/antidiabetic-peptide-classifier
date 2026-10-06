#!/usr/bin/env bash
#SBATCH --job-name=esm2_seed_screen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:30:00  # five forward passes over 3,596 candidates, no training
#SBATCH --account=chem-data-2023  # REQUIRED, job won't run without it
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=om894@york.ac.uk
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err

# Selection-stability run behind the five-seed shortlist. Scores the screening
# pool with each of the five DoRA adapters from the phase 3.3 seed refits, so
# the 412 discoveries can be reported as a selection frequency across seeds
# rather than as the seed-42 fit's output.
#
# Nothing is trained and no embeddings are recomputed, which is why the walltime
# is short beside the fine-tuning jobs. The job fails immediately with the list
# of missing adapters if any seed refit was never run or was cleaned up.
#
#   sbatch phase5.8_seed_screen.sbatch
#   SEEDS=43,44 sbatch phase5.8_seed_screen.sbatch   # to fill in a subset
#
# Expects the flat ~/adp/ layout from viking_runbook.md, not the repo tree, with
# screening_candidates.csv in ~/adp/screening/ from the phase 5.2 run and the
# per-seed adapters in ~/adp/models/.

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
python phase5.8_seed_screen.py
