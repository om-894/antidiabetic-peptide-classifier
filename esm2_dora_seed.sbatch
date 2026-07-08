#!/usr/bin/env bash
#SBATCH --job-name=esm2_seed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --account=chem-data-2023
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=om894@york.ac.uk
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err

set -e
module purge
module load Python/3.11.3-GCCcore-12.3.0
source ~/esm2env/bin/activate

export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export DATA=data/dataset_split.csv
export SKIP_OOF=1
export OUT_NPZ=predictions/esm2_dora_seed${SEED}_predictions.npz
export OUT_ADAPTER=models/esm2_seed${SEED}_adapter

nvidia-smi
python phase3.3_esm2_dora.py