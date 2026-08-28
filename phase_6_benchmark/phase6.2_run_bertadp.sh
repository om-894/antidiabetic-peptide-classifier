#!/usr/bin/env bash
# Phase 6.2: run BertADP (Xie et al. 2025) on an input CSV.
#
# Automates the external-benchmark inference. Clones the BertADP repo, builds an
# isolated venv at BertADP's pinned versions so it cannot clash with the project
# environment, runs their released BertADP.py and copies the result into benchmark/.
# The clone and the venv are built once, so later calls only run inference.
#
# INPUTS  a CSV of sequences to score, written by phase 6.1
# OUTPUTS  benchmark/<input>_pred.csv (Sequence, Positive_Probability, Prediction)
# REQUIREMENTS  git, python3 and network access. Clones Xie's repo to ~/BertADP and
#               builds a venv at ~/bertadp_env, both outside this project. ProtBert
#               downloads to the HuggingFace cache on the first run
# Usage  bash phase_6_benchmark/phase6.2_run_bertadp.sh benchmark/bertadp_hardneg_test.csv

set -e

INPUT="${1:?usage: bash phase6.2_run_bertadp.sh <input_csv>}"
PROJECT="$(pwd)"
REPO="$HOME/BertADP"
VENV="$HOME/bertadp_env"

# 1. clone BertADP once (trained model + released inference script)
if [ ! -d "$REPO" ]; then
    git clone https://github.com/xiexq007/BertADP.git "$REPO"
fi

# 2. build the isolated venv once, pinned to BertADP's requirements
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q pandas==2.2.3 numpy==1.26.4 torch==2.5.0 \
        datasets==3.0.1 transformers==4.44.0 peft==0.15.1 accelerate
fi

# 3. run BertADP (writes prediction_result.csv in the repo dir; ProtBert downloads on first run)
cd "$REPO"
"$VENV/bin/python" BertADP.py "$PROJECT/$INPUT"

# 4. copy the result back, named after the input
OUT="$PROJECT/benchmark/$(basename "${INPUT%.csv}")_pred.csv"
cp "$REPO/prediction_result.csv" "$OUT"
echo "saved -> $OUT"