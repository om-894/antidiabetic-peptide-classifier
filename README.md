# Hybrid deep learning and PLM embeddings for anti-diabetic peptide discovery

MSc Data Science project, Oliver McQuillan, 2025/26.

Scripts are numbered by pipeline phase and run in that order. Every `.py` script
carries a header docstring giving its inputs, outputs and requirements. Paths are
relative to this folder, so run from the repo root. Python 3.12.3 locally, 3.11.3
on Viking.

## Exceptions to running in order

- **Phases 6.1 to 6.3 come before 4.1.** `phase4.1_statistical_evaluation.py`
  reads `results/phase6_3_benchmark_tests.csv` so all five significance tests are
  Holm-corrected as one family, and raises FileNotFoundError without it.
- **`phase3.5` and `phase4.2` run as two processes each**, `MODELS=xgb,rf` then
  `MODELS=cnn`. torch and xgboost each bundle their own libomp and segfault when
  imported together on macOS.
- **Four steps run on Viking**: `phase3.3` (main fine-tune, and again per seed for
  42 to 46), `phase3.6.3` and `phase5.2`. See `viking_runbook.md`.
- **Two steps are web submissions.** AlgPred 2.0 results are saved to
  `screening/algpred2_safe.csv`, HADDOCK 2.4 archives to `docking/haddock_runs/`.
- `figures.ipynb` runs last, writing `figures/` plus
  `results/figures_nearest_negative.csv` and `results/figures_charge_by_class.csv`.

## External tools

cd-hit v4.8.1 on PATH, called by phases 1.3, 3.6.2 and 5.1. ToxinPred2 as a CLI,
invoked from `phase5.4`. AlgPred 2.0 and HADDOCK 2.4 on their web servers.
BertADP, cloned and run into its own venv by `phase6.2_run_bertadp.sh`.
`requirements.txt` is a full freeze of the local environment.

## Layout

`data/` released dataset, built negatives and split. `predictions/` out-of-fold
and test probabilities per learner. `models/` fitted learners and the DoRA
adapter. `screening/` candidate pool, scores, safety calls, tiered shortlist.
`docking/` receptor, peptide ensembles, restraints, HADDOCK archives.
`benchmark/` BertADP inputs and predictions. `results/` every table the report
quotes. `figures/` exported figures.

## Declaration of AI use in code

Generative AI was used in producing approximately 10% of the code in this
repository. Following the module guidance, this is a general account rather
than a line-by-line one.

It was used to draft certain functions and to solve specific problems within
the project. These include improving the efficiency of individual functions
and working around issues I could not resolve from documentation or Stack
Overflow. Claude was the tool used.

All code submitted here is my responsibility and I have read and understood
every part of it.