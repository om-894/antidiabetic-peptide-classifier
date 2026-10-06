# Phase 7: second-draft revision analyses

Everything added for the second draft of the JCIM manuscript, in response to the
supervisor's ten comments and three priorities. Nothing in phases 1 to 6 was
changed. The published model, the published split and the 412 discoveries are
all untouched; this phase measures them rather than replacing them.

## What each script answers

| Script | Question | Outputs |
|---|---|---|
| `phase7.1_audit_overlap.py` | How close is the test set to the training set, given that 870 sequences bypassed clustering? | `results/phase7_1_overlap_*` |
| `phase7.2_audit_positives.py` | How redundant is the 966-sequence positive class? | `results/phase7_2_positives_*` |
| `phase7.3_audit_folds.py` | Does cluster-level leakage across the 5 CV folds inflate the out-of-fold calibration the threshold was set on? | `results/phase7_3_folds_*` |
| `phase7.4_audit_discovery.py` | How far is each discovery from the nearest training positive? | `results/phase7_4_discovery_*` |
| `phase7.5_audit_seeds.py` | Is the 0.877 the best seed or the mean, and do frozen features remove the seed sensitivity? | `results/phase7_5_seeds_summary.txt` |
| `phase7.6_ablation/` | Can negative-class composition be separated from length and charge? | `results/phase7_6_*` |
| `phase5.8_seed_screen.py` | What does the screen return under each of the five ESM-2 seeds? | `screening/screening_esm2_seeds.npz` |
| `phase7.7_seed_stability.py` | How much of the shortlist survives a change of seed? | `results/phase7_6_seed_stability.*` |

`phase5.8_seed_screen.py` keeps its phase-5 number because it is a screening
pass rather than an audit, and because the Viking staging commands already refer
to it by that name. It lives here only so the revision is reviewable in one
place.

## Running order

The audits are independent and can run in any order. The ablation is a pipeline:

```bash
python phase7.6_ablation/phase7.6.1_build_arms.py   # fetches Swiss-Prot once, then caches
python phase7.6_ablation/phase7.6.2_embed.py        # ~2,750 sequences, MPS or CUDA
python phase7.6_ablation/phase7.6.3_fit.py          # 80 XGBoost fits, a few minutes
python phase7.6_ablation/phase7.6.4_family_split.py # needs 7.6.3's results file
```

The seed screen runs on Viking (see `viking_runbook.md` for the environment) and
`phase7.7_seed_stability.py` consumes its output locally.

## Two things that will bite a rerun

**Use the framework interpreter, not pyenv.** `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`
carries xgboost 3.1.2, which is what fitted `models/base_xgb.joblib`. The pyenv
3.12.3 install has xgboost 3.3.0 and gives a different fit from identical data
and seeds. Mixing them is how the first draft ended up with a point estimate and
a seed column from two different libraries.

**Single fits cannot compare the arms.** `subsample=0.8` draws training rows in
the order given, so permuting them moves test AUC by about 0.004, which is the
size of the smallest difference between arms. Every arm is therefore reported as
the mean over ten permutations with its standard deviation. A single fit of arm A
reads 0.857 or 0.862 depending on row order alone.

## Reproducibility notes

`work/` holds regenerable intermediates and is not needed to read the results.
Two of them are worth keeping rather than regenerating:

- `swissprot_pool.txt` is the GO-filtered Swiss-Prot draw behind arms E and F.
  UniProt changes over time, so a fresh fetch will not reproduce these arms
  exactly. The cached pool is what the manuscript's numbers come from.
- `ablation_embeddings.npz` is 12.8 MB of frozen ESM-2 vectors for the sequences
  the arms add, extracted with the same encoder and pooling as phase 2.1. It
  saves a GPU pass on rerun.

`manuscript/` holds the scripts that built the second-draft `.docx` files from
the first-draft ones. They operate on documents outside this repository and are
included for provenance rather than for rerunning. They are not part of the
analysis pipeline.

## Where these land in the paper

Table 5 and Tables S11 to S14 are generated from `results/phase7_*`, so the
manuscript cannot drift from the analysis. Section 2.1, Section 2.3, Section
2.5, Section 3.1, Section 3.6, Section 4.1 and Section 4.3 all carry numbers
from this phase.
