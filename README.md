# QCNN Data-Efficiency Benchmark for Network Intrusion Detection

Code and results for:

> **Ansatz Expressivity and Dataset-Dependent Scaling of QCNN Competitiveness in
> Network Intrusion Detection**
> Submitted to ICAIIC 2027 (draft — not yet accepted; camera-ready details TBD)

A systematic benchmark of Quantum Convolutional Neural Networks (QCNNs) for
network intrusion detection: 5 ansätze (12–51 trainable parameters) × 5
training sizes (N = 50–5,000) × 3 datasets (NSL-KDD, ToN_IoT, CICIDS-2017) —
375 QCNN training runs, plus a 60-run large-N extension (N up to 17,534) —
compared against classical 1D-CNN and SVM baselines under matched conditions.

## What's in this repo

```
QCNN/                 Vendored circuit implementation (Apache 2.0, see QCNN/NOTICE.md)
src/                   Preprocessing (per dataset), classical baselines, metrics
experiments/           Entry-point scripts (one per dataset + a CICIDS prep step)
config/config.yaml     Reference listing of experiment settings (not auto-loaded —
                        the scripts below hardcode the same values; kept for
                        documentation / quick lookup)
results/<dataset>/<ansatz>/results_final.json
                        every metric (F1, accuracy, margins, etc.) for every
                        run, for every (dataset, ansatz, N) combination
                        reported in the paper — field-by-field meaning in
                        results/README.md
figures/                PNG versions of the paper's figures (vector PDFs used in
                        the actual submission live in the paper's own repo,
                        not here)
gen_figures.py          Regenerates all figures from results/
```

`<dataset>` is one of `nsl-kdd`, `cicids`, `toniot`; `<ansatz>` is one of
`uttn`, `u15`, `uso4`, `u5`, `usu4` (Table 0's ansätze, lowercased), plus
`uttn_largeN` / `u5_largeN` for the N ∈ {10,000, 17,534} extension (only
these two ansätze were re-run at large N, across all 3 datasets) and two
one-off diagnostic folders, each with just 1 record instead of 25:
- `cicids/sanity`, `toniot/sanity` — a single U_TTN run at N=5,000, done
  before committing to the full 5-ansatz × 5-N × 5-rep sweep on each newly
  added dataset, just to check QCNN/CNN/SVM all reach a sane F1 first
- `cicids/u5_steps1000` — U_5 re-run with 5x the training steps (1,000 vs.
  200), the targeted check in paper Section VI.D for whether CICIDS-2017's
  underfitting is just undertraining (it isn't — see the paper)

Note: trained QCNN circuit parameters (`params/*.npy`) and intermediate
checkpoints (`results_partial.json`) are produced locally when you run an
experiment but aren't tracked here — `results_final.json` already carries
every metric used anywhere in the paper, and the parameters themselves are
exactly reproducible by rerunning with the documented seeds, so versioning
them added ~400 small files without adding anything verifiable.

## Quick start

```bash
pip install -r requirements.txt
```

### 1. Get the data (not included in this repo — see licensing note below)

| Dataset | Where | Put it at |
|---|---|---|
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html | `data/raw/KDDTrain+.txt`, `data/raw/KDDTest+.txt` |
| CICIDS-2017 | https://www.unb.ca/cic/datasets/ids-2017.html (8 daily CSVs) | `data/raw-cicids/extracted/*.csv` |
| ToN_IoT (Network subset) | Alsaedi et al. 2020, IEEE Access — see paper for the dataset citation; hosted by UNSW Canberra Cyber | `data/raw/toniot/train_test_network.csv` |

CICIDS-2017 needs one extra step before it can be used (concatenates the 8
files, binarizes labels, drops rows with Infinity/NaN from zero-duration
flows — see paper Section IV.C for exactly what this does):

```bash
python experiments/prepare_cicids.py   # writes data/processed/cicids_clean.pkl
```

### 2. Reproduce an experiment

```bash
# NSL-KDD, minimal ansatz (U_TTN, 12 params) — this is Table I / Fig. 1
python experiments/run_experiment.py

# A different ansatz / dataset / results folder
python experiments/run_experiment.py --ansatz U_15 --u_params 4 --results_dir results/nsl-kdd/u15
python experiments/run_experiment_cicids.py --ansatz U_5 --u_params 10 --results_dir results/cicids/u5
python experiments/run_experiment_toniot.py --ansatz U_SU4 --u_params 15 --results_dir results/toniot/usu4
```

`--ansatz`/`--u_params` pairs used in the paper: `U_TTN`/2, `U_15`/4,
`U_SO4`/6, `U_5`/10, `U_SU4`/15 (see paper Table 0 / `QCNN/unitary.py`).

Every run trains QCNN, 1D-CNN, and SVM on identical data splits/seeds and
writes `results_final.json` (+ the trained QCNN parameters, as `params/*.npy`,
gitignored locally) to `--results_dir`. The `results_final.json` files already
in this repo are the exact outputs the paper's tables and figures were
generated from — you don't need to rerun anything to inspect or reuse them.

### 3. Regenerate figures

```bash
python gen_figures.py
```

## Data licensing note

Raw and processed dataset files are **not** committed to this repo (see
`.gitignore`) — NSL-KDD and CICIDS-2017 are redistributed by their own
official hosts under their own terms, and re-hosting a copy here isn't
necessary; ToN_IoT is likewise left to the official source rather than
bundled. Download instructions above; `data/raw/` and `data/processed/` are
gitignored so this stays true locally too.

## Vendored code

`QCNN/` contains 4 files (`QCNN_circuit.py`, `Training.py`, `embedding.py`,
`unitary.py`) taken unmodified from Hur, Kim, and Park's public QCNN
implementation (https://github.com/takh04/QCNN, Apache License 2.0). See
`QCNN/NOTICE.md` for details. Everything else in this repository is
original and licensed under the top-level `LICENSE` (MIT).

## Citation

If this is useful, please cite the paper (details will be filled in once
accepted) and, for the QCNN circuit implementation, Hur et al.:

```bibtex
@article{hur2022quantum,
  title   = {Quantum convolutional neural network for classical data classification},
  author  = {Hur, Tak and Kim, Leeseok and Park, Daniel K},
  journal = {Quantum Machine Intelligence},
  volume  = {4},
  pages   = {3},
  year    = {2022}
}
```
