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
results_*/              Full experiment outputs — results_final.json (metrics per
                        run) + params/*.npy (trained QCNN parameters) for every
                        (dataset, ansatz, N) combination reported in the paper
figures/                Final PDF/PNG figures used in the paper
gen_figures.py          Regenerates all figures from results_*/
```

Naming convention: `results_ansatz_*` = NSL-KDD ansatz sweep, `results_cicids_*`
/ `results_toniot_*` = the same sweep on the other two datasets, `*_largeN` =
the N ∈ {10,000, 17,534} extension (U_TTN and U_5 only, all 3 datasets).

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
python experiments/run_experiment.py --ansatz U_15 --u_params 4 --results_dir results_ansatz_u15
python experiments/run_experiment_cicids.py --ansatz U_5 --u_params 10 --results_dir results_cicids_u5
python experiments/run_experiment_toniot.py --ansatz U_SU4 --u_params 15 --results_dir results_toniot_usu4
```

`--ansatz`/`--u_params` pairs used in the paper: `U_TTN`/2, `U_15`/4,
`U_SO4`/6, `U_5`/10, `U_SU4`/15 (see paper Table 0 / `QCNN/unitary.py`).

Every run trains QCNN, 1D-CNN, and SVM on identical data splits/seeds and
writes `results_final.json` (+ `params/*.npy` per run) to `--results_dir`.
The `results_*/` folders already in this repo are the exact outputs the
paper's tables and figures were generated from — you don't need to rerun
anything to inspect or reuse them.

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
