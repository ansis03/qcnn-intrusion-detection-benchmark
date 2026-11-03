# Results

Policy names in file names and fields: `stratified` = ST, `balance` = CB, `coverage` = MC,
`uniform` = TB. Circuit names follow Hur et al.: `U_TTN` (circuit 1, 12 parameters),
`U_15` (3, 18), `U_SO4` (6, 24), `U_5` (7, 36), `U_SU4` (9, 51).

## main/<dataset>/

One file per model, `<circuit>_N<size>_seed<seed>_<policy>.json`, 500 per dataset:

| Field | Meaning |
|---|---|
| `N`, `seed`, `circuit`, `method` | subset size, seed (42-46), circuit, sampling policy |
| `attack_records`, `selected_types` | attack records and attack types in the training subset |
| `auc_macro` | AUC averaged over attack types (each type against all 3,000 evaluation benign records) |
| `auc_weighted` | the same, weighted by the test attack mixture |
| `recall_macro`, `recall_weighted` | recall at the 1% FPR threshold, averaged over types / weighted by the test mixture |
| `fpr` | FPR on the 1,500 evaluation benign records not used for the threshold |
| `threshold` | the 1,485th smallest of the 1,500 calibration benign scores |
| `per_type` | for each evaluated attack type: `auc`, `recall`, `test_count` (records in the whole test set), `evaluated_count` (evaluation records), `subset_count` (records in the training subset) |

Other files per dataset:

- `models.npz`: `keys` (model keys), `scores` (evaluation scores, one row per model, in the
  order of `evaluation.npz`), `final_params` (trained parameters, padded with NaN to 51).
- `evaluation.npz`: `row_ids` (rows of the test set), `labels`, `types`, `names` (evaluated
  attack types), `calibration_positions` and `test_normal_positions` (the two halves of the
  benign records), `test_counts` and `weights` (test-mixture weights of the types).
- `pool.json`: size of the training pool and its attack records per type.

## checks/

All checks use the subsets, initial parameters and evaluation set of the main experiment.

- `rescore/<dataset>/`: circuits 1 and 9, CB, MC and TB, every N and seed, rescored without
  retraining. `test0.001` and `test0.01` set thresholds on half of a larger test-side benign
  pool (NSL-KDD: all 9,711 test benign records; CICIDS2017: 20,000) and measure the FPR on the
  other half. `train0.01` sets the 1% threshold on 10,000 training-pool benign records.
- `long/<dataset>/`: the N = 500 models of circuits 1 and 9 retrained with 1,000 iterations.
- `cnn/<dataset>/`: a one-layer CNN (`src/classical.py`; Adam, 60 epochs, five
  initializations `CNN0`-`CNN4`) trained on the CB, MC and TB subsets.
- `fullpool/<dataset>/`: the same CNN trained on the entire training pool, five initializations.

## replication/summary.csv

The earlier unpaired experiment (paper Sec. IV-A): ST, CB and TB subsets drawn independently,
five circuits, five subset sizes and five seeds per dataset (1,125 models). The file lists
the summary values of the type-averaged and mixture-weighted AUC and recall, and the
achieved test FPR, at target FPRs of 1% and 5%. Its training code is not part of this release.
