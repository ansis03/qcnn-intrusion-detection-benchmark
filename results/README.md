# Results schema

Every `<dataset>/<ansatz>/results_final.json` has the same shape: a JSON list
where each element is one training run (one `(N, rep)` pair). Example
(trimmed):

```json
{
  "N": 50,
  "rep": 1,
  "seed": 42,
  "QCNN": {
    "accuracy": 0.574, "f1": 0.288, "precision": 0.878, "recall": 0.172,
    "train_accuracy": 0.84, "train_f1": 0.789,
    "gen_gap_acc": 0.266, "gen_gap_f1": 0.502,
    "final_loss": 0.484,
    "margin": { "mean": 0.302, "min": -0.286, "std": 0.281,
                "negative_ratio": 0.16,
                "hist_counts": [...], "hist_edges": [...] },
    "param_file": "N50_rep1_seed42.npy"
  },
  "CNN": { "accuracy": 0.755, "f1": 0.728, "precision": 0.988, "recall": 0.576 },
  "SVM": { "accuracy": 0.776, "f1": 0.759, "precision": 0.984, "recall": 0.617 }
}
```

| Field | Meaning |
|---|---|
| `N` | Training-set size for this run (one of the 5 swept values, or 10000/17534 in a `*_largeN` folder) |
| `rep` | Repeat index, 1–5. `seed = 41 + rep`, i.e. 42–46 — same 5 seeds across every ansatz/dataset so runs are comparable |
| `QCNN` / `CNN` / `SVM` | Metrics for each model, evaluated on the held-out test set (binary classification, positive class = Attack). **Note:** QCNN is evaluated on a fixed balanced 1,000-sample test subset, while CNN/SVM use the full test set — see paper Section IV.D/V.A for why, and a robustness check confirming this doesn't drive the reported gap |
| `accuracy` / `f1` / `precision` / `recall` | Standard binary-classification metrics |
| `train_accuracy` / `train_f1` (QCNN only) | Same metrics, computed on the training set instead of the test set |
| `gen_gap_acc` / `gen_gap_f1` (QCNN only) | Train − test gap (generalization gap, paper Section V.D) |
| `final_loss` (QCNN only) | MSE training loss at the last optimization step |
| `margin` (QCNN only) | Training-set output-margin diagnostics, $y \cdot f(x) \in [-1, 1]$ (paper Section V.D): `mean`/`min`/`std`, `negative_ratio` (fraction of misclassified training points), and a 10-bin histogram (`hist_counts` over `hist_edges`) |
| `param_file` (QCNN only) | Filename the trained circuit parameters were saved under when this run originally executed. Not included in this repo — see the main [README](../README.md#whats-in-this-repo) note on why, and how to regenerate them |

`CNN`/`SVM` have no training-set or margin fields — they're classical models
without the QCNN-specific diagnostics used in the paper's analysis.
