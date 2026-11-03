"""QCNN training and scoring (paper Sec. III-C), using the circuits of Hur et al. in QCNN/."""
import sys
import numpy as np
import pennylane as qml
from pennylane import numpy as anp
from sklearn.metrics import roc_auc_score
from .data import ROOT

sys.path.insert(0, str(ROOT/'QCNN'))
import QCNN_circuit as circuit  # noqa: E402

# Convolutional circuits 1, 3, 6, 7 and 9 of Hur et al. and the parameters of one block.
# A QCNN has 3*block + 6 trainable parameters: 12, 18, 24, 36 and 51.
CIRCUITS = {'U_TTN': 2, 'U_15': 4, 'U_SO4': 6, 'U_5': 10, 'U_SU4': 15}
ITERATIONS, BATCH, LR, MOMENTUM = 200, 25, 0.01, 0.9


def initial_params(name, seed):
    """Standard normal initial parameters; the same seed for every policy."""
    return np.random.default_rng(100000 + seed).normal(size=3*CIRCUITS[name] + 6)


def batch_positions(n, seed, iterations=ITERATIONS, base=200000):
    """Mini-batch positions (iterations x 25, drawn with replacement), shared by every
    policy and circuit, so records common to two subsets are drawn in the same mini-batches."""
    return np.random.default_rng(base + seed + 1000*n).integers(0, n, size=(iterations, BATCH))


def scores(x, params, name):
    """Pauli-Z expectation of the readout qubit, in [-1, 1]; +1 means attack."""
    return np.concatenate([np.asarray(circuit.QCNN(anp.array(c, requires_grad=False), params, name, CIRCUITS[name], 'Angle',
                                                   cost_fn='mse')) for c in np.array_split(x, max(1, (len(x) + 255)//256))])


def train(x, y, idx, positions, init, name):
    """Nesterov momentum on the mean squared error, targets +1 (attack) and -1 (benign)."""
    size = CIRCUITS[name]
    xn = anp.array(x[idx], requires_grad=False); yn = anp.array(2*y[idx] - 1, requires_grad=False)
    p = anp.array(np.array(init, copy=True), requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(LR, momentum=MOMENTUM); losses = []
    for b in positions:
        xb, yb = xn[b], yn[b]
        p, cost = opt.step_and_cost(lambda v: anp.mean((circuit.QCNN(xb, v, name, size, 'Angle', cost_fn='mse') - yb)**2), p)
        losses.append(float(cost))
    return np.asarray(p), losses


def threshold(benign_scores, target=0.01):
    """Threshold for a target FPR on held-out benign scores: with 1,500 scores and a 1% target,
    the 1,485th smallest score, so at most 15 of them lie above it."""
    z = np.sort(benign_scores); m = len(z)
    return float(z[m - int(np.floor(target*m + 1e-9)) - 1])


def evaluate(s, ev, types, idx):
    """Per-type AUC (against all 3,000 evaluation benign records) and recall at the 1% FPR threshold,
    their type average (macro) and test-mixture weighted average, and the FPR on the other 1,500 benign records."""
    lab, ety = ev['labels'], ev['types']
    benign = np.flatnonzero(lab == 0)
    thr = threshold(s[ev['calibration_positions']])
    per = []
    for k, count in zip(ev['names'], ev['test_counts']):
        a = np.flatnonzero(ety == k); i = np.r_[benign, a]
        per.append({'type': k, 'auc': float(roc_auc_score(lab[i], s[i])), 'recall': float(np.mean(s[a] > thr)),
                    'test_count': int(count), 'evaluated_count': int(len(a)), 'subset_count': int(np.sum(types[idx] == k))})
    auc = np.array([r['auc'] for r in per]); rec = np.array([r['recall'] for r in per]); w = ev['weights']
    return {'auc_macro': float(auc.mean()), 'auc_weighted': float(w @ auc),
            'recall_macro': float(rec.mean()), 'recall_weighted': float(w @ rec),
            'fpr': float(np.mean(s[ev['test_normal_positions']] > thr)), 'threshold': thr, 'per_type': per}
