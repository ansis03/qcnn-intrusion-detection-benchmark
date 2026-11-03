"""Read the released results of the main experiment (results/main/<dataset>/)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT/'results/main'
DS = ['NSL-KDD', 'CICIDS2017', 'TON_IoT']
POLICIES = ['stratified', 'balance', 'coverage', 'uniform']
SHORT = {'stratified': 'ST', 'balance': 'CB', 'coverage': 'MC', 'uniform': 'TB'}
NAME = {'stratified': 'Stratified', 'balance': 'Class-balanced', 'coverage': 'Minimum coverage', 'uniform': 'Type-balanced'}
CIRCUITS = {'U_TTN': (1, 12), 'U_15': (3, 18), 'U_SO4': (6, 24), 'U_5': (7, 36), 'U_SU4': (9, 51)}  # Hur et al. number, parameters
NS = [50, 100, 500, 1000, 5000]
METRICS = ('auc_macro', 'auc_weighted', 'recall_macro', 'recall_weighted', 'fpr')


def _json(ds):
    return [json.loads(q.read_text(encoding='utf-8')) for q in sorted((MAIN/ds).glob('U_*.json'))]


def runs(ds):
    """One row per model."""
    return pd.DataFrame([{k: r[k] for k in ('key', 'N', 'circuit', 'seed', 'method', *METRICS)} for r in _json(ds)])


def per_type(ds):
    """One row per model and evaluated attack type."""
    return pd.DataFrame([{'N': r['N'], 'circuit': r['circuit'], 'seed': r['seed'], 'method': r['method'], **t}
                         for r in _json(ds) for t in r['per_type']])


def scores(ds):
    """Evaluation scores of every model, keyed by model key."""
    z = np.load(MAIN/ds/'models.npz')
    return dict(zip(z['keys'].tolist(), z['scores']))


def evaluation(ds):
    return dict(np.load(MAIN/ds/'evaluation.npz'))


def pool_shares(ds):
    """Share of each attack type among the training-pool attacks."""
    c = json.loads((MAIN/ds/'pool.json').read_text(encoding='utf-8'))['attack_types']
    total = sum(c.values())
    return {k: v/total for k, v in c.items()}
