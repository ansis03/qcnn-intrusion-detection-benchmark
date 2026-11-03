"""Datasets, preprocessing and the evaluation set (paper Sec. III-A and III-D).

Each dataset is reduced to eight features: one-hot encoding of categorical columns,
MinMax scaling fitted on the training pool, PCA to eight components, and scaling to
[-pi, pi] for angle encoding. Raw files go in data/raw/ (see README).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/raw'
DATASETS = ['NSL-KDD', 'CICIDS2017', 'TON_IoT']

NSL_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty']

EVAL_CAP = 400         # evaluation attacks per type (all of them if fewer)
EVAL_BENIGN = 3000     # evaluation benign records: 1,500 for the threshold, 1,500 for the FPR
CALIBRATION = 1500


def _project(x_train, x_test):
    """MinMax scaling and PCA fitted on the training pool, then scaling to [-pi, pi]."""
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train); x_test = scaler.transform(x_test)
    pca = PCA(n_components=8, random_state=42)
    x_train = pca.fit_transform(x_train); x_test = pca.transform(x_test)
    m = np.abs(x_train).max()
    return x_train/m*np.pi, x_test/m*np.pi


def load_nsl():
    """KDDTrain+ is the training pool and KDDTest+ the test set."""
    train = pd.read_csv(RAW/'KDDTrain+.txt', names=NSL_COLUMNS)
    test = pd.read_csv(RAW/'KDDTest+.txt', names=NSL_COLUMNS)
    for df in (train, test):
        df['y'] = (df['label'] != 'normal').astype(int)
    both = pd.get_dummies(pd.concat([train, test], keys=['train', 'test']), columns=['protocol_type', 'service', 'flag'])
    tr, te = both.loc['train'], both.loc['test']
    cols = [c for c in tr.columns if c not in ('label', 'difficulty', 'y')]
    x, xt = _project(tr[cols].values.astype(float), te[cols].values.astype(float))
    return x, xt, tr['y'].values, te['y'].values, train['label'].to_numpy(dtype=str), test['label'].to_numpy(dtype=str)


def load_cicids():
    """The eight MachineLearningCVE CSV files, split 80:20 (stratified by class, seed 42)."""
    data = []
    for f in sorted((RAW/'cicids2017').glob('*.csv')):
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        data.append(df)
    data = pd.concat(data, ignore_index=True)
    # The web-attack labels contain a broken en dash (U+FFFD) in the distributed CSV.
    label = data['Label'].astype(str).str.strip().str.replace('�', '–', regex=False)
    y = (label != 'BENIGN').astype(int)
    cols = [c for c in data.columns if c != 'Label' and pd.api.types.is_numeric_dtype(data[c])]
    x = data[cols].replace([np.inf, -np.inf], np.nan)
    ok = x.notna().all(axis=1).values
    x, y, label = x.values[ok], y.values[ok], label.values[ok]
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    x, xt = _project(x[tr], x[te])
    return x, xt, y[tr], y[te], np.array([str(v) for v in label[tr]]), np.array([str(v) for v in label[te]])


def load_toniot():
    """The train-test network file, deduplicated and split 80:20 (stratified by class, seed 42)."""
    df = pd.read_csv(RAW/'toniot/train_test_network.csv', low_memory=False).drop_duplicates()
    numeric = ['duration', 'src_bytes', 'dst_bytes', 'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes']
    num = df[numeric].astype(float).apply(np.log1p)
    cat = pd.get_dummies(df[['proto', 'service', 'conn_state']].astype(str))
    x = pd.concat([num.reset_index(drop=True), cat.reset_index(drop=True)], axis=1).values.astype(float)
    y, label = df['label'].values.astype(int), df['type'].values
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    x, xt = _project(x[tr], x[te])
    return x, xt, y[tr], y[te], np.array([str(v) for v in label[tr]]), np.array([str(v) for v in label[te]])


LOADERS = {'NSL-KDD': load_nsl, 'CICIDS2017': load_cicids, 'TON_IoT': load_toniot}


def load(ds):
    """Training pool (x, y, types) and test set (xt, yt, tt); y is 1 for attacks."""
    x, xt, y, yt, types, tt = LOADERS[ds]()
    return np.asarray(x, float), np.asarray(xt, float), np.asarray(y), np.asarray(yt), types, tt


def evaluation(xt, yt, tt, y, types):
    """Evaluation set drawn from the test set.

    Up to 400 records of every attack type seen in training and 3,000 benign records
    (seed 0). The benign records are split into 1,500 for the threshold and 1,500 for
    the FPR (seed 20260922). The mixture weights are the type shares of the whole test set.
    """
    rng = np.random.default_rng(0)
    rows = []
    for k in sorted(set(types[y == 1].tolist())):
        idx = np.flatnonzero((yt == 1) & (tt == k))
        take = min(len(idx), EVAL_CAP)
        if take:
            rows.append(rng.choice(idx, take, replace=False) if take < len(idx) else idx)
    rows.append(rng.choice(np.flatnonzero(yt == 0), EVAL_BENIGN, replace=False))
    rows = np.concatenate(rows); rng.shuffle(rows)
    labels, etypes = yt[rows], tt[rows]
    benign = np.random.default_rng(20260922).permutation(np.flatnonzero(labels == 0))
    names = sorted(set(etypes[labels == 1]))
    counts = np.array([np.sum(tt == k) for k in names], dtype=float)
    return {'row_ids': rows, 'x': xt[rows], 'labels': labels, 'types': etypes, 'names': names,
            'calibration_positions': benign[:CALIBRATION], 'test_normal_positions': benign[CALIBRATION:],
            'test_counts': counts.astype(int), 'weights': counts/counts.sum()}
