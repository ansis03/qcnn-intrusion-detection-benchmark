"""NSL-KDD 전처리 — Hur et al. 2022 방식 (PCA → angle encoding 준비)"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit


COL_NAMES = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'label','difficulty'
]

CAT_COLS = ['protocol_type', 'service', 'flag']


def load_and_preprocess(train_path: str, test_path: str, n_components: int = 6):
    """
    NSL-KDD 로드 → one-hot → 정규화 → PCA → angle encoding 범위로 스케일.
    반환: X_train, X_test, y_train, y_test (numpy array)
    """
    df_train = pd.read_csv(train_path, names=COL_NAMES)
    df_test  = pd.read_csv(test_path,  names=COL_NAMES)

    # 이진 레이블
    df_train['y'] = (df_train['label'] != 'normal').astype(int)
    df_test['y']  = (df_test['label']  != 'normal').astype(int)

    # 범주형 → one-hot (train 기준 컬럼 고정)
    df_all = pd.concat([df_train, df_test], keys=['train','test'])
    df_all = pd.get_dummies(df_all, columns=CAT_COLS)
    df_train = df_all.loc['train']
    df_test  = df_all.loc['test']

    feature_cols = [c for c in df_train.columns if c not in ['label','difficulty','y']]

    X_train = df_train[feature_cols].values.astype(float)
    X_test  = df_test[feature_cols].values.astype(float)
    y_train = df_train['y'].values
    y_test  = df_test['y'].values

    # MinMax 정규화 (train fit → test transform)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # PCA → n_components 차원 (= qubit 수)
    pca = PCA(n_components=n_components, random_state=42)
    X_train = pca.fit_transform(X_train)
    X_test  = pca.transform(X_test)

    # Angle encoding 범위: [-π, π]
    max_val = np.abs(X_train).max()
    X_train = X_train / max_val * np.pi
    X_test  = X_test  / max_val * np.pi

    return X_train, X_test, y_train, y_test


def sample_train(X_train, y_train, n: int, seed: int):
    """N개 stratified 샘플 추출."""
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(sss.split(X_train, y_train))
    return X_train[idx], y_train[idx]
