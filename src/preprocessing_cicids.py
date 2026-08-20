"""CICIDS-2017 전처리 — NSL-KDD(preprocessing.py)와 동일한 파이프라인(PCA → angle encoding).

NSL-KDD와 달리 CICIDS-2017은 train/test가 미리 나뉘어 있지 않으므로,
정제된 전체 풀(experiments/prepare_cicids.py 산출물)에서 stratified train/test 분할을 먼저 수행한다.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def load_and_preprocess(clean_pkl_path: str, n_components: int = 8,
                         test_size: float = 0.2, seed: int = 42):
    """
    정제된 CICIDS pickle(experiments/prepare_cicids.py 산출물) 로드 →
    stratified train/test 분할 → MinMax 정규화(train fit) → PCA(train fit) → angle encoding 스케일.
    반환: X_train, X_test, y_train, y_test (numpy array) — preprocessing.py와 동일 인터페이스.
    """
    data = pd.read_pickle(clean_pkl_path)
    y = data['y'].values
    X = data.drop(columns=['y']).values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    pca = PCA(n_components=n_components, random_state=seed)
    X_train = pca.fit_transform(X_train)
    X_test  = pca.transform(X_test)

    max_val = np.abs(X_train).max()
    X_train = X_train / max_val * np.pi
    X_test  = X_test  / max_val * np.pi

    return X_train, X_test, y_train, y_test
