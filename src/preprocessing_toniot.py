"""
ToN_IoT(Network 서브셋) 전처리.

구조가 CICIDS-2017(CICFlowMeter 78개 연속 통계)보다 NSL-KDD(protocol_type/service/flag
저카디널리티 범주형 + 수치 통계)에 더 가까워서, preprocessing.py 패턴을 따름.

컬럼 처리 방침(dataset_landscape.md §4 참고):
- 드롭: src_ip/dst_ip(비일반화 식별자), src_port/dst_port(포트 암기로 인한 정확도 부풀림 방지,
  CICIDS-2017 Destination-Port 이슈와 동일 우려), type(다중클래스, binary label만 사용),
  dns_query/ssl_subject/ssl_issuer/http_uri/http_user_agent(고카디널리티 자유텍스트),
  나머지 DNS/SSL/HTTP 조건부 필드(62~100% 결측 placeholder "-", service 컬럼과 중복 정보)
- 유지 수치(log1p 변환, 두꺼운 꼬리 분포 대응): duration, src_bytes, dst_bytes, missed_bytes,
  src_pkts, src_ip_bytes, dst_pkts, dst_ip_bytes
- 유지 범주형(원-핫): proto, service, conn_state
- 완전 중복 행(9.7%)은 split 전에 제거 (train/test 리키지 방지)
- 공식 train/test 분할 파일이 없어서 CICIDS-2017과 동일하게 직접 stratified split
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

NUMERIC_COLS = [
    'duration', 'src_bytes', 'dst_bytes', 'missed_bytes',
    'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
]
CAT_COLS = ['proto', 'service', 'conn_state']


def load_and_preprocess(csv_path: str, n_components: int = 8,
                         test_size: float = 0.2, seed: int = 42):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()

    y = df['label'].values.astype(int)

    num = df[NUMERIC_COLS].astype(float).apply(np.log1p)
    cat = pd.get_dummies(df[CAT_COLS].astype(str))

    X = pd.concat([num.reset_index(drop=True), cat.reset_index(drop=True)], axis=1).values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=n_components, random_state=seed)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    max_val = np.abs(X_train).max()
    X_train = X_train / max_val * np.pi
    X_test = X_test / max_val * np.pi

    return X_train, X_test, y_train, y_test
