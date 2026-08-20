"""
CICIDS-2017 원본 CSV(요일별 8개 파일) 결합 + 정제.

- 컬럼명 공백 제거
- Label 이진화: BENIGN=0, 그 외(공격)=1  (NSL-KDD의 label!=normal 관례와 동일)
- Flow Bytes/s, Flow Packets/s 등에서 발생하는 Infinity/NaN 제거
  (0-duration flow에서 나누기 0으로 발생 — CICIDS의 알려진 이슈)
- 결과를 code/data/processed/cicids_clean.pkl 로 저장 (재실행 시 8개 CSV 재파싱 불필요)

사용법: python experiments/prepare_cicids.py
"""

import os, glob, time
import numpy as np
import pandas as pd

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE, 'data', 'raw-cicids', 'extracted')
OUT_DIR    = os.path.join(BASE, 'data', 'processed')
OUT_PATH   = os.path.join(OUT_DIR, 'cicids_clean.pkl')

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, '*.csv')))
    print(f"발견된 CSV: {len(csv_files)}개")
    for f in csv_files:
        print(f"  {os.path.basename(f)}")

    dfs = []
    t0 = time.time()
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df)
        print(f"  로드: {os.path.basename(f)}  {df.shape}  ({time.time()-t0:.0f}s)")

    data = pd.concat(dfs, ignore_index=True)
    del dfs
    print(f"\n결합 후: {data.shape}")

    # Label 정제 + 이진화
    data['Label'] = data['Label'].astype(str).str.strip()
    print("\n원본 Label 분포:")
    print(data['Label'].value_counts())

    data['y'] = (data['Label'] != 'BENIGN').astype(int)

    # 특징 컬럼: Label 제외 전부 (전부 수치형이어야 함)
    feature_cols = [c for c in data.columns if c not in ('Label', 'y')]
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(data[c])]
    if non_numeric:
        print(f"\n⚠ 비수치형 컬럼 발견, 제거: {non_numeric}")
        feature_cols = [c for c in feature_cols if c not in non_numeric]

    X = data[feature_cols].copy()
    y = data['y'].copy()

    # Infinity → NaN → 해당 행 제거
    n_before = len(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    mask_valid = X.notna().all(axis=1)
    n_invalid = (~mask_valid).sum()
    print(f"\nInfinity/NaN 포함 행: {n_invalid} / {n_before} ({n_invalid/n_before:.2%}) → 제거")

    X = X[mask_valid].reset_index(drop=True)
    y = y[mask_valid].reset_index(drop=True)

    print(f"\n최종: X={X.shape}, attack ratio={y.mean():.1%}")

    clean = X.copy()
    clean['y'] = y
    clean.to_pickle(OUT_PATH)
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
