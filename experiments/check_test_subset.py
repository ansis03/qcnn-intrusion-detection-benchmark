"""
Table I의 QCNN(balanced 1,000샘플)과 CNN/SVM(전체 22,544샘플) 테스트셋 불일치가
결과에 영향을 주는지 확인: CNN/SVM을 원래와 동일한 학습 데이터로 재학습하되,
QCNN과 '같은' balanced 1,000샘플 서브셋에서 평가해서 F1 격차가 유지되는지 본다.
사용법: python experiments/check_test_subset.py
"""
import os, sys, json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'src'))

from preprocessing import load_and_preprocess, sample_train
from evaluate import compute_metrics

N_QUBITS   = 8
N_SIZES    = [50, 100, 500, 1000, 5000]
N_REPEATS  = 5
QCNN_EVAL  = 1000
# NSL-KDD 원본은 리포에 포함하지 않음 — README 안내에 따라 data/raw/ 에 직접 다운로드
TRAIN_PATH = os.path.join(BASE, 'data', 'raw', 'KDDTrain+.txt')
TEST_PATH  = os.path.join(BASE, 'data', 'raw', 'KDDTest+.txt')


def train_cnn(X_tr, y_tr, X_te, n_epochs=100, batch_size=32, lr=0.001):
    import torch, torch.nn as nn
    from classical import CNN1D
    model = CNN1D(input_dim=N_QUBITS, n_filters=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    X_t = torch.FloatTensor(X_tr); y_t = torch.FloatTensor(y_tr)
    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_t[idx]), y_t[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = (model(torch.FloatTensor(X_te)) > 0).numpy().astype(int)
    return preds


def train_svm(X_tr, y_tr, X_te):
    from classical import build_svm
    svm = build_svm()
    svm.fit(X_tr, y_tr)
    return svm.predict(X_te)


def main():
    X_train, X_test, y_train, y_test = load_and_preprocess(TRAIN_PATH, TEST_PATH, n_components=N_QUBITS)

    rng = np.random.default_rng(0)
    idx0 = np.where(y_test == 0)[0]
    idx1 = np.where(y_test == 1)[0]
    half = QCNN_EVAL // 2
    qcnn_test_idx = np.concatenate([
        rng.choice(idx0, min(half, len(idx0)), replace=False),
        rng.choice(idx1, min(half, len(idx1)), replace=False),
    ])
    X_qte = X_test[qcnn_test_idx]
    y_qte = y_test[qcnn_test_idx]
    print(f"QCNN balanced subset: {len(y_qte)} samples, attack ratio={y_qte.mean():.1%}")
    print(f"Full KDDTest+: {len(y_test)} samples, attack ratio={y_test.mean():.1%}")

    results = {}
    for N in N_SIZES:
        cnn_f1s, svm_f1s = [], []
        for rep in range(N_REPEATS):
            seed = 42 + rep
            Xn, yn = sample_train(X_train, y_train, N, seed)

            y_pred_cnn = train_cnn(Xn, yn, X_qte)
            m_cnn = compute_metrics(y_qte, y_pred_cnn)
            cnn_f1s.append(m_cnn['f1'])

            y_pred_svm = train_svm(Xn, yn, X_qte)
            m_svm = compute_metrics(y_qte, y_pred_svm)
            svm_f1s.append(m_svm['f1'])

            print(f"  N={N:5d} rep={rep+1}  CNN f1={m_cnn['f1']:.3f}  SVM f1={m_svm['f1']:.3f}")

        results[N] = {
            'CNN': (float(np.mean(cnn_f1s)), float(np.std(cnn_f1s))),
            'SVM': (float(np.mean(svm_f1s)), float(np.std(svm_f1s))),
        }

    out_path = os.path.join(BASE, 'results', 'nsl-kdd', 'uttn', 'test_subset_check.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n=== N | CNN (on balanced 1k subset) | SVM (on balanced 1k subset) ===")
    for N in N_SIZES:
        c = results[N]['CNN']; s = results[N]['SVM']
        print(f"  N={N:5d}  CNN: {c[0]:.3f} ± {c[1]:.3f}   SVM: {s[0]:.3f} ± {s[1]:.3f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
