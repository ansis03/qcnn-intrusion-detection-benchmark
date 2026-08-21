"""
메인 실험 스크립트
사용법: python experiments/run_experiment.py
        python experiments/run_experiment.py --n_sizes 50 100 500
        python experiments/run_experiment.py --steps 50  (quick test)
        python experiments/run_experiment.py --ansatz U_15 --u_params 4 --results_dir results/nsl-kdd/u15
"""

import argparse, sys, os, json, time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'src'))
sys.path.insert(0, os.path.join(BASE, 'QCNN', 'QCNN'))

from preprocessing import load_and_preprocess, sample_train
from evaluate import compute_metrics, save_result
import QCNN_circuit
import Training as QCNN_Training


# ── Config ────────────────────────────────────────────────────────────────────
N_QUBITS    = 8
N_SIZES     = [50, 100, 500, 1000, 5000]
N_REPEATS   = 5
STEPS       = 200
BATCH_SIZE  = 25
QCNN_EVAL   = 1000
ANSATZ           = 'U_TTN'  # 기본값 — 하위 호환 유지
U_PARAMS         = 2
RESULTS_DIR_NAME = 'results/nsl-kdd/uttn'
# NSL-KDD 원본은 리포에 포함하지 않음 — README 안내에 따라 data/raw/ 에 직접 다운로드
TRAIN_PATH  = os.path.join(BASE, 'data', 'raw', 'KDDTrain+.txt')
TEST_PATH   = os.path.join(BASE, 'data', 'raw', 'KDDTest+.txt')


# ── QCNN 예측 ─────────────────────────────────────────────────────────────────
def qcnn_predict(X, params, ansatz=ANSATZ, u_params=U_PARAMS, threshold=0.0):
    preds = [QCNN_circuit.QCNN(x, params, ansatz, u_params, 'Angle', cost_fn='mse') for x in X]
    return np.array([1 if float(p) > threshold else 0 for p in preds])


def qcnn_raw_outputs(X, params, ansatz=ANSATZ, u_params=U_PARAMS):
    """예측값 raw 출력 반환 (margin 계산용, 범위 [-1, 1])."""
    return np.array([float(QCNN_circuit.QCNN(x, params, ansatz, u_params, 'Angle', cost_fn='mse'))
                     for x in X])


def compute_margins(X_tr, y_tr_pm, params, ansatz=ANSATZ, u_params=U_PARAMS):
    """
    마진 = y_i * f(x_i), y_i in {+1,-1}, f(x_i) in [-1,1].
    양수: 올바른 예측 (클수록 확신), 음수: 오분류.
    반환: dict (mean, min, std, negative_ratio, histogram)
    """
    outputs = qcnn_raw_outputs(X_tr, params, ansatz=ansatz, u_params=u_params)
    margins = np.array(y_tr_pm, dtype=float) * outputs
    hist, edges = np.histogram(margins, bins=10, range=(-1.0, 1.0))
    return {
        'mean':           float(np.mean(margins)),
        'min':            float(np.min(margins)),
        'std':            float(np.std(margins)),
        'negative_ratio': float((margins < 0).mean()),   # 오분류 비율
        'hist_counts':    hist.tolist(),
        'hist_edges':     edges.tolist(),
    }


def save_params(params, N, rep, seed, params_dir):
    """학습된 QCNN 파라미터를 .npy로 저장."""
    path = os.path.join(params_dir, f'N{N}_rep{rep}_seed{seed}.npy')
    np.save(path, np.array(params))
    return path


# ── CNN1D 학습 + 예측 ──────────────────────────────────────────────────────────
def train_cnn(X_tr, y_tr, X_te, y_te, n_epochs=100, batch_size=32, lr=0.001):
    import torch
    import torch.nn as nn
    from classical import CNN1D

    model = CNN1D(input_dim=N_QUBITS, n_filters=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.FloatTensor(X_tr)
    y_t = torch.FloatTensor(y_tr)

    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            out = model(X_t[idx])
            loss = criterion(out, y_t[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_te))
        preds = (logits > 0).numpy().astype(int)
    return preds


# ── SVM 학습 + 예측 ───────────────────────────────────────────────────────────
def train_svm(X_tr, y_tr, X_te):
    from classical import build_svm
    svm = build_svm()
    svm.fit(X_tr, y_tr)
    return svm.predict(X_te)


# ── 메인 실험 루프 ─────────────────────────────────────────────────────────────
def run(n_sizes, n_repeats, steps, ansatz=ANSATZ, u_params=U_PARAMS, results_dir_name=RESULTS_DIR_NAME):
    results_dir = os.path.join(BASE, results_dir_name)
    params_dir = os.path.join(results_dir, 'params')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)

    print(f"=== 앤자츠: {ansatz} (파라미터/블록={u_params}) → 결과 저장: {results_dir_name}/ ===")
    print("=== NSL-KDD 로드 ===")
    X_train, X_test, y_train, y_test = load_and_preprocess(
        TRAIN_PATH, TEST_PATH, n_components=N_QUBITS
    )
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"Attack ratio - Train: {y_train.mean():.1%}  Test: {y_test.mean():.1%}")

    # QCNN 전용 고정 test 서브셋 (빠른 평가)
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

    # QCNN 학습 파라미터 덮어쓰기
    QCNN_Training.steps      = steps
    QCNN_Training.batch_size = BATCH_SIZE

    all_results = []

    for N in n_sizes:
        print(f"\n{'='*50}")
        print(f"  N = {N}")
        print(f"{'='*50}")

        for rep in range(n_repeats):
            seed = 42 + rep
            Xn, yn = sample_train(X_train, y_train, N, seed)
            yn_pm = np.where(yn == 1, 1, -1)  # QCNN용 {1, -1}

            t0 = time.time()

            # ── QCNN ──
            print(f"  [N={N}, rep={rep+1}/{n_repeats}] QCNN({ansatz}) training ({steps} steps)...")
            loss_history, params = QCNN_Training.circuit_training(
                Xn, list(yn_pm),
                U=ansatz, U_params=u_params,
                embedding_type='Angle',
                circuit='QCNN',
                cost_fn='mse'
            )
            # 파라미터 저장
            param_path = save_params(params, N, rep+1, seed, params_dir)

            y_pred_qcnn = qcnn_predict(X_qte, params, ansatz=ansatz, u_params=u_params)
            m_qcnn = compute_metrics(y_qte, y_pred_qcnn)
            print(f"    QCNN  → acc={m_qcnn['accuracy']:.4f}, f1={m_qcnn['f1']:.4f}  ({time.time()-t0:.0f}s)")

            # ── CNN1D ──
            t1 = time.time()
            y_pred_cnn = train_cnn(Xn, yn, X_test, y_test)
            m_cnn = compute_metrics(y_test, y_pred_cnn)
            print(f"    CNN1D → acc={m_cnn['accuracy']:.4f}, f1={m_cnn['f1']:.4f}  ({time.time()-t1:.0f}s)")

            # ── SVM ──
            t2 = time.time()
            y_pred_svm = train_svm(Xn, yn, X_test)
            m_svm = compute_metrics(y_test, y_pred_svm)
            print(f"    SVM   → acc={m_svm['accuracy']:.4f}, f1={m_svm['f1']:.4f}  ({time.time()-t2:.0f}s)")

            # Train 평가 + 마진 계산 (gen gap & margin analysis용)
            y_pred_qcnn_tr = qcnn_predict(Xn, params, ansatz=ansatz, u_params=u_params)
            m_qcnn_tr = compute_metrics(yn, y_pred_qcnn_tr)
            margins = compute_margins(Xn, yn_pm, params, ansatz=ansatz, u_params=u_params)
            final_loss = float(loss_history[-1])

            row = {
                'N': N, 'rep': rep+1, 'seed': seed,
                'QCNN': {
                    **m_qcnn,
                    'train_accuracy': m_qcnn_tr['accuracy'],
                    'train_f1':       m_qcnn_tr['f1'],
                    'gen_gap_acc':    m_qcnn_tr['accuracy'] - m_qcnn['accuracy'],
                    'gen_gap_f1':     m_qcnn_tr['f1']      - m_qcnn['f1'],
                    'final_loss':     final_loss,
                    'margin':         margins,
                    'param_file':     os.path.basename(param_path),
                },
                'CNN':  m_cnn,
                'SVM':  m_svm,
            }
            all_results.append(row)

            # 중간 저장
            out_path = os.path.join(results_dir, 'results_partial.json')
            with open(out_path, 'w') as f:
                json.dump(all_results, f, indent=2)

    # 최종 저장
    final_path = os.path.join(results_dir, 'results_final.json')
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== 완료 ===")
    print(f"결과 저장: {final_path}")

    # 요약 출력
    print("\n[요약] 모델별 F1 (평균 ± std)")
    for N in n_sizes:
        rows = [r for r in all_results if r['N'] == N]
        for model in ['QCNN', 'CNN', 'SVM']:
            f1s = [r[model]['f1'] for r in rows]
            print(f"  N={N:5d}  {model}: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_sizes', nargs='+', type=int, default=N_SIZES)
    parser.add_argument('--n_repeats', type=int, default=N_REPEATS)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--ansatz', type=str, default=ANSATZ)
    parser.add_argument('--u_params', type=int, default=U_PARAMS)
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR_NAME)
    args = parser.parse_args()

    run(args.n_sizes, args.n_repeats, args.steps,
        ansatz=args.ansatz, u_params=args.u_params, results_dir_name=args.results_dir)
