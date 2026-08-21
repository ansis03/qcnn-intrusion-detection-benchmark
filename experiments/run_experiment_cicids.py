"""
CICIDS-2017 실험 — run_experiment.py(NSL-KDD)와 동일한 프로토콜, 앤자츠 파라미터화 지원.

사전 준비: CICIDS-2017 원본 8개 CSV를 data/raw-cicids/extracted/ 에 받은 뒤
`python experiments/prepare_cicids.py` 를 먼저 실행해 data/processed/cicids_clean.pkl 을
생성해야 한다(원본 CSV·정제 pkl 모두 용량 문제로 리포에는 포함하지 않음. 자세한
컬럼 처리 방식은 README와 paper.md Section IV.C 참고).

사용법: python experiments/run_experiment_cicids.py
        python experiments/run_experiment_cicids.py --ansatz U_15 --u_params 4 --results_dir results/cicids/u15
"""

import argparse, sys, os, json, time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'src'))
sys.path.insert(0, os.path.join(BASE, 'QCNN', 'QCNN'))

from preprocessing_cicids import load_and_preprocess
from evaluate import compute_metrics
import QCNN_circuit
import Training as QCNN_Training
from run_experiment import (
    qcnn_predict, qcnn_raw_outputs, compute_margins, save_params,
    train_cnn, train_svm, sample_train,
    N_QUBITS, BATCH_SIZE, QCNN_EVAL, ANSATZ, U_PARAMS, RESULTS_DIR_NAME,
)

# CICIDS-2017 정제 데이터는 리포에 포함하지 않음 — README 안내에 따라 로컬에서 생성
CLEAN_PATH = os.path.join(BASE, 'data', 'processed', 'cicids_clean.pkl')

N_SIZES   = [50, 100, 500, 1000, 5000]
N_REPEATS = 5
STEPS     = 200


def run(n_sizes, n_repeats, steps, ansatz=ANSATZ, u_params=U_PARAMS, results_dir_name=RESULTS_DIR_NAME):
    results_dir = os.path.join(BASE, results_dir_name)
    params_dir = os.path.join(results_dir, 'params')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)

    print(f"=== 앤자츠: {ansatz} (파라미터/블록={u_params}) → 결과 저장: {results_dir_name}/ ===")
    print("=== CICIDS-2017 로드 ===")
    X_train, X_test, y_train, y_test = load_and_preprocess(CLEAN_PATH, n_components=N_QUBITS)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"Attack ratio - Train: {y_train.mean():.1%}  Test: {y_test.mean():.1%}")

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
            yn_pm = np.where(yn == 1, 1, -1)

            t0 = time.time()

            print(f"  [N={N}, rep={rep+1}/{n_repeats}] QCNN({ansatz}) training ({steps} steps)...")
            loss_history, params = QCNN_Training.circuit_training(
                Xn, list(yn_pm),
                U=ansatz, U_params=u_params,
                embedding_type='Angle',
                circuit='QCNN',
                cost_fn='mse'
            )
            param_path = save_params(params, N, rep+1, seed, params_dir)

            y_pred_qcnn = qcnn_predict(X_qte, params, ansatz=ansatz, u_params=u_params)
            m_qcnn = compute_metrics(y_qte, y_pred_qcnn)
            print(f"    QCNN  → acc={m_qcnn['accuracy']:.4f}, f1={m_qcnn['f1']:.4f}  ({time.time()-t0:.0f}s)")

            t1 = time.time()
            y_pred_cnn = train_cnn(Xn, yn, X_test, y_test)
            m_cnn = compute_metrics(y_test, y_pred_cnn)
            print(f"    CNN1D → acc={m_cnn['accuracy']:.4f}, f1={m_cnn['f1']:.4f}  ({time.time()-t1:.0f}s)")

            t2 = time.time()
            y_pred_svm = train_svm(Xn, yn, X_test)
            m_svm = compute_metrics(y_test, y_pred_svm)
            print(f"    SVM   → acc={m_svm['accuracy']:.4f}, f1={m_svm['f1']:.4f}  ({time.time()-t2:.0f}s)")

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

            out_path = os.path.join(results_dir, 'results_partial.json')
            with open(out_path, 'w') as f:
                json.dump(all_results, f, indent=2)

    final_path = os.path.join(results_dir, 'results_final.json')
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== 완료 ===")
    print(f"결과 저장: {final_path}")

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
    parser.add_argument('--results_dir', type=str, default='results/cicids/uttn')
    args = parser.parse_args()

    run(args.n_sizes, args.n_repeats, args.steps,
        ansatz=args.ansatz, u_params=args.u_params, results_dir_name=args.results_dir)
