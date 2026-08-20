"""실험 결과 저장·평가 유틸리티"""

import json, time, os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
    }


def save_result(result: dict, results_dir: str):
    """타임스탬프 기반 파일명으로 저장 — 실험 덮어쓰기 방지."""
    os.makedirs(results_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{results_dir}/result_{ts}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {fname}")
    return fname
