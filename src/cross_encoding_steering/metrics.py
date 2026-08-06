from __future__ import annotations

import numpy as np


def unit_vector(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None) -> float:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(map(int, true.tolist())) | set(map(int, pred.tolist())))
    scores = []
    for label in labels:
        tp = float(((true == label) & (pred == label)).sum())
        fp = float(((true != label) & (pred == label)).sum())
        fn = float(((true == label) & (pred != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def js_divergence(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    return float(0.5 * (kl_pm + kl_qm))
