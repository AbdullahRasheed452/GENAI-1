import numpy as np


def confusion_matrix(y_true, y_pred, num_classes=4):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def summarize(cm):
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }
