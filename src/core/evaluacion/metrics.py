from __future__ import annotations

from typing import Sequence
import math

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def calcular_metricas(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener la misma longitud.")
    if len(y_true) == 0:
        raise ValueError("No se pueden calcular métricas sobre una evaluación vacía.")
    kappa = float(cohen_kappa_score(y_true, y_pred))
    if not math.isfinite(kappa):
        kappa = None
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": kappa,
    }


def generar_informe_clasificacion(y_true, y_pred, class_names):
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
