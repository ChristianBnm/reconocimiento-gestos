from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from src.core.utils.io import load_json
from src.core.utils.paths import ProjectPaths

METRICS = (
    "n_samples",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "balanced_accuracy",
    "cohen_kappa",
)


def cargar_comparacion(
    root: str | Path,
    scenario: str,
    models: Iterable[str],
    test_source: str,
) -> list[dict]:
    """Carga resultados de modelos comparables sin asumir que todos existen.

    Devuelve una fila por modelo. Si faltan métricas, la fila se conserva con
    ``status='missing_metrics'`` para que un resumen no oculte experimentos
    todavía no ejecutados.
    """
    paths = ProjectPaths(Path(root).expanduser().resolve())
    rows: list[dict] = []
    for model_name in models:
        results_dir = paths.model_results_dir(model_name, scenario)
        metrics_path = results_dir / f"metrics_{test_source}.json"
        config_path = results_dir / "config.json"
        row = {
            "scenario": scenario,
            "model": model_name,
            "test_source": test_source,
            "status": "ok" if metrics_path.is_file() else "missing_metrics",
            **{metric: None for metric in METRICS},
            "class_count": None,
        }
        if metrics_path.is_file():
            metrics = load_json(metrics_path)
            for metric in METRICS:
                row[metric] = metrics.get(metric)
        if config_path.is_file():
            config = load_json(config_path)
            row["class_count"] = len(config.get("class_names", [])) or config.get("num_classes")
        rows.append(row)
    return rows


def ordenar_comparacion(
    rows: Iterable[Mapping], metric: str = "f1_macro", descending: bool = True
) -> list[dict]:
    """Ordena resultados colocando primero los valores disponibles.

    Los registros sin la métrica solicitada se conservan al final. El orden
    original se mantiene entre registros empatados gracias al sort estable.
    """
    if metric not in METRICS:
        raise ValueError(f"Métrica desconocida: {metric}. Disponibles: {', '.join(METRICS)}")
    materialized = [dict(row) for row in rows]
    return sorted(
        materialized,
        key=lambda row: (row.get(metric) is not None, row.get(metric) or 0.0),
        reverse=descending,
    )
