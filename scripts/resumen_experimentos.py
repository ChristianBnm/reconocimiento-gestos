from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from src.core.configuracion.catalogo import cargar_catalogos
from src.core.configuracion.settings import list_models
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


def main() -> None:
    _, scenarios = cargar_catalogos()
    parser = argparse.ArgumentParser(
        description="Consolida métricas por escenario, modelo y fuente de prueba."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--salida", default="results/resumen_experimentos.csv")
    parser.add_argument("--escenarios", nargs="+", choices=list(scenarios), default=list(scenarios))
    parser.add_argument("--modelos", nargs="+", choices=list_models(), default=list_models())
    parser.add_argument("--fuentes", nargs="+", default=None, help="Fuentes de prueba; por defecto usa todas las configuradas.")
    args = parser.parse_args()

    paths = ProjectPaths(Path(args.root).resolve())
    source_names, _ = cargar_catalogos()
    sources = args.fuentes or list(source_names)
    rows = []
    for scenario in args.escenarios:
        for model in args.modelos:
            results_dir = paths.model_results_dir(model, scenario)
            config = results_dir / "config.json"
            for source in sources:
                metrics_path = results_dir / f"metrics_{source}.json"
                row = {"scenario": scenario, "model": model, "test_source": source, "status": "ok"}
                if not metrics_path.is_file():
                    row["status"] = "missing_metrics"
                    row.update({key: None for key in METRICS})
                else:
                    metrics = load_json(metrics_path)
                    row.update({key: metrics.get(key) for key in METRICS})
                if config.is_file():
                    row["class_count"] = len(load_json(config).get("class_names", []))
                else:
                    row["class_count"] = None
                rows.append(row)

    output = Path(args.salida)
    if not output.is_absolute():
        output = paths.root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["scenario", "model", "test_source", "class_count", "status", *METRICS]
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Resumen guardado en {paths.relative_to_root(output)} ({len(rows)} combinaciones).")


if __name__ == "__main__":
    main()
