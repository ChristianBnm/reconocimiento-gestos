from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from src.core.configuracion.catalogo import cargar_catalogos
from src.core.configuracion.settings import list_models
from src.core.evaluacion.comparacion import METRICS, cargar_comparacion, ordenar_comparacion


def main():
    _, scenarios = cargar_catalogos()
    parser = argparse.ArgumentParser(description="Compara métricas de modelos en un escenario/fuente.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--escenario", choices=list(scenarios), default="combinado")
    parser.add_argument("--fuente", default="original", help="Fuente evaluada (nombre de una fuente configurada).")
    parser.add_argument("--modelos", nargs="+", default=list_models(), choices=list_models())
    args = parser.parse_args()

    rows = ordenar_comparacion(
        cargar_comparacion(args.root, args.escenario, args.modelos, args.fuente),
        metric="f1_macro",
    )
    columns = ("model", "status", "n_samples", "accuracy", "precision_macro", "recall_macro", "f1_macro", "balanced_accuracy", "cohen_kappa")
    print(" | ".join(columns))
    print("-" * 130)

    def fmt(value):
        if value is None:
            return "N/D"
        return str(value) if isinstance(value, int) else f"{value:.6f}"

    for row in rows:
        print(" | ".join(fmt(row.get(column)) for column in columns))


if __name__ == "__main__":
    main()
