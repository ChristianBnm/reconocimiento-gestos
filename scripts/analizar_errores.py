from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.core.configuracion.settings import list_models
from src.core.utils.paths import ProjectPaths


def main():
    parser = argparse.ArgumentParser(description="Copia y lista los errores de clasificación.")
    parser.add_argument("--modelo", required=True, choices=list_models())
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    paths = ProjectPaths(Path(args.root).resolve())
    results_dir = paths.model_results_dir(args.modelo)
    predictions_path = results_dir / "predicciones.csv"
    errors_dir = results_dir / "errores"
    errors_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        raise FileNotFoundError(f"No se encontró: {predictions_path}")

    data = pd.read_csv(predictions_path)
    errors = data[data["real"] != data["prediccion"]]

    print(f"Cantidad de errores: {len(errors)}")

    for _, row in errors.iterrows():
        source = paths.root / Path(row["ruta"])
        destination = errors_dir / (
            f"real_{row['real']}_pred_{row['prediccion']}_{source.name}"
        )
        if source.exists():
            shutil.copy2(source, destination)
        print(
            f"{row['real']} -> {row['prediccion']} | "
            f"confianza={float(row['confianza']) * 100:.2f}% | {row['ruta']}"
        )


if __name__ == "__main__":
    main()
