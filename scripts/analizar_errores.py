from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.core.configuracion.settings import list_models
from src.core.utils.paths import ProjectPaths


def analizar_errores(modelo: str, root: str | Path = ".") -> Path:
    # Lista y copia las imágenes mal clasificadas desde predicciones.csv.
    paths = ProjectPaths(Path(root).resolve())
    results_dir = paths.model_results_dir(modelo)
    predictions_path = results_dir / "predicciones.csv"
    errors_dir = results_dir / "errores"

    if not predictions_path.exists():
        raise FileNotFoundError(f"No se encontró: {predictions_path}")

    data = pd.read_csv(predictions_path)
    required_columns = {"ruta", "real", "prediccion", "confianza"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en {predictions_path}: {', '.join(sorted(missing))}"
        )

    errors = data[data["real"] != data["prediccion"]]

    # Elimina archivos anteriores para no conservar errores obsoletos.
    if errors_dir.exists():
        for item in errors_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    errors_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cantidad de errores: {len(errors)}")

    for _, row in errors.iterrows():
        source = paths.root / Path(row["ruta"])
        destination = errors_dir / (
            f"real_{row['real']}_pred_{row['prediccion']}_{source.name}"
        )

        if source.exists():
            shutil.copy2(source, destination)
        else:
            print(f"No se encontró la imagen original: {source}")

        print(
            f"{row['real']} -> {row['prediccion']} | "
            f"confianza={float(row['confianza']) * 100:.2f}% | {row['ruta']}"
        )

    return errors_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copia y lista los errores de clasificación."
    )
    parser.add_argument("--modelo", required=True, choices=list_models())
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    analizar_errores(args.modelo, args.root)


if __name__ == "__main__":
    main()
