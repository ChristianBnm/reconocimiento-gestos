from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import pandas as pd

from src.core.configuracion.catalogo import cargar_catalogos
from src.core.configuracion.settings import list_models
from src.core.utils.paths import ProjectPaths


def analizar_errores(
    modelo: str, root: str | Path = ".", escenario: str = "combinado", fuente: str = "original"
) -> Path:
    # Lista y copia las imágenes mal clasificadas desde predicciones.csv.
    paths = ProjectPaths(Path(root).resolve())
    results_dir = paths.model_results_dir(modelo, escenario)
    predictions_path = results_dir / f"predicciones_{fuente}.csv"
    errors_dir = results_dir / f"errores_{fuente}"

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
        source = Path(row["ruta"])
        if not source.is_absolute():
            source = paths.root / source
        destination = errors_dir / (
            f"{int(row.name):06d}_real_{row['real']}_pred_{row['prediccion']}_{source.name}"
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
    _, scenarios = cargar_catalogos()
    parser.add_argument("--modelo", required=True, choices=list_models())
    parser.add_argument("--escenario", choices=list(scenarios), default="combinado")
    parser.add_argument("--fuente", required=True, help="Fuente de prueba, por ejemplo original o complementario.")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    analizar_errores(args.modelo, args.root, args.escenario, args.fuente)


if __name__ == "__main__":
    main()
