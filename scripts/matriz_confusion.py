from __future__ import annotations

import argparse
from pathlib import Path

import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.core.configuracion.catalogo import cargar_catalogos
from src.core.configuracion.settings import list_models
from src.core.utils.io import load_json
from src.core.utils.paths import ProjectPaths


def generar_matriz_confusion(
    modelo: str,
    root: str | Path = ".",
    escenario: str = "combinado",
    fuente: str = "original",
) -> Path:
    paths = ProjectPaths(Path(root).resolve())
    results_dir = paths.model_results_dir(modelo, escenario)
    predictions_path = results_dir / f"predicciones_{fuente}.csv"
    output_path = results_dir / f"matriz_confusion_{fuente}.png"
    config_path = results_dir / "config.json"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"No se encontró: {predictions_path}")
    data = pd.read_csv(predictions_path)
    required_columns = {"real", "prediccion"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {', '.join(sorted(missing))}")
    if not config_path.is_file():
        raise FileNotFoundError(f"No se encontró la configuración que define las clases: {config_path}")
    class_names = load_json(config_path)["class_names"]
    labels = list(class_names)
    matrix = confusion_matrix(data["real"], data["prediccion"], labels=labels)
    fig_width = max(10, min(24, 0.45 * len(labels)))
    fig_height = max(8, min(22, 0.42 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    try:
        ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
            ax=ax, xticks_rotation=90, values_format="d", colorbar=False
        )
        ax.set_title(f"Matriz de confusión - {modelo} ({escenario}, {fuente})")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"Matriz guardada en: {output_path}")
    return output_path


def main() -> None:
    _, scenarios = cargar_catalogos()
    parser = argparse.ArgumentParser(description="Genera una matriz de confusión a partir de las predicciones.")
    parser.add_argument("--modelo", required=True, choices=list_models())
    parser.add_argument("--escenario", choices=list(scenarios), default="combinado")
    parser.add_argument("--fuente", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    generar_matriz_confusion(args.modelo, args.root, args.escenario, args.fuente)


if __name__ == "__main__":
    main()
