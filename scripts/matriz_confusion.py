from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.core.configuracion.settings import CLASS_NAMES, list_models
from src.core.utils.paths import ProjectPaths


def generar_matriz_confusion(modelo: str, root: str | Path = ".") -> Path:
    """Genera y guarda la matriz de confusión a partir de predicciones.csv."""
    paths = ProjectPaths(Path(root).resolve())
    results_dir = paths.model_results_dir(modelo)
    predictions_path = results_dir / "predicciones.csv"
    output_path = results_dir / "matriz_confusion.png"

    if not predictions_path.exists():
        raise FileNotFoundError(f"No se encontró: {predictions_path}")

    data = pd.read_csv(predictions_path)
    required_columns = {"real", "prediccion"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en {predictions_path}: {', '.join(sorted(missing))}"
        )

    labels = list(CLASS_NAMES)
    matrix = confusion_matrix(data["real"], data["prediccion"], labels=labels)

    fig, ax = plt.subplots(figsize=(16, 14))
    try:
        ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
            ax=ax, xticks_rotation=90, values_format="d"
        )

        # Resalta las celdas con un único caso, como en el script original.
        for text in ax.texts:
            if text.get_text() == "1":
                text.set_color("white")
                text.set_fontweight("bold")
                text.set_bbox(
                    dict(
                        facecolor="black",
                        edgecolor="black",
                        boxstyle="square,pad=0.2",
                    )
                )

        ax.set_title(f"Matriz de confusión - {modelo}")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)

    print(f"Matriz guardada en: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera la matriz de confusión.")
    parser.add_argument("--modelo", required=True, choices=list_models())
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    generar_matriz_confusion(args.modelo, args.root)


if __name__ == "__main__":
    main()
