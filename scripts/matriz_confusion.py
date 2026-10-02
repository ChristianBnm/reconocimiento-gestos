from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.core.configuracion.settings import CLASS_NAMES, list_models
from src.core.utils.paths import ProjectPaths


def main():
    parser = argparse.ArgumentParser(description="Genera la matriz de confusión.")
    parser.add_argument("--modelo", required=True, choices=list_models())
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    paths = ProjectPaths(Path(args.root).resolve())
    results_dir = paths.model_results_dir(args.modelo)
    predictions_path = results_dir / "predicciones.csv"
    output_path = results_dir / "matriz_confusion.png"

    if not predictions_path.exists():
        raise FileNotFoundError(f"No se encontró: {predictions_path}")

    data = pd.read_csv(predictions_path)
    labels = list(CLASS_NAMES)
    matrix = confusion_matrix(
        data["real"],
        data["prediccion"],
        labels=labels,
    )

    fig, ax = plt.subplots(figsize=(16, 14))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=ax,
        xticks_rotation=90,
        values_format="d",
    )
    ax.set_title(f"Matriz de confusión - {args.modelo}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Matriz guardada en: {output_path}")


if __name__ == "__main__":
    main()
