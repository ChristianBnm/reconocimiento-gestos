from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix


def guardar_matriz_confusion(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    output_dir: Path,
    prefix: str = "confusion_matrix",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # Guardar matriz de cantidades en CSV.
    csv_path = output_dir / f"{prefix}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["real\\prediccion", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *(int(value) for value in row)])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_classes = len(class_names)
    fig_width = max(12, min(24, 0.55 * n_classes))
    fig_height = max(10, min(24, 0.50 * n_classes))

    def plot_matrix(data, title, output_path, normalized=False):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        image = ax.imshow(data, interpolation="nearest", cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicción",
            ylabel="Etiqueta real",
            title=title,
        )

        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)

        # Anotar las celdas no vacías para evitar llenar la figura de ceros.
        max_value = float(np.max(data)) if data.size else 0.0
        threshold = max_value * 0.55

        for i in range(n_classes):
            for j in range(n_classes):
                value = data[i, j]

                if value == 0:
                    continue

                if normalized:
                    label = f"{value:.0%}"
                else:
                    label = f"{int(value)}"

                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6 if n_classes > 30 else 8,
                    color="white" if value > threshold else "black",
                )

        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Matriz original: cantidades absolutas.
    plot_matrix(
        matrix,
        "Matriz de confusión",
        output_dir / f"{prefix}.png",
    )

    # Matriz normalizada: proporción de aciertos y errores por clase real.
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )

    plot_matrix(
        normalized_matrix,
        "Matriz de confusión normalizada por clase real",
        output_dir / f"{prefix}_normalizada.png",
        normalized=True,
    )

def guardar_errores_prediccion(
    predictions_csv: Path,
    project_root: Path,
    output_dir: Path,
    class_names: Sequence[str],
) -> dict[str, int]:
    """Exporta errores de clasificación y copia sus imágenes para inspección visual.

    La estructura de salida agrupa imágenes por clase real y clase predicha.
    Solo reemplaza el directorio de errores de la fuente que se está evaluando.
    """
    import pandas as pd

    predictions_csv = Path(predictions_csv)
    project_root = Path(project_root).expanduser().resolve()
    output_dir = Path(output_dir)

    if not predictions_csv.is_file():
        raise FileNotFoundError(f"No existe el CSV de predicciones: {predictions_csv}")

    required = {"ruta", "real", "prediccion", "confianza"}
    df = pd.read_csv(predictions_csv)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"El CSV de predicciones no contiene las columnas requeridas: {sorted(missing)}"
        )

    errors = df[df["real"].astype(str) != df["prediccion"].astype(str)].copy()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "imagenes"

    records = []
    copied = 0
    missing_images = 0

    def safe_label(value: str) -> str:
        # Las etiquetas actuales son simples (0-9, A-Z, _); sanitizamos igualmente.
        value = str(value).strip()
        return value if value and all(c.isalnum() or c in "-_" for c in value) else "etiqueta"

    for row_number, row in enumerate(errors.to_dict(orient="records"), start=1):
        real = str(row["real"])
        predicted = str(row["prediccion"])
        original_path = Path(str(row["ruta"]))
        if not original_path.is_absolute():
            original_path = project_root / original_path
        original_path = original_path.resolve()

        relative_copy = ""
        status = "copiada"
        if original_path.is_file():
            group_dir = images_dir / f"real_{safe_label(real)}" / f"predicha_{safe_label(predicted)}"
            group_dir.mkdir(parents=True, exist_ok=True)
            # Prefijo secuencial para evitar colisiones de nombres entre carpetas fuente.
            destination = group_dir / f"error_{row_number:04d}_{original_path.name}"
            shutil.copy2(original_path, destination)
            relative_copy = destination.relative_to(output_dir).as_posix()
            copied += 1
        else:
            status = "imagen_no_encontrada"
            missing_images += 1

        records.append({
            "fuente": row.get("fuente", ""),
            "ruta_original": str(row["ruta"]),
            "clase_real": real,
            "clase_predicha": predicted,
            "confianza": float(row["confianza"]),
            "imagen_copiada": relative_copy,
            "estado": status,
        })

    report_path = output_dir / "resumen_errores.csv"
    fieldnames = [
        "fuente", "ruta_original", "clase_real", "clase_predicha",
        "confianza", "imagen_copiada", "estado",
    ]
    with report_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Resumen de los pares de confusión más frecuentes para orientar la revisión.
    pair_counts: dict[tuple[str, str], int] = {}
    for record in records:
        pair = (record["clase_real"], record["clase_predicha"])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    with (output_dir / "resumen_por_confusion.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["clase_real", "clase_predicha", "cantidad"])
        for (real, predicted), count in sorted(
            pair_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        ):
            writer.writerow([real, predicted, count])

    return {
        "total_evaluadas": int(len(df)),
        "total_errores": int(len(errors)),
        "imagenes_copiadas": copied,
        "imagenes_no_encontradas": missing_images,
    }
