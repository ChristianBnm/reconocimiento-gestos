from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch

from src.core.configuracion.settings import CLASS_NAMES
from src.core.dataset.loader import DatasetSubset, GestosDataset


@dataclass
class PredictionResult:
    indices: list[int]
    paths: list[Path]
    y_true: list[int]
    y_pred: list[int]
    confidence: list[float]


def evaluar_modelo(
    model: torch.nn.Module,
    dataset_test: DatasetSubset,
    batch_size: int,
    dispositivo: torch.device,
):
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(dispositivo.type == "cuda"),
    )

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    confidence: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(dispositivo, non_blocking=True)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            conf, pred = probabilities.max(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            confidence.extend(conf.cpu().tolist())

    indices = list(dataset_test.indices)
    paths = [dataset_test.dataset.records[i].path for i in indices]

    return PredictionResult(
        indices=indices,
        paths=paths,
        y_true=y_true,
        y_pred=y_pred,
        confidence=confidence,
    )


def guardar_predicciones(
    result: PredictionResult,
    dataset: GestosDataset,
    project_root: Path,
    ruta: Path,
) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)

    with ruta.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["indice", "ruta", "real", "prediccion", "confianza"])

        for index, path, real, pred, conf in zip(
            result.indices,
            result.paths,
            result.y_true,
            result.y_pred,
            result.confidence,
        ):
            writer.writerow([
                index,
                path.resolve().relative_to(project_root.resolve()).as_posix(),
                dataset.clases[real],
                dataset.clases[pred],
                float(conf),
            ])


def predecir_imagen(
    model: torch.nn.Module,
    image_path: Path,
    transform,
    dispositivo: torch.device,
    class_names=CLASS_NAMES,
) -> tuple[str, float]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"No se pudo cargar la imagen: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(image=image)["image"].unsqueeze(0).to(dispositivo)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)

    return class_names[prediction.item()], float(confidence.item())
