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
    num_workers: int = 0,
):
    from torch.utils.data import DataLoader

    if batch_size < 1:
        raise ValueError("batch_size debe ser mayor que cero.")
    loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(dispositivo.type == "cuda"),
    )
    if len(loader) == 0:
        raise ValueError("El conjunto de prueba está vacío.")

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    confidence: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(dispositivo, non_blocking=True)
            logits = model(images)
            if logits.ndim != 2:
                raise ValueError(f"El modelo devolvió logits con forma inesperada: {tuple(logits.shape)}")
            probabilities = torch.softmax(logits, dim=1)
            conf, pred = probabilities.max(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            confidence.extend(conf.cpu().tolist())

    indices = list(dataset_test.indices)
    paths = [dataset_test.dataset.records[index].path for index in indices]
    if not (len(indices) == len(y_true) == len(y_pred) == len(confidence) == len(paths)):
        raise RuntimeError("La cantidad de predicciones no coincide con el conjunto de prueba.")
    return PredictionResult(indices, paths, y_true, y_pred, confidence)


def guardar_predicciones(
    result: PredictionResult,
    dataset: GestosDataset,
    project_root: Path,
    ruta: Path,
) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    root = project_root.resolve()
    with ruta.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["indice_dataset", "fuente", "ruta", "real", "prediccion", "confianza"])
        for index, path, real, pred, conf in zip(
            result.indices, result.paths, result.y_true, result.y_pred, result.confidence
        ):
            try:
                display_path = path.resolve().relative_to(root).as_posix()
            except ValueError:
                display_path = path.resolve().as_posix()
            record = dataset.records[index]
            writer.writerow([
                index,
                record.source,
                display_path,
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
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"No se pudo cargar la imagen: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(image=image)["image"].unsqueeze(0).to(dispositivo)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        if logits.shape[-1] != len(class_names):
            raise ValueError(
                f"El modelo devuelve {logits.shape[-1]} clases, pero el vocabulario tiene {len(class_names)}."
            )
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)
    return class_names[prediction.item()], float(confidence.item())
