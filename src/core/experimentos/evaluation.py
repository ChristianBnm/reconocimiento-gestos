
"""Evaluación reproducible de ejecuciones de experimentos."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from src.core.configuracion.settings import ExperimentConfig
from src.core.dataset.experiment import (
    ProjectPaths,
    prepare_sources,
    subset_for,
)
from src.core.dataset.transforms import build_eval_transform
from src.core.experimentos.schemas import RunRecord
from src.core.experimentos.service import ExperimentService
from src.core.modelos.factory import crear_modelo


def _guardar_json_atomico(path: Path, contenido: dict) -> None:
    """Guarda el resultado sin dejar un JSON parcial ante un error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporal = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporal_path = Path(temporal)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as archivo:
            json.dump(
                contenido,
                archivo,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            archivo.write("\n")
            archivo.flush()
            os.fsync(archivo.fileno())
        os.replace(temporal_path, path)
    except BaseException:
        temporal_path.unlink(missing_ok=True)
        raise


def _configuracion_del_checkpoint(
    checkpoint: dict,
    spec,
) -> ExperimentConfig:
    """Reconstruye la configuración utilizada durante el entrenamiento."""
    config_guardada = checkpoint.get("experiment_config")
    if not isinstance(config_guardada, dict):
        raise ValueError("El checkpoint no contiene experiment_config.")

    config = ExperimentService._build_config(spec)

    campos = {
        "image_size",
        "train_size",
        "val_size",
        "test_size",
        "seed",
        "split_seed",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "epochs",
        "patience",
        "min_delta",
        "num_workers",
        "pin_memory",
        "freeze_backbone",
        "horizontal_flip_probability",
        "affine_probability",
        "brightness_contrast_probability",
        "normalization_mean",
        "normalization_std",
    }
    cambios = {
        key: value
        for key, value in config_guardada.items()
        if key in campos
    }

    for key in ("normalization_mean", "normalization_std"):
        if key in cambios:
            cambios[key] = tuple(cambios[key])

    return __import__("dataclasses").replace(config, **cambios)


def evaluar_ejecucion(
    root: str | Path,
    run_id: str,
    *,
    overwrite: bool = False,
) -> dict:
    """Evalúa el checkpoint de una ejecución sobre su partición test.

    Guarda evaluation.json junto al historial y la configuración del run.
    No entrena el modelo ni crea o regenera particiones.
    """
    root = Path(root).expanduser().resolve()
    service = ExperimentService(root)

    run: RunRecord = service.get_run(run_id)
    if run.status != "completed":
        raise ValueError(
            f"La ejecución {run_id!r} no está completada: {run.status!r}."
        )
    if not run.checkpoint_path:
        raise ValueError("La ejecución no tiene un checkpoint asociado.")

    checkpoint_path = Path(run.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No se encuentra el checkpoint: {checkpoint_path}"
        )

    spec = service.get_experiment(run.experiment_id)
    run_dir = checkpoint_path.parent
    output_path = run_dir.parent.parent / "results" / "unused"
    # El resultado se guarda en results/experiments/<experiment_id>/<run_id>.
    output_path = (
        root / "results" / "experiments"
        / run.experiment_id / run.run_id / "evaluation.json"
    )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Ya existe {output_path}. Usá overwrite=True para reemplazarlo."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    metadata = checkpoint.get("metadata") or {}
    if metadata.get("experiment_id") != run.experiment_id:
        raise ValueError("El checkpoint no coincide con el experimento.")
    if metadata.get("run_id") != run.run_id:
        raise ValueError("El checkpoint no coincide con la ejecución.")

    config = _configuracion_del_checkpoint(checkpoint, spec)
    class_names = tuple(checkpoint.get("class_names", ()))
    num_classes = int(checkpoint.get("num_classes", 0))

    if not class_names or len(class_names) != num_classes:
        raise ValueError("Las clases del checkpoint no son válidas.")

    source_assignments = [
        dataset for dataset in spec.datasets if dataset.role != "unused"
    ]
    if not source_assignments:
        raise ValueError("No hay fuentes activas para evaluar.")

    source_names = [dataset.source_name for dataset in source_assignments]
    path_overrides = {dataset.source_name: dataset.path for dataset in source_assignments}
    paths = ProjectPaths(root)
    prepared = prepare_sources(
        paths, config, requested_sources=source_names,
        path_overrides=path_overrides, create_missing_splits=False,
        regenerate_splits=False,
    )

    architecture = checkpoint.get("architecture")
    if not architecture:
        raise ValueError("El checkpoint no declara la arquitectura.")
    model = crear_modelo(
        architecture, num_classes=num_classes, pretrained=False, dispositivo=device
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    transform = build_eval_transform(config=config)
    tests: dict[str, dict] = {}
    external_evaluations: dict[str, dict] = {}
    for assignment in source_assignments:
        name = assignment.source_name
        dataset = subset_for(prepared[name], "test", transform)
        loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory and device.type == "cuda",
        )
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        criterion = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        total_correct = 0
        total = 0
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                if logits.ndim != 2 or logits.shape[1] != num_classes:
                    raise ValueError("La salida del modelo no coincide con las clases.")
                total_loss += criterion(logits, labels).item()
                predictions = logits.argmax(dim=1)
                total_correct += (predictions == labels).sum().item()
                total += labels.numel()
                encoded = labels * num_classes + predictions
                confusion += torch.bincount(
                    encoded, minlength=num_classes * num_classes
                ).reshape(num_classes, num_classes).cpu()
        if total == 0:
            raise ValueError(f"La partición test de '{name}' está vacía.")

        tp = confusion.diag().double()
        support = confusion.sum(dim=1).double()
        predicted_count = confusion.sum(dim=0).double()
        precision = torch.where(
            predicted_count > 0, tp / predicted_count.clamp(min=1),
            torch.zeros_like(tp),
        )
        recall = torch.where(
            support > 0, tp / support.clamp(min=1), torch.zeros_like(tp)
        )
        f1 = torch.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall).clamp(min=1e-15),
            torch.zeros_like(tp),
        )
        weighted_f1 = (f1 * support).sum().item() / support.sum().item()
        per_source = {
            "source": name, "role": assignment.role, "partition": "test",
            "num_samples": total,
            "metrics": {
                "loss": total_loss / total,
                "accuracy": total_correct / total,
                "precision_macro": precision.mean().item(),
                "recall_macro": recall.mean().item(),
                "f1_macro": f1.mean().item(),
                "f1_weighted": weighted_f1,
            },
            "per_class": {
                class_name: {
                    "support": int(support[index].item()),
                    "precision": precision[index].item(),
                    "recall": recall[index].item(),
                    "f1": f1[index].item(),
                } for index, class_name in enumerate(class_names)
            },
            "confusion_matrix": confusion.tolist(),
        }
        target = external_evaluations if assignment.role == "external_eval" else tests
        target[name] = per_source

    resultado = {
        "experiment_id": run.experiment_id, "run_id": run.run_id,
        "model": checkpoint["model"], "architecture": architecture,
        "checkpoint_path": paths.relative_to_root(checkpoint_path),
        "device": str(device), "num_classes": num_classes,
        "tests": tests, "external_evaluations": external_evaluations,
    }
    _guardar_json_atomico(output_path, resultado)
    resultado["evaluation_path"] = paths.relative_to_root(output_path)
    return resultado
