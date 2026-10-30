
"""Benchmark reproducible de inferencia para ejecuciones completadas."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.core.experimentos.evaluation import (
    _configuracion_del_checkpoint,
)
from src.core.experimentos.service import ExperimentService
from src.core.dataset.experiment import ProjectPaths, prepare_sources, subset_for
from src.core.dataset.transforms import build_eval_transform
from src.core.modelos.factory import crear_modelo


def benchmark_ejecucion(
    root: str | Path,
    run_id: str,
    *,
    warmup_batches: int = 5,
) -> dict:
    root = Path(root).expanduser().resolve()
    service = ExperimentService(root)
    run = service.get_run(run_id)

    if run.status != "completed" or not run.checkpoint_path:
        raise ValueError("La ejecución debe estar completada y tener checkpoint.")

    checkpoint_path = Path(run.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    spec = service.get_experiment(run.experiment_id)

    if (checkpoint.get("metadata") or {}).get("run_id") != run.run_id:
        raise ValueError("El checkpoint no corresponde a esta ejecución.")

    config = _configuracion_del_checkpoint(checkpoint, spec)
    source_assignments = [
        d for d in spec.datasets if d.role == "train_source"
    ]
    source_names = [d.source_name for d in source_assignments]
    path_overrides = {d.source_name: d.path for d in source_assignments}

    prepared = prepare_sources(
        ProjectPaths(root),
        config,
        requested_sources=source_names,
        path_overrides=path_overrides,
        create_missing_splits=False,
        regenerate_splits=False,
    )

    datasets = [
        subset_for(
            prepared[name], "test",
            build_eval_transform(config=config),
        )
        for name in source_names
    ]
    if len(datasets) != 1:
        raise ValueError(
            "Este benchmark requiere una única fuente de test."
        )

    loader = DataLoader(
        datasets[0],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
    )

    model = crear_modelo(
        checkpoint["architecture"],
        num_classes=int(checkpoint["num_classes"]),
        pretrained=False,
        dispositivo=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    def sincronizar() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # Precalentamiento: no se incluye en los tiempos reportados.
    with torch.inference_mode():
        for index, (images, _) in enumerate(loader):
            if index >= warmup_batches:
                break
            images = images.to(device)
            model(images)
    sincronizar()

    # Tiempo del modelo: solo transferencias y pasadas forward.
    model_seconds = 0.0
    samples = 0
    batches = 0

    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device)
            sincronizar()
            start = time.perf_counter()
            model(images)
            sincronizar()
            model_seconds += time.perf_counter() - start
            samples += images.shape[0]
            batches += 1

    # Tiempo extremo a extremo: incluye carga del DataLoader y transformaciones.
    sincronizar()
    start = time.perf_counter()
    end_to_end_samples = 0

    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device)
            model(images)
            end_to_end_samples += images.shape[0]

    sincronizar()
    end_to_end_seconds = time.perf_counter() - start

    result = {
        "run_id": run.run_id,
        "experiment_id": run.experiment_id,
        "model": checkpoint["model"],
        "architecture": checkpoint["architecture"],
        "device": str(device),
        "partition": "test",
        "num_samples": samples,
        "batch_size": config.batch_size,
        "warmup_batches": warmup_batches,
        "model_inference_seconds": model_seconds,
        "model_ms_per_image": model_seconds * 1000 / samples,
        "model_images_per_second": samples / model_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "end_to_end_ms_per_image": (
            end_to_end_seconds * 1000 / end_to_end_samples
        ),
        "batches_measured": batches,
    }

    output_path = (
        root / "results" / "experiments"
        / run.experiment_id / run.run_id / "benchmark.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result["benchmark_path"] = str(output_path.relative_to(root))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--warmup-batches", type=int, default=5)
    args = parser.parse_args()

    print(json.dumps(
        benchmark_ejecucion(
            args.root,
            args.run_id,
            warmup_batches=args.warmup_batches,
        ),
        ensure_ascii=False,
        indent=2,
    ))
