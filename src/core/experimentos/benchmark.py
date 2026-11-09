"""Benchmark reproducible de inferencia para ejecuciones completadas.

Mide varios tamaños de lote sin entrenar ni modificar checkpoints/datasets.
Conserva la validación de fingerprint del benchmark original.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.core.experimentos.evaluation import (
    _configuracion_del_checkpoint,
    _guardar_json_atomico,
)
from src.core.experimentos.service import ExperimentService
from src.core.dataset.experiment import ProjectPaths, prepare_sources, subset_for
from src.core.dataset.transforms import build_eval_transform
from src.core.modelos.factory import crear_modelo
from src.core.dataset.split import dataset_fingerprint


def _percentile(values: list[float], percentile: float) -> float | None:
    """Percentil lineal; devuelve None si no hay observaciones."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validar_batch_sizes(batch_sizes: tuple[int, ...] | list[int]) -> list[int]:
    if not batch_sizes:
        raise ValueError("Debe indicarse al menos un batch size.")
    result: list[int] = []
    for value in batch_sizes:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("Todos los batch sizes deben ser enteros positivos.")
        if value not in result:
            result.append(value)
    return result


def _medir_batch_size(
    model: torch.nn.Module,
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
    warmup_batches: int,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and device.type == "cuda",
    )

    # Calentamiento separado: no forma parte de ninguna métrica.
    with torch.inference_mode():
        for index, (images, _) in enumerate(loader):
            if index >= warmup_batches:
                break
            images = images.to(
                device, non_blocking=(pin_memory and device.type == "cuda")
            )
            model(images)
    _sync(device)

    # Forward puro: las imágenes ya están en el dispositivo antes de iniciar
    # el cronómetro. Se registran latencias por lote y por imagen del lote.
    forward_batch_seconds: list[float] = []
    transfer_batch_seconds: list[float] = []
    samples = 0

    with torch.inference_mode():
        for images, _ in loader:
            count = int(images.shape[0])

            _sync(device)
            transfer_start = time.perf_counter()
            device_images = images.to(
                device, non_blocking=(pin_memory and device.type == "cuda")
            )
            _sync(device)
            transfer_batch_seconds.append(time.perf_counter() - transfer_start)

            _sync(device)
            forward_start = time.perf_counter()
            model(device_images)
            _sync(device)
            forward_batch_seconds.append(time.perf_counter() - forward_start)
            samples += count

    if samples == 0:
        raise ValueError(
            "La partición test no contiene muestras para realizar el benchmark."
        )

    forward_seconds = sum(
        duration for duration in forward_batch_seconds
    )
    transfer_seconds = sum(transfer_batch_seconds)

    # Extremo a extremo: iteración del DataLoader, lectura/transformaciones,
    # transferencia y forward. Incluye el último lote aunque sea incompleto.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    e2e_batch_seconds: list[float] = []
    e2e_samples = 0
    _sync(device)
    e2e_start = time.perf_counter()

    with torch.inference_mode():
        for images, _ in loader:
            count = int(images.shape[0])
            batch_start = time.perf_counter()
            device_images = images.to(
                device, non_blocking=(pin_memory and device.type == "cuda")
            )
            model(device_images)
            _sync(device)
            e2e_batch_seconds.append(time.perf_counter() - batch_start)
            e2e_samples += count

    _sync(device)
    e2e_seconds = time.perf_counter() - e2e_start

    if e2e_samples != samples:
        raise RuntimeError(
            "La cantidad de muestras cambió entre pasadas del benchmark."
        )
    if forward_seconds <= 0 or e2e_seconds <= 0:
        raise ValueError("El benchmark produjo una duración no positiva.")

    forward_per_image = [
        duration * 1000.0 / batch_size_actual
        for duration, batch_size_actual in zip(
            forward_batch_seconds,
            [
                min(batch_size, samples - index * batch_size)
                for index in range(len(forward_batch_seconds))
            ],
        )
    ]
    e2e_per_image = [
        duration * 1000.0 / min(batch_size, samples - index * batch_size)
        for index, duration in enumerate(e2e_batch_seconds)
    ]

    gpu_memory = None
    if device.type == "cuda":
        gpu_memory = {
            "peak_allocated_mb": round(
                torch.cuda.max_memory_allocated(device) / (1024 ** 2), 3
            ),
            "peak_reserved_mb": round(
                torch.cuda.max_memory_reserved(device) / (1024 ** 2), 3
            ),
        }

    return {
        "batch_size": batch_size,
        "num_samples": samples,
        "num_batches": len(forward_batch_seconds),
        "model_forward_seconds": round(forward_seconds, 6),
        "model_forward_ms_per_image": round(
            forward_seconds * 1000.0 / samples, 6
        ),
        "model_images_per_second": round(samples / forward_seconds, 4),
        "forward_batch_ms_p50": round(
            _percentile([x * 1000.0 for x in forward_batch_seconds], 0.50), 6
        ),
        "forward_batch_ms_p95": round(
            _percentile([x * 1000.0 for x in forward_batch_seconds], 0.95), 6
        ),
        "forward_ms_per_image_p50": round(
            _percentile(forward_per_image, 0.50), 6
        ),
        "forward_ms_per_image_p95": round(
            _percentile(forward_per_image, 0.95), 6
        ),
        "transfer_seconds": round(transfer_seconds, 6),
        "transfer_ms_per_image": round(
            transfer_seconds * 1000.0 / samples, 6
        ),
        "end_to_end_seconds": round(e2e_seconds, 6),
        "end_to_end_ms_per_image": round(
            e2e_seconds * 1000.0 / e2e_samples, 6
        ),
        "end_to_end_images_per_second": round(
            e2e_samples / e2e_seconds, 4
        ),
        "end_to_end_batch_ms_p50": round(
            _percentile([x * 1000.0 for x in e2e_batch_seconds], 0.50), 6
        ),
        "end_to_end_batch_ms_p95": round(
            _percentile([x * 1000.0 for x in e2e_batch_seconds], 0.95), 6
        ),
        "end_to_end_ms_per_image_p50": round(
            _percentile(e2e_per_image, 0.50), 6
        ),
        "end_to_end_ms_per_image_p95": round(
            _percentile(e2e_per_image, 0.95), 6
        ),
        "gpu_memory": gpu_memory,
    }


def benchmark_ejecucion(
    root: str | Path,
    run_id: str,
    *,
    warmup_batches: int = 5,
    batch_sizes: tuple[int, ...] | list[int] = (1, 32),
) -> dict:
    root = Path(root).expanduser().resolve()

    if (
        isinstance(warmup_batches, bool)
        or not isinstance(warmup_batches, int)
        or warmup_batches < 0
    ):
        raise ValueError("warmup_batches debe ser un entero mayor o igual a cero.")

    batch_sizes = _validar_batch_sizes(batch_sizes)
    service = ExperimentService(root)
    run = service.get_run(run_id)

    if run.status != "completed" or not run.checkpoint_path:
        raise ValueError("La ejecución debe estar completada y tener checkpoint.")

    checkpoint_path = Path(run.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No se encontró el checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_size_bytes = checkpoint_path.stat().st_size
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    spec = service.get_experiment(run.experiment_id)

    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("El checkpoint no contiene metadata válida.")
    if metadata.get("run_id") != run.run_id:
        raise ValueError("El checkpoint no corresponde a esta ejecución.")

    expected_fingerprints = metadata.get("fingerprints")
    if not isinstance(expected_fingerprints, dict):
        raise ValueError("El checkpoint no contiene fingerprints de los datasets.")

    config = _configuracion_del_checkpoint(checkpoint, spec, service)
    source_assignments = [
        item for item in spec.datasets if item.role == "train_source"
    ]
    source_names = [item.source_name for item in source_assignments]
    path_overrides = {
        item.source_name: item.path for item in source_assignments
    }

    if len(source_names) != 1:
        raise ValueError("Este benchmark requiere una única fuente de test.")

    prepared = prepare_sources(
        ProjectPaths(root),
        config,
        requested_sources=source_names,
        path_overrides=path_overrides,
        create_missing_splits=False,
        regenerate_splits=False,
        source_assignments=source_assignments,
    )

    source_name = source_names[0]
    if source_name not in expected_fingerprints:
        raise ValueError(
            f"El checkpoint no contiene el fingerprint de '{source_name}'."
        )

    actual_fingerprint = dataset_fingerprint(prepared[source_name].dataset)
    if actual_fingerprint != expected_fingerprints[source_name]:
        raise ValueError(
            "El fingerprint del dataset actual no coincide con el utilizado "
            "durante el entrenamiento. Benchmark cancelado por integridad."
        )

    dataset = subset_for(
        prepared[source_name],
        "test",
        build_eval_transform(config=config),
    )

    model = crear_modelo(
        checkpoint["architecture"],
        num_classes=int(checkpoint["num_classes"]),
        pretrained=False,
        dispositivo=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    measurements = []
    for batch_size in batch_sizes:
        measurements.append(
            _medir_batch_size(
                model,
                dataset,
                batch_size=batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                device=device,
                warmup_batches=warmup_batches,
            )
        )

    result = {
        "run_id": run.run_id,
        "experiment_id": run.experiment_id,
        "model": checkpoint["model"],
        "architecture": checkpoint["architecture"],
        "device": str(device),
        "partition": "test",
        "dataset_fingerprint": actual_fingerprint,
        "warmup_batches": warmup_batches,
        "checkpoint": {
            "size_bytes": checkpoint_size_bytes,
            "size_mb": round(checkpoint_size_bytes / (1024 ** 2), 4),
        },
        "measurements": measurements,
        "environment": {
            "training": {
                "device": metadata.get("device"),
                "gpu_name": metadata.get("gpu_name"),
                "python_version": metadata.get("python_version"),
                "torch_version": metadata.get("torch_version"),
                "cuda_version": metadata.get("cuda_version"),
            },
            "benchmark": {
                "device": str(device),
                "gpu_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                "python_version": platform.python_version(),
                "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
            },
        },
        "notes": {
            "model_forward": (
                "Solo forward del modelo; los tensores ya están en el dispositivo."
            ),
            "transfer": "Solo transferencia CPU-dispositivo, sincronizada.",
            "end_to_end": (
                "DataLoader, lectura, transformaciones, transferencia y forward."
            ),
            "percentiles": (
                "P50/P95 calculados sobre lotes; las métricas por imagen "
                "normalizan cada lote por su cantidad real de imágenes."
            ),
        },
    }

    output_path = (
        root / "results" / "experiments"
        / run.experiment_id / run.run_id / "benchmark.json"
    )
    _guardar_json_atomico(output_path, result)
    result["benchmark_path"] = str(output_path.relative_to(root))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark de inferencia de una ejecución completada."
    )
    parser.add_argument("--root", default="/workspace")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument(
        "--batch-sizes",
        default="1,32",
        help="Tamaños separados por coma. Ejemplo: 1,32",
    )
    args = parser.parse_args()

    try:
        selected_batch_sizes = [
            int(value.strip())
            for value in args.batch_sizes.split(",")
            if value.strip()
        ]
    except ValueError as exc:
        parser.error(f"--batch-sizes inválido: {exc}")

    output = benchmark_ejecucion(
        args.root,
        args.run_id,
        warmup_batches=args.warmup_batches,
        batch_sizes=selected_batch_sizes,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
