#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark de inferencia para los cuatro modelos de la Fase A.
No modifica el frontend ni vuelve a entrenar modelos.

Ejecutar desde la raíz del proyecto:
    python scripts/benchmark_inferencia.py
Opcional:
    python scripts/benchmark_inferencia.py --device cuda --repeticiones 3

Mide:
- Inferencia del modelo con tensores ya preparados (model-only).
- Pipeline de inferencia con lectura, preprocesamiento y modelo (end-to-end).
- Batch 1 y batch 32.
- Latencia, percentiles, throughput y pico de memoria CUDA.
Los resultados se guardan en results/benchmark_inferencia/.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.configuracion.settings import build_experiment_config
from src.core.dataset.loader import (
    DatasetSubset,
    GestosDataset,
    cargar_split,
)
from src.core.dataset.transforms import build_eval_transform
from src.core.modelos.factory import crear_modelo, cargar_checkpoint, descongelar_modelo


RUNS = [
    ("MNET-001", "MNET-001-R01", "mobilenetv2"),
    ("DENS-001", "DENS-001-R01", "densenet201"),
    ("VITT-001", "VITT-001-R01", "vit_tiny"),
    ("VITB-001", "VITB-001-R01", "vit_base"),
]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_run(root: Path, experiment_id: str, run_id: str, model_name: str,
             device: torch.device, source_override: Path | None):
    run_dir = root / "results" / "experiments" / experiment_id / run_id
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No existe la configuración: {config_path}")

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = saved.get("metadata", {})
    class_names = saved.get("class_names")
    if not class_names:
        raise ValueError(f"No hay class_names en {config_path}")

    # Recupera exactamente los parámetros de evaluación guardados.
    config_fields = {
        "image_size", "train_size", "val_size", "test_size", "seed",
        "split_seed", "batch_size", "learning_rate", "weight_decay",
        "epochs", "patience", "min_delta", "num_workers", "pin_memory",
        "freeze_backbone", "horizontal_flip_probability",
        "affine_probability", "brightness_contrast_probability",
        "normalization_mean", "normalization_std",
    }
    overrides = {k: saved[k] for k in config_fields if k in saved}
    overrides["device"] = device.type
    config = build_experiment_config(model_name, class_names=class_names, **overrides)

    split_path_value = metadata.get("split_paths", {}).get("original")
    if not split_path_value:
        raise ValueError(f"No se encontró split original en {config_path}")
    split_path = Path(split_path_value)
    if not split_path.is_absolute():
        split_path = root / split_path

    source_value = (
        str(source_override) if source_override
        else saved.get("dataset_assignments", [{}])[0].get("path", "data/Clases_200")
    )
    source_path = Path(source_value)
    if not source_path.is_absolute():
        source_path = root / source_path

    dataset = GestosDataset(
        source_path,
        transformacion=None,
        class_names=class_names,
        source_names=["original"],
        deduplicate_exact_duplicates=True,
    )
    split = cargar_split(
        split_path, dataset,
        expected_seed=int(saved.get("split_seed", saved.get("seed", 42))),
        train_size=float(saved.get("train_size", 0.70)),
        val_size=float(saved.get("val_size", 0.15)),
        test_size=float(saved.get("test_size", 0.15)),
    )
    eval_dataset = DatasetSubset(
        dataset, split.test, transformacion=build_eval_transform(config=config)
    )

    checkpoint = root / "checkpoints" / "experiments" / experiment_id / run_id / "mejor.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No se encontró el mejor checkpoint: {checkpoint}\n"
            "Verificá que la carpeta checkpoints/ esté presente en el proyecto."
        )

    model = crear_modelo(
        arquitectura=config.model.architecture,
        num_classes=config.num_classes,
        pretrained=False,
        dispositivo=device,
    )
    cargar_checkpoint(
        model, checkpoint, device,
        class_names=class_names,
        expected_architecture=config.model.architecture,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, eval_dataset, config, checkpoint, len(split.test), metadata


def benchmark_model_only(model: torch.nn.Module, dataset, device: torch.device,
                         batch_size: int, warmup_batches: int,
                         repetitions: int) -> dict[str, Any]:
    # Usa ejemplos reales del test, transformados con el pipeline de evaluación.
    n = min(max(batch_size, 1), len(dataset))
    sample = torch.stack([dataset[i][0] for i in range(n)]).to(device)
    batch = sample[:batch_size]
    if batch.shape[0] < batch_size:
        batch = sample

    with torch.inference_mode():
        for _ in range(warmup_batches):
            model(batch)
        sync(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        batch_latencies = []
        for _ in range(repetitions):
            sync(device)
            start = time.perf_counter()
            model(batch)
            sync(device)
            batch_latencies.append((time.perf_counter() - start) * 1000.0)

    count = int(batch.shape[0])
    total_seconds = sum(batch_latencies) / 1000.0
    mean_batch_ms = statistics.mean(batch_latencies)
    return {
        "batch_size_real": count,
        "latencia_lote_media_ms": round(mean_batch_ms, 4),
        "latencia_por_imagen_media_ms": round(mean_batch_ms / count, 4),
        "latencia_lote_p50_ms": round(percentile(batch_latencies, 0.50), 4),
        "latencia_lote_p95_ms": round(percentile(batch_latencies, 0.95), 4),
        "throughput_imagenes_s": round((count * len(batch_latencies)) / total_seconds, 3)
            if total_seconds else None,
        "gpu_peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 3)
            if device.type == "cuda" else None,
        "gpu_peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 2**20, 3)
            if device.type == "cuda" else None,
    }


def benchmark_end_to_end(model: torch.nn.Module, dataset, device: torch.device,
                         batch_size: int, num_workers: int) -> dict[str, Any]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    latencies_ms: list[float] = []
    images_processed = 0
    iterator = iter(loader)

    with torch.inference_mode():
        while True:
            sync(device)
            start = time.perf_counter()
            try:
                images, _labels = next(iterator)
            except StopIteration:
                break
            images = images.to(device, non_blocking=True)
            model(images)
            sync(device)
            elapsed = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed)
            images_processed += int(images.shape[0])

    total_ms = sum(latencies_ms)
    return {
        "imagenes_procesadas": images_processed,
        "lotes_procesados": len(latencies_ms),
        "latencia_lote_media_ms": round(statistics.mean(latencies_ms), 4)
            if latencies_ms else None,
        "latencia_lote_p50_ms": round(percentile(latencies_ms, 0.50), 4),
        "latencia_lote_p95_ms": round(percentile(latencies_ms, 0.95), 4),
        "latencia_por_imagen_media_ms": round(total_ms / images_processed, 4)
            if images_processed else None,
        "throughput_imagenes_s": round(images_processed / (total_ms / 1000.0), 3)
            if total_ms else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark de inferencia de Fase A.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--repeticiones", type=int, default=100,
                        help="Repeticiones para medir el forward del modelo.")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Pasadas de calentamiento para el forward.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--source", type=Path, default=None,
                        help="Ruta al dataset definitivo; por defecto usa la ruta guardada en config.json.")
    parser.add_argument("--output", type=Path,
                        default=Path("results/benchmark_inferencia"))
    args = parser.parse_args()

    if args.repeticiones < 10 or args.warmup < 1:
        parser.error("--repeticiones debe ser >= 10 y --warmup >= 1.")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero PyTorch no detecta una GPU disponible.")

    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    environment = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "image_size": 224,
        "test_partition_only": True,
        "batches": [1, 32],
        "warmup_batches_model_only": args.warmup,
        "repeticiones_model_only": args.repeticiones,
        "num_workers_end_to_end": args.num_workers,
        "note": (
            "Model-only excluye decodificación/preprocesamiento. End-to-end incluye "
            "DataLoader, lectura, transformación, transferencia y forward. "
            "La sincronización CUDA se aplica antes y después de cada medición."
        ),
    }

    for experiment_id, run_id, model_name in RUNS:
        print(f"\n[{run_id}] Cargando modelo y partición test...")
        model, dataset, config, checkpoint, test_count, metadata = load_run(
            ROOT, experiment_id, run_id, model_name, device, args.source
        )
        if test_count != 1110:
            print(f"  AVISO: el test tiene {test_count} imágenes (se esperaban 1110).")

        model_storage_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in list(model.parameters()) + list(model.buffers())
        )
        model_parameters = sum(parameter.numel() for parameter in model.parameters())

        for batch_size in (1, 32):
            print(f"  Midiendo batch {batch_size}: model-only...")
            model_only = benchmark_model_only(
                model, dataset, device, batch_size,
                args.warmup, args.repeticiones,
            )
            print(f"  Midiendo batch {batch_size}: end-to-end...")
            end_to_end = benchmark_end_to_end(
                model, dataset, device, batch_size, args.num_workers
            )
            row = {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "model": model_name,
                "architecture": config.model.architecture,
                "device": str(device),
                "gpu_name": environment["gpu_name"],
                "batch_size": batch_size,
                "test_images": test_count,
                "parameters_total": model_parameters,
                "model_weights_and_buffers_mb": round(model_storage_bytes / 2**20, 3),
                "checkpoint_path": str(checkpoint.relative_to(ROOT)),
                **{f"model_only_{k}": v for k, v in model_only.items()},
                **{f"end_to_end_{k}": v for k, v in end_to_end.items()},
            }
            rows.append(row)
            print(
                f"    Model-only: {model_only['latencia_por_imagen_media_ms']} ms/imagen; "
                f"{model_only['throughput_imagenes_s']} img/s"
            )
            print(
                f"    End-to-end: {end_to_end['latencia_por_imagen_media_ms']} ms/imagen; "
                f"{end_to_end['throughput_imagenes_s']} img/s"
            )

        del model, dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = output_dir / "benchmark_inferencia_fase_A.csv"
    json_path = output_dir / "benchmark_inferencia_fase_A.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    payload = {
        "environment": environment,
        "methodology": {
            "dataset": "dataset_definitivo (ruta y fingerprint validados mediante split guardado)",
            "partition": "test",
            "input_resolution": "224x224",
            "preprocessing": "build_eval_transform del proyecto para cada arquitectura",
            "models": [run_id for _, run_id, _ in RUNS],
            "batches": [1, 32],
            "model_only": "Tensor ya decodificado y transformado; mide solo forward del modelo.",
            "end_to_end": "Incluye lectura/decodificación, transformación, DataLoader, transferencia y forward.",
            "repetitions": args.repeticiones,
            "warmup": args.warmup,
        },
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nBenchmark finalizado.")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
