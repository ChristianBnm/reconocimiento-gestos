from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch

from src.core.configuracion.settings import (
    build_experiment_config,
    list_models,
)
from src.core.modelos.factory import cargar_checkpoint, crear_modelo
from src.core.utils.paths import ProjectPaths


def sincronizar(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentil(valores: list[float], p: float) -> float:
    """Calcula un percentil por nearest-rank, sin depender de NumPy."""
    ordenados = sorted(valores)
    indice = max(0, math.ceil((p / 100.0) * len(ordenados)) - 1)
    return ordenados[indice]


def medir_modelo(
    modelo_nombre: str,
    paths: ProjectPaths,
    device: torch.device,
    warmup: int,
    runs: int,
) -> dict:
    config = build_experiment_config(modelo_nombre)
    checkpoint = paths.model_checkpoint_dir(modelo_nombre) / "mejor.pth"

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No existe el checkpoint de {modelo_nombre}: {checkpoint}"
        )

    # Cargamos la arquitectura sin pesos preentrenados.
    # Los pesos entrenados se cargan desde nuestro checkpoint.
    model = crear_modelo(
        config.model.architecture,
        config.num_classes,
        pretrained=False,
    )
    cargar_checkpoint(model, checkpoint, device)
    model.eval()

    parametros = sum(p.numel() for p in model.parameters())
    bytes_parametros = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    bytes_buffers = sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    tam_checkpoint = checkpoint.stat().st_size

    # Simulamos una entrada individual de 224x224.
    # No se incluye la lectura ni la transformación de imágenes.
    x = torch.randn(
        1,
        3,
        config.image_size,
        config.image_size,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        memoria_base = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        memoria_base = None

    tiempos_ms: list[float] = []

    with torch.inference_mode():
        # Calentamiento: excluido de las estadísticas.
        for _ in range(warmup):
            _ = model(x)

        sincronizar(device)

        # Medición de inferencias individuales.
        for _ in range(runs):
            sincronizar(device)
            inicio = time.perf_counter()

            _ = model(x)

            sincronizar(device)
            tiempos_ms.append(
                (time.perf_counter() - inicio) * 1000.0
            )

    if device.type == "cuda":
        memoria_pico = torch.cuda.max_memory_allocated(device)
        memoria_reservada_pico = torch.cuda.max_memory_reserved(device)
        memoria_extra_pico = max(
            0,
            memoria_pico - memoria_base,
        )
    else:
        memoria_pico = None
        memoria_reservada_pico = None
        memoria_extra_pico = None

    resultado = {
        "modelo": modelo_nombre,
        "familia": config.model.family,
        "arquitectura": config.model.architecture,
        "dispositivo": str(device),
        "tamano_entrada": [
            1,
            3,
            config.image_size,
            config.image_size,
        ],
        "parametros_totales": parametros,
        "parametros_mb_aproximados": round(
            bytes_parametros / (1024**2), 3
        ),
        "buffers_mb_aproximados": round(
            bytes_buffers / (1024**2), 3
        ),
        "checkpoint": paths.relative_to_root(checkpoint),
        "checkpoint_mb": round(
            tam_checkpoint / (1024**2), 3
        ),
        "warmup": warmup,
        "repeticiones": runs,
        "latencia_media_ms": round(
            statistics.mean(tiempos_ms), 3
        ),
        "latencia_mediana_ms": round(
            statistics.median(tiempos_ms), 3
        ),
        "latencia_p95_ms": round(
            percentil(tiempos_ms, 95), 3
        ),
        "throughput_aproximado_imagenes_s": round(
            1000.0 / statistics.mean(tiempos_ms), 2
        ),
        "gpu_memoria_asignada_pico_mb": (
            round(memoria_pico / (1024**2), 2)
            if memoria_pico is not None
            else None
        ),
        "gpu_memoria_extra_pico_mb": (
            round(memoria_extra_pico / (1024**2), 2)
            if memoria_extra_pico is not None
            else None
        ),
        "gpu_memoria_reservada_pico_mb": (
            round(memoria_reservada_pico / (1024**2), 2)
            if memoria_reservada_pico is not None
            else None
        ),
        "alcance_medicion": (
            "forward del modelo con batch=1; excluye lectura, "
            "preprocesamiento y postprocesamiento"
        ),
    }

    del x, model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mide tamaño y latencia de inferencia "
            "de los modelos entrenados."
        )
    )

    parser.add_argument(
        "--root",
        default=".",
        help="Raíz del repositorio",
    )

    parser.add_argument(
        "--modelos",
        nargs="+",
        choices=list_models(),
        default=list_models(),
        help="Modelos a medir; por defecto mide todos",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Inferencias de calentamiento",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Inferencias medidas por modelo",
    )

    parser.add_argument(
        "--output-dir",
        default="results/benchmark",
        help="Directorio de salida relativo a la raíz",
    )

    args = parser.parse_args()

    if args.warmup < 0 or args.runs < 1:
        parser.error(
            "--warmup debe ser >= 0 y --runs debe ser >= 1"
        )

    root = Path(args.root).resolve()
    paths = ProjectPaths(root)

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    gpu_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None
    )

    print(
        f"Dispositivo: {device}"
        + (f" ({gpu_name})" if gpu_name else "")
    )
    print(
        f"Warm-up: {args.warmup}; "
        f"repeticiones medidas: {args.runs}\n"
    )

    resultados = []

    for name in args.modelos:
        print(f"Midiendo {name}...")

        resultado = medir_modelo(
            name,
            paths,
            device,
            args.warmup,
            args.runs,
        )

        resultados.append(resultado)

        print(
            f"  params={resultado['parametros_totales']:,} | "
            f"checkpoint={resultado['checkpoint_mb']} MB | "
            f"latencia mediana="
            f"{resultado['latencia_mediana_ms']} ms | "
            f"p95={resultado['latencia_p95_ms']} ms"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "fecha_local_ejecucion": (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_disponible": torch.cuda.is_available(),
        "version_cuda_pytorch": torch.version.cuda,
        "gpu": gpu_name,
        "nota": (
            "Medición de inferencia del forward del modelo con "
            "batch=1. No incluye lectura de imagen, transformaciones, "
            "DataLoader ni postprocesamiento."
        ),
        "resultados": resultados,
    }

    json_path = (
        output_dir / f"benchmark_modelos_{stamp}.json"
    )
    csv_path = (
        output_dir / f"benchmark_modelos_{stamp}.csv"
    )

    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    columnas = (
        list(resultados[0].keys()) if resultados else []
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columnas,
        )
        writer.writeheader()
        writer.writerows(resultados)

    print("\nArchivos generados:")
    print(f"  {paths.relative_to_root(json_path)}")
    print(f"  {paths.relative_to_root(csv_path)}")


if __name__ == "__main__":
    main()