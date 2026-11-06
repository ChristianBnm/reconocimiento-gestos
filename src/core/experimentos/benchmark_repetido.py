
"""Repite benchmarks de inferencia y resume sus medianas."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from src.core.experimentos.benchmark import benchmark_ejecucion
from src.core.experimentos.evaluation import _guardar_json_atomico


def repetir_benchmark(
    root: str | Path,
    run_id: str,
    *,
    repeticiones: int = 5,
    warmup_batches: int = 5,
) -> dict:
    if (not isinstance(repeticiones, int) or isinstance(repeticiones, bool) or repeticiones < 3):
        raise ValueError(
            "La cantidad de repeticiones debe ser un entero de al menos 3."
        )

    root = Path(root).expanduser().resolve()
    resultados = []

    for numero in range(1, repeticiones + 1):
        print(
            f"Medición {numero}/{repeticiones} para run {run_id}",
            flush=True,
        )
        resultado = benchmark_ejecucion(
            root,
            run_id,
            warmup_batches=warmup_batches,
        )
        resultado.pop("benchmark_path", None)
        resultados.append({
            "repeticion": numero,
            **resultado,
        })

    campos_consistencia = (
        "experiment_id",
        "model",
        "architecture",
        "device",
        "partition",
        "num_samples",
        "batch_size",
    )

    referencia = resultados[0]

    for resultado in resultados[1:]:
        diferencias = [
            campo
            for campo in campos_consistencia
            if resultado[campo] != referencia[campo]
        ]

        if diferencias:
            raise ValueError(
                "Las mediciones no son comparables. "
                f"Diferencias en: {', '.join(diferencias)}."
            )

    metricas = (
        "model_inference_seconds",
        "model_ms_per_image",
        "model_images_per_second",
        "end_to_end_seconds",
        "end_to_end_ms_per_image",
    )

    resumen = {
        nombre: {
            "mediana": statistics.median(
                r[nombre] for r in resultados
            ),
            "minimo": min(r[nombre] for r in resultados),
            "maximo": max(r[nombre] for r in resultados),
        }
        for nombre in metricas
    }

    salida = {
        "run_id": run_id,
        "model": resultados[0]["model"],
        "device": resultados[0]["device"],
        "num_samples": resultados[0]["num_samples"],
        "repeticiones": repeticiones,
        "warmup_batches": warmup_batches,
        "resultados": resultados,
        "resumen": resumen,
    }

    output_path = (
        root / "results" / "experiments"
        / resultados[0]["experiment_id"]
        / run_id / "benchmark_repeticiones.json"
    )

    _guardar_json_atomico(output_path, salida)

    print(json.dumps({
        "model": salida["model"],
        "repeticiones": repeticiones,
        "resumen": resumen,
        "archivo": str(output_path.relative_to(root)),
    }, ensure_ascii=False, indent=2))

    return salida


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--warmup-batches", type=int, default=5)
    args = parser.parse_args()

    repetir_benchmark(
        args.root,
        args.run_id,
        repeticiones=args.repeticiones,
        warmup_batches=args.warmup_batches,
    )
