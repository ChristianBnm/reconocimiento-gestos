from __future__ import annotations

import argparse
import json
import shutil
import platform
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.core.configuracion.settings import (
    build_experiment_config,
    cargar_clases,
    get_model_spec,
    list_models,
)


from src.core.dataset.split import dataset_fingerprint
from src.core.dataset.experiment import (
    SCENARIO_SOURCES,
    SOURCE_NAMES,
    PreparedSource,
    prepare_sources,
    subset_for,
)


from src.core.dataset.transforms import build_eval_transform, build_train_transform
from src.core.evaluacion.metrics import calcular_metricas, generar_informe_clasificacion
from src.core.evaluacion.predictor import evaluar_modelo, guardar_predicciones, predecir_imagen
from src.core.evaluacion.reporting import (
    guardar_errores_prediccion,
    guardar_matriz_confusion,
)
from src.core.modelos.factory import (
    adaptar_normalizacion_a_modelo, cargar_checkpoint, congelar_backbone,
    crear_modelo, descongelar_modelo,
)


from src.core.entrenamiento.trainer import Trainer
from src.core.utils.io import load_json, save_json
from src.core.utils.paths import ProjectPaths
from src.core.utils.reproducibilidad import establecer_semilla


def _class_names(args, root: Path) -> tuple[str, ...]:
    class_file = args.archivo_clases
    if class_file:
        candidate = Path(class_file).expanduser()
        if not candidate.is_absolute() and not candidate.is_file():
            candidate = root / candidate
        return cargar_clases(candidate)
    return cargar_clases()


def _path_overrides(args) -> dict[str, str | Path]:
    result: dict[str, str | Path] = {}
    if args.dataset_original:
        result["original"] = args.dataset_original
    if args.dataset_complementario:
        result["complementario"] = args.dataset_complementario
    for item in args.fuente_ruta or []:
        if "=" not in item:
            raise ValueError("Cada --fuente-ruta debe tener el formato NOMBRE=RUTA.")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError("Cada --fuente-ruta debe tener nombre y ruta no vacíos.")
        if name in result:
            raise ValueError(f"Se indicó más de una ruta para la fuente '{name}'.")
        result[name] = path
    return result


def _build_config(args, class_names):
    overrides = {
        "seed": args.semilla,
        "split_seed": args.semilla_split,
        "batch_size": args.lote,
        "epochs": args.epocas,
        "num_workers": args.workers,
        "freeze_backbone": not args.ajuste_fino,
    }
    return build_experiment_config(args.modelo, class_names=class_names, **overrides)


def _loader(dataset, config, device, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2**32)
        import random
        import numpy as np
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=(config.pin_memory and device.type == "cuda"),
        generator=generator,
        worker_init_fn=seed_worker if config.num_workers > 0 else None,
        persistent_workers=(config.num_workers > 0),
    )


def _save_experiment_config(
    paths: ProjectPaths,
    results_dir: Path,
    config,
    scenario: str,
    training_sources: list[str],
    prepared: dict[str, PreparedSource],
):
    payload = config.to_dict()
    payload.update({
        "scenario": scenario,
        "training_sources": training_sources,
        "source_paths": {
            name: paths.relative_to_root(source.dataset.carpetas_imagenes[0])
            for name, source in prepared.items()
        },
        "splits": {
            name: {
                "path": paths.relative_to_root(source.split_path),
                "fingerprint": dataset_fingerprint(source.dataset),
                "counts": {
                    "unique_records": len(source.dataset),
                    "exact_duplicate_files_excluded": len(source.dataset.excluded_duplicates),
                    "train": len(source.split.train),
                    "val": len(source.split.val),
                    "test": len(source.split.test),
                },
            }
            for name, source in prepared.items()
        },
    })
    save_json(payload, results_dir / "config.json")


def command_train(args):
    paths = ProjectPaths(Path(args.root).expanduser().resolve())
    class_names = _class_names(args, paths.root)
    config = _build_config(args, class_names)
    scenario_sources = SCENARIO_SOURCES[args.escenario]
    checkpoint_dir, results_dir = paths.experiment_dirs(args.escenario, args.modelo)
    checkpoint_path = checkpoint_dir / "mejor.pth"

    has_existing_artifacts = checkpoint_path.exists() or (
        results_dir.exists() and any(results_dir.iterdir())
    )
    if has_existing_artifacts and not args.sobrescribir:
        raise FileExistsError(
            f"Ya existen artefactos para {args.escenario}/{args.modelo}. "
            "--sobrescribir para reemplazar únicamente este experimento."
        )
    if args.sobrescribir:
        # La eliminación se restringe al directorio exclusivo del experimento elegido.
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        if results_dir.exists():
            shutil.rmtree(results_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    establecer_semilla(config.seed)
    prepared = prepare_sources(
        paths,
        config,
        requested_sources=SOURCE_NAMES,
        path_overrides=_path_overrides(args),
        create_missing_splits=True,
        regenerate_splits=args.regenerar_splits,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = crear_modelo(
        config.model.architecture,
        config.num_classes,
        pretrained=config.model.pretrained,
        dispositivo=device,
    )


    # La normalización queda alineada con los metadatos del checkpoint preentrenado
    # y viaja dentro de la configuración guardada en el checkpoint del experimento.
    config = adaptar_normalizacion_a_modelo(config, model)
    train_transform = build_train_transform(config=config)
    eval_transform = build_eval_transform(config=config)
    train_parts = [
        subset_for(prepared[name], "train", train_transform)
        for name in scenario_sources
    ]
    val_parts = [
        subset_for(prepared[name], "val", eval_transform)
        for name in scenario_sources
    ]
    train_dataset = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_dataset = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)

    train_loader = _loader(train_dataset, config, device, shuffle=True, seed=config.seed)
    val_loader = _loader(val_dataset, config, device, shuffle=False, seed=config.seed)
    model = congelar_backbone(model) if config.freeze_backbone else descongelar_modelo(model)

    _save_experiment_config(
        paths, results_dir, config, args.escenario, list(scenario_sources), prepared
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_path=checkpoint_path,
        history_path=results_dir / "history.json",
        metadata={
            "scenario": args.escenario,
            "training_sources": list(scenario_sources),
            "split_paths": {name: paths.relative_to_root(source.split_path) for name, source in prepared.items()},
            "source_paths": {
                name: paths.relative_to_root(source.dataset.carpetas_imagenes[0])
                for name, source in prepared.items()
            },
            "torch_version": str(torch.__version__),
            "device": str(device),
        },
    )
    # Medición de recursos del entrenamiento. La memoria CUDA corresponde
    # al asignador de PyTorch; la RAM máxima se informa con la métrica del SO.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    trainer.fit()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started

    try:
        import resource
        max_rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reporta KiB; macOS reporta bytes.
        max_rss_mb = max_rss_raw / (1024 if platform.system() == "Linux" else 1024 * 1024)
    except (ImportError, AttributeError, OSError):
        max_rss_mb = None

    try:
        checkpoint_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    except OSError:
        checkpoint_mb = None

    epoch_history_path = results_dir / "history.json"
    epoch_history = load_json(epoch_history_path) if epoch_history_path.is_file() else []
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    resource_metrics = {
        "scenario": args.escenario,
        "model": args.modelo,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "epochs_completed": len(epoch_history),
        "training_time_seconds": round(training_seconds, 4),
        "mean_epoch_time_seconds": round(sum(row.get("epoch_seconds", 0.0) for row in epoch_history) / len(epoch_history), 4) if epoch_history else None,
        "max_process_ram_mb": round(max_rss_mb, 2) if max_rss_mb is not None else None,
        "gpu_peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2) if device.type == "cuda" else None,
        "gpu_peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / (1024 * 1024), 2) if device.type == "cuda" else None,
        "parameters_total": total_parameters,
        "parameters_trainable": trainable_parameters,
        "checkpoint_size_mb": round(checkpoint_mb, 3) if checkpoint_mb is not None else None,
        "note_ram": "Máximo RSS del proceso según el sistema operativo; no equivale a memoria exclusiva del modelo.",
        "note_gpu_memory": "Pico de memoria gestionada por el asignador de PyTorch; no incluye necesariamente todas las asignaciones externas a PyTorch.",
    }
    save_json(resource_metrics, results_dir / "metricas_recursos_entrenamiento.json")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Resultados: {results_dir}")
    print(f"Métricas de recursos: {results_dir / 'metricas_recursos_entrenamiento.json'}")


def _read_checkpoint_metadata(checkpoint_path: Path, device: torch.device) -> dict:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    return payload if isinstance(payload, dict) and "model_state_dict" in payload else {}


def _config_for_checkpoint(args, checkpoint_path: Path, device: torch.device):
    metadata = _read_checkpoint_metadata(checkpoint_path, device)
    saved_names = metadata.get("class_names")
    class_names = tuple(saved_names) if saved_names else _class_names(
        args, Path(args.root).expanduser().resolve()
    )
    saved_config = metadata.get("experiment_config", {})
    model_name = metadata.get("model", args.modelo)
    architecture = metadata.get("architecture", get_model_spec(args.modelo).architecture)
    spec = get_model_spec(model_name) if model_name in list_models() else get_model_spec(args.modelo)
    if spec.architecture != architecture:
        from src.core.configuracion.settings import ModelSpec
        spec = ModelSpec(model_name, architecture, spec.family, pretrained=False)

    # En evaluación e inferencia prevalece la configuración guardada en el checkpoint.
    valid_fields = {
        key: saved_config[key]
        for key in (
            "image_size", "train_size", "val_size", "test_size", "seed", "split_seed", "batch_size",
            "learning_rate", "weight_decay", "epochs", "patience", "min_delta", "num_workers",
            "pin_memory", "freeze_backbone", "horizontal_flip_probability", "affine_probability",
            "brightness_contrast_probability", "normalization_mean", "normalization_std",
        )
        if key in saved_config
    }
    for key in ("normalization_mean", "normalization_std"):
        if key in valid_fields:
            valid_fields[key] = tuple(valid_fields[key])
    if "normalization_mean" not in valid_fields:
        # El pipeline anterior normalizaba con media/desviación 0.5 por canal.
        valid_fields["normalization_mean"] = (0.5, 0.5, 0.5)
        valid_fields["normalization_std"] = (0.5, 0.5, 0.5)
    if model_name != args.modelo:
        raise ValueError(
            f"El checkpoint indica el modelo '{model_name}', pero se solicitó '{args.modelo}'."
        )
    config = build_experiment_config(model_name, class_names=class_names, **valid_fields)
    return config, metadata, spec


def command_evaluate(args):
    paths = ProjectPaths(Path(args.root).expanduser().resolve())
    checkpoint_path = paths.model_checkpoint_dir(args.modelo, args.escenario) / "mejor.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, metadata, spec = _config_for_checkpoint(args, checkpoint_path, device)
    results_dir = paths.model_results_dir(args.modelo, args.escenario)
    results_dir.mkdir(parents=True, exist_ok=True)
    saved_path_overrides = {}
    saved_metadata = metadata.get("metadata", {}) if isinstance(metadata, dict) else {}
    saved_path_overrides.update(saved_metadata.get("source_paths", {}))
    saved_config_path = results_dir / "config.json"
    if not saved_path_overrides and saved_config_path.is_file():
        saved_path_overrides.update(load_json(saved_config_path).get("source_paths", {}))
    # Las opciones de la línea de comandos prevalecen sobre las rutas guardadas.
    saved_path_overrides.update(_path_overrides(args))
    prepared = prepare_sources(
        paths,
        config,
        requested_sources=SOURCE_NAMES,
        path_overrides=saved_path_overrides,
        create_missing_splits=False,
    )
    model = crear_modelo(
        spec.architecture,
        config.num_classes,
        pretrained=False,
        dispositivo=device,
    )
    cargar_checkpoint(
        model,
        checkpoint_path,
        device,
        class_names=config.class_names,
        expected_architecture=spec.architecture,
    )
    eval_transform = build_eval_transform(config=config)
    selected_sources = SOURCE_NAMES if args.fuente == "ambas" else (args.fuente,)
    summary = {"scenario": args.escenario, "model": args.modelo, "tests": {}}

    for source_name in selected_sources:
        source = prepared[source_name]
        test_dataset = subset_for(source, "test", eval_transform)
        result = evaluar_modelo(
            model,
            test_dataset,
            config.batch_size,
            device,
            num_workers=config.num_workers,
        )
        metrics = calcular_metricas(result.y_true, result.y_pred)
        report = generar_informe_clasificacion(result.y_true, result.y_pred, config.class_names)
        prefix = f"{args.escenario}_{args.modelo}_{source_name}"
        predictions_path = results_dir / f"predicciones_{source_name}.csv"
        guardar_predicciones(result, source.dataset, paths.root, predictions_path)
        error_summary = guardar_errores_prediccion(
            predictions_path,
            paths.root,
            results_dir / "errores" / source_name,
            config.class_names,
        )
        save_json(error_summary, results_dir / f"resumen_errores_{source_name}.json")
        print(
            f"Errores {source_name}: {error_summary['total_errores']}/"
            f"{error_summary['total_evaluadas']} | "
            f"imágenes copiadas: {error_summary['imagenes_copiadas']} | "
            f"no encontradas: {error_summary['imagenes_no_encontradas']}"
        )
        save_json(metrics, results_dir / f"metrics_{source_name}.json")
        save_json(report, results_dir / f"classification_report_{source_name}.json")
        guardar_matriz_confusion(
            result.y_true,
            result.y_pred,
            config.class_names,
            results_dir,
            prefix=f"matriz_confusion_{source_name}",
        )
        summary["tests"][source_name] = metrics
        print(f"\nEvaluación sobre {source_name}:")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    save_json(summary, results_dir / "resumen_evaluacion.json")
    print(f"Artefactos: {results_dir}")


def command_infer(args):
    paths = ProjectPaths(Path(args.root).expanduser().resolve())
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else paths.model_checkpoint_dir(args.modelo, args.escenario) / "mejor.pth"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, _metadata, spec = _config_for_checkpoint(args, checkpoint_path, device)
    model = crear_modelo(
        spec.architecture,
        config.num_classes,
        pretrained=False,
        dispositivo=device,
    )
    cargar_checkpoint(
        model,
        checkpoint_path,
        device,
        class_names=config.class_names,
        expected_architecture=spec.architecture,
    )
    transform = build_eval_transform(config=config)
    label, confidence = predecir_imagen(
        model,
        Path(args.imagen),
        transform,
        device,
        class_names=config.class_names,
    )
    print(f"Predicción: {label}")
    print(f"Confianza: {confidence:.4f}")


def _add_common_arguments(parser):
    parser.add_argument("--root", default=".", help="Raíz del proyecto.")
    parser.add_argument("--archivo-clases", help="JSON de clases alternativo; por defecto usa src/core/configuracion/clases.json.")
    parser.add_argument("--dataset-original", help="Ruta alternativa al dataset original.")
    parser.add_argument("--dataset-complementario", help="Ruta alternativa al dataset complementario.")
    parser.add_argument(
        "--fuente-ruta", action="append", default=[], metavar="NOMBRE=RUTA",
        help="Sobrescribe la ruta de una fuente definida en fuentes.json; se puede repetir.",
    )


def _add_training_arguments(parser):
    parser.add_argument("--semilla", type=int, default=42, help="Semilla de inicialización/entrenamiento.")
    parser.add_argument("--semilla-split", type=int, default=42, help="Semilla independiente para las particiones compartidas del dataset.")
    parser.add_argument("--lote", type=int, default=32)
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--ajuste-fino", action="store_true", help="Entrena todos los parámetros en vez de congelar el backbone.")


def build_parser():
    parser = argparse.ArgumentParser(description="Pipeline extensible de reconocimiento de gestos estáticos.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("entrenar", aliases=["train"])
    _add_common_arguments(train)
    _add_training_arguments(train)
    train.add_argument("--modelo", required=True, choices=list_models())
    train.add_argument("--escenario", choices=list(SCENARIO_SOURCES), default="combinado")
    train.add_argument("--regenerar-splits", action="store_true", help="Regenera particiones guardadas; usar solo si se desea cambiar la división compartida.")
    train.add_argument("--sobrescribir", action="store_true", help="Reemplaza los artefactos solo de este escenario/modelo.")
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluar", aliases=["evaluate"])
    _add_common_arguments(evaluate)
    evaluate.add_argument("--modelo", required=True, choices=list_models())
    evaluate.add_argument("--escenario", choices=list(SCENARIO_SOURCES), default="combinado")
    evaluate.add_argument(
        "--fuente", choices=[*SOURCE_NAMES, "ambas"], default="ambas",
        help="Fuente de prueba; por defecto evalúa todas las fuentes configuradas independientemente.",
    )
    evaluate.set_defaults(func=command_evaluate)

    infer = subparsers.add_parser("inferir", aliases=["infer"])
    _add_common_arguments(infer)
    infer.add_argument("--modelo", required=True, choices=list_models())
    infer.add_argument("--escenario", choices=list(SCENARIO_SOURCES), default="combinado")
    infer.add_argument("--imagen", required=True)
    infer.add_argument("--checkpoint", help="Ruta explícita a un checkpoint; permite inferencia con checkpoints legacy.")
    infer.set_defaults(func=command_infer)

    info = subparsers.add_parser("modelos", help="Lista las arquitecturas disponibles.")
    info.set_defaults(func=lambda args: [
        print(f"{name}: {get_model_spec(name).architecture} ({get_model_spec(name).family})")
        for name in list_models()
    ])

    classes = subparsers.add_parser("clases", help="Lista el vocabulario y su índice configurado.")
    classes.add_argument("--archivo-clases")
    classes.set_defaults(func=lambda args: [
        print(f"{index:>3}  {name}")
        for index, name in enumerate(cargar_clases(args.archivo_clases))
    ])

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
