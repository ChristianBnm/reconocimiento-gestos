
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from src.core.configuracion.settings import (
    cargar_clases,
    get_model_spec,
    list_models,
)
from src.core.dataset.experiment import SOURCE_NAMES
from src.core.evaluacion.predictor import predecir_imagen
from src.core.experimentos.benchmark import benchmark_ejecucion
from src.core.experimentos.evaluation import (
    _configuracion_del_checkpoint,
    evaluar_ejecucion,
)
from src.core.experimentos.schemas import (
    DATASET_ROLES,
    DatasetAssignment,
    ExperimentSpec,
)
from src.core.experimentos.service import ExperimentService
from src.core.modelos.factory import crear_modelo
from src.core.dataset.transforms import build_eval_transform


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------

def _resolve_root(args) -> Path:
    return Path(args.root).expanduser().resolve()


def _resolve_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if device_config == "cpu":
        return torch.device("cpu")

    if device_config == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Se solicitó CUDA, pero no está disponible."
            )
        return torch.device("cuda")

    raise ValueError(
        f"Dispositivo no válido: {device_config!r}."
    )


def _parse_dataset_assignments(
    values: list[str],
) -> tuple[DatasetAssignment, ...]:
    """
    Formato:
        NOMBRE=RUTA=ROL

    Ejemplo:
        original=data/Clases_200=train_source
    """
    assignments = []
    assigned_names = set()

    for value in values:
        name_path, separator, role = value.rpartition("=")

        if not separator:
            raise ValueError(
                f"Dataset inválido: {value!r}. "
                "Formato esperado: NOMBRE=RUTA=ROL."
            )

        name, separator, path = name_path.partition("=")

        name = name.strip()
        path = path.strip()
        role = role.strip()

        if not separator or not name or not path or not role:
            raise ValueError(
                f"Dataset inválido: {value!r}. "
                "Formato esperado: NOMBRE=RUTA=ROL."
            )

        if name not in SOURCE_NAMES:
            raise ValueError(
                f"Fuente desconocida: {name!r}. "
                f"Fuentes disponibles: {', '.join(SOURCE_NAMES)}."
            )

        if name in assigned_names:
            raise ValueError(
                f"La fuente {name!r} fue asignada más de una vez."
            )

        if role not in DATASET_ROLES:
            raise ValueError(
                f"Rol inválido: {role!r}. "
                f"Roles disponibles: {', '.join(DATASET_ROLES)}."
            )

        assigned_names.add(name)

        assignments.append(
            DatasetAssignment(
                source_name=name,
                path=path,
                role=role,
            )
        )

    if not assignments:
        raise ValueError(
            "Se debe especificar al menos un dataset."
        )

    if not any(
        item.role == "train_source"
        for item in assignments
    ):
        raise ValueError(
            "Se necesita al menos una fuente train_source."
        )

    return tuple(assignments)


def _build_parameters(args) -> dict:
    """
    Construye únicamente los parámetros indicados por el usuario.
    Los restantes conservan los valores predeterminados
    de ExperimentConfig.
    """
    parameters = {
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "device": args.device,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "horizontal_flip_probability": args.flip_probability,
        "affine_probability": args.affine_probability,
        "brightness_contrast_probability": (
            args.brightness_contrast_probability
        ),
    }

    parameters = {
        key: value
        for key, value in parameters.items()
        if value is not None
    }

    if args.ajuste_fino:
        parameters["freeze_backbone"] = False

    elif args.congelar_backbone:
        parameters["freeze_backbone"] = True

    return parameters


def _print_json(data) -> None:
    print(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


# ---------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------

def command_train(args) -> None:
    root = _resolve_root(args)
    service = ExperimentService(root)

    datasets = _parse_dataset_assignments(args.dataset)

    parameters = _build_parameters(args)

    experiment_id = (
        args.experiment_id
        or service.next_experiment_id(args.modelo)
    )

    spec = ExperimentSpec(
        name=args.nombre,
        description=args.descripcion or "",
        model_name=args.modelo,
        datasets=datasets,
        parameters=parameters,
        class_file=args.archivo_clases,
        experiment_id=experiment_id,
    )

    # Validación y persistencia de la definición.
    service.create_experiment(spec)

    # Se registra una ejecución correlativa.
    run = service.create_run(experiment_id)

    print(f"Experimento: {experiment_id}")
    print(f"Ejecución: {run.run_id}")
    print("Iniciando entrenamiento...")

    # El pipeline completo pertenece a ExperimentService.
    completed_run = service.run_experiment(
        experiment_id,
        run_id=run.run_id,
    )

    print(f"\nEstado: {completed_run.status}")
    print(f"Experimento: {completed_run.experiment_id}")
    print(f"Ejecución: {completed_run.run_id}")

    if completed_run.checkpoint_path:
        print(f"Checkpoint: {completed_run.checkpoint_path}")


# ---------------------------------------------------------------------
# Experimentos
# ---------------------------------------------------------------------

def command_list_experiments(args) -> None:
    service = ExperimentService(_resolve_root(args))

    experiments = [
        service.get_experiment(experiment_id).to_dict()
        for experiment_id in service.list_experiments()
    ]

    _print_json(experiments)


def command_show_experiment(args) -> None:
    service = ExperimentService(_resolve_root(args))

    spec = service.get_experiment(args.experiment_id)

    _print_json(spec.to_dict())


def command_delete_experiment(args) -> None:
    service = ExperimentService(_resolve_root(args))

    service.delete_experiment(args.experiment_id)

    print(
        f"Experimento eliminado: {args.experiment_id}"
    )


# ---------------------------------------------------------------------
# Ejecuciones
# ---------------------------------------------------------------------

def command_run(args) -> None:
    service = ExperimentService(_resolve_root(args))

    run = service.create_run(args.experiment_id)

    print(f"Ejecución creada: {run.run_id}")

    completed_run = service.run_experiment(
        args.experiment_id,
        run_id=run.run_id,
    )

    _print_json(completed_run.to_dict())


def command_list_runs(args) -> None:
    service = ExperimentService(_resolve_root(args))

    runs = service.list_runs(args.experiment_id)

    _print_json([
        run.to_dict()
        for run in runs
    ])


def command_show_run(args) -> None:
    service = ExperimentService(_resolve_root(args))

    run = service.get_run(args.run_id)

    _print_json(run.to_dict())


# ---------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------

def command_evaluate(args) -> None:
    root = _resolve_root(args)

    result = evaluar_ejecucion(
        root,
        args.run_id,
        overwrite=args.sobrescribir,
    )

    _print_json(result)


# ---------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------

def command_benchmark(args) -> None:
    result = benchmark_ejecucion(
        _resolve_root(args),
        args.run_id,
        warmup_batches=args.warmup_batches,
    )

    _print_json(result)


# ---------------------------------------------------------------------
# Inferencia individual
# ---------------------------------------------------------------------

def command_infer(args) -> None:
    root = _resolve_root(args)
    service = ExperimentService(root)

    run = service.get_run(args.run_id)

    if run.status != "completed":
        raise ValueError(
            "La ejecución debe estar completada para realizar inferencias."
        )

    if not run.checkpoint_path:
        raise ValueError(
            "La ejecución no tiene un checkpoint asociado."
        )

    checkpoint_path = Path(run.checkpoint_path)

    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No se encuentra el checkpoint: {checkpoint_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "El checkpoint no tiene un formato válido."
        )

    spec = service.get_experiment(run.experiment_id)

    config = _configuracion_del_checkpoint(
        checkpoint,
        spec,
        service,
    )

    model = crear_modelo(
        checkpoint["architecture"],
        num_classes=config.num_classes,
        pretrained=False,
        dispositivo=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    transform = build_eval_transform(config=config)

    label, confidence = predecir_imagen(
        model,
        Path(args.imagen),
        transform,
        device,
        class_names=config.class_names,
    )

    result = {
        "run_id": run.run_id,
        "experiment_id": run.experiment_id,
        "image": str(Path(args.imagen)),
        "prediction": label,
        "confidence": confidence,
        "device": str(device),
    }

    _print_json(result)


# ---------------------------------------------------------------------
# Información general
# ---------------------------------------------------------------------

def command_models(args) -> None:
    models = [
        {
            "name": name,
            "architecture": get_model_spec(name).architecture,
            "family": get_model_spec(name).family,
        }
        for name in list_models()
    ]

    _print_json(models)


def command_classes(args) -> None:
    class_file = args.archivo_clases

    if class_file:
        path = Path(class_file).expanduser()

        if not path.is_absolute():
            path = _resolve_root(args) / path

        classes = cargar_clases(path)
    else:
        classes = cargar_clases()

    for index, name in enumerate(classes):
        print(f"{index:>3}  {name}")


# ---------------------------------------------------------------------
# Argumentos comunes
# ---------------------------------------------------------------------

def _add_root_argument(parser) -> None:
    parser.add_argument(
        "--root",
        default=".",
        help="Directorio raíz del proyecto.",
    )


def _add_training_parameters(parser) -> None:
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
    )

    parser.add_argument(
        "--train-size",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--flip-probability",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--affine-probability",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--brightness-contrast-probability",
        type=float,
        default=None,
    )

    freeze_group = parser.add_mutually_exclusive_group()

    freeze_group.add_argument(
        "--ajuste-fino",
        action="store_true",
        help="Descongela y entrena todos los parámetros.",
    )

    freeze_group.add_argument(
        "--congelar-backbone",
        action="store_true",
        help="Congela el backbone y entrena el clasificador.",
    )


# ---------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "CLI para experimentos de reconocimiento "
            "de gestos estáticos LSA."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ENTRENAR
    train = subparsers.add_parser(
        "entrenar",
        aliases=["train"],
        help="Crear un experimento y ejecutar su entrenamiento.",
    )

    _add_root_argument(train)

    train.add_argument(
        "--modelo",
        required=True,
        choices=list_models(),
    )

    train.add_argument(
        "--nombre",
        required=True,
        help="Nombre descriptivo del experimento.",
    )

    train.add_argument(
        "--descripcion",
        default="",
    )

    train.add_argument(
        "--experiment-id",
        default=None,
        help="ID explícito. Si se omite, se genera automáticamente.",
    )

    train.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NOMBRE=RUTA=ROL",
        help=(
            "Asignación de dataset. Se puede repetir. "
            "Roles: train_source, final_test, external_eval, unused."
        ),
    )

    train.add_argument(
        "--archivo-clases",
        default=None,
        help="Archivo JSON alternativo de clases.",
    )

    _add_training_parameters(train)

    train.set_defaults(func=command_train)

    # EJECUTAR EXPERIMENTO EXISTENTE
    run = subparsers.add_parser(
        "ejecutar",
        aliases=["run"],
        help="Crear una ejecución de un experimento existente.",
    )

    _add_root_argument(run)

    run.add_argument(
        "--experiment-id",
        required=True,
    )

    run.set_defaults(func=command_run)

    # LISTAR EXPERIMENTOS
    experiments = subparsers.add_parser(
        "experimentos",
        aliases=["experiments"],
        help="Listar experimentos registrados.",
    )

    _add_root_argument(experiments)

    experiments.set_defaults(
        func=command_list_experiments
    )

    # CONSULTAR EXPERIMENTO
    show_experiment = subparsers.add_parser(
        "ver-experimento",
        help="Consultar una definición de experimento.",
    )

    _add_root_argument(show_experiment)

    show_experiment.add_argument(
        "--experiment-id",
        required=True,
    )

    show_experiment.set_defaults(
        func=command_show_experiment
    )

    # ELIMINAR EXPERIMENTO
    delete_experiment = subparsers.add_parser(
        "eliminar-experimento",
        help="Eliminar un experimento sin ejecuciones asociadas.",
    )

    _add_root_argument(delete_experiment)

    delete_experiment.add_argument(
        "--experiment-id",
        required=True,
    )

    delete_experiment.set_defaults(
        func=command_delete_experiment
    )

    # LISTAR EJECUCIONES
    runs = subparsers.add_parser(
        "ejecuciones",
        aliases=["runs"],
        help="Listar ejecuciones registradas.",
    )

    _add_root_argument(runs)

    runs.add_argument(
        "--experiment-id",
        default=None,
    )

    runs.set_defaults(func=command_list_runs)

    # CONSULTAR EJECUCIÓN
    show_run = subparsers.add_parser(
        "ver-ejecucion",
        help="Consultar el estado de una ejecución.",
    )

    _add_root_argument(show_run)

    show_run.add_argument(
        "--run-id",
        required=True,
    )

    show_run.set_defaults(func=command_show_run)

    # EVALUAR
    evaluate = subparsers.add_parser(
        "evaluar",
        aliases=["evaluate"],
        help="Evaluar una ejecución completada.",
    )

    _add_root_argument(evaluate)

    evaluate.add_argument(
        "--run-id",
        required=True,
    )

    evaluate.add_argument(
        "--sobrescribir",
        action="store_true",
        help="Permite reemplazar evaluation.json.",
    )

    evaluate.set_defaults(func=command_evaluate)

    # BENCHMARK
    benchmark = subparsers.add_parser(
        "benchmark",
        help="Medir el rendimiento de inferencia de una ejecución.",
    )

    _add_root_argument(benchmark)

    benchmark.add_argument(
        "--run-id",
        required=True,
    )

    benchmark.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
    )

    benchmark.set_defaults(func=command_benchmark)

    # INFERIR
    infer = subparsers.add_parser(
        "inferir",
        aliases=["infer"],
        help="Predecir el gesto de una imagen individual.",
    )

    _add_root_argument(infer)

    infer.add_argument(
        "--run-id",
        required=True,
    )

    infer.add_argument(
        "--imagen",
        required=True,
    )

    infer.set_defaults(func=command_infer)

    # MODELOS
    models = subparsers.add_parser(
        "modelos",
        help="Listar arquitecturas disponibles.",
    )

    models.set_defaults(func=command_models)

    # CLASES
    classes = subparsers.add_parser(
        "clases",
        help="Listar el vocabulario de clases.",
    )

    _add_root_argument(classes)

    classes.add_argument(
        "--archivo-clases",
        default=None,
    )

    classes.set_defaults(func=command_classes)

    return parser


# ---------------------------------------------------------------------
# Entrada principal
# ---------------------------------------------------------------------

def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)

    except Exception as exc:
        print(
            f"Error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
