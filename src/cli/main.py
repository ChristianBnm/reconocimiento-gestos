from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.core.configuracion.settings import (
    build_experiment_config,
    get_model_spec,
    list_models,
)
from src.core.dataset.loader import DatasetSubset, GestosDataset
from src.core.dataset.split import cargar_split, dividir_dataset, guardar_split
from src.core.dataset.transforms import transformacion_train, transformacion_eval
from src.core.utils.io import load_json, save_json
from src.core.utils.paths import ProjectPaths
from src.core.utils.reproducibilidad import establecer_semilla
from scripts.matriz_confusion import generar_matriz_confusion
from scripts.analizar_errores import analizar_errores


def build_dataset(paths: ProjectPaths, config):
    return GestosDataset([
        paths.dataset_base,
        paths.dataset_complementario,
    ], transformacion=None, class_names=config.class_names)


def build_loaders(dataset, split, config, dispositivo):
    train_dataset = DatasetSubset(dataset, split.train, transformacion_train)
    val_dataset = DatasetSubset(dataset, split.val, transformacion_eval)
    test_dataset = DatasetSubset(dataset, split.test, transformacion_eval)

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=(config.pin_memory and dispositivo.type == "cuda"),
        generator=generator,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, val_loader, test_loader, test_dataset


def save_experiment_config(paths: ProjectPaths, config):
    _, results_dir = paths.ensure_model_dirs(config.model.name)
    payload = config.to_dict()
    payload["paths"] = {
        "dataset_base": "data/Clases_200",
        "dataset_complementario": "data/Clases_200_comp",
    }
    save_json(payload, results_dir / "config.json")


def command_train(args):
    import torch
    from src.core.entrenamiento.trainer import Trainer
    from src.core.modelos.factory import crear_modelo, congelar_backbone


    paths = ProjectPaths(Path(args.root).resolve())
    config = build_experiment_config(args.modelo)
    establecer_semilla(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir, results_dir = paths.ensure_model_dirs(args.modelo)
    dataset = build_dataset(paths, config)
    split = dividir_dataset(
        dataset.etiquetas,
        config.train_size,
        config.val_size,
        config.test_size,
        config.seed,
    )
    guardar_split(
        results_dir / "split.json",
        split,
        dataset,
        config.seed,
        config.train_size,
        config.val_size,
        config.test_size,
    )
    save_experiment_config(paths, config)

    train_loader, val_loader, _, _ = build_loaders(dataset, split, config, device)
    model = crear_modelo(
        config.model.architecture,
        config.num_classes,
        pretrained=config.model.pretrained,
    )

    congelar_backbone(model)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_path=checkpoint_dir / "mejor.pth",
        history_path=results_dir / "history.json",
    )
    trainer.fit()


def command_evaluate(args):
    import torch
    from src.core.evaluacion.metrics import calcular_metricas, generar_informe_clasificacion
    from src.core.evaluacion.predictor import evaluar_modelo, guardar_predicciones
    from src.core.modelos.factory import cargar_checkpoint, crear_modelo


    paths = ProjectPaths(Path(args.root).resolve())
    config = build_experiment_config(args.modelo)
    _, results_dir = paths.ensure_model_dirs(args.modelo)
    checkpoint_dir = paths.model_checkpoint_dir(args.modelo)

    dataset = build_dataset(paths, config)
    split_path = results_dir / "split.json"

    if not split_path.exists():
        if not args.crear_split:
            raise FileNotFoundError(
                f"No existe {split_path}. Entrená primero o usá --crear-split para un checkpoint legacy."
            )
        split = dividir_dataset(
            dataset.etiquetas,
            config.train_size,
            config.val_size,
            config.test_size,
            config.seed,
        )
        guardar_split(
            split_path,
            split,
            dataset,
            config.seed,
            config.train_size,
            config.val_size,
            config.test_size,
        )
    else:
        split = cargar_split(split_path, dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, _, test_dataset = build_loaders(dataset, split, config, device)

    model = crear_modelo(config.model.architecture, config.num_classes, pretrained=False)
    cargar_checkpoint(model, checkpoint_dir / "mejor.pth", device)

    result = evaluar_modelo(model, test_dataset, config.batch_size, device)
    metrics = calcular_metricas(result.y_true, result.y_pred)
    classification_report = generar_informe_clasificacion(
        result.y_true, result.y_pred, config.class_names
    )

    guardar_predicciones(
        result,
        dataset,
        paths.root,
        results_dir / "predicciones.csv",
    )
    save_json(metrics, results_dir / "metrics.json")
    save_json(classification_report, results_dir / "classification_report.json")

    generar_matriz_confusion(args.modelo, paths.root)
    analizar_errores(args.modelo, paths.root)

    print(json.dumps(metrics, indent=2))
    print(f"Predicciones: {results_dir / 'predicciones.csv'}")


def command_infer(args):
    import torch
    from src.core.evaluacion.predictor import predecir_imagen
    from src.core.modelos.factory import cargar_checkpoint, crear_modelo

    paths = ProjectPaths(Path(args.root).resolve())
    config = build_experiment_config(args.modelo)
    checkpoint = paths.model_checkpoint_dir(args.modelo) / "mejor.pth"
    model = crear_modelo(config.model.architecture, config.num_classes, pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cargar_checkpoint(model, checkpoint, device)

    pred, confidence = predecir_imagen(
        model,
        Path(args.imagen),
        transformacion_eval,
        device,
        config.class_names,
    )
    print(f"Predicción: {pred}")
    print(f"Confianza: {confidence:.4f}")


def build_parser():
    parser = argparse.ArgumentParser(description="Pipeline de reconocimiento de gestos.")
    parser.add_argument("--root", default=".", help="Raíz del proyecto.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("entrenar", aliases=["train"])
    train.add_argument("--modelo", required=True, choices=list_models())
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluar", aliases=["evaluate"])
    evaluate.add_argument("--modelo", required=True, choices=list_models())
    evaluate.add_argument(
        "--crear-split",
        action="store_true",
        help="Genera un split nuevo si no existe (solo para checkpoints legacy).",
    )
    evaluate.set_defaults(func=command_evaluate)

    infer = subparsers.add_parser("inferir", aliases=["infer"])
    infer.add_argument("--modelo", required=True, choices=list_models())
    infer.add_argument("--imagen", required=True)
    infer.set_defaults(func=command_infer)

    info = subparsers.add_parser("modelos", help="Lista las arquitecturas disponibles.")
    info.set_defaults(func=lambda args: [
        print(f"{name}: {get_model_spec(name).architecture} ({get_model_spec(name).family})")
        for name in list_models()
    ])

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
