
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from threading import Event
from typing import Any
from src.core.utils.io import load_json, save_json
from torch.utils.data import ConcatDataset, DataLoader

import time
import uuid
import torch
import random
import platform
import numpy as np




from src.core.configuracion.settings import (
    ExperimentConfig,
    build_experiment_config,
    cargar_clases,
    get_model_spec,
)
from src.core.dataset.experiment import (
    prepare_sources,
    subset_for,
)
from src.core.configuracion.discovery import (
    get_dataset_catalog,
    inspect_dataset_directory,
)
from src.core.dataset.split import dataset_fingerprint
from src.core.dataset.transforms import (
    build_eval_transform,
    build_train_transform,
)
from src.core.dataset.preprocessing import ImagePreprocessor
from src.core.entrenamiento.trainer import Trainer, TrainingCancelled
from src.core.experimentos.registry import ExperimentRegistry
from src.core.experimentos.schemas import (
    DATASET_ROLES,
    ExperimentSpec,
    RunRecord,
)
from src.core.modelos.factory import (
    adaptar_normalizacion_a_modelo,
    congelar_backbone,
    crear_modelo,
    descongelar_modelo,
)
from src.core.utils.io import save_json
from src.core.utils.paths import ProjectPaths
from src.core.utils.reproducibilidad import establecer_semilla


class ExperimentService:
    """Coordina experimentos sin duplicar el pipeline de entrenamiento."""

    def __init__(
        self,
        root: str | Path,
        registry: ExperimentRegistry | None = None,
    ) -> None:
        self.paths = ProjectPaths(Path(root).expanduser().resolve())
        self.registry = registry or ExperimentRegistry(
            self.paths.results_dir / "registry"
        )

    @staticmethod
    def _model_id_prefix(model_name: str) -> str:
        prefixes = {
            "mobilenetv2": "MNET",
            "densenet201": "DENS",
            "vit_tiny": "VITT",
            "vit_base": "VITB",
        }
        return prefixes.get(model_name.casefold(), model_name[:4].upper())

    def next_experiment_id(self, model_name: str) -> str:
        """Genera una ID legible y correlativa por familia de modelo."""
        prefix = self._model_id_prefix(model_name)
        highest = 0
        for experiment_id in self.registry.list_experiments():
            if experiment_id.startswith(prefix + "-"):
                suffix = experiment_id[len(prefix) + 1:]
                if suffix.isdigit():
                    highest = max(highest, int(suffix))
        return f"{prefix}-{highest + 1:03d}"

    def create_experiment(self, spec: ExperimentSpec) -> ExperimentSpec:
        """Valida y persiste una definición nueva e inmutable por ID."""
        self.validate_experiment(spec)
        if spec.experiment_id in self.registry.list_experiments():
            raise FileExistsError(
                f"Ya existe el experimento {spec.experiment_id!r}; "
                "creá otro experimento para guardar una configuración distinta."
            )
        self.registry.save_experiment(spec)
        return spec

    def get_experiment(self, experiment_id: str) -> ExperimentSpec:
        return self.registry.load_experiment(experiment_id)

    def delete_experiment(self, experiment_id: str) -> None:
        """Elimina la definición solo si no tiene ejecuciones asociadas."""
        self.registry.delete_experiment(experiment_id)

    def list_experiments(self) -> list[str]:
        return self.registry.list_experiments()

    def create_run(self, experiment_id: str, run_id: str | None = None) -> RunRecord:
        """Registra una ejecución pendiente sin iniciar todavía el entrenamiento."""
        spec = self.get_experiment(experiment_id)
        if run_id is None:
            prefix = f"{spec.experiment_id}-R"
            highest = 0
            for existing in self.registry.list_runs(spec.experiment_id):
                if existing.run_id.startswith(prefix):
                    suffix = existing.run_id[len(prefix):]
                    if suffix.isdigit():
                        highest = max(highest, int(suffix))
            run_id = f"{prefix}{highest + 1:02d}"
        run = RunRecord(
            run_id=run_id,
            experiment_id=spec.experiment_id,
            status="pending",
        )
        self.registry.save_run(run)
        return run

    def get_run(self, run_id: str) -> RunRecord:
        return self.registry.load_run(run_id)

    def list_runs(self, experiment_id: str | None = None) -> list[RunRecord]:
        return self.registry.list_runs(experiment_id)

    def validate_experiment(self, spec: ExperimentSpec) -> None:
        if not isinstance(spec, ExperimentSpec):
            raise TypeError("spec debe ser una instancia de ExperimentSpec.")

        get_model_spec(spec.model_name)

        catalog = get_dataset_catalog(self.paths.root)
        available_sources = {
            source["name"]: source for source in catalog["sources"]
        }
        invalid_catalog_sources = {
            source["name"]: source for source in catalog["invalid_sources"]
        }

        unknown_sources = sorted(
            {
                dataset.source_name
                for dataset in spec.datasets
                if dataset.source_name not in available_sources
                and dataset.source_name not in invalid_catalog_sources
            }
        )
        if unknown_sources:
            raise ValueError(
                "Fuentes no detectadas en data/ o no configuradas: "
                + ", ".join(unknown_sources)
            )

        rejected_sources = sorted(
            name for name in {
                dataset.source_name for dataset in spec.datasets
            } if name in invalid_catalog_sources
        )
        if rejected_sources:
            details = "; ".join(
                f"{name}: {invalid_catalog_sources[name]['reason']}"
                for name in rejected_sources
            )
            raise ValueError("Hay fuentes con estructura no válida: " + details)

        invalid_roles = sorted(
            {
                dataset.role
                for dataset in spec.datasets
                if dataset.role not in DATASET_ROLES
            }
        )
        if invalid_roles:
            raise ValueError(
                "Roles de dataset no válidos: " + ", ".join(invalid_roles)
            )

        config = self._build_config(spec)

        for dataset in spec.datasets:
            if dataset.role == "unused":
                continue
            path = Path(dataset.path).expanduser()
            if not path.is_absolute():
                path = self.paths.root / path
            inspect_dataset_directory(path, expected_class_names=config.class_names)

    def _build_config(self, spec: ExperimentSpec) -> ExperimentConfig:
        allowed = {
            item.name
            for item in fields(ExperimentConfig)
        }

        forbidden = {"model", "class_names"} & set(spec.parameters)
        if forbidden:
            raise ValueError(
                "Estos parámetros no se pueden sobrescribir directamente: "
                + ", ".join(sorted(forbidden))
            )

        unknown = set(spec.parameters) - allowed
        if unknown:
            raise ValueError(
                "Parámetros de configuración desconocidos: "
                + ", ".join(sorted(unknown))
            )

        return build_experiment_config(
            spec.model_name,
            class_names=self._resolve_class_names(spec),
            **spec.parameters,
        )

    def _resolve_class_names(
        self,
        spec: ExperimentSpec,
    ) -> tuple[str, ...]:
        if spec.class_file is None:
            return cargar_clases()

        path = Path(spec.class_file).expanduser()

        if not path.is_absolute():
            path = self.paths.root / path

        return cargar_clases(path)


    @staticmethod
    def _resolve_device(device_config: str) -> torch.device:
        """Resuelve el dispositivo respetando la configuración del experimento."""
        if device_config == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        if device_config == "cpu":
            return torch.device("cpu")

        if device_config == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Se solicitó el dispositivo 'cuda', "
                    "pero CUDA no está disponible."
                )
            return torch.device("cuda")

        raise ValueError(
            f"Dispositivo no válido: {device_config!r}. "
            "Opciones permitidas: 'auto', 'cpu' o 'cuda'."
        )


    @staticmethod
    def _make_loader(
        dataset,
        config: ExperimentConfig,
        device: torch.device,
        *,
        shuffle: bool,
        seed: int,
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(seed)

        def seed_worker(worker_id: int) -> None:
            worker_seed = torch.initial_seed() % (2**32)
            random.seed(worker_seed)
            np.random.seed(worker_seed)

        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory and device.type == "cuda",
            generator=generator,
            worker_init_fn=seed_worker if config.num_workers > 0 else None,
            persistent_workers=config.num_workers > 0,
        )

    def run_experiment(
        self,
        experiment_id: str,
        run_id: str | None = None,
        cancel_event: Event | None = None,
    ) -> RunRecord:
        """Ejecuta el entrenamiento y persiste el estado de la ejecución."""
        run_id = run_id or uuid.uuid4().hex
        spec = None
        try:
            # Cargar primero el registro de ejecución: cualquier error posterior,
            # incluida la validación de datasets personalizados, debe quedar
            # asociado a un estado terminal y no dejar la ejecución en pending.
            try:
                run = self.registry.load_run(run_id)
            except FileNotFoundError:
                run = self.create_run(experiment_id, run_id=run_id)

            if run.experiment_id != experiment_id:
                raise ValueError("La ejecución no pertenece al experimento indicado.")
            if run.status != "pending":
                raise ValueError(
                    f"La ejecución {run_id!r} no está pendiente: {run.status!r}."
                )
            self.registry.update_run_status(run_id, "running")

            spec = self.get_experiment(experiment_id)
            self.validate_experiment(spec)
            config = self._build_config(spec)
            device = self._resolve_device(config.device)

            train_assignments = [
                dataset
                for dataset in spec.datasets
                if dataset.role == "train_source"
            ]
            if not train_assignments:
                raise ValueError(
                    "Se necesita al menos una fuente con rol 'train_source'."
                )

            # Cada ejecución tiene directorios propios: nunca sobrescribe
            # checkpoints ni resultados de otra ejecución.
            artifact_root = (
                self.paths.results_dir
                / "experiments"
                / spec.experiment_id
                / run_id
            )
            checkpoint_dir = (
                self.paths.checkpoints_dir
                / "experiments"
                / spec.experiment_id
                / run_id
            )
            artifact_root.mkdir(parents=True, exist_ok=False)
            checkpoint_dir.mkdir(parents=True, exist_ok=False)

            checkpoint_path = checkpoint_dir / "mejor.pth"
            history_path = artifact_root / "history.json"

            establecer_semilla(config.seed)

            active_assignments = [
                item for item in spec.datasets if item.role != "unused"
            ]
            path_overrides = {
                item.source_name: item.path for item in active_assignments
            }
            active_source_names = [
                item.source_name for item in active_assignments
            ]
            source_names = [item.source_name for item in train_assignments]

            # Se preparan también las fuentes reservadas para evaluación. Sus
            # particiones train/val nunca se incorporan al entrenamiento.
            prepared = prepare_sources(
                self.paths,
                config,
                requested_sources=active_source_names,
                path_overrides=path_overrides,
                create_missing_splits=True,
                source_assignments=active_assignments,
            )


            model_spec = get_model_spec(spec.model_name)

            model = crear_modelo(
                arquitectura=config.model.architecture,
                num_classes=config.num_classes,
                pretrained=config.model.pretrained,
                dispositivo=device,
            )

            config = adaptar_normalizacion_a_modelo(config, model)

            preprocessor = ImagePreprocessor(
                config.preprocessing_pipeline, config.preprocessing_options
            )
            train_transform = build_train_transform(config=config)
            eval_transform = build_eval_transform(config=config)

            train_parts = [
                subset_for(prepared[name], "train", train_transform, preprocessor)
                for name in source_names
            ]
            val_parts = [
                subset_for(prepared[name], "val", eval_transform, preprocessor)
                for name in source_names
            ]

            train_dataset = (
                train_parts[0]
                if len(train_parts) == 1
                else ConcatDataset(train_parts)
            )
            val_dataset = (
                val_parts[0]
                if len(val_parts) == 1
                else ConcatDataset(val_parts)
            )

            train_loader = self._make_loader(
                train_dataset, config, device,
                shuffle=True, seed=config.seed,
            )
            val_loader = self._make_loader(
                val_dataset, config, device,
                shuffle=False, seed=config.seed,
            )

            model = (
                congelar_backbone(model)
                if config.freeze_backbone
                else descongelar_modelo(model)
            )

            metadata: dict[str, Any] = {
                "experiment_id": spec.experiment_id,
                "run_id": run_id,
                "model_name": spec.model_name,
                "preprocessing_pipeline": config.preprocessing_pipeline,
                "preprocessing_options": config.preprocessing_options,
                "training_sources": source_names,
                "dataset_roles": {item.source_name: item.role for item in active_assignments},
                "source_paths": {
                    item.source_name: self.paths.relative_to_root(
                        (
                            Path(item.path).expanduser()
                            if Path(item.path).expanduser().is_absolute()
                            else self.paths.root / Path(item.path).expanduser()
                        ).resolve()
                    )
                    for item in active_assignments
                },

                "split_paths": {
                    name: self.paths.relative_to_root(prepared[name].split_path)
                    for name in source_names
                },
                
                "fingerprints": {
                    name: dataset_fingerprint(prepared[name].dataset)
                    for name in active_source_names
                },
                "device": str(device),
                "torch_version": str(torch.__version__),
                "device": str(device),
                "torch_version": str(torch.__version__),
                "python_version": platform.python_version(),
                "gpu_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                "cuda_version": torch.version.cuda,
            }

            config_payload = config.to_dict()
            config_payload.update({
                "experiment_id": spec.experiment_id,
                "run_id": run_id,
                "experiment_name": spec.name,
                "description": spec.description,
                "dataset_assignments": [
                    item.to_dict() for item in spec.datasets
                ],
                "metadata": metadata,
            })
            save_json(config_payload, artifact_root / "config.json")

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                checkpoint_path=checkpoint_path,
                history_path=history_path,
                metadata=metadata,
                cancel_event=cancel_event,
            )


            # Medición de recursos del entrenamiento.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

            training_started = time.perf_counter()
            trainer.fit()

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            training_seconds = time.perf_counter() - training_started

            # No registrar métricas de éxito si se solicitó cancelar.
            if cancel_event is not None and cancel_event.is_set():
                raise TrainingCancelled(
                    "El usuario solicitó detener el entrenamiento."
                )

            # Memoria RAM máxima del proceso según el sistema operativo.
            try:
                import resource

                max_rss_raw = resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss

                # Linux informa KiB; macOS informa bytes.
                max_rss_mb = max_rss_raw / (
                    1024 if platform.system() == "Linux"
                    else 1024 * 1024
                )
            except (ImportError, AttributeError, OSError):
                max_rss_mb = None

            try:
                checkpoint_mb = (
                    checkpoint_path.stat().st_size / (1024 * 1024)
                )
            except OSError:
                checkpoint_mb = None

            epoch_history = (
                load_json(history_path)
                if history_path.is_file()
                else []
            )

            trainable_parameters = sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )
            total_parameters = sum(
                p.numel() for p in model.parameters()
            )

            resource_metrics = {
                "experiment_id": spec.experiment_id,
                "run_id": run_id,
                "model": spec.model_name,
                "device": str(device),
                "gpu_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                "python_version": platform.python_version(),
                "pytorch_version": str(torch.__version__),
                "cuda_available": torch.cuda.is_available(),
                "image_size": config.image_size,
                "batch_size": config.batch_size,
                "epochs_completed": len(epoch_history),
                "training_time_seconds": round(training_seconds, 4),
                "mean_epoch_time_seconds": (
                    round(
                        sum(
                            row.get("epoch_seconds", 0.0)
                            for row in epoch_history
                        ) / len(epoch_history),
                        4,
                    )
                    if epoch_history
                    else None
                ),
                "max_process_ram_mb": (
                    round(max_rss_mb, 2)
                    if max_rss_mb is not None
                    else None
                ),
                "gpu_peak_allocated_mb": (
                    round(
                        torch.cuda.max_memory_allocated(device)
                        / (1024 * 1024),
                        2,
                    )
                    if device.type == "cuda"
                    else None
                ),
                "gpu_peak_reserved_mb": (
                    round(
                        torch.cuda.max_memory_reserved(device)
                        / (1024 * 1024),
                        2,
                    )
                    if device.type == "cuda"
                    else None
                ),
                "parameters_total": total_parameters,
                "parameters_trainable": trainable_parameters,
                "checkpoint_size_mb": (
                    round(checkpoint_mb, 3)
                    if checkpoint_mb is not None
                    else None
                ),
                "note_ram": (
                    "Máximo RSS del proceso según el sistema operativo; "
                    "no equivale a memoria exclusiva del modelo."
                ),
                "note_gpu_memory": (
                    "Pico de memoria gestionada por el asignador de PyTorch; "
                    "no incluye necesariamente todas las asignaciones "
                    "externas a PyTorch."
                ),
            }

            save_json(
                resource_metrics,
                artifact_root / "metricas_recursos_entrenamiento.json",
            )

            if not checkpoint_path.is_file():
                raise RuntimeError(
                    "El entrenamiento terminó sin generar el checkpoint esperado."
                )

            return self.registry.update_run_status(
                run_id,
                "completed",
                checkpoint_path=self.paths.relative_to_root(checkpoint_path),
            )

        except TrainingCancelled as exc:
            try:
                current = self.registry.load_run(run_id)
                if current.status in {"pending", "running"}:
                    return self.registry.update_run_status(
                        run_id,
                        "cancelled",
                        error_message=str(exc),
                    )
            except Exception:
                raise
            return self.registry.load_run(run_id)
        except Exception as exc:
            try:
                current = self.registry.load_run(run_id)
                if (
                    current.status in {"pending", "running"}
                ):
                    self.registry.update_run_status(
                        run_id,
                        "failed",
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
            except Exception as status_exc:
                # No reemplazar el error original por un fallo de registro.
                import logging

                logging.getLogger(__name__).error(
                    "No se pudo registrar el fallo de la ejecución %s: %s: %s",
                    run_id,
                    type(status_exc).__name__,
                    status_exc,
                )
            raise
