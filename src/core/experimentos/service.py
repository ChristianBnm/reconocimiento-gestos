
from __future__ import annotations

import random
import uuid
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.core.configuracion.settings import (
    ExperimentConfig,
    build_experiment_config,
    cargar_clases,
    get_model_spec,
)
from src.core.dataset.experiment import (
    SOURCE_NAMES,
    prepare_sources,
    subset_for,
)
from src.core.dataset.split import dataset_fingerprint
from src.core.dataset.transforms import (
    build_eval_transform,
    build_train_transform,
)
from src.core.entrenamiento.trainer import Trainer
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

        unknown_sources = sorted(
            {
                dataset.source_name
                for dataset in spec.datasets
                if dataset.source_name not in SOURCE_NAMES
            }
        )
        if unknown_sources:
            raise ValueError(
                "Fuentes no definidas en el catálogo del proyecto: "
                + ", ".join(unknown_sources)
            )

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

        self._build_config(spec)

        for dataset in spec.datasets:
            if dataset.role == "unused":
                continue
            path = Path(dataset.path).expanduser()
            if not path.is_absolute():
                path = self.paths.root / path
            if not path.is_dir():
                raise FileNotFoundError(
                    f"No existe el directorio de la fuente "
                    f"'{dataset.source_name}': {path}"
                )

    @staticmethod
    def _build_config(spec: ExperimentSpec) -> ExperimentConfig:
        """Convierte parámetros de la definición a la configuración existente."""
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
            class_names=cargar_clases(),
            **spec.parameters,
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
    ) -> RunRecord:
        """Ejecuta el entrenamiento y persiste el estado de la ejecución."""
        spec = self.get_experiment(experiment_id)
        self.validate_experiment(spec)

        run_id = run_id or uuid.uuid4().hex
        try:
            try:
                run = self.registry.load_run(run_id)
            except FileNotFoundError:
                run = self.create_run(spec.experiment_id, run_id=run_id)

            if run.experiment_id != spec.experiment_id:
                raise ValueError("La ejecución no pertenece al experimento indicado.")
            if run.status != "pending":
                raise ValueError(
                    f"La ejecución {run_id!r} no está pendiente: {run.status!r}."
                )
            self.registry.update_run_status(run_id, "running")

            config = self._build_config(spec)
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
            )

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model_spec = get_model_spec(spec.model_name)
            model = crear_modelo(
                model_spec.architecture,
                config.num_classes,
                pretrained=model_spec.pretrained,
                dispositivo=device,
            )
            config = adaptar_normalizacion_a_modelo(config, model)

            train_transform = build_train_transform(config=config)
            eval_transform = build_eval_transform(config=config)

            train_parts = [
                subset_for(prepared[name], "train", train_transform)
                for name in source_names
            ]
            val_parts = [
                subset_for(prepared[name], "val", eval_transform)
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
                    for name in active_source_names
                },
                "fingerprints": {
                    name: dataset_fingerprint(prepared[name].dataset)
                    for name in active_source_names
                },
                "device": str(device),
                "torch_version": str(torch.__version__),
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
            )
            trainer.fit()

            if not checkpoint_path.is_file():
                raise RuntimeError(
                    "El entrenamiento terminó sin generar el checkpoint esperado."
                )

            return self.registry.update_run_status(
                run_id,
                "completed",
                checkpoint_path=self.paths.relative_to_root(checkpoint_path),
            )

        except Exception as exc:
            try:
                current = self.registry.load_run(run_id)
                if (
                    current.experiment_id == spec.experiment_id
                    and current.status in {"pending", "running"}
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