from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from src.core.experimentos.schemas import DatasetAssignment

from src.core.configuracion.catalogo import cargar_catalogos, construir_mapa_fuentes
from src.core.configuracion.discovery import get_source_definitions
from src.core.configuracion.settings import ExperimentConfig
from src.core.dataset.loader import DatasetSubset, GestosDataset
from src.core.utils.paths import ProjectPaths

from src.core.dataset.split import (
    DatasetSplit, cargar_split, dataset_fingerprint, dividir_dataset, guardar_split
)

from src.core.dataset.global_inventory import (
    build_global_inventory,
    validate_global_preflight,
)


SOURCE_DEFINITIONS, SCENARIO_DEFINITIONS = cargar_catalogos()
SCENARIO_SOURCES: dict[str, tuple[str, ...]] = {
    name: tuple(spec["sources"]) for name, spec in SCENARIO_DEFINITIONS.items()
}
SOURCE_NAMES = tuple(SOURCE_DEFINITIONS)


@dataclass
class PreparedSource:
    name: str
    dataset: GestosDataset
    split: DatasetSplit
    split_path: Path


def source_paths(
    paths: ProjectPaths,
    overrides: Mapping[str, str | Path] | None = None,
    source_definitions: Mapping[str, dict] | None = None,
):
    definitions = source_definitions or get_source_definitions(paths.root)
    return construir_mapa_fuentes(paths.root, definitions, overrides)


def prepare_sources(
    paths: ProjectPaths,
    config: ExperimentConfig,
    requested_sources: Sequence[str] = SOURCE_NAMES,
    path_overrides: Mapping[str, str | Path] | None = None,
    create_missing_splits: bool = False,
    regenerate_splits: bool = False,
    source_assignments: Sequence | None = None,
) -> dict[str, PreparedSource]:
    source_definitions = get_source_definitions(paths.root)
    unknown = sorted(set(requested_sources) - set(source_definitions))
    if unknown:
        raise ValueError(f"Fuentes desconocidas o no válidas: {', '.join(unknown)}")

    locations = source_paths(paths, path_overrides, source_definitions)

    if source_assignments is not None:
        inventory_assignments = [
            DatasetAssignment(
                source_name=assignment.source_name,
                path=str(locations[assignment.source_name]),
                role=assignment.role,
            )
            for assignment in source_assignments
            if assignment.role != "unused"
        ]

        inventory = build_global_inventory(
            assignments=inventory_assignments,
            class_names=config.class_names,
        )
    else:
        inventory = None

    if source_assignments is not None:
        missing_sources = {
            assignment.source_name
            for assignment in source_assignments
            if assignment.role != "unused"
        } - set(requested_sources)

        if missing_sources:
            raise ValueError(
                "Hay fuentes activas que no se prepararán: "
                + ", ".join(sorted(missing_sources))
            )

    excluded_paths_by_source: dict[str, set[Path]] = {}

    training_sources = (
        [
            assignment.source_name
            for assignment in source_assignments
            if assignment.role == "train_source"
        ]
        if source_assignments is not None
        else None
    )

    if inventory is not None:
        validate_global_preflight(
            inventory=inventory,
        class_names=config.class_names,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        seed=config.split_seed,
        training_sources=training_sources,
    )

        for excluded in inventory.excluded_training_copies:
            excluded_paths_by_source.setdefault(
            excluded.source_name, set()
        ).add(excluded.path.resolve())

        _guardar_manifest_exclusiones_globales(
            inventory,
            paths.root / "manifests" / "exclusiones_globales.csv",
        )

    prepared: dict[str, PreparedSource] = {}

    for name in requested_sources:
        dataset = GestosDataset(
            locations[name],
            transformacion=None,
            class_names=config.class_names,
            source_names=[name],
            require_all_classes=True,
            deduplicate_exact_duplicates=source_definitions[name].get(
                "deduplicate_exact_duplicates", True
            ),
            excluded_paths=excluded_paths_by_source.get(name, set()),
        )
        fingerprint = dataset_fingerprint(dataset)
        split_path = paths.source_split_path(
            name,
            dataset_fingerprint=fingerprint,
            seed=config.split_seed,
            train_size=config.train_size,
            val_size=config.val_size,
            test_size=config.test_size,
        )
        _guardar_manifest_duplicados(dataset, split_path.with_suffix(".duplicados_exactos.csv"))
        if regenerate_splits or not split_path.exists():
            if not create_missing_splits and not regenerate_splits:
                raise FileNotFoundError(
                    f"No existe la partición de '{name}': {split_path}. "
                    "Entrená primero o autorizá la creación de particiones."
                )
            split = dividir_dataset(
                dataset.etiquetas,
                config.train_size,
                config.val_size,
                config.test_size,
                config.split_seed,
                grupos=[record.group_id for record in dataset.records],
            )
            guardar_split(
                split_path,
                split,
                dataset,
                config.split_seed,
                config.train_size,
                config.val_size,
                config.test_size,
            )
        else:
            split = cargar_split(
                split_path,
                dataset,
                expected_seed=config.split_seed,
                train_size=config.train_size,
                val_size=config.val_size,
                test_size=config.test_size,
            )
        prepared[name] = PreparedSource(name, dataset, split, split_path)

        _validate_group_partition_consistency(
            prepared,
            training_sources=(
            set(training_sources)
            if training_sources is not None
            else None
            ),
        )

    return prepared


def _guardar_manifest_duplicados(dataset: GestosDataset, ruta: Path) -> None:
    """Registra las copias exactas omitidas en memoria; nunca borra archivos fuente."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "fuente", "clase", "ruta_conservada", "ruta_duplicada_excluida", "sha256", "motivo"
        ])
        for duplicate in dataset.excluded_duplicates:
            writer.writerow([
                duplicate.source,
                duplicate.class_name,
                duplicate.kept_relative_path,
                duplicate.relative_path,
                duplicate.group_id,
                "duplicado_exacto_sha256",
            ])


def _guardar_manifest_exclusiones_globales(inventory, ruta: Path) -> None:
    """Audita las copias de entrenamiento excluidas por deduplicación global."""
    ruta.parent.mkdir(parents=True, exist_ok=True)

    with ruta.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "fuente_excluida",
            "clase",
            "ruta_duplicada_excluida",
            "fuente_conservada",
            "ruta_conservada",
            "sha256",
            "motivo",
        ])

        for excluded in inventory.excluded_training_copies:
            writer.writerow([
                excluded.source_name,
                excluded.class_name,
                excluded.relative_path,
                excluded.kept_source_name,
                excluded.kept_relative_path,
                excluded.sha256,
                excluded.reason,
            ])



def _validate_group_partition_consistency(
    sources: Mapping[str, PreparedSource],
    training_sources: set[str] | None = None,
) -> None:
    """Detecta duplicados entre particiones de las fuentes de entrenamiento."""

    assignments: dict[str, tuple[str, str, Path, int]] = {}

    for source_name, source in sources.items():
        if training_sources is not None and source_name not in training_sources:
            continue

        split_by_index: dict[int, str] = {}

        for partition_name in ("train", "val", "test"):
            for index in getattr(source.split, partition_name):
                split_by_index[index] = partition_name

        for index, record in enumerate(source.dataset.records):
            partition = split_by_index[index]
            previous = assignments.get(record.group_id)

            if previous and previous[3] != record.label:
                raise ValueError(
                    "Se detectó contenido idéntico etiquetado con clases "
                    "distintas entre fuentes de entrenamiento: "
                    f"{previous[2]} ({previous[3]}) y "
                    f"{record.path} ({record.label}). "
                    "Resolvé la inconsistencia de etiquetas antes de entrenar."
                )

            if previous and previous[0] != partition:
                raise ValueError(
                    "Se detectó una imagen idéntica asignada a particiones "
                    "distintas entre fuentes de entrenamiento: "
                    f"{previous[2]} ({previous[0]}) y "
                    f"{record.path} ({partition}). "
                    "Para evitar fuga de datos, revisá los duplicados y "
                    "regenerá las particiones después de resolverlos."
                )

            if previous is None:
                assignments[record.group_id] = (
                    partition,
                    source_name,
                    record.path,
                    record.label,
                )


def subset_for(source: PreparedSource, partition: str, transform=None) -> DatasetSubset:
    if partition not in {"train", "val", "test"}:
        raise ValueError(f"Partición desconocida: {partition}")
    return DatasetSubset(source.dataset, getattr(source.split, partition), transform)
