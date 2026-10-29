from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from src.core.configuracion.catalogo import cargar_catalogos, construir_mapa_fuentes
from src.core.configuracion.settings import ExperimentConfig
from src.core.dataset.loader import DatasetSubset, GestosDataset
from src.core.dataset.split import (
    DatasetSplit, cargar_split, dataset_fingerprint, dividir_dataset, guardar_split
)
from src.core.utils.paths import ProjectPaths


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


def source_paths(paths: ProjectPaths, overrides: Mapping[str, str | Path] | None = None):
    return construir_mapa_fuentes(paths.root, SOURCE_DEFINITIONS, overrides)


def prepare_sources(
    paths: ProjectPaths,
    config: ExperimentConfig,
    requested_sources: Sequence[str] = SOURCE_NAMES,
    path_overrides: Mapping[str, str | Path] | None = None,
    create_missing_splits: bool = False,
    regenerate_splits: bool = False,
) -> dict[str, PreparedSource]:
    unknown = sorted(set(requested_sources) - set(SOURCE_NAMES))
    if unknown:
        raise ValueError(f"Fuentes desconocidas: {', '.join(unknown)}")
    locations = source_paths(paths, path_overrides)
    prepared: dict[str, PreparedSource] = {}

    for name in requested_sources:
        dataset = GestosDataset(
            locations[name],
            transformacion=None,
            class_names=config.class_names,
            source_names=[name],
            require_all_classes=True,
            deduplicate_exact_duplicates=SOURCE_DEFINITIONS[name].get(
                "deduplicate_exact_duplicates", True
            ),
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

    _validate_group_partition_consistency(prepared)
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


def _validate_group_partition_consistency(sources: Mapping[str, PreparedSource]) -> None:
    """Detecta copias exactas que quedarían a ambos lados de una partición."""
    assignments: dict[str, tuple[str, str, Path, int]] = {}
    for source_name, source in sources.items():
        split_by_index: dict[int, str] = {}
        for partition_name in ("train", "val", "test"):
            for index in getattr(source.split, partition_name):
                split_by_index[index] = partition_name
        for index, record in enumerate(source.dataset.records):
            partition = split_by_index[index]
            previous = assignments.get(record.group_id)
            if previous and previous[3] != record.label:
                raise ValueError(
                    "Se detectó contenido idéntico etiquetado con clases distintas entre fuentes: "
                    f"{previous[2]} ({previous[3]}) y {record.path} ({record.label}). "
                    "Resolvé la inconsistencia de etiquetas antes de entrenar."
                )
            if previous and previous[0] != partition:
                raise ValueError(
                    "Se detectó una imagen idéntica asignada a particiones distintas entre fuentes: "
                    f"{previous[2]} ({previous[0]}) y {record.path} ({partition}). "
                    "Para evitar fuga de datos, revisá los duplicados y regenerá las particiones "
                    "después de resolverlos."
                )
            if previous is None:
                assignments[record.group_id] = (partition, source_name, record.path, record.label)


def subset_for(source: PreparedSource, partition: str, transform=None) -> DatasetSubset:
    if partition not in {"train", "val", "test"}:
        raise ValueError(f"Partición desconocida: {partition}")
    return DatasetSubset(source.dataset, getattr(source.split, partition), transform)
