from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from sklearn.model_selection import train_test_split

from src.core.utils.io import save_json


@dataclass(frozen=True)
class DatasetSplit:
    train: list[int]
    val: list[int]
    test: list[int]

    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def validate(self, dataset_size: int) -> None:
        parts = {"train": self.train, "val": self.val, "test": self.test}
        flattened: list[int] = []
        for name, indices in parts.items():
            if not indices:
                raise ValueError(f"La partición '{name}' está vacía.")
            if any(index < 0 or index >= dataset_size for index in indices):
                raise ValueError(f"La partición '{name}' contiene índices fuera del dataset.")
            flattened.extend(indices)
        if len(flattened) != len(set(flattened)):
            raise ValueError("Hay índices repetidos o solapados entre particiones.")
        if set(flattened) != set(range(dataset_size)):
            raise ValueError("Las particiones no cubren exactamente el dataset completo.")


def _validate_ratios(train_size: float, val_size: float, test_size: float) -> None:
    if any(ratio <= 0 or ratio >= 1 for ratio in (train_size, val_size, test_size)):
        raise ValueError("Las proporciones de train, val y test deben estar entre 0 y 1.")
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size debe ser igual a 1.0")


def dividir_dataset(
    etiquetas: Sequence[int],
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
    grupos: Sequence[str] | None = None,
) -> DatasetSplit:
    """Divide un dataset sin separar copias con el mismo identificador de grupo.

    La estratificación se realiza por grupo (no por imagen) si se pasan grupos.
    Cada grupo debe pertenecer a una sola clase; grupos conflictivos se rechazan.
    """
    _validate_ratios(train_size, val_size, test_size)
    if len(etiquetas) == 0:
        raise ValueError("No se puede particionar un dataset vacío.")
    if grupos is not None and len(grupos) != len(etiquetas):
        raise ValueError("La cantidad de grupos debe coincidir con la cantidad de etiquetas.")

    indices = list(range(len(etiquetas)))
    if grupos is None:
        train_indices, temp_indices = _stratified_split(
            indices, list(etiquetas), val_size + test_size, seed
        )
        val_fraction = val_size / (val_size + test_size)
        val_indices, test_indices = _stratified_split(
            temp_indices,
            [etiquetas[index] for index in temp_indices],
            1.0 - val_fraction,
            seed,
        )
    else:
        group_to_indices: dict[str, list[int]] = {}
        group_to_label: dict[str, int] = {}
        for index, (label, group) in enumerate(zip(etiquetas, grupos)):
            group = str(group)
            group_to_indices.setdefault(group, []).append(index)
            previous = group_to_label.setdefault(group, int(label))
            if previous != int(label):
                raise ValueError(
                    "Se detectó contenido idéntico asignado a clases distintas; "
                    f"grupo {group!r} tiene etiquetas {previous} y {label}."
                )
        group_ids = sorted(group_to_indices)
        group_labels = [group_to_label[group] for group in group_ids]
        train_groups, temp_groups = _stratified_split(
            group_ids, group_labels, val_size + test_size, seed
        )
        temp_label_by_group = {group: group_to_label[group] for group in temp_groups}
        val_fraction = val_size / (val_size + test_size)
        val_groups, test_groups = _stratified_split(
            temp_groups,
            [temp_label_by_group[group] for group in temp_groups],
            1.0 - val_fraction,
            seed,
        )

        def expand(groups_for_split):
            return sorted(index for group in groups_for_split for index in group_to_indices[group])

        train_indices = expand(train_groups)
        val_indices = expand(val_groups)
        test_indices = expand(test_groups)

    result = DatasetSplit(
        train=sorted(train_indices),
        val=sorted(val_indices),
        test=sorted(test_indices),
    )
    result.validate(len(etiquetas))
    if grupos is not None:
        group_partition: dict[str, str] = {}
        for partition_name, part_indices in (
            ("train", result.train), ("val", result.val), ("test", result.test)
        ):
            for index in part_indices:
                group = str(grupos[index])
                previous = group_partition.setdefault(group, partition_name)
                if previous != partition_name:
                    raise RuntimeError(f"El grupo {group!r} quedó en más de una partición.")
    return result


def _stratified_split(indices, labels, test_size: float, seed: int):
    try:
        return train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=labels
        )
    except ValueError as exc:
        raise ValueError(
            "No se puede crear una partición estratificada con estos datos. "
            "Verificá que cada clase tenga suficientes grupos únicos para train, "
            "validación y prueba. No se aplicará una división no estratificada automáticamente."
        ) from exc


def dataset_fingerprint(dataset) -> str:
    """Fingerprint portable de fuente/ruta relativa/etiqueta/contenido."""
    digest = hashlib.sha256()
    for record in dataset.records:
        line = f"KEEP\t{record.source}\t{record.relative_path}\t{record.label}\t{record.group_id}\n"
        digest.update(line.encode("utf-8"))
    for duplicate in getattr(dataset, "excluded_duplicates", []):
        line = (
            f"EXCLUDE_EXACT_DUPLICATE\t{duplicate.source}\t{duplicate.relative_path}\t"
            f"{duplicate.kept_relative_path}\t{duplicate.label}\t{duplicate.group_id}\n"
        )
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def guardar_split(
    ruta: Path,
    split: DatasetSplit,
    dataset,
    seed: int,
    train_size: float,
    val_size: float,
    test_size: float,
) -> None:
    split.validate(len(dataset))
    payload = {
        "version": 2,
        "seed": seed,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "dataset_size": len(dataset),
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "class_names": list(dataset.clases),
        "source_names": list(dataset.source_names),
        "grouping": "sha256_image_content",
        "deduplicate_exact_duplicates": getattr(dataset, "deduplicate_exact_duplicates", False),
        "exact_duplicate_files_excluded": len(getattr(dataset, "excluded_duplicates", [])),
        "indices": asdict(split),
    }
    save_json(payload, ruta)


def cargar_split(
    ruta: Path,
    dataset,
    expected_seed: int | None = None,
    train_size: float | None = None,
    val_size: float | None = None,
    test_size: float | None = None,
) -> DatasetSplit:
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el split guardado: {ruta}")
    with ruta.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("dataset_size") != len(dataset):
        raise ValueError("El tamaño del dataset no coincide con el usado para crear el split.")
    if payload.get("class_names") != list(dataset.clases):
        raise ValueError("Las clases actuales no coinciden con las del split guardado.")
    if expected_seed is not None and payload.get("seed") != expected_seed:
        raise ValueError(
            f"El split guardado usa semilla {payload.get('seed')}, pero la configuración solicita "
            f"{expected_seed}. Usá --regenerar-splits para cambiar explícitamente las particiones."
        )
    for key, requested in (("train_size", train_size), ("val_size", val_size), ("test_size", test_size)):
        if requested is not None and abs(float(payload.get(key, -1)) - requested) > 1e-6:
            raise ValueError(
                f"El split guardado tiene {key}={payload.get(key)}, pero se solicita {requested}. "
                "Usá --regenerar-splits para cambiar explícitamente las particiones."
            )
    if payload.get("source_names", list(dataset.source_names)) != list(dataset.source_names):
        raise ValueError("Las fuentes actuales no coinciden con las del split guardado.")
    if payload.get("dataset_fingerprint") != dataset_fingerprint(dataset):
        raise ValueError(
            "El contenido o la estructura del dataset cambió. Regenerá explícitamente las particiones."
        )
    indices = payload.get("indices", {})
    split = DatasetSplit(
        train=list(indices.get("train", [])),
        val=list(indices.get("val", [])),
        test=list(indices.get("test", [])),
    )
    split.validate(len(dataset))
    group_partition: dict[str, str] = {}
    for partition_name in ("train", "val", "test"):
        for index in getattr(split, partition_name):
            group = dataset.records[index].group_id
            previous = group_partition.setdefault(group, partition_name)
            if previous != partition_name:
                raise ValueError(
                    "El split guardado separa copias exactas de una misma imagen entre "
                    f"'{previous}' y '{partition_name}'. Regenerá las particiones."
                )
    return split
