
"""Inventario global de imágenes para validar identidad y roles de datasets."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from src.core.dataset.split import dividir_dataset



IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

TRAIN_ROLE = "train_source"
EVALUATION_ROLES = {"final_test", "external_eval"}
ACTIVE_ROLES = {TRAIN_ROLE, *EVALUATION_ROLES}
VALID_ROLES = ACTIVE_ROLES | {"unused"}



@dataclass(frozen=True)
class InventoryRecord:
    """Identidad de una imagen encontrada en una fuente activa."""

    sha256: str
    path: Path
    relative_path: str
    class_name: str
    source_name: str
    role: str


@dataclass(frozen=True)
class ExcludedTrainingCopy:
    """Auditoría de una copia de entrenamiento excluida por duplicación."""

    sha256: str
    path: Path
    relative_path: str
    class_name: str
    source_name: str
    kept_path: Path
    kept_relative_path: str
    kept_source_name: str
    reason: str = "duplicate_training_content"


@dataclass(frozen=True)
class GlobalInventory:
    """Inventario global y propuesta determinista de copias canónicas."""

    records: tuple[InventoryRecord, ...]
    duplicate_groups: Mapping[str, tuple[InventoryRecord, ...]]
    canonical_by_hash: Mapping[str, InventoryRecord]
    excluded_training_copies: tuple[ExcludedTrainingCopy, ...]

    @property
    def duplicate_count(self) -> int:
        """Cantidad de registros adicionales dentro de grupos duplicados."""
        return sum(len(group) - 1 for group in self.duplicate_groups.values())



def _validate_class_names(class_names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(class_names)

    if not names:
        raise ValueError("class_names no puede estar vacío.")

    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Todos los nombres de clase deben ser textos no vacíos.")

    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Los nombres de clase deben ser únicos sin distinguir mayúsculas.")

    for name in names:
        if name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(
                f"Nombre de clase inválido para un directorio: {name!r}."
            )

    return names


def _validate_assignment(assignment: object) -> tuple[str, str, Path]:
    """Valida una asignación y devuelve nombre, rol y ruta resuelta."""
    for attribute in ("source_name", "path", "role"):
        if not hasattr(assignment, attribute):
            raise ValueError(
                f"Asignación inválida: falta el atributo '{attribute}'."
            )

    raw_name = assignment.source_name
    raw_role = assignment.role
    raw_path = assignment.path

    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("El nombre de la fuente debe ser un texto no vacío.")

    if not isinstance(raw_role, str) or raw_role.strip() not in VALID_ROLES:
        raise ValueError(
            f"Rol no reconocido para la fuente {raw_name!r}: {raw_role!r}."
        )

    if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
        raise ValueError(
            f"La ruta de la fuente {raw_name!r} debe ser un texto o Path no vacío."
        )

    root = Path(raw_path).expanduser().resolve()
    return raw_name.strip(), raw_role.strip(), root


def _validate_source_roots(
    sources: list[tuple[str, str, Path]],
) -> None:
    """Rechaza directorios de fuentes duplicados o anidados."""
    for index, (name, _, root) in enumerate(sources):
        if not root.is_dir():
            raise FileNotFoundError(
                f"No existe el directorio de la fuente '{name}': {root}"
            )

        for other_name, _, other_root in sources[:index]:
            if root == other_root:
                raise ValueError(
                    f"Las fuentes '{other_name}' y '{name}' apuntan al mismo "
                    f"directorio: {root}."
                )

            if root in other_root.parents or other_root in root.parents:
                raise ValueError(
                    f"Las rutas de las fuentes '{other_name}' y '{name}' "
                    "están anidadas. Deben ser directorios independientes."
                )


def build_global_inventory(
    assignments: Iterable[object],
    class_names: Iterable[str],
) -> GlobalInventory:
    """Escanea fuentes activas y valida etiquetas y fugas entre roles.

    Cada asignación debe exponer `source_name`, `path` y `role`.
    Las fuentes `unused` se ignoran.

    No elimina archivos ni excluye registros de los datasets.
    """
    names = _validate_class_names(class_names)

    normalized_assignments: list[tuple[str, str, Path]] = []
    seen_sources: set[str] = set()

    for assignment in assignments:
        source_name, role, root = _validate_assignment(assignment)

        if role == "unused":
            continue

        normalized_name = source_name.casefold()
        if normalized_name in seen_sources:
            raise ValueError(
                f"La fuente '{source_name}' está asignada más de una vez."
            )

        seen_sources.add(normalized_name)
        normalized_assignments.append((source_name, role, root))

    _validate_source_roots(normalized_assignments)

    records: list[InventoryRecord] = []

    for source_name, role, root in normalized_assignments:
        for class_name in names:
            class_dir = root / class_name

            if not class_dir.is_dir():
                continue

            for path in sorted(
                class_dir.iterdir(),
                key=lambda item: item.name.casefold(),
            ):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                digest = hashlib.sha256(path.read_bytes()).hexdigest()

                records.append(
                    InventoryRecord(
                        sha256=digest,
                        path=path,
                        relative_path=path.relative_to(root).as_posix(),
                        class_name=class_name,
                        source_name=source_name,
                        role=role,
                    )
                )

    records_by_hash: dict[str, list[InventoryRecord]] = defaultdict(list)

    for record in records:
        records_by_hash[record.sha256].append(record)

    for digest, group in records_by_hash.items():
        labels = {record.class_name for record in group}

        if len(labels) > 1:
            details = "; ".join(
                f"{record.source_name}:{record.relative_path} "
                f"(clase={record.class_name})"
                for record in group
            )
            raise ValueError(
                "Contenido idéntico asignado a clases distintas. "
                f"SHA-256={digest}. Registros: {details}"
            )

        train_records = [
            record for record in group if record.role == TRAIN_ROLE
        ]
        evaluation_records = [
            record for record in group if record.role in EVALUATION_ROLES
        ]

        if train_records and evaluation_records:
            details = "; ".join(
                f"{record.source_name}:{record.relative_path} "
                f"(rol={record.role})"
                for record in group
            )
            raise ValueError(
                "Posible fuga de datos: contenido idéntico entre "
                "entrenamiento y evaluación. "
                f"SHA-256={digest}. Registros: {details}"
            )


    duplicate_groups = {
        digest: tuple(group)
        for digest, group in records_by_hash.items()
        if len(group) > 1
    }

    # Elegir la primera copia de entrenamiento según el orden de asignaciones.
    source_priority = {
        source_name.casefold(): index
        for index, (source_name, role, _) in enumerate(normalized_assignments)
        if role == TRAIN_ROLE
    }

    canonical_by_hash: dict[str, InventoryRecord] = {}
    excluded_training_copies: list[ExcludedTrainingCopy] = []

    for digest, group in records_by_hash.items():
        train_records = [
            record for record in group
            if record.role == TRAIN_ROLE
        ]

        if train_records:
            ordered_train_records = sorted(
                train_records,
                key=lambda record: (
                    source_priority[record.source_name.casefold()],
                    record.relative_path.casefold(),
                    record.relative_path,
                ),
            )

            canonical = ordered_train_records[0]
            canonical_by_hash[digest] = canonical

            for record in ordered_train_records[1:]:
                excluded_training_copies.append(
                    ExcludedTrainingCopy(
                        sha256=digest,
                        path=record.path,
                        relative_path=record.relative_path,
                        class_name=record.class_name,
                        source_name=record.source_name,
                        kept_path=canonical.path,
                        kept_relative_path=canonical.relative_path,
                        kept_source_name=canonical.source_name,
                    )
                )

        else:
            # Para hashes solo presentes en evaluación, conservar la primera
            # referencia como canónica sin excluir ninguna evaluación.
            canonical_by_hash[digest] = min(
                group,
                key=lambda record: (
                    record.source_name.casefold(),
                    record.relative_path.casefold(),
                    record.relative_path,
                ),
            )

    return GlobalInventory(
        records=tuple(records),
        duplicate_groups=duplicate_groups,
        canonical_by_hash=canonical_by_hash,
        excluded_training_copies=tuple(excluded_training_copies),
    )


def validate_global_preflight(
    inventory: GlobalInventory,
    class_names: Iterable[str],
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
    training_sources: Iterable[str] | None = None,
) -> None:
    """Valida la cobertura y viabilidad de los splits tras las exclusiones.

    Utiliza las copias canónicas propuestas por el inventario. No modifica
    archivos, datasets ni particiones guardadas.
    """
    names = _validate_class_names(class_names)
    class_to_label = {name: index for index, name in enumerate(names)}

    excluded_paths = {
        record.path
        for record in inventory.excluded_training_copies
    }

    effective_by_source: dict[str, list[InventoryRecord]] = defaultdict(list)

    for record in inventory.records:
        if record.role != TRAIN_ROLE:
            continue

        if record.path in excluded_paths:
            continue

        canonical = inventory.canonical_by_hash.get(record.sha256)

        # Solo aceptar la copia canónica del grupo.
        if canonical is None or canonical.path != record.path:
            continue

        effective_by_source[record.source_name].append(record)

    sources_to_validate = (
        list(training_sources)
        if training_sources is not None
        else list(effective_by_source)
    )

    for source_name in sources_to_validate:
        records = effective_by_source.get(source_name, [])

        if not records:
            raise ValueError(
                f"Preflight fallido para la fuente '{source_name}': "
                "no conserva imágenes después de las exclusiones globales."
            )

        present_classes = {record.class_name for record in records}
        missing_classes = sorted(set(names) - present_classes)

        if missing_classes:
            raise ValueError(
                f"Preflight fallido para la fuente '{source_name}': "
                "no conserva imágenes para las clases requeridas: "
                f"{', '.join(missing_classes)}."
            )

        labels = [class_to_label[record.class_name] for record in records]
        groups = [record.sha256 for record in records]

        try:
            dividir_dataset(
                etiquetas=labels,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                seed=seed,
                grupos=groups,
            )
        except ValueError as exc:
            raise ValueError(
                f"Preflight fallido para la fuente '{source_name}': "
                "los grupos únicos restantes no permiten generar "
                "particiones estratificadas válidas. "
                f"Detalle: {exc}"
            ) from exc
