from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from src.core.configuracion.catalogo import cargar_catalogos
from src.core.configuracion.settings import cargar_clases
from src.core.dataset.loader import IMAGE_EXTENSIONS


IGNORED_DIRECTORY_NAMES = {".git", "__pycache__", ".ipynb_checkpoints"}


def _resolve_project_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def inspect_dataset_directory(
    dataset_path: str | Path,
    expected_class_names: Sequence[str] | None = None,
) -> dict:
    """Inspecciona la estructura por carpetas sin cargar imágenes en memoria.

    Cada subdirectorio inmediato del dataset debe representar una clase y
    contener al menos una imagen soportada directamente dentro de esa carpeta.
    La decodificación completa de las imágenes continúa a cargo de GestosDataset
    cuando se prepara una ejecución.
    """
    root = Path(dataset_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio del dataset: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"La ruta del dataset no es una carpeta: {root}")

    class_dirs = sorted(
        (
            child for child in root.iterdir()
            if child.is_dir()
            and not child.name.startswith(".")
            and child.name not in IGNORED_DIRECTORY_NAMES
        ),
        key=lambda item: (item.name.casefold(), item.name),
    )
    if not class_dirs:
        raise ValueError(
            "No se encontraron carpetas de clases. Se espera una carpeta por clase."
        )

    names_by_casefold: dict[str, Path] = {}
    for directory in class_dirs:
        key = directory.name.casefold()
        if key in names_by_casefold:
            raise ValueError(
                "Hay carpetas de clases que solo se diferencian por mayúsculas/minúsculas: "
                f"'{names_by_casefold[key].name}' y '{directory.name}'."
            )
        names_by_casefold[key] = directory

    if expected_class_names is None:
        class_names = tuple(directory.name for directory in class_dirs)
    else:
        class_names = tuple(expected_class_names)
        if not class_names or len(set(class_names)) != len(class_names):
            raise ValueError("El vocabulario esperado está vacío o contiene duplicados.")
        actual_names = {directory.name for directory in class_dirs}
        expected_names = set(class_names)
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        if missing or extra:
            details = []
            if missing:
                details.append("faltan clases: " + ", ".join(missing))
            if extra:
                details.append("sobran carpetas: " + ", ".join(extra))
            raise ValueError("La estructura no coincide con el vocabulario: " + "; ".join(details))

    image_counts: dict[str, int] = {}
    for class_name in class_names:
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise ValueError(f"No existe la carpeta de la clase '{class_name}'.")

        images = sorted(
            (
                item for item in class_dir.iterdir()
                if item.is_file() and item.suffix.casefold() in IMAGE_EXTENSIONS
            ),
            key=lambda item: item.name.casefold(),
        )
        if not images:
            nested_images = any(
                child.is_dir() and any(
                    file.is_file() and file.suffix.casefold() in IMAGE_EXTENSIONS
                    for file in child.rglob("*")
                )
                for child in class_dir.iterdir()
            )
            if nested_images:
                raise ValueError(
                    f"La clase '{class_name}' contiene imágenes en subcarpetas. "
                    "El loader espera los archivos directamente dentro de cada clase."
                )
            raise ValueError(f"La carpeta de clase '{class_name}' no contiene imágenes.")

        empty_files = [image.name for image in images if image.stat().st_size == 0]
        if empty_files:
            raise ValueError(
                f"La clase '{class_name}' contiene archivos de imagen vacíos: "
                + ", ".join(empty_files[:5])
            )
        image_counts[class_name] = len(images)

    root_images = [
        item.name for item in root.iterdir()
        if item.is_file() and item.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    if root_images:
        raise ValueError(
            "Hay imágenes directamente en la raíz del dataset. "
            "Cada imagen debe estar dentro de una carpeta de clase: "
            + ", ".join(sorted(root_images)[:5])
        )

    return {
        "class_names": class_names,
        "images_by_class": image_counts,
        "num_classes": len(class_names),
        "num_images": sum(image_counts.values()),
    }


def _write_class_manifest(
    project_root: Path,
    source_name: str,
    dataset_path: Path,
    class_names: Sequence[str],
) -> str:
    """Escribe un vocabulario inmutable y direccionado por contenido."""
    payload = {"classes": list(class_names)}
    signature = json.dumps(
        {
            "source_name": source_name,
            "dataset_path": str(dataset_path.resolve()),
            "classes": list(class_names),
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(signature).hexdigest()[:12]
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source_name).strip("._-") or "dataset"
    manifest_dir = project_root / "results" / "dataset_catalog"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{safe_name}-{digest}.classes.json"

    if not manifest_path.is_file():
        fd, temporary = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_dir
        )
        temporary_path = Path(temporary)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, manifest_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    return manifest_path.relative_to(project_root).as_posix()


def _discover_uncached(project_root_text: str) -> dict:
    project_root = Path(project_root_text).resolve()
    data_root = project_root / "data"
    configured_sources, _ = cargar_catalogos()
    default_classes = cargar_clases()

    valid_sources: list[dict] = []
    invalid_sources: list[dict] = []
    configured_paths: set[Path] = set()
    known_names: set[str] = set()

    for source_name, definition in configured_sources.items():
        dataset_path = _resolve_project_path(project_root, definition["path"])
        configured_paths.add(dataset_path)
        known_names.add(source_name.casefold())
        try:
            inspection = inspect_dataset_directory(dataset_path, default_classes)
            valid_sources.append({
                "name": source_name,
                "path": dataset_path.relative_to(project_root).as_posix()
                if dataset_path.is_relative_to(project_root) else str(dataset_path),
                "description": definition.get("description", "Fuente configurada"),
                "class_file": None,
                "classes": list(inspection["class_names"]),
                "num_classes": inspection["num_classes"],
                "num_images": inspection["num_images"],
                "images_by_class": inspection["images_by_class"],
                "deduplicate_exact_duplicates": definition.get(
                    "deduplicate_exact_duplicates", True
                ),
                "origin": "configured",
            })
        except (OSError, ValueError, RuntimeError) as exc:
            invalid_sources.append({
                "name": source_name,
                "path": definition["path"],
                "reason": str(exc),
                "origin": "configured",
            })

    if data_root.is_dir():
        for dataset_path in sorted(data_root.iterdir(), key=lambda item: item.name.casefold()):
            if (
                not dataset_path.is_dir()
                or dataset_path.name.startswith(".")
                or dataset_path.name in IGNORED_DIRECTORY_NAMES
            ):
                continue

            resolved_path = dataset_path.resolve()
            if resolved_path in configured_paths:
                continue

            source_name = dataset_path.name
            if source_name.casefold() in known_names:
                invalid_sources.append({
                    "name": source_name,
                    "path": dataset_path.relative_to(project_root).as_posix(),
                    "reason": "El nombre del directorio entra en conflicto con una fuente configurada.",
                    "origin": "automatic",
                })
                continue

            try:
                inspection = inspect_dataset_directory(resolved_path)
                class_file = _write_class_manifest(
                    project_root,
                    source_name,
                    resolved_path,
                    inspection["class_names"],
                )
                valid_sources.append({
                    "name": source_name,
                    "path": resolved_path.relative_to(project_root).as_posix(),
                    "description": "Dataset detectado automáticamente en data/.",
                    "class_file": class_file,
                    "classes": list(inspection["class_names"]),
                    "num_classes": inspection["num_classes"],
                    "num_images": inspection["num_images"],
                    "images_by_class": inspection["images_by_class"],
                    "deduplicate_exact_duplicates": True,
                    "origin": "automatic",
                })
                known_names.add(source_name.casefold())
            except (OSError, ValueError, RuntimeError) as exc:
                invalid_sources.append({
                    "name": source_name,
                    "path": resolved_path.relative_to(project_root).as_posix(),
                    "reason": str(exc),
                    "origin": "automatic",
                })

    return {"sources": valid_sources, "invalid_sources": invalid_sources}


@lru_cache(maxsize=8)
def _cached_dataset_catalog(project_root_text: str) -> dict:
    return _discover_uncached(project_root_text)


def get_dataset_catalog(
    project_root: str | Path,
    *,
    refresh: bool = False,
) -> dict:
    """Devuelve el catálogo detectado; se calcula una vez por proceso/root."""
    root_text = str(Path(project_root).expanduser().resolve())
    if refresh:
        _cached_dataset_catalog.cache_clear()
    return copy.deepcopy(_cached_dataset_catalog(root_text))


def clear_dataset_catalog_cache() -> None:
    """Limpia la caché (útil para pruebas automatizadas)."""
    _cached_dataset_catalog.cache_clear()


def get_source_definitions(project_root: str | Path) -> dict[str, dict]:
    """Adapta el catálogo detectado al formato usado por prepare_sources()."""
    definitions: dict[str, dict] = {}
    for source in get_dataset_catalog(project_root)["sources"]:
        definitions[source["name"]] = {
            "path": source["path"],
            "description": source["description"],
            "deduplicate_exact_duplicates": source["deduplicate_exact_duplicates"],
        }
    return definitions
