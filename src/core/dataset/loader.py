from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from torch.utils.data import Dataset

from src.core.configuracion.settings import CLASS_NAMES


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    class_name: str
    source: str
    relative_path: str
    group_id: str


@dataclass(frozen=True)
class ExcludedDuplicate:
    path: Path
    relative_path: str
    kept_path: Path
    kept_relative_path: str
    label: int
    class_name: str
    source: str
    group_id: str


class GestosDataset(Dataset):
    """Dataset de clasificación organizado por una carpeta por clase.

    ``class_names`` define explícitamente el orden de etiquetas. Cada imagen
    recibe un SHA-256 como ``group_id`` para impedir que copias exactas caigan
    en particiones distintas. Se conserva la ruta y la fuente para auditar los
    experimentos. No se eliminan archivos del disco.
    """

    def __init__(
        self,
        carpetas_imagenes: str | Path | Sequence[str | Path],
        transformacion=None,
        class_names: Sequence[str] | None = None,
        source_names: Sequence[str] | None = None,
        require_all_classes: bool = True,
        deduplicate_exact_duplicates: bool = True,
    ) -> None:
        if isinstance(carpetas_imagenes, (str, Path)):
            carpetas_imagenes = [carpetas_imagenes]
        roots = [Path(root).expanduser().resolve() for root in carpetas_imagenes]
        if not roots:
            raise ValueError("Debe indicarse al menos una carpeta de dataset.")
        if len(set(roots)) != len(roots):
            raise ValueError("Se indicó una misma carpeta de dataset más de una vez.")
        for root in roots:
            if not root.exists():
                raise FileNotFoundError(f"No existe el dataset: {root}")
            if not root.is_dir():
                raise NotADirectoryError(f"No es una carpeta: {root}")

        names = tuple(class_names) if class_names is not None else CLASS_NAMES
        if not names or len(set(names)) != len(names):
            raise ValueError("class_names debe ser una secuencia no vacía y sin duplicados.")
        self.clases = list(names)
        self.class_to_idx = {name: index for index, name in enumerate(names)}
        self.carpetas_imagenes = roots
        self.transformacion = transformacion

        if source_names is None:
            sources = [root.name for root in roots]
        else:
            sources = list(source_names)
        if len(sources) != len(roots) or len(set(sources)) != len(sources):
            raise ValueError("source_names debe tener un nombre único por carpeta.")
        self.source_names = sources
        self.deduplicate_exact_duplicates = bool(deduplicate_exact_duplicates)
        self.excluded_duplicates: list[ExcludedDuplicate] = []
        self.records = self._scan_records(require_all_classes=require_all_classes)
        self.rutas_imagenes = [record.path for record in self.records]
        self.etiquetas = [record.label for record in self.records]

    def _scan_records(self, require_all_classes: bool) -> list[ImageRecord]:
        records: list[ImageRecord] = []
        per_class = {class_name: 0 for class_name in self.clases}
        first_by_source_hash: dict[tuple[str, str], ImageRecord] = {}
        for root in self.carpetas_imagenes:
            unknown_dirs = []
            for child in root.iterdir():
                if not child.is_dir() or child.name in self.class_to_idx:
                    continue
                if any(
                    path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
                    for path in child.iterdir()
                ):
                    unknown_dirs.append(child.name)
            if unknown_dirs:
                raise ValueError(
                    f"La fuente {root} contiene carpetas con imágenes que no están en el vocabulario: "
                    + ", ".join(sorted(unknown_dirs))
                    + ". Agregá las etiquetas a clases.json o mové esas carpetas fuera del dataset."
                )

        # Orden estable: vocabulario configurado, fuentes configuradas, nombre de archivo.
        for class_name in self.clases:
            label = self.class_to_idx[class_name]
            for root, source in zip(self.carpetas_imagenes, self.source_names):
                class_dir = root / class_name
                if not class_dir.is_dir():
                    continue
                for path in sorted(class_dir.iterdir(), key=lambda item: item.name.casefold()):
                    if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
                        continue
                    raw = path.read_bytes()
                    group_id = hashlib.sha256(raw).hexdigest()
                    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if decoded is None:
                        raise ValueError(f"Imagen ilegible o corrupta: {path}")
                    relative_path = path.relative_to(root).as_posix()
                    key = (source, group_id)
                    previous = first_by_source_hash.get(key)
                    if previous is not None and previous.label != label:
                        raise ValueError(
                            "Contenido idéntico asignado a clases distintas dentro de una fuente: "
                            f"{previous.path} ({previous.class_name}) y {path} ({class_name})."
                        )
                    if previous is not None and self.deduplicate_exact_duplicates:
                        self.excluded_duplicates.append(
                            ExcludedDuplicate(
                                path=path,
                                relative_path=relative_path,
                                kept_path=previous.path,
                                kept_relative_path=previous.relative_path,
                                label=label,
                                class_name=class_name,
                                source=source,
                                group_id=group_id,
                            )
                        )
                        continue
                    record = ImageRecord(
                        path=path,
                        label=label,
                        class_name=class_name,
                        source=source,
                        relative_path=relative_path,
                        group_id=group_id,
                    )
                    records.append(record)
                    first_by_source_hash.setdefault(key, record)
                    per_class[class_name] += 1

        if not records:
            raise RuntimeError("No se encontraron imágenes en las carpetas de clases configuradas.")
        missing = [name for name, count in per_class.items() if count == 0]
        if require_all_classes and missing:
            raise ValueError(
                "No se encontraron imágenes para estas clases configuradas: "
                + ", ".join(missing)
                + ". Revisá clases.json y la estructura del dataset."
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def load_image(self, index: int):
        path = self.records[index].path
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"No se pudo cargar la imagen: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int):
        image = self.load_image(index)
        label = self.records[index].label
        if self.transformacion is not None:
            transformed = self.transformacion(image=image)
            image = transformed["image"] if isinstance(transformed, dict) else transformed
        return image, label


class DatasetSubset(Dataset):
    """Vista de un dataset a partir de índices, con transformación propia."""

    def __init__(self, dataset: GestosDataset, indices: Iterable[int], transformacion=None):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        if any(index < 0 or index >= len(dataset) for index in self.indices):
            raise IndexError("El subset contiene índices fuera del rango del dataset.")
        self.transformacion = transformacion

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        original_index = self.indices[index]
        image = self.dataset.load_image(original_index)
        label = self.dataset.records[original_index].label
        if self.transformacion is not None:
            transformed = self.transformacion(image=image)
            image = transformed["image"] if isinstance(transformed, dict) else transformed
        return image, label

    def original_index(self, index: int) -> int:
        return self.indices[index]
