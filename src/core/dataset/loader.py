from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
from torch.utils.data import Dataset

from src.core.configuracion.settings import CLASS_NAMES


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    class_name: str


class GestosDataset(Dataset):
    """Dataset de imágenes organizado por carpetas de clase.

    Las clases se determinan mediante `class_names`, de modo que el mapeo
    clase -> índice no depende del orden del sistema de archivos.
    """

    def __init__(
        self,
        carpetas_imagenes: str | Path | Sequence[str | Path],
        transformacion=None,
        class_names: Sequence[str] = CLASS_NAMES,
    ) -> None:
        if isinstance(carpetas_imagenes, (str, Path)):
            carpetas_imagenes = [carpetas_imagenes]

        self.carpetas_imagenes = [Path(p) for p in carpetas_imagenes]
        self.transformacion = transformacion
        self.clases = list(class_names)
        self.class_to_idx = {clase: i for i, clase in enumerate(self.clases)}

        self.records = self._scan_records()

        # Compatibilidad con la API anterior.
        self.rutas_imagenes = [record.path for record in self.records]
        self.etiquetas = [record.label for record in self.records]

    def _scan_records(self) -> list[ImageRecord]:
        records: list[ImageRecord] = []

        for root in self.carpetas_imagenes:
            if not root.exists():
                raise FileNotFoundError(f"No existe el dataset: {root}")
            if not root.is_dir():
                raise NotADirectoryError(f"No es una carpeta: {root}")

        for class_name in self.clases:
            label = self.class_to_idx[class_name]

            for root in self.carpetas_imagenes:
                class_dir = root / class_name
                if not class_dir.is_dir():
                    continue

                for path in sorted(class_dir.iterdir(), key=lambda p: p.name.lower()):
                    if not path.is_file():
                        continue
                    if path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    records.append(
                        ImageRecord(
                            path=path,
                            label=label,
                            class_name=class_name,
                        )
                    )

        if not records:
            raise RuntimeError("No se encontraron imágenes válidas en el dataset.")

        return records

    def __len__(self) -> int:
        return len(self.records)

    def load_image(self, index: int):
        path = self.records[index].path
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"No se pudo cargar la imagen: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int):
        image = self.load_image(index)
        label = self.records[index].label

        if self.transformacion is not None:
            image = self.transformacion(image=image)["image"]

        return image, label


class DatasetSubset(Dataset):
    """Vista de un dataset usando índices concretos y una transformación."""

    def __init__(self, dataset: GestosDataset, indices: Iterable[int], transformacion=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.transformacion = transformacion

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        original_index = self.indices[index]
        image = self.dataset.load_image(original_index)
        label = self.dataset.records[original_index].label

        if self.transformacion is not None:
            image = self.transformacion(image=image)["image"]

        return image, label

    def original_index(self, index: int) -> int:
        return self.indices[index]
