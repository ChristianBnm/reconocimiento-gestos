from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    train: list[int]
    val: list[int]
    test: list[int]

    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


def dividir_dataset(
    etiquetas: Sequence[int],
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> DatasetSplit:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size debe ser igual a 1.0")

    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("Las proporciones de train, val y test deben ser mayores que 0.")

    indices = list(range(len(etiquetas)))

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=etiquetas,
    )

    val_fraction_of_temp = val_size / (val_size + test_size)
    temp_labels = [etiquetas[i] for i in temp_indices]

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=1 - val_fraction_of_temp,
        random_state=seed,
        stratify=temp_labels,
    )

    return DatasetSplit(
        train=sorted(train_indices),
        val=sorted(val_indices),
        test=sorted(test_indices),
    )


def dataset_fingerprint(dataset) -> str:
    """Hash estable de ruta + etiqueta para validar el dataset del split."""
    digest = hashlib.sha256()
    for record in dataset.records:
        line = f"{record.label}\t{record.path.resolve()}\n".encode("utf-8")
        digest.update(line)
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
    payload = {
        "version": 1,
        "seed": seed,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "dataset_size": len(dataset),
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "class_names": list(dataset.clases),
        "indices": asdict(split),
    }

    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def cargar_split(ruta: Path, dataset) -> DatasetSplit:
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el split guardado: {ruta}")

    with ruta.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if payload.get("dataset_size") != len(dataset):
        raise ValueError(
            "El tamaño del dataset actual no coincide con el dataset usado para crear el split."
        )

    if payload.get("class_names") != list(dataset.clases):
        raise ValueError("Las clases actuales no coinciden con las del split guardado.")

    fingerprint = dataset_fingerprint(dataset)
    if payload.get("dataset_fingerprint") != fingerprint:
        raise ValueError(
            "El fingerprint del dataset no coincide. El split ya no corresponde al dataset actual."
        )

    indices = payload["indices"]
    return DatasetSplit(
        train=list(indices["train"]),
        val=list(indices["val"]),
        test=list(indices["test"]),
    )
