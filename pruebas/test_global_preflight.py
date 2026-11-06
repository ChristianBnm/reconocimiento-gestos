
from types import SimpleNamespace

import pytest

from src.core.dataset.global_inventory import build_global_inventory
from src.core.dataset.split import dividir_dataset

from src.core.dataset.global_inventory import (
    build_global_inventory,
    validate_global_preflight,
)



def assignment(name, path, role="train_source"):
    return SimpleNamespace(
        source_name=name,
        path=str(path),
        role=role,
    )


def write_image(root, class_name, filename, content):
    path = root / class_name / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def validate_effective_source(records, source_name, config):
    """Simula la validación de una fuente después de las exclusiones."""
    effective = [
        record
        for record in records
        if record.source_name == source_name
    ]

    classes = set(config["class_names"])
    present_classes = {record.class_name for record in effective}
    missing_classes = sorted(classes - present_classes)

    if missing_classes:
        raise ValueError(
            f"La fuente '{source_name}' no conserva imágenes para las clases: "
            f"{', '.join(missing_classes)}"
        )

    labels = [config["class_names"].index(record.class_name) for record in effective]
    groups = [record.sha256 for record in effective]

    try:
        dividir_dataset(
            etiquetas=labels,
            train_size=config["train_size"],
            val_size=config["val_size"],
            test_size=config["test_size"],
            seed=config["seed"],
            grupos=groups,
        )
    except ValueError as exc:
        raise ValueError(
            f"La fuente '{source_name}' no tiene grupos suficientes "
            "para generar las particiones."
        ) from exc

    return effective


def test_preflight_acepta_fuente_con_grupos_suficientes(tmp_path):
    source = tmp_path / "original"

    for index in range(30):
        write_image(source, "A", f"a_{index}.jpg", f"imagen-a-{index}".encode())
        write_image(source, "B", f"b_{index}.jpg", f"imagen-b-{index}".encode())

    inventory = build_global_inventory(
        [assignment("original", source)],
        class_names=("A", "B"),
    )

    validate_global_preflight(
        inventory=inventory,
        class_names=("A", "B"),
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        seed=42,
    )


def test_preflight_rechaza_fuente_sin_una_clase(tmp_path):
    source = tmp_path / "original"

    for index in range(10):
        write_image(
            source,
            "A",
            f"a_{index}.jpg",
            f"imagen-a-{index}".encode(),
        )

    inventory = build_global_inventory(
        [assignment("original", source)],
        class_names=("A", "B"),
    )

    config = {
        "class_names": ("A", "B"),
        "train_size": 0.7,
        "val_size": 0.15,
        "test_size": 0.15,
        "seed": 42,
    }

    with pytest.raises(ValueError, match="original.*B"):
        validate_effective_source(inventory.records, "original", config)


def test_preflight_rechaza_grupos_insuficientes_para_particionar(tmp_path):
    source = tmp_path / "original"

    # Solo dos grupos únicos de la clase A: insuficientes para dos
    # divisiones estratificadas consecutivas.
    write_image(source, "A", "a1.jpg", b"imagen-a1")
    write_image(source, "A", "a2.jpg", b"imagen-a2")

    # La clase B sí tiene grupos suficientes.
    for index in range(20):
        write_image(
            source,
            "B",
            f"b_{index}.jpg",
            f"imagen-b-{index}".encode(),
        )

    inventory = build_global_inventory(
        [assignment("original", source)],
        class_names=("A", "B"),
    )

    config = {
        "class_names": ("A", "B"),
        "train_size": 0.7,
        "val_size": 0.15,
        "test_size": 0.15,
        "seed": 42,
    }

    with pytest.raises(ValueError, match="original.*grupos suficientes"):
        validate_effective_source(inventory.records, "original", config)


def test_preflight_no_modifica_los_archivos_originales(tmp_path):
    source = tmp_path / "original"
    image_path = write_image(source, "A", "a.jpg", b"imagen-original")

    inventory = build_global_inventory(
        [assignment("original", source)],
        class_names=("A",),
    )

    config = {
        "class_names": ("A",),
        "train_size": 0.7,
        "val_size": 0.15,
        "test_size": 0.15,
        "seed": 42,
    }

    # Aunque esta configuración no alcanza para todas las particiones,
    # la validación no debe alterar la imagen fuente.
    try:
        validate_effective_source(inventory.records, "original", config)
    except ValueError:
        pass

    assert image_path.exists()
    assert image_path.read_bytes() == b"imagen-original"
