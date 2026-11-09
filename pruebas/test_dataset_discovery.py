
from __future__ import annotations

import cv2
import numpy as np

from src.core.configuracion.discovery import get_dataset_catalog


def _crear_imagen(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    imagen = np.zeros((16, 16, 3), dtype=np.uint8)
    imagen[:, :] = (80, 120, 160)
    assert cv2.imwrite(str(path), imagen)


def _crear_dataset(root, classes):
    for class_name in classes:
        _crear_imagen(root / class_name / f"{class_name.lower()}_001.jpg")


def test_discovery_detecta_dataset_automatico_con_vocabulario_propio(tmp_path):
    dataset = tmp_path / "data" / "gestos_personalizados"
    _crear_dataset(dataset, ("ALTO", "AYUDA", "OK"))

    catalog = get_dataset_catalog(tmp_path, refresh=True)

    source = next(
        item for item in catalog["sources"]
        if item["name"] == "gestos_personalizados"
    )

    assert source["origin"] == "automatic"
    assert source["classes"] == ["ALTO", "AYUDA", "OK"]
    assert source["num_classes"] == 3
    assert source["num_images"] == 3
    assert source["class_file"] is not None

    manifest = tmp_path / source["class_file"]
    assert manifest.is_file()


def test_discovery_informa_motivo_de_dataset_invalido(tmp_path):
    dataset = tmp_path / "data" / "dataset_invalido"
    (dataset / "CLASE_VACIA").mkdir(parents=True)

    catalog = get_dataset_catalog(tmp_path, refresh=True)

    invalid = next(
        item for item in catalog["invalid_sources"]
        if item["name"] == "dataset_invalido"
    )

    assert invalid["origin"] == "automatic"
    assert "no contiene imágenes" in invalid["reason"]


def test_discovery_actualiza_catalogo_al_refrescar(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()

    first = data_root / "dataset_inicial"
    _crear_dataset(first, ("A", "B"))

    initial_catalog = get_dataset_catalog(tmp_path, refresh=True)
    initial_names = {
        item["name"] for item in initial_catalog["sources"]
    }
    assert "dataset_inicial" in initial_names

    second = data_root / "dataset_nuevo"
    _crear_dataset(second, ("C", "D"))

    cached_catalog = get_dataset_catalog(tmp_path)
    cached_names = {
        item["name"] for item in cached_catalog["sources"]
    }
    assert "dataset_nuevo" not in cached_names

    refreshed_catalog = get_dataset_catalog(tmp_path, refresh=True)
    refreshed_names = {
        item["name"] for item in refreshed_catalog["sources"]
    }
    assert "dataset_nuevo" in refreshed_names


def test_discovery_refresh_reconoce_dataset_extendido_agregado_despues(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    get_dataset_catalog(tmp_path, refresh=True)

    extended = data_root / "dataset_extendido"
    _crear_dataset(extended, ("0", "1", "A"))

    # Sin refresh el catálogo conserva la instantánea anterior.
    assert "dataset_extendido" not in {
        item["name"] for item in get_dataset_catalog(tmp_path)["sources"]
    }

    refreshed = get_dataset_catalog(tmp_path, refresh=True)
    source = next(
        item for item in refreshed["sources"]
        if item["name"] == "dataset_extendido"
    )
    assert source["classes"] == ["0", "1", "A"]
    assert source["class_file"]
    assert (tmp_path / source["class_file"]).is_file()
