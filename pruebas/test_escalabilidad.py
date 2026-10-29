from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.core.configuracion.settings import (
    ExperimentConfig,
    ModelSpec,
    cargar_clases,
)
from src.core.dataset.loader import GestosDataset
from src.core.dataset.split import DatasetSplit, dividir_dataset, guardar_split, cargar_split
from src.core.entrenamiento.trainer import Trainer
from src.core.modelos.factory import cargar_checkpoint


def test_cargar_clases_admite_lista_y_objeto(tmp_path):
    path = tmp_path / "clases.json"
    path.write_text(json.dumps(["GESTO_A", "GESTO_B", "VERBO_1"]), encoding="utf-8")
    assert cargar_clases(path) == ("GESTO_A", "GESTO_B", "VERBO_1")
    path.write_text(json.dumps({"classes": ["A", "B"]}), encoding="utf-8")
    assert cargar_clases(path) == ("A", "B")


def test_cargar_clases_rechaza_duplicadas(tmp_path):
    path = tmp_path / "clases.json"
    path.write_text('["A", "A"]', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicados"):
        cargar_clases(path)


def test_configuracion_admite_cantidad_variable_de_clases():
    config = ExperimentConfig(
        model=ModelSpec("modelo_test", "arquitectura_test", "test"),
        class_names=("A", "B", "VERBO_HOLA"),
    )
    assert config.num_classes == 3
    assert config.class_to_idx == {"A": 0, "B": 1, "VERBO_HOLA": 2}
    assert config.to_dict()["num_classes"] == 3


def _write_rgb_image(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def test_dataset_usa_vocabulario_explicito_y_fingerprint_de_contenido(tmp_path):
    root = tmp_path / "datos"
    for class_name in ("GESTO_Z", "GESTO_A"):
        (root / class_name).mkdir(parents=True)
        for i in range(4):
            _write_rgb_image(root / class_name / f"{i}.jpg", i + (10 if class_name == "GESTO_Z" else 30))
    dataset = GestosDataset(root, class_names=("GESTO_Z", "GESTO_A"), source_names=["test"])
    assert dataset.clases == ["GESTO_Z", "GESTO_A"]
    assert dataset.class_to_idx == {"GESTO_Z": 0, "GESTO_A": 1}
    image, label = dataset[0]
    assert image.shape == (32, 32, 3)
    assert label == 0


def test_split_agrupa_contenido_identico_y_cubre_todos_los_indices():
    labels = []
    groups = []
    for label in range(3):
        for group_idx in range(12):
            group = f"clase{label}-imagen{group_idx}"
            labels.append(label)
            groups.append(group)
            if group_idx in {0, 2}:
                # Otra copia del mismo contenido: debe quedar en la misma partición.
                labels.append(label)
                groups.append(group)
    split = dividir_dataset(labels, 0.7, 0.15, 0.15, 123, grupos=groups)
    split.validate(len(labels))
    group_parts = {}
    for part_name in ("train", "val", "test"):
        for index in getattr(split, part_name):
            group_parts.setdefault(groups[index], set()).add(part_name)
    assert all(len(parts) == 1 for parts in group_parts.values())
    assert sorted(split.train + split.val + split.test) == list(range(len(labels)))


def test_split_rechaza_mismo_grupo_con_etiquetas_distintas():
    with pytest.raises(ValueError, match="clases distintas"):
        dividir_dataset([0, 1, 0, 1, 0, 1], 0.5, 0.25, 0.25, 42, grupos=["x", "x", "a", "b", "c", "d"])


def test_split_guardado_detecta_cambio_en_contenido(tmp_path):
    root = tmp_path / "datos"
    for class_name in ("A", "B"):
        (root / class_name).mkdir(parents=True)
        for i in range(10):
            _write_rgb_image(root / class_name / f"{i}.jpg", i + (100 if class_name == "B" else 0))
    dataset = GestosDataset(root, class_names=("A", "B"), source_names=["original"])
    split = dividir_dataset(dataset.etiquetas, 0.7, 0.15, 0.15, 42, grupos=[r.group_id for r in dataset.records])
    path = tmp_path / "split.json"
    guardar_split(path, split, dataset, 42, 0.7, 0.15, 0.15)
    assert cargar_split(path, dataset, 42, 0.7, 0.15, 0.15) == split

    _write_rgb_image(root / "A" / "0.jpg", 999)
    modified_dataset = GestosDataset(root, class_names=("A", "B"), source_names=["original"])
    with pytest.raises(ValueError, match="cambió"):
        cargar_split(path, modified_dataset, 42, 0.7, 0.15, 0.15)


def test_checkpoint_guarda_y_valida_mapeo_de_clases(tmp_path):
    class_names = ("A", "B", "NUEVO_GESTO")
    config = ExperimentConfig(
        model=ModelSpec("test", "test_arch", "test", pretrained=False),
        class_names=class_names,
        epochs=1,
        patience=1,
        freeze_backbone=False,
    )
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, len(class_names)))
    images = torch.randn(12, 3, 2, 2)
    labels = torch.tensor([0, 1, 2] * 4)
    loader = DataLoader(TensorDataset(images, labels), batch_size=3)
    checkpoint = tmp_path / "mejor.pth"
    history = tmp_path / "history.json"
    Trainer(model, loader, loader, config, checkpoint, history).fit()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["class_names"] == list(class_names)
    assert payload["class_to_idx"] == {"A": 0, "B": 1, "NUEVO_GESTO": 2}
    model_same = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, len(class_names)))
    cargar_checkpoint(model_same, checkpoint, torch.device("cpu"), class_names=class_names)
    model_same_wrong_vocab = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, len(class_names)))
    with pytest.raises(ValueError, match="vocabulario"):
        cargar_checkpoint(
            model_same_wrong_vocab,
            checkpoint,
            torch.device("cpu"),
            class_names=("B", "A", "NUEVO_GESTO"),
        )


def test_catalogo_de_fuentes_y_escenarios_es_configurable(tmp_path):
    from src.core.configuracion.catalogo import cargar_catalogos

    sources_path = tmp_path / "fuentes.json"
    scenarios_path = tmp_path / "escenarios.json"
    sources_path.write_text(
        json.dumps({"captura_a": {"path": "data/a"}, "captura_b": {"path": "data/b"}}),
        encoding="utf-8",
    )
    scenarios_path.write_text(
        json.dumps({"solo_a": {"sources": ["captura_a"]}, "ambas": {"sources": ["captura_a", "captura_b"]}}),
        encoding="utf-8",
    )
    sources, scenarios = cargar_catalogos(sources_path, scenarios_path)
    assert set(sources) == {"captura_a", "captura_b"}
    assert scenarios["ambas"]["sources"] == ["captura_a", "captura_b"]


def test_ruta_split_versiona_dataset_semilla_y_proporciones(tmp_path):
    from src.core.utils.paths import ProjectPaths

    paths = ProjectPaths(tmp_path)
    first = paths.source_split_path("original", "abc123456789", 42, 0.7, 0.15, 0.15)
    changed_dataset = paths.source_split_path("original", "def123456789", 42, 0.7, 0.15, 0.15)
    changed_seed = paths.source_split_path("original", "abc123456789", 43, 0.7, 0.15, 0.15)
    assert first != changed_dataset
    assert first != changed_seed


def test_dataset_falla_si_hay_una_clase_no_configurada(tmp_path):
    root = tmp_path / "datos"
    (root / "A").mkdir(parents=True)
    (root / "B").mkdir()
    (root / "A" / "a.jpg").parent.mkdir(exist_ok=True)
    _write_rgb_image(root / "A" / "a.jpg", 1)
    _write_rgb_image(root / "B" / "b.jpg", 2)
    with pytest.raises(ValueError, match="no están en el vocabulario"):
        GestosDataset(root, class_names=("A",))


def test_prepare_sources_reutiliza_particiones_y_versiona_cambios(tmp_path):
    from src.core.dataset.experiment import SOURCE_NAMES, prepare_sources
    from src.core.utils.paths import ProjectPaths

    data_roots = {"original": tmp_path / "original", "complementario": tmp_path / "complementario"}
    for source_idx, (source_name, root) in enumerate(data_roots.items()):
        for class_idx, class_name in enumerate(("A", "B", "C")):
            class_dir = root / class_name
            class_dir.mkdir(parents=True)
            for image_idx in range(12):
                _write_rgb_image(
                    class_dir / f"{image_idx:03d}.jpg",
                    seed=1000 + source_idx * 1000 + class_idx * 100 + image_idx,
                )
    config = ExperimentConfig(
        model=ModelSpec("unit_test", "unit_test_arch", "test", pretrained=False),
        class_names=("A", "B", "C"),
        split_seed=17,
    )
    paths = ProjectPaths(tmp_path / "repo")
    overrides = {name: path for name, path in data_roots.items()}
    first = prepare_sources(
        paths,
        config,
        requested_sources=SOURCE_NAMES,
        path_overrides=overrides,
        create_missing_splits=True,
    )
    second = prepare_sources(
        paths,
        config,
        requested_sources=SOURCE_NAMES,
        path_overrides=overrides,
        create_missing_splits=False,
    )
    for name in SOURCE_NAMES:
        assert first[name].split_path == second[name].split_path
        assert first[name].split == second[name].split
        assert first[name].split_path.is_file()

    old_path = first["original"].split_path
    _write_rgb_image(data_roots["original"] / "A" / "000.jpg", seed=987654)
    changed = prepare_sources(
        paths,
        config,
        requested_sources=SOURCE_NAMES,
        path_overrides=overrides,
        create_missing_splits=True,
    )
    assert changed["original"].split_path != old_path
    assert old_path.is_file()
    assert changed["original"].split_path.is_file()


def test_dataset_excluye_duplicados_exactos_sin_borrar_archivos(tmp_path):
    import shutil

    root = tmp_path / "datos"
    class_dir = root / "A"
    class_dir.mkdir(parents=True)
    first = class_dir / "A_0001.jpg"
    second = class_dir / "A_0002.jpg"
    third = class_dir / "A_0003.jpg"
    _write_rgb_image(first, 111)
    shutil.copyfile(first, second)
    _write_rgb_image(third, 112)

    dataset = GestosDataset(root, class_names=("A",), source_names=["original"])
    assert len(dataset) == 2
    assert len(dataset.excluded_duplicates) == 1
    assert dataset.excluded_duplicates[0].path == second
    assert first.is_file() and second.is_file() and third.is_file()

    keep_all = GestosDataset(
        root,
        class_names=("A",),
        source_names=["original"],
        deduplicate_exact_duplicates=False,
    )
    assert len(keep_all) == 3
    assert len(keep_all.excluded_duplicates) == 0


def test_comparacion_conserva_modelos_pendientes_y_ordena_metricas(tmp_path):
    from src.core.evaluacion.comparacion import cargar_comparacion, ordenar_comparacion
    from src.core.utils.io import save_json
    from src.core.utils.paths import ProjectPaths

    paths = ProjectPaths(tmp_path)
    _, results_dir = paths.experiment_dirs("combinado", "modelo_a")
    save_json({"class_names": ["A", "B", "CLASE CON ESPACIOS"]}, results_dir / "config.json")
    save_json({"n_samples": 10, "accuracy": 0.9, "f1_macro": 0.85}, results_dir / "metrics_original.json")
    rows = cargar_comparacion(tmp_path, "combinado", ["modelo_a", "modelo_b"], "original")
    assert rows[0]["status"] == "ok"
    assert rows[0]["class_count"] == 3
    assert rows[1]["status"] == "missing_metrics"
    ordered = ordenar_comparacion(rows, "f1_macro")
    assert [row["model"] for row in ordered] == ["modelo_a", "modelo_b"]


def test_matriz_confusion_csv_escapa_etiquetas_complejas(tmp_path):
    import csv
    from src.core.evaluacion.reporting import guardar_matriz_confusion

    guardar_matriz_confusion(
        [0, 1], [0, 1], ["GESTO, A", "GESTO B"], tmp_path, prefix="test_matrix"
    )
    with (tmp_path / "test_matrix.csv").open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    assert rows[0][1:] == ["GESTO, A", "GESTO B"]
    assert rows[1][0] == "GESTO, A"


def test_normalizacion_se_adapta_a_metadatos_del_modelo():
    from src.core.modelos.factory import adaptar_normalizacion_a_modelo

    config = ExperimentConfig(
        model=ModelSpec("test", "test_arch", "test", pretrained=False),
        class_names=("A", "B"),
    )

    class FakeModel:
        pretrained_cfg = {"mean": (0.5, 0.5, 0.5), "std": (0.25, 0.25, 0.25)}

    updated = adaptar_normalizacion_a_modelo(config, FakeModel())
    assert updated.normalization_mean == (0.5, 0.5, 0.5)
    assert updated.normalization_std == (0.25, 0.25, 0.25)
    assert config.normalization_mean != updated.normalization_mean
