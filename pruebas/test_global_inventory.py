
from types import SimpleNamespace

import pytest

from src.core.dataset.global_inventory import build_global_inventory


def assignment(name, path, role):
    return SimpleNamespace(source_name=name, path=str(path), role=role)


def write_image(root, class_name, filename, content):
    path = root / class_name / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_identifica_mismo_contenido_entre_fuentes_de_entrenamiento(tmp_path):
    original = tmp_path / "original"
    complementario = tmp_path / "complementario"

    write_image(original, "A", "uno.jpg", b"contenido-identico")
    write_image(complementario, "A", "copia.jpg", b"contenido-identico")

    inventory = build_global_inventory(
        [
            assignment("original", original, "train_source"),
            assignment("complementario", complementario, "train_source"),
        ],
        class_names=("A",),
    )

    assert len(inventory.records) == 2
    assert len(inventory.duplicate_groups) == 1
    assert inventory.duplicate_count == 1

    group = next(iter(inventory.duplicate_groups.values()))
    assert {record.source_name for record in group} == {
        "original",
        "complementario",
    }


def test_rechaza_mismo_contenido_asignado_a_clases_distintas(tmp_path):
    original = tmp_path / "original"
    complementario = tmp_path / "complementario"

    write_image(original, "A", "uno.jpg", b"mismo-contenido")
    write_image(complementario, "B", "dos.jpg", b"mismo-contenido")

    with pytest.raises(ValueError, match="clases distintas"):
        build_global_inventory(
            [
                assignment("original", original, "train_source"),
                assignment("complementario", complementario, "train_source"),
            ],
            class_names=("A", "B"),
        )


@pytest.mark.parametrize("evaluation_role", ["final_test", "external_eval"])
def test_rechaza_duplicado_entre_entrenamiento_y_evaluacion(
    tmp_path, evaluation_role
):
    train = tmp_path / "train"
    evaluation = tmp_path / "evaluation"

    write_image(train, "A", "train.jpg", b"contenido-compartido")
    write_image(evaluation, "A", "eval.jpg", b"contenido-compartido")

    with pytest.raises(ValueError, match="fuga de datos"):
        build_global_inventory(
            [
                assignment("original", train, "train_source"),
                assignment("evaluacion", evaluation, evaluation_role),
            ],
            class_names=("A",),
        )


def test_conserva_identidad_independiente_de_fuentes_de_evaluacion(tmp_path):
    final_test = tmp_path / "final_test"
    external_eval = tmp_path / "external_eval"

    write_image(final_test, "A", "test.jpg", b"contenido-evaluacion")
    write_image(external_eval, "A", "externa.jpg", b"contenido-evaluacion")

    inventory = build_global_inventory(
        [
            assignment("original", final_test, "final_test"),
            assignment("complementario", external_eval, "external_eval"),
        ],
        class_names=("A",),
    )

    assert len(inventory.records) == 2
    assert inventory.duplicate_count == 1
    assert {record.role for record in inventory.records} == {
        "final_test",
        "external_eval",
    }


def test_ignora_fuentes_unused(tmp_path):
    unused = tmp_path / "unused"
    write_image(unused, "A", "imagen.jpg", b"contenido")

    inventory = build_global_inventory(
        [assignment("fuente_sin_usar", unused, "unused")],
        class_names=("A",),
    )

    assert inventory.records == ()
    assert inventory.duplicate_groups == {}


def test_no_modifica_los_archivos_originales(tmp_path):
    source = tmp_path / "original"
    image_path = write_image(source, "A", "imagen.jpg", b"contenido-original")

    build_global_inventory(
        [assignment("original", source, "train_source")],
        class_names=("A",),
    )

    assert image_path.exists()
    assert image_path.read_bytes() == b"contenido-original"



def test_rechaza_asignacion_sin_source_name(tmp_path):
    root = tmp_path / "original"
    root.mkdir()

    assignment_without_name = SimpleNamespace(
        path=str(root),
        role="train_source",
    )

    with pytest.raises(ValueError, match="source_name"):
        build_global_inventory(
            [assignment_without_name],
            class_names=("A",),
        )


def test_rechaza_nombre_de_clase_invalido(tmp_path):
    root = tmp_path / "original"
    root.mkdir()

    with pytest.raises(ValueError, match="Nombre de clase inválido"):
        build_global_inventory(
            [assignment("original", root, "train_source")],
            class_names=("../A",),
        )


def test_rechaza_nombres_de_clase_duplicados_sin_distinguir_mayusculas(
    tmp_path,
):
    root = tmp_path / "original"
    root.mkdir()

    with pytest.raises(ValueError, match="únicos"):
        build_global_inventory(
            [assignment("original", root, "train_source")],
            class_names=("A", "a"),
        )


def test_rechaza_dos_fuentes_que_apuntan_al_mismo_directorio(tmp_path):
    root = tmp_path / "original"
    root.mkdir()

    with pytest.raises(ValueError, match="mismo directorio"):
        build_global_inventory(
            [
                assignment("original", root, "train_source"),
                assignment("complementario", root / "." , "train_source"),
            ],
            class_names=("A",),
        )


def test_rechaza_fuentes_con_directorios_anidados(tmp_path):
    parent = tmp_path / "original"
    child = parent / "subfuente"
    child.mkdir(parents=True)

    with pytest.raises(ValueError, match="anidadas"):
        build_global_inventory(
            [
                assignment("original", parent, "train_source"),
                assignment("complementario", child, "train_source"),
            ],
            class_names=("A",),
        )


def test_ignora_fuente_unused_antes_de_validar_su_ruta():
    inventory = build_global_inventory(
        [
            assignment(
                "fuente_sin_usar",
                "/ruta/que/no/existe",
                "unused",
            )
        ],
        class_names=("A",),
    )

    assert inventory.records == ()
    assert inventory.duplicate_groups == {}



def test_copia_canonica_prioriza_el_orden_de_asignaciones(tmp_path):
    original = tmp_path / "original"
    complementario = tmp_path / "complementario"

    original_path = write_image(original, "A", "original.jpg", b"imagen-1")
    complementario_path = write_image(
        complementario, "A", "complementario.jpg", b"imagen-1"
    )

    inventory = build_global_inventory(
        [
            assignment("original", original, "train_source"),
            assignment("complementario", complementario, "train_source"),
        ],
        class_names=("A",),
    )

    # Especificación futura: la primera fuente de entrenamiento gana.
    canonical = inventory.canonical_by_hash

    assert canonical[canonical_path_hash(original_path)].path == original_path
    assert canonical[canonical_path_hash(complementario_path)].path == original_path


def test_audita_copia_excluida_sin_borrar_archivos(tmp_path):
    original = tmp_path / "original"
    complementario = tmp_path / "complementario"

    original_path = write_image(original, "A", "original.jpg", b"imagen-1")
    complementario_path = write_image(
        complementario, "A", "copia.jpg", b"imagen-1"
    )

    inventory = build_global_inventory(
        [
            assignment("original", original, "train_source"),
            assignment("complementario", complementario, "train_source"),
        ],
        class_names=("A",),
    )

    # Especificación futura: la copia no canónica queda auditada.
    excluded = inventory.excluded_training_copies

    assert len(excluded) == 1
    assert excluded[0].path == complementario_path
    assert excluded[0].kept_path == original_path
    assert original_path.exists()
    assert complementario_path.exists()


def test_no_excluye_duplicados_entre_fuentes_de_evaluacion(tmp_path):
    final_test = tmp_path / "final_test"
    external_eval = tmp_path / "external_eval"

    final_path = write_image(final_test, "A", "test.jpg", b"imagen-eval")
    external_path = write_image(
        external_eval, "A", "externa.jpg", b"imagen-eval"
    )

    inventory = build_global_inventory(
        [
            assignment("original", final_test, "final_test"),
            assignment("complementario", external_eval, "external_eval"),
        ],
        class_names=("A",),
    )

    assert len(inventory.records) == 2
    assert inventory.excluded_training_copies == ()
    assert {record.path for record in inventory.records} == {
        final_path,
        external_path,
    }


def canonical_path_hash(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
