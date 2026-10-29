
"""Pruebas para los esquemas de experimentos y ejecuciones."""

from copy import deepcopy
from datetime import datetime

import pytest

from src.core.experimentos.schemas import (
    DatasetAssignment,
    ExperimentSpec,
    RunRecord,
    utc_now,
)


# ============================================================
# DatasetAssignment
# ============================================================


@pytest.mark.parametrize(
    "role",
    ["train_source", "final_test", "external_eval", "unused"],
)
def test_dataset_assignment_acepta_roles_validos(role):
    dataset = DatasetAssignment(
        source_name="dataset_original",
        path="data/original",
        role=role,
    )

    assert dataset.source_name == "dataset_original"
    assert dataset.path == "data/original"
    assert dataset.role == role


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_name", ""),
        ("source_name", "   "),
        ("path", ""),
        ("path", "   "),
        ("role", "invalid_role"),
    ],
)
def test_dataset_assignment_rechaza_datos_invalidos(field, value):
    data = {
        "source_name": "dataset_original",
        "path": "data/original",
        "role": "train_source",
    }
    data[field] = value

    with pytest.raises(ValueError):
        DatasetAssignment(**data)


def test_dataset_assignment_round_trip():
    original = DatasetAssignment(
        source_name="dataset_original",
        path="data/original",
        role="train_source",
    )

    restored = DatasetAssignment.from_dict(original.to_dict())

    assert restored == original


def test_dataset_assignment_from_dict_rechaza_datos_no_dict():
    with pytest.raises(ValueError):
        DatasetAssignment.from_dict(["dataset_original", "data/original"])


# ============================================================
# ExperimentSpec
# ============================================================


@pytest.fixture
def dataset_entrenamiento():
    return DatasetAssignment(
        source_name="original",
        path="data/original",
        role="train_source",
    )


@pytest.fixture
def experiment_spec(dataset_entrenamiento):
    return ExperimentSpec(
        name="experimento_prueba",
        model_name="mobilenetv2",
        datasets=(dataset_entrenamiento,),
        parameters={"epochs": 3, "seed": 42},
        description="Experimento de prueba.",
    )


def test_experiment_spec_acepta_configuracion_valida(experiment_spec):
    assert experiment_spec.name == "experimento_prueba"
    assert experiment_spec.model_name == "mobilenetv2"
    assert len(experiment_spec.datasets) == 1
    assert experiment_spec.parameters == {"epochs": 3, "seed": 42}
    assert experiment_spec.description == "Experimento de prueba."
    assert experiment_spec.experiment_id


def test_experiment_spec_generacion_automatica_de_id(dataset_entrenamiento):
    first = ExperimentSpec(
        name="experimento_1",
        model_name="mobilenetv2",
        datasets=(dataset_entrenamiento,),
    )
    second = ExperimentSpec(
        name="experimento_2",
        model_name="mobilenetv2",
        datasets=(dataset_entrenamiento,),
    )

    assert first.experiment_id
    assert second.experiment_id
    assert first.experiment_id != second.experiment_id


def test_experiment_spec_round_trip(experiment_spec):
    restored = ExperimentSpec.from_dict(experiment_spec.to_dict())

    assert restored == experiment_spec
    assert restored.experiment_id == experiment_spec.experiment_id
    assert restored.datasets == experiment_spec.datasets
    assert restored.parameters == experiment_spec.parameters


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", ""),
        ("name", "   "),
        ("model_name", ""),
        ("model_name", "   "),
        ("experiment_id", ""),
        ("experiment_id", "   "),
        ("description", None),
        ("datasets", ()),
        ("datasets", []),
        ("datasets", ("no_es_un_dataset",)),
        ("parameters", None),
        ("parameters", []),
    ],
)
def test_experiment_spec_rechaza_configuracion_invalida(
    dataset_entrenamiento, field, value
):
    data = {
        "name": "experimento_prueba",
        "model_name": "mobilenetv2",
        "datasets": (dataset_entrenamiento,),
        "parameters": {},
        "description": "",
        "experiment_id": "exp-001",
    }
    data[field] = value

    with pytest.raises(ValueError):
        ExperimentSpec(**data)


def test_experiment_spec_rechaza_datasets_sin_fuente_de_entrenamiento():
    dataset = DatasetAssignment(
        source_name="test_final",
        path="data/test",
        role="final_test",
    )

    with pytest.raises(ValueError, match="train_source"):
        ExperimentSpec(
            name="experimento_prueba",
            model_name="mobilenetv2",
            datasets=(dataset,),
        )


def test_experiment_spec_rechaza_nombres_de_dataset_duplicados():
    first = DatasetAssignment(
        source_name="Original",
        path="data/original",
        role="train_source",
    )
    second = DatasetAssignment(
        source_name="original",
        path="data/otra_fuente",
        role="external_eval",
    )

    with pytest.raises(ValueError, match="misma fuente"):
        ExperimentSpec(
            name="experimento_prueba",
            model_name="mobilenetv2",
            datasets=(first, second),
        )


def test_experiment_spec_copia_parametros_al_construirse(
    dataset_entrenamiento,
):
    parameters = {
        "epochs": 3,
        "config": {"batch_size": 16},
    }

    experiment = ExperimentSpec(
        name="experimento_prueba",
        model_name="mobilenetv2",
        datasets=(dataset_entrenamiento,),
        parameters=parameters,
    )

    parameters["epochs"] = 99
    parameters["config"]["batch_size"] = 1

    assert experiment.parameters["epochs"] == 3
    assert experiment.parameters["config"]["batch_size"] == 16


def test_experiment_spec_to_dict_no_expone_referencias_mutables(
    experiment_spec,
):
    data = experiment_spec.to_dict()
    data["parameters"]["epochs"] = 999

    assert experiment_spec.parameters["epochs"] == 3


def test_experiment_spec_from_dict_rechaza_datos_no_dict():
    with pytest.raises(ValueError):
        ExperimentSpec.from_dict([])


# ============================================================
# RunRecord
# ============================================================


def test_run_record_acepta_estado_pending():
    record = RunRecord(
        run_id="run-001",
        experiment_id="exp-001",
        status="pending",
    )

    assert record.status == "pending"
    assert record.checkpoint_path is None
    assert record.error_message is None
    assert datetime.fromisoformat(record.created_at)


@pytest.mark.parametrize("status", ["pending", "running", "cancelled"])
def test_run_record_acepta_estados_sin_checkpoint(status):
    record = RunRecord(
        run_id="run-001",
        experiment_id="exp-001",
        status=status,
        finished_at=utc_now() if status == "cancelled" else None,
    )

    assert record.status == status
    assert record.checkpoint_path is None


def test_run_record_acepta_estado_completado_y_round_trip():
    record = RunRecord(
        run_id="run-001",
        experiment_id="exp-001",
        status="completed",
        checkpoint_path="checkpoints/model.pth",
        created_at=utc_now(),
        finished_at=utc_now(),
    )

    restored = RunRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.checkpoint_path == "checkpoints/model.pth"
    assert restored.finished_at is not None


def test_run_record_acepta_estado_failed_sin_checkpoint():
    record = RunRecord(
        run_id="run-001",
        experiment_id="exp-001",
        status="failed",
        error_message="Error de prueba",
        finished_at=utc_now(),
    )

    assert record.status == "failed"
    assert record.checkpoint_path is None
    assert record.error_message == "Error de prueba"
    assert record.finished_at is not None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", ""),
        ("run_id", "   "),
        ("experiment_id", ""),
        ("experiment_id", "   "),
        ("status", "unknown"),
        ("checkpoint_path", ""),
        ("error_message", 123),
        ("created_at", "fecha-invalida"),
    ],
)
def test_run_record_rechaza_datos_invalidos(field, value):
    data = {
        "run_id": "run-001",
        "experiment_id": "exp-001",
        "status": "pending",
    }
    data[field] = value

    with pytest.raises(ValueError):
        RunRecord(**data)


def test_run_record_rechaza_completado_sin_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        RunRecord(
            run_id="run-001",
            experiment_id="exp-001",
            status="completed",
            finished_at=utc_now(),
        )


def test_run_record_rechaza_failed_sin_mensaje_de_error():
    with pytest.raises(ValueError, match="indicar el error"):
        RunRecord(
            run_id="run-001",
            experiment_id="exp-001",
            status="failed",
            finished_at=utc_now(),
        )


def test_run_record_rechaza_estado_terminal_sin_finished_at():
    with pytest.raises(ValueError, match="finished_at"):
        RunRecord(
            run_id="run-001",
            experiment_id="exp-001",
            status="cancelled",
        )


def test_run_record_rechaza_finished_at_en_estado_no_terminal():
    with pytest.raises(ValueError, match="no finalizada"):
        RunRecord(
            run_id="run-001",
            experiment_id="exp-001",
            status="running",
            finished_at=utc_now(),
        )


def test_run_record_from_dict_rechaza_datos_no_dict():
    with pytest.raises(ValueError):
        RunRecord.from_dict("no es un diccionario")
