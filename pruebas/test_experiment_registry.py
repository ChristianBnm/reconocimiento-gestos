
from datetime import datetime, timezone

import json
import pytest

from src.core.experimentos.registry import ExperimentRegistry
from src.core.experimentos.schemas import (
    DatasetAssignment,
    ExperimentSpec,
    RunRecord,
    utc_now,
)


@pytest.fixture
def registry(tmp_path):
    return ExperimentRegistry(tmp_path / "registry")


@pytest.fixture
def experiment():
    return ExperimentSpec(
        name="experimento_prueba",
        model_name="mobilenetv2",
        datasets=(
            DatasetAssignment(
                source_name="original",
                path="data/original",
                role="train_source",
            ),
        ),
        parameters={"epochs": 3, "seed": 42},
    )


def test_guarda_y_recupera_experimento(registry, experiment):
    registry.save_experiment(experiment)

    loaded = registry.load_experiment(experiment.experiment_id)

    assert loaded == experiment
    assert registry.list_experiments() == [experiment.experiment_id]


def test_id_del_experimento_se_conserva(registry, experiment):
    registry.save_experiment(experiment)

    loaded = registry.load_experiment(experiment.experiment_id)

    assert loaded.experiment_id == experiment.experiment_id


def test_actualiza_experimento_sin_cambiar_id(registry, experiment):
    registry.save_experiment(experiment)

    updated = ExperimentSpec(
        experiment_id=experiment.experiment_id,
        name="experimento_actualizado",
        model_name=experiment.model_name,
        datasets=experiment.datasets,
        parameters={"epochs": 5},
    )
    registry.save_experiment(updated)

    assert registry.load_experiment(
        experiment.experiment_id
    ) == updated


def test_elimina_experimento_sin_ejecuciones(registry, experiment):
    registry.save_experiment(experiment)
    registry.delete_experiment(experiment.experiment_id)

    assert registry.list_experiments() == []

    with pytest.raises(FileNotFoundError):
        registry.load_experiment(experiment.experiment_id)


def test_no_elimina_experimento_con_ejecuciones(registry, experiment):
    registry.save_experiment(experiment)
    registry.save_run(
        RunRecord(
            run_id="run-001",
            experiment_id=experiment.experiment_id,
            status="pending",
        )
    )

    with pytest.raises(ValueError, match="ejecuciones"):
        registry.delete_experiment(experiment.experiment_id)


def test_guarda_y_recupera_ejecucion(registry, experiment):
    registry.save_experiment(experiment)
    run = RunRecord(
        run_id="run-001",
        experiment_id=experiment.experiment_id,
        status="pending",
    )

    registry.save_run(run)

    assert registry.load_run("run-001") == run


def test_rechaza_ejecucion_de_experimento_inexistente(registry):
    run = RunRecord(
        run_id="run-001",
        experiment_id="inexistente",
        status="pending",
    )

    with pytest.raises(FileNotFoundError):
        registry.save_run(run)


def test_rechaza_run_id_duplicado(registry, experiment):
    registry.save_experiment(experiment)
    run = RunRecord(
        run_id="run-001",
        experiment_id=experiment.experiment_id,
        status="pending",
    )
    registry.save_run(run)

    with pytest.raises(FileExistsError):
        registry.save_run(run)


def test_transicion_pending_a_running(registry, experiment):
    registry.save_experiment(experiment)
    registry.save_run(
        RunRecord(
            run_id="run-001",
            experiment_id=experiment.experiment_id,
            status="pending",
        )
    )

    updated = registry.update_run_status("run-001", "running")

    assert updated.status == "running"
    assert updated.finished_at is None


def test_transicion_running_a_completed(registry, experiment):
    registry.save_experiment(experiment)
    registry.save_run(
        RunRecord(
            run_id="run-001",
            experiment_id=experiment.experiment_id,
            status="pending",
        )
    )
    registry.update_run_status("run-001", "running")

    updated = registry.update_run_status(
        "run-001",
        "completed",
        checkpoint_path="checkpoints/best.pth",
    )

    assert updated.status == "completed"
    assert updated.checkpoint_path == "checkpoints/best.pth"
    assert updated.finished_at is not None
    datetime.fromisoformat(updated.finished_at)


def test_transicion_running_a_failed(registry, experiment):
    registry.save_experiment(experiment)
    registry.save_run(
        RunRecord(
            run_id="run-001",
            experiment_id=experiment.experiment_id,
            status="pending",
        )
    )
    registry.update_run_status("run-001", "running")

    updated = registry.update_run_status(
        "run-001",
        "failed",
        error_message="Error de prueba",
    )

    assert updated.status == "failed"
    assert updated.error_message == "Error de prueba"
    assert updated.finished_at is not None


def test_rechaza_transicion_directa_pending_a_completed(
    registry, experiment
):
    registry.save_experiment(experiment)
    registry.save_run(
        RunRecord(
            run_id="run-001",
            experiment_id=experiment.experiment_id,
            status="pending",
        )
    )

    with pytest.raises(ValueError, match="Transición"):
        registry.update_run_status(
            "run-001",
            "completed",
            checkpoint_path="checkpoints/best.pth",
        )


def test_no_permite_actualizar_ejecucion_finalizada(
    registry, experiment
):
    registry.save_experiment(experiment)
    registry.save_run(
        RunRecord(
            run_id="run-001",
            experiment_id=experiment.experiment_id,
            status="pending",
        )
    )
    registry.update_run_status("run-001", "cancelled")

    with pytest.raises(ValueError, match="ya finalizó"):
        registry.update_run_status("run-001", "running")


def test_lista_ejecuciones_por_experimento(registry):
    first = ExperimentSpec(
        name="primero",
        model_name="mobilenetv2",
        datasets=(
            DatasetAssignment("original", "data/original", "train_source"),
        ),
    )
    second = ExperimentSpec(
        name="segundo",
        model_name="mobilenetv2",
        datasets=(
            DatasetAssignment("original", "data/original", "train_source"),
        ),
    )
    registry.save_experiment(first)
    registry.save_experiment(second)

    registry.save_run(
        RunRecord("run-001", first.experiment_id, "pending")
    )
    registry.save_run(
        RunRecord("run-002", second.experiment_id, "pending")
    )

    assert [
        run.run_id for run in registry.list_runs(first.experiment_id)
    ] == ["run-001"]


def test_detecta_json_corrupto(registry):
    path = registry.experiments_dir / "exp-001.json"
    path.write_text("{json invalido", encoding="utf-8")

    with pytest.raises(ValueError, match="corrupto"):
        registry.load_experiment("exp-001")


def test_rechaza_ids_con_rutas(registry, experiment):
    with pytest.raises(ValueError):
        registry._validate_id("../exp-001", "experiment_id")


def test_archivo_persistido_es_json_legible(registry, experiment):
    registry.save_experiment(experiment)

    path = (
        registry.experiments_dir
        / f"{experiment.experiment_id}.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == experiment.experiment_id
    assert data["name"] == "experimento_prueba"


def test_run_con_fecha_invalida_se_rechaza(experiment):
    with pytest.raises(ValueError, match="ISO 8601"):
        RunRecord(
            run_id="run-001",
            experiment_id=experiment.experiment_id,
            status="pending",
            created_at="fecha-invalida",
        )
