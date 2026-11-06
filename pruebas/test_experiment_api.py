
import pytest

from fastapi.testclient import TestClient
from src.api.routes import runs as runs_routes
from src.api.main import app
from src.api import dependencies
from src.api.routes import experiments as experiments_routes

from src.core.experimentos.schemas import (
    DatasetAssignment,
    ExperimentSpec,
)
from src.core.experimentos.service import ExperimentService


@pytest.fixture
def client(tmp_path, monkeypatch):
    service = ExperimentService(root=tmp_path)

    monkeypatch.setattr(
        experiments_routes,
        "experiment_service",
        service,
    )
    monkeypatch.setattr(
        runs_routes,
        "experiment_service",
        service,
    )

    return TestClient(app), service


@pytest.fixture
def experiment_spec():
    return ExperimentSpec(
        experiment_id="MNET-TEST-001",
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


@pytest.fixture(autouse=True)
def reset_run_state():
    if dependencies.training_lock.locked():
        dependencies.training_lock.release()

    with dependencies.run_cancel_events_lock:
        dependencies.run_cancel_events.clear()

    yield

    if dependencies.training_lock.locked():
        dependencies.training_lock.release()

    with dependencies.run_cancel_events_lock:
        dependencies.run_cancel_events.clear()


def test_delete_experiment_returns_204(client, experiment_spec):
    http, service = client
    service.registry.save_experiment(experiment_spec)

    response = http.delete(
        f"/api/experiments/{experiment_spec.experiment_id}"
    )

    assert response.status_code == 204
    assert response.content == b""
    assert (
        experiment_spec.experiment_id
        not in service.list_experiments()
    )


def test_delete_nonexistent_experiment_returns_404(client):
    http, _ = client


    response = http.delete("/api/experiments/NO-EXISTE")

    assert response.status_code == 404


def test_delete_experiment_with_runs_returns_409(
    client,
    experiment_spec,
):
    http, service = client
    service.registry.save_experiment(experiment_spec)

    service.create_run(
        experiment_spec.experiment_id,
        run_id="run-test-001",
    )

    response = http.delete(
        f"/api/experiments/{experiment_spec.experiment_id}"
    )

    assert response.status_code == 409
    assert (
        experiment_spec.experiment_id
        in service.list_experiments()
    )


def test_start_run_rejects_concurrent_execution(
    client,
    experiment_spec,
    monkeypatch,
):
    http, service = client
    service.registry.save_experiment(experiment_spec)

    submitted_tasks = []

    class FakeExecutor:
        def submit(self, fn, *args, **kwargs):
            submitted_tasks.append((fn, args, kwargs))

    monkeypatch.setattr(
        runs_routes,
        "executor",
        FakeExecutor(),
    )

    response_first = http.post(
        f"/api/experiments/{experiment_spec.experiment_id}/runs"
    )

    response_second = http.post(
        f"/api/experiments/{experiment_spec.experiment_id}/runs"
    )

    assert response_first.status_code == 202
    assert response_second.status_code == 409
    assert len(submitted_tasks) == 1

    run_id = response_first.json()["run_id"]
    run = service.get_run(run_id)

    assert run.status == "pending"



def test_start_run_releases_lock_when_submit_fails(
    client,
    experiment_spec,
    monkeypatch,
):
    http, service = client
    service.registry.save_experiment(experiment_spec)

    class FailingExecutor:
        def submit(self, fn, *args, **kwargs):
            raise RuntimeError("Error simulado del executor")

    monkeypatch.setattr(
        runs_routes,
        "executor",
        FailingExecutor(),
    )

    response = http.post(
        f"/api/experiments/{experiment_spec.experiment_id}/runs"
    )

    assert response.status_code == 500
    assert not dependencies.training_lock.locked()

    runs = service.list_runs(experiment_spec.experiment_id)
    assert len(runs) == 1
    assert runs[0].status == "failed"


def test_run_experiment_task_releases_lock_on_exception(
    monkeypatch,
):
    def failing_run_experiment(
        experiment_id,
        run_id=None,
        cancel_event=None,
    ):
        raise RuntimeError("Error simulado del entrenamiento")

    monkeypatch.setattr(
        dependencies,
        "experiment_service",
        type(
            "FakeService",
            (),
            {"run_experiment": staticmethod(failing_run_experiment)},
        )(),
    )

    dependencies.training_lock.acquire()

    try:
        with pytest.raises(RuntimeError, match="Error simulado"):
            dependencies.run_experiment_task(
                "MNET-TEST-001",
                "run-test-error",
            )

        assert not dependencies.training_lock.locked()
    finally:
        # Protección adicional para evitar contaminar otras pruebas.
        if dependencies.training_lock.locked():
            dependencies.training_lock.release()



def test_cancel_run_keeps_lock_until_task_finishes(
    client,
    experiment_spec,
    monkeypatch,
):
    http, service = client
    service.registry.save_experiment(experiment_spec)

    submitted_tasks = []

    class FakeExecutor:
        def submit(self, fn, *args, **kwargs):
            submitted_tasks.append((fn, args, kwargs))

    monkeypatch.setattr(
        runs_routes,
        "executor",
        FakeExecutor(),
    )

    response = http.post(
        f"/api/experiments/{experiment_spec.experiment_id}/runs"
    )

    assert response.status_code == 202
    run_id = response.json()["run_id"]

    cancel_response = http.post(
        f"/api/runs/{run_id}/cancel"
    )

    assert cancel_response.status_code == 202
    assert cancel_response.json()["status"] == "cancellation_requested"

    # Solicitar la cancelación no debe liberar el bloqueo.
    assert dependencies.training_lock.locked()

    second_response = http.post(
        f"/api/experiments/{experiment_spec.experiment_id}/runs"
    )

    assert second_response.status_code == 409

    # Simular el final de la tarea en segundo plano.
    with dependencies.run_cancel_events_lock:
        dependencies.run_cancel_events.pop(run_id, None)

    dependencies.training_lock.release()

    third_response = http.post(
        f"/api/experiments/{experiment_spec.experiment_id}/runs"
    )

    assert third_response.status_code == 202
