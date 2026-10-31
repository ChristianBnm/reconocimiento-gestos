
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
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
