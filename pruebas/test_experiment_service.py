from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.experimentos.schemas import (
    DatasetAssignment,
    ExperimentSpec,
)
from src.core.experimentos.service import ExperimentService


def make_spec(dataset_path: Path) -> ExperimentSpec:
    return ExperimentSpec(
        name="Experimento de prueba",
        model_name="mobilenetv2",
        datasets=(
            DatasetAssignment(
                source_name="original",
                path=str(dataset_path),
                role="train_source",
            ),
        ),
        parameters={},
        description="Prueba automatizada",
    )


def test_create_and_load_experiment(tmp_path):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()

    service = ExperimentService(tmp_path)
    spec = make_spec(dataset_path)

    created = service.create_experiment(spec)
    loaded = service.get_experiment(created.experiment_id)

    assert loaded.experiment_id == spec.experiment_id
    assert loaded.name == spec.name
    assert loaded.model_name == spec.model_name
    assert service.list_experiments()


def test_validate_experiment_rejects_missing_dataset(tmp_path):
    service = ExperimentService(tmp_path)
    spec = make_spec(tmp_path / "dataset_inexistente")

    with pytest.raises(FileNotFoundError):
        service.validate_experiment(spec)


def test_validate_experiment_rejects_unknown_parameter(tmp_path):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()

    spec = ExperimentSpec(
        name="Experimento inválido",
        model_name="mobilenetv2",
        datasets=(
            DatasetAssignment(
                source_name="original",
                path=str(dataset_path),
                role="train_source",
            ),
        ),
        parameters={"parametro_que_no_existe": 123},
    )

    service = ExperimentService(tmp_path)

    with pytest.raises(ValueError, match="Parámetros de configuración desconocidos"):
        service.validate_experiment(spec)


def test_run_experiment_marks_completed_with_checkpoint(tmp_path):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()

    service = ExperimentService(tmp_path)
    spec = service.create_experiment(make_spec(dataset_path))

    def fake_fit(self):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.write_bytes(b"checkpoint de prueba")

    with (
        patch(
            "src.core.experimentos.service.ExperimentService.validate_experiment"
        ),
        patch(
            "src.core.experimentos.service.prepare_sources",
            side_effect=RuntimeError("fallo de preparación"),
        ),
    ):
        with pytest.raises(RuntimeError, match="fallo de preparación"):
            service.run_experiment(spec.experiment_id)

    runs = service.list_runs(spec.experiment_id)

    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert "fallo de preparación" in runs[0].error_message


def test_run_experiment_records_training_error(tmp_path):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()

    service = ExperimentService(tmp_path)
    spec = service.create_experiment(make_spec(dataset_path))

    with (
        patch(
            "src.core.experimentos.service.ExperimentService.validate_experiment"
        ),
        patch(
            "src.core.experimentos.service.prepare_sources",
            side_effect=RuntimeError("fallo simulado"),
        ),
    ):
        with pytest.raises(RuntimeError, match="fallo simulado"):
            service.run_experiment(spec.experiment_id)

    run = service.list_runs(spec.experiment_id)[0]

    assert run.status == "failed"
    assert run.error_message == "RuntimeError: fallo simulado"
    assert run.finished_at is not None


def test_run_experiment_marks_completed_with_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    import src.core.experimentos.service as service_module

    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()

    service = ExperimentService(tmp_path)
    spec = service.create_experiment(make_spec(dataset_path))

    # Simula una fuente preparada sin cargar imágenes reales.
    fake_dataset = [0, 1]

    monkeypatch.setattr(
        service_module,
        "prepare_sources",
        lambda *args, **kwargs: {
            "original": SimpleNamespace(
                name="original",
                dataset=fake_dataset,
                split_path=tmp_path / "split.json",
            )
        },
    )
    monkeypatch.setattr(
        service_module,
        "dataset_fingerprint",
        lambda dataset: "fingerprint-de-prueba",
    )
    monkeypatch.setattr(
        service_module,
        "subset_for",
        lambda *args, **kwargs: fake_dataset,
    )
    monkeypatch.setattr(
        service_module,
        "build_train_transform",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        service_module,
        "build_eval_transform",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        service_module,
        "crear_modelo",
        lambda *args, **kwargs: torch.nn.Linear(2, 2),
    )
    monkeypatch.setattr(
        service_module,
        "adaptar_normalizacion_a_modelo",
        lambda config, model: config,
    )
    monkeypatch.setattr(
        service_module,
        "descongelar_modelo",
        lambda model: model,
    )
    monkeypatch.setattr(
        service_module,
        "establecer_semilla",
        lambda seed: None,
    )

    # Evita crear DataLoaders reales.
    monkeypatch.setattr(
        ExperimentService,
        "_make_loader",
        staticmethod(lambda *args, **kwargs: object()),
    )

    class FakeTrainer:
        def __init__(
            self,
            model,
            train_loader,
            val_loader,
            config,
            checkpoint_path,
            history_path,
            metadata=None,
        ):
            self.checkpoint_path = checkpoint_path

        def fit(self):
            self.checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            self.checkpoint_path.write_bytes(b"checkpoint de prueba")

    monkeypatch.setattr(service_module, "Trainer", FakeTrainer)

    monkeypatch.setattr(
    service_module,
    "congelar_backbone",
    lambda model: model,
    )

    run = service.run_experiment(spec.experiment_id)

    assert run.status == "completed"
    assert run.experiment_id == spec.experiment_id
    assert run.checkpoint_path is not None
    assert (service.paths.root / run.checkpoint_path).is_file()
    assert run.finished_at is not None

    persisted_run = service.get_run(run.run_id)
    assert persisted_run.status == "completed"
    assert persisted_run.checkpoint_path == run.checkpoint_path


def test_run_experiment_preserves_original_error_if_failure_recording_fails(
    tmp_path, monkeypatch
):
    import src.core.experimentos.service as service_module

    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()

    service = ExperimentService(tmp_path)
    spec = service.create_experiment(make_spec(dataset_path))

    original_error = RuntimeError("fallo original del entrenamiento")

    monkeypatch.setattr(
        service_module.ExperimentService,
        "validate_experiment",
        lambda self, spec: None,
    )
    monkeypatch.setattr(
        service_module,
        "prepare_sources",
        lambda *args, **kwargs: (_ for _ in ()).throw(original_error),
    )
    
    original_update = service.registry.update_run_status
    def update_status_with_failure_on_error(
        run_id, status, **kwargs
    ):
        if status == "failed":
            raise OSError("fallo al persistir el estado")
        return original_update(run_id, status, **kwargs)

    monkeypatch.setattr(
        service.registry,
        "update_run_status",
        update_status_with_failure_on_error,
    )

    with pytest.raises(RuntimeError, match="fallo original del entrenamiento"):
        service.run_experiment(spec.experiment_id)
