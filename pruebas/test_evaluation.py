
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import torch

from src.core.experimentos.schemas import (
    DatasetAssignment,
    ExperimentSpec,
    RunRecord,
)


@pytest.mark.parametrize(
    ("role", "expected_partition", "expected_collection"),
    [
        ("train_source", "test", "tests"),
        ("final_test", "full", "tests"),
        ("external_eval", "full", "external_evaluations"),
    ],
)
def test_evaluation_uses_correct_partition_by_role(
    tmp_path,
    monkeypatch,
    role,
    expected_partition,
    expected_collection,
):
    import src.core.experimentos.evaluation as evaluation

    run_id = "run-test"
    experiment_id = "experiment-test"
    source_name = "original" if role == "train_source" else "dataset-test"

    dataset_path = tmp_path / source_name
    dataset_path.mkdir()
    (tmp_path / "dataset-unused").mkdir()

    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    spec = ExperimentSpec(
        name="Experimento de prueba",
        model_name="mobilenetv2",
        datasets=(
            DatasetAssignment(
                source_name="original",
                path=str(tmp_path / "original"),
                role="train_source",
            ),
            *(
                ()
                if role == "train_source"
                else (
                    DatasetAssignment(
                        source_name=source_name,
                        path=str(dataset_path),
                        role=role,
                    ),
                )
            ),
            DatasetAssignment(
                source_name="dataset-unused",
                path=str(tmp_path / "dataset-unused"),
                role="unused",
            ),
        ),
        parameters={},
    )

    run = RunRecord(
        run_id=run_id,
        experiment_id=experiment_id,
        status="completed",
        checkpoint_path=str(checkpoint_path),
        finished_at=datetime.now(timezone.utc).isoformat(),
    )

    class FakeService:
        def __init__(self, root):
            self.paths = SimpleNamespace(
                root=tmp_path,
                relative_to_root=lambda path: str(path),
            )

        def get_run(self, requested_run_id):
            assert requested_run_id == run_id
            return run

        def get_experiment(self, requested_experiment_id):
            assert requested_experiment_id == experiment_id
            return spec

        def _build_config(self, spec):
            return SimpleNamespace(
                model=SimpleNamespace(architecture="mobilenetv2"),
                class_names=("A", "B"),
                device="cpu",
                batch_size=2,
                num_workers=0,
                pin_memory=False,
            )

    model_state_dict = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 224 * 224, 2),
    ).state_dict()

    checkpoint = {
        "metadata": {
            "run_id": run_id,
            "fingerprints": {
                assignment.source_name: "fingerprint"
                for assignment in spec.datasets
                if assignment.role != "unused"
            },
        },
        "experiment_config": {},
        "class_names": ["A", "B"],
        "num_classes": 2,
        "class_to_idx": {"A": 0, "B": 1},
        "architecture": "mobilenetv2",
        "model": "mobilenetv2",
        "model_state_dict": model_state_dict,
    }

    fake_dataset = [(torch.zeros(3, 224, 224), 0)]

    prepared = {
        assignment.source_name: SimpleNamespace(
            name=assignment.source_name,
            dataset=fake_dataset,
            split=None,
            split_path=None,
        )
        for assignment in spec.datasets
    }

    calls = {"subset": [], "full": []}

    monkeypatch.setattr(
        evaluation,
        "ExperimentService",
        FakeService,
    )
    monkeypatch.setattr(
        evaluation.torch,
        "load",
        lambda *args, **kwargs: checkpoint,
    )
    monkeypatch.setattr(
        evaluation,
        "_seleccionar_dispositivo",
        lambda device: torch.device("cpu"),
    )
    monkeypatch.setattr(
        evaluation,
        "_configuracion_del_checkpoint",
        lambda checkpoint, spec, service: service._build_config(spec),
    )
    monkeypatch.setattr(
        evaluation,
        "_validar_checkpoint",
        lambda checkpoint, run, config: (("A", "B"), 2),
    )
    def fake_prepare_sources(*args, **kwargs):
        assignments = kwargs["source_assignments"]

        assert all(
            assignment.role != "unused"
            for assignment in assignments
        )

        return prepared

    monkeypatch.setattr(
        evaluation,
        "prepare_sources",
        fake_prepare_sources,
    )

    monkeypatch.setattr(
        evaluation,
        "dataset_fingerprint",
        lambda dataset: "fingerprint",
    )
    monkeypatch.setattr(
        evaluation,
        "crear_modelo",
        lambda *args, **kwargs: torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 224 * 224, 2),
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "build_eval_transform",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        evaluation,
        "subset_for",
        lambda source, partition, transform: (
            calls["subset"].append((source.name, partition))
            or fake_dataset
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "full_dataset_for",
        lambda source, transform: (
            calls["full"].append(source.name)
            or fake_dataset
        ),
    )

    result = evaluation.evaluar_ejecucion(
        tmp_path,
        run_id,
    )

    assert calls["subset"] == [("original", "test")]

    assert calls["full"] == (
        []
        if role == "train_source"
        else [source_name]
    )

    evaluated = result[expected_collection][source_name]

    assert evaluated["role"] == role
    assert evaluated["partition"] == expected_partition
    assert evaluated["num_samples"] == len(fake_dataset)

    assert "dataset-unused" not in result["tests"]
    assert "dataset-unused" not in result["external_evaluations"]
