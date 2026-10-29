
"""Persistencia local de experimentos y ejecuciones."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.core.experimentos.schemas import (
    ExperimentSpec,
    RunRecord,
    RUN_STATUSES,
    TERMINAL_RUN_STATUSES,
    utc_now,
)


class ExperimentRegistry:
    """Almacena definiciones de experimentos y sus ejecuciones."""

    _TRANSITIONS = {
        "pending": {"running", "failed", "cancelled"},
        "running": {"completed", "failed", "cancelled"},
        "completed": set(),
        "failed": set(),
        "cancelled": set(),
    }

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.experiments_dir = self.root / "experiments"
        self.runs_dir = self.root / "runs"

        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_id(value: str, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} no puede estar vacío.")
        if value in {".", ".."} or Path(value).name != value:
            raise ValueError(f"{field} no es válido: {value!r}.")
        if any(char in value for char in ("/", "\\", "\0")):
            raise ValueError(f"{field} no es válido: {value!r}.")
        return value

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        """Reemplaza el JSON de forma atómica."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(
                    data,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())

            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"El archivo JSON está corrupto: {path}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(f"Se esperaba un objeto JSON en {path}.")
        return data

    def save_experiment(self, experiment: ExperimentSpec) -> None:
        if not isinstance(experiment, ExperimentSpec):
            raise TypeError(
                "experiment debe ser una instancia de ExperimentSpec."
            )

        experiment_id = self._validate_id(
            experiment.experiment_id,
            "experiment_id",
        )
        path = self.experiments_dir / f"{experiment_id}.json"

        # Si existe, solo se permite actualizar la misma definición por ID.
        if path.exists():
            stored = ExperimentSpec.from_dict(self._read_json(path))
            if stored.experiment_id != experiment.experiment_id:
                raise ValueError("El ID del experimento no coincide.")

        self._write_json(path, experiment.to_dict())

    def load_experiment(self, experiment_id: str) -> ExperimentSpec:
        experiment_id = self._validate_id(experiment_id, "experiment_id")
        path = self.experiments_dir / f"{experiment_id}.json"

        if not path.is_file():
            raise FileNotFoundError(
                f"No existe el experimento {experiment_id!r}."
            )

        experiment = ExperimentSpec.from_dict(self._read_json(path))
        if experiment.experiment_id != experiment_id:
            raise ValueError(
                f"El ID del archivo no coincide con el experimento: {path}"
            )
        return experiment

    def list_experiments(self) -> list[str]:
        return sorted(
            path.stem for path in self.experiments_dir.glob("*.json")
        )

    def delete_experiment(self, experiment_id: str) -> None:
        experiment_id = self._validate_id(experiment_id, "experiment_id")
        path = self.experiments_dir / f"{experiment_id}.json"

        if not path.is_file():
            raise FileNotFoundError(
                f"No existe el experimento {experiment_id!r}."
            )

        if self.list_runs(experiment_id):
            raise ValueError(
                "No se puede eliminar un experimento con ejecuciones "
                "asociadas. Conservá el historial."
            )

        path.unlink()

    def save_run(self, run: RunRecord) -> None:
        """Crea una ejecución nueva; no permite sobrescribir su historial."""
        if not isinstance(run, RunRecord):
            raise TypeError("run debe ser una instancia de RunRecord.")

        run_id = self._validate_id(run.run_id, "run_id")
        experiment_id = self._validate_id(
            run.experiment_id,
            "experiment_id",
        )

        # Comprueba existencia y validez de la definición asociada.
        self.load_experiment(experiment_id)

        path = self.runs_dir / f"{run_id}.json"
        if path.exists():
            raise FileExistsError(
                f"La ejecución {run_id!r} ya existe."
            )

        self._write_json(path, run.to_dict())

    def load_run(self, run_id: str) -> RunRecord:
        run_id = self._validate_id(run_id, "run_id")
        path = self.runs_dir / f"{run_id}.json"

        if not path.is_file():
            raise FileNotFoundError(f"No existe la ejecución {run_id!r}.")

        run = RunRecord.from_dict(self._read_json(path))
        if run.run_id != run_id:
            raise ValueError(
                f"El ID del archivo no coincide con la ejecución: {path}"
            )
        return run

    def update_run_status(
        self,
        run_id: str,
        status: str,
        *,
        checkpoint_path: str | None = None,
        error_message: str | None = None,
    ) -> RunRecord:
        """Actualiza una ejecución validando la transición de estado."""
        if status not in RUN_STATUSES:
            raise ValueError(f"Estado no válido: {status!r}.")

        current = self.load_run(run_id)

        if current.status in TERMINAL_RUN_STATUSES:
            raise ValueError(
                f"La ejecución ya finalizó con estado {current.status!r}."
            )

        allowed = self._TRANSITIONS[current.status]
        if status not in allowed:
            raise ValueError(
                f"Transición no permitida: {current.status!r} -> {status!r}."
            )

        changes: dict[str, Any] = {"status": status}

        if checkpoint_path is not None:
            changes["checkpoint_path"] = checkpoint_path
        if error_message is not None:
            changes["error_message"] = error_message

        if status in TERMINAL_RUN_STATUSES:
            changes["finished_at"] = utc_now()

        updated = replace(current, **changes)

        self._write_json(
            self.runs_dir / f"{updated.run_id}.json",
            updated.to_dict(),
        )
        return updated

    def list_runs(
        self,
        experiment_id: str | None = None,
    ) -> list[RunRecord]:
        """Lista ejecuciones, opcionalmente filtradas por experimento."""
        if experiment_id is not None:
            experiment_id = self._validate_id(
                experiment_id,
                "experiment_id",
            )

        runs: list[RunRecord] = []
        for path in sorted(self.runs_dir.glob("*.json")):
            run = RunRecord.from_dict(self._read_json(path))
            if experiment_id is None or run.experiment_id == experiment_id:
                runs.append(run)

        return runs
