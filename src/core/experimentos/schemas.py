
"""Modelos de datos para la definición y ejecución de experimentos."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

DatasetRole = Literal[
    "train_source",
    "final_test",
    "external_eval",
    "unused",
]

RunStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
]

DATASET_ROLES = {
    "train_source",
    "final_test",
    "external_eval",
    "unused",
}

RUN_STATUSES = {
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
}

TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}


def utc_now() -> str:
    """Devuelve la fecha y hora actual en UTC, en formato ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def _require_nonempty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} debe ser un texto no vacío.")


@dataclass(frozen=True)
class DatasetAssignment:
    source_name: str
    path: str
    role: DatasetRole

    def __post_init__(self) -> None:
        _require_nonempty_string(self.source_name, "source_name")
        _require_nonempty_string(self.path, "path")

        if self.role not in DATASET_ROLES:
            raise ValueError(f"Rol no válido: {self.role!r}.")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetAssignment:
        if not isinstance(data, dict):
            raise ValueError("La asignación del dataset debe ser un objeto.")
        return cls(**data)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    model_name: str
    datasets: tuple[DatasetAssignment, ...]
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    experiment_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, "name")
        _require_nonempty_string(self.model_name, "model_name")
        _require_nonempty_string(self.experiment_id, "experiment_id")

        if not isinstance(self.description, str):
            raise ValueError("description debe ser un texto.")

        if not isinstance(self.datasets, tuple) or not self.datasets:
            raise ValueError("Debe declararse al menos una fuente de datos.")

        if not all(isinstance(d, DatasetAssignment) for d in self.datasets):
            raise ValueError(
                "datasets debe contener objetos DatasetAssignment."
            )

        names = [d.source_name.casefold() for d in self.datasets]
        if len(names) != len(set(names)):
            raise ValueError(
                "No se puede asignar más de un rol a la misma fuente."
            )

        if not any(d.role == "train_source" for d in self.datasets):
            raise ValueError(
                "Debe existir al menos una fuente con rol 'train_source'."
            )

        if not isinstance(self.parameters, dict):
            raise ValueError("parameters debe ser un diccionario.")

        # Evita compartir accidentalmente diccionarios anidados con el llamador.
        object.__setattr__(self, "parameters", deepcopy(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "model_name": self.model_name,
            "datasets": [d.to_dict() for d in self.datasets],
            "parameters": deepcopy(self.parameters),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentSpec:
        if not isinstance(data, dict):
            raise ValueError("El experimento debe ser un objeto.")

        return cls(
            experiment_id=data.get("experiment_id", uuid4().hex),
            name=data["name"],
            model_name=data["model_name"],
            datasets=tuple(
                DatasetAssignment.from_dict(d) for d in data["datasets"]
            ),
            parameters=data.get("parameters", {}),
            description=data.get("description", ""),
        )


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    experiment_id: str
    status: RunStatus
    checkpoint_path: str | None = None
    error_message: str | None = None
    created_at: str = field(default_factory=utc_now)
    finished_at: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.run_id, "run_id")
        _require_nonempty_string(self.experiment_id, "experiment_id")

        if self.status not in RUN_STATUSES:
            raise ValueError(f"Estado no válido: {self.status!r}.")

        if self.checkpoint_path is not None:
            _require_nonempty_string(self.checkpoint_path, "checkpoint_path")

        if self.error_message is not None and not isinstance(
            self.error_message, str
        ):
            raise ValueError("error_message debe ser un texto o None.")

        for field_name in ("created_at", "finished_at"):
            value = getattr(self, field_name)
            if value is None and field_name == "finished_at":
                continue
            _require_nonempty_string(value, field_name)
            try:
                datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(
                    f"{field_name} debe tener formato ISO 8601."
                ) from exc

        is_terminal = self.status in TERMINAL_RUN_STATUSES
        if is_terminal and self.finished_at is None:
            raise ValueError(
                "Una ejecución finalizada debe tener finished_at."
            )
        if not is_terminal and self.finished_at is not None:
            raise ValueError(
                "Una ejecución no finalizada no debe tener finished_at."
            )

        if self.status == "completed" and not self.checkpoint_path:
            raise ValueError(
                "Una ejecución completada debe indicar su checkpoint."
            )

        if self.status == "failed" and not self.error_message:
            raise ValueError(
                "Una ejecución fallida debe indicar el error."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        if not isinstance(data, dict):
            raise ValueError("La ejecución debe ser un objeto.")
        return cls(**data)
