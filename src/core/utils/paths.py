from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolver rutas sin fijar fuentes, modelos o escenarios en el código."""

    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def dataset_base(self) -> Path:
        return self.data_dir / "Clases_200"

    @property
    def dataset_complementario(self) -> Path:
        return self.data_dir / "Clases_200_comp"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def partitions_dir(self) -> Path:
        return self.results_dir / "particiones"

    def source_split_path(
        self,
        source_name: str,
        dataset_fingerprint: str | None = None,
        seed: int = 42,
        train_size: float = 0.70,
        val_size: float = 0.15,
        test_size: float = 0.15,
    ) -> Path:
        self._validate_component(source_name, "nombre de fuente")
        if dataset_fingerprint is None:
            filename = "split.json"  # ruta histórica, solo por compatibilidad externa
        else:
            fingerprint = dataset_fingerprint[:16]
            ratios = f"tr{train_size:.6f}-va{val_size:.6f}-te{test_size:.6f}"
            filename = f"{fingerprint}_seed{seed}_{ratios}.json"
        return self.partitions_dir / source_name / filename

    def model_checkpoint_dir(self, model_name: str, scenario: str | None = None) -> Path:
        self._validate_component(model_name, "nombre de modelo")
        if scenario:
            self._validate_component(scenario, "nombre de escenario")
        base = self.checkpoints_dir / scenario if scenario else self.checkpoints_dir
        return base / model_name

    def model_results_dir(self, model_name: str, scenario: str | None = None) -> Path:
        self._validate_component(model_name, "nombre de modelo")
        if scenario:
            self._validate_component(scenario, "nombre de escenario")
        base = self.results_dir / scenario if scenario else self.results_dir
        return base / model_name

    def experiment_dirs(self, scenario: str, model_name: str) -> tuple[Path, Path]:
        checkpoint_dir = self.model_checkpoint_dir(model_name, scenario)
        results_dir = self.model_results_dir(model_name, scenario)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir, results_dir

    def ensure_model_dirs(self, model_name: str) -> tuple[Path, Path]:
        """Compatibilidad con scripts históricos que usan results/<modelo>."""
        checkpoint_dir = self.model_checkpoint_dir(model_name)
        results_dir = self.model_results_dir(model_name)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir, results_dir

    @staticmethod
    def _validate_component(value: str, description: str) -> None:
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"Componente inválido para {description}: {value!r}")

    def relative_to_root(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.resolve().as_posix()
