from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Rutas estándar del proyecto."""

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

    def model_checkpoint_dir(self, model_name: str) -> Path:
        return self.checkpoints_dir / model_name

    def model_results_dir(self, model_name: str) -> Path:
        return self.results_dir / model_name

    def ensure_model_dirs(self, model_name: str) -> tuple[Path, Path]:
        checkpoint_dir = self.model_checkpoint_dir(model_name)
        results_dir = self.model_results_dir(model_name)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir, results_dir

    def relative_to_root(self, path: Path) -> str:
        return path.resolve().relative_to(self.root.resolve()).as_posix()
