import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from fastapi import HTTPException

from src.core.experimentos.service import ExperimentService


# Raíz del proyecto: directorio que contiene src/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Instancias compartidas por todos los routers.
experiment_service = ExperimentService(root=PROJECT_ROOT)
training_lock = Lock()
executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="lsa-training",
)


def run_experiment_task(experiment_id: str, run_id: str) -> None:
    """Ejecuta el entrenamiento en segundo plano y libera el lock."""
    try:
        experiment_service.run_experiment(
            experiment_id,
            run_id=run_id,
        )
    finally:
        training_lock.release()


def read_artifact(run_id: str, filename: str) -> dict:
    """Lee un artefacto JSON asociado a una ejecución."""
    try:
        run = experiment_service.get_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    artifact_path = (
        PROJECT_ROOT
        / "results"
        / "experiments"
        / run.experiment_id
        / run.run_id
        / filename
    )

    if not artifact_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Todavía no existe el artefacto '{filename}' "
                f"para la ejecución {run_id}."
            ),
        )

    try:
        return json.loads(
            artifact_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo leer {filename}.",
        ) from exc
