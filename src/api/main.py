from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
import json

from fastapi import FastAPI, HTTPException, Response

from src.core.configuracion.catalogo import cargar_catalogos
from src.core.configuracion.settings import get_model_spec, list_models
from src.core.experimentos.evaluation import evaluar_ejecucion
from src.core.experimentos.schemas import ExperimentSpec
from src.core.experimentos.service import ExperimentService

app = FastAPI(
    title="LSA Gesture Recognition API",
    description="Gestionar experimentos y ejecuciones de reconocimiento de gestos.",
    version="0.2.0",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
experiment_service = ExperimentService(root=PROJECT_ROOT)
training_lock = Lock()
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lsa-training")


def _run_experiment_task(experiment_id: str, run_id: str) -> None:
    try:
        experiment_service.run_experiment(experiment_id, run_id=run_id)
    finally:
        training_lock.release()


def _run_artifact_path(run_id: str, filename: str) -> Path:
    run = experiment_service.get_run(run_id)
    return (
        PROJECT_ROOT / "results" / "experiments"
        / run.experiment_id / run.run_id / filename
    )


def _read_artifact(run_id: str, filename: str) -> dict:
    run = experiment_service.get_run(run_id)
    path = _run_artifact_path(run_id, filename)
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Todavía no existe el artefacto '{filename}' para la ejecución {run_id}.",
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo leer {filename}.") from exc


@app.get("/api/health", tags=["Sistema"])
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "tesis_gestos"}


@app.get("/api/models", tags=["Modelos"])
def get_models() -> list[dict[str, str]]:
    return [
        {"name": name, "architecture": get_model_spec(name).architecture,
         "family": get_model_spec(name).family}
        for name in list_models()
    ]


@app.get("/api/sources", tags=["Datos"])
def get_sources() -> dict:
    sources, scenarios = cargar_catalogos()
    return {
        "sources": [
            {"name": name, "description": spec.get("description", ""),
             "path": spec["path"]}
            for name, spec in sources.items()
        ],
        "scenarios": [
            {"name": name, "description": spec.get("description", ""),
             "sources": spec["sources"]}
            for name, spec in scenarios.items()
        ],
    }


@app.post("/api/experiments", tags=["Experimentos"], status_code=201)
def create_experiment(payload: dict) -> dict:
    try:
        data = dict(payload)
        if not data.get("experiment_id"):
            model_name = data.get("model_name")
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("model_name es obligatorio para generar la ID.")
            data["experiment_id"] = experiment_service.next_experiment_id(model_name)
        spec = ExperimentSpec.from_dict(data)
        return experiment_service.create_experiment(spec).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/experiments", tags=["Experimentos"])
def list_experiments() -> list[dict]:
    result = []
    for experiment_id in experiment_service.list_experiments():
        spec = experiment_service.get_experiment(experiment_id)
        result.append(spec.to_dict())
    return result


@app.get("/api/experiments/{experiment_id}", tags=["Experimentos"])
def get_experiment(experiment_id: str) -> dict:
    try:
        return experiment_service.get_experiment(experiment_id).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/experiments/{experiment_id}/runs", tags=["Ejecuciones"], status_code=202)
def start_run(experiment_id: str, response: Response) -> dict:
    try:
        experiment_service.get_experiment(experiment_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not training_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Ya hay un entrenamiento o evaluación en curso.")

    try:
        run = experiment_service.create_run(experiment_id)
        try:
            executor.submit(_run_experiment_task, experiment_id, run.run_id)
        except Exception:
            # Si la cola no acepta el trabajo, no dejar el run eternamente pendiente.
            experiment_service.registry.update_run_status(
                run.run_id, "failed", error_message="No se pudo iniciar la tarea de entrenamiento."
            )
            raise
    except Exception as exc:
        training_lock.release()
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"No se pudo iniciar el entrenamiento: {exc}") from exc

    response.headers["Location"] = f"/api/runs/{run.run_id}"
    return run.to_dict()


@app.get("/api/runs", tags=["Ejecuciones"])
def list_runs(experiment_id: str | None = None) -> list[dict]:
    try:
        return [run.to_dict() for run in experiment_service.list_runs(experiment_id)]
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}", tags=["Ejecuciones"])
def get_run(run_id: str) -> dict:
    try:
        return experiment_service.get_run(run_id).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/history", tags=["Ejecuciones"])
def get_run_history(run_id: str) -> dict:
    try:
        return _read_artifact(run_id, "history.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/config", tags=["Ejecuciones"])
def get_run_config(run_id: str) -> dict:
    try:
        return _read_artifact(run_id, "config.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/results", tags=["Ejecuciones"])
def get_run_results(run_id: str) -> dict:
    try:
        return _read_artifact(run_id, "evaluation.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/evaluate", tags=["Evaluación"])
def evaluate_run(run_id: str) -> dict:
    try:
        run = experiment_service.get_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if run.status != "completed":
        raise HTTPException(status_code=409, detail="Solo se pueden evaluar ejecuciones completadas.")
    if not training_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Ya hay un entrenamiento o evaluación en curso.")
    try:
        return evaluar_ejecucion(PROJECT_ROOT, run_id, overwrite=True)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        training_lock.release()
