
from fastapi import APIRouter, HTTPException, Response, status
from threading import Event

from src.api.dependencies import (
    executor,
    experiment_service,
    read_artifact,
    run_experiment_task,
    run_cancel_events,
    run_cancel_events_lock,
    training_lock,
)

router = APIRouter(tags=["Ejecuciones"])


@router.post(
    "/api/experiments/{experiment_id}/runs",
    status_code=status.HTTP_202_ACCEPTED,
)
def start_run(experiment_id: str, response: Response) -> dict:
    try:
        experiment_service.get_experiment(experiment_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not training_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Ya hay un entrenamiento o evaluación en curso.",
        )

    try:
        run = experiment_service.create_run(experiment_id)
        with run_cancel_events_lock:
            run_cancel_events[run.run_id] = Event()

        try:
            executor.submit(
                run_experiment_task,
                experiment_id,
                run.run_id,
            )
        except Exception:
            with run_cancel_events_lock:
                run_cancel_events.pop(run.run_id, None)
            experiment_service.registry.update_run_status(
                run.run_id,
                "failed",
                error_message=(
                    "No se pudo iniciar la tarea de entrenamiento."
                ),
            )
            raise

    except Exception as exc:
        training_lock.release()

        if isinstance(exc, HTTPException):
            raise

        raise HTTPException(
            status_code=500,
            detail=f"No se pudo iniciar el entrenamiento: {exc}",
        ) from exc

    response.headers["Location"] = f"/api/runs/{run.run_id}"

    return run.to_dict()


@router.get("/api/runs")
def list_runs(experiment_id: str | None = None) -> list[dict]:
    try:
        return [
            run.to_dict()
            for run in experiment_service.list_runs(experiment_id)
        ]
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        return experiment_service.get_run(run_id).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/api/runs/{run_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
def cancel_run(run_id: str) -> dict:
    try:
        run = experiment_service.get_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if run.status not in {"pending", "running"}:
        raise HTTPException(
            status_code=409,
            detail=f"No se puede detener una ejecución con estado '{run.status}'.",
        )

    with run_cancel_events_lock:
        cancel_event = run_cancel_events.get(run_id)
        if cancel_event is None:
            raise HTTPException(
                status_code=409,
                detail="No hay una señal de cancelación activa para esta ejecución.",
            )
        cancel_event.set()

    return {"run_id": run_id, "status": "cancellation_requested"}


@router.get("/api/runs/{run_id}/history")
def get_run_history(run_id: str) -> list[dict]:
    return read_artifact(run_id, "history.json")


@router.get("/api/runs/{run_id}/config")
def get_run_config(run_id: str) -> dict:
    return read_artifact(run_id, "config.json")


@router.get("/api/runs/{run_id}/results")
def get_run_results(run_id: str) -> dict:
    return read_artifact(run_id, "evaluation.json")
