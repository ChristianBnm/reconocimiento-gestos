
from fastapi import APIRouter, HTTPException

from src.core.experimentos.evaluation import evaluar_ejecucion

from src.api.dependencies import (
    PROJECT_ROOT,
    experiment_service,
    training_lock,
)

router = APIRouter(tags=["Evaluación"])


@router.post("/api/runs/{run_id}/evaluate")
def evaluate_run(run_id: str) -> dict:
    try:
        run = experiment_service.get_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if run.status != "completed":
        raise HTTPException(
            status_code=409,
            detail="Solo se pueden evaluar ejecuciones completadas.",
        )

    if not training_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Ya hay un entrenamiento o evaluación en curso.",
        )

    try:
        return evaluar_ejecucion(
            PROJECT_ROOT,
            run_id,
            overwrite=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        training_lock.release()
