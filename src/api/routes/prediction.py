from fastapi import APIRouter, HTTPException, Request
from src.api.dependencies import PROJECT_ROOT, experiment_service
from src.core.evaluacion.prediccion_individual import predecir_imagen

router = APIRouter(tags=["Predicción individual"])

@router.post("/api/runs/{run_id}/predict")
async def predict_image(run_id: str, request: Request) -> dict:
    if request.headers.get("content-type", "").split(";")[0] not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Formato no admitido. Subí una imagen JPG o PNG.")
    content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="La imagen supera el límite de 10 MB.")
    try:
        run = experiment_service.get_run(run_id)
        if run.status != "completed":
            raise HTTPException(status_code=409, detail="La ejecución debe estar completada.")
        return predecir_imagen(PROJECT_ROOT, run_id, content)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
