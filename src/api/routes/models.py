from fastapi import APIRouter

from src.core.configuracion.settings import (
    get_model_spec,
    list_models,
)

router = APIRouter(tags=["Modelos"])


@router.get("/api/models")
def get_models() -> list[dict[str, str]]:
    models = []

    for name in list_models():
        spec = get_model_spec(name)
        models.append({
            "name": name,
            "architecture": spec.architecture,
            "family": spec.family,
        })

    return models