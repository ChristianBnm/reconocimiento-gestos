from fastapi import APIRouter

from src.core.configuracion.settings import (
    get_model_spec,
    list_models,
)

router = APIRouter(tags=["Modelos"])


@router.get("/api/models")
def get_models() -> list[dict[str, str]]:
    return [
        {
            "name": name,
            "architecture": get_model_spec(name).architecture,
            "family": get_model_spec(name).family,
        }
        for name in list_models()
    ]
