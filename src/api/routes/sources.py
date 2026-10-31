from fastapi import APIRouter
from src.core.configuracion.catalogo import cargar_catalogos

router = APIRouter(tags=["Datos"])


@router.get("/api/sources")
def get_sources() -> dict:
    sources, scenarios = cargar_catalogos()

    return {
        "sources": [
            {
                "name": name,
                "description": spec.get("description", ""),
                "path": spec["path"],
            }
            for name, spec in sources.items()
        ],
        "scenarios": [
            {
                "name": name,
                "description": spec.get("description", ""),
                "sources": spec["sources"],
            }
            for name, spec in scenarios.items()
        ],
    }
