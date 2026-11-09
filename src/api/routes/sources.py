from fastapi import APIRouter, Query

from src.api.dependencies import PROJECT_ROOT
from src.core.configuracion.catalogo import cargar_catalogos
from src.core.configuracion.discovery import get_dataset_catalog

router = APIRouter(tags=["Datos"])


@router.get("/api/sources")
def get_sources(refresh: bool = Query(False)) -> dict:
    """Devuelve datasets detectados; refresh=True vuelve a escanear data/."""
    _, scenarios = cargar_catalogos()
    catalog = get_dataset_catalog(PROJECT_ROOT, refresh=refresh)

    return {
        "sources": catalog["sources"],
        "invalid_sources": catalog["invalid_sources"],
        "scenarios": [
            {
                "name": name,
                "description": spec.get("description", ""),
                "sources": spec["sources"],
            }
            for name, spec in scenarios.items()
        ],
    }
