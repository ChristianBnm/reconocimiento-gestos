
from fastapi import APIRouter

router = APIRouter(tags=["Sistema"])


@router.get("/api/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "tesis_gestos",
    }
