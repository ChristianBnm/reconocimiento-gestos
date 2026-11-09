from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.dependencies import PROJECT_ROOT, recover_interrupted_runs
from src.api.routes import evaluation, experiments, health, models, runs, sources
from src.core.configuracion.discovery import get_dataset_catalog


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Recupera estados que quedaron activos en el proceso anterior.
    recover_interrupted_runs()
    # Detecta datasets al iniciar; el endpoint /sources permite refrescarlo luego.
    get_dataset_catalog(PROJECT_ROOT)
    yield


app = FastAPI(
    title="LSA Gesture Recognition API",
    description="Gestionar experimentos y ejecuciones de reconocimiento de gestos.",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(sources.router)
app.include_router(experiments.router)
app.include_router(runs.router)
app.include_router(evaluation.router)
