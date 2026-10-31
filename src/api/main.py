# API de reconocimiento de gestos LSA.

from fastapi import FastAPI

from src.api.routes import evaluation, experiments, health, models, runs, sources

app = FastAPI(
    title="LSA Gesture Recognition API",
    description="Gestionar experimentos y ejecuciones de reconocimiento de gestos.",
    version="0.2.0",
)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(sources.router)
app.include_router(experiments.router)
app.include_router(runs.router)
app.include_router(evaluation.router)
