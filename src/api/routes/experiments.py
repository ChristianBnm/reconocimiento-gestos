from fastapi import APIRouter, HTTPException, Response, status
from src.api.dependencies import experiment_service
from src.core.experimentos.schemas import ExperimentSpec

router = APIRouter(tags=["Experimentos"])


@router.post(
    "/api/experiments",
    status_code=status.HTTP_201_CREATED,
)


def create_experiment(payload: dict) -> dict:
    try:
        data = dict(payload)

        if not data.get("experiment_id"):
            model_name = data.get("model_name")

            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError(
                    "model_name es obligatorio para generar la ID."
                )

            data["experiment_id"] = (
                experiment_service.next_experiment_id(model_name)
            )

        spec = ExperimentSpec.from_dict(data)

        return experiment_service.create_experiment(spec).to_dict()

    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/api/experiments")
def list_experiments() -> list[dict]:
    result = []

    for experiment_id in experiment_service.list_experiments():
        spec = experiment_service.get_experiment(experiment_id)
        result.append(spec.to_dict())

    return result


@router.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict:
    try:
        return experiment_service.get_experiment(
            experiment_id
        ).to_dict()

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete(
    "/api/experiments/{experiment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)

def delete_experiment(experiment_id: str) -> Response:
    try:
        experiment_service.delete_experiment(experiment_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)