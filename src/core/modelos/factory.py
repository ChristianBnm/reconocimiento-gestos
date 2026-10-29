from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch

from src.core.configuracion.settings import ExperimentConfig
from src.core.utils.hardware import obtener_dispositivo


def crear_modelo(
    arquitectura: str,
    num_classes: int,
    pretrained: bool = True,
    dispositivo: torch.device | None = None,
):
    if num_classes < 1:
        raise ValueError("num_classes debe ser mayor que cero.")
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            "Falta timm. Instalá las dependencias del proyecto antes de crear modelos."
        ) from exc

    model = timm.create_model(
        arquitectura,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model.to(dispositivo or obtener_dispositivo())


def adaptar_normalizacion_a_modelo(
    config: ExperimentConfig, modelo: torch.nn.Module
) -> ExperimentConfig:
    """Devuelve la configuración con la normalización declarada por timm.

    Diferentes checkpoints preentrenados pueden esperar estadísticas distintas.
    Se intenta primero la API oficial de timm y luego sus metadatos del modelo;
    si no existen, se conservan los valores configurados explícitamente.
    """
    mean = std = None
    try:
        import timm

        data_config = timm.data.resolve_model_data_config(modelo)
        mean = data_config.get("mean")
        std = data_config.get("std")
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    if mean is None or std is None:
        pretrained_cfg = getattr(modelo, "pretrained_cfg", None)
        if isinstance(pretrained_cfg, dict):
            mean = mean if mean is not None else pretrained_cfg.get("mean")
            std = std if std is not None else pretrained_cfg.get("std")
        if mean is None or std is None:
            default_cfg = getattr(modelo, "default_cfg", None)
            if isinstance(default_cfg, dict):
                mean = mean if mean is not None else default_cfg.get("mean")
                std = std if std is not None else default_cfg.get("std")

    if mean is None or std is None:
        return config
    mean = tuple(float(value) for value in mean)
    std = tuple(float(value) for value in std)
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("La configuración de normalización del modelo no es válida.")
    return replace(config, normalization_mean=mean, normalization_std=std)


def congelar_backbone(modelo: torch.nn.Module):
    """Congela el backbone y deja entrenable el clasificador final."""
    for parameter in modelo.parameters():
        parameter.requires_grad = False

    classifier = modelo.get_classifier() if hasattr(modelo, "get_classifier") else None
    if classifier is None:
        raise ValueError(
            "La arquitectura no expone un clasificador con get_classifier(); "
            "definí una estrategia de congelamiento específica para ese modelo."
        )
    classifier_parameters = list(classifier.parameters()) if hasattr(classifier, "parameters") else []
    if not classifier_parameters:
        raise ValueError("El clasificador no expone parámetros entrenables.")
    for parameter in classifier_parameters:
        parameter.requires_grad = True
    return modelo


def descongelar_modelo(modelo: torch.nn.Module):
    for parameter in modelo.parameters():
        parameter.requires_grad = True
    return modelo


def cargar_checkpoint(
    modelo: torch.nn.Module,
    ruta: Path,
    dispositivo: torch.device,
    class_names: Sequence[str] | None = None,
    expected_architecture: str | None = None,
):
    """Carga checkpoints versionados y state_dicts legacy.

    Si el checkpoint tiene metadatos, valida el vocabulario y la arquitectura
    antes de cargar pesos para evitar interpretar índices con otra etiqueta.
    Devuelve el payload para que inferencia pueda recuperar sus metadatos.
    """
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el checkpoint: {ruta}")
    payload = torch.load(ruta, map_location=dispositivo, weights_only=True)
    metadata: dict = {}
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        metadata = payload
    elif isinstance(payload, dict) and payload and all(
        isinstance(key, str) for key in payload
    ):
        state_dict = payload
        warnings.warn(
            "Checkpoint legacy sin metadatos de vocabulario; se omite la validación de etiquetas.",
            UserWarning,
            stacklevel=2,
        )
    else:
        raise ValueError(f"Formato de checkpoint no reconocido: {ruta}")

    if expected_architecture and metadata.get("architecture"):
        if metadata["architecture"] != expected_architecture:
            raise ValueError(
                f"El checkpoint es de '{metadata['architecture']}', no de '{expected_architecture}'."
            )
    if class_names is not None and metadata.get("class_names") is not None:
        if list(class_names) != list(metadata["class_names"]):
            raise ValueError(
                "El vocabulario solicitado no coincide con el del checkpoint. "
                "No se cargarán pesos con un mapeo de etiquetas diferente."
            )
    elif metadata:
        warnings.warn(
            "Checkpoint legacy sin vocabulario explícito: no se puede validar el orden de las clases.",
            UserWarning,
            stacklevel=2,
        )
    modelo.load_state_dict(state_dict, strict=True)
    return metadata
