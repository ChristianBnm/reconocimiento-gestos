from __future__ import annotations

from pathlib import Path

import timm
import torch

from src.core.utils.hardware import obtener_dispositivo


def crear_modelo(arquitectura: str, num_classes: int, pretrained: bool = True):
    modelo = timm.create_model(
        arquitectura,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    dispositivo = obtener_dispositivo()
    return modelo.to(dispositivo)


def congelar_backbone(modelo):
    # Congela parámetros del modelo y deja entrenable el clasificador.
    for parametro in modelo.parameters():
        parametro.requires_grad = False

    classifier = modelo.get_classifier()
    if classifier is None:
        raise ValueError("La arquitectura no expone un clasificador mediante get_classifier().")

    for parametro in classifier.parameters():
        parametro.requires_grad = True

    return modelo


def cargar_checkpoint(modelo, ruta: Path, dispositivo: torch.device):
    # Carga checkpoints nuevos y también state_dicts legacy.
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint: {ruta}")

    payload = torch.load(
        ruta,
        map_location=dispositivo,
        weights_only=True,
    )

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload

    modelo.load_state_dict(state_dict, strict=True)
    return modelo
