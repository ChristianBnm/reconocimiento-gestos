import torch
import timm


def obtener_dispositivo():
    """Devuelve CUDA si está disponible; de lo contrario, CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def crear_modelo(arquitectura, num_classes, pretrained=True):
    """
    Crea un modelo utilizando timm y lo mueve al dispositivo disponible.
    """
    dispositivo = obtener_dispositivo()

    modelo = timm.create_model(
        arquitectura,
        pretrained=pretrained,
        num_classes=num_classes
    )

    modelo.to(dispositivo)

    return modelo


def congelar_backbone(modelo):
    """
    Congela todos los parámetros del modelo y deja entrenable
    únicamente el clasificador final.
    """
    for parametro in modelo.parameters():
        parametro.requires_grad = False

    for parametro in modelo.get_classifier().parameters():
        parametro.requires_grad = True

    return modelo