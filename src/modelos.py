import torch
import timm


def obtener_dispositivo():
    """Devuelve CUDA si está disponible; de lo contrario, CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def crear_modelo(arquitectura, num_classes, pretrained=True):
    """
    Crea un modelo utilizando timm y lo mueve al dispositivo disponible.

    Parameters
    ----------
    arquitectura : str
        Nombre de la arquitectura soportada por timm.
    num_classes : int
        Cantidad de clases de salida.
    pretrained : bool
        Indica si se utilizan pesos preentrenados.

    Returns
    -------
    torch.nn.Module
        Modelo configurado.
    """
    dispositivo = obtener_dispositivo()

    modelo = timm.create_model(arquitectura, pretrained=pretrained, num_classes=num_classes)

    modelo.to(dispositivo)

    return modelo