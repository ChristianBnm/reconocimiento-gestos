import torch


def obtener_dispositivo() -> torch.device:
    """Usa CUDA cuando está disponible y CPU en caso contrario."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
