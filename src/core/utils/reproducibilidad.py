from __future__ import annotations

import os
import random

import numpy as np
import torch


def establecer_semilla(seed: int, deterministic: bool = False) -> None:
    """Configura fuentes habituales de aleatoriedad.

    `deterministic=False` mantiene el comportamiento actual sin forzar
    operaciones deterministas potencialmente más lentas o no soportadas.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Evita que cuDNN seleccione algoritmos distintos según el benchmark.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic

    if deterministic:
        torch.use_deterministic_algorithms(True)
