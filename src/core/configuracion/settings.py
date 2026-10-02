from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple


_CLASS_FILE = Path(__file__).with_name("clases.json")
CLASS_NAMES: Tuple[str, ...] = tuple(__import__("json").loads(_CLASS_FILE.read_text(encoding="utf-8")))
NUM_CLASSES = len(CLASS_NAMES)


@dataclass(frozen=True)
class ModelSpec:
    """Especificación de un modelo disponible para los experimentos."""

    name: str
    architecture: str
    family: str
    pretrained: bool = True


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "mobilenetv2": ModelSpec(
        name="mobilenetv2",
        architecture="mobilenetv2_100",
        family="CNN",
    ),
    "vit_tiny": ModelSpec(
        name="vit_tiny",
        architecture="vit_tiny_patch16_224",
        family="Vision Transformer",
    ),
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuración completa y serializable de un experimento."""

    model: ModelSpec
    num_classes: int = NUM_CLASSES
    image_size: int = 224
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 30
    patience: int = 5
    min_delta: float = 0.0
    num_workers: int = 0
    pin_memory: bool = True

    @property
    def class_names(self) -> Tuple[str, ...]:
        return CLASS_NAMES

    def to_dict(self) -> dict:
        data = asdict(self)
        data["class_names"] = list(self.class_names)
        return data


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        disponibles = ", ".join(MODEL_REGISTRY)
        raise ValueError(
            f"Modelo '{name}' no soportado. Disponibles: {disponibles}"
        ) from exc


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def build_experiment_config(model_name: str) -> ExperimentConfig:
    spec = get_model_spec(model_name)
    return ExperimentConfig(model=spec)


def project_root() -> Path:
    """Devuelve la raíz del repositorio a partir de este archivo."""
    return Path(__file__).resolve().parents[3]
