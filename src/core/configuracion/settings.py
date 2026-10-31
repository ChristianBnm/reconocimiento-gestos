from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal
from typing import Mapping, Sequence


_CONFIG_DIR = Path(__file__).resolve().parent
_CLASS_FILE = _CONFIG_DIR / "clases.json"


def cargar_clases(path: str | Path | None = None) -> tuple[str, ...]:
    """Carga el vocabulario en tiempo de ejecución, nunca al importar el módulo.

    Se acepta el formato histórico (lista JSON) y el formato extensible
    {"classes": [..]}. El orden del archivo es el orden de los índices.
    """
    class_file = Path(path) if path else _CLASS_FILE
    if not class_file.is_absolute() and not class_file.exists():
        class_file = Path.cwd() / class_file
    if not class_file.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de clases: {class_file}")

    with class_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, Mapping):
        payload = payload.get("classes")
    if not isinstance(payload, list) or not payload:
        raise ValueError("El archivo de clases debe contener una lista JSON no vacía.")

    names = tuple(str(name).strip() for name in payload)
    if any(not name for name in names):
        raise ValueError("Los nombres de las clases no pueden estar vacíos.")
    if len(set(names)) != len(names):
        raise ValueError("El archivo de clases contiene nombres duplicados.")
    if any(name in {".", ".."} or "/" in name or "\\" in name for name in names):
        raise ValueError("Los nombres de clase no pueden contener separadores de ruta.")
    return names


CLASS_NAMES: tuple[str, ...] = cargar_clases()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    architecture: str
    family: str
    pretrained: bool = True


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "mobilenetv2": ModelSpec("mobilenetv2", "mobilenetv2_100", "CNN"),
    "densenet201": ModelSpec("densenet201", "densenet201", "CNN"),
    "vit_tiny": ModelSpec("vit_tiny", "vit_tiny_patch16_224", "Vision Transformer"),
    "vit_base": ModelSpec("vit_base", "vit_base_patch16_224", "Vision Transformer"),
}


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelSpec
    class_names: tuple[str, ...] = field(default_factory=lambda: CLASS_NAMES)
    image_size: int = 224
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42
    split_seed: int = 42
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 30
    patience: int = 5
    min_delta: float = 0.0
    num_workers: int = 0
    pin_memory: bool = True
    freeze_backbone: bool = True
    horizontal_flip_probability: float = 0.0
    affine_probability: float = 0.5
    brightness_contrast_probability: float = 0.2
    normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    device: Literal["auto", "cpu", "cuda"] = "auto"

    def __post_init__(self) -> None:
        if not self.class_names:
            raise ValueError("El vocabulario debe contener al menos una clase.")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names contiene clases duplicadas.")
        if self.image_size <= 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("image_size, batch_size y epochs deben ser positivos.")
        if self.patience < 1 or self.num_workers < 0:
            raise ValueError("patience debe ser >= 1 y num_workers >= 0.")
        ratios = (self.train_size, self.val_size, self.test_size)
        if any(ratio <= 0 or ratio >= 1 for ratio in ratios):
            raise ValueError("Las proporciones de train, val y test deben estar entre 0 y 1.")
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("Las proporciones de train, val y test deben sumar 1.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.min_delta < 0:
            raise ValueError("learning_rate debe ser positivo; weight_decay y min_delta no pueden ser negativos.")
        for field_name in (
            "horizontal_flip_probability",
            "affine_probability",
            "brightness_contrast_probability",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} debe estar entre 0 y 1.")
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError("La normalización debe tener tres canales RGB.")
        if any(value <= 0 for value in self.normalization_std):
            raise ValueError("Los valores de desviación estándar deben ser positivos.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError(
        "device debe ser 'auto', 'cpu' o 'cuda'."
        )

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.class_names)}

    def to_dict(self) -> dict:
        data = asdict(self)
        data["num_classes"] = self.num_classes
        data["class_to_idx"] = self.class_to_idx
        return data


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Modelo '{name}' no soportado. Disponibles: {available}") from exc


def list_models() -> list[str]:
    return list(MODEL_REGISTRY)


def build_experiment_config(
    model_name: str,
    class_names: Sequence[str] | None = None,
    **overrides,
) -> ExperimentConfig:
    spec = get_model_spec(model_name)
    names = tuple(class_names) if class_names is not None else CLASS_NAMES
    return ExperimentConfig(model=spec, class_names=names, **overrides)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]
