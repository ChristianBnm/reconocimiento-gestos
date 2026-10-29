from __future__ import annotations

import cv2

from src.core.configuracion.settings import ExperimentConfig


IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _resolve_config(
    image_size: int | None = None,
    config: ExperimentConfig | None = None,
):
    if config is not None:
        return (
            config.image_size,
            config.normalization_mean,
            config.normalization_std,
            config.horizontal_flip_probability,
            config.affine_probability,
            config.brightness_contrast_probability,
        )
    return (
        image_size or IMAGE_SIZE,
        MEAN,
        STD,
        0.0,
        0.5,
        0.2,
    )


def build_train_transform(
    image_size: int | None = None,
    config: ExperimentConfig | None = None,
):
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as exc:
        raise RuntimeError(
            "Falta Albumentations. Instalá las dependencias de requirements.txt."
        ) from exc

    size, mean, std, flip_probability, affine_probability, brightness_probability = (
        _resolve_config(image_size, config)
    )
    transforms = [A.Resize(size, size)]
    # El volteo está desactivado por defecto: algunas etiquetas pueden cambiar
    # de significado al reflejar una mano. Activarlo exige justificación por clase.
    if flip_probability > 0:
        transforms.append(A.HorizontalFlip(p=flip_probability))
    transforms.extend(
        [
            A.Affine(
                translate_percent=0.05,
                scale=(0.95, 1.05),
                rotate=(-15, 15),
                p=affine_probability,
            ),
            A.RandomBrightnessContrast(p=brightness_probability),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def build_eval_transform(
    image_size: int | None = None,
    config: ExperimentConfig | None = None,
):
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as exc:
        raise RuntimeError(
            "Falta Albumentations. Instalá las dependencias de requirements.txt."
        ) from exc

    size, mean, std, *_ = _resolve_config(image_size, config)
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def filtro_contornos(image, **kwargs):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return {"image": cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)}


def build_contour_transform(
    image_size: int = IMAGE_SIZE,
    config: ExperimentConfig | None = None,
):
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as exc:
        raise RuntimeError(
            "Falta Albumentations. Instalá las dependencias de requirements.txt."
        ) from exc

    size, mean, std, flip_probability, affine_probability, brightness_probability = (
        _resolve_config(image_size, config)
    )
    transforms = [A.Resize(size, size)]
    if flip_probability > 0:
        transforms.append(A.HorizontalFlip(p=flip_probability))
    transforms.extend([
        A.Affine(
            translate_percent=0.05,
            scale=(0.95, 1.05),
            rotate=(-15, 15),
            p=affine_probability,
        ),
        A.RandomBrightnessContrast(p=brightness_probability),
        A.Lambda(image=filtro_contornos),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    return A.Compose(transforms)


class _LazyTransform:
    """Compatibilidad con scripts existentes sin importar Albumentations al inicio."""

    def __init__(self, builder):
        self.builder = builder
        self._transform = None

    def __call__(self, **kwargs):
        if self._transform is None:
            self._transform = self.builder()
        return self._transform(**kwargs)


transformacion_train = _LazyTransform(build_train_transform)
transformacion_eval = _LazyTransform(build_eval_transform)
