from __future__ import annotations

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
IMAGE_SIZE = 224


def _normalizacion():
    return A.Normalize(mean=MEAN, std=STD)


def build_train_transform(image_size: int = IMAGE_SIZE):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=0.05,
            scale=(0.95, 1.05),
            rotate=(-15, 15),
            p=0.5,
        ),
        A.RandomBrightnessContrast(p=0.2),
        _normalizacion(),
        ToTensorV2(),
    ])


def build_eval_transform(image_size: int = IMAGE_SIZE):
    return A.Compose([
        A.Resize(image_size, image_size),
        _normalizacion(),
        ToTensorV2(),
    ])


def filtro_contornos(image, **kwargs):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return {"image": cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)}


def build_contour_transform(image_size: int = IMAGE_SIZE):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=0.05,
            scale=(0.95, 1.05),
            rotate=(-15, 15),
            p=0.5,
        ),
        A.RandomBrightnessContrast(p=0.2),
        A.Lambda(image=filtro_contornos),
        _normalizacion(),
        ToTensorV2(),
    ])


transformacion_train = build_train_transform()
transformacion_eval = build_eval_transform()
