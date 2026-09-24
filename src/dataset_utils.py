import os
import random

import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


# ============================================================
# Transformaciones de entrenamiento
# ============================================================

transformacion = A.Compose([
    A.HorizontalFlip(p=0.5),

    A.Affine(
        translate_percent=0.05,
        scale=(0.95, 1.05),
        rotate=(-15, 15),
        p=0.5
    ),

    A.RandomBrightnessContrast(p=0.2),

    A.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    ),

    ToTensorV2()
])


# ============================================================
# Filtro opcional de contornos
# ============================================================

def filtro_contornos(image, **kwargs):
    """Resalta los contornos de la imagen utilizando Canny."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return {"image": edges_rgb}


# ============================================================
# Transformaciones alternativas con contornos
# ============================================================

transformacion_contornos = A.Compose([
    A.HorizontalFlip(p=0.5),

    A.Affine(
        translate_percent=0.05,
        scale=(0.95, 1.05),
        rotate=(-15, 15),
        p=0.5
    ),

    A.RandomBrightnessContrast(p=0.2),

    A.Lambda(image=filtro_contornos),

    A.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    ),

    ToTensorV2()
])


# ============================================================
# Dataset
# ============================================================

class GestosDataset(Dataset):

    def __init__(self, carpetas_imagenes, transformacion=None):

        if isinstance(carpetas_imagenes, str):
            carpetas_imagenes = [carpetas_imagenes]

        self.carpetas_imagenes = carpetas_imagenes
        self.transformacion = transformacion

        self.rutas_imagenes = []
        self.etiquetas = []

        # Las clases se obtienen de la primera carpeta raíz.
        self.clases = sorted(
            nombre
            for nombre in os.listdir(carpetas_imagenes[0])
            if os.path.isdir(
                os.path.join(carpetas_imagenes[0], nombre)
            )
        )

        # Recorrer todas las clases y carpetas.
        for indice_clase, nombre_clase in enumerate(self.clases):

            for carpeta in carpetas_imagenes:

                carpeta_clase = os.path.join(
                    carpeta,
                    nombre_clase
                )

                if not os.path.isdir(carpeta_clase):
                    continue

                for nombre_archivo in os.listdir(carpeta_clase):

                    ruta_completa = os.path.join(
                        carpeta_clase,
                        nombre_archivo
                    )

                    if os.path.isfile(ruta_completa):
                        self.rutas_imagenes.append(ruta_completa)
                        self.etiquetas.append(indice_clase)

    def __len__(self):
        return len(self.rutas_imagenes)

    def __getitem__(self, indice):

        ruta = self.rutas_imagenes[indice]

        imagen = cv2.imread(ruta)

        if imagen is None:
            raise ValueError(
                f"No se pudo cargar la imagen: {ruta}"
            )

        imagen = cv2.cvtColor(
            imagen,
            cv2.COLOR_BGR2RGB
        )

        etiqueta = self.etiquetas[indice]

        if self.transformacion:
            imagen = self.transformacion(
                image=imagen
            )["image"]

        return imagen, etiqueta


# ============================================================
# Visualización
# ============================================================

def mostrar_imagenes_por_clase(
    dataset,
    clase,
    num_imagenes=5
):
    """
    Muestra imágenes aleatorias pertenecientes a una clase.
    """

    if clase not in dataset.clases:
        print(
            f"La clase '{clase}' no existe en el dataset."
        )
        return

    indice_clase = dataset.clases.index(clase)

    indices = [
        i
        for i, etiqueta in enumerate(dataset.etiquetas)
        if etiqueta == indice_clase
    ]

    if not indices:
        print(
            f"No hay imágenes disponibles para la clase '{clase}'."
        )
        return

    indices_muestra = random.sample(
        indices,
        min(num_imagenes, len(indices))
    )

    plt.figure(figsize=(12, 4))

    for posicion, indice in enumerate(indices_muestra):

        imagen = cv2.imread(
            dataset.rutas_imagenes[indice]
        )

        imagen = cv2.cvtColor(
            imagen,
            cv2.COLOR_BGR2RGB
        )

        plt.subplot(
            1,
            len(indices_muestra),
            posicion + 1
        )

        plt.imshow(imagen)
        plt.title(clase)
        plt.axis("off")

    plt.tight_layout()
    plt.show()