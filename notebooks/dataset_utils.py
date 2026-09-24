# ===============================
# Utilidades para cargar y transformar el dataset
# ===============================

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ===============================
# Transformaciones de aumento de datos y normalización para ViT
# ===============================
transformacion = A.Compose([
    A.HorizontalFlip(p = 0.5),
    A.Affine(
    translate_percent=0.05,       # equivalente a shift_limit=0.05
    scale=(0.95, 1.05),           # equivalente a scale_limit=0.05
    rotate=(-15, 15),             # equivalente a rotate_limit=15
    p=0.5
),
    A.RandomBrightnessContrast(p = 0.2),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Normaliza a float y rango [-1,1]
    ToTensorV2()  # Debe ser la última
])

# ===============================
# Filtro opcional para resaltar contornos con Canny
# ===============================
def filtro_contornos(image, **kwargs):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)          # Convertir a escala de grises
    edges = cv2.Canny(gray, 100, 200)                       # Detección de bordes con Canny
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)     # Volver a 3 canales (RGB) para ViT/ResNet
    return {"image": edges_rgb}

# ===============================
# Transformaciones alternativas con contornos
# ===============================
transformacion_contornos = A.Compose([
    A.HorizontalFlip(p = 0.5),
    A.Affine(
    translate_percent=0.05,       # equivalente a shift_limit=0.05
    scale=(0.95, 1.05),           # equivalente a scale_limit=0.05
    rotate=(-15, 15),             # equivalente a rotate_limit=15
    p=0.5
),
    
    A.RandomBrightnessContrast(p = 0.2),
    A.Lambda(image = filtro_contornos),                # Aplica filtro de contornos el 50% de las veces
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# ===============================
# Clase para cargar el dataset de gestos
# ===============================
class GestosDataset(Dataset):
    def __init__(self, carpetas_imagenes, transformacion=None):
        if isinstance(carpetas_imagenes, str):
            carpetas_imagenes = [carpetas_imagenes]

        self.carpetas_imagenes = carpetas_imagenes
        self.transformacion = transformacion
        self.rutas_imagenes = []                                                            # Paths completos a cada archivo de imagen.
        self.etiquetas = []                                                                 # Label entero correspondiente a cada imagen
        self.clases = sorted(os.listdir(carpetas_imagenes[0]))                              # Leer nombres de subcarpetas y ordenarlos.

        # Recorrer todas las carpetas raíz
        for indice_clase, nombre_clase in enumerate(self.clases):
            for carpeta in carpetas_imagenes:
                carpeta_clase = os.path.join(carpeta, nombre_clase)
                if os.path.isdir(carpeta_clase):
                    for nombre_archivo in os.listdir(carpeta_clase):
                        ruta_completa = os.path.join(carpeta_clase, nombre_archivo)
                        self.rutas_imagenes.append(ruta_completa)
                        self.etiquetas.append(indice_clase)

    def __len__(self):
        return len(self.rutas_imagenes)                             # Para hacer len(dataset).

    def __getitem__(self, indice):
        # Leer imagen con OpenCV (por defecto carga en BGR)
        imagen = cv2.imread(self.rutas_imagenes[indice])
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)            # Convertir a RGB

        etiqueta = self.etiquetas[indice]
        
        if self.transformacion:
            # Albumentations espera un diccionario con la clave "image"
            imagen = self.transformacion(image=imagen)["image"]
            
        return imagen, etiqueta


# ===============================
# Función de visualización
# ===============================
def mostrar_imagenes_por_clase(dataset, clase, num_imagenes=5):
    """
    Muestra imágenes aleatorias de una clase específica del dataset.
    
    Parámetros:
        dataset: instancia de GestosDataset
        clase: nombre de la clase (string, ej. "A")
        num_imagenes: cantidad de imágenes a mostrar
    """
    if clase not in dataset.clases:
        print(f"La clase '{clase}' no existe en el dataset.")
        return

    # Buscar índice de la clase
    idx_clase = dataset.clases.index(clase)

    # Obtener índices de todas las imágenes de esa clase
    indices = [i for i, y in enumerate(dataset.etiquetas) if y == idx_clase]

    # Seleccionar al azar
    indices_muestra = random.sample(indices, min(num_imagenes, len(indices)))

    # Mostrar imágenes
    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(indices_muestra):
        img = cv2.imread(dataset.rutas_imagenes[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(1, len(indices_muestra), i+1)
        plt.imshow(img)
        plt.title(clase)
        plt.axis("off")
    plt.show()
