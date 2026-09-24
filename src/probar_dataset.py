from pathlib import Path

import torch
import os

from dataset_utils import GestosDataset, transformacion


# ============================================================
# Configuración de rutas
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_BASE = ROOT_DIR / "data" / "Clases_200"
DATASET_COMPLEMENTARIO = ROOT_DIR / "data" / "Clases_200_comp"


# ============================================================
# Verificación de rutas
# ============================================================

print("=" * 60)
print("PRUEBA DEL DATASET")
print("=" * 60)

print(f"\nDataset base:          {DATASET_BASE}")
print(f"Dataset complementario: {DATASET_COMPLEMENTARIO}")

if not DATASET_BASE.exists():
    raise FileNotFoundError(
        f"No se encontro el dataset base: {DATASET_BASE}"
    )

if not DATASET_COMPLEMENTARIO.exists():
    raise FileNotFoundError(
        f"No se encontro el dataset complementario: "
        f"{DATASET_COMPLEMENTARIO}"
    )


# ============================================================
# Crear dataset
# ============================================================

dataset = GestosDataset(
    carpetas_imagenes=[
        str(DATASET_BASE),
        str(DATASET_COMPLEMENTARIO)
    ],
    transformacion=transformacion
)

print("\n--- Cantidad de rutas por raíz REAL ---")

for carpeta in dataset.carpetas_imagenes:
    cantidad = sum(
        1
        for ruta in dataset.rutas_imagenes
        if ruta.startswith(carpeta + "/")
    )

    print(carpeta, "->", cantidad)




print("\n--- Carpetas utilizadas ---")

for carpeta in dataset.carpetas_imagenes:
    print(carpeta)

print(f"\nImágenes cargadas por GestosDataset: {len(dataset)}")

print("\n--- Imágenes por raíz según GestosDataset ---")

for carpeta in dataset.carpetas_imagenes:
    cantidad = sum(
        ruta.startswith(carpeta)
        for ruta in dataset.rutas_imagenes
    )

    print(f"{carpeta}: {cantidad}")


# ============================================================
# Información general
# ============================================================

print("\n--- Información general ---")

print(f"Cantidad de clases:   {len(dataset.clases)}")
print(f"Cantidad de imágenes: {len(dataset)}")

print("\nClases:")
print(dataset.clases)


# ============================================================
# Cantidad de imágenes por clase
# ============================================================

print("\n--- Imágenes por clase ---")

for indice, clase in enumerate(dataset.clases):

    cantidad = dataset.etiquetas.count(indice)

    print(f"{clase:>2}: {cantidad}")


# ============================================================
# Probar una imagen
# ============================================================

print("\n--- Prueba de carga ---")

imagen, etiqueta = dataset[0]

print(f"Tipo de imagen:       {type(imagen)}")
print(f"Forma de la imagen:   {imagen.shape}")
print(f"Tipo de dato:         {imagen.dtype}")
print(f"Etiqueta:             {etiqueta}")
print(f"Clase:                {dataset.clases[etiqueta]}")


# ============================================================
# Probar CUDA
# ============================================================

print("\n--- Prueba de GPU ---")

dispositivo = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Dispositivo: {dispositivo}")

imagen = imagen.to(dispositivo)

print(f"Imagen en dispositivo: {imagen.device}")

# ============================================================
# Distribucion por dataset de origen
# ============================================================

print("\n--- Distribucion por dataset de origen ---")

cantidad_base = sum(
    Path(ruta).is_relative_to(DATASET_BASE)
    for ruta in dataset.rutas_imagenes
)

cantidad_complementario = sum(
    Path(ruta).is_relative_to(DATASET_COMPLEMENTARIO)
    for ruta in dataset.rutas_imagenes
)

print(f"Dataset base:           {cantidad_base}")
print(f"Dataset complementario: {cantidad_complementario}")
print(f"Total:                   {cantidad_base + cantidad_complementario}")



print("\n--- Comprobación de clases compartidas ---")

for clase in ["0", "A", "P", "T"]:
    print(f"\nClase: {clase}")

    for carpeta in [
        DATASET_BASE,
        DATASET_COMPLEMENTARIO
    ]:
        carpeta_clase = carpeta / clase

        if carpeta_clase.exists():
            archivos = [
                archivo
                for archivo in carpeta_clase.iterdir()
                if archivo.is_file()
            ]

            print(
                f"  {carpeta.name}: "
                f"{len(archivos)} archivos"
            )
        else:
            print(
                f"  {carpeta.name}: NO EXISTE"
            )


# ============================================================
# Resultado
# ============================================================

print("\n" + "=" * 60)
print("PRUEBA FINALIZADA CORRECTAMENTE")
print("=" * 60)