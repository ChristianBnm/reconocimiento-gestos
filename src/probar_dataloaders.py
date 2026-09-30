from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset_utils import (
    GestosDataset,
    DatasetSubset,
    transformacion_train,
    transformacion_eval
)

from configuracion import (
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    SEED,
    BATCH_SIZE
)


# ============================================================
# Rutas
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_BASE = ROOT_DIR / "data" / "Clases_200"
DATASET_COMPLEMENTARIO = ROOT_DIR / "data" / "Clases_200_comp"


# ============================================================
# Dataset completo
# ============================================================

dataset_completo = GestosDataset(
    carpetas_imagenes=[
        str(DATASET_BASE),
        str(DATASET_COMPLEMENTARIO)
    ],
    transformacion=None
)

indices = list(range(len(dataset_completo)))
etiquetas = dataset_completo.etiquetas


# ============================================================
# División Train / Validation / Test
# ============================================================

indices_train, indices_temp = train_test_split(
    indices,
    test_size=VAL_SIZE + TEST_SIZE,
    random_state=SEED,
    stratify=etiquetas
)

proporcion_val = VAL_SIZE / (VAL_SIZE + TEST_SIZE)

etiquetas_temp = [
    etiquetas[i]
    for i in indices_temp
]

indices_val, indices_test = train_test_split(
    indices_temp,
    test_size=1 - proporcion_val,
    random_state=SEED,
    stratify=etiquetas_temp
)


# ============================================================
# Crear subconjuntos
# ============================================================

dataset_train = DatasetSubset(
    dataset=dataset_completo,
    indices=indices_train,
    transformacion=transformacion_train
)

dataset_val = DatasetSubset(
    dataset=dataset_completo,
    indices=indices_val,
    transformacion=transformacion_eval
)

dataset_test = DatasetSubset(
    dataset=dataset_completo,
    indices=indices_test,
    transformacion=transformacion_eval
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    dataset_val,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Información general
# ============================================================

print("=" * 60)
print("PRUEBA DE DATALOADERS")
print("=" * 60)

print(f"\nDataset completo: {len(dataset_completo)}")

print("\nTrain:")
print(f"  Imágenes: {len(dataset_train)}")
print(f"  Batches:  {len(train_loader)}")

print("\nValidation:")
print(f"  Imágenes: {len(dataset_val)}")
print(f"  Batches:  {len(val_loader)}")

print("\nTest:")
print(f"  Imágenes: {len(dataset_test)}")
print(f"  Batches:  {len(test_loader)}")


# ============================================================
# Probar batch de Train
# ============================================================

imagenes_train, etiquetas_train = next(iter(train_loader))

print("\n--- Batch Train ---")
print(f"Imágenes:  {imagenes_train.shape}")
print(f"Etiquetas: {etiquetas_train.shape}")
print(f"Tipo:      {imagenes_train.dtype}")


# ============================================================
# Probar batch de Validation
# ============================================================

imagenes_val, etiquetas_val = next(iter(val_loader))

print("\n--- Batch Validation ---")
print(f"Imágenes:  {imagenes_val.shape}")
print(f"Etiquetas: {etiquetas_val.shape}")
print(f"Tipo:      {imagenes_val.dtype}")


# ============================================================
# Probar batch de Test
# ============================================================

imagenes_test, etiquetas_test = next(iter(test_loader))

print("\n--- Batch Test ---")
print(f"Imágenes:  {imagenes_test.shape}")
print(f"Etiquetas: {etiquetas_test.shape}")
print(f"Tipo:      {imagenes_test.dtype}")


# ============================================================
# Probar GPU
# ============================================================

dispositivo = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

imagenes_train = imagenes_train.to(dispositivo)
etiquetas_train = etiquetas_train.to(dispositivo)

print("\n--- Prueba de GPU ---")
print(f"Dispositivo: {dispositivo}")
print(f"Imágenes:    {imagenes_train.device}")
print(f"Etiquetas:   {etiquetas_train.device}")


# ============================================================
# Resultado
# ============================================================

print("\n" + "=" * 60)
print("PRUEBA FINALIZADA CORRECTAMENTE")
print("=" * 60)