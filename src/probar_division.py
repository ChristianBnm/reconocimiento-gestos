from pathlib import Path

from sklearn.model_selection import train_test_split

from dataset_utils import GestosDataset
from configuracion import (
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    SEED
)


# ============================================================
# Rutas
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_BASE = ROOT_DIR / "data" / "Clases_200"
DATASET_COMPLEMENTARIO = ROOT_DIR / "data" / "Clases_200_comp"


# ============================================================
# Validación de configuración
# ============================================================

total_porcentajes = TRAIN_SIZE + VAL_SIZE + TEST_SIZE

if abs(total_porcentajes - 1.0) > 1e-6:
    raise ValueError(
        "TRAIN_SIZE + VAL_SIZE + TEST_SIZE debe ser igual a 1.0"
    )


# ============================================================
# Cargar dataset
# ============================================================

dataset = GestosDataset(
    carpetas_imagenes=[
        str(DATASET_BASE),
        str(DATASET_COMPLEMENTARIO)
    ],
    transformacion=None
)

indices = list(range(len(dataset)))
etiquetas = dataset.etiquetas


# ============================================================
# División Train / Validation / Test
# ============================================================

indices_train, indices_temp = train_test_split(
    indices,
    test_size=VAL_SIZE + TEST_SIZE,
    random_state=SEED,
    stratify=etiquetas
)


# Proporción de validation dentro del conjunto temporal.
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
# Resultados
# ============================================================

print("\n========================================")
print("DIVISIÓN DEL DATASET")
print("========================================")

print(f"Total:       {len(indices):5d}")
print(f"Train:       {len(indices_train):5d}")
print(f"Validation:  {len(indices_val):5d}")
print(f"Test:        {len(indices_test):5d}")

print("\nPorcentajes configurados:")
print(f"Train:       {TRAIN_SIZE:.0%}")
print(f"Validation:  {VAL_SIZE:.0%}")
print(f"Test:        {TEST_SIZE:.0%}")

print(f"\nSeed: {SEED}")



from collections import Counter


# ============================================================
# Distribución por clase
# ============================================================

contador_train = Counter(
    etiquetas[i]
    for i in indices_train
)

contador_val = Counter(
    etiquetas[i]
    for i in indices_val
)

contador_test = Counter(
    etiquetas[i]
    for i in indices_test
)


print("\n========================================")
print("DISTRIBUCIÓN POR CLASE")
print("========================================")

print(
    f"{'Clase':<8}"
    f"{'Total':>8}"
    f"{'Train':>8}"
    f"{'Val':>8}"
    f"{'Test':>8}"
)

for indice, clase in enumerate(dataset.clases):
    print(
        f"{clase:<8}"
        f"{etiquetas.count(indice):>8}"
        f"{contador_train[indice]:>8}"
        f"{contador_val[indice]:>8}"
        f"{contador_test[indice]:>8}"
    )