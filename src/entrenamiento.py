import torch
from pathlib import Path
from modelos import crear_modelo, congelar_backbone
from torch.utils.data import DataLoader
from division_dataset import dividir_dataset

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
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS
)

# ============================================================
# Rutas
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_BASE = ROOT_DIR / "data" / "Clases_200"
DATASET_COMPLEMENTARIO = ROOT_DIR / "data" / "Clases_200_comp"


# ============================================================
# Dataset
# ============================================================

dataset = GestosDataset(
    carpetas_imagenes=[
        str(DATASET_BASE),
        str(DATASET_COMPLEMENTARIO)
    ],
    transformacion=None
)


# ============================================================
# División
# ============================================================

indices_train, indices_val, indices_test = dividir_dataset(
    etiquetas=dataset.etiquetas,
    train_size=TRAIN_SIZE,
    val_size=VAL_SIZE,
    test_size=TEST_SIZE,
    seed=SEED
)


# ============================================================
# Subconjuntos
# ============================================================

dataset_train = DatasetSubset(
    dataset=dataset,
    indices=indices_train,
    transformacion=transformacion_train
)

dataset_val = DatasetSubset(
    dataset=dataset,
    indices=indices_val,
    transformacion=transformacion_eval
)

dataset_test = DatasetSubset(
    dataset=dataset,
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
# Información
# ============================================================

print("=" * 60)
print("PREPARACIÓN DEL ENTRENAMIENTO")
print("=" * 60)

print(f"\nDataset total: {len(dataset)}")
print(f"Train:         {len(dataset_train)}")
print(f"Validation:    {len(dataset_val)}")
print(f"Test:          {len(dataset_test)}")

print(f"\nBatch size: {BATCH_SIZE}")


# ============================================================
# Modelo
# ============================================================

from modelos import crear_modelo

modelo = crear_modelo(
    arquitectura="mobilenetv2_100",
    num_classes=37,
    pretrained=True
)

modelo = congelar_backbone(modelo)

# Función de pérdida
criterio = torch.nn.CrossEntropyLoss()

# Optimizador
optimizador = torch.optim.AdamW(
    modelo.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

def congelar_backbone(modelo):

    for parametro in modelo.parameters():
        parametro.requires_grad = False

    for parametro in modelo.get_classifier().parameters():
        parametro.requires_grad = True

    return modelo


# ============================================================
# Prueba del modelo
# ============================================================

parametros_totales = sum(
    parametro.numel()
    for parametro in modelo.parameters()
)

parametros_entrenables = sum(
    parametro.numel()
    for parametro in modelo.parameters()
    if parametro.requires_grad
)

print("\n--- Parámetros ---")
print(f"Totales:       {parametros_totales:,}")
print(f"Entrenables:   {parametros_entrenables:,}")

modelo.eval()

dispositivo = next(modelo.parameters()).device



print("\n--- Entrenamiento ---")
print(f"Loss:           CrossEntropyLoss")
print(f"Optimizador:    AdamW")
print(f"Learning rate:  {LEARNING_RATE}")
print(f"Weight decay:   {WEIGHT_DECAY}")
print(f"Épocas máximas: {EPOCHS}")