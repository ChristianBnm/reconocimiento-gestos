import json
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
    MODELO,
    ARQUITECTURA,
    NUM_CLASSES,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE
)

# ============================================================
# Rutas
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_BASE = ROOT_DIR / "data" / "Clases_200"
DATASET_COMPLEMENTARIO = ROOT_DIR / "data" / "Clases_200_comp"

CHECKPOINT_DIR = (
    ROOT_DIR / "checkpoints" / MODELO
)

RESULTS_DIR = (
    ROOT_DIR / "results" / MODELO
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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


def establecer_semilla(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

establecer_semilla(SEED)

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
# Epocas
# ============================================================

def entrenar_epoca(modelo, dataloader, criterio, optimizador, dispositivo):
    """
    Ejecuta una época completa de entrenamiento.

    Devuelve:
        loss_promedio: pérdida promedio de la época.
        accuracy: porcentaje de predicciones correctas.
    """
    modelo.train()

    perdida_total = 0.0
    predicciones_correctas = 0
    total_imagenes = 0

    for imagenes, etiquetas in dataloader:

        imagenes = imagenes.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)

        # Reiniciar gradientes
        optimizador.zero_grad()

        # Forward
        salidas = modelo(imagenes)

        # Calcular pérdida
        perdida = criterio(salidas, etiquetas)

        # Backward
        perdida.backward()

        # Actualizar parámetros
        optimizador.step()

        # Acumular pérdida
        perdida_total += perdida.item() * imagenes.size(0)

        # Calcular predicciones
        _, predicciones = torch.max(salidas, 1)

        predicciones_correctas += (
            predicciones == etiquetas
        ).sum().item()

        total_imagenes += etiquetas.size(0)

    loss_promedio = perdida_total / total_imagenes
    accuracy = predicciones_correctas / total_imagenes

    return loss_promedio, accuracy

def validar_epoca(modelo, dataloader, criterio, dispositivo):
    """
    Evalúa el modelo sobre un conjunto de validación.

    No modifica los parámetros del modelo.

    Devuelve:
        loss_promedio: pérdida promedio de validación.
        accuracy: porcentaje de predicciones correctas.
    """
    modelo.eval()

    perdida_total = 0.0
    predicciones_correctas = 0
    total_imagenes = 0

    with torch.no_grad():

        for imagenes, etiquetas in dataloader:

            imagenes = imagenes.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)

            # Forward
            salidas = modelo(imagenes)

            # Calcular pérdida
            perdida = criterio(salidas, etiquetas)

            # Acumular pérdida
            perdida_total += perdida.item() * imagenes.size(0)

            # Calcular predicciones
            _, predicciones = torch.max(salidas, 1)

            predicciones_correctas += (
                predicciones == etiquetas
            ).sum().item()

            total_imagenes += etiquetas.size(0)

    loss_promedio = perdida_total / total_imagenes
    accuracy = predicciones_correctas / total_imagenes

    return loss_promedio, accuracy

# ============================================================
# Modelo
# ============================================================

modelo = crear_modelo(
    arquitectura = ARQUITECTURA,
    num_classes = NUM_CLASSES,
    pretrained = True
)

modelo = congelar_backbone(modelo)
dispositivo = next(modelo.parameters()).device

# Función de pérdida
criterio = torch.nn.CrossEntropyLoss()

# Optimizador
optimizador = torch.optim.AdamW(
    modelo.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("\n" + "=" * 60)
print("INICIO DEL ENTRENAMIENTO")
print("=" * 60)

mejor_val_loss = float("inf")
epocas_sin_mejora = 0

historial = []

for epoca in range(1, EPOCHS + 1):

    loss_train, accuracy_train = entrenar_epoca(
        modelo=modelo,
        dataloader=train_loader,
        criterio=criterio,
        optimizador=optimizador,
        dispositivo=dispositivo
    )

    loss_val, accuracy_val = validar_epoca(
        modelo=modelo,
        dataloader=val_loader,
        criterio=criterio,
        dispositivo=dispositivo
    )

    historial.append({
        "epoca": epoca,
        "train_loss": loss_train,
        "train_accuracy": accuracy_train,
        "val_loss": loss_val,
        "val_accuracy": accuracy_val
    })

    print(
        f"\nÉpoca {epoca}/{EPOCHS}"
    )

    print(
        f"Train      | "
        f"Loss: {loss_train:.4f} | "
        f"Accuracy: {accuracy_train:.4f}"
    )

    print(
        f"Validation | "
        f"Loss: {loss_val:.4f} | "
        f"Accuracy: {accuracy_val:.4f}"
    )

    # Comprobar mejora en validation
    if loss_val < mejor_val_loss:

        mejor_val_loss = loss_val
        epocas_sin_mejora = 0

        ruta_checkpoint = CHECKPOINT_DIR / "mejor.pth"

        torch.save(
            modelo.state_dict(),
            ruta_checkpoint
        )

        print("✓ Nuevo mejor modelo guardado.")

    else:

        epocas_sin_mejora += 1

        print(
            f"Sin mejora durante "
            f"{epocas_sin_mejora}/{PATIENCE} épocas."
        )

    # Early Stopping
    if epocas_sin_mejora >= PATIENCE:

        print("\nEarly Stopping activado.")
        break

# ============================================================
# Guardar historial
# ============================================================

ruta_historial = RESULTS_DIR / "history.json"

with open(
    ruta_historial,
    "w",
    encoding="utf-8"
) as archivo:

    json.dump(
        historial,
        archivo,
        indent=4
    )

print(
    f"\nHistorial guardado en: {ruta_historial}"
)