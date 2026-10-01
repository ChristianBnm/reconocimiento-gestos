import csv
import torch

from pathlib import Path
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from dataset_utils import (
    GestosDataset,
    DatasetSubset,
    transformacion_eval
)

from division_dataset import dividir_dataset

from configuracion import (
    MODELO,
    ARQUITECTURA,
    NUM_CLASSES,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    SEED,
    BATCH_SIZE
)

from modelos import crear_modelo


def obtener_nombres_clases(dataset):
    """
    Obtiene los nombres de las clases en el mismo orden
    utilizado por GestosDataset.
    """
    return dataset.clases


def evaluar_modelo(modelo, dataloader, dispositivo):
    """
    Obtiene etiquetas reales, predicciones y confianza
    para cada imagen del conjunto evaluado.
    """

    modelo.eval()

    etiquetas_reales = []
    predicciones = []
    confianzas = []

    with torch.no_grad():

        for imagenes, etiquetas in dataloader:

            imagenes = imagenes.to(dispositivo)

            salidas = modelo(imagenes)

            probabilidades = torch.softmax(salidas, dim=1)

            confianzas_batch, predicciones_batch = torch.max(
                probabilidades,
                dim=1
            )

            etiquetas_reales.extend(
                etiquetas.numpy()
            )

            predicciones.extend(
                predicciones_batch.cpu().numpy()
            )

            confianzas.extend(
                confianzas_batch.cpu().numpy()
            )

    return etiquetas_reales, predicciones, confianzas


# ============================================================
# Rutas
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_BASE = ROOT_DIR / "data" / "Clases_200"
DATASET_COMPLEMENTARIO = ROOT_DIR / "data" / "Clases_200_comp"

CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / MODELO
RESULTS_DIR = ROOT_DIR / "results" / MODELO

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ruta_checkpoint = CHECKPOINT_DIR / "mejor.pth"


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

nombres_clases = dataset.clases

print("\nClases:")
print(nombres_clases)

# ============================================================
# División del dataset
# ============================================================

indices_train, indices_val, indices_test = dividir_dataset(
    etiquetas=dataset.etiquetas,
    train_size=TRAIN_SIZE,
    val_size=VAL_SIZE,
    test_size=TEST_SIZE,
    seed=SEED
)


# ============================================================
# Dataset de Test
# ============================================================

dataset_test = DatasetSubset(
    dataset=dataset,
    indices=indices_test,
    transformacion=transformacion_eval
)

test_loader = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Modelo
# ============================================================

modelo = crear_modelo(
    arquitectura = ARQUITECTURA,
    num_classes = NUM_CLASSES,
    pretrained = False
)

dispositivo = next(modelo.parameters()).device


# ============================================================
# Cargar checkpoint
# ============================================================

if not ruta_checkpoint.exists():
    raise FileNotFoundError(
        f"No se encontró el checkpoint: {ruta_checkpoint}"
    )

modelo.load_state_dict(
    torch.load(
        ruta_checkpoint,
        map_location=dispositivo,
        weights_only=True
    )
)

modelo.to(dispositivo)
modelo.eval()


# ============================================================
# Evaluación
# ============================================================

etiquetas_reales, predicciones, confianzas = evaluar_modelo(
    modelo=modelo,
    dataloader=test_loader,
    dispositivo=dispositivo
)


# ============================================================
# Guardar predicciones
# ============================================================

ruta_predicciones = RESULTS_DIR / "predicciones.csv"

with open(
    ruta_predicciones,
    "w",
    newline="",
    encoding="utf-8"
) as archivo:

    escritor = csv.writer(archivo)

    escritor.writerow([
        "indice",
        "ruta",
        "real",
        "prediccion",
        "confianza"
    ])

    for indice, (real, prediccion, confianza) in enumerate(
        zip(
            etiquetas_reales,
            predicciones,
            confianzas
        )
    ):

        indice_original = indices_test[indice]

        ruta_imagen = dataset.rutas_imagenes[indice_original]

        escritor.writerow([
            indice_original,
            ruta_imagen,
            nombres_clases[real],
            nombres_clases[prediccion],
            float(confianza)
        ])

print(f"\nPredicciones guardadas en:")
print(ruta_predicciones)


# ============================================================
# Métricas
# ============================================================

accuracy = accuracy_score(
    etiquetas_reales,
    predicciones
)

precision = precision_score(
    etiquetas_reales,
    predicciones,
    average="macro",
    zero_division=0
)

recall = recall_score(
    etiquetas_reales,
    predicciones,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    etiquetas_reales,
    predicciones,
    average="macro",
    zero_division=0
)


print("\n" + "=" * 60)
print("RESULTADOS SOBRE TEST")
print("=" * 60)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\n" + "=" * 60)