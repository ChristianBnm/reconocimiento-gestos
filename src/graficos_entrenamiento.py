from pathlib import Path
from configuracion import MODELO

import json
import matplotlib.pyplot as plt


# ============================================================
# Rutas
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT_DIR / "results" / MODELO

RUTA_HISTORIAL = RESULTS_DIR / "history.json"


# ============================================================
# Cargar historial
# ============================================================

if not RUTA_HISTORIAL.exists():
    raise FileNotFoundError(
        f"No se encontró el historial: {RUTA_HISTORIAL}"
    )

with open(
    RUTA_HISTORIAL,
    "r",
    encoding="utf-8"
) as archivo:

    historial = json.load(archivo)


# ============================================================
# Extraer datos
# ============================================================

epocas = [
    registro["epoca"]
    for registro in historial
]

train_loss = [
    registro["train_loss"]
    for registro in historial
]

val_loss = [
    registro["val_loss"]
    for registro in historial
]

train_accuracy = [
    registro["train_accuracy"]
    for registro in historial
]

val_accuracy = [
    registro["val_accuracy"]
    for registro in historial
]


# ============================================================
# Gráfico de Loss
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epocas,
    train_loss,
    label="Train Loss"
)

plt.plot(
    epocas,
    val_loss,
    label="Validation Loss"
)

plt.xlabel("Época")
plt.ylabel("Loss")
plt.title(f"Loss durante el entrenamiento - {MODELO}")

plt.legend()
plt.grid(True)

plt.tight_layout()


ruta_loss = RESULTS_DIR / "loss.png"


plt.savefig(
    ruta_loss,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Gráfico de Accuracy
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epocas,
    train_accuracy,
    label="Train Accuracy"
)

plt.plot(
    epocas,
    val_accuracy,
    label="Validation Accuracy"
)

plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.title(f"Accuracy durante el entrenamiento - {MODELO}")

plt.legend()
plt.grid(True)

plt.tight_layout()


ruta_accuracy = RESULTS_DIR / "accuracy.png"


plt.savefig(
    ruta_accuracy,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Resultados
# ============================================================

print("\n" + "=" * 60)
print("GRÁFICOS DE ENTRENAMIENTO")
print("=" * 60)

print(f"\nLoss guardado en:")
print(ruta_loss)

print(f"\nAccuracy guardado en:")
print(ruta_accuracy)