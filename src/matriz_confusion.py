from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from configuracion import MODELO


# ============================================================
# Rutas
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT_DIR / "results" / MODELO

RUTA_PREDICCIONES = (
    RESULTS_DIR / "predicciones.csv"
)

RUTA_SALIDA = (
    RESULTS_DIR / "matriz_confusion.png"
)


# ============================================================
# Cargar predicciones
# ============================================================

if not RUTA_PREDICCIONES.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo: {RUTA_PREDICCIONES}"
    )

datos = pd.read_csv(RUTA_PREDICCIONES)


# ============================================================
# Clases
# ============================================================

nombres_clases = sorted(
    datos["real"].unique(),
    key=lambda clase: (
        0 if clase.isdigit() else 1,
        clase
    )
)


# ============================================================
# Matriz de confusión
# ============================================================

matriz = confusion_matrix(
    datos["real"],
    datos["prediccion"],
    labels=nombres_clases
)


print("\n" + "=" * 60)
print("MATRIZ DE CONFUSIÓN")
print("=" * 60)

print("\nClases:")
print(nombres_clases)


# ============================================================
# Visualización
# ============================================================

plt.figure(figsize=(16, 14))

sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=nombres_clases,
    yticklabels=nombres_clases,
    linewidths=0.5,
    linecolor="white",
    cbar=True
)

plt.xlabel("Clase predicha")
plt.ylabel("Clase real")
plt.title(f"Matriz de confusión - {MODELO}")

plt.tight_layout()

plt.savefig(
    RUTA_SALIDA,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(f"\nMatriz de confusión guardada en:")
print(RUTA_SALIDA)