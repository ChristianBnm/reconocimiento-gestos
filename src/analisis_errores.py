import shutil
import pandas as pd

from pathlib import Path
from configuracion import MODELO


# ============================================================
# Rutas
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT_DIR / "results" / MODELO

RUTA_PREDICCIONES = (
    RESULTS_DIR / "predicciones.csv"
)

ERRORS_DIR = RESULTS_DIR / "errores"

ERRORS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Cargar predicciones
# ============================================================

if not RUTA_PREDICCIONES.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo: {RUTA_PREDICCIONES}"
    )

datos = pd.read_csv(RUTA_PREDICCIONES)


# ============================================================
# Filtrar errores
# ============================================================

errores = datos[
    datos["real"] != datos["prediccion"]
]


# ============================================================
# Mostrar resultados
# ============================================================

print("\n" + "=" * 60)
print("ANÁLISIS DE ERRORES")
print("=" * 60)

print(f"\nCantidad de errores: {len(errores)}")


# ============================================================
# Guardar imágenes de los errores
# ============================================================

for _, error in errores.iterrows():

    ruta_imagen = Path(error["ruta"])

    nombre_imagen = ruta_imagen.name

    nombre_destino = (
        f"real_{error['real']}"
        f"_pred_{error['prediccion']}"
        f"_{nombre_imagen}"
    )

    ruta_destino = ERRORS_DIR / nombre_destino

    shutil.copy2(
        ruta_imagen,
        ruta_destino
    )

    print(
        f"\nÍndice: {int(error['indice'])}"
    )

    print(
        f"Real: {error['real']} "
        f"-> Predicción: {error['prediccion']}"
    )

    print(
        f"Confianza: "
        f"{float(error['confianza']) * 100:.2f}%"
    )

    print(
        f"Imagen: {ruta_imagen}"
    )

    print(
        f"Copia guardada en: {ruta_destino}"
    )