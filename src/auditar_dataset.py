import hashlib

from pathlib import Path

# ============================================================
# Rutas de los datasets
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

DATASETS = {
    "Clases_200": DATA_DIR / "Clases_200",
    "Clases_200_comp": DATA_DIR / "Clases_200_comp",
}

# ============================================================
# Extensiones de imagen admitidas
# ============================================================

EXTENSIONES_IMAGEN = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

def contar_imagenes_por_clase(ruta_dataset):
    conteo = {}

    for carpeta_clase in sorted(ruta_dataset.iterdir()):

        if not carpeta_clase.is_dir():
            continue

        cantidad = sum(
            1
            for archivo in carpeta_clase.iterdir()
            if archivo.is_file()
            and archivo.suffix.lower() in EXTENSIONES_IMAGEN
        )

        conteo[carpeta_clase.name] = cantidad

    return conteo

for nombre_dataset, ruta_dataset in DATASETS.items():

    print("\n" + "=" * 60)
    print(f"DATASET: {nombre_dataset}")
    print("=" * 60)

    conteo = contar_imagenes_por_clase(ruta_dataset)

    total = 0

    for clase, cantidad in conteo.items():
        print(f"{clase:>3} → {cantidad:>5}")

        total += cantidad

    print("-" * 60)
    print(f"TOTAL → {total}")


# ============================================================
# Muestras de nombres de archivos
# ============================================================

def mostrar_muestras_nombres(ruta_dataset, muestras_por_clase=5):
    for carpeta_clase in sorted(ruta_dataset.iterdir()):

        if not carpeta_clase.is_dir():
            continue

        archivos = sorted(
            archivo
            for archivo in carpeta_clase.iterdir()
            if archivo.is_file()
            and archivo.suffix.lower() in EXTENSIONES_IMAGEN
        )

        print(f"\nClase: {carpeta_clase.name}")

        for archivo in archivos[:muestras_por_clase]:
            print(f"  {archivo.name}")


# ============================================================
# Mostrar nombres representativos
# ============================================================

for nombre_dataset, ruta_dataset in DATASETS.items():

    print("\n" + "=" * 60)
    print(f"NOMBRES DE ARCHIVOS: {nombre_dataset}")
    print("=" * 60)

    mostrar_muestras_nombres(ruta_dataset)


# ============================================================
# Hash SHA-256
# ============================================================

def calcular_hash(ruta_archivo):
    sha256 = hashlib.sha256()

    with open(ruta_archivo, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            sha256.update(bloque)

    return sha256.hexdigest()

# ============================================================
# Buscar duplicados
# ============================================================

def buscar_duplicados(ruta_dataset):
    hashes = {}

    for carpeta_clase in sorted(ruta_dataset.iterdir()):

        if not carpeta_clase.is_dir():
            continue

        for archivo in sorted(carpeta_clase.iterdir()):

            if not archivo.is_file():
                continue

            if archivo.suffix.lower() not in EXTENSIONES_IMAGEN:
                continue

            hash_archivo = calcular_hash(archivo)

            if hash_archivo not in hashes:
                hashes[hash_archivo] = []

            hashes[hash_archivo].append(archivo)

    duplicados = {
        hash_archivo: rutas
        for hash_archivo, rutas in hashes.items()
        if len(rutas) > 1
    }

    return duplicados


# ============================================================
# Informe de duplicados
# ============================================================

for nombre_dataset, ruta_dataset in DATASETS.items():

    print("\n" + "=" * 60)
    print(f"DUPLICADOS: {nombre_dataset}")
    print("=" * 60)

    duplicados = buscar_duplicados(ruta_dataset)

    if not duplicados:
        print("No se encontraron duplicados exactos.")
        continue

    cantidad_grupos = len(duplicados)

    cantidad_archivos = sum(
        len(rutas)
        for rutas in duplicados.values()
    )

    print(f"Grupos de duplicados: {cantidad_grupos}")
    print(f"Archivos involucrados: {cantidad_archivos}")

    for hash_archivo, rutas in duplicados.items():

        print(f"\nHash: {hash_archivo}")

        for ruta in rutas:
            print(f"  {ruta}")