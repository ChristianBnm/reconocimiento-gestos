#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Construye dataset_definitivo a partir de Clases_Extend.

- Espera 37 carpetas de clase: 0-9, A-Z y _.
- Elimina duplicados exactos dentro de cada clase.
- Extrae embeddings con ResNet-50 preentrenada en ImageNet (solo para seleccionar).
- Agrupa cada clase en N clusters (200 por defecto) y elige un representante por cluster.
- N se controla con --count (por ejemplo, 100, 200 o 300).
- Informa la similitud perceptual de las imágenes seleccionadas; no fuerza cuotas de distancias.
- Redimensiona las imágenes seleccionadas a 224x224 y aplica una mejora suave.
- No genera aumentaciones artificiales.
- No modifica la fuente original.

Dependencias:
    pip install opencv-python numpy pandas scikit-learn torch torchvision tqdm

Primera ejecución:
    torchvision puede descargar los pesos preentrenados de ResNet-50.
    Se recomienda ejecutar con conexión a Internet la primera vez.

Ejemplo:
    python construir_dataset_definitivo.py ^
        --input "E:/Mis Documentos/Christian/Tesis/tesis_gestos/data/Clases_Extend" ^
        --output "E:/Mis Documentos/Christian/Tesis/tesis_gestos/data/dataset_definitivo" ^
        --count 200

El directorio de salida debe NO existir previamente.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm


CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(ord("A"), ord("Z") + 1)] + ["_"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_quality_score(image_rgb: np.ndarray) -> float:
    """
    Puntaje auxiliar, NO un umbral de descarte.
    Combina nitidez y contraste moderados. Se usa solo para desempatar
    candidatos cercanos al centro de un cluster.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    # Escalas suaves para evitar que la nitidez domine completamente.
    return float(np.log1p(max(sharpness, 0.0)) + 0.15 * np.log1p(max(contrast, 0.0)))


def enhance_image(image_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """Replica de forma conservadora la mejora del script anterior."""
    image = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_CUBIC)

    # CLAHE moderado sobre luminancia en espacio LAB.
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    image = cv2.cvtColor(
        cv2.merge((l_channel, a_channel, b_channel)),
        cv2.COLOR_LAB2RGB,
    )

    # Denoising y enfoque moderados. No se aplica blur artificial,
    # motion blur, ruido, flip ni rotaciones en el dataset definitivo.
    image = cv2.fastNlMeansDenoisingColored(
        image, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21
    )
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
    image = cv2.addWeighted(image, 1.6, blurred, -0.6, 0)

    # Gamma suave, equivalente al criterio del script original.
    gamma = 1.05
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    image = cv2.LUT(image, table)
    return np.clip(image, 0, 255).astype(np.uint8)



def perceptual_hash(image_rgb: np.ndarray) -> np.ndarray:
    """pHash compacto de 64 bits para describir similitud perceptual aproximada."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low_freq = dct[:8, :8].copy()
    values = low_freq.flatten()
    median = np.median(values[1:])  # excluye el componente DC
    return values > median


def selected_similarity_summary(selected_items: list[tuple], max_pairs: int = 50000) -> dict:
    """
    Resume distancias pHash de los seleccionados. No elimina pares ni impone
    cuotas: sirve para auditar cuánta similitud queda en la selección.
    """
    hashes = [perceptual_hash(item[2]) for item in selected_items]
    n = len(hashes)
    if n < 2:
        return {"pairs": 0, "min_distance": None, "median_distance": None,
                "mean_distance": None, "pairs_distance_le_2": 0,
                "pairs_distance_le_4": 0}
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(int(np.count_nonzero(hashes[i] != hashes[j])))
            if len(distances) >= max_pairs:
                break
        if len(distances) >= max_pairs:
            break
    arr = np.asarray(distances, dtype=np.int16)
    return {
        "pairs": int(len(arr)),
        "min_distance": int(arr.min()),
        "median_distance": float(np.median(arr)),
        "mean_distance": float(arr.mean()),
        "pairs_distance_le_2": int(np.count_nonzero(arr <= 2)),
        "pairs_distance_le_4": int(np.count_nonzero(arr <= 4)),
        "interpretation": "Indicador descriptivo; no equivale a duplicados exactos ni a diversidad gestual."
    }


class EmbeddingExtractor:
    def __init__(self, device: str = "auto", batch_size: int = 32):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size

        weights = ResNet50_Weights.DEFAULT
        self.preprocess = weights.transforms()
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(self.device)

        print(f"Modelo de embeddings: ResNet-50 ImageNet | dispositivo: {self.device}")

    def extract(self, images: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        vectors = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = []
                for image in images[start : start + self.batch_size]:
                    pil_image = Image.fromarray(image).convert("RGB")
                    batch.append(self.preprocess(pil_image))
                tensor = torch.stack(batch).to(self.device)
                features = self.model(tensor)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                vectors.append(features.cpu().numpy())

        return np.concatenate(vectors, axis=0).astype(np.float32)


def discover_classes(
    input_dir: Path,
    requested_classes: list[str] | None = None,
) -> dict[str, list[Path]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de entrada: {input_dir}")

    found = {
        folder.name: sorted(
            file for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )
        for folder in input_dir.iterdir()
        if folder.is_dir()
    }

    selected_classes = requested_classes or CLASSES
    invalid = sorted(set(selected_classes) - set(CLASSES))
    missing = sorted(set(selected_classes) - set(found))
    if invalid or missing:
        raise ValueError(
            f"Clases inválidas: {invalid}. Carpetas faltantes: {missing}. "
            "Se esperan nombres 0-9, A-Z y _."
        )

    if requested_classes is None:
        unexpected = sorted(set(found) - set(CLASSES))
        if unexpected:
            raise ValueError(f"Carpetas inesperadas en el dataset: {unexpected}")

    return {class_name: found[class_name] for class_name in selected_classes}


def remove_exact_duplicates(paths: list[Path]):
    unique_paths = []
    seen_hashes = {}
    duplicate_rows = []

    for path in paths:
        file_hash = sha256_file(path)
        if file_hash in seen_hashes:
            duplicate_rows.append({
                "file": str(path),
                "duplicate_of": str(seen_hashes[file_hash]),
                "sha256": file_hash,
                "status": "duplicate_exact",
            })
            continue
        seen_hashes[file_hash] = path
        unique_paths.append((path, file_hash))

    return unique_paths, duplicate_rows


def select_class_representatives(
    class_name: str,
    records: list[tuple[Path, str]],
    extractor: EmbeddingExtractor,
    target_count: int,
    seed: int,
):
    if len(records) < target_count:
        raise ValueError(
            f"Clase {class_name}: quedan {len(records)} imágenes únicas; "
            f"se necesitan al menos {target_count}."
        )

    valid_records = []
    rgb_images = []
    rejected = []

    for path, file_hash in records:
        image = read_rgb(path)
        if image is None:
            rejected.append({
                "file": str(path),
                "sha256": file_hash,
                "status": "unreadable",
            })
            continue
        valid_records.append((path, file_hash))
        rgb_images.append(image)

    if len(valid_records) < target_count:
        raise ValueError(
            f"Clase {class_name}: después de descartar archivos ilegibles "
            f"quedan {len(valid_records)} imágenes; se necesitan {target_count}."
        )

    embeddings = extractor.extract(rgb_images)

    # KMeans entrega grupos visuales. Se elige un representante cercano
    # al centro de cada grupo; la calidad solo desempata entre candidatos
    # muy próximos al centro, sin descartar automáticamente imágenes.
    kmeans = KMeans(
        n_clusters=target_count,
        random_state=seed,
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_

    selected_indices = []
    cluster_distances = {}
    for cluster_id in range(target_count):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue

        distances = np.linalg.norm(embeddings[members] - centers[cluster_id], axis=1)
        order = np.argsort(distances)
        # Entre los cinco candidatos más cercanos, priorizar calidad moderada.
        top = members[order[: min(5, len(order))]]
        scores = [image_quality_score(rgb_images[i]) for i in top]
        best_local = int(np.argmax(scores))
        chosen = int(top[best_local])

        selected_indices.append(chosen)
        cluster_distances[chosen] = float(
            np.linalg.norm(embeddings[chosen] - centers[cluster_id])
        )

    if len(selected_indices) != target_count:
        raise RuntimeError(
            f"Clase {class_name}: se seleccionaron {len(selected_indices)} "
            f"representantes, se esperaban {target_count}."
        )

    selected_set = set(selected_indices)
    selection_rows = []
    selected = []

    for i, (path, file_hash) in enumerate(valid_records):
        is_selected = i in selected_set
        cluster_id = int(labels[i])
        selection_rows.append({
            "class": class_name,
            "file": str(path),
            "sha256": file_hash,
            "selected": is_selected,
            "cluster": cluster_id,
            "distance_to_cluster_center": (
                cluster_distances.get(i, float(
                    np.linalg.norm(embeddings[i] - centers[cluster_id])
                ))
            ),
            "quality_score": image_quality_score(rgb_images[i]),
            "status": "selected" if is_selected else "not_selected",
        })
        if is_selected:
            selected.append((path, file_hash, rgb_images[i], cluster_id))

    return selected, selection_rows, rejected


def main():
    parser = argparse.ArgumentParser(
        description="Selecciona 200 imágenes diversas por clase y genera dataset 224x224."
    )
    parser.add_argument("--input", required=True, help="Ruta a Clases_Extend")
    parser.add_argument("--output", required=True, help="Ruta nueva para dataset_definitivo")
    parser.add_argument("--count", type=int, default=200, help="Imágenes por clase: 100, 200, 300 u otro entero (default: 200)")
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=CLASSES,
        default=None,
        help="Procesar solo estas clases. Ejemplo: --classes Q. Si se omite, procesa las 37.",
    )
    parser.add_argument("--size", type=int, default=224, help="Tamaño de salida (default: 224)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de KMeans")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Solo redimensionar; no aplicar CLAHE/denoising/unsharp/gamma.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if output_dir.exists():
        raise FileExistsError(
            f"La carpeta de salida ya existe: {output_dir}\n"
            "Elegí una carpeta nueva. El script no sobrescribe resultados."
        )
    if args.count < 1 or args.size < 1:
        raise ValueError("--count y --size deben ser mayores que cero.")

    selected_classes = args.classes or CLASSES
    class_files = discover_classes(input_dir, requested_classes=args.classes)
    total_source = sum(len(v) for v in class_files.values())
    print(f"Fuente: {input_dir}")
    print(f"Imágenes encontradas en las clases seleccionadas: {total_source}")
    print(f"Clases a procesar: {selected_classes}")
    print(f"Objetivo: {args.count} imágenes por clase")

    # Comprobar colisiones exactas entre clases: no se corrigen silenciosamente.
    hash_to_classes = defaultdict(set)
    for class_name, paths in class_files.items():
        for path in paths:
            try:
                hash_to_classes[sha256_file(path)].add(class_name)
            except OSError:
                pass
    cross_class_duplicates = [
        {"sha256": h, "classes": sorted(classes)}
        for h, classes in hash_to_classes.items()
        if len(classes) > 1
    ]

    extractor = EmbeddingExtractor(device=args.device, batch_size=args.batch_size)

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir.parent / f"{output_dir.name}_manifest.csv"
    duplicates_path = output_dir.parent / f"{output_dir.name}_duplicados.csv"
    report_path = output_dir.parent / f"{output_dir.name}_informe.json"

    manifest_rows = []
    duplicate_rows_all = []
    rejected_rows_all = []
    summary = {}

    try:
        for class_name in selected_classes:
            print(f"\nProcesando clase {class_name}...")
            records, duplicate_rows = remove_exact_duplicates(class_files[class_name])
            duplicate_rows_all.extend(
                [{"class": class_name, **row} for row in duplicate_rows]
            )

            selected, selection_rows, rejected = select_class_representatives(
                class_name=class_name,
                records=records,
                extractor=extractor,
                target_count=args.count,
                seed=args.seed,
            )
            similarity_stats = selected_similarity_summary(selected)
            rejected_rows_all.extend(
                [{"class": class_name, **row} for row in rejected]
            )

            class_output = output_dir / class_name
            class_output.mkdir(parents=True, exist_ok=True)

            for path, file_hash, original_rgb, cluster_id in tqdm(
                selected, desc=f"Guardando {class_name}", unit="img"
            ):
                processed = (
                    cv2.resize(
                        original_rgb, (args.size, args.size),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    if args.no_enhance
                    else enhance_image(original_rgb, args.size)
                )
                output_name = f"{path.stem}_selected.jpg"
                output_path = class_output / output_name
                bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                ok = cv2.imwrite(
                    str(output_path),
                    bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )
                if not ok:
                    raise IOError(f"No se pudo guardar: {output_path}")

                manifest_rows.append({
                    "class": class_name,
                    "source_file": str(path),
                    "output_file": str(output_path),
                    "source_sha256": file_hash,
                    "cluster": cluster_id,
                    "output_width": args.size,
                    "output_height": args.size,
                    "enhancement": not args.no_enhance,
                })

            for row in selection_rows:
                manifest_rows.append({
                    "class": row["class"],
                    "source_file": row["file"],
                    "output_file": "",
                    "source_sha256": row["sha256"],
                    "selected": row["selected"],
                    "cluster": row["cluster"],
                    "distance_to_cluster_center": row["distance_to_cluster_center"],
                    "quality_score": row["quality_score"],
                    "status": row["status"],
                })

            summary[class_name] = {
                "source_images": len(class_files[class_name]),
                "unique_images": len(records),
                "selected_images": len(selected),
                "exact_duplicates_removed": len(duplicate_rows),
                "unreadable_images": len(rejected),
                "selected_perceptual_similarity": similarity_stats,
            }

    except Exception:
        # Se conserva la carpeta parcial para diagnóstico; nunca se altera la fuente.
        raise

    # Manifest con filas de selección y filas de salida. El campo status
    # permite distinguir la selección de los archivos finales.
    manifest_fields = sorted({key for row in manifest_rows for key in row.keys()})
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    duplicate_fields = ["class", "file", "duplicate_of", "sha256", "status"]
    with duplicates_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=duplicate_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(duplicate_rows_all)

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "classes": len(selected_classes),
        "processed_classes": selected_classes,
        "target_per_class": args.count,
        "target_total": len(selected_classes) * args.count,
        "output_size": [args.size, args.size],
        "enhancement_applied": not args.no_enhance,
        "selection_model": "ResNet-50 pretrained on ImageNet; embeddings only",
        "clustering": "KMeans; one representative per cluster; count controlled by --count",
        "selection_policy": (
            "Representativeness with controlled redundancy; no forced pHash distance quotas. "
            "pHash statistics are reported for audit only."
        ),
        "random_seed": args.seed,
        "cross_class_exact_duplicate_hashes": cross_class_duplicates,
        "class_summary": summary,
        "note": (
            "El redimensionamiento no recupera detalle perdido en las imágenes 50x50. "
            "La selección por embeddings es un criterio de diversidad visual, "
            "no una validación lingüística de las señas."
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nProceso terminado.")
    print(f"Dataset: {output_dir}")
    print(f"Manifiesto: {manifest_path}")
    print(f"Duplicados: {duplicates_path}")
    print(f"Informe: {report_path}")
    print(f"Total esperado: {len(selected_classes) * args.count} imágenes.")


if __name__ == "__main__":
    main()
