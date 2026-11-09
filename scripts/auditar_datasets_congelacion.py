
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# Extensiones de imágenes admitidas por el proyecto.
EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}

# Fuentes oficiales utilizadas actualmente en la tesis.
SOURCES = ("Clases_200", "Clases_200_comp")


def load_classes(project_root: Path) -> list[str]:
    """Carga y valida el vocabulario de clases del proyecto."""

    path = project_root / "src/core/configuracion/clases.json"

    with path.open("r", encoding="utf-8") as f:
        classes = json.load(f)

    if (
        not isinstance(classes, list)
        or not classes
        or len(classes) != len(set(classes))
    ):
        raise ValueError(f"Vocabulario inválido en {path}")

    if not all(isinstance(name, str) and name for name in classes):
        raise ValueError(
            f"Todas las clases deben ser nombres de texto no vacíos: {path}"
        )

    return classes


def contains_images(folder: Path) -> bool:
    """Comprueba si una carpeta contiene imágenes admitidas."""

    try:
        return any(
            p.is_file() and p.suffix.casefold() in EXTENSIONS
            for p in folder.iterdir()
        )
    except OSError:
        return False


def audit(project_root: Path) -> dict:
    """Audita las fuentes oficiales sin modificar los datasets."""

    data_root = project_root / "data"

    classes = load_classes(project_root)
    class_set = set(classes)

    hashes = defaultdict(list)
    source_reports = {}

    corrupt = []
    unreadable = []
    unknown_dirs = []

    candidate_total = 0
    valid_total = 0

    for source in SOURCES:
        root = data_root / source

        if not root.is_dir():
            raise FileNotFoundError(
                f"No se encontró la fuente requerida: {root}"
            )

        counts = Counter({name: 0 for name in classes})

        candidates = 0
        valid = 0
        corrupt_count = 0

        # Detectar directorios con imágenes que no pertenecen
        # al vocabulario configurado.
        for child in root.iterdir():
            if (
                child.is_dir()
                and child.name not in class_set
                and contains_images(child)
            ):
                unknown_dirs.append({
                    "source": source,
                    "directory": child.name
                })

        # Recorrer las clases respetando el orden de clases.json.
        for class_name in classes:
            class_dir = root / class_name

            if not class_dir.is_dir():
                continue

            for path in sorted(
                class_dir.iterdir(),
                key=lambda p: p.name.casefold()
            ):
                if (
                    not path.is_file()
                    or path.suffix.casefold() not in EXTENSIONS
                ):
                    continue

                candidates += 1
                candidate_total += 1

                relative = path.relative_to(project_root).as_posix()

                try:
                    raw = path.read_bytes()

                except OSError as exc:
                    unreadable.append({
                        "source": source,
                        "class": class_name,
                        "path": relative,
                        "error": str(exc)
                    })
                    continue

                # Identificación exacta mediante SHA-256.
                digest = hashlib.sha256(raw).hexdigest()

                # Comprobar que OpenCV pueda decodificar la imagen.
                decoded = cv2.imdecode(
                    np.frombuffer(raw, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )

                if decoded is None:
                    corrupt_count += 1

                    corrupt.append({
                        "source": source,
                        "class": class_name,
                        "path": relative,
                        "sha256": digest
                    })

                    continue

                valid += 1
                valid_total += 1
                counts[class_name] += 1

                hashes[digest].append({
                    "source": source,
                    "class": class_name,
                    "path": relative
                })

        missing = [
            name for name in classes
            if counts[name] == 0
        ]

        source_reports[source] = {
            "candidate_files": candidates,
            "valid_images": valid,
            "corrupt_images": corrupt_count,
            "class_counts": dict(counts),
            "missing_classes": missing,
            "class_coverage": len(classes) - len(missing),
            "total_classes": len(classes)
        }

    # Identificar duplicados exactos y conflictos de etiquetas.
    duplicates = []
    cross_source = []
    cross_label = []

    for digest, records in hashes.items():

        if len(records) < 2:
            continue

        item = {
            "sha256": digest,
            "copy_count": len(records),
            "copies": records
        }

        duplicates.append(item)

        if len({r["source"] for r in records}) > 1:
            cross_source.append(item)

        if len({r["class"] for r in records}) > 1:
            cross_label.append(item)

    return {
        "metadata": {
            "generated_at": datetime.now().astimezone().isoformat(),
            "project_root": str(project_root),
            "sources": list(SOURCES),
            "classes": classes,
            "class_count": len(classes),
            "read_only_for_datasets": True,
            "scope": (
                "Solo Clases_200 y Clases_200_comp; "
                "otros directorios de data/ no se auditan."
            )
        },

        "summary": {
            "candidate_files": candidate_total,
            "valid_images": valid_total,
            "corrupt_images": len(corrupt),
            "unreadable_files": len(unreadable),
            "exact_duplicate_groups": len(duplicates),
            "cross_source_duplicate_groups": len(cross_source),
            "cross_label_conflict_groups": len(cross_label),
            "unknown_class_directories": len(unknown_dirs)
        },

        "sources": source_reports,
        "duplicates": duplicates,
        "cross_source_duplicates": cross_source,
        "cross_label_conflicts": cross_label,
        "corrupt_images": corrupt,
        "unreadable_files": unreadable,
        "unknown_class_directories": unknown_dirs
    }


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Auditoría de solo lectura de "
            "Clases_200 y Clases_200_comp."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None
    )

    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()

    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "results/dataset_audit"
    )

    report = audit(root)

    output.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output / f"auditoria_dataset_{stamp}.json"
    csv_path = output / f"conteo_por_clase_{stamp}.csv"

    # Informe completo.
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Conteo por fuente y clase.
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["source", "class", "valid_images"]
        )

        writer.writeheader()

        for source, details in report["sources"].items():

            for class_name, count in details["class_counts"].items():

                writer.writerow({
                    "source": source,
                    "class": class_name,
                    "valid_images": count
                })

    # Resumen en consola.
    print("\n=== AUDITORÍA DE DATASETS ===")

    for key, value in report["summary"].items():
        print(f"{key}: {value}")

    for source, details in report["sources"].items():

        print(
            f"\n[{source}] "
            f"{details['valid_images']} imágenes válidas; "
            f"cobertura "
            f"{details['class_coverage']}/"
            f"{details['total_classes']}"
        )

        if details["missing_classes"]:
            print(
                "Clases sin imágenes:",
                ", ".join(details["missing_classes"])
            )

    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")

    # Código de salida 1 si existen problemas relevantes.
    issues = any(
        report["summary"][key]
        for key in (
            "corrupt_images",
            "unreadable_files",
            "cross_label_conflict_groups",
            "unknown_class_directories"
        )
    )

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
