from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from src.core.dataset.loader import IMAGE_EXTENSIONS


def calcular_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_dataset(root: Path):
    records = []
    if not root.exists():
        raise FileNotFoundError(root)

    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(class_dir.iterdir(), key=lambda p: p.name.lower()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append((class_dir.name, path))
    return records


def audit_dataset(root: Path) -> dict:
    records = scan_dataset(root)
    counts = {}
    unreadable = []
    hashes = {}

    for class_name, path in records:
        counts[class_name] = counts.get(class_name, 0) + 1

        image = cv2.imread(str(path))
        if image is None:
            unreadable.append(str(path))

        digest = calcular_hash(path)
        hashes.setdefault(digest, []).append(str(path))

    duplicates = {
        digest: paths
        for digest, paths in hashes.items()
        if len(paths) > 1
    }

    return {
        "dataset": str(root),
        "total": len(records),
        "counts": counts,
        "unreadable": unreadable,
        "duplicate_groups": len(duplicates),
        "duplicate_files": sum(len(paths) for paths in duplicates.values()),
        "duplicates": duplicates,
    }


def main():
    parser = argparse.ArgumentParser(description="Audita datasets sin modificarlos.")
    parser.add_argument("dataset", nargs="+", help="Una o más carpetas de dataset.")
    parser.add_argument("--json", dest="json_output", help="Ruta para guardar informe JSON.")
    args = parser.parse_args()

    reports = [audit_dataset(Path(root).resolve()) for root in args.dataset]

    for report in reports:
        print("=" * 60)
        print(f"DATASET: {report['dataset']}")
        print(f"TOTAL: {report['total']}")
        print(f"GRUPOS DUPLICADOS: {report['duplicate_groups']}")
        print(f"ARCHIVOS INVOLUCRADOS: {report['duplicate_files']}")
        print(f"IMÁGENES NO LEGIBLES: {len(report['unreadable'])}")

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(reports, file, indent=2, ensure_ascii=False)
        print(f"Informe JSON: {output}")


if __name__ == "__main__":
    main()
