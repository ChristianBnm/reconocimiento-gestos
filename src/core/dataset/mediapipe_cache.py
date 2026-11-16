from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

from src.core.dataset.split import dataset_fingerprint

MEDIAPIPE_PIPELINES = {
    "mediapipe_roi",
    "mediapipe_roi_normalized",
    "mediapipe_roi_canny",
    "mediapipe_roi_rgb_edges",
}

FIELDS = [
    "fuente", "particion", "indice_dataset", "clase", "ruta",
    "detectada", "x1", "y1", "x2", "y2", "motivo_exclusion",
]


def _load_manifest(path: Path) -> dict[int, dict]:
    """Carga tanto detecciones positivas como negativas para poder reanudar."""
    cached: dict[int, dict] = {}
    if not path.is_file():
        return cached
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            try:
                cached[int(row["indice_dataset"])] = row
            except (KeyError, TypeError, ValueError):
                continue
    return cached


def _write_manifest(path: Path, rows: list[dict], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    mode = "a" if append and exists else "w"
    with path.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
        file.flush()


def prepare_mediapipe_cache(
    prepared_sources, source_names, preprocessor, cache_root: Path,
    *, log_every: int = 100, flush_every: int = 100,
):
    """Detecta manos una vez y persiste progresivamente solo ROI/estado en CSV.

    El manifiesto se actualiza por lotes, incluyendo fallos de detección.
    Si el proceso se interrumpe, la siguiente ejecución reutiliza lo ya escrito.
    """
    started = time.perf_counter()
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    audit_rows, summaries, boxes_by_source = [], {}, {}

    options = json.dumps(preprocessor.options, sort_keys=True, ensure_ascii=False, default=str)
    config_key = hashlib.sha256(
        (preprocessor.pipeline + "|" + options).encode("utf-8")
    ).hexdigest()[:16]

    print(
        f"[MediaPipe] Inicio de preparación. Pipeline={preprocessor.pipeline}; "
        f"fuentes={', '.join(source_names)}; caché={cache_root}",
        flush=True,
    )

    for source_name in source_names:
        source = prepared_sources[source_name]
        # Las fuentes con split usan train/val/test. Las fuentes externas
        # (external_eval/final_test) no tienen split y se procesan completas
        # bajo el manifiesto external_rois.csv.
        partitions = (
            ("train", "val", "test")
            if source.split is not None
            else ("external",)
        )

        fingerprint = dataset_fingerprint(source.dataset)
        source_cache = cache_root / source_name / f"{fingerprint[:16]}_{config_key}"
        source_cache.mkdir(parents=True, exist_ok=True)
        boxes_by_source[source_name] = {}
        print(
            f"[MediaPipe] Fuente={source_name}; imágenes={len(source.dataset)}; "
            f"fingerprint={fingerprint[:16]}",
            flush=True,
        )

        for partition in partitions:
            indices = (
                [int(index) for index in getattr(source.split, partition)]
                if source.split is not None
                else list(range(len(source.dataset)))
            )
            manifest_path = source_cache / f"{partition}_rois.csv"
            cached_rows = _load_manifest(manifest_path)
            valid: list[int] = []
            detected = excluded = reused = processed = 0
            pending_rows: list[dict] = []
            part_started = time.perf_counter()

            print(
                f"[MediaPipe] {source_name}/{partition}: inicio "
                f"({len(indices)} imágenes; {len(cached_rows)} registros reutilizables).",
                flush=True,
            )

            for position, index in enumerate(indices, start=1):
                record = source.dataset.records[index]
                old = cached_rows.get(index)

                # Un registro solo se reutiliza si coincide también la ruta relativa.
                if old is not None and old.get("ruta") == record.relative_path:
                    reused += 1
                    ok = str(old.get("detectada", "")).lower() == "true"
                    if ok:
                        try:
                            box = tuple(int(old[k]) for k in ("x1", "y1", "x2", "y2"))
                            if len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
                                valid.append(index)
                                boxes_by_source[source_name][index] = box
                                detected += 1
                                continue
                        except (KeyError, TypeError, ValueError):
                            pass
                    excluded += 1
                    audit_rows.append(old)
                    continue

                reason = ""
                box = None
                try:
                    image = source.dataset.load_image(index)
                    box = preprocessor.detect_mediapipe_box(image)
                except RuntimeError as exc:
                    reason = str(exc)
                    expected_failure = any(term in reason.casefold() for term in (
                        "no detectó ninguna mano", "región roi inválida",
                        "recorte vacío", "roi vacía o inválida",
                    ))
                    if not expected_failure:
                        raise RuntimeError(
                            f"Error inesperado al detectar MediaPipe en {record.relative_path}: {exc}"
                        ) from exc

                ok = box is not None
                if ok:
                    valid.append(index)
                    detected += 1
                    boxes_by_source[source_name][index] = box
                else:
                    excluded += 1

                row = {
                    "fuente": source_name, "particion": partition,
                    "indice_dataset": index, "clase": record.class_name,
                    "ruta": record.relative_path, "detectada": str(ok).lower(),
                    "x1": box[0] if ok else "", "y1": box[1] if ok else "",
                    "x2": box[2] if ok else "", "y2": box[3] if ok else "",
                    "motivo_exclusion": reason,
                }
                pending_rows.append(row)
                audit_rows.append(row)
                processed += 1

                if len(pending_rows) >= max(1, flush_every):
                    _write_manifest(manifest_path, pending_rows, append=True)
                    pending_rows.clear()

                if position % max(1, log_every) == 0 or position == len(indices):
                    elapsed = time.perf_counter() - part_started
                    rate = position / elapsed if elapsed else 0.0
                    remaining = (len(indices) - position) / rate if rate else 0.0
                    print(
                        f"[MediaPipe] {source_name}/{partition}: "
                        f"{position}/{len(indices)} ({position/len(indices):.1%}); "
                        f"detectadas={detected}; excluidas={excluded}; "
                        f"reutilizadas={reused}; {rate:.1f} img/s; "
                        f"ETA≈{remaining/60:.1f} min",
                        flush=True,
                    )

            if pending_rows:
                _write_manifest(manifest_path, pending_rows, append=True)
                pending_rows.clear()

            summaries[f"{source_name}:{partition}"] = {
                "total": len(indices), "detectadas": detected,
                "excluidas": excluded, "reutilizadas": reused,
                "procesadas_en_esta_ejecucion": processed,
                "cobertura": detected / len(indices) if indices else 0.0,
                "manifiesto": str(manifest_path),
                "duracion_segundos": round(time.perf_counter() - part_started, 2),
            }
            if not valid:
                raise RuntimeError(
                    f"MediaPipe no dejó imágenes válidas en {source_name}/{partition}."
                )
            setattr(source, f"_mediapipe_{partition}_indices", valid)
            print(
                f"[MediaPipe] {source_name}/{partition}: FINALIZADO; "
                f"detectadas={detected}/{len(indices)}; "
                f"excluidas={excluded}; reutilizadas={reused}; "
                f"manifiesto={manifest_path}",
                flush=True,
            )

    audit_path = cache_root / "auditoria_mediapipe.csv"
    # Reconstruye auditoría desde los manifiestos completos para incluir también
    # registros reutilizados y mantenerla íntegra después de una reanudación.
    all_rows = []
    for source_name in source_names:
        source = prepared_sources[source_name]
        fingerprint = dataset_fingerprint(source.dataset)
        source_cache = cache_root / source_name / f"{fingerprint[:16]}_{config_key}"
        partitions = (
            ("train", "val", "test")
            if source.split is not None
            else ("external",)
        )
        for partition in partitions:
            manifest = source_cache / f"{partition}_rois.csv"
            if manifest.exists():
                with manifest.open("r", newline="", encoding="utf-8") as file:
                    all_rows.extend(csv.DictReader(file))
    _write_manifest(audit_path, all_rows, append=False)

    summary_path = cache_root / "resumen_mediapipe.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[MediaPipe] PREPARACIÓN COMPLETA en {time.perf_counter()-started:.1f}s. "
        f"Auditoría={audit_path}; resumen={summary_path}",
        flush=True,
    )
    return boxes_by_source, summaries, audit_path, summary_path
