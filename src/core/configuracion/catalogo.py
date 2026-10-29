from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

CONFIG_DIR = Path(__file__).resolve().parent


def _read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def cargar_catalogos(
    sources_path: str | Path | None = None,
    scenarios_path: str | Path | None = None,
) -> tuple[dict, dict]:
    sources_file = Path(sources_path) if sources_path else CONFIG_DIR / "fuentes.json"
    scenarios_file = Path(scenarios_path) if scenarios_path else CONFIG_DIR / "escenarios.json"
    sources = _read_json(sources_file)
    scenarios = _read_json(scenarios_file)
    if not isinstance(sources, dict) or not sources:
        raise ValueError("fuentes.json debe ser un objeto JSON no vacío.")
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("escenarios.json debe ser un objeto JSON no vacío.")
    for name, spec in sources.items():
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"Identificador de fuente inválido: {name!r}")
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str) or not spec["path"].strip():
            raise ValueError(f"La fuente '{name}' debe definir una ruta no vacía.")
        if "deduplicate_exact_duplicates" in spec and not isinstance(spec["deduplicate_exact_duplicates"], bool):
            raise ValueError(
                f"deduplicate_exact_duplicates para '{name}' debe ser true o false."
            )
    for name, spec in scenarios.items():
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"Identificador de escenario inválido: {name!r}")
        sources_for_scenario = spec.get("sources") if isinstance(spec, dict) else None
        if not isinstance(sources_for_scenario, list) or not sources_for_scenario:
            raise ValueError(f"El escenario '{name}' debe definir una lista no vacía de fuentes.")
        if any(not isinstance(source_name, str) or not source_name.strip() for source_name in sources_for_scenario):
            raise ValueError(f"Las fuentes del escenario '{name}' deben ser identificadores de texto no vacíos.")
        unknown = sorted(set(sources_for_scenario) - set(sources))
        if unknown:
            raise ValueError(
                f"El escenario '{name}' refiere fuentes no configuradas: {', '.join(unknown)}"
            )
        if len(set(sources_for_scenario)) != len(sources_for_scenario):
            raise ValueError(f"El escenario '{name}' repite fuentes.")
    return sources, scenarios


def construir_mapa_fuentes(
    root: Path,
    source_definitions: Mapping[str, dict],
    overrides: Mapping[str, str | Path] | None = None,
) -> dict[str, Path]:
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(source_definitions))
    if unknown:
        raise ValueError(f"Se pasaron rutas para fuentes no configuradas: {', '.join(unknown)}")
    result: dict[str, Path] = {}
    for name, spec in source_definitions.items():
        raw_path = overrides.get(name, spec["path"])
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / path
        result[name] = path.resolve()
    reverse: dict[Path, str] = {}
    for name, path in result.items():
        if path in reverse:
            raise ValueError(
                f"Las fuentes '{reverse[path]}' y '{name}' apuntan a la misma carpeta: {path}. "
                "Configurá una ruta distinta por fuente para evitar duplicar su peso en el experimento."
            )
        reverse[path] = name
    return result
