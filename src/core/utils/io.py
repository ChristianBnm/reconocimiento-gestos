from __future__ import annotations

import json
from pathlib import Path


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    temporary.replace(path)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
