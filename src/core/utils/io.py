from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
