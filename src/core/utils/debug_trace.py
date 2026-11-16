from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def trace(event: str, *, log_path=None, **details) -> None:
    """Traza temporal JSONL + consola; nunca debe ocultar el error original."""
    record = {
        "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
        "unix": time.time(),
        "pid": os.getpid(),
        "event": event,
        **details,
    }
    line = json.dumps(record, ensure_ascii=False, default=str)
    print(f"[DEBUG-TRAIN] {line}", flush=True)
    if log_path:
        try:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\\n")
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as exc:
            print(f"[DEBUG-TRAIN] No se pudo escribir JSONL: {type(exc).__name__}: {exc}", flush=True)


def trace_exception(event: str, *, log_path=None, exc: BaseException, **details) -> None:
    trace(event, log_path=log_path, error_type=type(exc).__name__,
          error=str(exc), traceback=traceback.format_exc(), **details)
