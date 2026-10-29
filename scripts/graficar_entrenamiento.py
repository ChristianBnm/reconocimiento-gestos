from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import matplotlib.pyplot as plt

from src.core.configuracion.catalogo import cargar_catalogos
from src.core.configuracion.settings import list_models
from src.core.utils.paths import ProjectPaths


def main():
    _, scenarios = cargar_catalogos()
    parser = argparse.ArgumentParser(description="Grafica el historial de entrenamiento.")
    parser.add_argument("--modelo", required=True, choices=list_models())
    parser.add_argument("--escenario", choices=list(scenarios), default="combinado")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    paths = ProjectPaths(Path(args.root).resolve())
    results_dir = paths.model_results_dir(args.modelo, args.escenario)
    history_path = results_dir / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"No se encontró: {history_path}")

    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)
    epochs = [record.get("epoch", record.get("epoca")) for record in history]
    train_loss = [record["train_loss"] for record in history]
    val_loss = [record["val_loss"] for record in history]
    train_acc = [record["train_accuracy"] for record in history]
    val_acc = [record["val_accuracy"] for record in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, label="Train Loss")
    ax.plot(epochs, val_loss, label="Validation Loss")
    ax.set(xlabel="Época", ylabel="Loss", title=f"Loss - {args.modelo} ({args.escenario})")
    ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(results_dir / "loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_acc, label="Train Accuracy")
    ax.plot(epochs, val_acc, label="Validation Accuracy")
    ax.set(xlabel="Época", ylabel="Accuracy", title=f"Accuracy - {args.modelo} ({args.escenario})")
    ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(results_dir / "accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráficos guardados en: {results_dir}")


if __name__ == "__main__":
    main()
