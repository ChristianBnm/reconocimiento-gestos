from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.core.configuracion.settings import list_models
from src.core.utils.paths import ProjectPaths


def main():
    parser = argparse.ArgumentParser(description="Grafica history.json.")
    parser.add_argument("--modelo", required=True, choices=list_models())
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    paths = ProjectPaths(Path(args.root).resolve())
    results_dir = paths.model_results_dir(args.modelo)
    history_path = results_dir / "history.json"

    if not history_path.exists():
        raise FileNotFoundError(f"No se encontró: {history_path}")

    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)

    epochs = [r["epoca"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    train_acc = [r["train_accuracy"] for r in history]
    val_acc = [r["val_accuracy"] for r in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, label="Train Loss")
    ax.plot(epochs, val_loss, label="Validation Loss")
    ax.set(xlabel="Época", ylabel="Loss", title=f"Loss - {args.modelo}")
    ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(results_dir / "loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_acc, label="Train Accuracy")
    ax.plot(epochs, val_acc, label="Validation Accuracy")
    ax.set(xlabel="Época", ylabel="Accuracy", title=f"Accuracy - {args.modelo}")
    ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(results_dir / "accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Gráficos guardados en: {results_dir}")


if __name__ == "__main__":
    main()
