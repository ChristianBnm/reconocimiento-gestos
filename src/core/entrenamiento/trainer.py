from __future__ import annotations

import json
from pathlib import Path

import torch

from src.core.configuracion.settings import ExperimentConfig
from src.core.modelos.factory import crear_modelo, congelar_backbone


class Trainer:
    """Encapsula el ciclo de entrenamiento, validación y checkpointing."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        config: ExperimentConfig,
        checkpoint_path: Path,
        history_path: Path,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.history_path = history_path
        self.device = next(model.parameters()).device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size

        return total_loss / total, correct / total

    def validate_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += batch_size

        return total_loss / total, correct / total

    def _save_history(self, history: list[dict]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.history_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
        temporary.replace(self.history_path)

    def _save_checkpoint(self, epoch: int, val_loss: float, val_accuracy: float) -> None:
        payload = {
            "format_version": 2,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "model": self.config.model.name,
            "architecture": self.config.model.architecture,
            "num_classes": self.config.num_classes,
            "model_state_dict": self.model.state_dict(),
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self.checkpoint_path)

    def fit(self) -> list[dict]:
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        history: list[dict] = []

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.validate_epoch()

            record = {
                "epoca": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
            history.append(record)
            self._save_history(history)

            print(
                f"Época {epoch}/{self.config.epochs} | "
                f"Train loss={train_loss:.4f} acc={train_accuracy:.4f} | "
                f"Val loss={val_loss:.4f} acc={val_accuracy:.4f}"
            )

            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_loss, val_accuracy)
                print("  ✓ Nuevo mejor checkpoint.")
            else:
                epochs_without_improvement += 1
                print(
                    f"  Sin mejora: {epochs_without_improvement}/{self.config.patience}"
                )

            if epochs_without_improvement >= self.config.patience:
                print("  Early Stopping activado.")
                break

        return history


def entrenar(
    config: ExperimentConfig,
    train_loader,
    val_loader,
    checkpoint_path: Path,
    history_path: Path,
):
    model = crear_modelo(
        architecture=config.model.architecture,
        num_classes=config.num_classes,
        pretrained=config.model.pretrained,
    )
    model = congelar_backbone(model)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )
    return trainer.fit()
