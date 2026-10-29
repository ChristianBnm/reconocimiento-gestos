from __future__ import annotations

from pathlib import Path
import time

import torch

from src.core.configuracion.settings import ExperimentConfig
from src.core.modelos.factory import crear_modelo, congelar_backbone, descongelar_modelo
from src.core.utils.io import save_json


class Trainer:
    """Ciclo de entrenamiento/validación con early stopping y checkpoints portables."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        config: ExperimentConfig,
        checkpoint_path: Path,
        history_path: Path,
        metadata: dict | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.history_path = Path(history_path)
        try:
            self.device = next(model.parameters()).device
        except StopIteration as exc:
            raise ValueError("El modelo no tiene parámetros.") from exc
        self.criterion = torch.nn.CrossEntropyLoss()
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise ValueError("El modelo no tiene parámetros entrenables.")
        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.metadata = metadata or {}

    def _run_epoch(self, loader, training: bool) -> tuple[float, float]:
        if len(loader) == 0:
            raise ValueError("El DataLoader de entrenamiento/validación está vacío.")
        if training and self.config.freeze_backbone:
            # Mantiene congeladas también las estadísticas de BatchNorm del backbone.
            self.model.eval()
            classifier = self.model.get_classifier() if hasattr(self.model, "get_classifier") else None
            if hasattr(classifier, "train"):
                classifier.train(True)
        else:
            self.model.train(training)
        total_loss = 0.0
        correct = 0
        total = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                if logits.ndim != 2 or logits.shape[1] != self.config.num_classes:
                    raise ValueError(
                        f"La salida del modelo debe tener forma [lote, {self.config.num_classes}], "
                        f"pero se obtuvo {tuple(logits.shape)}."
                    )
                loss = self.criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "La pérdida dejó de ser finita. Revisá los datos, la tasa de aprendizaje y las transformaciones."
                    )
                if training:
                    loss.backward()
                    self.optimizer.step()
                batch_size = int(labels.size(0))
                total_loss += float(loss.item()) * batch_size
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += batch_size
        return total_loss / total, correct / total

    def train_epoch(self) -> tuple[float, float]:
        return self._run_epoch(self.train_loader, training=True)

    def validate_epoch(self) -> tuple[float, float]:
        return self._run_epoch(self.val_loader, training=False)

    def _save_checkpoint(self, epoch: int, val_loss: float, val_accuracy: float) -> None:
        payload = {
            "format_version": 3,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "model": self.config.model.name,
            "architecture": self.config.model.architecture,
            "num_classes": self.config.num_classes,
            "class_names": list(self.config.class_names),
            "class_to_idx": self.config.class_to_idx,
            "experiment_config": self.config.to_dict(),
            "metadata": self.metadata,
            "model_state_dict": self.model.state_dict(),
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(self.checkpoint_path)

    def fit(self) -> list[dict]:
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        history: list[dict] = []

        for epoch in range(1, self.config.epochs + 1):
            epoch_started = time.perf_counter()
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.validate_epoch()
            epoch_seconds = time.perf_counter() - epoch_started
            record = {
                "epoch": epoch,
                "epoch_seconds": round(epoch_seconds, 6),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
            history.append(record)
            save_json(history, self.history_path)
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
                print(f"  Sin mejora: {epochs_without_improvement}/{self.config.patience}")

            if epochs_without_improvement >= self.config.patience:
                print("  Early stopping activado.")
                break
        return history


def entrenar(
    config: ExperimentConfig,
    train_loader,
    val_loader,
    checkpoint_path: Path,
    history_path: Path,
    metadata: dict | None = None,
):
    model = crear_modelo(
        architecture=config.model.architecture,
        num_classes=config.num_classes,
        pretrained=config.model.pretrained,
    )
    model = congelar_backbone(model) if config.freeze_backbone else descongelar_modelo(model)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        metadata=metadata,
    )
    return trainer.fit()
