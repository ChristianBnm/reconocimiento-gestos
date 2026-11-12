"""Inferencia de una imagen externa usando el checkpoint de una ejecución."""
from __future__ import annotations
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from src.core.experimentos.service import ExperimentService
from src.core.experimentos.evaluation import _configuracion_del_checkpoint, _validar_checkpoint, _seleccionar_dispositivo
from src.core.modelos.factory import crear_modelo
from src.core.dataset.preprocessing import ImagePreprocessor
from src.core.dataset.transforms import build_eval_transform


def predecir_imagen(root: str | Path, run_id: str, image_bytes: bytes) -> dict:
    root = Path(root).expanduser().resolve()
    service = ExperimentService(root)
    run = service.get_run(run_id)
    if run.status != "completed":
        raise ValueError("La predicción solo está disponible para ejecuciones completadas.")
    if not run.checkpoint_path:
        raise FileNotFoundError("La ejecución no tiene un checkpoint asociado.")
    checkpoint_path = Path(run.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No se encontró el checkpoint: {checkpoint_path}")
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("No se pudo leer la imagen. Usá JPG o PNG válido.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    spec = service.get_experiment(run.experiment_id)
    config = _configuracion_del_checkpoint(checkpoint, spec, service)
    class_names, num_classes = _validar_checkpoint(checkpoint, run, config)
    device = _seleccionar_dispositivo(config.device)
    model = crear_modelo(config.model.architecture, num_classes, pretrained=False, dispositivo=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    options = dict(config.preprocessing_options or {})
    preprocessor = ImagePreprocessor(config.preprocessing_pipeline, options)
    try:
        processed = preprocessor(rgb)
    finally:
        hands = getattr(preprocessor, "_hands", None)
        if hands is not None:
            hands.close()
    transform = build_eval_transform(config=config)
    tensor = transform(image=processed)["image"].unsqueeze(0).to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probabilities, k=min(3, num_classes))
    elapsed_ms = (time.perf_counter() - started) * 1000
    top = [{"class_name": class_names[int(i)], "probability": float(v)} for v, i in zip(values.cpu(), indices.cpu())]
    return {
        "run_id": run_id,
        "experiment_id": run.experiment_id,
        "model": checkpoint.get("model"),
        "architecture": checkpoint.get("architecture", config.model.architecture),
        "predicted_class": top[0]["class_name"],
        "confidence": top[0]["probability"],
        "top_predictions": top,
        "inference_time_ms": round(elapsed_ms, 3),
        "preprocessing_pipeline": config.preprocessing_pipeline,
        "image_size": config.image_size,
        "device": str(device),
    }
