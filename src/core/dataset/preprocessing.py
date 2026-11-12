from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

PIPELINES = {
    "rgb": "RGB original",
    "mediapipe_roi": "MediaPipe + ROI",
    "mediapipe_roi_normalized": "MediaPipe + ROI + normalización",
    "mediapipe_roi_canny": "MediaPipe + ROI + Canny",
    "mediapipe_roi_rgb_edges": "MediaPipe + ROI + RGB + bordes",
    "yolo_roi": "YOLO + ROI",
}


class ImagePreprocessor:
    """Transformación determinista previa a las aumentaciones del entrenamiento."""

    def __init__(self, pipeline: str = "rgb", options: dict[str, Any] | None = None):
        self.pipeline = pipeline or "rgb"
        self.options = dict(options or {})
        if self.pipeline not in PIPELINES:
            raise ValueError(f"Pipeline no soportado: {self.pipeline}. Opciones: {', '.join(PIPELINES)}")
        self._hands = None
        self._yolo = None
        if self.pipeline.startswith("mediapipe_"):
            try:
                import mediapipe as mp
            except ImportError as exc:
                raise RuntimeError("El pipeline MediaPipe requiere instalar mediapipe en el backend.") from exc
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=1,
                min_detection_confidence=float(self.options.get("min_detection_confidence", 0.5)),
            )
        elif self.pipeline == "yolo_roi":
            weights = self.options.get("weights")
            if not weights:
                raise ValueError("YOLO + ROI requiere preprocessing_options.weights con un modelo detector de manos.")
            weights_path = Path(weights).expanduser()
            if not weights_path.is_file():
                raise FileNotFoundError(f"No se encontró el modelo YOLO: {weights_path}")
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("El pipeline YOLO requiere instalar ultralytics en el backend.") from exc
            self._yolo = YOLO(str(weights_path))

    def _crop_roi(self, image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        margin = float(self.options.get("roi_margin", 0.20))
        if not 0 <= margin <= 2:
            raise ValueError("roi_margin debe estar entre 0 y 2.")
        pad_x, pad_y = int((x2 - x1) * margin), int((y2 - y1) * margin)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError("El detector produjo una región ROI inválida.")
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            raise RuntimeError("El detector produjo un recorte vacío.")
        return roi

    def _mediapipe_roi(self, image: np.ndarray) -> np.ndarray:
        result = self._hands.process(image)
        if not result.multi_hand_landmarks:
            raise RuntimeError("MediaPipe no detectó ninguna mano en la imagen.")
        height, width = image.shape[:2]
        points = result.multi_hand_landmarks[0].landmark
        box = (max(0, int(min(p.x for p in points) * width)),
               max(0, int(min(p.y for p in points) * height)),
               min(width, int(max(p.x for p in points) * width) + 1),
               min(height, int(max(p.y for p in points) * height) + 1))
        return self._crop_roi(image, box)

    def _yolo_roi(self, image: np.ndarray) -> np.ndarray:
        results = self._yolo.predict(
            source=image, conf=float(self.options.get("confidence", 0.25)),
            verbose=False, imgsz=int(self.options.get("input_size", 640)),
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            raise RuntimeError("YOLO no detectó ninguna mano en la imagen.")
        best = int(boxes.conf.argmax().item())
        coords = boxes.xyxy[best].detach().cpu().numpy().astype(int)
        return self._crop_roi(image, tuple(coords.tolist()))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.pipeline == "rgb":
            return image
        roi = self._mediapipe_roi(image) if self.pipeline.startswith("mediapipe_") else self._yolo_roi(image)
        if self.pipeline == "mediapipe_roi_normalized":
            lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
            lightness, a, b = cv2.split(lab)
            lightness = cv2.createCLAHE(
                clipLimit=float(self.options.get("clahe_clip_limit", 2.0)), tileGridSize=(8, 8)
            ).apply(lightness)
            roi = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2RGB)
        elif self.pipeline in {"mediapipe_roi_canny", "mediapipe_roi_rgb_edges"}:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, int(self.options.get("canny_low", 100)), int(self.options.get("canny_high", 200)))
            edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            if self.pipeline == "mediapipe_roi_canny":
                roi = edge_rgb
            else:
                alpha = float(self.options.get("edge_blend", 0.35))
                if not 0 <= alpha <= 1:
                    raise ValueError("edge_blend debe estar entre 0 y 1.")
                roi = cv2.addWeighted(roi, 1.0 - alpha, edge_rgb, alpha, 0)
        return roi
