from pathlib import Path

import cv2
import numpy as np

from src.core.dataset.loader import GestosDataset


def test_dataset_ordena_clases_y_carga_imagen(tmp_path: Path):
    for index, class_name in enumerate(["B", "A"]):
        class_dir = tmp_path / class_name
        class_dir.mkdir()

        # Imágenes distintas para evitar duplicados entre etiquetas.
        image = np.full(
            (32, 32, 3),
            fill_value=50 + index * 100,
            dtype=np.uint8,
        )
        cv2.imwrite(str(class_dir / "img.jpg"), image)

    dataset = GestosDataset(tmp_path, class_names=["A", "B"])

    assert dataset.clases == ["A", "B"]
