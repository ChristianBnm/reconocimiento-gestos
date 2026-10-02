from pathlib import Path

import cv2
import numpy as np

from src.core.dataset.loader import GestosDataset


def test_dataset_ordena_clases_y_carga_imagen(tmp_path: Path):
    for class_name in ["B", "A"]:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(class_dir / "img.jpg"), image)

    dataset = GestosDataset(tmp_path, class_names=["A", "B"])
    tensor_like, label = dataset[0]

    assert dataset.clases == ["A", "B"]
    assert label == 0
    assert tensor_like.shape[0] == 32
