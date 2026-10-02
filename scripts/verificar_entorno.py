from __future__ import annotations

import importlib


def main():
    import torch

    print("=== Entorno ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    print(f"CUDA de PyTorch: {torch.version.cuda}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Capacidad: {torch.cuda.get_device_capability(0)}")

    for name in ["timm", "albumentations", "cv2", "sklearn", "pandas"]:
        try:
            module = importlib.import_module(name)
            print(f"{name}: {getattr(module, '__version__', 'ok')}")
        except Exception as exc:
            print(f"{name}: ERROR - {exc}")


if __name__ == "__main__":
    main()
