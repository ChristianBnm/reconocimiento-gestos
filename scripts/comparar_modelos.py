from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.configuracion.settings import list_models
from src.core.utils.paths import ProjectPaths


def main():
    parser = argparse.ArgumentParser(description="Compara metrics.json de modelos entrenados.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--modelos", nargs="+", default=list_models(), choices=list_models())
    args = parser.parse_args()

    paths = ProjectPaths(Path(args.root).resolve())
    print("modelo | accuracy | precision_macro | recall_macro | f1_macro | balanced_accuracy | cohen_kappa")
    print("-" * 110)

    for model_name in args.modelos:
        path = paths.model_results_dir(model_name) / "metrics.json"
        if not path.exists():
            print(f"{model_name} | sin metrics.json")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"{model_name} | "
            f"{data['accuracy']:.6f} | "
            f"{data['precision_macro']:.6f} | "
            f"{data['recall_macro']:.6f} | "
            f"{data['f1_macro']:.6f} | "
            f"{data['balanced_accuracy']:.6f} | "
            f"{data['cohen_kappa']:.6f}"
        )


if __name__ == "__main__":
    main()
