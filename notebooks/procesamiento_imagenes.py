"""
procesamiento_imagenes.py

Uso:
    - Ejecutar desde un notebook o como script:
        python procesamiento_imagenes.py

Descripción:
    Aplica mejoras de imagen + generación de aumentaciones por imagen
    y guarda todo en una carpeta de salida manteniendo la estructura por clase.
"""

import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# -------------------------
# CONFIGURACIÓN (modificar)
# -------------------------
INPUT_DIR = r"E:\Mis Documentos\Christian\Tesis\tesis_gestos\data\Clases_200"
OUTPUT_DIR = r"E:\Mis Documentos\Christian\Tesis\tesis_gestos\data\Clases_200_proc"  # carpeta nueva
RESIZE = (224, 224)
augment_factor = 2        # cuántas imágenes aumentadas crear por imagen original (0 = solo procesado)
apply_deterministic_enhance = True  # aplicar CLAHE / unsharp / denoise antes de aumentaciones

# Clase -> multiplicador extra (opcional). Ej: {"A": 3} -> generar 3 veces más para clase A.
class_extra_multiplier = {
    # "A": 2,
    # "B": 1
}

# Albumentations pipeline para aumentaciones aleatorias
alb_pipeline = A.Compose([
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8,8), p=0.4),
    ], p=0.8),
    A.OneOf([
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.MotionBlur(blur_limit=3, p=0.3),
        A.MedianBlur(blur_limit=3, p=0.2),
    ], p=0.4),
    A.OneOf([
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1,0.5), p=0.2),
        A.NoOp()
    ], p=0.3),
    A.Sharpen(alpha=(0.2, 0.6), lightness=(0.5, 1.0), p=0.3),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=12, p=0.3),
    # mantener tamaño y recortar/resize al final si hace falta
])

# -------------------------
# FUNCIONES DE ENHANCEMENT
# -------------------------
def apply_clahe_rgb(img_rgb):
    # img_rgb: numpy HWC uint8 RGB
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    img2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return img2

def unsharp_mask(img_rgb, kernel_size=(5,5), sigma=1.0, amount=1.0, threshold=0):
    # Unsharp mask: sharpen by subtracting blurred image
    blurred = cv2.GaussianBlur(img_rgb, kernel_size, sigma)
    sharpened = cv2.addWeighted(img_rgb, 1 + amount, blurred, -amount, 0)
    return sharpened

def denoise_image(img_rgb):
    # Fast Non-local Means denoising (works on RGB)
    return cv2.fastNlMeansDenoisingColored(img_rgb, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def enhance_pipeline(img_rgb):
    # Aplicar varios pasos de mejora (ajustá parámetros si querés)
    img = img_rgb.copy()
    img = apply_clahe_rgb(img)
    img = denoise_image(img)
    img = unsharp_mask(img, kernel_size=(5,5), sigma=1.0, amount=0.8)
    # Aplicar gamma levemente (aumenta o reduce)
    img = adjust_gamma(img, gamma=1.05)
    return img

# -------------------------
# UTILS
# -------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_jpeg(path, img_bgr, quality=95):
    # img_bgr: OpenCV BGR uint8. Guarda con buena calidad.
    cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

# -------------------------
# PROCESADO PRINCIPAL
# -------------------------
def process_dataset(input_dir, output_dir, resize=RESIZE, augment_factor=augment_factor):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    clases = sorted([p.name for p in input_dir.iterdir() if p.is_dir()])
    print(f"Clases encontradas: {len(clases)} -> {clases}")

    for clase in clases:
        in_clase = input_dir / clase
        out_clase = output_dir / clase
        ensure_dir(out_clase)
        files = sorted([p for p in in_clase.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

        # multiplicador por clase
        extra_mult = class_extra_multiplier.get(clase, 1)

        for fpath in tqdm(files, desc=f"Clase {clase}", unit="img"):
            try:
                img_bgr = cv2.imread(str(fpath))
                if img_bgr is None:
                    print("No se pudo leer:", fpath)
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                # Resize base
                img_rgb = cv2.resize(img_rgb, resize)

                # 1) Guardar imagen procesada determinística
                if apply_deterministic_enhance:
                    img_rgb_proc = enhance_pipeline(img_rgb)
                else:
                    img_rgb_proc = img_rgb

                # Convertir a BGR para guardado con cv2
                img_bgr_proc = cv2.cvtColor(img_rgb_proc, cv2.COLOR_RGB2BGR)
                base_name = fpath.stem
                out_name_proc = out_clase / f"{base_name}_proc.jpg"
                save_jpeg(str(out_name_proc), img_bgr_proc)

                # 2) Generar augmentaciones aleatorias
                # calcular cuántas a generar: augment_factor * extra_mult
                total_aug = augment_factor * extra_mult
                for i in range(total_aug):
                    # Albumentations opera en formato uint8 RGB HWC
                    augmented = alb_pipeline(image=img_rgb_proc)["image"]
                    # A veces A devuelve float32; asegurarse uint8
                    if augmented.dtype != np.uint8:
                        augmented = (np.clip(augmented, 0, 255)).astype(np.uint8)
                    augmented = cv2.resize(augmented, resize)  # asegurar tamaño
                    out_name = out_clase / f"{base_name}_aug_{i}.jpg"
                    save_jpeg(str(out_name), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print("Error procesando", fpath, e)

    print("Procesado finalizado. Salida en:", output_dir)

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    process_dataset(INPUT_DIR, OUTPUT_DIR, resize=RESIZE, augment_factor=augment_factor)
