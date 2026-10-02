# Migración al refactor

## Estructura anterior → nueva

| Antes | Ahora |
|---|---|
| `src/configuracion.py` | `src/core/configuracion/settings.py` |
| `src/dataset_utils.py` | `src/core/dataset/loader.py` + `transforms.py` |
| `src/division_dataset.py` | `src/core/dataset/split.py` |
| `src/modelos.py` | `src/core/modelos/factory.py` |
| `src/entrenamiento.py` | `src/core/entrenamiento/trainer.py` + CLI |
| `src/evaluacion.py` | `src/core/evaluacion/predictor.py` + CLI |
| `src/matriz_confusion.py` | `scripts/matriz_confusion.py` |
| `src/analisis_errores.py` | `scripts/analizar_errores.py` |
| `src/graficos_entrenamiento.py` | `scripts/graficar_entrenamiento.py` |
| `src/auditar_dataset.py` | `auditoria/auditar_dataset.py` |
| `src/probar_*.py` | `pruebas/` |

## Comandos principales

```bash
python3 -m src.cli.main modelos
python3 -m src.cli.main entrenar --modelo mobilenetv2
python3 -m src.cli.main entrenar --modelo vit_tiny
python3 -m src.cli.main evaluar --modelo mobilenetv2
python3 -m src.cli.main evaluar --modelo vit_tiny
python3 scripts/matriz_confusion.py --modelo mobilenetv2
python3 scripts/analizar_errores.py --modelo mobilenetv2
python3 scripts/graficar_entrenamiento.py --modelo mobilenetv2
python3 -m src.cli.main inferir --modelo mobilenetv2 --imagen data/ejemplo.jpg
```

Los entry points `src/entrenamiento.py` y `src/evaluacion.py` se mantienen temporalmente para facilitar la transición:

```bash
python3 src/entrenamiento.py --modelo vit_tiny
python3 src/evaluacion.py --modelo vit_tiny
```

## Cambios metodológicos deliberadamente conservadores

El refactor **no agrega AMP ni scheduler** y mantiene los hiperparámetros base actuales. Sí mejora la trazabilidad del experimento: el split exacto se guarda, se valida con fingerprint y la configuración se registra en JSON.

Los resultados 99.95% / 99.79% obtenidos antes del refactor deben considerarse resultados históricos del código/protocolo anterior hasta volver a ejecutar el pipeline refactorizado con el dataset definitivo.
