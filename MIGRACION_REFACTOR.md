# Notas de migración del pipeline

El refactor actual conserva la organización modular del proyecto y agrega controles de escalabilidad y reproducibilidad. La guía de uso vigente, los comandos de entrenamiento por escenario y la estructura de artefactos están documentados en [`README.md`](README.md).

## Cambios principales

- `settings.py`: el número de clases se deriva de un vocabulario configurable; ya no es un valor fijo usado por todos los componentes.
- `clases.json`: orden de etiquetas estable y explícito. Se acepta también el formato `{"classes": [...]}`.
- `fuentes.json` y `escenarios.json`: definen fuentes y combinaciones de entrenamiento fuera del código.
- `loader.py`: guarda el origen, la ruta relativa y el SHA-256 de cada imagen, valida que las imágenes sean legibles y detecta carpetas de clase no declaradas.
- `split.py`: divide por grupos de contenido idéntico, valida las particiones y guarda fingerprints para detectar cambios de contenido.
- `experiment.py`: prepara splits compartidos por fuente y verifica duplicados exactos entre fuentes.
- `transforms.py`: normalización resuelta desde la configuración de cada modelo preentrenado de timm, transformaciones configurables y flip horizontal desactivado por defecto.
- `factory.py`: carga diferida de `timm`, selección explícita de dispositivo, congelamiento o ajuste fino y validación del vocabulario de checkpoints.
- `trainer.py`: checkpoint con configuración y mapeo de etiquetas, escritura atómica de artefactos y tratamiento de BatchNorm en el modo de extractor de características.
- `predictor.py`, `metrics.py` y `reporting.py`: evaluación independiente por fuente, CSV de predicciones, métricas e informes por clase y matrices de confusión.
- `main.py`: comandos para los tres escenarios de entrenamiento, evaluación por fuente e inferencia con el vocabulario guardado en el checkpoint.
- `pruebas/`: pruebas automatizadas para vocabulario variable, carga de datos, particiones y checkpoints.

## Compatibilidad

Se conserva el cargador para checkpoints antiguos basados en `state_dict`. Si estos no guardan los nombres y el orden de las clases, el código lo advierte porque no es posible comprobar el mapeo de etiquetas. Para inferencia con ellos puede pasarse `--checkpoint` con la ruta histórica. Los nuevos experimentos escriben checkpoints bajo `checkpoints/<escenario>/<modelo>/` y resultados bajo `results/<escenario>/<modelo>/`, para no mezclar sus artefactos con los históricos.

Las métricas históricas del protocolo anterior no deben combinarse directamente con las del nuevo pipeline: cambió el preprocesamiento, se desactivó por defecto el volteo horizontal y las particiones ahora agrupan duplicados exactos. Todos los modelos y escenarios que se comparen deben entrenarse usando la misma versión de vocabulario y los mismos archivos de partición.
