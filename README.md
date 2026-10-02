# Reconocimiento de gestos

Refactor del proyecto de tesis para separar responsabilidades, mejorar reproducibilidad y preparar una futura interfaz gráfica.

## Arquitectura

```text
src/
├── core/
│   ├── configuracion/
│   ├── dataset/
│   ├── modelos/
│   ├── entrenamiento/
│   ├── evaluacion/
│   └── utils/
├── cli/
│   ├── main.py
│   ├── entrenamiento.py      # compatibilidad
│   └── evaluacion.py         # compatibilidad
├── entrenamiento.py          # compatibilidad directa
└── evaluacion.py             # compatibilidad directa

scripts/
pruebas/
auditoria/
data/
checkpoints/
results/
docs/
notebooks/
```

## Ejecución con Docker

```bash
docker compose build
docker compose up -d
docker compose exec tesis python3 -m src.cli.main modelos
docker compose exec tesis python3 -m src.cli.main entrenar --modelo mobilenetv2
docker compose exec tesis python3 -m src.cli.main evaluar --modelo mobilenetv2
docker compose exec tesis python3 scripts/matriz_confusion.py --modelo mobilenetv2
docker compose exec tesis python3 scripts/analizar_errores.py --modelo mobilenetv2
python3 scripts/graficar_entrenamiento.py --modelo mobilenetv2
python3 scripts/comparar_modelos.py
python3 scripts/verificar_entorno.py
```

## Experimentos

Modelos registrados:

- `mobilenetv2` → `mobilenetv2_100`
- `vit_tiny` → `vit_tiny_patch16_224`

Cada experimento guarda en `results/<modelo>/`:

- `config.json`
- `split.json`
- `history.json`
- `metrics.json`
- `classification_report.json`
- `predicciones.csv`
- gráficos y análisis derivados

Y en `checkpoints/<modelo>/`:

- `mejor.pth`

## Importante

El refactor separa la infraestructura del código de ML, pero no elimina las limitaciones metodológicas ya identificadas: el split sigue siendo por imagen y la evaluación externa sigue siendo una etapa pendiente. Antes de presentar resultados definitivos, hay que volver a ejecutar el pipeline con el dataset definitivo y comprobar la independencia de los conjuntos.
