# Flujo del proyecto

```text
Docker / entorno
      ↓
Dataset → split reproducible → transforms → DataLoader
      ↓
Modelo pretrained → backbone congelado → entrenamiento
      ↓
checkpoint + history.json + split.json + config.json
      ↓
evaluación sobre TEST
      ↓
metrics.json + predicciones.csv
      ↓
matriz de confusión / análisis de errores / generalización externa
      ↓
inferencia
```

## Principios del refactor

- El código de dominio queda separado por responsabilidad.
- La selección del modelo se realiza desde la CLI.
- El split exacto queda persistido y se valida mediante fingerprint.
- Los resultados y checkpoints quedan separados por modelo.
- Los paths guardados en `predicciones.csv` son relativos al proyecto.
- El checkpoint nuevo conserva metadatos y sigue admitiendo checkpoints legacy que sean un `state_dict`.
- No se introduce scheduler ni AMP en esta etapa: se preserva el protocolo experimental actual.
