# Uso de datasets extendidos en Gesture Lab (V3)

## Ubicación esperada

El descubrimiento automático busca carpetas de datasets directamente dentro de `data/`.
Por ejemplo:

```text
data/
├── Clases_200/
│   ├── 0/
│   ├── 1/
│   └── ...
├── Clases_200_comp/
│   ├── 0/
│   ├── 1/
│   └── ...
└── dataset_extendido/
    ├── 0/
    ├── 1/
    ├── A/
    └── ...
```

Cada carpeta inmediata del dataset representa una clase. Las imágenes deben estar
ubicadas directamente dentro de la carpeta de clase; no se admiten subcarpetas
adicionales por clase en esta versión.

## Incorporar un dataset nuevo

1. Copiar la carpeta del dataset dentro de `data/`.
2. Verificar que cada clase tenga imágenes y que no existan carpetas extra con imágenes.
3. Abrir Gesture Lab y entrar en **Recursos disponibles**.
4. Pulsar **Actualizar datasets**. No es necesario reiniciar FastAPI.
5. Confirmar que el dataset aparezca como válido, con la cantidad de clases e imágenes esperadas.
6. Crear el experimento y asignar el dataset extendido como **Entrenamiento** si se desea aprender desde él.

El sistema genera un manifiesto de clases en `results/dataset_catalog/`. Ese manifiesto
fija el orden del vocabulario utilizado por el experimento y se guarda en la configuración
del mismo. No editarlo manualmente.

## Compatibilidad

- El dataset de entrenamiento define el vocabulario del experimento.
- Las fuentes asignadas a prueba final o evaluación externa deben contener exactamente
  las mismas etiquetas (aunque el orden físico de las carpetas no importa).
- Si el dataset tiene clases diferentes, incompletas o carpetas adicionales con imágenes,
  aparecerá en **Carpetas rechazadas** con el motivo.
- Los archivos de imágenes admitidos son JPG, JPEG, PNG, BMP, WEBP, TIF y TIFF.
- El descubrimiento automático de esta versión inspecciona solo los subdirectorios
  inmediatos de `data/`; no busca recursivamente dentro de carpetas arbitrarias.

## Si el entrenamiento no inicia

Revisar la ejecución en **Ejecuciones recientes**. Los errores de validación se registran
como `failed` y deben mostrar el mensaje en la columna Estado. Si el backend se reinicia,
las ejecuciones que estaban `pending` o `running` se marcan como interrumpidas para que
no bloqueen nuevos entrenamientos.
