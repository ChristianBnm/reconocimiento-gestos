# Reconocimiento de Gestos Estáticos

Pipeline reproducible y modular para investigar la clasificación de configuraciones manuales estáticas a partir de imágenes. El alcance actual es la clasificación de las clases alfanuméricas incluidas en el vocabulario configurado. El proyecto constituye una base técnica para incorporar más gestos estáticos; no reconoce por sí solo una lengua de señas completa ni secuencias dinámicas.

## Principios de diseño

- **Vocabulario configurable:** las etiquetas e índices provienen de `src/core/configuracion/clases.json`. La cantidad de clases no está fijada en el código ni en el tamaño de la última capa.
- **Fuentes y escenarios configurables:** las fuentes de imágenes se declaran en `src/core/configuracion/fuentes.json`, y los escenarios de entrenamiento en `src/core/configuracion/escenarios.json`.
- **Separación de responsabilidades:** carga de datos, particionado, transformaciones, fábrica de modelos, entrenamiento, evaluación y persistencia están desacoplados.
- **Trazabilidad:** los splits guardan fingerprint del contenido, vocabulario, semilla y proporciones; los checkpoints registran arquitectura, clases, configuración y escenario.
- **Tratamiento explícito de duplicados exactos:** por defecto, se conserva una sola copia de cada contenido SHA-256 por fuente durante la carga; los archivos del dataset nunca se borran. Las exclusiones se guardan en un manifiesto CSV. Si se desactiva esa política en `fuentes.json`, las copias se mantienen en el dataset pero se agrupan para evitar que queden en particiones diferentes. También se comprueba que las copias entre fuentes permanezcan en la misma partición.
- **Resultados separados:** cada combinación de escenario/modelo usa sus propios directorios y no mezcla los artefactos de otros experimentos.

La comprobación por SHA-256 detecta copias byte a byte; no identifica por sí sola imágenes visualmente similares recodificadas, capturas consecutivas ni imágenes de una misma persona. Esos casos requieren auditoría adicional y metadatos de captura cuando estén disponibles. El número de registros usado para entrenar puede ser menor que el número de archivos originales; el manifiesto y `config.json` registran la cantidad excluida.

## Estructura

```text
src/
├── cli/main.py
└── core/
    ├── configuracion/
    │   ├── clases.json
    │   ├── fuentes.json
    │   ├── escenarios.json
    │   ├── catalogo.py
    │   └── settings.py
    ├── dataset/
    │   ├── loader.py
    │   ├── split.py
    │   ├── experiment.py
    │   └── transforms.py
    ├── modelos/factory.py
    ├── entrenamiento/trainer.py
    ├── evaluacion/
    │   ├── metrics.py
    │   ├── predictor.py
    │   └── reporting.py
    └── utils/
pruebas/
scripts/
```

## Preparar el entorno

Las dependencias principales están declaradas en `requirements.txt`; las pruebas, en `requirements-dev.txt`.

```bash
docker compose build
docker compose up -d
docker compose exec tesis python -m src.cli.main modelos
docker compose exec tesis python -m src.cli.main clases
```

El flujo está pensado para ejecutarse desde la raíz del repositorio. También puede correrse en un entorno Python con las dependencias instaladas.

## Fuentes y escenarios

`fuentes.json` asigna un identificador a cada origen de imágenes y su ruta relativa a la raíz del proyecto. `escenarios.json` define qué fuentes se utilizan para entrenar. La configuración inicial ofrece:

| Escenario | Datos usados en entrenamiento | Validación para selección del modelo |
|---|---|---|
| `original` | Partición de entrenamiento de la fuente original | Validación de la fuente original |
| `complementario` | Partición de entrenamiento de la fuente complementaria | Validación de la fuente complementaria |
| `combinado` | Particiones de entrenamiento de ambas fuentes | Validación de ambas fuentes |

Los conjuntos de prueba de ambas fuentes se mantienen reservados para la evaluación final. El comando `evaluar` calcula métricas independientes sobre cada fuente, de modo que los tres escenarios de entrenamiento se puedan comparar con la misma matriz de evaluación.

Rutas esperadas por defecto:

```text
data/
├── Clases_200/
│   ├── 0/
│   ├── 1/
│   └── ...
└── Clases_200_comp/
    ├── 0/
    ├── 1/
    └── ...
```

Cada carpeta de fuente debe tener una subcarpeta por cada clase configurada. Si usás otra ubicación, podés cambiar las rutas en `fuentes.json`, o pasar opciones en la ejecución:

```bash
python -m src.cli.main entrenar --modelo mobilenetv2 --escenario original \
  --dataset-original /ruta/al/original \
  --dataset-complementario /ruta/al/complementario
```

Para una fuente adicional definida en `fuentes.json`, se puede usar `--fuente-ruta nombre=/ruta/dataset`. Los nuevos escenarios se configuran en `escenarios.json` sin cambiar la lógica central del pipeline.

## Ejecutar los experimentos

Cada arquitectura registrada se puede entrenar en cada escenario. Para los cuatro modelos y tres escenarios de la tesis, son 12 entrenamientos principales:

```bash
python -m src.cli.main entrenar --modelo mobilenetv2 --escenario original
python -m src.cli.main entrenar --modelo mobilenetv2 --escenario complementario
python -m src.cli.main entrenar --modelo mobilenetv2 --escenario combinado
```

Repetí con `densenet201`, `vit_tiny` y `vit_base`. Parámetros útiles:

```bash
# Ajustar duración, tamaño de lote, semilla o workers
python -m src.cli.main entrenar --modelo vit_tiny --escenario combinado \
  --epocas 30 --lote 32 --semilla 42 --semilla-split 42 --workers 0

# Entrenar todos los parámetros, en vez de congelar el backbone
python -m src.cli.main entrenar --modelo mobilenetv2 --escenario combinado --ajuste-fino
```

Por defecto se mantiene congelado el backbone y se entrena el clasificador. El entrenamiento registra el modo usado. En este modo las capas congeladas permanecen en evaluación, para evitar modificar accidentalmente las estadísticas de BatchNorm.

Las particiones se generan la primera vez y se guardan en `results/particiones/<fuente>/<fingerprint>_seed<semilla>_<proporciones>.json`. Se reutilizan entre arquitecturas y escenarios para facilitar comparaciones justas. El fingerprint incluye rutas relativas, etiquetas y hashes SHA-256 del contenido; si cambia el contenido o el vocabulario, se crea una partición versionada nueva y se conservan las anteriores.

La semilla de entrenamiento (`--semilla`) es independiente de la semilla del split (`--semilla-split`). Para comparar arquitecturas, mantené la misma semilla del split y las mismas proporciones. Cambiar `--semilla-split` crea otra versión de las particiones; `--regenerar-splits` vuelve a calcular la partición de esa versión de forma explícita. Una vez que un checkpoint existe, el entrenamiento se detiene para evitar sobrescribir resultados por accidente. `--sobrescribir` reemplaza únicamente el directorio del escenario/modelo seleccionado.

## Evaluar e inferir

```bash
# Evalúa el conjunto de prueba original y el complementario de manera independiente
python -m src.cli.main evaluar --modelo mobilenetv2 --escenario combinado

# Evalúa solo una fuente
python -m src.cli.main evaluar --modelo mobilenetv2 --escenario original --fuente original

# Predice una imagen usando el vocabulario y preprocesamiento del checkpoint
python -m src.cli.main inferir --modelo mobilenetv2 --escenario combinado --imagen /ruta/imagen.jpg
```

La evaluación genera métricas agregadas, informe por clase, predicciones CSV y matriz de confusión por fuente. La inferencia recupera del checkpoint el orden de etiquetas utilizado durante el entrenamiento; si el vocabulario local cambió, no se vuelve a interpretar silenciosamente un índice con otra etiqueta.

## Comparar y analizar resultados

```bash
# Consolidar escenarios x modelos x fuentes de prueba en un CSV
python scripts/resumen_experimentos.py

# Comparar arquitecturas dentro de un escenario y una fuente de prueba
python scripts/comparar_modelos.py --escenario combinado --fuente original

# Graficar entrenamiento, inspeccionar errores o medir coste de inferencia
python scripts/graficar_entrenamiento.py --modelo vit_tiny --escenario combinado
python scripts/analizar_errores.py --modelo vit_tiny --escenario combinado --fuente original
python scripts/benchmark_modelos.py --escenario combinado
```

## Ubicación de artefactos

```text
checkpoints/<escenario>/<modelo>/mejor.pth
results/
├── particiones/<fuente>/
│   ├── <fingerprint>_seed<semilla>_<proporciones>.json
│   └── <mismo-prefijo>.duplicados_exactos.csv
└── <escenario>/<modelo>/
    ├── config.json
    ├── history.json
    ├── metrics_<fuente>.json
    ├── classification_report_<fuente>.json
    ├── predicciones_<fuente>.csv
    ├── matriz_confusion_<fuente>.csv
    └── matriz_confusion_<fuente>.png
```

Los datasets, checkpoints y resultados se mantienen fuera de Git mediante `.gitignore`. Los checkpoints nuevos contienen vocabulario, configuración y metadatos de procedencia. Para hacer inferencia con un checkpoint antiguo, puede indicarse su ruta: `python -m src.cli.main inferir --modelo mobilenetv2 --imagen /ruta/imagen.jpg --checkpoint checkpoints/mobilenetv2/mejor.pth`. Los checkpoints legacy sin vocabulario explícito se cargan con aviso y no permiten verificar el mapeo de clases.

## Incorporar nuevas clases estáticas

1. Agregá el identificador de la clase en `clases.json`. El orden de ese archivo define los índices, así que mantenelo estable una vez iniciados los experimentos.
2. Creá la carpeta con ese mismo identificador en cada fuente para la cual se espera evaluar la clase.
3. Incorporá imágenes verificadas de esa clase en cada fuente donde deba ser evaluada y revisá que el número de grupos únicos permita una división estratificada de entrenamiento, validación y prueba.
4. Entrená nuevos checkpoints. El cambio de vocabulario produce automáticamente nuevos fingerprints y nuevas rutas de particiones, conservando la versión anterior. Un checkpoint entrenado con el vocabulario anterior no reconoce automáticamente clases nuevas.
5. Registrá el nuevo experimento y volvé a evaluar también las clases anteriores para detectar regresiones o pérdida de capacidad.

El pipeline no borra archivos fuente ni resuelve automáticamente muestras ambiguas. Por defecto excluye del dataset en memoria las copias exactas repetidas dentro de cada fuente y registra cada exclusión; esto se puede desactivar mediante `deduplicate_exact_duplicates: false` en la definición de una fuente, aunque las copias seguirán agrupadas al partir los datos. Las imágenes de calidad dudosa y las casi duplicadas requieren revisión. La probabilidad de volteo horizontal está desactivada por defecto porque no debe asumirse que el reflejo conserva el significado de cada gesto.

### Límite sobre verbos y señas dinámicas

Agregar más configuraciones manuales estáticas puede resolverse ampliando las etiquetas y datos. En cambio, verbos o señas cuyo significado depende de la trayectoria, la orientación cambiante, la expresión facial o el contexto requieren datos secuenciales y una modalidad temporal. Eso no se logra simplemente agregando nombres al JSON y debe abordarse como una extensión técnica posterior.

## Pruebas

```bash
python -m pytest -q
```

Las pruebas cubren vocabulario variable, validación de configuración, carga del dataset, exclusión auditable de duplicados exactos en memoria, particionado agrupado, detección de contenido modificado, versionado de splits, configuración de fuentes y metadatos/mapeo de etiquetas en checkpoints. Ejecutá también una prueba pequeña de entrenamiento/inferencia en el entorno con GPU y dependencias instaladas antes de lanzar las ejecuciones completas.

## Estado experimental

La arquitectura de código facilita experimentos reproducibles, pero no constituye evidencia de desempeño. Los resultados históricos deben considerarse preliminares hasta volver a ejecutar el pipeline sobre las particiones definitivas y analizar la generalización. La separación por imagen tampoco demuestra generalización a personas no vistas; para esa conclusión hace falta una partición por persona/sesión identificada o un conjunto externo verdaderamente independiente.
